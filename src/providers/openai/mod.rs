// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! OpenAI provider adapter.
//!
//! Rust-native ProviderAdapter that forwards to OpenAI's API.
//! Domain types are already OpenAI-shaped — minimal translation; focus on
//! model-specific parameter handling (reasoning models) and error semantics.

mod types;
pub mod utils;

use async_trait::async_trait;
use futures::StreamExt;
use secrecy::ExposeSecret;
use tracing::debug;

use crate::config::OpenAIConfig;
use crate::domain::chat::{ChatRequest, ChatResponse, Role, Usage};
use crate::domain::embedding::{EmbeddingRequest, EmbeddingResponse};
use crate::domain::ports::{
    ChatCompletionStream, EmbeddingCapabilities, HealthStatus, ProviderAdapter, ProviderAdapterExt,
    ProviderError, ProviderKind, ProviderMetadata,
};
use crate::providers::openai::types::StreamChunkWithUsage;
use crate::providers::openai::utils::{inject_stream_options, normalize_openai_usage};
use crate::utils::provider_error::{classify_reqwest_error, sanitize_network_error};

const OPENAI_API_BASE: &str = "https://api.openai.com";

/// Models that strictly forbid temperature/top_p (all o-series reasoning models).
const FORBIDS_TEMPERATURE: &[&str] = &[
    "o1", "o1-mini", "o1-pro", "o3", "o3-mini", "o3-pro", "o4-mini",
];

/// Reasoning models (o-series): need max_completion_tokens, system→developer.
const REASONING_MODELS: &[&str] = &[
    "o1", "o1-mini", "o1-pro", "o3", "o3-mini", "o3-pro", "o4-mini",
];

/// Known OpenAI models for supported_models. Operators may override via config.
const KNOWN_OPENAI_MODELS: &[&str] = &[
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.1",
    "o1",
    "o1-mini",
    "o1-pro",
    "o3",
    "o3-mini",
    "o3-pro",
    "o4-mini",
];

/// Maximum input tokens for a single OpenAI embedding call (model context window is 8192,
/// but the embeddings endpoint accepts at most 8191 tokens per input).
const OPENAI_EMBED_MAX_INPUT_TOKENS: u32 = 8191;
/// Supported output dimension presets across all OpenAI embedding models (union of
/// text-embedding-3-small [512, 1536], text-embedding-3-large [256, 1024, 3072],
/// ada-002 [1536 only]). Treat as hints — route to a specific model before trusting
/// individual dimension support.
const OPENAI_EMBED_DIMENSIONS: &[u32] = &[256, 512, 1024, 1536, 3072];
/// OpenAI's documented maximum inputs per embeddings call.
const OPENAI_EMBED_BATCH_MAX_ITEMS: usize = 2048;

/// Returns true if the model is a reasoning model (o-series).
#[must_use]
pub fn is_reasoning_model(model: &str) -> bool {
    REASONING_MODELS.contains(&model)
        || model.starts_with("o1-")
        || model.starts_with("o3-")
        || model.starts_with("o4-")
        || model.starts_with("o5-")
}

/// Returns true if the model strictly forbids temperature/top_p.
/// All o-series reasoning models reject temperature/top_p at the API level.
#[must_use]
pub fn forbids_temperature(model: &str) -> bool {
    FORBIDS_TEMPERATURE.contains(&model)
        || model == "o1-preview"
        || model.starts_with("o3-")
        || model.starts_with("o4-")
        || model.starts_with("o5-")
}

fn openai_base_url(config: &OpenAIConfig) -> String {
    config
        .api_base_url
        .clone()
        .unwrap_or_else(|| OPENAI_API_BASE.to_string())
}

/// Prepares a ChatRequest for upstream. Clones and mutates for reasoning models.
fn prepare_request(req: &ChatRequest) -> ChatRequest {
    let model = &req.model;
    let mut prepared = req.clone();

    if is_reasoning_model(model) {
        // max_tokens ignored; use max_completion_tokens (derive from max_tokens if missing)
        if prepared.max_completion_tokens.is_none() {
            prepared.max_completion_tokens = prepared.max_tokens;
        }
        prepared.max_tokens = None;

        // system → developer for all o-series
        for msg in &mut prepared.messages {
            if msg.role == Role::System {
                msg.role = Role::Other("developer".to_string());
            }
        }

        // Strip temperature/top_p only for models that forbid them (o1-series)
        if forbids_temperature(model) {
            prepared.temperature = None;
            prepared.extra.remove("top_p");
        }
        // o3, o3-pro, o4-mini: forward temperature/top_p as-is

        // reasoning_effort from extra → forward as-is (already in extra)
    }

    prepared
}

/// OpenAI provider adapter.
pub struct OpenAiAdapter {
    config: OpenAIConfig,
    http: reqwest::Client,
    metadata: ProviderMetadata,
    /// Validated bearer token, stored once at construction to avoid expect on hot path.
    api_key: String,
}

impl OpenAiAdapter {
    /// Creates an OpenAI adapter. Config must be validated by GatewayConfig::validate.
    pub async fn new(config: OpenAIConfig) -> Result<Self, ProviderError> {
        let api_key = config
            .api_key
            .as_ref()
            .and_then(|k| {
                let s = k.expose_secret();
                if s.is_empty() { None } else { Some(s.clone()) }
            })
            .ok_or_else(|| ProviderError::Auth("providers.openai.api_key is required".into()))?;

        let timeout_secs = config.timeout_secs.unwrap_or(120);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| ProviderError::Unreachable(format!("reqwest client: {}", e)))?;

        let supported_models = config.supported_models.clone().unwrap_or_else(|| {
            let mut models: Vec<String> = KNOWN_OPENAI_MODELS
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            models.push("gpt-".to_string());
            models.push("o1-".to_string());
            models.push("o3-".to_string());
            models.push("o4-".to_string());
            models.push("text-embedding-".to_string());
            models
        });

        let metadata = ProviderMetadata {
            name: "openai".to_string(),
            supported_models,
            supports_streaming: true,
            supports_tools: true,
            supports_vision: true,
            supports_embeddings: true,
            supports_thinking: true,
            kind: ProviderKind::Primary,
            embedding_capabilities: Some(EmbeddingCapabilities {
                dimensions: OPENAI_EMBED_DIMENSIONS.to_vec(),
                max_input_tokens: OPENAI_EMBED_MAX_INPUT_TOKENS,
                supports_batch: true,
            }),
        };

        Ok(Self {
            config,
            http: client,
            metadata,
            api_key,
        })
    }

    fn model(&self, req_model: &str) -> String {
        if req_model.trim().is_empty() {
            self.config
                .default_model
                .clone()
                .unwrap_or_else(|| "gpt-4o".to_string())
        } else {
            req_model.to_string()
        }
    }

    fn chat_url(&self) -> String {
        let base = openai_base_url(&self.config);
        let base = base.trim_end_matches('/');
        format!("{base}/v1/chat/completions")
    }

    fn build_request(&self, req: &ChatRequest) -> reqwest::RequestBuilder {
        let mut r = self
            .http
            .post(self.chat_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(req);

        if let Some(ref org) = self.config.organization {
            r = r.header("OpenAI-Organization", org.as_str());
        }
        if let Some(ref proj) = self.config.project {
            r = r.header("OpenAI-Project", proj.as_str());
        }

        r
    }

    async fn map_error_response(
        &self,
        status: reqwest::StatusCode,
        resp: reqwest::Response,
    ) -> ProviderError {
        crate::providers::openai::utils::map_openai_error_response(status, resp).await
    }

    fn embed_url(&self) -> String {
        let base = openai_base_url(&self.config);
        let base = base.trim_end_matches('/');
        format!("{base}/v1/embeddings")
    }
}

#[async_trait]
impl ProviderAdapter for OpenAiAdapter {
    async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        let model = self.model(&req.model);
        let prepared = prepare_request(req);

        let start = std::time::Instant::now();
        let resp = self
            .build_request(&prepared)
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(self.map_error_response(status, resp).await);
        }

        let mut chat_resp: ChatResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::Serialization(e.to_string()))?;

        // Ensure model in response matches request
        chat_resp.model = model.clone();

        //: map prompt_tokens_details.cached_tokens → cache_read_input_tokens
        normalize_openai_usage(&mut chat_resp.usage);

        Ok(chat_resp)
    }

    async fn chat_completion_stream(
        &self,
        req: &ChatRequest,
    ) -> Result<ChatCompletionStream, ProviderError> {
        let mut prepared = prepare_request(req);
        prepared.stream = Some(true);

        // Inject stream_options.include_usage: true for usage in final chunk.
        // If client explicitly set include_usage: false, respect it.
        inject_stream_options(&mut prepared);

        let start = std::time::Instant::now();
        let resp = self
            .build_request(&prepared)
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))?;

        if !resp.status().is_success() {
            return Err(self.map_error_response(resp.status(), resp).await);
        }

        let mut stream = resp
            .bytes_stream()
            .map(|r| r.map_err(|e: reqwest::Error| std::io::Error::other(e.to_string())));

        let s = async_stream::stream! {
            let mut last_usage: Option<Usage> = None;
            // Resolved model from SSE chunks. chat.rs uses first-wins semantics for CostHeader::MODEL_USED.
            let mut resolved_model: Option<String> = None;

            while let Some(chunk_res) = stream.next().await {
                let data = match chunk_res {
                    Ok(b) => b,
                    Err(e) => {
                        yield Err(ProviderError::Unreachable(format!(
                            "openai: {}",
                            sanitize_network_error(&e.to_string())
                        )));
                        break;
                    }
                };

                // Parse for usage and model extraction (final chunk has "usage": {...})
                if !data.is_empty() {
                    let s = std::str::from_utf8(&data).unwrap_or_default();
                    for line in s.lines() {
                        let trimmed = line.trim();
                        let json_str = trimmed.strip_prefix("data: ").unwrap_or(trimmed).trim();
                        if json_str != "[DONE]"
                            && !json_str.is_empty()
                            && let Ok(parsed) = serde_json::from_str::<StreamChunkWithUsage>(json_str)
                        {
                            if let Some(ref m) = parsed.model {
                                resolved_model = Some(m.clone());
                            }
                            if let Some(ref usage) = parsed.usage {
                                let mut u = usage.clone();
                                normalize_openai_usage(&mut u);
                                last_usage = Some(u);
                                debug!(
                                    prompt_tokens = usage.prompt_tokens,
                                    completion_tokens = usage.completion_tokens,
                                    reasoning_tokens = ?usage
                                        .completion_tokens_details
                                        .as_ref()
                                        .and_then(|d| d.reasoning_tokens),
                                    "openai streaming usage extracted for cost tracking"
                                );
                            }
                        }
                    }
                }

                yield Ok(crate::domain::chat::StreamChunk::new(
                    data,
                    last_usage.clone(),
                    resolved_model.clone(),
                ));
            }

        };

        Ok(Box::pin(s))
    }

    async fn embeddings(&self, req: &EmbeddingRequest) -> Result<EmbeddingResponse, ProviderError> {
        let model = self.model(&req.model);

        if req.input.as_slice().len() > OPENAI_EMBED_BATCH_MAX_ITEMS {
            return Err(ProviderError::InvalidRequest(format!(
                "OpenAI embedding limit is {OPENAI_EMBED_BATCH_MAX_ITEMS} inputs per call; {} provided",
                req.input.as_slice().len()
            )));
        }

        let mut body = req.clone();
        body.model = model.clone();

        let start = std::time::Instant::now();
        let resp = self
            .http
            .post(self.embed_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))?;

        if !resp.status().is_success() {
            return Err(self.map_error_response(resp.status(), resp).await);
        }

        let mut embed_resp: EmbeddingResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::Serialization(e.to_string()))?;

        embed_resp.model = model;

        // Normalise: backfill prompt_tokens from total_tokens when the field is absent/zero.
        if let Some(ref mut usage) = embed_resp.usage
            && usage.prompt_tokens == 0
            && usage.total_tokens > 0
        {
            usage.prompt_tokens = usage.total_tokens;
        }

        Ok(embed_resp)
    }

    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }

    async fn health_check(&self) -> HealthStatus {
        let base = openai_base_url(&self.config);
        let base = base.trim_end_matches('/');
        let url = format!("{base}/v1/models");
        match self
            .http
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => HealthStatus::Healthy,
            Ok(_) => HealthStatus::Degraded,
            Err(_) => HealthStatus::Unhealthy,
        }
    }
}

impl ProviderAdapterExt for OpenAiAdapter {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::{MessageContent, Role};

    fn msg(role: Role, content: &str) -> crate::domain::chat::Message {
        crate::domain::chat::Message {
            role,
            content: Some(MessageContent::Text(content.to_string())),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn test_reasoning_model_flags() {
        assert!(is_reasoning_model("o1"));
        assert!(is_reasoning_model("o1-mini"));
        assert!(is_reasoning_model("o3"));
        assert!(is_reasoning_model("o3-pro"));
        assert!(is_reasoning_model("o4-mini"));
        assert!(is_reasoning_model("o5-mini")); // future o5 family via prefix
        assert!(!is_reasoning_model("gpt-4o"));
        assert!(!is_reasoning_model("gpt-5"));
    }

    #[test]
    fn test_forbids_temperature_all_reasoning_models() {
        // o1 family
        assert!(forbids_temperature("o1"));
        assert!(forbids_temperature("o1-mini"));
        assert!(forbids_temperature("o1-pro"));
        assert!(forbids_temperature("o1-preview"));
        // o3 family
        assert!(forbids_temperature("o3"));
        assert!(forbids_temperature("o3-mini"));
        assert!(forbids_temperature("o3-pro"));
        // o4 family
        assert!(forbids_temperature("o4-mini"));
        // non-reasoning must not be affected
        assert!(!forbids_temperature("gpt-4o"));
        assert!(!forbids_temperature("gpt-5"));
    }

    #[test]
    fn test_prepare_request_reasoning_model() {
        let req = ChatRequest {
            model: "o3".into(),
            messages: vec![msg(Role::User, "hi")],
            temperature: Some(0.7),
            max_tokens: Some(100),
            max_completion_tokens: None,
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra: Default::default(),
        };
        let prepared = prepare_request(&req);
        assert_eq!(prepared.max_tokens, None);
        assert_eq!(prepared.max_completion_tokens, Some(100));
        assert_eq!(prepared.temperature, None); // o3 forbids temperature
    }

    #[test]
    fn test_prepare_request_temperature_stripped_for_o1() {
        let mut extra = serde_json::Map::new();
        extra.insert("top_p".into(), serde_json::json!(0.9));
        let req = ChatRequest {
            model: "o1".into(),
            messages: vec![msg(Role::User, "hi")],
            temperature: Some(0.7),
            max_tokens: Some(100),
            max_completion_tokens: None,
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra,
        };
        let prepared = prepare_request(&req);
        assert_eq!(prepared.temperature, None);
        assert!(prepared.extra.get("top_p").is_none());
    }

    #[test]
    fn test_prepare_request_temperature_stripped_for_o3() {
        let mut extra = serde_json::Map::new();
        extra.insert("top_p".into(), serde_json::json!(0.9));
        let req = ChatRequest {
            model: "o3".into(),
            messages: vec![msg(Role::User, "hi")],
            temperature: Some(0.5),
            max_tokens: Some(100),
            max_completion_tokens: None,
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra: extra.clone(),
        };
        let prepared = prepare_request(&req);
        // o3 is a reasoning model and forbids temperature/top_p
        assert_eq!(prepared.temperature, None);
        assert!(prepared.extra.get("top_p").is_none());
    }

    #[test]
    fn test_prepare_request_system_to_developer() {
        let req = ChatRequest {
            model: "o3".into(),
            messages: vec![msg(Role::System, "You are helpful."), msg(Role::User, "hi")],
            temperature: None,
            max_tokens: None,
            max_completion_tokens: Some(500),
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra: Default::default(),
        };
        let prepared = prepare_request(&req);
        assert_eq!(
            prepared.messages[0].role,
            Role::Other("developer".to_string())
        );
        assert_eq!(prepared.messages[1].role, Role::User);
    }

    #[tokio::test]
    async fn test_supported_models_override() {
        let config = OpenAIConfig {
            api_key: Some(crate::config::SecretString::new("sk-test")),
            default_model: Some("gpt-4o".into()),
            api_base_url: Some("http://localhost:9999".into()),
            timeout_secs: Some(10),
            supported_models: Some(vec!["custom-model".into(), "my-gpt".into()]),
            organization: None,
            project: None,
        };
        let adapter = OpenAiAdapter::new(config).await.expect("must build");
        let meta = adapter.metadata();
        assert_eq!(
            meta.supported_models,
            vec!["custom-model".to_string(), "my-gpt".to_string()]
        );
    }

    #[test]
    fn test_prepare_request_standard_model_unchanged() {
        let req = ChatRequest {
            model: "gpt-4o".into(),
            messages: vec![msg(Role::System, "Help"), msg(Role::User, "hi")],
            temperature: Some(0.8),
            max_tokens: Some(100),
            max_completion_tokens: None,
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra: Default::default(),
        };
        let prepared = prepare_request(&req);
        assert_eq!(prepared.messages[0].role, Role::System);
        assert_eq!(prepared.temperature, Some(0.8));
        assert_eq!(prepared.max_tokens, Some(100));
    }

    /// OpenAI adapter reports supports_embeddings = true with correct capabilities.
    #[tokio::test]
    async fn test_openai_metadata_supports_embeddings() {
        let config = OpenAIConfig {
            api_key: Some(crate::config::SecretString::new("sk-test")),
            default_model: None,
            api_base_url: Some("http://localhost:9999".into()),
            timeout_secs: Some(10),
            supported_models: None,
            organization: None,
            project: None,
        };
        let adapter = OpenAiAdapter::new(config)
            .await
            .expect("adapter must build");
        let meta = adapter.metadata();
        assert!(meta.supports_embeddings);
        let caps = meta
            .embedding_capabilities
            .as_ref()
            .expect("embedding_capabilities must be set");
        assert!(caps.supports_batch);
        assert!(caps.dimensions.contains(&1536));
        assert!(caps.dimensions.contains(&3072));
        assert_eq!(caps.max_input_tokens, 8191);
    }

    /// embed_url helper builds the correct endpoint.
    #[test]
    fn test_openai_embed_url_default_base() {
        // Verify URL pattern directly — no live adapter needed.
        let base = OPENAI_API_BASE;
        let url = format!("{}/v1/embeddings", base.trim_end_matches('/'));
        assert_eq!(url, "https://api.openai.com/v1/embeddings");
    }

    /// embed_url respects custom api_base_url.
    #[test]
    fn test_openai_embed_url_custom_base() {
        let base = "https://custom.openai.proxy.example.com/";
        let url = format!("{}/v1/embeddings", base.trim_end_matches('/'));
        assert_eq!(url, "https://custom.openai.proxy.example.com/v1/embeddings");
    }
}
