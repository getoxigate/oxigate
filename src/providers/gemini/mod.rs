// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Google Gemini / Vertex AI provider adapter.
//!
//! Dual auth (API key + Vertex OAuth), OpenAI-compatible interface.
//! ThinkingConfig activation, model taxonomy, multimodal Part types.

mod auth;
mod translate;
mod types;

use async_trait::async_trait;
use bitflags::bitflags;
use futures::StreamExt;
use reqwest::Url;
use secrecy::ExposeSecret;
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::warn;

use crate::config::{GeminiConfig, GeminiMode};
use crate::domain::chat::{ChatRequest, ChatResponse, StreamChunk};
use crate::domain::embedding::{EmbeddingRequest, EmbeddingResponse};
use crate::domain::ports::{
    ChatCompletionStream, EmbeddingCapabilities, HealthStatus, ProviderAdapter, ProviderAdapterExt,
    ProviderError, ProviderKind, ProviderMetadata,
};
use crate::providers::gemini::auth::{GeminiApiKey, VertexOAuthTokens};
use crate::providers::gemini::types::{
    EmbedContent, EmbedContentItem, EmbedPart, EmbedStatistics, GeminiBatchEmbedItem,
    GeminiBatchEmbedRequest, GeminiBatchEmbedResponse, GeminiChatResponse, GeminiEmbeddingRequest,
    GeminiSingleEmbedResponse, VertexEmbeddingInstance, VertexEmbeddingRequest,
};
use crate::utils::provider_error::{classify_reqwest_error, sanitize_network_error};

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com";
const GEMINI_V1BETA: &str = "/v1beta/models";
const GEMINI_DEFAULT_TASK_TYPE: &str = "RETRIEVAL_DOCUMENT";
/// Google's documented batchEmbedContents limit (items per call).
const GEMINI_BATCH_EMBED_MAX_ITEMS: usize = 100;
/// Conservative Vertex AI predict limit (items per call; varies by model, 250 is the lowest GA floor).
const VERTEX_EMBED_MAX_ITEMS: usize = 250;
/// Maximum input tokens for a single Gemini embedding input (text-embedding-004 / multilingual-002).
const GEMINI_EMBED_MAX_INPUT_TOKENS: u32 = 2048;
/// Supported output dimension presets for Gemini embedding models (768 native; 3072 max upscaled).
const GEMINI_EMBED_DIMENSIONS: &[u32] = &[768, 3072];

bitflags! {
    /// Per-model capability flags for thinking config and metadata.
    #[derive(Clone, Copy)]
    pub struct ModelFlags: u8 {
        const NONE = 0;
        /// Gemini 2.5: supports thinkingBudget (integer). Thinking tokens billed at separate tier.
        const THINKING_BUDGET = 1 << 0;
        /// Gemini 3.x: supports thinkingLevel (enum). Thinking tokens billed at output rate.
        const THINKING_LEVEL = 1 << 1;
        /// Model is deprecated; warn at startup if used as default_model.
        const DEPRECATED = 1 << 2;
        /// Model is in Preview; not GA.
        const PREVIEW = 1 << 3;
    }
}

// Mirror ModelFlags bits as raw u8 for const array init — bitflags! | is not const-compatible.
const F_THINKING_BUDGET: u8 = 1 << 0;
const F_THINKING_LEVEL: u8 = 1 << 1;
const F_DEPRECATED: u8 = 1 << 2;
const F_PREVIEW: u8 = 1 << 3;

/// Known Gemini model IDs. Operators may extend via providers.gemini.supported_models in YAML.
/// Element type is `(&str, u8)` rather than `(&str, ModelFlags)` because bitflags! `|` is not
/// const-compatible; `ModelFlags::from_bits_retain()` is called at lookup time in `model_flags()`.
pub(crate) const KNOWN_GEMINI_MODELS: &[(&str, u8)] = &[
    ("gemini-2.5-pro", F_THINKING_BUDGET),
    ("gemini-2.5-flash", F_THINKING_BUDGET),
    ("gemini-2.5-flash-lite", 0),
    ("gemini-2.0-flash", F_DEPRECATED),
    ("gemini-2.0-flash-lite", F_DEPRECATED),
    ("gemini-1.5-pro", F_DEPRECATED),
    ("gemini-1.5-flash", F_DEPRECATED),
    ("gemini-3.1-pro-preview", F_THINKING_LEVEL | F_PREVIEW),
    (
        "gemini-3.1-flash-lite-preview",
        F_THINKING_LEVEL | F_PREVIEW,
    ),
    ("gemini-3-flash-preview", F_THINKING_LEVEL | F_PREVIEW),
    ("text-embedding-004", 0),
    ("text-multilingual-embedding-002", 0),
    // Experimental embedding model; pricing matches text-embedding-004 ($0.0000001/token).
    ("gemini-embedding-exp-03-07", 0),
];

pub(crate) fn model_flags(model: &str) -> ModelFlags {
    if let Some((_, bits)) = KNOWN_GEMINI_MODELS.iter().find(|(id, _)| *id == model) {
        return ModelFlags::from_bits_retain(*bits);
    }
    if model.starts_with("gemini-2.5") {
        tracing::debug!("unknown model, guessing thinking flags by prefix: {model}");
        return ModelFlags::THINKING_BUDGET;
    }
    if model.starts_with("gemini-3") {
        tracing::debug!("unknown model, guessing thinking flags by prefix: {model}");
        return ModelFlags::THINKING_LEVEL | ModelFlags::PREVIEW;
    }
    ModelFlags::NONE
}

fn gemini_base_url(config: &GeminiConfig) -> String {
    config
        .api_base_url
        .clone()
        .unwrap_or_else(|| GEMINI_API_BASE.to_string())
}

fn vertex_base_url(config: &GeminiConfig, location: &str) -> String {
    config
        .vertex_base_url_override
        .clone()
        .unwrap_or_else(|| format!("https://{location}-aiplatform.googleapis.com"))
}

/// Auth mode for Gemini: API key or Vertex OAuth.
pub enum GeminiAuth {
    ApiKey(GeminiApiKey),
    Vertex(VertexOAuthTokens),
}

/// Google Gemini/Vertex AI provider adapter.
pub struct GeminiAdapter {
    config: GeminiConfig,
    http: reqwest::Client,
    auth: GeminiAuth,
    metadata: ProviderMetadata,
}

impl GeminiAdapter {
    /// Creates a Gemini adapter. Call only with config validated by [crate::config::GatewayConfig::validate];
    /// validation is done at startup via [crate::config::load_and_validate_config].
    pub async fn new(config: GeminiConfig) -> Result<Self, ProviderError> {
        let auth = match &config.mode {
            GeminiMode::Api => {
                let key = config
                    .api_key
                    .clone()
                    .expect("api_key validated by config layer when mode is api");
                GeminiAuth::ApiKey(GeminiApiKey(key))
            }
            GeminiMode::Vertex => {
                let project = config
                    .vertex_project
                    .clone()
                    .expect("vertex_project validated by config layer when mode is vertex");
                let location = config
                    .vertex_location
                    .clone()
                    .expect("vertex_location validated by config layer when mode is vertex");
                let sa = config.vertex_service_account_json.clone().expect(
                    "vertex_service_account_json validated by config layer when mode is vertex",
                );
                let tokens = VertexOAuthTokens::new(project, location, sa)
                    .await
                    .map_err(|e| ProviderError::Auth(e.to_string()))?;
                GeminiAuth::Vertex(tokens)
            }
        };

        let timeout_secs = config.timeout_secs.unwrap_or(120);
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| classify_reqwest_error(e, 0))?;

        let supported_models: Vec<String> = if let Some(ref models) = config.supported_models {
            models.clone()
        } else {
            KNOWN_GEMINI_MODELS
                .iter()
                .filter(|(_, bits)| (*bits & F_DEPRECATED) == 0)
                .map(|(id, _)| (*id).to_string())
                .collect()
        };

        const DEPRECATION_RETIREMENT_DATE: &str = "2026-06-01"; // update if Google extends deadline
        if let Some(ref default_model) = config.default_model
            && let Some((_, bits)) = KNOWN_GEMINI_MODELS
                .iter()
                .find(|(id, _)| *id == default_model.as_str())
            && (*bits & F_DEPRECATED) != 0
        {
            warn!(
                model = %default_model,
                "Gemini default model is deprecated and will be retired on {}",
                DEPRECATION_RETIREMENT_DATE
            );
        }

        let metadata = ProviderMetadata {
            name: "google".to_string(),
            supported_models,
            supports_streaming: true,
            supports_tools: true,
            supports_vision: false,
            supports_embeddings: true,
            supports_thinking: true,
            kind: ProviderKind::Primary,
            embedding_capabilities: Some(EmbeddingCapabilities {
                dimensions: GEMINI_EMBED_DIMENSIONS.to_vec(),
                max_input_tokens: GEMINI_EMBED_MAX_INPUT_TOKENS,
                supports_batch: true,
            }),
        };

        Ok(Self {
            config,
            http,
            auth,
            metadata,
        })
    }

    fn model(&self, req_model: &str) -> String {
        if req_model.is_empty() {
            self.config
                .default_model
                .clone()
                .unwrap_or_else(|| "gemini-2.0-flash".into())
        } else {
            req_model.to_string()
        }
    }

    fn build_chat_url(&self, model: &str, endpoint: &str) -> Result<Url, ProviderError> {
        let model = self.model(model);
        let url = match &self.auth {
            GeminiAuth::ApiKey(_) => {
                let base = gemini_base_url(&self.config);
                format!("{}{}/{}:{}", base, GEMINI_V1BETA, model, endpoint)
            }
            GeminiAuth::Vertex(t) => {
                let base = vertex_base_url(&self.config, t.location());
                format!(
                    "{}/v1/projects/{}/locations/{}/publishers/google/models/{}:{}",
                    base,
                    t.project(),
                    t.location(),
                    model,
                    endpoint
                )
            }
        };
        Url::parse(&url).map_err(|e| ProviderError::InvalidRequest(e.to_string()))
    }

    fn build_embed_url(&self, model: &str, endpoint: &str) -> Result<Url, ProviderError> {
        let model = self.model(model);
        let url = match &self.auth {
            GeminiAuth::ApiKey(_) => {
                let base = gemini_base_url(&self.config);
                let api_path = match self.config.embed_api_version.as_deref() {
                    Some(v) => format!("/{}/models", v),
                    None => "/v1/models".to_string(),
                };
                format!("{}{}/{}:{}", base, api_path, model, endpoint)
            }
            GeminiAuth::Vertex(t) => {
                let base = vertex_base_url(&self.config, t.location());
                format!(
                    "{}/v1/projects/{}/locations/{}/publishers/google/models/{}:{}",
                    base,
                    t.project(),
                    t.location(),
                    model,
                    endpoint
                )
            }
        };
        Url::parse(&url).map_err(|e| ProviderError::InvalidRequest(e.to_string()))
    }

    async fn apply_auth(&self, url: &mut Url) -> Result<(), ProviderError> {
        match &self.auth {
            GeminiAuth::ApiKey(k) => {
                k.apply_to_url(url);
                Ok(())
            }
            GeminiAuth::Vertex(_) => {
                // Vertex uses Bearer header at request time, not URL
                Ok(())
            }
        }
    }

    /// Applies Vertex OAuth Bearer header to a request; no-op for API key mode.
    async fn apply_vertex_bearer(
        &self,
        req: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder, ProviderError> {
        match &self.auth {
            GeminiAuth::ApiKey(_) => Ok(req),
            GeminiAuth::Vertex(t) => {
                let token = t
                    .get_token()
                    .await
                    .map_err(|e| ProviderError::Auth(e.to_string()))?;
                Ok(req.header("Authorization", format!("Bearer {}", token.expose_secret())))
            }
        }
    }

    async fn map_error_response(
        &self,
        status: reqwest::StatusCode,
        resp: reqwest::Response,
    ) -> ProviderError {
        let retry_after = resp
            .headers()
            .get("Retry-After")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        let body = resp.text().await.unwrap_or_default();
        let msg = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|j| {
                j.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .map(String::from)
            })
            .unwrap_or_else(|| body.clone());

        match status.as_u16() {
            400 => ProviderError::InvalidRequest(msg),
            401 => ProviderError::Auth(msg),
            403 => ProviderError::Auth(format!("forbidden: {msg}")),
            404 => ProviderError::UnknownModel(msg),
            429 => ProviderError::RateLimited { retry_after },
            500 | 502 | 503 => ProviderError::ProviderUnavailable(msg),
            _ => ProviderError::ProviderHttpError {
                status: status.as_u16(),
                body,
            },
        }
    }
}

#[async_trait]
impl ProviderAdapter for GeminiAdapter {
    async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        let model = self.model(&req.model);
        let request_id = req.request_id.as_deref().unwrap_or("unknown");

        let gemini_req = translate::openai_to_gemini(req, self.config.resolved_thinking_budget())?;

        let mut url = self
            .build_chat_url(&model, "generateContent")
            .map_err(|e| ProviderError::InvalidRequest(e.to_string()))?;

        self.apply_auth(&mut url).await?;

        let request = self.http.post(url.clone()).json(&gemini_req);
        let request = self.apply_vertex_bearer(request).await?;

        let start = std::time::Instant::now();
        let resp = request
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(self.map_error_response(status, resp).await);
        }

        let gemini_resp: GeminiChatResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::Serialization(e.to_string()))?;

        if let Some(c) = gemini_resp.candidates.first()
            && c.finish_reason.as_deref() == Some("SAFETY")
        {
            return Err(ProviderError::ContentFiltered(
                "content blocked by safety filter".into(),
            ));
        }

        translate::gemini_to_openai(&gemini_resp, &model, request_id)
            .map_err(|e| ProviderError::Translate(e.to_string()))
    }

    async fn chat_completion_stream(
        &self,
        req: &ChatRequest,
    ) -> Result<ChatCompletionStream, ProviderError> {
        let model = self.model(&req.model);
        let request_id = req.request_id.as_deref().unwrap_or("unknown");

        let gemini_req = translate::openai_to_gemini(req, self.config.resolved_thinking_budget())?;

        // Gemini API requires ?alt=sse for proper SSE streaming; Vertex returns NDJSON.
        let mut url = self
            .build_chat_url(&model, "streamGenerateContent")
            .map_err(|e| ProviderError::InvalidRequest(e.to_string()))?;
        if matches!(&self.auth, GeminiAuth::ApiKey(_)) {
            url.query_pairs_mut().append_pair("alt", "sse");
        }
        self.apply_auth(&mut url).await?;

        let request = self.http.post(url.clone()).json(&gemini_req);
        let request = self.apply_vertex_bearer(request).await?;

        let start = std::time::Instant::now();
        let resp = request
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))?;

        if !resp.status().is_success() {
            return Err(self.map_error_response(resp.status(), resp).await);
        }

        let stream = resp
            .bytes_stream()
            .map(|r| r.map_err(|e: reqwest::Error| std::io::Error::other(e.to_string())));

        let model_clone = model.clone();
        let request_id_clone = request_id.to_string();
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let s = async_stream::stream! {
            let mut reader = BufReader::new(tokio_util::io::StreamReader::new(stream));
            let mut line = String::new();

            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) => break,
                    Err(e) => {
                        yield Err(ProviderError::Unreachable(format!(
                            "gemini: {}",
                            sanitize_network_error(&e.to_string())
                        )));
                        break;
                    }
                    Ok(_) => {}
                }
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let json_str = trimmed
                    .strip_prefix("data: ")
                    .unwrap_or(trimmed)
                    .trim();
                if json_str == "[DONE]" {
                    break;
                }
                let chunk: GeminiChatResponse = match serde_json::from_str(json_str) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                let has_finish = chunk
                    .candidates
                    .first()
                    .and_then(|c| c.finish_reason.as_ref())
                    .is_some();

                if has_finish {
                    // Gemini Vertex sends usage_metadata in a separate chunk after the one with
                    // finish_reason. Try to read one more chunk for complete usage.
                    line.clear();
                    let usage_chunk: Option<GeminiChatResponse> = match reader.read_line(&mut line).await {
                        Ok(0) => None,
                        Err(_) => None,
                        Ok(_) => {
                            let trimmed = line.trim();
                            if trimmed.is_empty() {
                                None
                            } else {
                                let json_str = trimmed
                                    .strip_prefix("data: ")
                                    .unwrap_or(trimmed)
                                    .trim();
                                if json_str == "[DONE]" {
                                    None
                                } else {
                                    serde_json::from_str(json_str).ok()
                                }
                            }
                        }
                    };
                    let usage_from = usage_chunk
                        .as_ref()
                        .filter(|u| u.usage_metadata.is_some());
                    if let Ok(Some(sse)) = translate::gemini_stream_chunk_to_sse(
                        &chunk,
                        &model_clone,
                        &request_id_clone,
                        created,
                        true,
                        usage_from,
                    ) {
                        yield Ok(sse);
                    }
                    break;
                }

                if let Ok(Some(sse)) = translate::gemini_stream_chunk_to_sse(
                    &chunk,
                    &model_clone,
                    &request_id_clone,
                    created,
                    false,
                    None,
                ) {
                    yield Ok(sse);
                }
            }

            yield Ok(StreamChunk::new(
                bytes::Bytes::from("data: [DONE]\n\n"),
                None,
                Some(model_clone.clone()),
            ));
        };

        Ok(Box::pin(s))
    }

    async fn embeddings(&self, req: &EmbeddingRequest) -> Result<EmbeddingResponse, ProviderError> {
        let model = self.model(&req.model);
        let inputs = req.input.as_slice();

        if inputs.is_empty() {
            return Err(ProviderError::InvalidRequest(
                "input must not be empty".into(),
            ));
        }

        match &self.auth {
            GeminiAuth::ApiKey(_) => {
                let (items, total_tokens) = if inputs.len() == 1 {
                    // Single input: embedContent (lower latency than batch for one item)
                    let embed_req = GeminiEmbeddingRequest {
                        content: EmbedContent {
                            parts: vec![EmbedPart {
                                text: inputs[0].clone(),
                            }],
                        },
                        task_type: Some(GEMINI_DEFAULT_TASK_TYPE),
                    };
                    let mut url = self.build_embed_url(&model, "embedContent")?;
                    self.apply_auth(&mut url).await?;

                    let start = std::time::Instant::now();
                    let resp = self
                        .http
                        .post(url)
                        .json(&embed_req)
                        .send()
                        .await
                        .map_err(|e| {
                            classify_reqwest_error(e, start.elapsed().as_millis() as u64)
                        })?;

                    if !resp.status().is_success() {
                        return Err(self.map_error_response(resp.status(), resp).await);
                    }

                    let body: GeminiSingleEmbedResponse = resp
                        .json()
                        .await
                        .map_err(|e| ProviderError::Serialization(e.to_string()))?;

                    let tokens = body
                        .embedding
                        .statistics
                        .as_ref()
                        .map(|s| s.token_count)
                        .unwrap_or_else(|| {
                            warn!("gemini embedContent response missing tokenCount");
                            0
                        });
                    (vec![body.embedding], tokens)
                } else {
                    // Multiple inputs: batchEmbedContents (single round-trip, max 100 items per Google docs)
                    if inputs.len() > GEMINI_BATCH_EMBED_MAX_ITEMS {
                        return Err(ProviderError::InvalidRequest(format!(
                            "batchEmbedContents: Google API limit is {GEMINI_BATCH_EMBED_MAX_ITEMS} items per call; {} provided",
                            inputs.len()
                        )));
                    }
                    let batch_items: Vec<GeminiBatchEmbedItem> = inputs
                        .iter()
                        .map(|text| GeminiBatchEmbedItem {
                            model: format!("models/{}", model),
                            content: EmbedContent {
                                parts: vec![EmbedPart { text: text.clone() }],
                            },
                            task_type: Some(GEMINI_DEFAULT_TASK_TYPE),
                        })
                        .collect();

                    let batch_req = GeminiBatchEmbedRequest {
                        requests: batch_items,
                    };
                    let mut url = self.build_embed_url(&model, "batchEmbedContents")?;
                    self.apply_auth(&mut url).await?;

                    let start = std::time::Instant::now();
                    let resp = self
                        .http
                        .post(url)
                        .json(&batch_req)
                        .send()
                        .await
                        .map_err(|e| {
                            classify_reqwest_error(e, start.elapsed().as_millis() as u64)
                        })?;

                    if !resp.status().is_success() {
                        return Err(self.map_error_response(resp.status(), resp).await);
                    }

                    let body: GeminiBatchEmbedResponse = resp
                        .json()
                        .await
                        .map_err(|e| ProviderError::Serialization(e.to_string()))?;

                    let tokens: u64 = body
                        .embeddings
                        .iter()
                        .enumerate()
                        .map(|(i, e)| {
                            e.statistics
                                .as_ref()
                                .map(|s| s.token_count)
                                .unwrap_or_else(|| {
                                    warn!(
                                        index = i,
                                        "gemini batchEmbedContents element missing tokenCount"
                                    );
                                    0
                                })
                        })
                        .sum();
                    (body.embeddings, tokens)
                };

                Ok(translate::gemini_embedding_to_openai(
                    items,
                    &model,
                    total_tokens,
                ))
            }
            GeminiAuth::Vertex(_t) => {
                // Vertex AI embedding predict limits vary by model; use the lowest documented GA floor.
                if inputs.len() > VERTEX_EMBED_MAX_ITEMS {
                    return Err(ProviderError::InvalidRequest(format!(
                        "Vertex AI embedding limit is {VERTEX_EMBED_MAX_ITEMS} instances per call; {} provided",
                        inputs.len()
                    )));
                }
                let instances: Vec<VertexEmbeddingInstance> = inputs
                    .iter()
                    .map(|s| VertexEmbeddingInstance {
                        content: s.clone(),
                        task_type: Some(GEMINI_DEFAULT_TASK_TYPE),
                    })
                    .collect();

                let vertex_req = VertexEmbeddingRequest { instances };
                let url = self.build_embed_url(&model, "predict")?;
                let request = self.http.post(url).json(&vertex_req);
                let request = self.apply_vertex_bearer(request).await?;
                let start = std::time::Instant::now();
                let resp = request
                    .send()
                    .await
                    .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))?;

                if !resp.status().is_success() {
                    return Err(self.map_error_response(resp.status(), resp).await);
                }

                let json: serde_json::Value = resp
                    .json()
                    .await
                    .map_err(|e| ProviderError::Serialization(e.to_string()))?;

                let predictions = json
                    .get("predictions")
                    .and_then(|p| p.as_array())
                    .ok_or_else(|| {
                        ProviderError::Serialization("invalid vertex embedding response".into())
                    })?;

                let items: Vec<EmbedContentItem> = predictions
                    .iter()
                    .enumerate()
                    .filter_map(|(i, p)| {
                        let embed = p.get("embeddings")?;
                        let values = embed.get("values").and_then(|v| v.as_array()).map(|a| {
                            a.iter()
                                .filter_map(|x| x.as_f64().map(|f| f as f32))
                                .collect()
                        })?;
                        let statistics = embed
                            .get("statistics")
                            .and_then(|s| s.get("token_count"))
                            .and_then(|t| t.as_u64())
                            .map(|tc| EmbedStatistics { token_count: tc });
                        if statistics.is_none() {
                            warn!(index = i, "vertex embedding element missing token_count");
                        }
                        Some(EmbedContentItem { values, statistics })
                    })
                    .collect();

                let total_tokens: u64 = items
                    .iter()
                    .map(|e| e.statistics.as_ref().map(|s| s.token_count).unwrap_or(0))
                    .sum();

                Ok(translate::gemini_embedding_to_openai(
                    items,
                    &model,
                    total_tokens,
                ))
            }
        }
    }

    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }

    async fn health_check(&self) -> HealthStatus {
        let model = self.model("gemini-2.0-flash");
        let url = match &self.auth {
            GeminiAuth::ApiKey(_) => {
                let base = gemini_base_url(&self.config);
                format!("{}{}/{}", base, GEMINI_V1BETA, model)
            }
            GeminiAuth::Vertex(t) => {
                let base = vertex_base_url(&self.config, t.location());
                format!(
                    "{}/v1/projects/{}/locations/{}",
                    base,
                    t.project(),
                    t.location()
                )
            }
        };

        let mut req_url = match Url::parse(&url) {
            Ok(u) => u,
            Err(_) => return HealthStatus::Unhealthy,
        };
        self.apply_auth(&mut req_url).await.ok();

        let request = self.http.get(req_url);
        let request = match self.apply_vertex_bearer(request).await {
            Ok(r) => r,
            Err(_) => return HealthStatus::Unhealthy,
        };

        match request.send().await {
            Ok(r) if r.status().is_success() => HealthStatus::Healthy,
            Ok(_r) => HealthStatus::Unhealthy,
            Err(_) => HealthStatus::Unhealthy,
        }
    }
}

impl ProviderAdapterExt for GeminiAdapter {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{GeminiConfig, GeminiMode, SecretString};
    use tracing_test::traced_test;

    fn gemini_config_api(mock_base: &str) -> GeminiConfig {
        GeminiConfig {
            mode: GeminiMode::Api,
            api_key: Some(SecretString::new("test-key")),
            vertex_project: None,
            vertex_location: None,
            vertex_service_account_json: None,
            default_model: Some("gemini-2.0-flash".into()),
            timeout_secs: Some(10),
            api_base_url: Some(mock_base.to_string()),
            vertex_base_url_override: None,
            supported_models: None,
            default_thinking_budget: None,
            embed_api_version: None,
        }
    }

    #[traced_test]
    #[tokio::test]
    async fn test_deprecated_model_triggers_startup_warn() {
        let mock = wiremock::MockServer::start().await;
        let mut config = gemini_config_api(mock.uri().trim_end_matches('/'));
        config.default_model = Some("gemini-2.0-flash".into());
        let _adapter = GeminiAdapter::new(config)
            .await
            .expect("adapter must build");
        assert!(
            logs_contain("deprecated"),
            "must warn when default_model is deprecated"
        );
    }

    #[tokio::test]
    async fn test_supported_models_override_from_config() {
        let mock = wiremock::MockServer::start().await;
        let mut config = gemini_config_api(mock.uri().trim_end_matches('/'));
        config.supported_models = Some(vec!["custom-tuned-gemini-v1".into()]);
        let adapter = GeminiAdapter::new(config)
            .await
            .expect("adapter must build");
        let meta = adapter.metadata();
        assert_eq!(
            meta.supported_models,
            vec!["custom-tuned-gemini-v1"],
            "operator override must be used verbatim"
        );
    }

    #[tokio::test]
    async fn test_supported_models_default_excludes_deprecated() {
        let mock = wiremock::MockServer::start().await;
        let config = gemini_config_api(mock.uri().trim_end_matches('/'));
        let adapter = GeminiAdapter::new(config)
            .await
            .expect("adapter must build");
        let meta = adapter.metadata();
        assert!(
            !meta
                .supported_models
                .contains(&"gemini-2.0-flash".to_string()),
            "deprecated models must be excluded from default list"
        );
        assert!(
            meta.supported_models
                .contains(&"gemini-2.5-pro".to_string()),
            "non-deprecated thinking models must be included"
        );
    }

    #[test]
    fn test_supports_thinking_flag_set_for_thinking_models() {
        let flags_25_pro = model_flags("gemini-2.5-pro");
        assert!(
            flags_25_pro.contains(ModelFlags::THINKING_BUDGET),
            "gemini-2.5-pro must have THINKING_BUDGET"
        );

        let flags_31_pro = model_flags("gemini-3.1-pro-preview");
        assert!(
            flags_31_pro.contains(ModelFlags::THINKING_LEVEL),
            "gemini-3.1-pro-preview must have THINKING_LEVEL"
        );

        let flags_20 = model_flags("gemini-2.0-flash");
        assert!(
            !flags_20.contains(ModelFlags::THINKING_BUDGET)
                && !flags_20.contains(ModelFlags::THINKING_LEVEL),
            "gemini-2.0-flash must not have thinking flags"
        );

        let flags_embed = model_flags("text-embedding-004");
        assert!(
            !flags_embed.contains(ModelFlags::THINKING_BUDGET)
                && !flags_embed.contains(ModelFlags::THINKING_LEVEL),
            "embedding models must not have thinking flags"
        );
    }

    /// build_embed_url uses /v1/models by default for API-key mode.
    #[tokio::test]
    async fn test_build_embed_url_default_api_version() {
        let mock = wiremock::MockServer::start().await;
        let config = gemini_config_api(mock.uri().trim_end_matches('/'));
        let adapter = GeminiAdapter::new(config).await.expect("adapter created");
        let url = adapter
            .build_embed_url("text-embedding-004", "embedContent")
            .expect("url built");
        let url_str = url.to_string();
        assert!(
            url_str.contains("/v1/models/text-embedding-004:embedContent"),
            "default embed URL must use /v1/models: {url_str}"
        );
    }

    /// build_embed_url respects embed_api_version override.
    #[tokio::test]
    async fn test_build_embed_url_custom_api_version() {
        let mock = wiremock::MockServer::start().await;
        let mut config = gemini_config_api(mock.uri().trim_end_matches('/'));
        config.embed_api_version = Some("v1beta".to_string());
        let adapter = GeminiAdapter::new(config).await.expect("adapter created");
        let url = adapter
            .build_embed_url("text-embedding-004", "embedContent")
            .expect("url built");
        let url_str = url.to_string();
        assert!(
            url_str.contains("/v1beta/models/text-embedding-004:embedContent"),
            "custom embed_api_version must be used: {url_str}"
        );
    }

    /// build_embed_url builds batchEmbedContents endpoint correctly.
    #[tokio::test]
    async fn test_build_embed_url_batch_endpoint() {
        let mock = wiremock::MockServer::start().await;
        let config = gemini_config_api(mock.uri().trim_end_matches('/'));
        let adapter = GeminiAdapter::new(config).await.expect("adapter created");
        let url = adapter
            .build_embed_url("text-embedding-004", "batchEmbedContents")
            .expect("url built");
        assert!(
            url.to_string().contains(":batchEmbedContents"),
            "batch endpoint must be in URL"
        );
    }

    /// GEMINI_DEFAULT_TASK_TYPE constant value.
    #[test]
    fn test_gemini_default_task_type_value() {
        assert_eq!(GEMINI_DEFAULT_TASK_TYPE, "RETRIEVAL_DOCUMENT");
    }

    fn embedding_req(inputs: Vec<&str>) -> EmbeddingRequest {
        EmbeddingRequest {
            model: "text-embedding-004".into(),
            input: if inputs.len() == 1 {
                crate::domain::embedding::EmbeddingInput::Single(inputs[0].to_string())
            } else {
                crate::domain::embedding::EmbeddingInput::Batch(
                    inputs.iter().map(|s| s.to_string()).collect(),
                )
            },
            dimensions: None,
            encoding_format: None,
        }
    }

    #[tokio::test]
    async fn test_embeddings_rejects_empty_batch() {
        let mock = wiremock::MockServer::start().await;
        let config = gemini_config_api(mock.uri().trim_end_matches('/'));
        let adapter = GeminiAdapter::new(config)
            .await
            .expect("adapter must build");
        let req = EmbeddingRequest {
            model: "text-embedding-004".into(),
            input: crate::domain::embedding::EmbeddingInput::Batch(vec![]),
            dimensions: None,
            encoding_format: None,
        };
        let result = adapter.embeddings(&req).await;
        assert!(
            matches!(result, Err(ProviderError::InvalidRequest(_))),
            "empty batch must be rejected before any network call"
        );
    }

    #[tokio::test]
    async fn test_embeddings_rejects_all_empty_strings() {
        let mock = wiremock::MockServer::start().await;
        let config = gemini_config_api(mock.uri().trim_end_matches('/'));
        let adapter = GeminiAdapter::new(config)
            .await
            .expect("adapter must build");
        // The adapter guard checks only inputs.is_empty(); Single("") is a 1-element slice,
        // so it passes. The all-empty-string guard lives at the handler level.
        // This test documents that Single("") is not caught by the adapter — handler owns it.
        let req = embedding_req(vec![""]);
        // Single("") produces a 1-element slice; adapter forwards to provider.
        // The test asserts this does NOT return InvalidRequest from the adapter guard.
        let result = adapter.embeddings(&req).await;
        // No server listening → network error, not InvalidRequest.
        assert!(
            !matches!(result, Err(ProviderError::InvalidRequest(_))),
            "adapter should not reject Single(\"\") — handler owns that guard"
        );
    }

    #[tokio::test]
    async fn test_embeddings_rejects_over_limit_apikey_batch() {
        let mock = wiremock::MockServer::start().await;
        let config = gemini_config_api(mock.uri().trim_end_matches('/'));
        let adapter = GeminiAdapter::new(config)
            .await
            .expect("adapter must build");
        let inputs: Vec<&str> = vec!["text"; GEMINI_BATCH_EMBED_MAX_ITEMS + 1];
        let req = embedding_req(inputs);
        let result = adapter.embeddings(&req).await;
        assert!(
            matches!(result, Err(ProviderError::InvalidRequest(_))),
            "over-limit ApiKey batch must be rejected before any network call"
        );
    }

    #[tokio::test]
    async fn test_embeddings_rejects_over_limit_vertex_batch() {
        let mock = wiremock::MockServer::start().await;
        // Borrow metadata from an ApiKey adapter; only auth differs.
        let api_config = gemini_config_api(mock.uri().trim_end_matches('/'));
        let api_adapter = GeminiAdapter::new(api_config.clone())
            .await
            .expect("adapter must build");
        let vertex_adapter = GeminiAdapter {
            config: api_config,
            http: reqwest::Client::new(),
            auth: GeminiAuth::Vertex(VertexOAuthTokens::new_stub()),
            metadata: api_adapter.metadata.clone(),
        };
        let inputs: Vec<&str> = vec!["text"; VERTEX_EMBED_MAX_ITEMS + 1];
        let req = embedding_req(inputs);
        let result = vertex_adapter.embeddings(&req).await;
        assert!(
            matches!(result, Err(ProviderError::InvalidRequest(_))),
            "over-limit Vertex batch must be rejected before any network call"
        );
    }

    /// Gemini metadata has EmbeddingCapabilities with supports_batch.
    #[tokio::test]
    async fn test_gemini_metadata_embedding_capabilities() {
        let mock = wiremock::MockServer::start().await;
        let config = gemini_config_api(mock.uri().trim_end_matches('/'));
        let adapter = GeminiAdapter::new(config).await.expect("adapter created");
        let caps = adapter
            .metadata()
            .embedding_capabilities
            .as_ref()
            .expect("embedding_capabilities must be Some");
        assert!(caps.supports_batch, "Gemini must support batch embeddings");
        assert!(
            !caps.dimensions.is_empty(),
            "Gemini must declare supported dimensions"
        );
    }
}
