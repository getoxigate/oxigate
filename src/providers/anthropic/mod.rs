// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Anthropic Claude provider adapter.
//!
//! Translates OpenAI-compatible requests to Anthropic Messages API.
//! Supports chat completions (stream + non-stream), tool use, cache tokens, extended thinking.
//!
//! Observability: Model and cost are logged at the handler layer (api/chat.rs) — handler has
//! #[tracing::instrument] with model field; non-streaming logs cost/prompt_tokens/completion_tokens;
//! streaming emits oxigate.usage SSE events with cost headers.

mod translate;
mod types;

use async_trait::async_trait;
use futures::StreamExt;
use secrecy::ExposeSecret;
use tokio::io::{AsyncBufReadExt, BufReader};

use std::sync::{Arc, RwLock};

use crate::config::AnthropicConfig;
use crate::domain::chat::{ChatRequest, ChatResponse};
use crate::domain::ports::{
    ChatCompletionStream, HealthStatus, ProviderAdapter, ProviderAdapterExt, ProviderError,
    ProviderKind, ProviderMetadata,
};
use serde::de::DeserializeSeed;

use crate::domain::pricing::{PricingDb, snapshot_pricing_context};
use crate::domain::usage_accounting::{CacheWriteAccumulator, PricingContext};
use crate::providers::anthropic::translate::{
    DEFAULT_MAX_TOKENS, StreamErr, StreamTranslator, anthropic_to_chat_response,
    chat_request_to_anthropic, overflow_sse_event,
};
use crate::providers::anthropic::types::{MessagesRequest, MessagesResponse, MessagesResponseSeed};
use crate::providers::tool_limits::DEFAULT_TOOL_CALL_BUFFER_CAP_BYTES;
use crate::utils::provider_error::classify_reqwest_error;

const ANTHROPIC_API_BASE: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION_DEFAULT: &str = "2023-06-01";
const ANTHROPIC_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

/// Known Anthropic models with capability flags. (model_id, supports_thinking)
///
/// Entries removed here because Anthropic has retired them
/// (platform.claude.com/docs/en/about-claude/model-deprecations) — requests to a retired
/// model fail outright, so advertising it via /v1/models publishes false capability metadata:
/// claude-sonnet-4-20250514, claude-opus-4-20250514 (retired 2026-06-15), claude-opus-4-1-20250805
/// (retired 2026-08-05), claude-3-7-sonnet-20250219, claude-3-5-haiku-20241022 (retired
/// 2026-02-19), claude-3-5-sonnet-20241022 (retired 2025-10-28), claude-3-opus-20240229 (retired
/// 2026-01-05), claude-3-haiku-20240307 (retired 2026-04-20).
const KNOWN_ANTHROPIC_MODELS: &[(&str, bool)] = &[
    // ── Current generation ──────────────────────────────────────────────────
    ("claude-opus-4-6", true),
    ("claude-sonnet-4-6", true),
    ("claude-haiku-4-5-20251001", true),
    // ── Legacy (still available — migrate when convenient) ───────────────────
    ("claude-sonnet-4-5-20250929", true),
    ("claude-opus-4-5-20251101", true),
];

fn anthropic_base_url(config: &AnthropicConfig) -> String {
    config
        .api_base_url
        .clone()
        .unwrap_or_else(|| ANTHROPIC_API_BASE.to_string())
}

fn default_model(config: &AnthropicConfig) -> String {
    config
        .default_model
        .clone()
        .unwrap_or_else(|| "claude-sonnet-4-6".to_string())
}

fn default_max_tokens(config: &AnthropicConfig) -> u32 {
    config.default_max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)
}

/// Anthropic Claude provider adapter.
pub struct AnthropicAdapter {
    config: AnthropicConfig,
    http: reqwest::Client,
    metadata: ProviderMetadata,
    api_key: String,
    /// Effective tool-call streaming buffer cap — resolved once at construction.
    cap_bytes: usize,
    /// Hot-reload pricing holder. Each request clones one generation out of it before dispatch,
    /// so the cache-write class registry a response is accounted against and the rates it is
    /// priced with come from the same pricing database.
    pricing_db: Arc<RwLock<PricingDb>>,
}

impl AnthropicAdapter {
    /// Creates an Anthropic adapter. Config must be validated by GatewayConfig::validate.
    ///
    /// `pricing_db` is the same hot-reload holder the cost calculator reads; the adapter needs it
    /// to snapshot one pricing generation per request before dispatch.
    pub async fn new(
        config: AnthropicConfig,
        pricing_db: Arc<RwLock<PricingDb>>,
    ) -> Result<Self, ProviderError> {
        let api_key = config
            .api_key
            .as_ref()
            .and_then(|k| {
                let s = k.expose_secret();
                if s.is_empty() { None } else { Some(s.clone()) }
            })
            .ok_or_else(|| ProviderError::Auth("providers.anthropic.api_key is required".into()))?;

        let timeout_secs = config.timeout_secs.unwrap_or(120);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| ProviderError::Unreachable(format!("reqwest client: {}", e)))?;

        let supported_models = config.supported_models.clone().unwrap_or_else(|| {
            let mut models: Vec<String> = KNOWN_ANTHROPIC_MODELS
                .iter()
                .map(|(id, _)| (*id).to_string())
                .collect();
            models.push("claude-*".to_string());
            models
        });

        let metadata = ProviderMetadata {
            name: "anthropic".to_string(),
            supported_models,
            supports_streaming: true,
            supports_tools: true,
            supports_vision: false,
            supports_embeddings: false,
            supports_thinking: true,
            kind: ProviderKind::Primary,
            ..Default::default()
        };

        let cap_bytes = config
            .tool_call_buffer_cap_bytes
            .unwrap_or(DEFAULT_TOOL_CALL_BUFFER_CAP_BYTES);

        Ok(Self {
            config,
            http: client,
            metadata,
            api_key,
            cap_bytes,
            pricing_db,
        })
    }

    /// Snapshots this request's pricing generation.
    ///
    /// Called once per request, before the upstream request is dispatched, on the buffered path
    /// as well as the streaming one: a buffered request dispatched under one database but
    /// completing after a reload must not price against the next one while an equivalent stream
    /// prices against the first.
    fn pricing_context(&self) -> PricingContext {
        snapshot_pricing_context(&self.pricing_db)
    }

    fn model(&self, req_model: &str) -> String {
        if req_model.is_empty() {
            default_model(&self.config)
        } else {
            req_model.to_string()
        }
    }

    fn build_url(&self) -> String {
        let base = anthropic_base_url(&self.config);
        let base = base.trim_end_matches('/');
        format!("{}/v1/messages", base)
    }

    fn build_base_request(&self, body: &MessagesRequest, stream: bool) -> reqwest::RequestBuilder {
        let url = self.build_url();
        let accept = if stream {
            "text/event-stream"
        } else {
            "application/json"
        };
        let mut request = self
            .http
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header(
                "anthropic-version",
                self.config
                    .anthropic_version
                    .as_deref()
                    .unwrap_or(ANTHROPIC_VERSION_DEFAULT),
            )
            .header("content-type", "application/json")
            .header("accept", accept)
            .json(body);

        if body.thinking.is_some() {
            request = request.header("anthropic-beta", ANTHROPIC_THINKING_BETA);
        }
        request
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
            529 => ProviderError::ProviderUnavailable(format!("anthropic overloaded: {msg}")),
            500 | 502 | 503 => ProviderError::ProviderUnavailable(msg),
            _ => ProviderError::ProviderHttpError {
                status: status.as_u16(),
                body,
            },
        }
    }
}

#[async_trait]
impl ProviderAdapter for AnthropicAdapter {
    async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        let model = self.model(&req.model);
        let request_id = req.request_id.as_deref().unwrap_or("unknown");

        let anthropic_req = chat_request_to_anthropic(
            req,
            &default_model(&self.config),
            default_max_tokens(&self.config),
        )?;

        // Acquired before dispatch, matching the streaming path: a buffered request completing
        // after a reload must not price against a generation it was never dispatched under.
        let pricing_context = self.pricing_context();

        let request = self.build_base_request(&anthropic_req, false);
        let start = std::time::Instant::now();
        let resp = request
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(self.map_error_response(status, resp).await);
        }

        // Driven by a seed rather than `resp.json()`: the cache-write detail object has to be
        // credited into this request's accumulator as it is read, and a derived `Deserialize`
        // never invokes a seed.
        let body = resp
            .bytes()
            .await
            .map_err(|e| ProviderError::Serialization(e.to_string()))?;
        // The seed proposes the cache-write snapshot rather than crediting it: a body with
        // trailing data after it is not a response, and nothing it stated may reach billing.
        let (anthropic_resp, cache_write): (MessagesResponse, _) = {
            let mut de = serde_json::Deserializer::from_slice(&body);
            let parsed = MessagesResponseSeed::new(pricing_context.registry())
                .deserialize(&mut de)
                .map_err(|e| ProviderError::Serialization(e.to_string()))?;
            de.end()
                .map_err(|e| ProviderError::Serialization(e.to_string()))?;
            (parsed.response, parsed.cache_write)
        };
        let accumulator = cache_write
            .unwrap_or_else(|| CacheWriteAccumulator::new(pricing_context.registry().clone()));

        anthropic_to_chat_response(
            &anthropic_resp,
            &model,
            request_id,
            self.cap_bytes,
            &pricing_context,
            accumulator,
        )
    }

    async fn chat_completion_stream(
        &self,
        req: &ChatRequest,
    ) -> Result<ChatCompletionStream, ProviderError> {
        let model = self.model(&req.model);
        let request_id = req.request_id.as_deref().unwrap_or("unknown").to_string();

        let mut anthropic_req = chat_request_to_anthropic(
            req,
            &default_model(&self.config),
            default_max_tokens(&self.config),
        )?;
        anthropic_req.stream = Some(true);

        // Acquired before dispatch: the generation this response is accounted against must be the
        // one the request started under, not whichever database a mid-flight reload installed.
        let pricing_context = self.pricing_context();

        let request = self.build_base_request(&anthropic_req, true);
        let start = std::time::Instant::now();
        let resp = request
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))?;

        if !resp.status().is_success() {
            return Err(self.map_error_response(resp.status(), resp).await);
        }

        let model = model.clone();
        let stream = resp
            .bytes_stream()
            .map(|r| r.map_err(|e: reqwest::Error| std::io::Error::other(e.to_string())));

        let reader = BufReader::new(tokio_util::io::StreamReader::new(stream));
        let mut lines = reader.lines();

        let cap_bytes = self.cap_bytes;
        let mut translator = StreamTranslator::new(model, request_id, cap_bytes, pricing_context);

        let output = async_stream::stream! {
            while let Ok(Some(line)) = lines.next_line().await {
                if let Some(event) = translator.parse_event(&line) {
                    match translator.process_event(&event) {
                        Ok(Some(chunk)) => yield Ok(chunk),
                        Ok(None) => {}
                        Err(StreamErr::ProviderError(msg)) => {
                            let err_msg = msg.unwrap_or_else(|| "unknown stream error".into());
                            yield Err(ProviderError::ProviderUnavailable(err_msg));
                            return;
                        }
                        Err(StreamErr::BufferOverflow(e)) => {
                            // Mid-stream: headers already committed; emit terminal SSE event.
                            yield Ok(overflow_sse_event(&e));
                            return;
                        }
                    }
                }
            }
        };

        Ok(Box::pin(output))
    }

    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }

    async fn health_check(&self) -> HealthStatus {
        let url = anthropic_base_url(&self.config);
        let url = url.trim_end_matches('/');
        match self.http.head(url).send().await {
            Ok(resp) => {
                if resp.status().is_success() {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Degraded
                }
            }
            Err(_) => HealthStatus::Unhealthy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::KNOWN_ANTHROPIC_MODELS;

    /// The default advertised catalogue (`GET /v1/models` and provider dispatch both read
    /// `KNOWN_ANTHROPIC_MODELS`) must never list a model Anthropic has retired — a retired
    /// model's requests fail outright, so listing it publishes false capability metadata.
    /// This asserts against every model this file has confirmed retired via
    /// platform.claude.com/docs/en/about-claude/model-deprecations, so a future re-add of any
    /// of them fails this test instead of shipping silently.
    #[test]
    fn known_anthropic_models_excludes_retired_models() {
        const RETIRED: &[&str] = &[
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-opus-4-1-20250805",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ];
        for (id, _) in KNOWN_ANTHROPIC_MODELS {
            assert!(
                !RETIRED.contains(id),
                "KNOWN_ANTHROPIC_MODELS advertises retired model {id}"
            );
        }
    }
}

impl ProviderAdapterExt for AnthropicAdapter {}
