// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! OpenAI-compatible provider adapter.
//!
//! Forwards requests to any provider that speaks the OpenAI chat completions wire format
//! (DeepSeek, OpenRouter, Kimi, Qwen, etc.) with zero field transformation.
//! Cost tracking depends entirely on what the upstream provider emits in its response.

mod http;
mod sse;

pub use http::CompatHttpClient;
pub(crate) use sse::make_compat_sse_stream;

use std::sync::Arc;

use async_trait::async_trait;
use secrecy::ExposeSecret;
use serde::Deserialize;
use tracing::warn;

use crate::api::CHAT_COMPLETIONS_PATH;
use crate::config::OpenAICompatConfig;
use crate::domain::chat::{ChatRequest, ChatResponse, Choice, Usage};
use crate::domain::ports::{
    ChatCompletionStream, HealthStatus, ProviderAdapter, ProviderAdapterExt, ProviderError,
    ProviderKind, ProviderMetadata,
};
use crate::providers::openai::utils::{
    COMPAT_DEFAULT_ACCOUNTING, inject_stream_options, normalize_openai_usage,
};
use crate::utils::provider_error::{classify_reqwest_error, sanitize_network_error};

const DEFAULT_COMPAT_TIMEOUT_SECS: u64 = 120;

/// Wraps a non-streaming response body with an optional usage field.
///
/// `ChatResponse.usage` is required; deserializing raw bytes through `ChatResponse`
/// directly would fail when the upstream omits the field. This wrapper accepts absence
/// and lets the adapter emit a zero-cost warning instead of returning a deserialization error.
#[derive(Deserialize)]
struct CompatResponse {
    pub id: Option<String>,
    pub object: Option<String>,
    pub created: Option<i64>,
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<serde_json::Value>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// OpenAI-compatible provider adapter.
///
/// Registered per-instance from `providers.openai_compat[]` config. Zero request
/// transformation — re-serializes the `ChatRequest` (already deserialized by the
/// axum handler) and forwards it verbatim. Scan-only cost extraction from response.
pub struct OpenAICompatAdapter {
    config: OpenAICompatConfig,
    http: Arc<CompatHttpClient>,
    metadata: ProviderMetadata,
    /// Full chat URL: `{base_url}/v1/chat/completions`
    chat_url: String,
}

impl OpenAICompatAdapter {
    /// Constructs the adapter from validated config and a shared HTTP client.
    ///
    /// `http` is Arc-shared across all `openai_compat[]` instances; per-instance
    /// timeout is applied per-request in `build_request`.
    pub async fn new(
        config: OpenAICompatConfig,
        http: Arc<CompatHttpClient>,
    ) -> Result<Self, ProviderError> {
        let base = config.base_url.trim_end_matches('/');
        let chat_url = format!("{base}{CHAT_COMPLETIONS_PATH}");

        let (kind, supported_models) = match &config.supported_models {
            None => (ProviderKind::FallbackOnly, vec!["*".to_string()]),
            Some(ms) => (ProviderKind::Primary, ms.clone()),
        };

        let metadata = ProviderMetadata {
            name: config.name.clone(),
            supported_models,
            supports_streaming: true,
            supports_tools: config.supports_tools,
            supports_vision: false,
            supports_embeddings: false,
            supports_thinking: false,
            kind,
            ..Default::default()
        };

        Ok(Self {
            config,
            http,
            metadata,
            chat_url,
        })
    }

    /// Builds a POST request to the upstream chat URL with per-instance timeout.
    ///
    /// Accepts any body that converts to `reqwest::Body` — both `Vec<u8>` (re-serialize path)
    /// and `bytes::Bytes` (raw-forward path) implement `Into<reqwest::Body>`. The raw path
    /// passes `Bytes::clone()` which is O(1) (Arc refcount inc, no memcopy —).
    fn build_request(&self, body: impl Into<reqwest::Body>) -> reqwest::RequestBuilder {
        let timeout_secs = self
            .config
            .timeout_secs
            .unwrap_or(DEFAULT_COMPAT_TIMEOUT_SECS);
        let mut rb = self
            .http
            .inner
            .post(&self.chat_url)
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .header("Content-Type", "application/json")
            .body(body);
        if let Some(ref key) = self.config.api_key {
            let s = key.expose_secret();
            if !s.is_empty() {
                rb = rb.header("Authorization", format!("Bearer {s}"));
            }
        }
        rb
    }
}

/// Parses a successful compat non-streaming response body into a `ChatResponse`.
///
/// Shared by `chat_completion` and `try_forward_raw` to avoid duplicating the
/// normalization + `Choice` mapping logic.
fn parse_compat_response(
    bytes: &[u8],
    req_model: &str,
    provider_name: &str,
) -> Result<ChatResponse, ProviderError> {
    let compat: CompatResponse =
        serde_json::from_slice(bytes).map_err(|e| ProviderError::Serialization(e.to_string()))?;

    let mut usage = match compat.usage {
        Some(u) => u,
        None => {
            warn!(
                provider = %provider_name,
                "compat non-streaming: upstream returned no usage field; cost will be zero for this request"
            );
            Usage::default()
        }
    };
    normalize_openai_usage(&mut usage, COMPAT_DEFAULT_ACCOUNTING, None);

    let model = compat.model.unwrap_or_else(|| req_model.to_string());
    let choices = compat
        .choices
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            serde_json::from_value::<Choice>(c).map_err(|e| {
                ProviderError::Serialization(format!(
                    "compat({provider_name}): choice[{i}] parse error: {e}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ChatResponse {
        id: compat.id.unwrap_or_default(),
        object: compat
            .object
            .unwrap_or_else(|| "chat.completion".to_string()),
        created: compat.created.unwrap_or(0),
        model,
        choices,
        usage,
    })
}

#[async_trait]
impl ProviderAdapter for OpenAICompatAdapter {
    async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        let body =
            serde_json::to_vec(req).map_err(|e| ProviderError::Serialization(e.to_string()))?;

        let start = std::time::Instant::now();
        let resp = self
            .build_request(body)
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(
                crate::providers::openai::utils::map_openai_error_response(status, resp).await,
            );
        }

        let bytes = resp
            .bytes()
            .await
            .map_err(|e| ProviderError::Unreachable(sanitize_network_error(&e.to_string())))?;

        parse_compat_response(&bytes, &req.model, &self.config.name)
    }

    async fn chat_completion_stream(
        &self,
        req: &ChatRequest,
    ) -> Result<ChatCompletionStream, ProviderError> {
        let mut prepared = req.clone();
        prepared.stream = Some(true);

        if self.config.stream_options_support {
            inject_stream_options(&mut prepared);
        }

        let body = serde_json::to_vec(&prepared)
            .map_err(|e| ProviderError::Serialization(e.to_string()))?;

        let start = std::time::Instant::now();
        let resp = self
            .build_request(body)
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))?;

        if !resp.status().is_success() {
            return Err(crate::providers::openai::utils::map_openai_error_response(
                resp.status(),
                resp,
            )
            .await);
        }

        // No pricing context: a generic backend's cache-write semantics are unverified, so the
        // field is echoed and nothing is accounted from it.
        Ok(make_compat_sse_stream(
            resp,
            self.config.name.clone(),
            COMPAT_DEFAULT_ACCOUNTING,
            None,
        ))
    }

    /// Zero-copy non-streaming forwarding: raw inbound bytes flow directly to upstream.
    ///
    /// `ChatRequest` is immutable from handler entry, so `raw_body` and `req`
    /// are guaranteed consistent. `Bytes::clone()` is O(1).
    async fn try_forward_raw(
        &self,
        req: &ChatRequest,
        raw_body: &bytes::Bytes,
    ) -> Option<Result<ChatResponse, ProviderError>> {
        let start = std::time::Instant::now();
        let resp = match self
            .build_request(raw_body.clone())
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))
        {
            Ok(r) => r,
            Err(e) => return Some(Err(e)),
        };

        let status = resp.status();
        if !status.is_success() {
            return Some(Err(
                crate::providers::openai::utils::map_openai_error_response(status, resp).await,
            ));
        }

        let bytes = match resp
            .bytes()
            .await
            .map_err(|e| ProviderError::Unreachable(sanitize_network_error(&e.to_string())))
        {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };

        Some(parse_compat_response(&bytes, &req.model, &self.config.name))
    }

    /// Zero-copy streaming forwarding: raw inbound bytes flow directly to upstream.
    ///
    /// Returns `None` when stream_options injection is required (`stream_options_support: true`)
    /// or `req.stream != Some(true)` — dispatch falls back to `chat_completion_stream`.
    async fn try_forward_raw_stream(
        &self,
        req: &ChatRequest,
        raw_body: &bytes::Bytes,
    ) -> Option<Result<ChatCompletionStream, ProviderError>> {
        // Raw path only when client already set stream=true AND no injection is needed.
        if self.config.stream_options_support || req.stream != Some(true) {
            return None;
        }

        let start = std::time::Instant::now();
        let resp = match self
            .build_request(raw_body.clone())
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))
        {
            Ok(r) => r,
            Err(e) => return Some(Err(e)),
        };

        if !resp.status().is_success() {
            return Some(Err(
                crate::providers::openai::utils::map_openai_error_response(resp.status(), resp)
                    .await,
            ));
        }

        Some(Ok(make_compat_sse_stream(
            resp,
            self.config.name.clone(),
            COMPAT_DEFAULT_ACCOUNTING,
            None,
        )))
    }

    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }

    async fn health_check(&self) -> HealthStatus {
        // No standardised health endpoint for compat providers. Probe-based health
        // checks are not yet implemented; until then, return Unknown so routing does not
        // treat compat instances as actively-verified-healthy.
        HealthStatus::Unknown
    }
}

impl ProviderAdapterExt for OpenAICompatAdapter {}

#[cfg(test)]
mod tests {
    use super::sse::{CARRY_MAX_BYTES, extract_usage_from_sse_line, parse_sse_data};
    use super::*;
    use crate::domain::chat::{Message, MessageContent, Role, StreamChunk};
    use crate::domain::ports::ProviderError;
    use futures::StreamExt;
    use proptest::prelude::*;
    use tracing_test::traced_test;

    fn make_config(name: &str, stream_options: bool) -> OpenAICompatConfig {
        OpenAICompatConfig {
            name: name.to_string(),
            base_url: "http://localhost:9999".to_string(),
            api_key: Some(crate::config::SecretString::new("sk-test")),
            supported_models: None,
            stream_options_support: stream_options,
            supports_tools: false,
            timeout_secs: Some(5),
        }
    }

    fn make_http() -> Arc<CompatHttpClient> {
        Arc::new(CompatHttpClient::new().expect("test http client"))
    }

    fn minimal_request() -> ChatRequest {
        ChatRequest {
            model: "deepseek-chat".into(),
            messages: vec![Message {
                role: Role::User,
                content: Some(MessageContent::Text("hi".into())),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            stream: None,
            tools: None,
            parallel_tool_calls: None,
            request_id: None,
            extra: Default::default(),
        }
    }

    /// A compat backend's `cache_write_tokens` is echoed back to the client but never priced.
    ///
    /// The two halves are asserted separately because they are separate promises. Adding the
    /// field to `PromptTokensDetails` makes it round-trip through the compat lane, which
    /// deserializes the upstream payload and re-serializes it — faithful passthrough of an
    /// OpenAI-standard field the client asked its own backend for. What must *not* move is the
    /// money: a generic backend's cache-write semantics are unverified, so the compat lane
    /// supplies no pricing generation and the quantity is not accounted, priced, persisted or
    /// counted against a budget. The budget counter takes the finalized total cost and nothing
    /// else, so asserting that total pins it.
    #[test]
    fn compat_echoes_cache_write_tokens_without_pricing_them() {
        use crate::domain::ports::NanoUsd;
        use crate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
        use crate::domain::spend::SpendRecord;
        use crate::utils::cost_headers::build_cost_headers;

        let body = |details: &str| {
            format!(
                r#"{{"id":"c1","object":"chat.completion","created":1,"model":"gpt-5.6-terra",
                     "choices":[],
                     "usage":{{"prompt_tokens":10000,"completion_tokens":500,
                               "total_tokens":10500,"prompt_tokens_details":{details}}}}}"#
            )
        };
        let with_write = parse_compat_response(
            body(r#"{"cached_tokens":2000,"cache_write_tokens":1000}"#).as_bytes(),
            "gpt-5.6-terra",
            "compat",
        )
        .expect("payload parses");
        let without_write = parse_compat_response(
            body(r#"{"cached_tokens":2000}"#).as_bytes(),
            "gpt-5.6-terra",
            "compat",
        )
        .expect("payload parses");

        // Half one: the field reaches the client.
        let echoed = serde_json::to_value(&with_write.usage).expect("usage serializes");
        assert_eq!(
            echoed["prompt_tokens_details"]["cache_write_tokens"],
            serde_json::json!(1000),
            "an OpenAI-standard field the backend reported is passed through"
        );
        assert!(
            serde_json::to_value(&without_write.usage).expect("usage serializes")
                ["prompt_tokens_details"]
                .get("cache_write_tokens")
                .is_none(),
            "and is absent when the backend did not report it"
        );

        // Half two: nothing about the money moves.
        assert_eq!(
            with_write.usage.cache_creation_input_tokens, None,
            "the compat lane accounts no cache-write quantity"
        );
        assert_eq!(with_write.usage.cache_write.observation_count(), 0);

        let holder = Arc::new(std::sync::RwLock::new(
            PricingDb::load(
                BUNDLED_PRICING_JSON,
                &crate::config::PricingConfig::default(),
            )
            .expect("bundled pricing must load"),
        ));
        let priced =
            |usage| build_cost_headers("gpt-5.6-terra", usage, Arc::clone(&holder), false).1;
        let (a, b) = (priced(&with_write.usage), priced(&without_write.usage));

        // Pinned absolutely, not only against each other: two equal-but-wrong totals would
        // satisfy a comparison. `gpt-5.6-terra` base tier, cache-inclusive prompt — 8,000 × 2,000
        // input, 500 × 12,000 output, 2,000 × 200 cache read.
        assert_eq!(a.cost.total_cost, NanoUsd(22_400_000));
        assert_eq!(a.cost.total_cost, b.cost.total_cost);
        assert_eq!(a.cost.cache_write_cost, NanoUsd::zero());
        assert_eq!(a.cost.status, b.cost.status);
        assert_eq!(
            a.cost.status,
            crate::domain::usage_accounting::CostStatus::Exact
        );

        let identity = crate::domain::auth::RequestIdentity {
            id: "key-1".into(),
            org_id: "acme".into(),
            label: None,
            tags: std::collections::HashMap::new(),
        };
        let row =
            |accounting| SpendRecord::build(&identity, "gpt-5.6-terra", "compat", accounting, 1);
        let (row_a, row_b) = (row(&a), row(&b));
        assert_eq!(row_a.prompt_tokens, row_b.prompt_tokens);
        assert_eq!(row_a.completion_tokens, row_b.completion_tokens);
        assert_eq!(row_a.cache_read_tokens, row_b.cache_read_tokens);
        assert_eq!(row_a.thinking_tokens, row_b.thinking_tokens);
        assert_eq!(row_a.cost_nano_usd, row_b.cost_nano_usd);
        assert_eq!(row_a.cost_status, row_b.cost_status);
        assert!(
            row_a.usage_evidence.is_none(),
            "an unaccounted quantity leaves no evidence document behind"
        );
    }

    #[tokio::test]
    async fn new_with_valid_config_builds() {
        let adapter = OpenAICompatAdapter::new(make_config("deepseek", false), make_http())
            .await
            .expect("must build");
        assert_eq!(adapter.metadata().name, "deepseek");
    }

    #[tokio::test]
    async fn fallback_only_when_no_supported_models() {
        let adapter = OpenAICompatAdapter::new(make_config("deepseek", false), make_http())
            .await
            .expect("must build");
        assert_eq!(adapter.metadata().kind, ProviderKind::FallbackOnly);
        assert_eq!(adapter.metadata().supported_models, vec!["*"]);
    }

    #[tokio::test]
    async fn primary_when_supported_models_set() {
        let mut config = make_config("deepseek", false);
        config.supported_models = Some(vec!["deepseek-chat".to_string()]);
        let adapter = OpenAICompatAdapter::new(config, make_http())
            .await
            .expect("must build");
        assert_eq!(adapter.metadata().kind, ProviderKind::Primary);
        assert_eq!(
            adapter.metadata().supported_models,
            vec!["deepseek-chat".to_string()]
        );
    }

    #[tokio::test]
    async fn keyless_no_auth_header() {
        let config = OpenAICompatConfig {
            name: "local".to_string(),
            base_url: "http://localhost:11434".to_string(),
            api_key: None,
            supported_models: None,
            stream_options_support: false,
            supports_tools: false,
            timeout_secs: Some(5),
        };
        let adapter = OpenAICompatAdapter::new(config, make_http())
            .await
            .expect("must build");
        assert!(
            adapter.config.api_key.is_none(),
            "keyless config must produce no api_key"
        );
    }

    #[test]
    fn stream_options_not_injected_when_support_false() {
        let mut req = minimal_request();
        req.stream = Some(true);
        // stream_options_support=false: no injection should happen
        // We verify by checking the request has no stream_options after a simulated prepare
        let config = make_config("deepseek", false);
        // Simulate what the adapter does: only inject when stream_options_support=true
        if config.stream_options_support {
            inject_stream_options(&mut req);
        }
        assert!(
            req.extra.get("stream_options").is_none(),
            "stream_options must not be injected when stream_options_support=false"
        );
    }

    #[test]
    fn stream_options_injected_when_support_true() {
        let mut req = minimal_request();
        req.stream = Some(true);
        let config = make_config("openrouter", true);
        if config.stream_options_support {
            inject_stream_options(&mut req);
        }
        let v = req
            .extra
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(|v| v.as_bool());
        assert_eq!(v, Some(true));
    }

    #[test]
    fn stream_options_respects_client_false_even_when_support_true() {
        let mut req = minimal_request();
        req.extra.insert(
            "stream_options".into(),
            serde_json::json!({"include_usage": false}),
        );
        inject_stream_options(&mut req);
        let v = req
            .extra
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(|v| v.as_bool());
        assert_eq!(v, Some(false), "client false must not be overridden");
    }

    #[test]
    fn extract_usage_from_complete_sse_line() {
        let line = r#"data: {"id":"x","usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let usage = extract_usage_from_sse_line(line, COMPAT_DEFAULT_ACCOUNTING, None)
            .expect("must parse usage");
        assert_eq!(usage.accounting, COMPAT_DEFAULT_ACCOUNTING);
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
    }

    #[test]
    fn extract_usage_returns_none_for_done() {
        assert!(
            extract_usage_from_sse_line("data: [DONE]", COMPAT_DEFAULT_ACCOUNTING, None).is_none()
        );
    }

    #[test]
    fn extract_usage_returns_none_when_usage_null() {
        let line = r#"data: {"id":"x","usage":null}"#;
        assert!(extract_usage_from_sse_line(line, COMPAT_DEFAULT_ACCOUNTING, None).is_none());
    }

    #[test]
    fn parse_sse_data_accepts_no_space_after_colon() {
        // WHATWG SSE spec §9.2.6: "data:" with no trailing space is valid.
        // strip_prefix("data: ") would silently miss this form; strip_prefix("data:") + trim_start() must handle both.
        let json =
            r#"{"id":"x","usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}"#;
        let line = format!("data:{json}");
        let parsed = parse_sse_data(&line).expect("data: without space must parse");
        assert_eq!(parsed["usage"]["prompt_tokens"], 1);
        // Also verify the with-space form still works.
        let line_space = format!("data: {json}");
        let parsed_space = parse_sse_data(&line_space).expect("data: with space must parse");
        assert_eq!(parsed_space["usage"]["prompt_tokens"], 1);
    }

    #[tokio::test]
    async fn utf8_invalid_chunk_is_forwarded_not_dropped() {
        // Regression for the `continue` bug: a chunk that fails UTF-8 decoding must still
        // be yielded to the caller as Ok(bytes). Only the SSE scan is skipped; the bytes go through.
        use async_stream::stream as async_stream_gen;
        use axum::Router;
        use axum::body::Body;
        use axum::http::header;
        use axum::response::Response;
        use axum::routing::post;
        use bytes::Bytes;

        let router = Router::new().route(
            crate::api::CHAT_COMPLETIONS_PATH,
            post(|| async {
                let body = async_stream_gen! {
                    yield Result::<Bytes, std::convert::Infallible>::Ok(Bytes::from_static(
                        b"data: {\"id\":\"c1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
                    ));
                    // 0xFF byte is never valid UTF-8 — triggers the Err(_) arm.
                    yield Ok(Bytes::from(vec![0xFF, 0xFE, 0x80]));
                    yield Ok(Bytes::from_static(
                        b"data: {\"id\":\"c3\",\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}\n\n",
                    ));
                    yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                };
                Response::builder()
                    .status(200)
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .body(Body::from_stream(body))
                    .unwrap()
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test upstream");
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });

        let config = OpenAICompatConfig {
            name: "utf8-test".to_string(),
            base_url: format!("http://127.0.0.1:{port}"),
            api_key: None,
            supported_models: None,
            stream_options_support: false,
            supports_tools: false,
            timeout_secs: Some(5),
        };
        let adapter = OpenAICompatAdapter::new(config, make_http())
            .await
            .expect("must build");
        let mut s = adapter
            .chat_completion_stream(&minimal_request())
            .await
            .expect("stream must open");

        let mut chunks: Vec<bytes::Bytes> = vec![];
        while let Some(item) = s.next().await {
            chunks.push(
                item.expect("no stream error — invalid UTF-8 must not become Err")
                    .data,
            );
        }

        // All 4 upstream bytes chunks must be forwarded.
        assert_eq!(
            chunks.len(),
            4,
            "all chunks must be forwarded, including the invalid UTF-8 one"
        );
        // The invalid UTF-8 bytes must pass through verbatim.
        assert_eq!(chunks[1], bytes::Bytes::from(vec![0xFF, 0xFE, 0x80]));
        // With byte-level carry, the invalid bytes from chunk 1 (no \n) sit in carry and
        // prefix the first line of chunk 2, causing that line's from_utf8 to fail — usage
        // extraction for that contaminated line is skipped. All bytes are still forwarded.
        // A provider sending binary garbage mixed with SSE violates the text-protocol contract;
        // byte forwarding is preserved, SSE extraction is best-effort.
    }

    #[tokio::test]
    async fn health_check_returns_unknown() {
        let adapter = OpenAICompatAdapter::new(make_config("probe", false), make_http())
            .await
            .expect("adapter");
        assert_eq!(adapter.health_check().await, HealthStatus::Unknown);
    }

    // ── Carry-buffer overflow test ────────────────────────────────────────────────────

    #[tokio::test]
    async fn compat_carry_overflow_aborts_stream() {
        // Regression guard: the 1 MiB carry bound must be checked BEFORE extend_from_slice.
        // An upstream that emits > 1 MiB without \n must cause ProviderUnavailable, not OOM.
        use async_stream::stream as async_stream_gen;
        use axum::Router;
        use axum::body::Body;
        use axum::http::header;
        use axum::response::Response;
        use axum::routing::post;
        use bytes::Bytes;

        let oversized = Bytes::from(vec![b'x'; CARRY_MAX_BYTES + 1]);
        let router = Router::new().route(
            crate::api::CHAT_COMPLETIONS_PATH,
            post(move || {
                let chunk = oversized.clone();
                async move {
                    let body = async_stream_gen! {
                        // One chunk exceeding 1 MiB with no newline — triggers the overflow guard.
                        yield Result::<Bytes, std::convert::Infallible>::Ok(chunk);
                    };
                    Response::builder()
                        .status(200)
                        .header(header::CONTENT_TYPE, "text/event-stream")
                        .body(Body::from_stream(body))
                        .unwrap()
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });

        let config = OpenAICompatConfig {
            name: "overflow-test".to_string(),
            base_url: format!("http://127.0.0.1:{port}"),
            api_key: None,
            supported_models: None,
            stream_options_support: false,
            supports_tools: false,
            timeout_secs: Some(5),
        };
        let adapter = OpenAICompatAdapter::new(config, make_http())
            .await
            .expect("must build");
        let mut s = adapter
            .chat_completion_stream(&minimal_request())
            .await
            .expect("stream must open");

        let mut got_overflow = false;
        while let Some(item) = s.next().await {
            if let Err(ProviderError::ProviderUnavailable(_)) = item {
                got_overflow = true;
                break;
            }
        }
        assert!(
            got_overflow,
            "stream must yield ProviderUnavailable when carry overflows 1 MiB"
        );
    }

    // ── Proptest: byte-perfect forwarding under arbitrary chunking ────────────────────

    /// Splits `data` at cumulative step offsets derived from `steps`.
    /// Each u8 step value is treated as a minimum-1-byte advance.
    fn split_at_offsets(data: &[u8], steps: &[u8]) -> Vec<bytes::Bytes> {
        let mut chunks = Vec::new();
        let mut pos = 0usize;
        for &step in steps {
            let advance = (step as usize).max(1);
            let end = (pos + advance).min(data.len());
            if pos < end {
                chunks.push(bytes::Bytes::copy_from_slice(&data[pos..end]));
                pos = end;
            }
            if pos >= data.len() {
                break;
            }
        }
        if pos < data.len() {
            chunks.push(bytes::Bytes::copy_from_slice(&data[pos..]));
        }
        chunks
    }

    fn sse_event_strategy() -> impl Strategy<Value = Vec<u8>> {
        prop::bool::ANY.prop_map(|with_usage| {
            if with_usage {
                b"data: {\"id\":\"u\",\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n"
                    .to_vec()
            } else {
                b"data: {\"id\":\"c\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
                    .to_vec()
            }
        })
    }

    async fn drive_compat_adapter_with_chunks(chunks: Vec<bytes::Bytes>) -> Vec<StreamChunk> {
        use async_stream::stream as async_stream_gen;
        use axum::Router;
        use axum::body::Body;
        use axum::http::header;
        use axum::response::Response;
        use axum::routing::post;
        use bytes::Bytes;

        let router = Router::new().route(
            crate::api::CHAT_COMPLETIONS_PATH,
            post(move || {
                let c = chunks.clone();
                async move {
                    let body = async_stream_gen! {
                        for chunk in c {
                            yield Result::<Bytes, std::convert::Infallible>::Ok(chunk);
                        }
                    };
                    Response::builder()
                        .status(200)
                        .header(header::CONTENT_TYPE, "text/event-stream")
                        .body(Body::from_stream(body))
                        .unwrap()
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });

        let config = OpenAICompatConfig {
            name: "proptest-driver".to_string(),
            base_url: format!("http://127.0.0.1:{port}"),
            api_key: None,
            supported_models: None,
            stream_options_support: false,
            supports_tools: false,
            timeout_secs: Some(5),
        };
        let adapter = OpenAICompatAdapter::new(config, make_http()).await.unwrap();
        let mut s = adapter
            .chat_completion_stream(&minimal_request())
            .await
            .unwrap();
        let mut yielded = vec![];
        while let Some(Ok(chunk)) = s.next().await {
            yielded.push(chunk);
        }
        yielded
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn carry_buffer_preserves_bytes_under_arbitrary_chunking(
            events in prop::collection::vec(sse_event_strategy(), 1..10),
            steps in prop::collection::vec(any::<u8>(), 0..50),
        ) {
            let had_usage = events.iter().any(|e| e.windows(8).any(|w| w == b"\"usage\":{"));
            let full_bytes: Vec<u8> = events.into_iter().flatten().collect();
            let chunks = split_at_offsets(&full_bytes, &steps);

            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let yielded = rt.block_on(drive_compat_adapter_with_chunks(chunks));

            // Invariant 1: byte-perfect forwarding — gateway must not drop or alter any byte.
            let got: Vec<u8> = yielded.iter().flat_map(|c| c.data.to_vec()).collect();
            prop_assert_eq!(got, full_bytes);

            // Invariant 2: if a usage event was in the stream, the last chunk carries Some(usage).
            if had_usage && !yielded.is_empty() {
                prop_assert!(
                    yielded.last().and_then(|c| c.usage.as_ref()).is_some(),
                    "last chunk must carry Some(usage) when a usage event was present"
                );
            }
        }
    }

    /// A compat stream that ends without upstream usage says nothing about it from the lane.
    ///
    /// The missing-usage fact belongs to the single warning a completed request emits at
    /// finalization; a second WARN here would describe the same fact twice — and on a backend that
    /// reports no usage unless asked, and is not asked because `stream_options` injection is off,
    /// on every streamed request it ever serves. The flag governs injection, not parsing: a
    /// backend that volunteers `usage` is read normally with injection disabled, so it is the
    /// backend's behaviour and not the setting that makes this the every-request case.
    #[traced_test]
    #[tokio::test]
    async fn compat_stream_without_usage_emits_no_lane_local_warning() {
        let chunks = vec![
            bytes::Bytes::from_static(
                b"data: {\"id\":\"c\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            ),
            bytes::Bytes::from_static(b"data: [DONE]\n\n"),
        ];

        let yielded = drive_compat_adapter_with_chunks(chunks).await;

        assert!(
            yielded.iter().all(|c| c.usage.is_none()),
            "the fixture must be a genuinely usage-less stream for this assertion to mean anything"
        );
        assert!(
            !logs_contain("upstream returned no usage data"),
            "the missing-usage fact is the finalizer's to report, not the lane's"
        );
    }

    // ── try_forward_raw / try_forward_raw_stream unit tests ─────────────────

    /// Spin up a local axum mock that records the exact bytes received and responds with
    /// a minimal valid OpenAI chat completion JSON. Returns the (port, recorded_body_arc).
    async fn spawn_mock_upstream(
        response_json: &'static str,
    ) -> (u16, std::sync::Arc<tokio::sync::Mutex<bytes::Bytes>>) {
        use axum::extract::Request;
        use axum::{Router, body::Body, http::StatusCode, response::Response, routing::post};
        let captured: std::sync::Arc<tokio::sync::Mutex<bytes::Bytes>> =
            std::sync::Arc::new(tokio::sync::Mutex::new(bytes::Bytes::new()));
        let cap_clone = std::sync::Arc::clone(&captured);
        let router = Router::new().route(
            crate::api::CHAT_COMPLETIONS_PATH,
            post(move |req: Request| {
                let cap = std::sync::Arc::clone(&cap_clone);
                async move {
                    let body_bytes = axum::body::to_bytes(req.into_body(), 50 * 1024 * 1024)
                        .await
                        .expect("mock upstream: failed to read body");
                    *cap.lock().await = body_bytes;
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("Content-Type", "application/json")
                        .body(Body::from(response_json))
                        .expect("mock upstream: response builder")
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        (port, captured)
    }

    const MINIMAL_CHAT_RESPONSE: &str = r#"{
        "id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "deepseek-chat",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    }"#;

    #[tokio::test]
    async fn try_forward_raw_returns_some_and_body_is_byte_for_byte_identical() {
        let (port, captured) = spawn_mock_upstream(MINIMAL_CHAT_RESPONSE).await;
        let config = OpenAICompatConfig {
            name: "test".to_string(),
            base_url: format!("http://127.0.0.1:{port}"),
            api_key: None,
            supported_models: None,
            stream_options_support: false,
            supports_tools: false,
            timeout_secs: Some(5),
        };
        let adapter = OpenAICompatAdapter::new(config, make_http()).await.unwrap();
        let req = minimal_request();
        let raw = bytes::Bytes::from_static(
            b"{\"model\":\"deepseek-chat\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        );

        let result = adapter.try_forward_raw(&req, &raw).await;
        assert!(
            result.is_some(),
            "try_forward_raw must return Some for OpenAICompatAdapter"
        );
        assert!(result.unwrap().is_ok(), "try_forward_raw must succeed");

        let upstream_body = captured.lock().await.clone();
        assert_eq!(
            upstream_body, raw,
            "upstream body must be byte-for-byte identical to raw input"
        );
    }

    #[tokio::test]
    async fn try_forward_raw_default_returns_none_for_translation_adapter() {
        // The ProviderAdapter trait default returns None. Verify this via a minimal mock
        // that does NOT override try_forward_raw (same guarantee all translation adapters get).
        struct DefaultAdapter;

        #[async_trait::async_trait]
        impl ProviderAdapter for DefaultAdapter {
            async fn chat_completion(
                &self,
                _req: &ChatRequest,
            ) -> Result<ChatResponse, ProviderError> {
                Err(ProviderError::NotImplemented)
            }
            fn metadata(&self) -> &ProviderMetadata {
                unimplemented!()
            }
            async fn health_check(&self) -> crate::domain::ports::HealthStatus {
                crate::domain::ports::HealthStatus::Unknown
            }
        }

        let adapter = DefaultAdapter;
        let req = minimal_request();
        let raw = bytes::Bytes::from_static(b"{}");
        // Default impl must return None — translation adapters inherit this.
        assert!(adapter.try_forward_raw(&req, &raw).await.is_none());
        assert!(adapter.try_forward_raw_stream(&req, &raw).await.is_none());
    }

    #[tokio::test]
    async fn try_forward_raw_stream_returns_some_when_stream_true_and_no_options_support() {
        use async_stream::stream as async_stream_gen;
        use axum::extract::Request;
        use axum::{Router, body::Body, http::header, response::Response, routing::post};
        use bytes::Bytes;

        let captured: std::sync::Arc<tokio::sync::Mutex<Bytes>> =
            std::sync::Arc::new(tokio::sync::Mutex::new(Bytes::new()));
        let cap_clone = std::sync::Arc::clone(&captured);

        let router = Router::new().route(
            crate::api::CHAT_COMPLETIONS_PATH,
            post(move |req: Request| {
                let cap = std::sync::Arc::clone(&cap_clone);
                async move {
                    let body_bytes = axum::body::to_bytes(req.into_body(), 50 * 1024 * 1024)
                        .await
                        .expect("mock upstream: failed to read body");
                    *cap.lock().await = body_bytes;
                    let body = async_stream_gen! {
                        yield Result::<Bytes, std::convert::Infallible>::Ok(Bytes::from_static(
                            b"data: {\"id\":\"s1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
                        ));
                        yield Ok(Bytes::from_static(
                            b"data: {\"id\":\"s2\",\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n",
                        ));
                        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                    };
                    Response::builder()
                        .status(200)
                        .header(header::CONTENT_TYPE, "text/event-stream")
                        .body(Body::from_stream(body))
                        .unwrap()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });

        let config = OpenAICompatConfig {
            name: "stream-raw-test".to_string(),
            base_url: format!("http://127.0.0.1:{port}"),
            api_key: None,
            supported_models: None,
            stream_options_support: false, // raw path eligible
            supports_tools: false,
            timeout_secs: Some(5),
        };
        let adapter = OpenAICompatAdapter::new(config, make_http()).await.unwrap();
        let mut req = minimal_request();
        req.stream = Some(true); // client requested streaming

        let raw =
            Bytes::from_static(b"{\"model\":\"deepseek-chat\",\"messages\":[],\"stream\":true}");
        let result = adapter.try_forward_raw_stream(&req, &raw).await;
        assert!(
            result.is_some(),
            "must return Some when stream=true and stream_options_support=false"
        );
        let mut stream = result.unwrap().expect("stream must open");

        let mut all_data: Vec<Bytes> = vec![];
        while let Some(Ok(chunk)) = stream.next().await {
            all_data.push(chunk.data);
        }
        assert!(!all_data.is_empty(), "must receive at least one chunk");

        // Upstream received the original raw bytes (not re-serialized).
        let upstream_body = captured.lock().await.clone();
        assert_eq!(
            upstream_body, raw,
            "upstream body must be byte-for-byte identical to raw input"
        );
    }

    #[tokio::test]
    async fn try_forward_raw_stream_returns_none_when_stream_options_support_true() {
        let config = make_config("openrouter", true); // stream_options_support = true
        let adapter = OpenAICompatAdapter::new(config, make_http()).await.unwrap();
        let mut req = minimal_request();
        req.stream = Some(true);
        let raw = bytes::Bytes::from_static(b"{\"model\":\"x\",\"messages\":[],\"stream\":true}");
        let result = adapter.try_forward_raw_stream(&req, &raw).await;
        assert!(
            result.is_none(),
            "must return None when stream_options_support=true"
        );
    }

    #[tokio::test]
    async fn try_forward_raw_stream_returns_none_when_stream_not_true() {
        let config = make_config("deepseek", false); // stream_options_support = false
        let adapter = OpenAICompatAdapter::new(config, make_http()).await.unwrap();
        let req = minimal_request(); // req.stream = None
        let raw = bytes::Bytes::from_static(b"{\"model\":\"x\",\"messages\":[]}");
        let result = adapter.try_forward_raw_stream(&req, &raw).await;
        assert!(
            result.is_none(),
            "must return None when req.stream != Some(true)"
        );
    }

    // ── regression guard ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn compat_http_client_is_shared_across_instances() {
        // all compat adapters must be constructed with a shared Arc<CompatHttpClient>,
        // not create their own per-instance client. Arc::ptr_eq guards this invariant.
        let http = Arc::new(CompatHttpClient::new().expect("http"));
        let a = OpenAICompatAdapter::new(make_config("a", false), Arc::clone(&http))
            .await
            .expect("adapter a");
        let b = OpenAICompatAdapter::new(make_config("b", false), Arc::clone(&http))
            .await
            .expect("adapter b");
        assert!(
            Arc::ptr_eq(&a.http, &b.http),
            "both adapters must reference the same CompatHttpClient Arc"
        );
    }

    // -----------------------------------------------------------------------
    // Cache-write accounting at the adapter seam — the negative half
    //
    // A generic compat backend speaks the OpenAI wire format; that proves what fields it emits,
    // not how it counts them. So `cache_write_tokens` is echoed and nothing is accounted from it.
    // Asserted here, at the adapter, because a helper test cannot distinguish a lane that passes
    // no pricing context from one that was simply never wired to a backend.
    // -----------------------------------------------------------------------

    use crate::providers::openai::utils::cache_write_fixture as fixture;

    fn compat_config_for(mock_base: &str) -> OpenAICompatConfig {
        let mut cfg = make_config("compat-cache-probe", false);
        cfg.base_url = mock_base.to_string();
        cfg
    }

    fn cache_write_request() -> ChatRequest {
        let mut req = minimal_request();
        req.model = fixture::MODEL.to_string();
        req
    }

    /// A compat instance echoes the field and prices nothing from it.
    ///
    /// The response body *does* change relative to before the field existed on
    /// `PromptTokensDetails` — that is faithful passthrough of an OpenAI-standard field the
    /// client asked its own backend for. What must not change is the money: cost, status and the
    /// quantities every downstream surface is derived from.
    #[tokio::test]
    async fn a_compat_instance_echoes_a_cache_write_without_accounting_it() {
        let mock = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path(crate::api::CHAT_COMPLETIONS_PATH))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": fixture::MODEL,
                    "choices": [],
                    "usage": fixture::usage_json(),
                })),
            )
            .mount(&mock)
            .await;

        let adapter = OpenAICompatAdapter::new(compat_config_for(&mock.uri()), make_http())
            .await
            .expect("adapter must build");
        let resp = adapter
            .chat_completion(&cache_write_request())
            .await
            .expect("chat must succeed");

        assert_eq!(
            resp.usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cache_write_tokens),
            Some(1_000),
            "the upstream's own field is passed through to the client"
        );
        fixture::assert_unaccounted_and_unbilled(&resp.usage);
    }

    /// The same on the streamed path, which shares `make_compat_sse_stream` with Azure.
    ///
    /// Sharing that helper is exactly why this assertion is worth its cost: the helper now
    /// carries a per-lane pricing context, and compat's is `None`.
    #[tokio::test]
    async fn a_compat_stream_echoes_a_cache_write_without_accounting_it() {
        let mock = wiremock::MockServer::start().await;
        let sse = format!(
            "data: {}\n\ndata: [DONE]\n\n",
            serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "model": fixture::MODEL,
                "choices": [],
                "usage": fixture::usage_json(),
            })
        );
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path(crate::api::CHAT_COMPLETIONS_PATH))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_raw(sse, "text/event-stream")
                    .insert_header("Content-Type", "text/event-stream"),
            )
            .mount(&mock)
            .await;

        let adapter = OpenAICompatAdapter::new(compat_config_for(&mock.uri()), make_http())
            .await
            .expect("adapter must build");
        let mut stream = adapter
            .chat_completion_stream(&cache_write_request())
            .await
            .expect("stream must open");

        let mut last_usage: Option<Usage> = None;
        while let Some(chunk) = stream.next().await {
            if let Some(u) = chunk.expect("no stream error").usage {
                last_usage = Some(u);
            }
        }

        fixture::assert_unaccounted_and_unbilled(
            &last_usage.expect("the terminal chunk carries usage"),
        );
    }

    // ── Terminal-chunk marking ────────────────────────────────────────────────────────
    //
    // The flag is derived from the carry buffer's line reassembly, never from the raw bytes of
    // the chunk being yielded. A network read is not an SSE line: the terminator can arrive split
    // across two reads or bundled behind a data frame, so a byte-suffix test on the yielded chunk
    // is wrong in both directions. Nor is a line an event — the terminator is complete only once
    // the blank line closing it has been read. These cases drive `make_compat_sse_stream` over a
    // body whose read boundaries are fixed by the test rather than by the kernel, which is what
    // makes the split case reproducible at all.

    /// Feeds an exact sequence of network reads through the compat SSE stream.
    ///
    /// The response is assembled in-process from the given frames, so each element of `reads`
    /// is delivered as one `bytes_stream` item. Driving the adapter over a socket would let the
    /// transport coalesce the frames and silently turn the split case into the whole-line case.
    async fn compat_chunks_for_reads(reads: &[&'static [u8]]) -> Vec<StreamChunk> {
        let frames: Vec<Result<bytes::Bytes, std::io::Error>> = reads
            .iter()
            .map(|r| Ok(bytes::Bytes::from_static(r)))
            .collect();
        let body = reqwest::Body::wrap_stream(futures::stream::iter(frames));
        let resp = reqwest::Response::from(axum::http::Response::new(body));

        let mut stream = super::sse::make_compat_sse_stream(
            resp,
            "marking-test".to_string(),
            COMPAT_DEFAULT_ACCOUNTING,
            None,
        );
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            out.push(item.expect("no stream error"));
        }
        out
    }

    /// The one chunk whose read completed the terminator is terminal; nothing before it is.
    fn assert_only_last_is_final(chunks: &[StreamChunk]) {
        let marked: Vec<usize> = chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.is_final)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            marked,
            vec![chunks.len() - 1],
            "exactly the last chunk must be terminal, got marks at {marked:?} of {} chunks",
            chunks.len()
        );
    }

    #[tokio::test]
    async fn compat_marks_the_read_carrying_a_whole_done_line() {
        let chunks = compat_chunks_for_reads(&[
            b"data: {\"id\":\"c1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            b"data: [DONE]\n\n",
        ])
        .await;

        assert_eq!(chunks.len(), 2, "one chunk per read");
        assert_only_last_is_final(&chunks);
    }

    /// A terminator split across two reads is only a terminator once the second read completes
    /// the line. A suffix test on the yielded bytes would see `NE]\n\n` and miss it entirely.
    #[tokio::test]
    async fn compat_marks_the_read_that_completes_a_split_done_line() {
        let chunks = compat_chunks_for_reads(&[
            b"data: {\"id\":\"c1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DO",
            b"NE]\n\n",
        ])
        .await;

        assert_eq!(chunks.len(), 2, "one chunk per read");
        assert_only_last_is_final(&chunks);
    }

    /// A terminator bundled behind a usage frame in one read marks that read, and the usage the
    /// same read reported still rides on it — the marking must not displace the accounting.
    #[tokio::test]
    async fn compat_marks_a_bundled_done_read_and_keeps_its_usage() {
        let chunks = compat_chunks_for_reads(&[
            b"data: {\"id\":\"c1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            b"data: {\"id\":\"c2\",\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\ndata: [DONE]\n\n",
        ])
        .await;

        assert_eq!(chunks.len(), 2, "one chunk per read");
        assert_only_last_is_final(&chunks);
        let usage = chunks[1]
            .usage
            .as_ref()
            .expect("the bundled read reported usage");
        assert_eq!(usage.prompt_tokens, 7);
        assert_eq!(usage.completion_tokens, 3);
    }

    /// The terminator event ends at the blank line after the marker, not at the marker's own
    /// newline. When the separator lands in the next read, that read is the terminal one — and it
    /// carries only the delimiter. Marking the earlier read would stop the stream with the event
    /// still open, and the gateway's trailing event would merge into it.
    #[tokio::test]
    async fn compat_marks_the_read_that_closes_the_done_event() {
        let chunks = compat_chunks_for_reads(&[b"data: [DONE]\n", b"\n"]).await;

        assert_eq!(chunks.len(), 2, "one chunk per read");
        assert_only_last_is_final(&chunks);
    }

    /// Only an empty line dispatches an event. A line of spaces is an unrecognized field line,
    /// which the spec ignores and which leaves the terminator open — so the read carrying it is
    /// not the terminal one, and the read carrying the real delimiter after it is.
    #[tokio::test]
    async fn compat_does_not_close_the_done_event_on_a_whitespace_line() {
        let chunks = compat_chunks_for_reads(&[b"data: [DONE]\n", b" \n", b"\n"]).await;

        assert_eq!(chunks.len(), 3, "one chunk per read");
        assert_only_last_is_final(&chunks);
    }

    /// An event's payload is the concatenation of all its `data:` lines, so an event that merely
    /// *ends* with the marker is not a bare terminator. Marking it would end the response with
    /// the content of that same event still unaccounted for — the handler stops at a terminal
    /// chunk, so everything after it is dropped.
    #[tokio::test]
    async fn compat_does_not_mark_a_multi_line_event_ending_in_the_marker() {
        let chunks = compat_chunks_for_reads(&[
            b"data: {\"id\":\"c1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\ndata: [DONE]\n\n",
        ])
        .await;

        assert!(
            chunks.iter().all(|c| !c.is_final),
            "an event whose payload is more than the bare marker is not the terminator"
        );
    }

    /// A stream that never sends the terminator marks nothing. The handler's fallback covers it;
    /// asserting a completion here would be a lie about an upstream that simply stopped.
    #[tokio::test]
    async fn compat_marks_nothing_when_the_stream_ends_without_a_done_line() {
        let chunks = compat_chunks_for_reads(&[
            b"data: {\"id\":\"c1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        ])
        .await;

        assert!(
            chunks.iter().all(|c| !c.is_final),
            "no chunk may claim a completion the upstream never signalled"
        );
    }
}
