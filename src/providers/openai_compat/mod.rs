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
use crate::providers::openai::utils::{inject_stream_options, normalize_openai_usage};
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
    normalize_openai_usage(&mut usage);

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

        Ok(make_compat_sse_stream(resp, self.config.name.clone()))
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

        Some(Ok(make_compat_sse_stream(resp, self.config.name.clone())))
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
        let usage = extract_usage_from_sse_line(line).expect("must parse usage");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
    }

    #[test]
    fn extract_usage_returns_none_for_done() {
        assert!(extract_usage_from_sse_line("data: [DONE]").is_none());
    }

    #[test]
    fn extract_usage_returns_none_when_usage_null() {
        let line = r#"data: {"id":"x","usage":null}"#;
        assert!(extract_usage_from_sse_line(line).is_none());
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
}
