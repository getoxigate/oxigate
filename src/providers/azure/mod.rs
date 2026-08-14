// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Azure OpenAI provider adapter .
//!
//! Deployment-based URL construction, `api-key` header auth, and always-on
//! `stream_options.include_usage: true` injection for non-zero streaming cost.
//! Embeddings deferred to; tool use/vision deferred to.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use secrecy::ExposeSecret;
use serde::Deserialize;
use tracing::warn;

use crate::config::AzureConfig;
use crate::domain::chat::{ChatRequest, ChatResponse, Choice, Usage};
use crate::domain::ports::{
    ChatCompletionStream, HealthStatus, ProviderAdapter, ProviderAdapterExt, ProviderError,
    ProviderKind, ProviderMetadata,
};
use crate::providers::openai::utils::{map_status_to_provider_error, normalize_openai_usage};
use crate::providers::openai_compat::{CompatHttpClient, make_compat_sse_stream};
use crate::utils::provider_error::{classify_reqwest_error, sanitize_network_error};

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const ERROR_BODY_CAP: usize = 4 * 1024;

/// Azure OpenAI provider adapter.
///
/// One instance per `azure[]` config entry. Builds deployment-based URLs,
/// injects `api-key` auth (never `Authorization: Bearer`), and always injects
/// `stream_options.include_usage: true` so streaming cost is non-zero.
pub struct AzureAdapter {
    config: AzureConfig,
    http: Arc<CompatHttpClient>,
    metadata: ProviderMetadata,
    /// Pre-computed: `{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}`
    chat_url: String,
}

impl AzureAdapter {
    /// Constructs the adapter from validated config and a shared HTTP client.
    ///
    /// `async` is intentional: managed-identity token fetch at startup (deferred) will
    /// require an async call here. Keeping the signature async avoids a breaking change
    /// to all call sites when that feature lands.
    pub async fn new(
        config: AzureConfig,
        http: Arc<CompatHttpClient>,
    ) -> Result<Self, ProviderError> {
        let chat_url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            config.endpoint.trim_end_matches('/'),
            config.deployment_name,
            config.api_version,
        );

        let (kind, supported_models) = match &config.supported_models {
            None => (ProviderKind::FallbackOnly, vec!["*".to_string()]),
            Some(ms) => (ProviderKind::Primary, ms.clone()),
        };

        let metadata = ProviderMetadata {
            name: config.name.clone(),
            supported_models,
            supports_streaming: true,
            supports_tools: false,      //
            supports_vision: false,     //
            supports_embeddings: false, //
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

    fn build_request(&self, body: impl Into<reqwest::Body>) -> reqwest::RequestBuilder {
        let timeout_secs = self.config.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS);
        self.http
            .inner
            .post(&self.chat_url)
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .header("Content-Type", "application/json")
            // Azure uses api-key; never set Authorization alongside it.
            .header("api-key", self.config.api_key.expose_secret())
            .body(body)
    }
}

/// Parses a successful Azure non-streaming response body into a `ChatResponse`.
fn parse_response(
    bytes: &[u8],
    req_model: &str,
    provider_name: &str,
) -> Result<ChatResponse, ProviderError> {
    // `usage` is optional — Azure can omit it; the wrapper lets us emit a zero-cost warning
    // instead of returning a deserialization error. `choices` uses `Vec<Choice>` directly
    // (single-pass) because Azure's choice format is identical to the domain type.
    #[derive(Deserialize)]
    struct AzureResponse {
        pub id: Option<String>,
        pub object: Option<String>,
        pub created: Option<i64>,
        pub model: Option<String>,
        #[serde(default)]
        pub choices: Vec<Choice>,
        #[serde(default)]
        pub usage: Option<Usage>,
    }

    let parsed: AzureResponse =
        serde_json::from_slice(bytes).map_err(|e| ProviderError::Serialization(e.to_string()))?;

    let mut usage = match parsed.usage {
        Some(u) => u,
        None => {
            warn!(
                provider = %provider_name,
                "azure non-streaming: upstream returned no usage field; cost will be zero for this request"
            );
            Usage::default()
        }
    };
    normalize_openai_usage(&mut usage);

    Ok(ChatResponse {
        id: parsed.id.unwrap_or_default(),
        object: parsed
            .object
            .unwrap_or_else(|| "chat.completion".to_string()),
        created: parsed.created.unwrap_or(0),
        model: parsed.model.unwrap_or_else(|| req_model.to_string()),
        choices: parsed.choices,
        usage,
    })
}

/// Classifies a pre-read Azure error body into a [`ProviderError`].
///
/// Separated from the async body-reader so tests can drive it with raw bytes
/// without constructing a live `reqwest::Response`.
///
/// Azure content-filter codes checked (must precede the generic 400 branch):
/// - `"content_filter"` — streaming-path block (OWASP A10)
/// - `"content_policy_violation"` — non-streaming structured error
/// - `inner_error.code: "ResponsibleAIPolicyViolation"` — legacy inner-error field
/// - `innererror.code` — older API alias for inner_error
fn classify_azure_error_body(
    status: reqwest::StatusCode,
    body: &[u8],
    retry_after: Option<u64>,
) -> ProviderError {
    let val = serde_json::from_slice::<serde_json::Value>(body).unwrap_or_default();

    // Must come before the generic 400 branch (which maps to InvalidRequest — retryable).
    if status == reqwest::StatusCode::BAD_REQUEST {
        let error_obj = &val["error"];
        let error_code = error_obj["code"].as_str().unwrap_or("");
        let inner_code = error_obj["inner_error"]["code"]
            .as_str()
            .or_else(|| error_obj["innererror"]["code"].as_str())
            .unwrap_or("");
        if error_code == "content_filter"
            || error_code == "content_policy_violation"
            || inner_code == "ResponsibleAIPolicyViolation"
        {
            return ProviderError::ContentFiltered(
                error_obj["message"]
                    .as_str()
                    .unwrap_or("content filtered by Azure policy")
                    .to_owned(),
            );
        }
    }

    let msg = val["error"]["message"]
        .as_str()
        .map(String::from)
        .unwrap_or_else(|| String::from_utf8_lossy(body).into_owned());

    map_status_to_provider_error(status, msg, retry_after)
}

/// Reads the response body (bounded) and delegates to [`classify_azure_error_body`].
async fn map_error_response(status: reqwest::StatusCode, resp: reqwest::Response) -> ProviderError {
    let retry_after = resp
        .headers()
        .get("Retry-After")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    // Bounded read: caps heap allocation so a hostile upstream cannot force large alloc
    // via an oversized error body. Mirrors the pattern in openai/utils.rs.
    let mut body: Vec<u8> = Vec::with_capacity(ERROR_BODY_CAP);
    let mut stream = resp.bytes_stream();
    while let Some(Ok(chunk)) = stream.next().await {
        let remaining = ERROR_BODY_CAP.saturating_sub(body.len());
        if remaining == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }

    classify_azure_error_body(status, &body, retry_after)
}

/// Forces `stream_options.include_usage = true` unconditionally.
///
/// Unlike `inject_stream_options`, which preserves a client-supplied `false`, this
/// always overrides. Azure cost tracking is mandatory and must not be defeatable by
/// client request options.
fn force_include_usage(req: &mut ChatRequest) {
    let mut opts = req
        .extra
        .get("stream_options")
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default();
    opts.insert("include_usage".into(), serde_json::json!(true));
    req.extra.insert("stream_options".into(), opts.into());
}

#[async_trait]
impl ProviderAdapter for AzureAdapter {
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
            return Err(map_error_response(status, resp).await);
        }

        let bytes = resp
            .bytes()
            .await
            .map_err(|e| ProviderError::Unreachable(sanitize_network_error(&e.to_string())))?;

        parse_response(&bytes, &req.model, &self.config.name)
    }

    async fn chat_completion_stream(
        &self,
        req: &ChatRequest,
    ) -> Result<ChatCompletionStream, ProviderError> {
        let mut prepared = req.clone();
        prepared.stream = Some(true);
        force_include_usage(&mut prepared);

        let body = serde_json::to_vec(&prepared)
            .map_err(|e| ProviderError::Serialization(e.to_string()))?;

        let start = std::time::Instant::now();
        let resp = self
            .build_request(body)
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(map_error_response(status, resp).await);
        }

        // Known gap: Azure can send a content-filter block mid-stream as an SSE chunk
        // with `content_filter_results` in the payload (distinct from the HTTP 400 path).
        // make_compat_sse_stream surfaces these as raw StreamChunk::Delta values; the
        // caller sees a non-error chunk with a refusal finish_reason, not ContentFiltered.
        // Full mid-stream content-filter mapping is deferred to.
        Ok(make_compat_sse_stream(resp, self.config.name.clone()))
    }

    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus::Unknown
    }
    // try_forward_raw / try_forward_raw_stream: inherit default None.
    // stream_options injection requires body re-serialization, defeating zero-copy.
}

impl ProviderAdapterExt for AzureAdapter {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SecretString;
    use crate::domain::chat::{Message, MessageContent, Role};

    fn make_config(endpoint: &str, deployment_name: &str, api_version: &str) -> AzureConfig {
        AzureConfig {
            name: "azure-test".to_string(),
            endpoint: endpoint.to_string(),
            deployment_name: deployment_name.to_string(),
            api_version: api_version.to_string(),
            api_key: SecretString::new("sk-test"),
            supported_models: None,
            timeout_secs: Some(5),
        }
    }

    fn make_http() -> Arc<CompatHttpClient> {
        Arc::new(CompatHttpClient::new().expect("test http client"))
    }

    fn minimal_request() -> ChatRequest {
        ChatRequest {
            model: "gpt-4o".into(),
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

    // --- URL construction ---

    #[tokio::test]
    async fn url_construction_deployment_based() {
        let adapter = AzureAdapter::new(
            make_config(
                "https://my-resource.openai.azure.com",
                "gpt-4o",
                "2024-10-21",
            ),
            make_http(),
        )
        .await
        .unwrap();
        assert_eq!(
            adapter.chat_url,
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        );
    }

    #[tokio::test]
    async fn url_construction_trims_trailing_slash() {
        let adapter = AzureAdapter::new(
            make_config(
                "https://my-resource.openai.azure.com/",
                "gpt-4o",
                "2024-10-21",
            ),
            make_http(),
        )
        .await
        .unwrap();
        assert_eq!(
            adapter.chat_url,
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        );
    }

    // --- Metadata ---

    #[tokio::test]
    async fn metadata_fallback_only_when_no_supported_models() {
        let adapter = AzureAdapter::new(
            make_config("https://x.openai.azure.com", "d", "v"),
            make_http(),
        )
        .await
        .unwrap();
        assert_eq!(adapter.metadata().kind, ProviderKind::FallbackOnly);
        assert!(
            adapter
                .metadata()
                .supported_models
                .contains(&"*".to_string())
        );
    }

    #[tokio::test]
    async fn metadata_primary_when_supported_models_set() {
        let mut cfg = make_config("https://x.openai.azure.com", "d", "v");
        cfg.supported_models = Some(vec!["gpt-4o".to_string()]);
        let adapter = AzureAdapter::new(cfg, make_http()).await.unwrap();
        assert_eq!(adapter.metadata().kind, ProviderKind::Primary);
        assert_eq!(adapter.metadata().supported_models, vec!["gpt-4o"]);
    }

    #[tokio::test]
    async fn metadata_name_matches_config() {
        let mut cfg = make_config("https://x.openai.azure.com", "d", "v");
        cfg.name = "azure-prod".to_string();
        let adapter = AzureAdapter::new(cfg, make_http()).await.unwrap();
        assert_eq!(adapter.metadata().name, "azure-prod");
    }

    // --- Error classification (classify_azure_error_body) ---

    fn status(code: u16) -> reqwest::StatusCode {
        reqwest::StatusCode::from_u16(code).unwrap()
    }

    #[test]
    fn error_content_filter_code_returns_content_filtered() {
        let body = serde_json::json!({
            "error": { "code": "content_filter", "message": "blocked by policy" }
        })
        .to_string();
        let err = classify_azure_error_body(status(400), body.as_bytes(), None);
        assert!(
            matches!(err, ProviderError::ContentFiltered(ref m) if m == "blocked by policy"),
            "unexpected: {err:?}"
        );
    }

    #[test]
    fn error_content_policy_violation_returns_content_filtered() {
        let body = serde_json::json!({
            "error": { "code": "content_policy_violation", "message": "policy violation" }
        })
        .to_string();
        let err = classify_azure_error_body(status(400), body.as_bytes(), None);
        assert!(matches!(err, ProviderError::ContentFiltered(_)), "{err:?}");
    }

    #[test]
    fn error_responsible_ai_via_inner_error_returns_content_filtered() {
        let body = serde_json::json!({
            "error": {
                "code": "other",
                "message": "filtered",
                "inner_error": { "code": "ResponsibleAIPolicyViolation" }
            }
        })
        .to_string();
        let err = classify_azure_error_body(status(400), body.as_bytes(), None);
        assert!(matches!(err, ProviderError::ContentFiltered(_)), "{err:?}");
    }

    #[test]
    fn error_responsible_ai_via_innererror_alias_returns_content_filtered() {
        let body = serde_json::json!({
            "error": {
                "code": "other",
                "message": "filtered",
                "innererror": { "code": "ResponsibleAIPolicyViolation" }
            }
        })
        .to_string();
        let err = classify_azure_error_body(status(400), body.as_bytes(), None);
        assert!(matches!(err, ProviderError::ContentFiltered(_)), "{err:?}");
    }

    #[test]
    fn error_generic_400_returns_invalid_request() {
        let body = serde_json::json!({
            "error": { "code": "invalid_request_error", "message": "bad param" }
        })
        .to_string();
        let err = classify_azure_error_body(status(400), body.as_bytes(), None);
        assert!(matches!(err, ProviderError::InvalidRequest(_)), "{err:?}");
    }

    #[test]
    fn error_429_with_retry_after_returns_rate_limited() {
        let body = serde_json::json!({ "error": { "message": "quota exceeded" } }).to_string();
        let err = classify_azure_error_body(status(429), body.as_bytes(), Some(30));
        assert!(
            matches!(
                err,
                ProviderError::RateLimited {
                    retry_after: Some(30)
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn error_503_returns_provider_unavailable() {
        let body = serde_json::json!({ "error": { "message": "service unavailable" } }).to_string();
        let err = classify_azure_error_body(status(503), body.as_bytes(), None);
        assert!(
            matches!(err, ProviderError::ProviderUnavailable(_)),
            "{err:?}"
        );
    }

    // --- Streaming usage injection ---

    #[test]
    fn force_include_usage_sets_true_when_absent() {
        let mut req = minimal_request();
        force_include_usage(&mut req);
        let v = req
            .extra
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(|v| v.as_bool());
        assert_eq!(v, Some(true));
    }

    #[test]
    fn force_include_usage_overrides_client_false() {
        let mut req = minimal_request();
        req.extra.insert(
            "stream_options".into(),
            serde_json::json!({"include_usage": false}),
        );
        force_include_usage(&mut req);
        let v = req
            .extra
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(|v| v.as_bool());
        assert_eq!(
            v,
            Some(true),
            "client false must be overridden for Azure cost tracking"
        );
    }

    #[test]
    fn force_include_usage_preserves_other_stream_options_keys() {
        let mut req = minimal_request();
        req.extra.insert(
            "stream_options".into(),
            serde_json::json!({"include_usage": false, "other_key": "value"}),
        );
        force_include_usage(&mut req);
        let opts = req.extra.get("stream_options").unwrap();
        assert_eq!(
            opts.get("include_usage").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            opts.get("other_key").and_then(|v| v.as_str()),
            Some("value")
        );
    }
}
