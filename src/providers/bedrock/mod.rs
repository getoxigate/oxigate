// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! AWS Bedrock Converse API provider adapter .
//!
//! Phase 1: Claude (anthropic.*) models only via the Converse API.
//! Non-Claude prefixes (meta.*, amazon.*, mistral.*, cohere.*) return UnknownModel immediately.
//!
//! Credentials resolve at startup: config fields → env vars (AWS_ACCESS_KEY_ID /
//! AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN). Startup fails fast on missing credentials.
//!
//! Signing: SigV4 with service name "bedrock" (not "bedrock-runtime") — see signing.rs.
//! Streaming: AWS EventStream binary framing — see eventstream.rs.

pub mod eventstream;
pub mod signing;
pub mod translate;

use std::time::Instant;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use secrecy::{ExposeSecret, SecretString};
use tracing::{info, warn};

use crate::config::BedrockConfig;
use crate::domain::chat::{ChatRequest, ChatResponse, StreamChunk, Usage};
use crate::domain::ports::{
    ChatCompletionStream, HealthStatus, ProviderAdapter, ProviderAdapterExt, ProviderError,
    ProviderKind, ProviderMetadata,
};
use crate::providers::tool_limits::FEATURE_BEDROCK_STREAMING_TOOL_USE;
use crate::utils::provider_error::classify_reqwest_error;
use crate::utils::sse::openai_chat_completion_envelope;

use self::eventstream::{ConverseEvent, EventStreamParser};
use self::signing::BedrockSigner;
use self::translate::{
    ConverseResponse, chat_request_to_converse, converse_response_to_chat, map_stop_reason,
};

const DEFAULT_MODEL: &str = "anthropic.claude-3-5-sonnet-20241022-v2:0";

// Default model list for /v1/models advertisement only.
// Routing uses check_model_prefix (anthropic.* wildcard) — this list does NOT gate requests.
// Operators can override via providers.bedrock.supported_models in config.
// Update when AWS publishes new model IDs; missing entries here do not break routing.
const KNOWN_BEDROCK_MODELS: &[&str] = &[
    "anthropic.claude-opus-4-7-20251001-v1:0",
    "anthropic.claude-sonnet-4-6-20251001-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
];

pub struct BedrockAdapter {
    default_model: Option<String>,
    http: reqwest::Client,
    metadata: ProviderMetadata,
    signer: BedrockSigner,
    base_url: String,
}

impl BedrockAdapter {
    pub async fn new(config: BedrockConfig) -> Result<Self, ProviderError> {
        if config.region.trim().is_empty() {
            return Err(ProviderError::InvalidRequest(
                "providers.bedrock.region is required".into(),
            ));
        }

        let (access_key_id, secret_access_key) = resolve_credentials(&config)?;
        let session_token = resolve_session_token(&config);

        let signer = BedrockSigner::new(
            access_key_id,
            SecretString::new(secret_access_key),
            session_token.map(SecretString::new),
            config.region.clone(),
        );

        let timeout_secs = config.timeout_secs.unwrap_or(120);
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| ProviderError::Unreachable(format!("reqwest client: {e}")))?;

        let supported_models = config.supported_models.clone().unwrap_or_else(|| {
            let mut models: Vec<String> =
                KNOWN_BEDROCK_MODELS.iter().map(|m| m.to_string()).collect();
            // Wildcard for future anthropic.* models operator may add
            models.push("anthropic.*".to_string());
            models
        });

        let metadata = ProviderMetadata {
            name: "bedrock".to_string(),
            supported_models,
            supports_streaming: true,
            supports_tools: true,
            supports_vision: false,     //: vision deferred
            supports_embeddings: false, //: embeddings deferred
            supports_thinking: false,   //: thinking deferred
            kind: ProviderKind::Primary,
            ..Default::default()
        };

        let base_url = config.endpoint_url.clone().unwrap_or_else(|| {
            format!(
                "https://bedrock-runtime.{}.amazonaws.com",
                config.region.trim()
            )
        });
        let base_url = base_url.trim_end_matches('/').to_string();

        let default_model = config.default_model;
        Ok(Self {
            default_model,
            http,
            metadata,
            signer,
            base_url,
        })
    }

    /// Returns the effective model ID (uses default if request is empty).
    fn model(&self, req_model: &str) -> String {
        if req_model.trim().is_empty() {
            self.default_model
                .clone()
                .unwrap_or_else(|| DEFAULT_MODEL.to_string())
        } else {
            req_model.to_string()
        }
    }

    /// Validates that the model prefix is `anthropic.*`.
    ///
    /// Phase 1 supports only Claude models on Bedrock. Other prefixes (meta.*, amazon.*, etc.)
    /// return `UnknownModel` immediately — no request is sent upstream.
    fn check_model_prefix(&self, model: &str) -> Result<(), ProviderError> {
        // Reject URL-special characters before they reach format!() URL construction.
        // Legitimate Bedrock model IDs never contain '/', '?', '#', or '%'.
        if model.contains('/') || model.contains('?') || model.contains('#') || model.contains('%')
        {
            return Err(ProviderError::UnknownModel(format!(
                "invalid model id '{model}': must not contain URL-special characters (/, ?, #, %)"
            )));
        }
        if model.starts_with("anthropic.") {
            Ok(())
        } else {
            Err(ProviderError::UnknownModel(format!(
                "bedrock phase-1 supports only anthropic.* models; got '{model}'. \
                 Other prefixes (meta.*, amazon.*, etc.) are not yet supported."
            )))
        }
    }

    /// Builds the Converse non-streaming URL.
    fn converse_url(&self, model: &str) -> String {
        format!("{}/model/{}/converse", self.base_url, model)
    }

    /// Builds the Converse streaming URL.
    fn converse_stream_url(&self, model: &str) -> String {
        format!("{}/model/{}/converse-stream", self.base_url, model)
    }

    /// Serializes the Converse request body.
    fn build_body(&self, req: &ChatRequest) -> Result<Vec<u8>, ProviderError> {
        let converse_req = chat_request_to_converse(req)?;
        serde_json::to_vec(&converse_req).map_err(|e| ProviderError::Serialization(e.to_string()))
    }

    /// Maps a Bedrock error response to a `ProviderError`.
    ///
    /// Parses `{"__type": "...", "message": "..."}` from the body first;
    /// falls back to HTTP status code classification.
    async fn map_bedrock_error(
        &self,
        status: reqwest::StatusCode,
        resp: reqwest::Response,
    ) -> ProviderError {
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        let body = resp.text().await.unwrap_or_default();
        let (error_type, message) = parse_bedrock_error_body(&body);

        match error_type.as_deref() {
            Some("AccessDeniedException") => ProviderError::Auth(
                message.unwrap_or_else(|| "access denied — check Bedrock IAM permissions".into()),
            ),
            Some("ValidationException") => {
                ProviderError::InvalidRequest(message.unwrap_or_else(|| body.clone()))
            }
            Some("ModelNotFoundException") => {
                ProviderError::UnknownModel(message.unwrap_or_else(|| body.clone()))
            }
            Some("ThrottlingException") => ProviderError::RateLimited { retry_after },
            Some("ModelStreamErrorException") | Some("ServiceUnavailableException") => {
                ProviderError::ProviderUnavailable(message.unwrap_or_else(|| body.clone()))
            }
            _ => match status.as_u16() {
                400 => ProviderError::InvalidRequest(body),
                401 | 403 => ProviderError::Auth(body),
                404 => ProviderError::UnknownModel(body),
                429 => ProviderError::RateLimited { retry_after },
                500 | 502 | 503 | 504 => ProviderError::ProviderUnavailable(body.clone()),
                _ => {
                    warn!(
                        status = status.as_u16(),
                        body = %body,
                        "bedrock: unmapped error status"
                    );
                    ProviderError::ProviderUnavailable(format!(
                        "bedrock HTTP {}: {}",
                        status.as_u16(),
                        body
                    ))
                }
            },
        }
    }
}

/// Parses `{"__type": "SomeName", "message": "..."}` from a Bedrock error body.
fn parse_bedrock_error_body(body: &str) -> (Option<String>, Option<String>) {
    let Ok(json) = serde_json::from_str::<serde_json::Value>(body) else {
        return (None, None);
    };
    let error_type = json
        .get("__type")
        .and_then(|v| v.as_str())
        .map(String::from);
    let message = json
        .get("message")
        .and_then(|v| v.as_str())
        .map(String::from);
    (error_type, message)
}

/// Returns from `secret` if non-empty, else calls `env_fn`.
///
/// Separated from the real `std::env::var` call so tests can inject a closure instead of
/// mutating env vars — `set_var` is UB under parallel test threads (Rust 1.80+).
fn resolve_from<S, F>(secret: Option<&S>, env_fn: F) -> Option<String>
where
    S: ExposeSecret<String>,
    F: FnOnce() -> Option<String>,
{
    secret
        .and_then(|s| {
            let v = s.expose_secret();
            if v.is_empty() { None } else { Some(v.clone()) }
        })
        .or_else(env_fn)
}

/// Returns the non-empty string from a config secret field, or falls back to an env var.
fn resolve_secret_or_env<S>(secret: Option<&S>, env_var: &str) -> Option<String>
where
    S: ExposeSecret<String>,
{
    resolve_from(secret, || {
        std::env::var(env_var).ok().filter(|s| !s.is_empty())
    })
}

/// Resolves AWS access key ID and secret key from config or environment.
fn resolve_credentials(config: &BedrockConfig) -> Result<(String, String), ProviderError> {
    let key_id = resolve_secret_or_env(config.access_key_id.as_ref(), "AWS_ACCESS_KEY_ID");
    let secret = resolve_secret_or_env(config.secret_access_key.as_ref(), "AWS_SECRET_ACCESS_KEY");
    match (key_id, secret) {
        (Some(k), Some(s)) => Ok((k, s)),
        _ => Err(ProviderError::Auth(
            "bedrock: no AWS credentials found. \
             Set providers.bedrock.access_key_id/secret_access_key or \
             AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env vars."
                .into(),
        )),
    }
}

/// Resolves optional STS session token from config or environment.
fn resolve_session_token(config: &BedrockConfig) -> Option<String> {
    resolve_secret_or_env(config.session_token.as_ref(), "AWS_SESSION_TOKEN")
}

/// Serializes an SSE envelope to `data: {...}\n\n` (plus `data: [DONE]\n\n` when `done` is true).
fn sse_frame(
    envelope: &serde_json::Map<String, serde_json::Value>,
    done: bool,
) -> Result<Bytes, ProviderError> {
    let json =
        serde_json::to_string(envelope).map_err(|e| ProviderError::Serialization(e.to_string()))?;
    if done {
        Ok(Bytes::from(format!("data: {json}\n\ndata: [DONE]\n\n")))
    } else {
        Ok(Bytes::from(format!("data: {json}\n\n")))
    }
}

#[async_trait]
impl ProviderAdapter for BedrockAdapter {
    async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        let model = self.model(&req.model);
        self.check_model_prefix(&model)?;

        let request_id = req.request_id.as_deref().unwrap_or("unknown");
        info!(model = %model, request_id = %request_id, "bedrock: chat_completion");
        let body = self.build_body(req)?;
        let url = self.converse_url(&model);

        let signed_headers = self.signer.sign_request("POST", &url, &body)?;

        let start = Instant::now();
        let resp = self
            .http
            .post(&url)
            .headers(signed_headers)
            .body(body)
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(self.map_bedrock_error(status, resp).await);
        }

        let converse_resp: ConverseResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::Serialization(e.to_string()))?;

        Ok(converse_response_to_chat(
            &converse_resp,
            &model,
            request_id,
        ))
    }

    async fn chat_completion_stream(
        &self,
        req: &ChatRequest,
    ) -> Result<ChatCompletionStream, ProviderError> {
        let tool_choice_val = req.extra.get("tool_choice");
        if !crate::domain::tool_schema::is_tool_choice_none(tool_choice_val)
            && req.tools.as_ref().is_some_and(|t| !t.is_empty())
        {
            return Err(ProviderError::NotYetSupported {
                feature: FEATURE_BEDROCK_STREAMING_TOOL_USE,
            });
        }
        let model = self.model(&req.model);
        self.check_model_prefix(&model)?;

        let request_id = req.request_id.as_deref().unwrap_or("unknown").to_string();
        info!(model = %model, request_id = %request_id, "bedrock: chat_completion_stream");
        let body = self.build_body(req)?;
        let url = self.converse_stream_url(&model);

        let signed_headers = self.signer.sign_request("POST", &url, &body)?;

        let start = Instant::now();
        let resp = self
            .http
            .post(&url)
            .headers(signed_headers)
            .body(body)
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, start.elapsed().as_millis() as u64))?;

        if !resp.status().is_success() {
            return Err(self.map_bedrock_error(resp.status(), resp).await);
        }

        let model_clone = model.clone();
        let mut bytes_stream = resp
            .bytes_stream()
            .map(|r| r.map_err(|e: reqwest::Error| std::io::Error::other(e.to_string())));

        let created = chrono::Utc::now().timestamp() as u64;

        let output = async_stream::stream! {
            let mut parser = EventStreamParser::new();
            // AWS event order: contentBlockDelta(s) → messageStop → metadata.
            // Buffer stop_reason at messageStop; emit [DONE] only after metadata arrives
            // so the cost middleware always receives token counts for billing.
            let mut pending_stop_reason: Option<String> = None;
            // OpenAI SSE spec: first chunk carries {"role":"assistant","content":""};
            // subsequent deltas carry {"content":...} only. Strict clients (openai-python,
            // LangChain) rely on this boundary to detect message start.
            let mut first_delta = true;

            'outer: while let Some(chunk_result) = bytes_stream.next().await {
                let chunk = match chunk_result {
                    Ok(b) => b,
                    Err(e) => {
                        yield Err(ProviderError::ProviderUnavailable(e.to_string()));
                        return;
                    }
                };

                let events = match parser.feed(&chunk) {
                    Ok(evts) => evts,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };

                for event in events {
                    match event {
                        ConverseEvent::ContentBlockDelta { text } => {
                            if first_delta {
                                first_delta = false;
                                let preamble = serde_json::json!({
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": null
                                });
                                let env = openai_chat_completion_envelope(
                                    created,
                                    &model_clone,
                                    &request_id,
                                    preamble,
                                );
                                let data = match sse_frame(&env, false) {
                                    Ok(b) => b,
                                    Err(e) => { yield Err(e); return; }
                                };
                                yield Ok(StreamChunk::new(data, None, Some(model_clone.clone())));
                            }
                            let choice = serde_json::json!({
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": null
                            });
                            let envelope = openai_chat_completion_envelope(
                                created,
                                &model_clone,
                                &request_id,
                                choice,
                            );
                            let data = match sse_frame(&envelope, false) {
                                Ok(b) => b,
                                Err(e) => { yield Err(e); return; }
                            };
                            yield Ok(StreamChunk::new(data, None, Some(model_clone.clone())));
                        }
                        ConverseEvent::MessageStop { stop_reason } => {
                            pending_stop_reason = Some(stop_reason);
                        }
                        ConverseEvent::Metadata { input_tokens, output_tokens } => {
                            let usage = Usage {
                                prompt_tokens: input_tokens,
                                completion_tokens: output_tokens,
                                total_tokens: input_tokens + output_tokens,
                                ..Default::default()
                            };
                            // take() clears pending_stop_reason so the post-loop fallback
                            // doesn't double-emit [DONE].
                            let taken = pending_stop_reason.take();
                            let finish =
                                taken.as_deref().map(map_stop_reason).unwrap_or("stop");
                            let choice = serde_json::json!({
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish
                            });
                            let mut envelope = openai_chat_completion_envelope(
                                created,
                                &model_clone,
                                &request_id,
                                choice,
                            );
                            let usage_val = match serde_json::to_value(&usage) {
                                Ok(v) => v,
                                Err(e) => {
                                    yield Err(ProviderError::Serialization(e.to_string()));
                                    return;
                                }
                            };
                            envelope.insert("usage".to_string(), usage_val);
                            let data = match sse_frame(&envelope, true) {
                                Ok(b) => b,
                                Err(e) => { yield Err(e); return; }
                            };
                            yield Ok(StreamChunk::new(data, Some(usage), Some(model_clone.clone())));
                            break 'outer;
                        }
                        ConverseEvent::StreamError(e) => {
                            yield Err(e);
                            return;
                        }
                        ConverseEvent::Ignored => {}
                    }
                }
            }

            // Fallback: stream ended after messageStop but without a metadata frame.
            // Rare (error paths), but we must still close the stream.
            if let Some(stop_reason) = pending_stop_reason {
                let finish = map_stop_reason(&stop_reason);
                let choice = serde_json::json!({
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish
                });
                let envelope = openai_chat_completion_envelope(
                    created,
                    &model_clone,
                    &request_id,
                    choice,
                );
                match sse_frame(&envelope, true) {
                    Ok(data) => {
                        yield Ok(StreamChunk::new(data, None, Some(model_clone.clone())));
                    }
                    Err(e) => {
                        yield Err(e);
                    }
                }
            }
        };

        Ok(Box::pin(output))
    }

    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }

    /// A 5xx response indicates service failure; 4xx (e.g. 403) means the endpoint is
    /// reachable but unauthorized — expected for an unauthenticated HEAD probe.
    async fn health_check(&self) -> HealthStatus {
        match self.http.head(&self.base_url).send().await {
            Ok(resp) if resp.status().is_server_error() => HealthStatus::Unhealthy,
            Ok(_) => HealthStatus::Healthy,
            Err(_) => HealthStatus::Unhealthy,
        }
    }
}

impl ProviderAdapterExt for BedrockAdapter {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(region: &str) -> BedrockConfig {
        BedrockConfig {
            region: region.to_string(),
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            endpoint_url: None,
            default_model: None,
            timeout_secs: None,
            supported_models: None,
        }
    }

    #[test]
    fn test_error_access_denied() {
        let body = r#"{"__type":"AccessDeniedException","message":"not allowed"}"#;
        let (t, m) = parse_bedrock_error_body(body);
        assert_eq!(t.as_deref(), Some("AccessDeniedException"));
        assert_eq!(m.as_deref(), Some("not allowed"));
    }

    #[test]
    fn test_error_throttling() {
        let body = r#"{"__type":"ThrottlingException","message":"too many"}"#;
        let (t, _m) = parse_bedrock_error_body(body);
        assert_eq!(t.as_deref(), Some("ThrottlingException"));
    }

    #[test]
    fn test_error_model_not_found() {
        let body = r#"{"__type":"ModelNotFoundException","message":"no such model"}"#;
        let (t, _m) = parse_bedrock_error_body(body);
        assert_eq!(t.as_deref(), Some("ModelNotFoundException"));
    }

    #[test]
    fn test_error_unknown_prefix() {
        // A BedrockAdapter with valid env creds would still return UnknownModel for meta.*
        // Test the prefix check directly.
        let adapter = BedrockAdapter {
            default_model: None,
            http: reqwest::Client::new(),
            metadata: ProviderMetadata {
                name: "bedrock".to_string(),
                supported_models: vec![],
                supports_streaming: true,
                supports_tools: false,
                supports_vision: false,
                supports_embeddings: false,
                supports_thinking: false,
                kind: ProviderKind::Primary,
                ..Default::default()
            },
            signer: BedrockSigner::new(
                "key".into(),
                SecretString::new("secret".into()),
                None,
                "us-east-1".into(),
            ),
            base_url: "https://bedrock-runtime.us-east-1.amazonaws.com".to_string(),
        };
        let err = adapter.check_model_prefix("meta.llama3-70b-instruct-v1:0");
        assert!(matches!(err, Err(ProviderError::UnknownModel(_))));
    }

    #[test]
    fn test_error_url_injection_chars_rejected() {
        let adapter = BedrockAdapter {
            default_model: None,
            http: reqwest::Client::new(),
            metadata: ProviderMetadata {
                name: "bedrock".to_string(),
                supported_models: vec![],
                supports_streaming: true,
                supports_tools: false,
                supports_vision: false,
                supports_embeddings: false,
                supports_thinking: false,
                kind: ProviderKind::Primary,
                ..Default::default()
            },
            signer: BedrockSigner::new(
                "key".into(),
                SecretString::new("secret".into()),
                None,
                "us-east-1".into(),
            ),
            base_url: "https://bedrock-runtime.us-east-1.amazonaws.com".to_string(),
        };
        // '/' path traversal
        assert!(matches!(
            adapter.check_model_prefix("anthropic.claude/../../admin"),
            Err(ProviderError::UnknownModel(_))
        ));
        // '?' query injection
        assert!(matches!(
            adapter.check_model_prefix("anthropic.claude?x=y"),
            Err(ProviderError::UnknownModel(_))
        ));
        // '#' fragment injection
        assert!(matches!(
            adapter.check_model_prefix("anthropic.claude#frag"),
            Err(ProviderError::UnknownModel(_))
        ));
        // '%' percent-encoding injection
        assert!(matches!(
            adapter.check_model_prefix("anthropic.claude%2F..%2Fadmin"),
            Err(ProviderError::UnknownModel(_))
        ));
    }

    #[test]
    fn test_credential_resolution_explicit_config() {
        let mut config = make_config("us-east-1");
        config.access_key_id = Some(crate::config::SecretString::new("explicit-key"));
        config.secret_access_key = Some(crate::config::SecretString::new("explicit-secret"));
        // Must succeed regardless of env vars
        let result = resolve_credentials(&config);
        assert!(result.is_ok());
        let (k, s) = result.unwrap();
        assert_eq!(k, "explicit-key");
        assert_eq!(s, "explicit-secret");
    }

    #[test]
    fn test_credential_resolution_env_fallback() {
        // Use resolve_from with a controlled closure — no env mutation, no data race.
        // (set_var is UB under parallel test threads since Rust 1.80.)

        // None secret → env closure used.
        let result = resolve_from(None::<&crate::config::SecretString>, || {
            Some("env-key".to_string())
        });
        assert_eq!(result.as_deref(), Some("env-key"));

        // Empty config field (treated as absent) → env closure used.
        let empty = crate::config::SecretString::new("");
        let result = resolve_from(Some(&empty), || Some("env-fallback".to_string()));
        assert_eq!(result.as_deref(), Some("env-fallback"));

        // Non-empty config field → env closure never called.
        let present = crate::config::SecretString::new("config-val");
        let result = resolve_from(Some(&present), || Some("env-ignored".to_string()));
        assert_eq!(result.as_deref(), Some("config-val"));
    }

    #[test]
    fn test_credential_resolution_missing_fails() {
        // Clear env vars for this check — use a known-bad config.
        let config = make_config("us-east-1");
        // If env vars are not set, this must fail with Auth error.
        // We simulate by calling with empty config and no env (best-effort).
        // If AWS_ACCESS_KEY_ID is in the environment this test becomes a no-op.
        if std::env::var("AWS_ACCESS_KEY_ID").is_ok() {
            return; // Skip in environments with real AWS creds
        }
        let result = resolve_credentials(&config);
        assert!(matches!(result, Err(ProviderError::Auth(_))));
    }
}
