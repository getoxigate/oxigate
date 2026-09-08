// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Chat completions handler — POST /v1/chat/completions.
//!
//! OpenAI-compatible request/response with auth stub and cost header injection.
//! SSE streaming supported when provider implements chat_completion_stream.
//! Per-identity spend tracking — Redis INCRBY + Postgres audit row on every request.
//! Structured cost log line (chat_completion_cost) emitted after every completed request.
//!
//! TODO(bench): add criterion benchmark for chat_completions handler dispatch;
//! target < 50µs P99.

use std::sync::Arc;

use async_stream::stream;
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderValue;
use axum::http::StatusCode;
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE, RETRY_AFTER};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures::stream::StreamExt;
use serde_json::json;
use thiserror::Error;
use tracing::warn;

use crate::api::AppState;
use crate::domain::auth::RequestIdentity;
use crate::domain::chat::ChatRequest;
use crate::domain::ports::{AttemptedMeta, ProviderError};
use crate::domain::spend::SpendRecord;
use crate::middleware::request_metrics::ProviderLabel;
use crate::observability::metrics::{COST_USD_TOTAL, ENDPOINT_CHAT};
use crate::utils::cost_headers;

/// Yield type for chat streaming: Bytes on success, Infallible (never Err).
type ChatStreamItem = Result<Bytes, std::convert::Infallible>;

/// The terminal `oxigate.usage` SSE payload.
///
/// Serialized rather than assembled with `format!`, so a member that needs escaping cannot
/// produce a malformed event. The four cost and token members keep the header names this event
/// has always used; `cost_status` is the request-wide status, which cannot travel as an HTTP
/// header on a streamed response because it is not known when the headers are sent.
#[derive(serde::Serialize)]
struct UsageEvent<'a> {
    #[serde(rename = "X-Oxigate-Request-Cost")]
    request_cost: &'a str,
    #[serde(rename = "X-Oxigate-Input-Tokens")]
    input_tokens: String,
    #[serde(rename = "X-Oxigate-Output-Tokens")]
    output_tokens: String,
    #[serde(rename = "X-Oxigate-Model-Used")]
    model_used: &'a str,
    cost_status: &'a str,
}

/// Everything one streamed request's finalization needs, captured before the response body is
/// created.
///
/// Owned rather than borrowed on purpose, and deliberately without a lifetime parameter: the
/// generator becomes the response body and must be `'static`, so it cannot hold a reference to
/// `AppState`. Assembling the values once and moving the whole thing across that boundary is what
/// keeps them together — carrying them as separate locals only defers reassembling them, one
/// argument at a time, at the point of use.
struct RequestAccounting {
    pricing_db: Arc<std::sync::RwLock<crate::domain::pricing::PricingDb>>,
    pool: Arc<tokio::sync::RwLock<crate::db::DbPool>>,
    redis: Arc<tokio::sync::RwLock<crate::redis_pool::RedisPool>>,
    /// Snapshot taken when the request arrived, so a request is priced against the config as it
    /// was at that moment even if a reload swaps it mid-stream.
    budget: crate::config::BudgetConfig,
    identity: RequestIdentity,
    request_id: String,
    provider_name: String,
    batch: bool,
    content_length: Option<u64>,
    request_start: std::time::Instant,
}

/// Finalizes a streamed request's accounting and returns its terminal `oxigate.usage` event.
///
/// Prices whatever usage the stream reported — or finalizes the request as cost-unavailable when
/// it reported none — emits the structured cost log line, increments the cost counter and
/// schedules the spend write. Row durability stays asynchronous: the database write is a spawned
/// task here as it is everywhere else. The event bytes are returned rather than sent, because
/// only the generator that owns the response body can forward them.
///
/// **This is a plain `fn`, not an `async fn`, and must stay one.** The caller runs it between
/// resolving the provider's terminal chunk and forwarding that chunk, and the ordering only buys
/// anything because nothing can interleave in between: a generator future is cancelled at a
/// suspension point, and a non-`async fn` cannot contain one. An `.await` added in here is a
/// compile error, which is the intent — relaxing the signature to allow one would silently
/// restore the behaviour where a client that stops reading at the last provider chunk is never
/// accounted at all.
///
/// The values it works from are passed rather than captured; that is what lets it be a `fn`
/// instead of a closure. Everything fixed for the request travels in one [`RequestAccounting`];
/// the two remaining parameters are exactly the two that differ between the call sites.
fn finalize_stream_accounting(
    acct: &RequestAccounting,
    last_seen_usage: Option<&crate::domain::chat::Usage>,
    model_used: &str,
) -> Bytes {
    // A stream that reported no usage is finalized as cost-unavailable rather than left
    // traceless: the request happened, and its identity, model, provider and latency are worth
    // recording even when its cost is not establishable. Both branches produce the same kind of
    // value, so the tail below runs once over one accounting result.
    //
    // On the reported-usage branch the header map is discarded — a streamed response's headers
    // were sent before the first chunk. Every value the terminal event reports comes from the
    // finalization result and the usage it was computed from, which is what the headers are built
    // from on the buffered path, so the two paths cannot drift.
    //
    // The two token members are the provider's reported totals, as the buffered path's
    // INPUT_TOKENS / OUTPUT_TOKENS headers are — not the billing buckets on the accounting, which
    // split cached tokens out on a cache-inclusive contract. They travel beside the accounting
    // rather than being read back off it for that reason.
    let (accounting, prompt_tokens_display, completion_tokens_display) = match last_seen_usage {
        Some(usage) => (
            cost_headers::build_cost_headers(
                model_used,
                usage,
                Arc::clone(&acct.pricing_db),
                acct.batch,
            )
            .1,
            usage.prompt_tokens,
            usage.completion_tokens,
        ),
        None => (
            cost_headers::finalize_missing_usage(
                model_used,
                Arc::clone(&acct.pricing_db),
                acct.batch,
            ),
            0,
            0,
        ),
    };
    cost_headers::report_finalized_warning(&acct.request_id, model_used, &accounting);
    let cost = accounting.cost.total_cost.to_display_string();
    let payload = UsageEvent {
        request_cost: &cost,
        input_tokens: prompt_tokens_display.to_string(),
        output_tokens: completion_tokens_display.to_string(),
        model_used,
        cost_status: accounting.cost.status.as_str(),
    };
    // Integers and short ASCII strings; serialization has no failing case. The fallback keeps the
    // event well-formed rather than truncating the stream.
    let data = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());

    let latency_ms = i32::try_from(acct.request_start.elapsed().as_millis()).unwrap_or_else(|_| {
        tracing::warn!("streaming request latency overflows i32; recording -1");
        -1
    });
    let record = SpendRecord::build(
        &acct.identity,
        model_used,
        &acct.provider_name,
        &accounting,
        latency_ms,
    );
    //: emit per-request cost counter (nano-USD).
    metrics::counter!(
        COST_USD_TOTAL,
        "provider" => acct.provider_name.clone(),
        "endpoint" => ENDPOINT_CHAT
    )
    .increment(accounting.cost.total_cost.as_u64());

    //: request size observability at DEBUG (stays local to chat path).
    tracing::debug!(
        request_id = %acct.request_id,
        request_body_bytes = ?acct.content_length,
        "chat_request_size"
    );
    crate::api::spawn_cost_log_and_spend(
        "chat_completion_cost",
        record,
        &acct.request_id,
        &cost,
        Arc::clone(&acct.pool),
        Arc::clone(&acct.redis),
        acct.budget.clone(),
    );

    Bytes::from(format!("event: oxigate.usage\ndata: {data}\n\n"))
}

/// Chat endpoint error with OpenAI-compatible JSON envelope.
#[derive(Debug, Error)]
pub enum ChatError {
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    #[error("provider unreachable: {0}")]
    ProviderUnreachable(String),
    #[error("upstream provider error: {status} - {body}")]
    ProviderError { status: u16, body: String },
    #[error("internal serialization error: {0}")]
    Serialization(String),
    #[error("feature not implemented")]
    NotImplemented,
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("auth error: {0}")]
    Auth(String),
    #[error("model not found: {0}")]
    UnknownModel(String),
    #[error("rate limited")]
    RateLimited {
        /// Seconds to wait before retry, if provided by provider.
        retry_after: Option<u64>,
    },
    #[error("provider unavailable: {0}")]
    ProviderUnavailable(String),
    #[error("content filtered: {0}")]
    ContentFiltered(String),
    #[error("not supported: {0}")]
    NotSupported(String),
    #[error("translation error: {0}")]
    TranslationError(String),
    /// All providers are in 429 cooldown . → HTTP 503 + Retry-After.
    #[error("all providers rate limited; retry after {retry_after}s")]
    AllProvidersRateLimited { retry_after: u64 },
    /// Internal routing misconfiguration (e.g. all weights zero). → HTTP 500.
    #[error("internal error: {0}")]
    Internal(String),
    /// Provider request or inter-chunk streaming timeout.
    #[error("provider timeout after {elapsed_ms}ms")]
    Timeout { elapsed_ms: u64 },
    /// Tool-use errors (choice unsupported, count exceeded, schema validation,
    /// buffer overflow, not yet supported). Wraps the ProviderError directly to
    /// avoid 1:1 boilerplate; mapped to HTTP responses in IntoResponse.
    #[error("{0}")]
    ToolError(crate::domain::ports::ProviderError),
}

impl From<ProviderError> for ChatError {
    fn from(e: ProviderError) -> Self {
        match e {
            ProviderError::Unreachable(msg) => ChatError::ProviderUnreachable(msg),
            ProviderError::ProviderHttpError { status, body } => {
                ChatError::ProviderError { status, body }
            }
            ProviderError::Serialization(s) => ChatError::Serialization(s),
            ProviderError::NotImplemented => ChatError::NotImplemented,
            ProviderError::InvalidRequest(s) => ChatError::InvalidRequest(s),
            ProviderError::Auth(s) => ChatError::Auth(s),
            ProviderError::UnknownModel(s) => ChatError::UnknownModel(s),
            ProviderError::RateLimited { retry_after } => ChatError::RateLimited { retry_after },
            ProviderError::ProviderUnavailable(s) => ChatError::ProviderUnavailable(s),
            ProviderError::ContentFiltered(s) => ChatError::ContentFiltered(s),
            ProviderError::NotSupported(s) => ChatError::NotSupported(s),
            ProviderError::Translate(s) => ChatError::TranslationError(s),
            ProviderError::AllProvidersRateLimited { retry_after } => {
                ChatError::AllProvidersRateLimited { retry_after }
            }
            ProviderError::Internal(s) => ChatError::Internal(s),
            ProviderError::Timeout { elapsed_ms } => ChatError::Timeout { elapsed_ms },
            ProviderError::ToolChoiceUnsupported { .. }
            | ProviderError::ToolCountExceeded { .. }
            | ProviderError::MalformedToolSchema { .. }
            | ProviderError::ToolCallBufferOverflow { .. }
            | ProviderError::NotYetSupported { .. } => ChatError::ToolError(e),
        }
    }
}

impl IntoResponse for ChatError {
    fn into_response(self) -> axum::response::Response {
        use crate::domain::ports::ProviderError;
        use crate::domain::tool_schema::{
            ERR_MALFORMED_TOOL_SCHEMA, ERR_NOT_YET_SUPPORTED, ERR_TOOL_CALL_BUFFER_OVERFLOW,
            ERR_TOOL_CHOICE_UNSUPPORTED, ERR_TOOL_COUNT_EXCEEDED, ERR_TYPE_GATEWAY_ERROR,
            SUPPORTED_TOOL_CHOICE_VALUES,
        };

        match self {
            // ── Tool-use errors (HTTP 400 / 502) ─────────────────────────────────────────────
            Self::ToolError(ProviderError::ToolCallBufferOverflow {
                provider,
                tool_call_id,
                cap_bytes,
            }) => (
                StatusCode::BAD_GATEWAY,
                Json(json!({
                    "error": {
                        "message": "tool call JSON exceeded the per-call buffer cap",
                        "type": ERR_TYPE_GATEWAY_ERROR,
                        "code": ERR_TOOL_CALL_BUFFER_OVERFLOW,
                        "provider": provider,
                        "tool_call_id": tool_call_id,
                        "cap_bytes": cap_bytes,
                    }
                })),
            )
                .into_response(),

            Self::ToolError(ProviderError::NotYetSupported { feature }) => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": format!("not yet supported: {feature}"),
                        "type": ERR_TYPE_GATEWAY_ERROR,
                        "code": ERR_NOT_YET_SUPPORTED,
                        "feature": feature,
                    }
                })),
            )
                .into_response(),

            Self::ToolError(ProviderError::ToolChoiceUnsupported {
                provider, requested, ..
            }) => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": format!("tool_choice not supported by {provider}: {requested}"),
                        "type": ERR_TOOL_CHOICE_UNSUPPORTED,
                        "code": ERR_TOOL_CHOICE_UNSUPPORTED,
                        "param": null,
                        "provider": provider,
                        "requested": requested,
                        "supported_values": SUPPORTED_TOOL_CHOICE_VALUES,
                    }
                })),
            )
                .into_response(),

            Self::ToolError(ProviderError::ToolCountExceeded {
                provider,
                requested,
                limit,
            }) => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": format!("tool count exceeded for {provider}: requested {requested}, limit {limit}"),
                        "type": ERR_TOOL_COUNT_EXCEEDED,
                        "code": ERR_TOOL_COUNT_EXCEEDED,
                        "param": null,
                        "provider": provider,
                        "requested": requested,
                        "limit": limit,
                    }
                })),
            )
                .into_response(),

            Self::ToolError(ProviderError::MalformedToolSchema { provider, reason }) => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": format!("malformed tool schema for {provider}: {reason}"),
                        "type": ERR_MALFORMED_TOOL_SCHEMA,
                        "code": ERR_MALFORMED_TOOL_SCHEMA,
                        "param": null,
                        "provider": provider,
                        "reason": reason,
                    }
                })),
            )
                .into_response(),

            Self::ToolError(other) => {
                // Non-tool ProviderError routed here — indicates a bug in From<ProviderError>.
                warn!(error = %other, "non-tool ProviderError in ChatError::ToolError");
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            }

            // ── Standard OpenAI error envelope ───────────────────────────────────────────────
            ref e @ (Self::Unauthorized(_)
            | Self::ProviderUnreachable(_)
            | Self::ProviderError { .. }
            | Self::Serialization(_)
            | Self::NotImplemented
            | Self::InvalidRequest(_)
            | Self::TranslationError(_)
            | Self::Auth(_)
            | Self::UnknownModel(_)
            | Self::RateLimited { .. }
            | Self::ProviderUnavailable(_)
            | Self::ContentFiltered(_)
            | Self::NotSupported(_)
            | Self::AllProvidersRateLimited { .. }
            | Self::Internal(_)
            | Self::Timeout { .. }) => {
                let (status, code) = match e {
                    Self::Unauthorized(_) => (StatusCode::UNAUTHORIZED, "unauthorized"),
                    Self::ProviderUnreachable(_) => {
                        (StatusCode::SERVICE_UNAVAILABLE, "provider_unreachable")
                    }
                    Self::ProviderError { .. } => (StatusCode::BAD_GATEWAY, "provider_error"),
                    Self::Serialization(_) => (StatusCode::BAD_GATEWAY, "internal_error"),
                    Self::NotImplemented => (StatusCode::NOT_IMPLEMENTED, "not_implemented"),
                    Self::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
                    Self::TranslationError(_) => (StatusCode::BAD_REQUEST, "translation_error"),
                    Self::Auth(_) => (StatusCode::UNAUTHORIZED, "authentication_error"),
                    Self::UnknownModel(_) => (StatusCode::NOT_FOUND, "invalid_request_error"),
                    Self::RateLimited { .. } => {
                        (StatusCode::TOO_MANY_REQUESTS, "rate_limit_exceeded")
                    }
                    Self::ProviderUnavailable(_) => {
                        (StatusCode::SERVICE_UNAVAILABLE, "provider_unavailable")
                    }
                    Self::ContentFiltered(_) => (StatusCode::BAD_REQUEST, "content_filtered"),
                    Self::NotSupported(_) => (StatusCode::BAD_REQUEST, "not_supported"),
                    Self::AllProvidersRateLimited { .. } => {
                        (StatusCode::SERVICE_UNAVAILABLE, "rate_limit_exceeded")
                    }
                    Self::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
                    Self::Timeout { .. } => (StatusCode::GATEWAY_TIMEOUT, "provider_timeout"),
                    // ToolError variants are exhaustively handled by earlier match arms above;
                    // this arm is never reached but required for compiler exhaustiveness.
                    Self::ToolError(_) => unreachable!("ToolError matched by earlier arms"),
                };
                let body = Json(json!({
                    "error": {
                        "message": e.to_string(),
                        "type": code,
                        "param": null,
                        "code": code
                    }
                }));
                let mut response = (status, body).into_response();
                if matches!(e, Self::Unauthorized(_) | Self::Auth(_)) {
                    response
                        .headers_mut()
                        .insert("WWW-Authenticate", HeaderValue::from_static("Bearer"));
                }
                if let Self::RateLimited {
                    retry_after: Some(secs),
                } = e
                {
                    response.headers_mut().insert(
                        RETRY_AFTER,
                        HeaderValue::from_str(&secs.to_string())
                            .expect("u64 decimal is always a valid HeaderValue"),
                    );
                }
                if let Self::AllProvidersRateLimited { retry_after } = e {
                    response.headers_mut().insert(
                        RETRY_AFTER,
                        HeaderValue::from_str(&retry_after.to_string())
                            .expect("u64 decimal is always a valid HeaderValue"),
                    );
                }
                response
            }
        }
    }
}

// ── M5: tool-error HTTP shape tests ──────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ports::ProviderError;

    /// The terminal usage event is serialized, not hand-built, and its four legacy members keep
    /// the exact header names clients already parse. `cost_status` joins them because a streamed
    /// response cannot carry the status as an HTTP header.
    #[test]
    fn test_usage_event_serializes_the_header_names_and_the_status() {
        use crate::domain::usage_accounting::CostStatus;
        use crate::utils::CostHeader;

        let payload = UsageEvent {
            request_cost: "0.001234",
            input_tokens: "5".to_string(),
            output_tokens: "2".to_string(),
            model_used: "gpt-4-0613",
            cost_status: CostStatus::RateFallback.as_str(),
        };
        let data = serde_json::to_string(&payload).expect("payload serializes");
        let parsed: serde_json::Value =
            serde_json::from_str(&data).expect("terminal event data must be valid JSON");

        assert_eq!(
            parsed
                .get(CostHeader::REQUEST_COST)
                .and_then(|v| v.as_str()),
            Some("0.001234")
        );
        assert_eq!(
            parsed
                .get(CostHeader::INPUT_TOKENS)
                .and_then(|v| v.as_str()),
            Some("5")
        );
        assert_eq!(
            parsed
                .get(CostHeader::OUTPUT_TOKENS)
                .and_then(|v| v.as_str()),
            Some("2")
        );
        assert_eq!(
            parsed.get(CostHeader::MODEL_USED).and_then(|v| v.as_str()),
            Some("gpt-4-0613")
        );
        assert_eq!(
            parsed.get("cost_status").and_then(|v| v.as_str()),
            Some(CostStatus::RateFallback.as_str())
        );
    }

    /// A model name that would need escaping produces valid JSON rather than a malformed event —
    /// the reason the payload is serialized instead of assembled with `format!`.
    #[test]
    fn test_usage_event_escapes_a_model_name_that_needs_it() {
        use crate::domain::usage_accounting::CostStatus;
        use crate::utils::CostHeader;

        let payload = UsageEvent {
            request_cost: "0.000000",
            input_tokens: "0".to_string(),
            output_tokens: "0".to_string(),
            model_used: r#"we"ird\model"#,
            cost_status: CostStatus::CostUnavailable.as_str(),
        };
        let data = serde_json::to_string(&payload).expect("payload serializes");
        let parsed: serde_json::Value =
            serde_json::from_str(&data).expect("terminal event data must be valid JSON");
        assert_eq!(
            parsed.get(CostHeader::MODEL_USED).and_then(|v| v.as_str()),
            Some(r#"we"ird\model"#)
        );
    }

    async fn response_json(r: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(r.into_body(), 1024 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).expect("response body must be valid JSON")
    }

    #[tokio::test]
    async fn test_tool_choice_unsupported_http_shape() {
        let e = ChatError::ToolError(ProviderError::ToolChoiceUnsupported {
            provider: "anthropic",
            requested: "bad".to_string(),
            supported_values: &["auto", "none", "required"],
        });
        let r = e.into_response();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        let j = response_json(r).await;
        assert_eq!(j["error"]["type"], "tool_choice_unsupported");
        assert_eq!(j["error"]["code"], "tool_choice_unsupported");
        assert_eq!(j["error"]["provider"], "anthropic");
        assert_eq!(j["error"]["requested"], "bad");
    }

    #[tokio::test]
    async fn test_tool_count_exceeded_http_shape() {
        let e = ChatError::ToolError(ProviderError::ToolCountExceeded {
            provider: "anthropic",
            requested: 100,
            limit: 64,
        });
        let r = e.into_response();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        let j = response_json(r).await;
        assert_eq!(j["error"]["type"], "tool_count_exceeded");
        assert_eq!(j["error"]["code"], "tool_count_exceeded");
        assert_eq!(j["error"]["provider"], "anthropic");
        assert_eq!(j["error"]["requested"], 100);
        assert_eq!(j["error"]["limit"], 64);
    }

    #[tokio::test]
    async fn test_malformed_tool_schema_http_shape() {
        let e = ChatError::ToolError(ProviderError::MalformedToolSchema {
            provider: "gateway",
            reason: "name_invalid",
        });
        let r = e.into_response();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        let j = response_json(r).await;
        assert_eq!(j["error"]["type"], "malformed_tool_schema");
        assert_eq!(j["error"]["code"], "malformed_tool_schema");
        assert_eq!(j["error"]["provider"], "gateway");
        assert_eq!(j["error"]["reason"], "name_invalid");
    }

    #[tokio::test]
    async fn test_tool_call_buffer_overflow_http_shape() {
        let e = ChatError::ToolError(ProviderError::ToolCallBufferOverflow {
            provider: "anthropic",
            tool_call_id: "toolu_01".to_string(),
            cap_bytes: 1024,
        });
        let r = e.into_response();
        assert_eq!(r.status(), StatusCode::BAD_GATEWAY);
        let j = response_json(r).await;
        assert_eq!(j["error"]["type"], "gateway_error");
        assert_eq!(j["error"]["code"], "tool_call_buffer_overflow");
        assert_eq!(j["error"]["provider"], "anthropic");
        assert_eq!(j["error"]["tool_call_id"], "toolu_01");
        assert_eq!(j["error"]["cap_bytes"], 1024);
    }

    #[tokio::test]
    async fn test_invalid_request_has_code_string() {
        let e = ChatError::InvalidRequest("bad param".to_string());
        let j = response_json(e.into_response()).await;
        assert_eq!(j["error"]["type"], "invalid_request_error");
        assert_eq!(j["error"]["code"], "invalid_request_error");
    }

    #[tokio::test]
    async fn test_translation_error_has_distinct_code() {
        let e = ChatError::TranslationError("format mismatch".to_string());
        let r = e.into_response();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        let j = response_json(r).await;
        assert_eq!(j["error"]["type"], "translation_error");
        assert_eq!(j["error"]["code"], "translation_error");
    }

    #[tokio::test]
    async fn test_not_yet_supported_http_shape() {
        let e = ChatError::ToolError(ProviderError::NotYetSupported {
            feature: "bedrock_streaming_tool_use",
        });
        let r = e.into_response();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        let j = response_json(r).await;
        assert_eq!(j["error"]["type"], "gateway_error");
        assert_eq!(j["error"]["code"], "not_yet_supported");
        assert_eq!(j["error"]["feature"], "bedrock_streaming_tool_use");
    }
}

/// Handles POST /v1/chat/completions.
#[tracing::instrument(skip_all, fields(model = tracing::field::Empty))]
pub async fn chat_completions(
    State(state): State<AppState>,
    axum::Extension(identity): axum::Extension<RequestIdentity>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, ChatError> {
    if !headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.split(';').next().unwrap_or("").trim() == "application/json")
        .unwrap_or(false)
    {
        return Err(ChatError::InvalidRequest(
            "Content-Type must be application/json".into(),
        ));
    }
    let req: ChatRequest =
        serde_json::from_slice(&body).map_err(|e| ChatError::InvalidRequest(e.to_string()))?;

    // Record model in the tracing span now that we've deserialized.
    tracing::Span::current().record("model", req.model.as_str());

    //: body is fully buffered by axum before the handler runs (same as Json<T>).
    let content_length: Option<u64> = Some(body.len() as u64);

    // Wall-clock start for latency measurement .
    let request_start = std::time::Instant::now();

    let request_id = req
        .request_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let mut req_with_id = req.clone();
    req_with_id.request_id = Some(request_id.clone());
    // SECURITY: Do not honor user-supplied batch flag. Gateway proxies to sync endpoints,
    // not /v1/batches; honoring it would let clients artificially halve reported cost.
    // TODO: Set batch=true only when request is routed to a real batch-compatible flow.
    let batch = false;

    if let Err(reason) = crate::domain::tool_schema::validate_request_tools(&req_with_id) {
        return Err(ChatError::ToolError(
            crate::domain::ports::ProviderError::MalformedToolSchema {
                provider: "gateway",
                reason,
            },
        ));
    }

    let provider = state.provider.read().await.clone();
    let provider_name = provider.metadata().name.clone();

    if req.stream.unwrap_or(false) {
        let (stream, meta) = match provider
            .chat_completion_stream_raw_with_trace(&req_with_id, &body)
            .await
        {
            Ok(routed) => routed,
            Err(e) => {
                let mut resp = ChatError::from(e).into_response();
                cost_headers::inject_zero_cost_headers(&mut resp, &req.model);
                return Ok(resp);
            }
        };
        let AttemptedMeta {
            providers: attempted_providers,
            models: attempted_models,
            fallback_trigger,
            fallback_dispatched,
        } = meta;
        let provider_name = attempted_providers.last().cloned().unwrap_or(provider_name);
        let expose_providers = state.security.read().await.expose_provider_names;
        let model = req.model.clone();
        let provider_name_for_ext = provider_name.clone(); //: for response extension
        // provider_name is moved into `acct`, which is moved into body_stream;
        // provider_name_for_ext is used after.
        //
        // Assembled here, outside the generator, because the generator becomes the response body
        // and must be `'static`. The budget read in particular has to happen here: it snapshots
        // the config as of the request's arrival, and taking it inside the generator would move
        // the snapshot to whenever the stream is first polled.
        let acct = RequestAccounting {
            pricing_db: Arc::clone(&state.pricing_db),
            pool: Arc::clone(&state.pool),
            redis: Arc::clone(&state.redis_pool),
            budget: state.budget_settings.read().await.clone(),
            identity: identity.clone(),
            request_id: request_id.clone(),
            provider_name,
            batch,
            content_length,
            request_start,
        };
        let body_stream = stream! {
            let mut first_model: Option<String> = None;
            let mut last_seen_usage: Option<crate::domain::chat::Usage> = None;
            // Tracks whether the stream ended via an error break. The post-loop emit must
            // only fire on clean EOF — not when the stream was interrupted mid-flight.
            let mut stream_error = false;
            // Set once accounting has run, so the clean-EOF block below cannot finalize a
            // request the terminal chunk already finalized.
            let mut finalized = false;
            let mut stream = std::pin::pin!(stream);
            while let Some(r) = stream.next().await {
                match r.map_err(ChatError::from) {
                    Ok(c) => {
                        // Bookkeeping runs before the chunk is forwarded, because the terminal
                        // chunk can carry the very usage finalization is about to read. Both
                        // accumulator rules keep the semantics they have always had; only their
                        // position moved.
                        if let Some(ref m) = c.model {
                            if let Some(ref prev) = first_model {
                                if prev != m {
                                    warn!(
                                        streaming_model_changed = true,
                                        expected = %prev,
                                        got = %m,
                                        "model changed mid-stream; using first"
                                    );
                                }
                            } else {
                                first_model = Some(m.clone());
                            }
                        }
                        // Last reported usage wins, and a chunk that reports none leaves the
                        // accumulator alone: providers that send usage in multiple chunks (e.g.
                        // Anthropic's message_start + message_delta) still produce exactly one
                        // log line and one spend record, and a terminal chunk carrying no usage
                        // does not erase what an earlier chunk reported.
                        if let Some(ref usage) = c.usage {
                            last_seen_usage = Some(usage.clone());
                        }
                        if c.is_final {
                            // The adapter has declared this chunk the clean end of a completed
                            // upstream response, so the request can be accounted now — before
                            // any of it is forwarded. Everything from here to the first `yield`
                            // is synchronous, so the request is accounted whether or not the
                            // client ever reads another byte; a client that stops at the
                            // provider's terminator, as the OpenAI SDKs do, is accounted all the
                            // same.
                            let model_used = first_model.as_deref().unwrap_or(&model);
                            let usage_event = finalize_stream_accounting(
                                &acct,
                                last_seen_usage.as_ref(),
                                model_used,
                            );
                            finalized = true;
                            yield ChatStreamItem::Ok(c.data);
                            yield ChatStreamItem::Ok(usage_event);
                            // Nothing after a clean terminal chunk is part of the response, so
                            // the provider stream is never polled again.
                            break;
                        }
                        yield ChatStreamItem::Ok(c.data);
                    }
                    Err(e) => {
                        // Known limitation: we emit oxigate.error and do not emit oxigate.usage.
                        // If the provider's final error chunk carried partial usage, that data is
                        // lost to the client. Conservative choice for now — avoids exposing
                        // potentially inconsistent state.
                        // HTTP status remains 200 because headers were already sent; errors are
                        // signaled only via the oxigate.error SSE event.
                        stream_error = true;
                        if let Some(ref u) = last_seen_usage {
                            warn!(
                                error = %e,
                                prompt_tokens = u.prompt_tokens,
                                completion_tokens = u.completion_tokens,
                                "stream interrupted; emitting oxigate.error"
                            );
                        } else {
                            warn!(
                                error = %e,
                                "stream interrupted; emitting oxigate.error (no partial usage available)"
                            );
                        }
                        let msg = json!({
                            "error": "stream_interrupted",
                            "message": e.to_string()
                        });
                        let event = format!("event: oxigate.error\ndata: {}\n\n", msg);
                        yield ChatStreamItem::Ok(Bytes::from(event));
                        break;
                    }
                }
            }

            // Clean end of stream with no chunk marked terminal. Adapters that mark their
            // terminal chunk finalize above and set `finalized`, so this is the fallback for a
            // stream that ended without one — a degraded termination the adapter would not
            // certify as a clean completion, or an adapter that has not opted into the
            // terminal-chunk contract. Such a request is still accounted, but only once the
            // consumer polls past the last chunk.
            //
            // Skipped on error-interrupted streams (stream_error = true) to avoid
            // double-counting partial usage: an interrupted stream is deliberately charged
            // nothing.
            if !stream_error && !finalized {
                // Reaching here means no chunk of this stream was marked terminal, so its
                // accounting depended on the consumer polling once more than the response
                // required. That is the pre-contract behaviour and it is not an error, but it
                // is worth being able to find: it is how a request silently loses its spend row
                // when a client stops reading. Logged with the provider so the cause —
                // a degraded termination, an adapter that has not adopted the contract, or an
                // upstream that sent no terminator — can be told apart per lane.
                tracing::debug!(
                    provider = %acct.provider_name,
                    "stream finalized at end of stream; no terminal chunk was marked"
                );
                let model_used = first_model.as_deref().unwrap_or(&model);
                let usage_event =
                    finalize_stream_accounting(&acct, last_seen_usage.as_ref(), model_used);
                yield ChatStreamItem::Ok(usage_event);
            }
        };
        // Client disconnect propagation: when axum drops this Body (client disconnects),
        // Tokio cancels the `body_stream` future at the next `.await` in the stream! closure.
        // The `ChatCompletionStream` is owned by that closure, so its reqwest TCP connection
        // is released automatically. No explicit CancellationToken is needed.
        // Verified by T-cancel in tests/integration/streaming.rs.
        let body = Body::from_stream(body_stream);
        let mut res = Response::new(body);
        *res.status_mut() = StatusCode::OK;
        res.headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        res.headers_mut()
            .insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        if expose_providers {
            crate::api::inject_attempted_headers(
                res.headers_mut(),
                &attempted_providers,
                &attempted_models,
                fallback_trigger.as_deref(),
                fallback_dispatched,
            );
        }
        //: inject provider label for RequestMetricsLayer (reads from response extensions).
        res.extensions_mut()
            .insert(ProviderLabel(provider_name_for_ext));
        tracing::info!(model = %req.model, "chat completion stream started");
        return Ok(res);
    }

    let (response, meta) = match provider
        .chat_completion_raw_with_trace(&req_with_id, &body)
        .await
    {
        Ok(r) => r,
        Err(e) => {
            let mut resp = ChatError::from(e).into_response();
            cost_headers::inject_zero_cost_headers(&mut resp, &req.model);
            return Ok(resp);
        }
    };

    let AttemptedMeta {
        providers: resp_attempted_providers,
        models: resp_attempted_models,
        fallback_trigger,
        fallback_dispatched,
    } = meta;
    let provider_name = resp_attempted_providers
        .last()
        .cloned()
        .unwrap_or(provider_name);

    let (cost_headers, accounting) = cost_headers::build_cost_headers(
        &response.model,
        &response.usage,
        Arc::clone(&state.pricing_db),
        batch,
    );
    cost_headers::report_finalized_warning(&request_id, &response.model, &accounting);

    let latency_ms = i32::try_from(request_start.elapsed().as_millis()).unwrap_or_else(|_| {
        tracing::warn!("request latency overflows i32; recording -1");
        -1
    });

    // +: structured cost log + spawn-and-forget spend write.
    let record = SpendRecord::build(
        &identity,
        &response.model,
        &provider_name,
        &accounting,
        latency_ms,
    );
    let cost_usd = accounting.cost.total_cost.to_display_string();
    let budget = state.budget_settings.read().await.clone();
    //: request size observability at DEBUG (stays local to chat path).
    tracing::debug!(
        request_id = %request_id,
        request_body_bytes = ?content_length,
        "chat_request_size"
    );
    crate::api::spawn_cost_log_and_spend(
        "chat_completion_cost",
        record,
        &request_id,
        &cost_usd,
        Arc::clone(&state.pool),
        Arc::clone(&state.redis_pool),
        budget,
    );

    //: emit per-request cost counter (nano-USD; divide by 1e9 in PromQL for USD).
    metrics::counter!(
        COST_USD_TOTAL,
        "provider" => provider_name.clone(),
        "endpoint" => ENDPOINT_CHAT
    )
    .increment(accounting.cost.total_cost.as_u64());

    let expose_providers = state.security.read().await.expose_provider_names;
    let mut resp = (StatusCode::OK, cost_headers, Json(response)).into_response();
    //: inject provider label for RequestMetricsLayer.
    resp.extensions_mut().insert(ProviderLabel(provider_name));
    if expose_providers {
        crate::api::inject_attempted_headers(
            resp.headers_mut(),
            &resp_attempted_providers,
            &resp_attempted_models,
            fallback_trigger.as_deref(),
            fallback_dispatched,
        );
    }
    Ok(resp)
}
