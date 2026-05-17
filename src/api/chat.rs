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
use crate::observability::metrics::COST_USD_TOTAL;
use crate::utils::{CostHeader, cost_headers};

/// Yield type for chat streaming: Bytes on success, Infallible (never Err).
type ChatStreamItem = Result<Bytes, std::convert::Infallible>;

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
        let pricing_db = Arc::clone(&state.pricing_db);
        let model = req.model.clone();
        let identity = identity.clone();
        let pool = Arc::clone(&state.pool);
        let redis = Arc::clone(&state.redis_pool);
        let budget = state.budget_settings.read().await.clone();
        let request_id = request_id.clone();
        let provider_name_for_ext = provider_name.clone(); //: for response extension
        // provider_name is moved into body_stream; provider_name_for_ext is used after.
        let body_stream = stream! {
            let mut first_model: Option<String> = None;
            let mut last_seen_usage: Option<crate::domain::chat::Usage> = None;
            // Tracks whether the stream ended via an error break. The post-loop emit must
            // only fire on clean EOF — not when the stream was interrupted mid-flight.
            let mut stream_error = false;
            let mut stream = std::pin::pin!(stream);
            while let Some(r) = stream.next().await {
                match r.map_err(ChatError::from) {
                    Ok(c) => {
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
                        yield ChatStreamItem::Ok(c.data);
                        // Accumulate the latest usage; actual emit happens after stream EOF
                        // so providers that send usage in multiple chunks (e.g. Anthropic's
                        // message_start + message_delta) produce exactly one log line and one
                        // spend record.
                        if let Some(ref usage) = c.usage {
                            last_seen_usage = Some(usage.clone());
                        }
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

            // +: emit oxigate.usage SSE event, structured cost log, and spend
            // write exactly once after stream EOF. Skipped on error-interrupted streams
            // (stream_error = true) to avoid double-counting partial usage.
            if !stream_error && let Some(ref usage) = last_seen_usage {
                    let model_used = first_model.as_deref().unwrap_or(&model);
                    let (headers, cost_breakdown, token_usage) = cost_headers::build_cost_headers(
                        model_used,
                        usage,
                        pricing_db.clone(),
                        batch,
                    );
                    let cost = headers
                        .get(CostHeader::REQUEST_COST)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("0");
                    let input = headers
                        .get(CostHeader::INPUT_TOKENS)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("0");
                    let output = headers
                        .get(CostHeader::OUTPUT_TOKENS)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("0");
                    let model_used_val = model_used.to_string();
                    let event = format!(
                        "event: oxigate.usage\ndata: {{\"{}\":\"{}\",\"{}\":\"{}\",\"{}\":\"{}\",\"{}\":\"{}\"}}\n\n",
                        CostHeader::REQUEST_COST,
                        cost,
                        CostHeader::INPUT_TOKENS,
                        input,
                        CostHeader::OUTPUT_TOKENS,
                        output,
                        CostHeader::MODEL_USED,
                        model_used_val,
                    );
                    yield ChatStreamItem::Ok(Bytes::from(event));

                    let latency_ms = i32::try_from(request_start.elapsed().as_millis())
                        .unwrap_or_else(|_| {
                            tracing::warn!(
                                "streaming request latency overflows i32; recording -1"
                            );
                            -1
                        });
                    let record = SpendRecord::build(
                        &identity,
                        &model_used_val,
                        &provider_name,
                        &token_usage,
                        &cost_breakdown,
                        latency_ms,
                    );
                    //: emit per-request cost counter (nano-USD).
                    metrics::counter!(
                        COST_USD_TOTAL,
                        "provider" => provider_name.clone()
                    )
                    .increment(cost_breakdown.total_cost.as_u64());

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
                        cost,
                        Arc::clone(&pool),
                        Arc::clone(&redis),
                        budget.clone(),
                    );
            }

            // Clean provider EOF with no usage in any chunk — not a client disconnect: on disconnect
            // Tokio cancels this future at the next suspension (e.g. stream.next().await), so control
            // never reaches this post-loop block. Use this message for log routing, not as a disconnect signal.
            if !stream_error && last_seen_usage.is_none() {
                tracing::warn!(
                    request_id = %request_id,
                    model = %model,
                    "stream_eof_no_usage"
                );
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

    let (cost_headers, cost_breakdown, token_usage) = cost_headers::build_cost_headers(
        &response.model,
        &response.usage,
        Arc::clone(&state.pricing_db),
        batch,
    );

    let latency_ms = i32::try_from(request_start.elapsed().as_millis()).unwrap_or_else(|_| {
        tracing::warn!("request latency overflows i32; recording -1");
        -1
    });

    // +: structured cost log + spawn-and-forget spend write.
    let record = SpendRecord::build(
        &identity,
        &response.model,
        &provider_name,
        &token_usage,
        &cost_breakdown,
        latency_ms,
    );
    let cost_usd = cost_breakdown.total_cost.to_display_string();
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
        "provider" => provider_name.clone()
    )
    .increment(cost_breakdown.total_cost.as_u64());

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
