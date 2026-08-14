// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Shared utilities extracted from the OpenAI adapter for use by OpenAICompatAdapter.

use futures::StreamExt;
use tracing::debug;

use crate::domain::chat::{ChatRequest, Usage};
use crate::domain::ports::ProviderError;

/// Maximum bytes read from an upstream error body. Prevents hostile upstreams from
/// forcing large allocations on the error path.
const ERROR_BODY_CAP: usize = 4 * 1024;

/// Maps `prompt_tokens_details.cached_tokens` → `cache_read_input_tokens`.
///
/// OpenAI reports cached prompt tokens in `prompt_tokens_details.cached_tokens`.
/// The domain `Usage` model uses `cache_read_input_tokens` for pricing. This function
/// bridges the two representations so the cost layer sees the correct cache signal.
pub fn normalize_openai_usage(usage: &mut Usage) {
    if let Some(ref d) = usage.prompt_tokens_details
        && d.cached_tokens.is_some()
    {
        usage.cache_read_input_tokens = d.cached_tokens;
    }
}

/// Injects `stream_options.include_usage: true` unless the client already set it to any value.
///
/// Without this injection OpenAI (and Azure) emit NO usage data in any streaming chunk
/// (`usage: null` on every chunk). The injection is mandatory for any cost tracking.
///
/// Cases:
/// - `Some(true)` → already set; log debug, no-op.
/// - `Some(false)` → client opted out; log debug, no-op (client value wins).
/// - `None` → inject `true`.
pub fn inject_stream_options(req: &mut ChatRequest) {
    let existing = req
        .extra
        .get("stream_options")
        .and_then(|o| o.get("include_usage"))
        .and_then(|v| v.as_bool());
    match existing {
        Some(true) => {
            debug!("stream_options.include_usage already true; cost tracking will be precise");
        }
        Some(false) => {
            debug!(
                "stream_options.include_usage=false from client; cost tracking will be imprecise for this request"
            );
        }
        None => {
            let mut opts = req
                .extra
                .get("stream_options")
                .and_then(|v| v.as_object().cloned())
                .unwrap_or_default();
            opts.insert("include_usage".into(), serde_json::json!(true));
            req.extra.insert("stream_options".into(), opts.into());
        }
    }
}

/// Maps an HTTP status code + pre-extracted message to a [`ProviderError`].
///
/// Shared primitive used by [`map_openai_error_response`] and `azure::map_error_response`.
/// Callers are responsible for reading the response body, extracting the message string,
/// and reading the `Retry-After` header before calling this function.
pub(crate) fn map_status_to_provider_error(
    status: reqwest::StatusCode,
    msg: String,
    retry_after: Option<u64>,
) -> ProviderError {
    match status.as_u16() {
        400 => ProviderError::InvalidRequest(msg),
        401 => ProviderError::Auth(msg),
        403 => ProviderError::Auth(format!("forbidden: {msg}")),
        404 => ProviderError::UnknownModel(msg),
        429 => ProviderError::RateLimited { retry_after },
        500 | 502 | 503 => ProviderError::ProviderUnavailable(msg),
        _ => ProviderError::ProviderHttpError {
            status: status.as_u16(),
            body: msg,
        },
    }
}

/// Shared error-response mapper for the OpenAI adapter family (OpenAI, compat).
///
/// Reads the `Retry-After` header and body (bounded at `ERROR_BODY_CAP`), extracts
/// `error.message` from the JSON body, then delegates to [`map_status_to_provider_error`].
///
/// Azure uses its own `map_error_response` to perform content-filter detection before
/// delegating to [`map_status_to_provider_error`] for the generic status table.
pub async fn map_openai_error_response(
    status: reqwest::StatusCode,
    resp: reqwest::Response,
) -> ProviderError {
    let retry_after = resp
        .headers()
        .get("Retry-After")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());
    // Bounded read: stop after ERROR_BODY_CAP bytes so a hostile upstream cannot force
    // a large heap allocation by sending a multi-MB error body.
    let mut body_bytes: Vec<u8> = Vec::with_capacity(ERROR_BODY_CAP);
    let mut stream = resp.bytes_stream();
    while let Some(Ok(chunk)) = stream.next().await {
        let remaining = ERROR_BODY_CAP.saturating_sub(body_bytes.len());
        if remaining == 0 {
            break;
        }
        body_bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }
    let msg = serde_json::from_slice::<serde_json::Value>(&body_bytes)
        .ok()
        .and_then(|j| {
            j.get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(String::from)
        })
        .unwrap_or_else(|| String::from_utf8_lossy(&body_bytes).into_owned());

    map_status_to_provider_error(status, msg, retry_after)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::{ChatRequest, Message, MessageContent, Role};

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

    #[test]
    fn normalize_usage_maps_cached_tokens() {
        let mut usage = Usage {
            prompt_tokens: 100,
            prompt_tokens_details: Some(crate::domain::chat::PromptTokensDetails {
                cached_tokens: Some(40),
            }),
            cache_read_input_tokens: None,
            ..Default::default()
        };
        normalize_openai_usage(&mut usage);
        assert_eq!(usage.cache_read_input_tokens, Some(40));
    }

    #[test]
    fn normalize_usage_noop_when_no_details() {
        let mut usage = Usage {
            prompt_tokens: 100,
            prompt_tokens_details: None,
            cache_read_input_tokens: None,
            ..Default::default()
        };
        normalize_openai_usage(&mut usage);
        assert_eq!(usage.cache_read_input_tokens, None);
    }

    #[test]
    fn inject_stream_options_adds_include_usage_when_absent() {
        let mut req = minimal_request();
        inject_stream_options(&mut req);
        let v = req
            .extra
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(|v| v.as_bool());
        assert_eq!(v, Some(true));
    }

    #[test]
    fn inject_stream_options_noop_when_already_true() {
        let mut req = minimal_request();
        req.extra.insert(
            "stream_options".into(),
            serde_json::json!({"include_usage": true}),
        );
        inject_stream_options(&mut req);
        let v = req
            .extra
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(|v| v.as_bool());
        assert_eq!(v, Some(true));
    }

    #[test]
    fn inject_stream_options_respects_client_false() {
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
        // Client opted out — must be preserved as false, not overridden.
        assert_eq!(v, Some(false));
    }
}
