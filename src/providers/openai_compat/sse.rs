// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! SSE stream parsing helpers for the OpenAI-compatible adapter.
//!
//! `make_compat_sse_stream` wraps a `reqwest::Response` into a `ChatCompletionStream`
//! with carry-buffer line reassembly, usage extraction, and model tracking.
//! The four helper fns are `pub(super)` so the parent module's test suite can unit-test
//! them without exposing them outside `openai_compat`.

use futures::StreamExt;
use tracing::warn;

use crate::domain::chat::{StreamChunk, Usage};
use crate::domain::ports::{ChatCompletionStream, ProviderError};
use crate::providers::openai::utils::normalize_openai_usage;
use crate::utils::provider_error::sanitize_network_error;

/// Maximum carry-buffer size for SSE line reassembly.
/// A misbehaving upstream that never sends `\n` would otherwise grow this without bound.
pub(super) const CARRY_MAX_BYTES: usize = 1024 * 1024; // 1 MiB

/// Wraps a successful streaming response into a `ChatCompletionStream`.
///
/// Shared by `chat_completion_stream` and `try_forward_raw_stream` to avoid
/// duplicating the SSE carry-buffer + usage-extraction logic.
pub(crate) fn make_compat_sse_stream(
    resp: reqwest::Response,
    provider_name: String,
) -> ChatCompletionStream {
    let mut raw_stream = resp
        .bytes_stream()
        .map(|r| r.map_err(|e: reqwest::Error| std::io::Error::other(e.to_string())));

    Box::pin(async_stream::stream! {
        let mut last_usage: Option<Usage> = None;
        let mut resolved_model: Option<String> = None;
        let mut carry: Vec<u8> = Vec::new();

        while let Some(chunk_res) = raw_stream.next().await {
            let data = match chunk_res {
                Ok(b) => b,
                Err(e) => {
                    yield Err(ProviderError::Unreachable(format!(
                        "compat({}): {}",
                        provider_name,
                        sanitize_network_error(&e.to_string())
                    )));
                    break;
                }
            };

            if carry.len() + data.len() > CARRY_MAX_BYTES {
                warn!(
                    provider = %provider_name,
                    carry_bytes = carry.len(),
                    chunk_bytes = data.len(),
                    "compat: SSE carry buffer would exceed 1 MiB — upstream sent no newlines; aborting stream"
                );
                yield Err(ProviderError::ProviderUnavailable(format!(
                    "compat({provider_name}): SSE line exceeded 1 MiB limit"
                )));
                break;
            }
            carry.extend_from_slice(&data);

            while let Some(pos) = carry.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = carry.drain(..=pos).collect();
                if let Ok(s) = std::str::from_utf8(&line_bytes) {
                    let trimmed = s.trim();
                    if trimmed.starts_with("data:") {
                        let Some(parsed) = parse_sse_data(trimmed) else {
                            continue;
                        };
                        if let Some(u) = extract_usage_from_value(&parsed) {
                            last_usage = Some(u);
                        }
                        if resolved_model.is_none() {
                            resolved_model = extract_model_from_value(&parsed);
                        }
                    }
                }
            }

            yield Ok(StreamChunk::new(data, last_usage.clone(), resolved_model.clone()));
        }

        if let Ok(s) = std::str::from_utf8(&carry) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                if let Some(u) = extract_usage_from_sse_line(trimmed) {
                    last_usage = Some(u);
                } else if trimmed != "[DONE]" {
                    warn!(
                        provider = %provider_name,
                        "compat: stream ended with incomplete SSE line — possible truncation"
                    );
                }
            }
        }

        if last_usage.is_none() {
            warn!(
                provider = %provider_name,
                model = %resolved_model.as_deref().unwrap_or("unknown"),
                "compat streaming: upstream returned no usage data; cost will be zero for this request"
            );
        }
    })
}

/// Strips the `data:` prefix (with or without trailing space — WHATWG SSE spec §9.2.6),
/// rejects `[DONE]`/empty, and parses the JSON payload.
pub(super) fn parse_sse_data(line: &str) -> Option<serde_json::Value> {
    let s = line.trim().strip_prefix("data:")?.trim_start();
    if s == "[DONE]" || s.is_empty() {
        return None;
    }
    serde_json::from_str(s).ok()
}

pub(super) fn extract_usage_from_value(v: &serde_json::Value) -> Option<Usage> {
    let usage_val = v.get("usage").filter(|u| !u.is_null())?;
    let mut u: Usage = serde_json::from_value(usage_val.clone()).ok()?;
    normalize_openai_usage(&mut u);
    Some(u)
}

/// Scan a reassembled SSE `data:` line for a `usage` field.
///
/// Returns `Some(usage)` when `usage` is present and non-null. Applies
/// `normalize_openai_usage` to map `prompt_tokens_details.cached_tokens` to
/// `cache_read_input_tokens` for any provider that emits the OpenAI cache shape.
pub(super) fn extract_usage_from_sse_line(line: &str) -> Option<Usage> {
    extract_usage_from_value(&parse_sse_data(line)?)
}

pub(super) fn extract_model_from_value(v: &serde_json::Value) -> Option<String> {
    v.get("model")?.as_str().map(String::from)
}
