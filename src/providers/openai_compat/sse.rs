// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! SSE stream parsing helpers for the OpenAI-compatible adapter.
//!
//! `make_compat_sse_stream` wraps a `reqwest::Response` into a `ChatCompletionStream`
//! with carry-buffer line reassembly, usage extraction, and model tracking.
//! The helper fns are `pub(super)` so the parent module's test suite can unit-test
//! them without exposing them outside `openai_compat`.

use futures::StreamExt;
use tracing::warn;

use crate::domain::chat::{StreamChunk, Usage, UsageAccounting};
use crate::domain::ports::{ChatCompletionStream, ProviderError};
use crate::domain::usage_accounting::PricingContext;
use crate::providers::openai::utils::normalize_openai_usage;
use crate::utils::provider_error::sanitize_network_error;

/// Maximum carry-buffer size for SSE line reassembly.
/// A misbehaving upstream that never sends `\n` would otherwise grow this without bound.
pub(super) const CARRY_MAX_BYTES: usize = 1024 * 1024; // 1 MiB

/// Wraps a successful streaming response into a `ChatCompletionStream`.
///
/// Shared by `chat_completion_stream` and `try_forward_raw_stream` to avoid
/// duplicating the SSE carry-buffer + usage-extraction logic — and shared across provider
/// contracts, not only across the two paths of one adapter.
///
/// `accounting` is therefore a required parameter: the caller knows which contract it is
/// streaming from and this function cannot, since every contract using it emits the same SSE
/// shape. Inferring it here would give them all the same answer.
///
/// `pricing_context` carries the same distinction onto the cache-write axis. Azure streams
/// through here and must account `cache_write_tokens` exactly as its buffered path does, while a
/// generic compat backend must not account it at all — so the value is the caller's, snapshotted
/// once before dispatch and owned by the stream, which outlives the call that built it.
pub(crate) fn make_compat_sse_stream(
    resp: reqwest::Response,
    provider_name: String,
    accounting: UsageAccounting,
    pricing_context: Option<PricingContext>,
) -> ChatCompletionStream {
    let mut raw_stream = resp
        .bytes_stream()
        .map(|r| r.map_err(|e: reqwest::Error| std::io::Error::other(e.to_string())));

    Box::pin(async_stream::stream! {
        let mut last_usage: Option<Usage> = None;
        let mut resolved_model: Option<String> = None;
        let mut carry: Vec<u8> = Vec::new();
        // A `data: [DONE]` line whose event has not been closed by its blank separator yet. It
        // survives across reads because the separator may arrive in a later one.
        let mut done_pending = false;
        // Whether the event currently being read has already had a `data:` line. An event's
        // payload is the concatenation of all its `data:` lines, so only a *sole* `data: [DONE]`
        // is a bare terminator.
        let mut event_has_data = false;

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

            // Whether this read completed the terminator *event*. Derived here rather than from
            // the bytes about to be yielded: a read is not a line. The marker can arrive split
            // across two reads, or bundled behind other frames in one — the carry buffer already
            // resolves both, and a byte test on `data` would resolve neither.
            //
            // A line is not an event either. An SSE event is dispatched by the empty line that
            // closes it, so the terminator is incomplete until that separator has been read.
            // Marking the read that carried `data: [DONE]` alone would declare the stream over
            // while the event was still half-written, and whatever the gateway appends next would
            // merge into it.
            let mut saw_done = false;

            while let Some(pos) = carry.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = carry.drain(..=pos).collect();
                // An event is dispatched by an *empty* line — exactly `\n` or `\r\n`, tested on
                // the raw bytes. A line of spaces is not empty: it is an unrecognized field line,
                // which the spec ignores and which leaves the event open. Trimming first would
                // make it look like the delimiter and close the terminator one line early.
                if line_bytes.as_slice() == b"\n" || line_bytes.as_slice() == b"\r\n" {
                    if done_pending {
                        saw_done = true;
                        done_pending = false;
                    }
                    event_has_data = false;
                    continue;
                }
                if let Ok(s) = std::str::from_utf8(&line_bytes) {
                    let trimmed = s.trim();
                    if trimmed.starts_with("data:") {
                        // The terminator is an event whose payload is exactly `[DONE]`, which
                        // means a *sole* `data:` line carrying the marker. A second `data:` line
                        // in the same event — before it or after it — makes the payload a
                        // concatenation, so the event is not a bare terminator and must not be
                        // marked: doing so would end the response while real content was still
                        // in it. Other fields carry nothing this parser reads and are ignored,
                        // as the spec ignores them — they do not close or invalidate the event.
                        done_pending = is_done_marker(trimmed) && !event_has_data;
                        event_has_data = true;
                        let Some(parsed) = parse_sse_data(trimmed) else {
                            continue;
                        };
                        if let Some(u) = extract_usage_from_value(&parsed, accounting, pricing_context.as_ref()) {
                            last_usage = Some(u);
                        }
                        if resolved_model.is_none() {
                            resolved_model = extract_model_from_value(&parsed);
                        }
                    }
                }
            }

            yield Ok(StreamChunk {
                is_final: saw_done,
                ..StreamChunk::new(data, last_usage.clone(), resolved_model.clone())
            });
        }

        if let Ok(s) = std::str::from_utf8(&carry) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                // Parsed only to tell a complete final line from a truncated one. Every chunk has
                // already been yielded by now, so a usage value found here has nothing left to
                // ride on — which is why the result is a boolean rather than a value.
                let complete_usage_line =
                    extract_usage_from_sse_line(trimmed, accounting, pricing_context.as_ref())
                        .is_some();
                if !complete_usage_line && trimmed != "[DONE]" {
                    warn!(
                        provider = %provider_name,
                        "compat: stream ended with incomplete SSE line — possible truncation"
                    );
                }
            }
        }
    })
}

/// Reports whether a complete SSE line is the `[DONE]` terminator.
///
/// Shares [`parse_sse_data`]'s prefix handling so the two agree on what a `data:` line carries —
/// that function discards the marker as a non-payload, and this one is what remembers it was seen.
pub(super) fn is_done_marker(line: &str) -> bool {
    line.trim()
        .strip_prefix("data:")
        .is_some_and(|s| s.trim_start() == "[DONE]")
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

/// Parses the `usage` object out of one SSE payload, normalized and carrying the caller's
/// accounting declaration and, where the lane declares one, its pricing generation.
pub(super) fn extract_usage_from_value(
    v: &serde_json::Value,
    accounting: UsageAccounting,
    pricing_context: Option<&PricingContext>,
) -> Option<Usage> {
    let usage_val = v.get("usage").filter(|u| !u.is_null())?;
    let mut u: Usage = serde_json::from_value(usage_val.clone()).ok()?;
    normalize_openai_usage(&mut u, accounting, pricing_context);
    Some(u)
}

/// Scan a reassembled SSE `data:` line for a `usage` field.
///
/// Returns `Some(usage)` when `usage` is present and non-null. Applies
/// `normalize_openai_usage` to map `prompt_tokens_details.cached_tokens` to
/// `cache_read_input_tokens` for any provider that emits the OpenAI cache shape.
pub(super) fn extract_usage_from_sse_line(
    line: &str,
    accounting: UsageAccounting,
    pricing_context: Option<&PricingContext>,
) -> Option<Usage> {
    extract_usage_from_value(&parse_sse_data(line)?, accounting, pricing_context)
}

pub(super) fn extract_model_from_value(v: &serde_json::Value) -> Option<String> {
    v.get("model")?.as_str().map(String::from)
}
