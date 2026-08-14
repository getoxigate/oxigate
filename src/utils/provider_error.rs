// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Safe error summarization for provider unreachable errors.
//!
//! Avoids leaking URLs or credentials from reqwest/io errors into API responses.

/// Returns a short, safe summary for client-facing unreachable messages.
#[must_use]
pub fn sanitize_network_error(msg: &str) -> String {
    let s = msg.to_lowercase();
    if s.contains("connection refused") {
        "connection refused".into()
    } else if s.contains("timed out") || s.contains("timeout") {
        "timeout".into()
    } else if s.contains("dns") || s.contains("name resolution") {
        "dns resolution failed".into()
    } else if s.contains("connection reset") {
        "connection reset".into()
    } else if s.contains("tls") || s.contains("ssl") || s.contains("certificate") {
        "tls error".into()
    } else {
        "network error".into()
    }
}

/// Returns a short, safe summary of a reqwest error for client-facing messages.
#[must_use]
pub fn sanitize_reqwest_error(e: &reqwest::Error) -> String {
    sanitize_network_error(&e.to_string())
}

/// Converts a reqwest error into a `ProviderError`, classifying timeouts separately.
///
/// Callers must record `let start = Instant::now()` before `.send().await` and pass
/// `start.elapsed().as_millis() as u64` as `elapsed_ms`.
#[must_use]
pub fn classify_reqwest_error(
    e: reqwest::Error,
    elapsed_ms: u64,
) -> crate::domain::ports::ProviderError {
    if e.is_timeout() {
        crate::domain::ports::ProviderError::Timeout { elapsed_ms }
    } else {
        crate::domain::ports::ProviderError::Unreachable(sanitize_reqwest_error(&e))
    }
}
