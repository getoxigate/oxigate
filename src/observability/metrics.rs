// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Metric definitions for OxiGate observability​.
//!
//! Defines counters and histograms for request, cost, fallback, and retry telemetry using
//! the `metrics` facade. `src/observability/mod.rs` installs the Prometheus exporter
//! (`metrics-exporter-prometheus`) so these definitions are automatically scraped via
//! `GET /metrics`.
//!
//! ## Cardinality guardrail
//! Labels are restricted to low-cardinality values (provider names, trigger strings, HTTP
//! method/status codes). Never use `key_id`, `user_id`, `model`, request IDs, or raw error
//! messages as labels.

// ---------------------------------------------------------------------------
// Metric names (constants — single source of truth for both emitter and tests)
// ---------------------------------------------------------------------------

// ---: baseline request + cost metrics ---

/// Counter: total LLM requests dispatched.
/// Labels: `method` (HTTP method), `status` (HTTP status code), `provider` (provider name).
pub const REQUESTS_TOTAL: &str = "oxigate_requests_total";

/// Histogram: end-to-end request latency in seconds (time-to-first-byte for streaming).
/// Label: `provider` (provider name).
/// Explicit buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0].
pub const REQUEST_DURATION_SECONDS: &str = "oxigate_request_duration_seconds";

/// Counter: accumulated request cost in nano-USD (NanoUsd units; divide by 1e9 for USD in PromQL).
/// Label: `provider` (provider name).
pub const COST_USD_TOTAL: &str = "oxigate_cost_usd_total";

/// Gauge: current number of in-flight LLM requests (concurrent connections).
/// No labels.
pub const ACTIVE_CONNECTIONS: &str = "oxigate_active_connections";

// ---: fallback + retry metrics ---

/// Counter: incremented once per fallback dispatch, labelled by the trigger type.
/// Label: `trigger` — snake_case trigger string (e.g. `rate_limit`, `timeout`).
pub const FALLBACK_TRIGGER_TOTAL: &str = "oxigate_fallback_trigger_total";

/// Counter: incremented once per retry attempt, labelled by provider and trigger.
/// Labels: `provider`, `trigger`.
pub const RETRY_ATTEMPT_TOTAL: &str = "oxigate_retry_attempt_total";

/// Counter: incremented once per skipped fallback target, labelled by skip reason.
/// Label: `reason` — snake_case skip reason (e.g. `trigger_not_allowed`, `in_cooldown`).
pub const FALLBACK_SKIP_TOTAL: &str = "oxigate_fallback_skip_total";

/// Histogram: start-to-terminal latency for the full fallback resolution pipeline (seconds).
pub const FALLBACK_RESOLUTION_SECONDS: &str = "oxigate_fallback_resolution_seconds";

/// Histogram: total number of dispatched attempts (retries + fallback targets) per request.
pub const FALLBACK_RESOLUTION_ATTEMPTS: &str = "oxigate_fallback_resolution_attempts";

// ---------------------------------------------------------------------------
// Emission helpers
// ---------------------------------------------------------------------------

/// Records metrics for a completed fallback dispatch pipeline.
///
/// Called from `dispatch_with_fallback` and `dispatch_stream_with_fallback` at terminal.
/// No-op when no recorder is installed (the default installs one).
/// `provider` parameter is reserved for future use (retry tracking in retry.rs).
pub fn record_dispatch(
    trigger: Option<&str>,
    attempts_dispatched: usize,
    skipped_count: usize,
    elapsed_secs: f64,
    _provider: &str,
) {
    if let Some(t) = trigger {
        metrics::counter!(FALLBACK_TRIGGER_TOTAL, "trigger" => t.to_owned()).increment(1);
    }
    if skipped_count > 0 {
        // Individual skip reasons are recorded at skip-time by the caller; record the total here.
        metrics::counter!(FALLBACK_SKIP_TOTAL, "reason" => "any").increment(skipped_count as u64);
    }
    metrics::histogram!(FALLBACK_RESOLUTION_SECONDS).record(elapsed_secs);
    metrics::histogram!(FALLBACK_RESOLUTION_ATTEMPTS).record(attempts_dispatched as f64);
}

/// Records a single same-provider retry attempt. Called from `retry_loop` for each retried error.
///
/// `provider`: the provider being retried. `trigger`: the trigger that caused the retry.
pub fn record_retry(provider: &str, trigger: &str) {
    metrics::counter!(
        RETRY_ATTEMPT_TOTAL,
        "provider" => provider.to_owned(),
        "trigger" => trigger.to_owned()
    )
    .increment(1);
}

/// Records a single skipped fallback target with its reason.
pub fn record_skip(reason: &str) {
    metrics::counter!(FALLBACK_SKIP_TOTAL, "reason" => reason.to_owned()).increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_name_constants_are_snake_case() {
        for name in [
            REQUESTS_TOTAL,
            REQUEST_DURATION_SECONDS,
            COST_USD_TOTAL,
            ACTIVE_CONNECTIONS,
            FALLBACK_TRIGGER_TOTAL,
            RETRY_ATTEMPT_TOTAL,
            FALLBACK_SKIP_TOTAL,
            FALLBACK_RESOLUTION_SECONDS,
            FALLBACK_RESOLUTION_ATTEMPTS,
        ] {
            assert!(
                name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
                "metric name {name:?} contains non-snake_case characters"
            );
        }
    }
}
