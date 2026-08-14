// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Retry loop and retryability predicate for ProviderRouter .

use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::Rng;

use crate::domain::ports::{ProviderAdapter, ProviderError};
use crate::observability::metrics;
use crate::providers::router::fallback_trace::{FetchAttempt, trigger_header_value};

use super::ProviderRouter;

/// Returns true if the error is transient and the same provider should be retried.
pub(super) fn is_retryable(e: &ProviderError) -> bool {
    matches!(
        e,
        ProviderError::RateLimited { .. }
            | ProviderError::ProviderUnavailable(_)
            | ProviderError::Unreachable(_)
            | ProviderError::Timeout { .. }
    )
}

impl ProviderRouter {
    /// Attempts a single-provider dispatch with exponential backoff retries.
    ///
    /// Retries only on transient errors (`RateLimited`, `ProviderUnavailable`, `Unreachable`).
    /// Non-transient errors (`Auth`, `InvalidRequest`, `ContentFiltered`, etc.) return immediately.
    ///
    /// `track_success`: when `true`, calls `on_response()` on the first `Ok` return. Pass `false`
    /// for streaming dispatch — stream-open success does not mean the stream was fully consumed;
    /// `GuardedStream` calls `on_response()` at stream EOF, which is the correct measurement point.
    /// Passing `true` for streaming would double-count latency and prematurely close HALF-OPEN.
    pub(super) async fn retry_loop<F, Fut, R>(
        &self,
        adapter: &Arc<dyn ProviderAdapter>,
        name: &str,
        model: &str,
        dispatch: F,
        track_success: bool,
    ) -> (Result<R, ProviderError>, Vec<FetchAttempt>)
    where
        F: Fn(Arc<dyn ProviderAdapter>) -> Fut,
        Fut: std::future::Future<Output = Result<R, ProviderError>>,
    {
        let mut last_err: Option<ProviderError> = None;
        let mut fetch_attempts: Vec<FetchAttempt> = Vec::new();
        for attempt in 0..=self.retry.max_retries {
            if attempt > 0 {
                let raw_delay = (self.retry.base_delay_ms as f64
                    * self.retry.multiplier.powi(attempt as i32 - 1))
                    as u64;
                let capped = raw_delay.min(self.retry.max_delay_ms);
                let jitter = rand::thread_rng().gen_range(0..=self.retry.jitter_ms);
                tokio::time::sleep(Duration::from_millis(capped + jitter)).await;
            }

            let start = Instant::now();
            let result = dispatch(Arc::clone(adapter)).await;
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

            // Record this dispatch attempt. trigger = previous failure's trigger (None for attempt 0).
            let error_class = result.as_ref().err().map(|e| e.error_class().to_owned());
            fetch_attempts.push(FetchAttempt {
                provider: name.to_string(),
                model: model.to_string(),
                is_retry: attempt > 0,
                trigger: last_err.as_ref().map(|e| e.to_trigger()),
                attempted: true,
                skip_reason: None,
                error_class,
                latency_ms: Some(latency_ms),
            });

            match result {
                Ok(v) => {
                    if track_success {
                        self.health.on_response(name, latency_ms).await;
                    }
                    return (Ok(v), fetch_attempts);
                }
                Err(e) if is_retryable(&e) => {
                    // Health tracking is independent of retry policy: a provider that returns a
                    // 429, 503, or timeout is distressed regardless of whether we choose to
                    // retry it. Update the cooldown/half-open state before the retry.on gate so
                    // subsequent routing decisions don't continue to favour a known-bad provider.
                    // Timeout is included for consistency with fallback.rs and streaming.rs,
                    // where it already participates in provider distress signalling.
                    if matches!(
                        &e,
                        ProviderError::RateLimited { .. }
                            | ProviderError::ProviderUnavailable(_)
                            | ProviderError::Timeout { .. }
                    ) {
                        self.health.on_rate_limit(name).await;
                    }
                    //: if retry.on is set, only retry when the trigger matches.
                    if let Some(allowed) = &self.retry.on
                        && !allowed.contains(&e.to_trigger())
                    {
                        return (Err(e), fetch_attempts);
                    }
                    // Only emit the retry metric when a retry will actually follow.
                    // On the last iteration (attempt == max_retries) no retry occurs,
                    // so counting here would overstate oxigate_retry_attempt_total.
                    if attempt < self.retry.max_retries {
                        tracing::warn!(
                            provider = %name,
                            attempt,
                            max_retries = self.retry.max_retries,
                            error = %e,
                            "retryable provider error; will retry"
                        );
                        metrics::record_retry(name, trigger_header_value(&e.to_trigger()));
                    }
                    last_err = Some(e);
                }
                Err(e) => return (Err(e), fetch_attempts),
            }
        }
        (
            Err(last_err.expect("loop ran at least once")),
            fetch_attempts,
        )
    }
}
