// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! GuardedStream and streaming fallback dispatch for ProviderRouter .

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use futures::Stream;
use tokio_stream::StreamExt as TokioStreamExt;

use crate::domain::chat::{ChatRequest, StreamChunk};
use crate::domain::ports::{ChatCompletionStream, ProviderError};
use crate::providers::health::InFlightGuard;
use crate::providers::router::fallback_trace::{
    DecisionOutcome, FallbackDecisionTrace, FetchAttempt, SkipReason,
};

use super::fallback::log_dispatch_terminal;

use super::ProviderRouter;

/// Wraps a `ChatCompletionStream` with an `InFlightGuard` so the in-flight counter
/// stays incremented for the full duration of the stream (not just until `await` returns).
///
/// wraps inner stream + guard.
/// adds mid-stream health tracking (on_rate_limit / on_response).
///
/// The inter-chunk timeout is pre-applied by the caller: `chat_completion_stream_routed`
/// maps `Timeout<S>` items to `ProviderError::Unreachable` before passing the stream
/// here, so `inner` has the standard `ChatCompletionStream` item type.
pub(super) struct GuardedStream {
    pub(super) inner: ChatCompletionStream,
    pub(super) _guard: InFlightGuard,
    pub(super) tracker: Arc<crate::providers::health::ProviderHealthTracker>,
    pub(super) provider_name: String,
    pub(super) stream_start: Instant,
}

// Pin<Box<T>>: Unpin because Box<T>: Unpin. Arc<T>: Unpin.
impl Unpin for GuardedStream {}

impl std::fmt::Debug for GuardedStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardedStream").finish_non_exhaustive()
    }
}

impl Stream for GuardedStream {
    type Item = Result<StreamChunk, ProviderError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // GuardedStream: Unpin — Pin::into_inner is safe.
        let this = Pin::into_inner(self);
        let result = this.inner.as_mut().poll_next(cx);
        // Fire-and-forget health tracker updates. Cannot await in poll_next.
        match &result {
            Poll::Ready(Some(Err(e))) => {
                if matches!(
                    e,
                    ProviderError::RateLimited { .. } | ProviderError::ProviderUnavailable(_)
                ) {
                    let tracker = Arc::clone(&this.tracker);
                    let name = this.provider_name.clone();
                    tokio::spawn(async move { tracker.on_rate_limit(&name).await });
                }
            }
            Poll::Ready(None) => {
                let latency_ms = this.stream_start.elapsed().as_secs_f64() * 1000.0;
                let tracker = Arc::clone(&this.tracker);
                let name = this.provider_name.clone();
                tokio::spawn(async move { tracker.on_response(&name, latency_ms).await });
            }
            _ => {}
        }
        result
    }
}

impl ProviderRouter {
    /// Wraps a raw stream with inter-chunk timeout and an in-flight guard.
    ///
    /// The guard keeps the provider's in-flight counter incremented for the full
    /// duration of stream consumption. The timeout terminates the stream with an error
    /// if no chunk arrives within `retry.stream_chunk_timeout_ms`.
    pub(super) fn wrap_stream_with_guard(
        &self,
        inner: ChatCompletionStream,
        guard: InFlightGuard,
        provider_name: String,
    ) -> ChatCompletionStream {
        let chunk_timeout_ms = self.retry.stream_chunk_timeout_ms;
        let timeout_duration = Duration::from_millis(chunk_timeout_ms);
        // Timeout<S> is !Unpin (contains tokio::time::Sleep). Box::pin makes it Unpin
        // (Pin<Box<T>>: Unpin since Box<T>: Unpin) so we can call .next() on &mut.
        let timed: ChatCompletionStream = Box::pin(async_stream::stream! {
            let mut timeout_stream = Box::pin(TokioStreamExt::timeout(inner, timeout_duration));
            loop {
                match TokioStreamExt::next(&mut timeout_stream).await {
                    Some(Ok(item)) => yield item,
                    Some(Err(_elapsed)) => {
                        yield Err(ProviderError::Timeout {
                            elapsed_ms: chunk_timeout_ms,
                        });
                        break;
                    }
                    None => break,
                }
            }
        });
        Box::pin(GuardedStream {
            inner: timed,
            _guard: guard,
            tracker: Arc::clone(&self.health),
            provider_name,
            stream_start: Instant::now(),
        })
    }

    /// Dispatches a streaming request with pre-stream fallback cascade.
    ///
    /// When `raw_body` is `Some`, each dispatch attempt calls `try_forward_raw_stream` first
    /// (zero-copy path); on `None` return from the adapter, falls back to
    /// `chat_completion_stream`. Pass `None` for the normal (re-serialize) path.
    ///
    /// **Pre-stream failures** (errors before the first chunk) trigger the same
    /// fallback cascade as non-streaming: the primary gets the configured retry loop,
    /// and each fallback target gets a single attempt.
    ///
    /// **Mid-stream failures** (errors after the first chunk is yielded) cannot be
    /// retried without buffering the entire response. They are surfaced as stream
    /// errors to the caller. Buffered mid-stream retry is explicitly deferred.
    pub(super) async fn dispatch_stream_with_fallback(
        &self,
        req: &ChatRequest,
        raw_body: Option<bytes::Bytes>,
    ) -> Result<(ChatCompletionStream, FallbackDecisionTrace), ProviderError> {
        let model = req.model.as_str();
        let mut attempts: Vec<FetchAttempt> = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();

        // --- Primary with retries (pre-stream only) ---
        let (primary_adapter, primary_guard) = match self.select_provider(model).await {
            Ok(pair) => pair,
            Err(e) => return Err(e),
        };
        let primary_name = primary_adapter.metadata().name.clone();
        visited.insert(format!("{}:{}", primary_name, model));

        let raw = raw_body.clone();
        let (primary_result, primary_attempts) = self
            .retry_loop(
                &primary_adapter,
                &primary_name,
                model,
                |a| {
                    let r = req.clone();
                    let raw = raw.clone();
                    async move {
                        if let Some(ref raw) = raw
                            && let Some(result) = a.try_forward_raw_stream(&r, raw).await
                        {
                            return result;
                        }
                        a.chat_completion_stream(&r).await
                    }
                },
                false, // streaming: GuardedStream calls on_response() at EOF; don't double-count
            )
            .await;
        attempts.extend(primary_attempts);

        let primary_err = match primary_result {
            Ok(stream) => {
                let trace = FallbackDecisionTrace {
                    source_provider: primary_name.clone(),
                    source_model: model.to_string(),
                    trigger: None,
                    matched_rule_index: None,
                    matched_rule_key: None,
                    attempts,
                    outcome: DecisionOutcome::Success,
                };
                return Ok((
                    self.wrap_stream_with_guard(stream, primary_guard, primary_name),
                    trace,
                ));
            }
            Err(e) => {
                drop(primary_guard);
                e
            }
        };

        let trigger = primary_err.to_trigger();

        // --- Find best matching fallback rule ---
        let matched_rule = self.best_matching_rule_indexed(&primary_name, model);

        if matched_rule.is_none() {
            let trace = FallbackDecisionTrace {
                source_provider: primary_name,
                source_model: model.to_string(),
                trigger: Some(trigger),
                matched_rule_index: None,
                matched_rule_key: None,
                attempts,
                outcome: DecisionOutcome::Exhausted,
            };
            log_dispatch_terminal(&trace);
            return Err(primary_err);
        }

        let (rule_index, rule) = matched_rule.unwrap();
        let rule_key = rule.key.clone();

        // --- Trigger gate ---
        if let Some(ref allowed_triggers) = rule.on
            && !allowed_triggers.contains(&trigger)
        {
            for target in &rule.targets {
                let target_name = target.provider_name().to_string();
                let target_model = target.model_override().unwrap_or(model);
                attempts.push(FetchAttempt {
                    provider: target_name,
                    model: target_model.to_string(),
                    is_retry: false,
                    trigger: Some(trigger.clone()),
                    attempted: false,
                    skip_reason: Some(SkipReason::TriggerNotAllowed),
                    error_class: None,
                    latency_ms: None,
                });
            }
            let trace = FallbackDecisionTrace {
                source_provider: primary_name,
                source_model: model.to_string(),
                trigger: Some(trigger.clone()),
                matched_rule_index: Some(rule_index),
                matched_rule_key: rule_key.clone(),
                attempts,
                outcome: DecisionOutcome::AbortedByPolicy,
            };
            log_dispatch_terminal(&trace);
            return Err(primary_err);
        }

        // --- Fallback cascade (single attempt per target, no retry) ---
        let fallback_targets = rule.targets.clone();
        let mut last_err: Option<ProviderError> = Some(primary_err);

        for target in &fallback_targets {
            let target_name = target.provider_name().to_string();
            let target_model = target.model_override().unwrap_or(model);
            let visit_key = format!("{}:{}", target_name, target_model);

            if visited.contains(&visit_key) {
                tracing::warn!(
                    provider = %target_name,
                    model = %target_model,
                    "streaming fallback cycle detected; skipping"
                );
                attempts.push(FetchAttempt {
                    provider: target_name,
                    model: target_model.to_string(),
                    is_retry: false,
                    trigger: Some(trigger.clone()),
                    attempted: false,
                    skip_reason: Some(SkipReason::DuplicateTarget),
                    error_class: None,
                    latency_ms: None,
                });
                continue;
            }
            visited.insert(visit_key);

            let fallback_adapter = match self.provider_by_name(&target_name) {
                Some(a) => a,
                None => {
                    tracing::warn!(
                        provider = %target_name,
                        "streaming fallback provider not found; skipping"
                    );
                    attempts.push(FetchAttempt {
                        provider: target_name,
                        model: target_model.to_string(),
                        is_retry: false,
                        trigger: Some(trigger.clone()),
                        attempted: false,
                        skip_reason: Some(SkipReason::ProviderNotFound),
                        error_class: None,
                        latency_ms: None,
                    });
                    continue;
                }
            };

            let candidates = self
                .health
                .candidates(
                    &[Arc::clone(&fallback_adapter)],
                    &self.routing_config.weights,
                    target_model,
                    &self.pricing_db,
                )
                .await;
            if candidates.is_empty() {
                tracing::debug!(
                    provider = %target_name,
                    model = %target_model,
                    "streaming fallback provider does not support resolved model or is in cooldown; skipping"
                );
                let skip_reason = if fallback_adapter
                    .metadata()
                    .supported_models
                    .iter()
                    .any(|m| m == "*" || m == target_model)
                {
                    SkipReason::InCooldown
                } else {
                    SkipReason::ModelUnsupported
                };
                attempts.push(FetchAttempt {
                    provider: target_name,
                    model: target_model.to_string(),
                    is_retry: false,
                    trigger: Some(trigger.clone()),
                    attempted: false,
                    skip_reason: Some(skip_reason),
                    error_class: None,
                    latency_ms: None,
                });
                continue;
            }

            let fallback_req = if target_model != model {
                req.with_model(target_model)
            } else {
                req.clone()
            };
            let guard = InFlightGuard::new(&self.health, &target_name);
            let start = Instant::now();

            // Skip the raw path when a fallback model override is active: the raw
            // bytes contain the client's original model name and would send the
            // wrong model upstream. Re-serialise via chat_completion_stream instead.
            let stream_result = if let Some(ref raw) = raw_body
                && target_model == model
            {
                match fallback_adapter
                    .try_forward_raw_stream(&fallback_req, raw)
                    .await
                {
                    Some(result) => result,
                    None => fallback_adapter.chat_completion_stream(&fallback_req).await,
                }
            } else {
                fallback_adapter.chat_completion_stream(&fallback_req).await
            };

            match stream_result {
                Ok(stream) => {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    attempts.push(FetchAttempt {
                        provider: target_name.clone(),
                        model: target_model.to_string(),
                        is_retry: false,
                        trigger: Some(trigger.clone()),
                        attempted: true,
                        skip_reason: None,
                        error_class: None,
                        latency_ms: Some(latency_ms),
                    });
                    let trace = FallbackDecisionTrace {
                        source_provider: primary_name,
                        source_model: model.to_string(),
                        trigger: Some(trigger),
                        matched_rule_index: Some(rule_index),
                        matched_rule_key: rule_key,
                        attempts,
                        outcome: DecisionOutcome::Success,
                    };
                    log_dispatch_terminal(&trace);
                    return Ok((
                        self.wrap_stream_with_guard(stream, guard, target_name),
                        trace,
                    ));
                }
                Err(e) => {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    if matches!(
                        &e,
                        ProviderError::RateLimited { .. }
                            | ProviderError::ProviderUnavailable(_)
                            | ProviderError::Timeout { .. }
                    ) {
                        self.health.on_rate_limit(&target_name).await;
                    }
                    attempts.push(FetchAttempt {
                        provider: target_name,
                        model: target_model.to_string(),
                        is_retry: false,
                        trigger: Some(trigger.clone()),
                        attempted: true,
                        skip_reason: None,
                        error_class: Some(e.error_class().to_owned()),
                        latency_ms: Some(latency_ms),
                    });
                    drop(guard);
                    last_err = Some(e);
                }
            }
        }

        let final_err = last_err.unwrap_or_else(|| {
            ProviderError::Internal("dispatch_stream_with_fallback: no providers tried".into())
        });
        let trace = FallbackDecisionTrace {
            source_provider: primary_name,
            source_model: model.to_string(),
            trigger: Some(trigger),
            matched_rule_index: Some(rule_index),
            matched_rule_key: rule_key,
            attempts,
            outcome: DecisionOutcome::Exhausted,
        };
        log_dispatch_terminal(&trace);
        Err(final_err)
    }
}
