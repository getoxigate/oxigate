// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Generic fallback cascade dispatch for ProviderRouter .

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use crate::config::{FallbackRule, FallbackTarget};
use crate::domain::chat::ChatRequest;
use crate::domain::embedding::EmbeddingRequest;
use crate::domain::ports::{ProviderAdapter, ProviderError};
use crate::observability::metrics;
use crate::providers::health::InFlightGuard;
use crate::providers::router::fallback_trace::{
    DecisionOutcome, FallbackDecisionTrace, FetchAttempt, SkipReason, trigger_header_value,
};

use super::ProviderRouter;

/// Private trait that allows `dispatch_with_fallback` to be generic over both
/// `ChatRequest` and `EmbeddingRequest` without exposing a public abstraction.
pub(super) trait DispatchableRequest: Clone + Send + Sync {
    fn model(&self) -> &str;
    fn with_model(&self, model: &str) -> Self;
}

impl DispatchableRequest for ChatRequest {
    fn model(&self) -> &str {
        &self.model
    }
    fn with_model(&self, model: &str) -> Self {
        ChatRequest::with_model(self, model)
    }
}

impl DispatchableRequest for EmbeddingRequest {
    fn model(&self) -> &str {
        &self.model
    }
    fn with_model(&self, model: &str) -> Self {
        EmbeddingRequest::with_model(self, model)
    }
}

impl ProviderRouter {
    /// Dispatches to the primary provider with retry, then cascades to fallback targets.
    ///
    /// Returns `(result, FallbackDecisionTrace)`. The trace records every attempt (retries +
    /// fallback targets) for logging, metrics, and `X-Fallback-Reason` header injection.
    ///
    /// Flat cascade: only the best-matching rule for the primary `(provider, model)` fires.
    pub(super) async fn dispatch_with_fallback<Req, F, Fut, R>(
        &self,
        req: &Req,
        dispatch_fn: F,
    ) -> (Result<R, ProviderError>, FallbackDecisionTrace)
    where
        Req: DispatchableRequest,
        F: Fn(Arc<dyn ProviderAdapter>, Req) -> Fut,
        Fut: std::future::Future<Output = Result<R, ProviderError>>,
    {
        let model = req.model();
        let mut attempts: Vec<FetchAttempt> = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();

        // --- Primary with retries ---
        let (primary_adapter, primary_guard) = match self.select_provider(model).await {
            Ok(pair) => pair,
            Err(e) => {
                let trace = FallbackDecisionTrace {
                    source_provider: String::new(),
                    source_model: model.to_string(),
                    trigger: None,
                    matched_rule_index: None,
                    matched_rule_key: None,
                    attempts,
                    outcome: DecisionOutcome::Exhausted,
                };
                return (Err(e), trace);
            }
        };
        let primary_name = primary_adapter.metadata().name.clone();
        visited.insert(format!("{}:{}", primary_name, model));

        // Primary attempt — attempt 0 (is_retry = false), then retries (is_retry = true).
        // retry_loop returns one FetchAttempt per dispatch iteration.
        let (primary_result, primary_attempts) = self
            .retry_loop(
                &primary_adapter,
                &primary_name,
                model,
                |a| {
                    let r = req.clone();
                    dispatch_fn(a, r)
                },
                true, // non-streaming: success accounting belongs here
            )
            .await;
        drop(primary_guard);
        attempts.extend(primary_attempts);

        if primary_result.is_ok() {
            let trace = FallbackDecisionTrace {
                source_provider: primary_name,
                source_model: model.to_string(),
                trigger: None,
                matched_rule_index: None,
                matched_rule_key: None,
                attempts,
                outcome: DecisionOutcome::Success,
            };
            return (primary_result, trace);
        }

        // Primary failed — classify trigger.
        // Safety: we checked `is_ok()` above and it was false.
        let Err(primary_err) = primary_result else {
            unreachable!()
        };
        let trigger = primary_err.to_trigger();

        // --- Find best matching fallback rule ---
        let matched_rule = self.best_matching_rule_indexed(&primary_name, model);

        if matched_rule.is_none() {
            // No rule — return primary error.
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
            return (Err(primary_err), trace);
        }

        let (rule_index, rule) = matched_rule.unwrap();
        let rule_key = rule.key.clone();

        // --- Trigger gate: check rule.on ---
        if let Some(ref allowed_triggers) = rule.on
            && !allowed_triggers.contains(&trigger)
        {
            // Trigger not allowed — record all targets as skipped.
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
                matched_rule_key: rule_key,
                attempts,
                outcome: DecisionOutcome::AbortedByPolicy,
            };
            log_dispatch_terminal(&trace);
            return (Err(primary_err), trace);
        }

        // --- Fallback cascade ---
        let fallback_targets: Vec<FallbackTarget> = rule.targets.clone();
        let mut last_err: Option<ProviderError> = None;

        for target in &fallback_targets {
            let target_name = target.provider_name().to_string();
            let target_model = target.model_override().unwrap_or(model);
            let visit_key = format!("{}:{}", target_name, target_model);

            if visited.contains(&visit_key) {
                tracing::warn!(
                    provider = %target_name,
                    model = %target_model,
                    "fallback cycle detected at runtime (same provider+model already attempted); skipping"
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
                        "fallback target provider not found in router; skipping"
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

            // Check model support via candidates() — returns empty if model not supported or
            // provider is in cooldown.
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
                    "fallback provider does not support resolved model or is in cooldown; skipping"
                );
                // Distinguish InCooldown vs ModelUnsupported based on metadata.
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
            let result = dispatch_fn(Arc::clone(&fallback_adapter), fallback_req).await;
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            drop(guard);

            match &result {
                Err(ProviderError::RateLimited { .. })
                | Err(ProviderError::ProviderUnavailable(_))
                | Err(ProviderError::Timeout { .. }) => {
                    self.health.on_rate_limit(&target_name).await;
                }
                Ok(_) => {
                    self.health.on_response(&target_name, latency_ms).await;
                }
                Err(_) => {}
            }

            let attempt_error_class = result.as_ref().err().map(|e| e.error_class().to_owned());
            let succeeded = result.is_ok();
            attempts.push(FetchAttempt {
                provider: target_name.clone(),
                model: target_model.to_string(),
                is_retry: false,
                trigger: Some(trigger.clone()),
                attempted: true,
                skip_reason: None,
                error_class: attempt_error_class,
                latency_ms: Some(latency_ms),
            });

            if succeeded {
                let trace = FallbackDecisionTrace {
                    source_provider: primary_name,
                    source_model: model.to_string(),
                    trigger: Some(trigger.clone()),
                    matched_rule_index: Some(rule_index),
                    matched_rule_key: rule_key,
                    attempts,
                    outcome: DecisionOutcome::Success,
                };
                log_dispatch_terminal(&trace);
                return (result, trace);
            }
            if let Err(e) = result {
                last_err = Some(e);
            }
        }

        // All attempts exhausted.
        let final_err = last_err.unwrap_or(primary_err);
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
        (Err(final_err), trace)
    }

    /// Finds the best-matching fallback rule for `(provider, model)`.
    ///
    /// Returns `Some((index, &rule))` where `index` is the 0-based position in
    /// `self.fallbacks`. Returns `None` when no rule matches.
    ///
    /// Match precedence (higher score wins; first rule wins ties):
    /// - Score 2: provider exact match AND model pattern match
    /// - Score 1: provider exact match only (no model constraint on rule)
    /// - Score 1: model pattern match only (no provider constraint on rule)
    /// - Score 0: no match
    pub(super) fn best_matching_rule_indexed<'a>(
        &'a self,
        provider: &str,
        model: &str,
    ) -> Option<(usize, &'a FallbackRule)> {
        let mut best_score: u8 = 0;
        let mut best: Option<(usize, &FallbackRule)> = None;
        for (i, rule) in self.fallbacks.iter().enumerate() {
            let provider_match = rule.provider.as_deref() == Some(provider);
            let model_match = rule.model.as_deref().is_some_and(|pat| {
                pat.strip_suffix('*')
                    .map_or(model == pat, |prefix| model.starts_with(prefix))
            });
            let score = match (rule.provider.is_some(), rule.model.is_some()) {
                (true, true) => {
                    if provider_match && model_match {
                        2
                    } else {
                        0
                    }
                }
                (true, false) => u8::from(provider_match),
                (false, true) => u8::from(model_match),
                (false, false) => 0,
            };
            if score > best_score {
                best_score = score;
                best = Some((i, rule));
            }
        }
        best
    }

    /// Looks up a provider adapter by name.
    pub(super) fn provider_by_name(&self, name: &str) -> Option<Arc<dyn ProviderAdapter>> {
        self.providers
            .iter()
            .find(|p| p.metadata().name == name)
            .map(Arc::clone)
    }
}

/// Emits metrics and a structured tracing event at dispatch terminal.
/// Level: DEBUG on success, INFO on terminal failure.
pub(super) fn log_dispatch_terminal(trace: &FallbackDecisionTrace) {
    // --- Metrics ---
    let attempt_count_dispatched = trace.attempts.iter().filter(|a| a.attempted).count();
    let skipped_count = trace.attempts.iter().filter(|a| !a.attempted).count();
    let total_secs: f64 = trace
        .attempts
        .iter()
        .filter_map(|a| a.latency_ms)
        .sum::<f64>()
        / 1000.0;
    let trigger_str = trace.trigger.as_ref().map(trigger_header_value);
    metrics::record_dispatch(
        trigger_str,
        attempt_count_dispatched,
        skipped_count,
        total_secs,
        &trace.source_provider,
    );
    // Record individual skip reasons.
    for a in trace.attempts.iter().filter(|a| !a.attempted) {
        if let Some(ref reason) = a.skip_reason {
            metrics::record_skip(&reason.to_string());
        }
    }
    let attempt_count = attempt_count_dispatched;
    let compact: Vec<String> = trace
        .attempts
        .iter()
        .map(|a| {
            if a.attempted {
                format!(
                    "{}:{}:retry={}:{}",
                    a.provider,
                    a.model,
                    a.is_retry,
                    a.error_class.as_deref().unwrap_or("ok")
                )
            } else {
                format!(
                    "{}:{}:skip={}",
                    a.provider,
                    a.model,
                    a.skip_reason
                        .as_ref()
                        .map(|r| r.to_string())
                        .unwrap_or_default()
                )
            }
        })
        .collect();
    let trigger_display = trigger_str.unwrap_or("none");
    let rule_key = trace.matched_rule_key.as_deref().unwrap_or("");
    let total_latency_ms = total_secs * 1000.0;

    match trace.outcome {
        DecisionOutcome::Success => {
            tracing::debug!(
                source_provider = %trace.source_provider,
                source_model = %trace.source_model,
                trigger = %trigger_display,
                matched_rule_index = ?trace.matched_rule_index,
                matched_rule_key = %rule_key,
                attempt_count,
                skipped_count,
                total_latency_ms,
                outcome = "success",
                attempts = %compact.join("|"),
                "fallback dispatch terminal"
            );
        }
        _ => {
            tracing::info!(
                source_provider = %trace.source_provider,
                source_model = %trace.source_model,
                trigger = %trigger_display,
                matched_rule_index = ?trace.matched_rule_index,
                matched_rule_key = %rule_key,
                attempt_count,
                skipped_count,
                total_latency_ms,
                outcome = %trace.outcome,
                attempts = %compact.join("|"),
                "fallback dispatch terminal"
            );
        }
    }
}
