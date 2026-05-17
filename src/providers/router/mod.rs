// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! ProviderRouter — composite ProviderAdapter that routes by model name using configurable strategies.
//!
//! Replaces first-match-wins with `RoutingStrategy`-based dispatch.
//! Supports `WeightedRandom`, `RateLimitAware`, and `LowestCost` strategies.
//! `ProviderHealthTracker` tracks 429-cooldown, EWMA latency, and in-flight counts.
//!
//! Adds retry loop (same-provider exponential backoff), fallback cascade
//! (flat rule-based), HALF-OPEN circuit-breaker, mid-stream health tracking,
//! and inter-chunk streaming deadline.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;

use crate::config::{FallbackRule, RetryConfig, RoutingConfig, SecurityConfig};
use crate::domain::chat::{ChatRequest, ChatResponse};
use crate::domain::embedding::{EmbeddingRequest, EmbeddingResponse};
use crate::domain::ports::{
    AttemptedMeta, ChatCompletionStream, EmbeddingCapabilities, HealthStatus, ProviderAdapter,
    ProviderAdapterExt, ProviderCandidate, ProviderError, ProviderKind, ProviderMetadata,
    RoutingContext, RoutingStrategy, StrategyError,
};
use crate::domain::pricing::PricingDb;
use crate::providers::health::{InFlightGuard, ProviderHealthTracker};
use crate::providers::router::fallback_trace::{
    DecisionOutcome, FallbackDecisionTrace, trigger_header_value,
};

mod fallback;
pub(crate) mod fallback_trace;
mod retry;
mod streaming;
#[cfg(test)]
mod tests;

/// Composite ProviderAdapter that routes requests using a configurable `RoutingStrategy`.
///
/// On each request:
/// 1. `ProviderHealthTracker::candidates()` builds a per-request snapshot.
/// 2. The strategy selects a candidate.
/// 3. `InFlightGuard` increments the in-flight counter (decrements on drop/panic).
/// 4. The candidate's adapter dispatches the request.
/// 5. Retry loop runs on transient errors .
/// 6. Fallback cascade runs when retries are exhausted .
/// 7. `on_rate_limit()` / `on_response()` keep health state updated throughout.
pub struct ProviderRouter {
    providers: Vec<Arc<dyn ProviderAdapter>>,
    aggregated_metadata: ProviderMetadata,
    strategy: Arc<dyn RoutingStrategy>,
    health: Arc<ProviderHealthTracker>,
    pricing_db: Arc<std::sync::RwLock<PricingDb>>,
    routing_config: RoutingConfig,
    retry: RetryConfig,
    fallbacks: Vec<FallbackRule>,
    security: SecurityConfig,
}

impl ProviderRouter {
    /// Creates a router with the given providers and routing strategy.
    ///
    /// Uses default retry/fallback/security config. `new_with_resilience` is preferred
    /// when building from `GatewayConfig`; this method is kept for tests.
    #[must_use]
    pub fn new(
        providers: Vec<Arc<dyn ProviderAdapter>>,
        strategy: Arc<dyn RoutingStrategy>,
        health: Arc<ProviderHealthTracker>,
        pricing_db: Arc<std::sync::RwLock<PricingDb>>,
        routing_config: RoutingConfig,
    ) -> Self {
        Self::new_with_resilience(
            providers,
            strategy,
            health,
            pricing_db,
            routing_config,
            RetryConfig::default(),
            vec![],
            SecurityConfig::default(),
        )
    }

    /// Creates a router with full resilience config (retry, fallbacks, security).
    ///
    /// Called by `build_from_config` to wire the full operator config.
    ///
    /// TODO: when mid-stream fallback adds `buffer_limit` and `commitment_point`,
    /// replace the individual parameters with a `ResilienceConfig { retry, fallbacks,
    /// security, streaming }` grouping struct, which will also eliminate this allow.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_resilience(
        providers: Vec<Arc<dyn ProviderAdapter>>,
        strategy: Arc<dyn RoutingStrategy>,
        health: Arc<ProviderHealthTracker>,
        pricing_db: Arc<std::sync::RwLock<PricingDb>>,
        routing_config: RoutingConfig,
        retry: RetryConfig,
        fallbacks: Vec<FallbackRule>,
        security: SecurityConfig,
    ) -> Self {
        let aggregated_metadata = Self::aggregate_metadata(&providers);
        Self {
            providers,
            aggregated_metadata,
            strategy,
            health,
            pricing_db,
            routing_config,
            retry,
            fallbacks,
            security,
        }
    }

    /// Returns `true` when `security.expose_provider_names` is enabled.
    pub fn expose_provider_names(&self) -> bool {
        self.security.expose_provider_names
    }

    /// Merges supported_models from all providers, deduplicated.
    fn aggregate_metadata(providers: &[Arc<dyn ProviderAdapter>]) -> ProviderMetadata {
        let mut supported_models: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for p in providers {
            let meta = p.metadata();
            for m in &meta.supported_models {
                if seen.insert(m.clone()) {
                    supported_models.push(m.clone());
                }
            }
        }

        let embedding_capabilities = {
            let caps: Vec<&EmbeddingCapabilities> = providers
                .iter()
                .filter_map(|p| p.metadata().embedding_capabilities.as_ref())
                .collect();
            if caps.is_empty() {
                None
            } else {
                let mut dims: Vec<u32> = caps
                    .iter()
                    .flat_map(|c| c.dimensions.iter().copied())
                    .collect();
                dims.sort_unstable();
                dims.dedup();
                Some(EmbeddingCapabilities {
                    dimensions: dims,
                    max_input_tokens: caps.iter().map(|c| c.max_input_tokens).min().unwrap_or(0),
                    supports_batch: caps.iter().any(|c| c.supports_batch),
                })
            }
        };

        ProviderMetadata {
            name: "router".to_string(),
            supported_models,
            supports_streaming: providers.iter().any(|p| p.metadata().supports_streaming),
            supports_tools: providers.iter().any(|p| p.metadata().supports_tools),
            supports_vision: providers.iter().any(|p| p.metadata().supports_vision),
            supports_embeddings: providers.iter().any(|p| p.metadata().supports_embeddings),
            supports_thinking: providers.iter().any(|p| p.metadata().supports_thinking),
            kind: ProviderKind::Primary,
            embedding_capabilities,
        }
    }

    /// Returns the list of configured providers (for models list endpoint, health checks).
    #[must_use]
    pub fn providers(&self) -> &[Arc<dyn ProviderAdapter>] {
        &self.providers
    }

    /// Runs the full candidate-selection pipeline for a given model.
    ///
    /// Returns the selected adapter and its live `InFlightGuard` (counter incremented).
    async fn select_provider(
        &self,
        model: &str,
    ) -> Result<(Arc<dyn ProviderAdapter>, InFlightGuard), ProviderError> {
        let candidates = self
            .health
            .candidates(
                &self.providers,
                &self.routing_config.weights,
                model,
                &self.pricing_db,
            )
            .await;

        if self.providers.is_empty() {
            return Err(ProviderError::ProviderUnavailable(
                "no providers configured; add at least one provider under `providers:` in your config".into(),
            ));
        }
        if candidates.is_empty() {
            return Err(ProviderError::UnknownModel(model.to_string()));
        }

        let candidate_refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model };
        let selected = self
            .strategy
            .select(&candidate_refs, &ctx)
            .map_err(Self::strategy_error_to_provider_error)?;

        let adapter = Arc::clone(&selected.adapter);
        let guard = InFlightGuard::new(&self.health, &selected.name);
        Ok((adapter, guard))
    }

    fn strategy_error_to_provider_error(err: StrategyError) -> ProviderError {
        match err {
            StrategyError::AllProvidersRateLimited { retry_after } => {
                ProviderError::AllProvidersRateLimited { retry_after }
            }
            StrategyError::NoEligibleCandidates => ProviderError::Internal(
                "no eligible provider candidates (all weights zero?)".into(),
            ),
        }
    }
}

#[async_trait]
impl ProviderAdapter for ProviderRouter {
    async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        ProviderAdapterExt::chat_completion_with_trace(self, req)
            .await
            .map(|(response, _)| response)
    }

    async fn chat_completion_stream(
        &self,
        req: &ChatRequest,
    ) -> Result<ChatCompletionStream, ProviderError> {
        ProviderAdapterExt::chat_completion_stream_with_trace(self, req)
            .await
            .map(|(stream, _)| stream)
    }

    async fn embeddings(&self, req: &EmbeddingRequest) -> Result<EmbeddingResponse, ProviderError> {
        ProviderAdapterExt::embeddings_with_trace(self, req)
            .await
            .map(|(resp, _)| resp)
    }

    fn metadata(&self) -> &ProviderMetadata {
        &self.aggregated_metadata
    }

    fn as_providers_slice(&self) -> Option<&[Arc<dyn ProviderAdapter>]> {
        Some(self.providers())
    }

    async fn health_check(&self) -> HealthStatus {
        let mut healthy = 0;
        let mut degraded = 0;
        let mut unknown = 0;
        for p in &self.providers {
            match p.health_check().await {
                HealthStatus::Healthy => healthy += 1,
                HealthStatus::Degraded => degraded += 1,
                HealthStatus::Unknown => unknown += 1,
                HealthStatus::Unhealthy => {}
            }
        }
        if healthy > 0 {
            HealthStatus::Healthy
        } else if unknown > 0 {
            // No verified-healthy providers, but some are unprobed — still routable.
            // Callers should treat Unknown as "available, not verified."
            HealthStatus::Unknown
        } else if degraded > 0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        }
    }
}

/// Extension trait implementation: full fallback + retry dispatch with routing metadata.
///
/// `ProviderAdapterExt`'s blanket impl handles leaf adapters; `ProviderRouter` overrides
/// both methods to run the complete dispatch pipeline and populate `AttemptedMeta` with
/// every `(provider, model)` pair tried (primary first, then fallback targets in order).
#[async_trait]
impl ProviderAdapterExt for ProviderRouter {
    async fn chat_completion_with_trace(
        &self,
        req: &ChatRequest,
    ) -> Result<(ChatResponse, AttemptedMeta), ProviderError> {
        let (result, trace) = self
            .dispatch_with_fallback(req, |adapter, r| async move {
                adapter.chat_completion(&r).await
            })
            .await;
        result.map(|response| (response, meta_from_trace(&trace)))
    }

    /// Streaming dispatch with pre-stream fallback cascade.
    ///
    /// Delegates to [`ProviderRouter::dispatch_stream_with_fallback`], which mirrors the
    /// non-streaming fallback behavior for errors that occur before the first chunk is
    /// yielded. Mid-stream failures are surfaced as stream errors; buffered retry is
    /// explicitly deferred.
    async fn chat_completion_stream_with_trace(
        &self,
        req: &ChatRequest,
    ) -> Result<(ChatCompletionStream, AttemptedMeta), ProviderError> {
        let (stream, trace) = self.dispatch_stream_with_fallback(req, None).await?;
        Ok((stream, meta_from_trace(&trace)))
    }

    /// Non-streaming dispatch with raw-bytes fast path through the full routing pipeline.
    ///
    /// Each adapter attempt in `dispatch_with_fallback` tries `try_forward_raw` first;
    /// on `None`, falls back to `chat_completion`. Translation adapters always return
    /// `None` and are unaffected. Only `OpenAICompatAdapter` takes the raw path.
    async fn chat_completion_raw_with_trace(
        &self,
        req: &ChatRequest,
        raw_body: &bytes::Bytes,
    ) -> Result<(ChatResponse, AttemptedMeta), ProviderError> {
        let raw = raw_body.clone();
        let original_model = req.model.clone();
        let (result, trace) = self
            .dispatch_with_fallback(req, move |adapter, r| {
                let raw = raw.clone();
                let orig = original_model.clone();
                async move {
                    // Skip the raw path when a fallback model override is active: the raw
                    // bytes contain the client's original model name and would send the
                    // wrong model upstream. Re-serialise via chat_completion instead.
                    if r.model == orig
                        && let Some(result) = adapter.try_forward_raw(&r, &raw).await
                    {
                        return result;
                    }
                    adapter.chat_completion(&r).await
                }
            })
            .await;
        result.map(|response| (response, meta_from_trace(&trace)))
    }

    /// Streaming dispatch with raw-bytes fast path through the full routing pipeline.
    async fn chat_completion_stream_raw_with_trace(
        &self,
        req: &ChatRequest,
        raw_body: &bytes::Bytes,
    ) -> Result<(ChatCompletionStream, AttemptedMeta), ProviderError> {
        let (stream, trace) = self
            .dispatch_stream_with_fallback(req, Some(raw_body.clone()))
            .await?;
        Ok((stream, meta_from_trace(&trace)))
    }

    async fn embeddings_with_trace(
        &self,
        req: &EmbeddingRequest,
    ) -> Result<(EmbeddingResponse, AttemptedMeta), ProviderError> {
        let (result, trace) = self
            .dispatch_with_fallback(
                req,
                |adapter, r| async move { adapter.embeddings(&r).await },
            )
            .await;
        result.map(|response| (response, meta_from_trace(&trace)))
    }
}

/// Builds `AttemptedMeta` from a `FallbackDecisionTrace`.
///
/// Providers/models are deduplicated by consecutive-provider grouping so that retry attempts
/// on the same provider don't inflate the `X-Oxigate-Attempted-*` headers. Each unique
/// provider role (primary, fallback target 1, fallback target 2, …) appears exactly once.
fn meta_from_trace(trace: &FallbackDecisionTrace) -> AttemptedMeta {
    // Deduplicate consecutive retries: keep the first entry per consecutive (provider, model)
    // group so that retry attempts on the same provider+model don't inflate the
    // X-Oxigate-Attempted-* headers. Each unique provider role (primary, fallback target 1,
    // fallback target 2, …) appears exactly once.
    let dispatched: Vec<_> = trace.attempts.iter().filter(|a| a.attempted).collect();
    let mut providers: Vec<String> = Vec::new();
    let mut models: Vec<String> = Vec::new();
    let mut last_key: Option<(&str, &str)> = None;
    for attempt in &dispatched {
        let key = (attempt.provider.as_str(), attempt.model.as_str());
        if last_key != Some(key) {
            providers.push(attempt.provider.clone());
            models.push(attempt.model.clone());
            last_key = Some(key);
        }
    }

    // X-Fallback-Reason: present only when ≥1 non-primary (fallback) target was dispatched.
    // A fallback attempt is one that was dispatched, is not a retry, and has a trigger set
    // (trigger is None only for the initial primary attempt).
    let fallback_dispatched = trace
        .attempts
        .iter()
        .any(|a| a.attempted && !a.is_retry && a.trigger.is_some());

    let fallback_trigger = if fallback_dispatched {
        trace
            .trigger
            .as_ref()
            .map(|t| trigger_header_value(t).to_owned())
    } else {
        None
    };

    // Outcome AbortedByPolicy: fallback_dispatched stays false even if trigger is set.
    let fallback_dispatched =
        fallback_dispatched && trace.outcome != DecisionOutcome::AbortedByPolicy;

    AttemptedMeta {
        providers,
        models,
        fallback_trigger,
        fallback_dispatched,
    }
}
