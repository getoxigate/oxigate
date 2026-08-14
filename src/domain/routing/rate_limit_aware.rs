// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! RateLimitAware routing strategy — skips providers in 429 cooldown.
//!
//! This is the ONLY strategy that hard-excludes cooling providers. All other
//! strategies receive the full unfiltered candidate slice .
//!
//! `AllProvidersRateLimited` is only emitted by this strategy.

use crate::domain::ports::{ProviderCandidate, RoutingContext, RoutingStrategy, StrategyError};
use crate::domain::routing::WeightedRandom;

/// Filters out cooling providers, then delegates to `WeightedRandom` for selection.
///
/// Returns `AllProvidersRateLimited { retry_after }` when all candidates are cooling,
/// where `retry_after` is the minimum remaining cooldown across all cooling providers —
/// i.e. the earliest a provider will become available again.
pub struct RateLimitAware;

impl RoutingStrategy for RateLimitAware {
    fn select<'s>(
        &self,
        candidates: &[&'s ProviderCandidate],
        ctx: &RoutingContext<'_>,
    ) -> Result<&'s ProviderCandidate, StrategyError> {
        let available: Vec<&'s ProviderCandidate> = candidates
            .iter()
            .copied()
            .filter(|c| !c.is_cooling_down)
            .collect();

        if available.is_empty() {
            // Compute the minimum remaining cooldown across all cooling providers.
            // The minimum represents the earliest point at which at least one provider
            // becomes available again — the correct value for a Retry-After header.
            // Floor at 1 so clients always get a positive wait value.
            let retry_after = candidates
                .iter()
                .filter(|c| c.is_cooling_down)
                .map(|c| c.cooldown_remaining_secs)
                .min()
                .unwrap_or(0)
                .max(1);
            return Err(StrategyError::AllProvidersRateLimited { retry_after });
        }

        // Delegate to WeightedRandom — available already contains only &ProviderCandidate
        // references; no cloning of the owned values is required.
        WeightedRandom.select(&available, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;

    use crate::domain::chat::{ChatRequest, ChatResponse};
    use crate::domain::ports::{
        HealthStatus, NanoUsd, ProviderAdapter, ProviderError, ProviderMetadata, RoutingContext,
        StrategyError,
    };

    use super::*;

    fn make_candidate(name: &str, cooling: bool) -> ProviderCandidate {
        make_candidate_with_remaining(name, cooling, if cooling { 60 } else { 0 })
    }

    fn make_candidate_with_remaining(
        name: &str,
        cooling: bool,
        remaining_secs: u64,
    ) -> ProviderCandidate {
        struct Stub(ProviderMetadata);
        #[async_trait]
        impl ProviderAdapter for Stub {
            async fn chat_completion(
                &self,
                _req: &ChatRequest,
            ) -> Result<ChatResponse, ProviderError> {
                Err(ProviderError::NotImplemented)
            }
            fn metadata(&self) -> &ProviderMetadata {
                &self.0
            }
            async fn health_check(&self) -> HealthStatus {
                HealthStatus::Healthy
            }
        }
        let meta = ProviderMetadata {
            name: name.to_string(),
            supported_models: vec!["*".to_string()],
            supports_streaming: false,
            supports_tools: false,
            supports_vision: false,
            supports_embeddings: false,
            supports_thinking: false,
            kind: Default::default(),
            ..Default::default()
        };
        ProviderCandidate {
            name: name.to_string(),
            adapter: Arc::new(Stub(meta)),
            weight: 1.0,
            in_flight: 0,
            latency_ewma_ms: 0.0,
            is_cooling_down: cooling,
            cooldown_remaining_secs: remaining_secs,
            cost_per_million_tokens: NanoUsd::MAX,
        }
    }

    #[test]
    fn test_rate_limit_aware_skips_cooling_providers() {
        let candidates = vec![
            make_candidate("cooling_provider", true),
            make_candidate("healthy_provider", false),
        ];
        let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model: "gpt-4o" };
        let strategy = RateLimitAware;
        for _ in 0..50 {
            let result = strategy.select(&refs, &ctx).unwrap();
            assert_eq!(result.name, "healthy_provider");
        }
    }

    #[test]
    fn test_rate_limit_aware_all_cooling_returns_min_retry_after() {
        // a has 30s remaining, b has 10s remaining — retry_after must be 10 (minimum).
        let candidates = vec![
            make_candidate_with_remaining("a", true, 30),
            make_candidate_with_remaining("b", true, 10),
        ];
        let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model: "gpt-4o" };
        let strategy = RateLimitAware;
        let err = strategy.select(&refs, &ctx).unwrap_err();
        match err {
            StrategyError::AllProvidersRateLimited { retry_after } => {
                assert_eq!(
                    retry_after, 10,
                    "retry_after must be the minimum remaining cooldown"
                );
            }
            other => panic!("expected AllProvidersRateLimited, got {other:?}"),
        }
    }

    #[test]
    fn test_rate_limit_aware_retry_after_floors_at_1() {
        // cooldown_remaining_secs = 0 (just expired but Redis key still present) → floor to 1.
        let candidates = vec![make_candidate_with_remaining("a", true, 0)];
        let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model: "gpt-4o" };
        let err = RateLimitAware.select(&refs, &ctx).unwrap_err();
        match err {
            StrategyError::AllProvidersRateLimited { retry_after } => {
                assert!(retry_after >= 1, "retry_after must be at least 1 second");
            }
            other => panic!("expected AllProvidersRateLimited, got {other:?}"),
        }
    }

    #[test]
    fn test_rate_limit_aware_partial_cooling_selects_healthy() {
        let candidates = vec![
            make_candidate("cooling", true),
            make_candidate("healthy1", false),
            make_candidate("healthy2", false),
        ];
        let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model: "gpt-4o" };
        let strategy = RateLimitAware;
        for _ in 0..50 {
            let result = strategy.select(&refs, &ctx).unwrap();
            assert_ne!(
                result.name, "cooling",
                "cooling provider must never be selected"
            );
        }
    }

    #[test]
    fn test_rate_limit_aware_retry_after_tie_uses_common_value() {
        // All providers have the same remaining cooldown — retry_after must be that value.
        let candidates = vec![
            make_candidate_with_remaining("a", true, 30),
            make_candidate_with_remaining("b", true, 30),
            make_candidate_with_remaining("c", true, 30),
        ];
        let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model: "m" };
        let err = RateLimitAware.select(&refs, &ctx).unwrap_err();
        match err {
            StrategyError::AllProvidersRateLimited { retry_after } => {
                assert_eq!(
                    retry_after, 30,
                    "retry_after must be 30 when all have 30s remaining"
                );
            }
            other => panic!("expected AllProvidersRateLimited, got {other:?}"),
        }
    }
}
