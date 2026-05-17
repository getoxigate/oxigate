// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! LowestCost routing strategy — selects provider with cheapest input cost.
//!
//! When all candidates have `NanoUsd::MAX` (unknown pricing), falls back to
//! `WeightedRandom` to avoid penalising new or misconfigured providers.

use crate::domain::ports::{
    NanoUsd, ProviderCandidate, RoutingContext, RoutingStrategy, StrategyError,
};
use crate::domain::routing::WeightedRandom;

/// Selects the provider with the lowest `cost_per_million_tokens`.
///
/// Ties are broken by stable position in the slice (first occurrence wins).
/// Providers with `NanoUsd::MAX` (unknown cost) are excluded unless ALL candidates
/// have unknown cost, in which case `WeightedRandom` is used as fallback.
///
/// **Pricing granularity**: costs are keyed by model name (via `PricingDb`), not by
/// provider. All providers serving the same model name receive identical cost values.
/// `LowestCost` therefore only meaningfully differentiates when providers serve
/// *different* model names (the common case for OpenAI/Anthropic/Gemini). Two
/// OpenAI-compatible backends serving the same model will see equal costs and
/// fall back to `WeightedRandom`. Per-provider pricing overrides are deferred to.
pub struct LowestCost;

impl RoutingStrategy for LowestCost {
    fn select<'s>(
        &self,
        candidates: &[&'s ProviderCandidate],
        ctx: &RoutingContext<'_>,
    ) -> Result<&'s ProviderCandidate, StrategyError> {
        // Respect weight == 0.0 (provider explicitly disabled). Zero-weight providers
        // are excluded by WeightedRandom; LowestCost must apply the same rule so that
        // a disabled provider never receives traffic regardless of its cost.
        let eligible: Vec<&'s ProviderCandidate> = candidates
            .iter()
            .copied()
            .filter(|c| c.weight > 0.0)
            .collect();

        if eligible.is_empty() {
            return Err(StrategyError::NoEligibleCandidates);
        }

        // Check if any eligible candidate has known pricing (cost != MAX).
        let has_known = eligible
            .iter()
            .any(|c| c.cost_per_million_tokens != NanoUsd::MAX);

        if !has_known {
            // All unknown — fall back to weighted random among eligible candidates.
            return WeightedRandom.select(&eligible, ctx);
        }

        // Select the lowest cost among eligible candidates with known pricing.
        // Exclude MAX entries (stable order → first occurrence wins on tie).
        eligible
            .iter()
            .copied()
            .filter(|c| c.cost_per_million_tokens != NanoUsd::MAX)
            .min_by_key(|c| c.cost_per_million_tokens)
            .ok_or(StrategyError::NoEligibleCandidates)
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

    fn make_candidate(name: &str, cost: NanoUsd) -> ProviderCandidate {
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
            supported_models: vec!["gpt-4o".to_string()],
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
            is_cooling_down: false,
            cooldown_remaining_secs: 0,
            cost_per_million_tokens: cost,
        }
    }

    #[test]
    fn test_lowest_cost_selects_cheapest() {
        let candidates = vec![
            make_candidate("expensive", NanoUsd::from_f64_usd(10.0)),
            make_candidate("cheap", NanoUsd::from_f64_usd(1.0)),
            make_candidate("medium", NanoUsd::from_f64_usd(5.0)),
        ];
        let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model: "gpt-4o" };
        let result = LowestCost.select(&refs, &ctx).unwrap();
        assert_eq!(result.name, "cheap");
    }

    #[test]
    fn test_lowest_cost_tiebreak_by_stable_order() {
        // Two candidates with equal cost — first in slice wins.
        let candidates = vec![
            make_candidate("first", NanoUsd::from_f64_usd(1.0)),
            make_candidate("second", NanoUsd::from_f64_usd(1.0)),
        ];
        let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model: "gpt-4o" };
        let result = LowestCost.select(&refs, &ctx).unwrap();
        assert_eq!(
            result.name, "first",
            "first in slice must win on equal cost"
        );
    }

    #[test]
    fn test_lowest_cost_all_unknown_falls_back_to_weighted_random() {
        // All candidates have MAX cost — must not error; should select one.
        let candidates = vec![
            make_candidate("a", NanoUsd::MAX),
            make_candidate("b", NanoUsd::MAX),
        ];
        let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model: "gpt-4o" };
        let result = LowestCost.select(&refs, &ctx);
        assert!(
            result.is_ok(),
            "all-unknown should fall back to WeightedRandom, not error"
        );
    }

    #[test]
    fn test_lowest_cost_excludes_unknown_when_known_exists() {
        // Mix: one known (cheap), one unknown (MAX). Should pick the known one.
        let candidates = vec![
            make_candidate("unknown", NanoUsd::MAX),
            make_candidate("known", NanoUsd::from_f64_usd(2.0)),
        ];
        let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model: "gpt-4o" };
        let result = LowestCost.select(&refs, &ctx).unwrap();
        assert_eq!(result.name, "known");
    }

    #[test]
    fn test_lowest_cost_no_candidates_returns_error() {
        let candidates: Vec<&ProviderCandidate> = vec![];
        let ctx = RoutingContext { model: "gpt-4o" };
        let err = LowestCost.select(&candidates, &ctx).unwrap_err();
        assert!(matches!(err, StrategyError::NoEligibleCandidates));
    }

    #[test]
    fn test_lowest_cost_zero_weight_excluded() {
        // A zero-weight provider is explicitly disabled and must never receive traffic,
        // even if it has the lowest (or only known) cost.
        let mut disabled = make_candidate("disabled", NanoUsd::from_f64_usd(0.01));
        disabled.weight = 0.0;
        let enabled = make_candidate("enabled", NanoUsd::from_f64_usd(10.0));
        let candidates = vec![disabled, enabled];
        let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model: "gpt-4o" };
        let result = LowestCost.select(&refs, &ctx).unwrap();
        assert_eq!(
            result.name, "enabled",
            "zero-weight provider must never be selected even when cheapest"
        );
    }
}
