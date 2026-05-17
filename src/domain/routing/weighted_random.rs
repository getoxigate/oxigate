// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! WeightedRandom routing strategy — Walker's Alias Method (weighted_rand 0.4).
//!
//! Selects providers proportionally to their configured weights. Providers with
//! weight 0.0 are excluded. Defers to `NoEligibleCandidates` when all weights are zero.

use rand::thread_rng;
use weighted_rand::builder::{NewBuilder, WalkerTableBuilder};

use crate::domain::ports::{ProviderCandidate, RoutingContext, RoutingStrategy, StrategyError};

/// Weighted random selection using Walker's Alias Method.
///
/// O(1) per call after O(n) table construction; P99 target < 5µs for 3 candidates.
pub struct WeightedRandom;

impl RoutingStrategy for WeightedRandom {
    fn select<'s>(
        &self,
        candidates: &[&'s ProviderCandidate],
        _ctx: &RoutingContext<'_>,
    ) -> Result<&'s ProviderCandidate, StrategyError> {
        let eligible: Vec<&ProviderCandidate> = candidates
            .iter()
            .copied()
            .filter(|c| c.weight > 0.0)
            .collect();

        if eligible.is_empty() {
            return Err(StrategyError::NoEligibleCandidates);
        }

        // Single candidate: skip table construction.
        if eligible.len() == 1 {
            return Ok(eligible[0]);
        }

        let weights: Vec<f32> = eligible.iter().map(|c| c.weight as f32).collect();
        let table = WalkerTableBuilder::new(&weights).build();
        let mut rng = thread_rng();
        let idx = table.next_rng(&mut rng);
        Ok(eligible[idx])
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

    fn make_candidate(name: &str, weight: f64) -> ProviderCandidate {
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
            weight,
            in_flight: 0,
            latency_ewma_ms: 0.0,
            is_cooling_down: false,
            cooldown_remaining_secs: 0,
            cost_per_million_tokens: NanoUsd::MAX,
        }
    }

    #[test]
    fn test_weighted_random_single_candidate() {
        let candidates = vec![make_candidate("openai", 1.0)];
        let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model: "gpt-4o" };
        let strategy = WeightedRandom;
        let result = strategy.select(&refs, &ctx).unwrap();
        assert_eq!(result.name, "openai");
    }

    #[test]
    fn test_weighted_random_zero_weight_excluded() {
        let candidates = vec![
            make_candidate("openai", 0.0),
            make_candidate("anthropic", 1.0),
        ];
        let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model: "gpt-4o" };
        let strategy = WeightedRandom;
        for _ in 0..50 {
            let result = strategy.select(&refs, &ctx).unwrap();
            assert_eq!(
                result.name, "anthropic",
                "zero-weight provider must never be selected"
            );
        }
    }

    #[test]
    fn test_weighted_random_no_eligible_candidates() {
        let candidates = vec![make_candidate("openai", 0.0)];
        let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model: "gpt-4o" };
        let strategy = WeightedRandom;
        let err = strategy.select(&refs, &ctx).unwrap_err();
        assert!(
            matches!(err, StrategyError::NoEligibleCandidates),
            "all-zero-weight must return NoEligibleCandidates"
        );
    }

    #[test]
    fn test_weighted_random_distribution_roughly_proportional() {
        let candidates = vec![make_candidate("high", 9.0), make_candidate("low", 1.0)];
        let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
        let ctx = RoutingContext { model: "m" };
        let strategy = WeightedRandom;
        let mut counts = std::collections::HashMap::new();
        for _ in 0..1000 {
            let r = strategy.select(&refs, &ctx).unwrap();
            *counts.entry(r.name.clone()).or_insert(0u32) += 1;
        }
        // "high" should be selected ~900/1000 ± noise. Accept 750..=980 as reasonable.
        let high = counts.get("high").copied().unwrap_or(0);
        assert!(
            (750..=980).contains(&high),
            "high-weight provider should be selected ~90% of the time; got {high}/1000"
        );
    }
}
