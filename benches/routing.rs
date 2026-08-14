// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};

use oxigate::domain::ports::{
    HealthStatus, NanoUsd, ProviderCandidate, RoutingContext, RoutingStrategy,
};
use oxigate::domain::routing::{LowestCost, WeightedRandom};

// Minimal stub adapter for benchmarks.
struct StubAdapter(oxigate::domain::ports::ProviderMetadata);

#[async_trait::async_trait]
impl oxigate::domain::ports::ProviderAdapter for StubAdapter {
    async fn chat_completion(
        &self,
        _req: &oxigate::domain::chat::ChatRequest,
    ) -> Result<oxigate::domain::chat::ChatResponse, oxigate::domain::ports::ProviderError> {
        Err(oxigate::domain::ports::ProviderError::NotImplemented)
    }
    fn metadata(&self) -> &oxigate::domain::ports::ProviderMetadata {
        &self.0
    }
    async fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }
}

fn make_candidate(name: &str, cost: NanoUsd) -> ProviderCandidate {
    let meta = oxigate::domain::ports::ProviderMetadata {
        name: name.to_string(),
        supported_models: vec!["gpt-4o".to_string()],
        supports_streaming: false,
        supports_tools: false,
        supports_vision: false,
        supports_embeddings: false,
        supports_thinking: false,
    };
    ProviderCandidate {
        name: name.to_string(),
        adapter: Arc::new(StubAdapter(meta)),
        weight: 1.0,
        in_flight: 0,
        latency_ewma_ms: 0.0,
        is_cooling_down: false,
        cooldown_remaining_secs: 0,
        cost_per_million_tokens: cost,
    }
}

fn bench_weighted_random_3_candidates(c: &mut Criterion) {
    let candidates = vec![
        make_candidate("openai", NanoUsd::from_f64_usd(10.0)),
        make_candidate("anthropic", NanoUsd::from_f64_usd(8.0)),
        make_candidate("passthrough", NanoUsd::from_f64_usd(1.0)),
    ];
    let ctx = RoutingContext { model: "gpt-4o" };
    let strategy = WeightedRandom;
    let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
    c.bench_function("weighted_random_3_candidates", |b| {
        b.iter(|| {
            let _ = strategy.select(&refs, &ctx);
        });
    });
}

fn bench_lowest_cost_3_candidates(c: &mut Criterion) {
    let candidates = vec![
        make_candidate("openai", NanoUsd::from_f64_usd(10.0)),
        make_candidate("anthropic", NanoUsd::from_f64_usd(8.0)),
        make_candidate("passthrough", NanoUsd::from_f64_usd(1.0)),
    ];
    let ctx = RoutingContext { model: "gpt-4o" };
    let strategy = LowestCost;
    let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
    c.bench_function("lowest_cost_3_candidates", |b| {
        b.iter(|| {
            let _ = strategy.select(&refs, &ctx);
        });
    });
}

criterion_group!(
    benches,
    bench_weighted_random_3_candidates,
    bench_lowest_cost_3_candidates
);
criterion_main!(benches);
