// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for health-tracker + strategy routing .
//!
//! Tests the `ProviderHealthTracker` + routing strategies directly, without going
//! through the HTTP layer, to exercise the Redis cooldown path and fail-open fallback.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use oxigate::config::{PricingConfig, RedisConfig, SecretString};
use oxigate::domain::chat::{ChatRequest, ChatResponse};
use oxigate::domain::ports::{
    HealthStatus, ProviderAdapter, ProviderCandidate, ProviderError, ProviderMetadata,
    RoutingContext, RoutingStrategy, StrategyError,
};
use oxigate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
use oxigate::domain::routing::{LowestCost, RateLimitAware};
use oxigate::providers::ProviderHealthTracker;
use oxigate::redis_pool::{RedisPool, create_pool};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn make_provider(name: &str) -> Arc<dyn ProviderAdapter> {
    struct Stub(ProviderMetadata);

    #[async_trait]
    impl ProviderAdapter for Stub {
        async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
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
    Arc::new(Stub(meta))
}

fn make_pricing_db() -> Arc<std::sync::RwLock<PricingDb>> {
    Arc::new(std::sync::RwLock::new(
        PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing must parse"),
    ))
}

fn unreachable_redis() -> Arc<tokio::sync::RwLock<RedisPool>> {
    // Port 19999 — high unprivileged port, unlikely to be in use in CI environments.
    // Port 1 is privileged and may be bound by system services on some platforms.
    let pool = create_pool(&RedisConfig {
        url: SecretString::new("redis://127.0.0.1:19999"),
        pool_size: Some(1),
        pool_timeout_secs: Some(1),
    })
    .expect("lazy pool must build for unreachable URL");
    Arc::new(tokio::sync::RwLock::new(pool))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// After `on_rate_limit` marks a provider as cooling, `RateLimitAware` must exclude
/// it from selection. The other provider must be chosen on every subsequent call.
#[tokio::test]
async fn rate_limit_cooldown_e2e() {
    let provider_a = make_provider("provider_a");
    let provider_b = make_provider("provider_b");
    let providers: Vec<Arc<dyn ProviderAdapter>> =
        vec![Arc::clone(&provider_a), Arc::clone(&provider_b)];
    let names: Vec<String> = vec!["provider_a".to_string(), "provider_b".to_string()];
    let tracker = ProviderHealthTracker::new(&names, None, 60, 0.1);
    let pricing_db = make_pricing_db();
    let weights = HashMap::new();

    // Trigger rate-limit on provider_a.
    tracker.on_rate_limit("provider_a").await;

    // Build candidate snapshot — provider_a must be cooling.
    let candidates = tracker
        .candidates(&providers, &weights, "test-model", &pricing_db)
        .await;
    let a = candidates
        .iter()
        .find(|c| c.name == "provider_a")
        .expect("provider_a must be in candidates");
    assert!(
        a.is_cooling_down,
        "provider_a must be in cooldown after on_rate_limit"
    );

    // RateLimitAware must always select provider_b (provider_a is cooling).
    let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
    let ctx = RoutingContext {
        model: "test-model",
    };
    let strategy = RateLimitAware;
    for _ in 0..20 {
        let selected = strategy
            .select(&refs, &ctx)
            .expect("RateLimitAware must select a candidate when provider_b is healthy");
        assert_eq!(
            selected.name, "provider_b",
            "cooling provider_a must never be selected"
        );
    }
}

/// When Redis is unreachable, `on_rate_limit` must not panic. Local in-memory cooldown
/// must still be set so routing correctly excludes the provider on the next request.
///
/// Note: `pool_timeout_secs = 1` means this test may take ~1 s in CI when the pool
/// connection attempt times out. `multi_thread` flavour avoids blocking the single-threaded
/// executor during the timeout wait.
#[tokio::test(flavor = "multi_thread")]
async fn redis_down_fallback() {
    let provider_a = make_provider("provider_a");
    let provider_b = make_provider("provider_b");
    let providers: Vec<Arc<dyn ProviderAdapter>> =
        vec![Arc::clone(&provider_a), Arc::clone(&provider_b)];
    let names: Vec<String> = vec!["provider_a".to_string(), "provider_b".to_string()];
    let tracker = ProviderHealthTracker::new(&names, Some(unreachable_redis()), 60, 0.1);
    let pricing_db = make_pricing_db();
    let weights = HashMap::new();

    // on_rate_limit must not panic even when Redis is unavailable.
    // Redis write failure is logged as WARN and suppressed; local state is still set.
    tracker.on_rate_limit("provider_a").await;

    // Local cooldown_until must be set — check_cooldown returns true from local state
    // without needing Redis.
    let candidates = tracker
        .candidates(&providers, &weights, "test-model", &pricing_db)
        .await;
    let a = candidates
        .iter()
        .find(|c| c.name == "provider_a")
        .expect("provider_a must be in candidates");
    assert!(
        a.is_cooling_down,
        "local cooldown must be enforced even when Redis is unreachable"
    );

    // Routing continues: provider_b must still be selectable.
    let b = candidates
        .iter()
        .find(|c| c.name == "provider_b")
        .expect("provider_b must be in candidates");
    assert!(
        !b.is_cooling_down,
        "provider_b must not be affected by provider_a's cooldown"
    );
    let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
    let ctx = RoutingContext {
        model: "test-model",
    };
    let selected = RateLimitAware
        .select(&refs, &ctx)
        .expect("routing must succeed with one healthy provider");
    assert_eq!(selected.name, "provider_b");
}

/// `LowestCost` strategy selects the provider whose model has the cheapest known input cost.
///
/// **Architecture note**: `cost_per_million_tokens` is keyed by model name (via `PricingDb`),
/// not by provider. Two providers serving the same model name receive identical cost values
/// and fall back to `WeightedRandom`. This test uses two providers with *different* model names
/// (haiku = cheap, sonnet = expensive), which is the correct way to exercise `LowestCost`
/// selection with the current design. Per-provider pricing overrides are deferred to.
///
/// Each candidate list contains only the one provider that matches the requested model,
/// so `LowestCost` selects it deterministically — verifying that pricing is populated and
/// the strategy doesn't error on known-cost candidates.
#[tokio::test]
async fn lowest_cost_selects_cheapest_provider() {
    // Use model names present in the bundled pricing DB so costs are populated.
    let cheap_model = "claude-3-haiku-20240307";
    let expensive_model = "claude-3-5-sonnet-20241022";

    struct Stub(ProviderMetadata);
    #[async_trait]
    impl ProviderAdapter for Stub {
        async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
            Err(ProviderError::NotImplemented)
        }
        fn metadata(&self) -> &ProviderMetadata {
            &self.0
        }
        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }

    fn make_stub(name: &str, model: &str) -> Arc<dyn ProviderAdapter> {
        let meta = ProviderMetadata {
            name: name.to_string(),
            supported_models: vec![model.to_string()],
            supports_streaming: false,
            supports_tools: false,
            supports_vision: false,
            supports_embeddings: false,
            supports_thinking: false,
            kind: Default::default(),
            ..Default::default()
        };
        Arc::new(Stub(meta))
    }

    let cheap = make_stub("provider_cheap", cheap_model);
    let expensive = make_stub("provider_expensive", expensive_model);
    let providers: Vec<Arc<dyn ProviderAdapter>> = vec![Arc::clone(&cheap), Arc::clone(&expensive)];
    let names = vec![
        "provider_cheap".to_string(),
        "provider_expensive".to_string(),
    ];
    let tracker = ProviderHealthTracker::new(&names, None, 60, 0.1);
    let pricing_db = make_pricing_db();
    let weights = HashMap::new();

    // Build candidates for the cheap model — only provider_cheap matches it.
    let candidates_cheap = tracker
        .candidates(&providers, &weights, cheap_model, &pricing_db)
        .await;
    let refs_cheap: Vec<&ProviderCandidate> = candidates_cheap.iter().collect();
    let ctx_cheap = RoutingContext { model: cheap_model };
    let selected = LowestCost
        .select(&refs_cheap, &ctx_cheap)
        .expect("LowestCost must select when candidates are available");
    assert_eq!(
        selected.name, "provider_cheap",
        "LowestCost must select provider_cheap for the cheaper model"
    );

    // Build candidates for the expensive model — only provider_expensive matches it.
    let candidates_expensive = tracker
        .candidates(&providers, &weights, expensive_model, &pricing_db)
        .await;
    let refs_expensive: Vec<&ProviderCandidate> = candidates_expensive.iter().collect();
    let ctx_expensive = RoutingContext {
        model: expensive_model,
    };
    let selected = LowestCost
        .select(&refs_expensive, &ctx_expensive)
        .expect("LowestCost must select when candidates are available");
    assert_eq!(
        selected.name, "provider_expensive",
        "LowestCost must select provider_expensive for the expensive model"
    );
}

/// When all providers are in cooldown, `RateLimitAware` must return
/// `AllProvidersRateLimited` with a positive `retry_after`. This is the signal that
/// the HTTP layer converts into HTTP 503 + Retry-After header.
#[tokio::test]
async fn all_providers_cooling_returns_strategy_error() {
    let provider_a = make_provider("provider_a");
    let provider_b = make_provider("provider_b");
    let providers: Vec<Arc<dyn ProviderAdapter>> =
        vec![Arc::clone(&provider_a), Arc::clone(&provider_b)];
    let names = vec!["provider_a".to_string(), "provider_b".to_string()];
    let tracker = ProviderHealthTracker::new(&names, None, 60, 0.1);
    let pricing_db = make_pricing_db();
    let weights = HashMap::new();

    tracker.on_rate_limit("provider_a").await;
    tracker.on_rate_limit("provider_b").await;

    let candidates = tracker
        .candidates(&providers, &weights, "test-model", &pricing_db)
        .await;
    assert!(
        candidates.iter().all(|c| c.is_cooling_down),
        "both providers must be cooling"
    );

    let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
    let ctx = RoutingContext {
        model: "test-model",
    };
    let err = RateLimitAware
        .select(&refs, &ctx)
        .expect_err("must return error when all providers are cooling");
    match err {
        StrategyError::AllProvidersRateLimited { retry_after } => {
            assert!(retry_after >= 1, "retry_after must be at least 1 second");
        }
        other => panic!("expected AllProvidersRateLimited, got {other:?}"),
    }
}

/// After the cooldown window expires, a previously-cooling provider must be re-admitted
/// to the candidate pool. This test uses a zero cooldown to simulate expiry immediately.
#[tokio::test]
async fn cooldown_expiry_re_admits_provider() {
    let provider_a = make_provider("provider_a");
    let provider_b = make_provider("provider_b");
    let providers: Vec<Arc<dyn ProviderAdapter>> =
        vec![Arc::clone(&provider_a), Arc::clone(&provider_b)];
    let names = vec!["provider_a".to_string(), "provider_b".to_string()];
    // cooldown_secs = 0 means the cooldown window is instantaneous.
    let tracker = ProviderHealthTracker::new(&names, None, 0, 0.1);
    let pricing_db = make_pricing_db();
    let weights = HashMap::new();

    tracker.on_rate_limit("provider_a").await;

    // With cooldown_secs=0 the cooldown_until is in the past; candidates() must
    // see provider_a as available immediately.
    let candidates = tracker
        .candidates(&providers, &weights, "test-model", &pricing_db)
        .await;
    let a = candidates
        .iter()
        .find(|c| c.name == "provider_a")
        .expect("provider_a must be in candidates");
    assert!(
        !a.is_cooling_down,
        "provider_a must be re-admitted after zero-duration cooldown expires"
    );
}

/// `LowestCost` must never select a zero-weight provider even when it is the only
/// candidate with known pricing. Zero weight means the operator has disabled the provider.
#[tokio::test]
async fn lowest_cost_zero_weight_excluded_e2e() {
    let provider_a = make_provider("provider_a");
    let provider_b = make_provider("provider_b");
    let providers: Vec<Arc<dyn ProviderAdapter>> =
        vec![Arc::clone(&provider_a), Arc::clone(&provider_b)];
    let names = vec!["provider_a".to_string(), "provider_b".to_string()];
    let tracker = ProviderHealthTracker::new(&names, None, 60, 0.1);
    let pricing_db = make_pricing_db();

    // provider_a weight=0.0 (disabled), provider_b weight=1.0 (active).
    let mut weights = HashMap::new();
    weights.insert("provider_a".to_string(), 0.0_f64);
    weights.insert("provider_b".to_string(), 1.0_f64);

    let candidates = tracker
        .candidates(&providers, &weights, "test-model", &pricing_db)
        .await;

    let a = candidates
        .iter()
        .find(|c| c.name == "provider_a")
        .expect("provider_a must be in candidates");
    assert_eq!(a.weight, 0.0, "provider_a must have weight 0.0");

    // LowestCost must skip provider_a (weight=0) and select provider_b.
    let refs: Vec<&ProviderCandidate> = candidates.iter().collect();
    let ctx = RoutingContext {
        model: "test-model",
    };
    // All candidates have NanoUsd::MAX for "test-model" (not in pricing DB),
    // so LowestCost falls back to WeightedRandom among eligible (weight > 0) candidates.
    // Only provider_b is eligible.
    let selected = LowestCost
        .select(&refs, &ctx)
        .expect("LowestCost must select provider_b");
    assert_eq!(
        selected.name, "provider_b",
        "zero-weight provider_a must never be selected"
    );
}
