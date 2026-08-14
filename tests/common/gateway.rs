// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! E2E gateway startup helper for integration tests.
//!
//! Uses axum-test's TestServer (no child process) for fast, deterministic tests.
//!
//! Constructs AppState in-tree since #[cfg(test)] on the lib is not set when
//! the library is built as a dependency of the integration test binary.

use std::sync::Arc;

use axum::Router;
#[cfg(feature = "test-hooks")]
use chrono::{DateTime, Utc};
use oxigate::api::{AppState, router, router_with_body_limit};
use oxigate::config::{AuthConfig, BudgetConfig, PricingConfig, SecurityConfig};
use oxigate::db::DbPool;
use oxigate::domain::ports::{NanoUsd, ProviderAdapterExt};
use oxigate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
use oxigate::middleware::budget::BudgetRuntimeConfig;
use oxigate::middleware::global_safety::GlobalSafetyRuntimeConfig;
use oxigate::providers::ProviderHealthTracker;
use oxigate::redis_pool::RedisPool;

fn test_app_state(
    pool: DbPool,
    redis: RedisPool,
    provider: Arc<dyn ProviderAdapterExt>,
    auth: AuthConfig,
    health: Arc<ProviderHealthTracker>,
) -> AppState {
    let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
        .expect("bundled pricing DB must parse in tests");
    AppState {
        pool: Arc::new(tokio::sync::RwLock::new(pool)),
        redis_pool: Arc::new(tokio::sync::RwLock::new(redis)),
        pricing_db: Arc::new(std::sync::RwLock::new(pricing_db)),
        provider: Arc::new(tokio::sync::RwLock::new(provider)),
        auth: Arc::new(tokio::sync::RwLock::new(auth)),
        global_safety: Arc::new(tokio::sync::RwLock::new(
            GlobalSafetyRuntimeConfig::default(), // None cap — no-op in all existing tests
        )),
        budget_settings: Arc::new(tokio::sync::RwLock::new(BudgetConfig::default())),
        budget: Arc::new(tokio::sync::RwLock::new(BudgetRuntimeConfig::default())),
        startup_time: 1,
        health,
        security: Arc::new(tokio::sync::RwLock::new(SecurityConfig::default())),
    }
}

fn test_gateway_router(
    pool: DbPool,
    redis: RedisPool,
    provider: Arc<dyn ProviderAdapterExt>,
    auth: AuthConfig,
    health: Arc<ProviderHealthTracker>,
) -> Router {
    router(test_app_state(pool, redis, provider, auth, health))
}

/// Gateway handle for E2E tests. Wraps axum-test TestServer.
pub struct TestGateway {
    pub server: axum_test::TestServer,
}

impl TestGateway {
    /// Spawns the gateway with the given pools and provider adapter.
    /// Uses auth bypass (key: None) and empty health tracker by default.
    pub async fn spawn(
        pool: DbPool,
        redis: RedisPool,
        provider: Arc<dyn ProviderAdapterExt>,
    ) -> Self {
        Self::spawn_with_auth(pool, redis, provider, AuthConfig::default()).await
    }

    /// Spawns the gateway with explicit auth config.
    pub async fn spawn_with_auth(
        pool: DbPool,
        redis: RedisPool,
        provider: Arc<dyn ProviderAdapterExt>,
        auth: AuthConfig,
    ) -> Self {
        Self::spawn_with_health_tracker(
            pool,
            redis,
            provider,
            auth,
            ProviderHealthTracker::new_for_test(&[]),
        )
        .await
    }

    /// Spawns the gateway with explicit auth config and a pre-built health tracker.
    ///
    /// Use this when tests need to assert on provider health values — production startup
    /// populates the tracker from `provider.health_check()`; tests inject it directly to
    /// avoid real network calls and keep tests hermetic.
    ///
    /// `startup_time` is hardcoded to `1` for determinism; tests should not assert on
    /// the exact value of `created` unless they also control `startup_time`.
    pub async fn spawn_with_health_tracker(
        pool: DbPool,
        redis: RedisPool,
        provider: Arc<dyn ProviderAdapterExt>,
        auth: AuthConfig,
        health: Arc<ProviderHealthTracker>,
    ) -> Self {
        let app_router = test_gateway_router(pool, redis, provider, auth, health);
        let server = axum_test::TestServer::new(app_router).expect("TestServer must build");
        Self { server }
    }

    /// Binds a real ephemeral TCP port so [`axum_test::TestServer::server_url`] works with an
    /// external HTTP client (e.g. streaming disconnect — `reqwest` cannot use mock transport).
    pub async fn spawn_random_http_port(
        pool: DbPool,
        redis: RedisPool,
        provider: Arc<dyn ProviderAdapterExt>,
    ) -> Self {
        let app_router = test_gateway_router(
            pool,
            redis,
            provider,
            AuthConfig::default(),
            ProviderHealthTracker::new_for_test(&[]),
        );
        let server = axum_test::TestServer::new_with_config(
            app_router,
            axum_test::TestServerConfig {
                transport: Some(axum_test::Transport::HttpRandomPort),
                ..Default::default()
            },
        )
        .expect("TestServer must build");
        Self { server }
    }

    /// Spawns the gateway with a specific BudgetConfig.
    /// Used by budget_e2e tests. Auth is explicitly passed.
    pub async fn spawn_with_budget(
        pool: DbPool,
        redis: RedisPool,
        provider: Arc<dyn ProviderAdapterExt>,
        auth: AuthConfig,
        budget: BudgetConfig,
    ) -> Self {
        let mut app_state = test_app_state(
            pool,
            redis,
            provider,
            auth,
            ProviderHealthTracker::new_for_test(&[]),
        );
        app_state.budget_settings = Arc::new(tokio::sync::RwLock::new(budget.clone()));
        app_state.budget = Arc::new(tokio::sync::RwLock::new(
            BudgetRuntimeConfig::from_budget_config(budget),
        ));
        let app_router = router(app_state);
        let server = axum_test::TestServer::new(app_router).expect("TestServer must build");
        Self { server }
    }

    /// Like [`Self::spawn_with_budget`], but returns shared [`BudgetRuntimeConfig`] for
    /// `test-hooks` clock injection (`now_override`).
    #[cfg(feature = "test-hooks")]
    pub async fn spawn_with_budget_and_clock(
        pool: DbPool,
        redis: RedisPool,
        provider: Arc<dyn ProviderAdapterExt>,
        auth: AuthConfig,
        budget: BudgetConfig,
        now_override: Option<DateTime<Utc>>,
    ) -> (Self, Arc<tokio::sync::RwLock<BudgetRuntimeConfig>>) {
        let pricing_db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing DB must parse in tests");
        let mut runtime = BudgetRuntimeConfig::from_budget_config(budget.clone());
        runtime.now_override = now_override;
        let budget_arc = Arc::new(tokio::sync::RwLock::new(runtime));
        let app_state = AppState {
            pool: Arc::new(tokio::sync::RwLock::new(pool)),
            redis_pool: Arc::new(tokio::sync::RwLock::new(redis)),
            pricing_db: Arc::new(std::sync::RwLock::new(pricing_db)),
            provider: Arc::new(tokio::sync::RwLock::new(provider)),
            auth: Arc::new(tokio::sync::RwLock::new(auth)),
            global_safety: Arc::new(tokio::sync::RwLock::new(
                GlobalSafetyRuntimeConfig::default(),
            )),
            budget_settings: Arc::new(tokio::sync::RwLock::new(budget.clone())),
            budget: Arc::clone(&budget_arc),
            startup_time: 1,
            health: ProviderHealthTracker::new_for_test(&[]),
            security: Arc::new(tokio::sync::RwLock::new(SecurityConfig::default())),
        };
        let app_router = router(app_state);
        let server = axum_test::TestServer::new(app_router).expect("TestServer must build");
        (Self { server }, budget_arc)
    }

    /// Spawns the gateway with a custom request body size limit.
    /// Used by request_size_limits tests to verify 413 enforcement without sending 50 MiB.
    pub async fn spawn_with_body_limit(
        pool: DbPool,
        redis: RedisPool,
        provider: Arc<dyn ProviderAdapterExt>,
        max_request_body_bytes: usize,
    ) -> Self {
        let app_state = test_app_state(
            pool,
            redis,
            provider,
            AuthConfig::default(),
            ProviderHealthTracker::new_for_test(&[]),
        );
        let app_router = router_with_body_limit(app_state, max_request_body_bytes);
        let server = axum_test::TestServer::new(app_router).expect("TestServer must build");
        Self { server }
    }

    /// Spawns the gateway with a specific `SecurityConfig`.
    /// Used by fallback_e2e tests to enable `expose_provider_names`. Auth is bypassed.
    pub async fn spawn_with_security(
        pool: DbPool,
        redis: RedisPool,
        provider: Arc<dyn ProviderAdapterExt>,
        security: SecurityConfig,
    ) -> Self {
        let mut app_state = test_app_state(
            pool,
            redis,
            provider,
            AuthConfig::default(),
            ProviderHealthTracker::new_for_test(&[]),
        );
        app_state.security = Arc::new(tokio::sync::RwLock::new(security));
        let app_router = router(app_state);
        let server = axum_test::TestServer::new(app_router).expect("TestServer must build");
        Self { server }
    }

    /// Spawns the gateway with team/tag budget config. Returns a live handle to the
    /// `BudgetConfig` Arc so callers can simulate SIGHUP by swapping config in-place:
    /// `*handle.write().await = updated_budget;`
    ///
    /// Note: this sets only `AppState::budget_settings` (TeamTagBudgetLayer). It does *not*
    /// update `AppState::budget` / `BudgetRuntimeConfig`, so per-identity BudgetLayer caps
    /// remain at defaults (i.e., inactive unless separately configured).
    pub async fn spawn_with_team_tag_budgets(
        pool: DbPool,
        redis: RedisPool,
        provider: Arc<dyn ProviderAdapterExt>,
        auth: AuthConfig,
        budget: BudgetConfig,
    ) -> (Self, Arc<tokio::sync::RwLock<BudgetConfig>>) {
        let budget_settings = Arc::new(tokio::sync::RwLock::new(budget));
        let mut state = test_app_state(
            pool,
            redis,
            provider,
            auth,
            ProviderHealthTracker::new_for_test(&[]),
        );
        state.budget_settings = Arc::clone(&budget_settings);
        let app_router = router(state);
        let server = axum_test::TestServer::new(app_router).expect("TestServer must build");
        (Self { server }, budget_settings)
    }

    /// Spawns the gateway with a specific global safety cap in NanoUsd.
    /// Used by global_safety_e2e tests. Auth is bypassed (key: None).
    pub async fn spawn_with_global_safety_cap(
        pool: DbPool,
        redis: RedisPool,
        provider: Arc<dyn ProviderAdapterExt>,
        cap_nano_usd: Option<NanoUsd>,
    ) -> Self {
        let mut app_state = test_app_state(
            pool,
            redis,
            provider,
            AuthConfig::default(),
            ProviderHealthTracker::new_for_test(&[]),
        );
        app_state.global_safety = Arc::new(tokio::sync::RwLock::new(
            GlobalSafetyRuntimeConfig::with_nano_usd_cap(cap_nano_usd),
        ));
        let app_router = router(app_state);
        let server = axum_test::TestServer::new(app_router).expect("TestServer must build");
        Self { server }
    }
}
