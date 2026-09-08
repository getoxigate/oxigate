// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Shared test helpers for integration tests.
//!
//! Re-exports submodules: containers, fixtures, gateway, stub_adapter, wiremock_stubs.

pub mod containers;
pub mod fixtures;
pub mod gateway;
pub mod stub_adapter;
pub mod wiremock_stubs;

use std::sync::OnceLock;
use std::time::Duration;

use oxigate::config::{RedisConfig, SecretString};
use oxigate::redis_pool::create_pool;

use metrics_exporter_prometheus::PrometheusHandle;

/// Exactly the last chunk is terminal, and every chunk before it is not.
///
/// Shared because every adapter lane asserts the same shape: the terminal-chunk contract is that
/// one chunk closes a completed response and nothing before it claims to.
pub fn assert_only_last_is_final(chunks: &[oxigate::domain::chat::StreamChunk]) {
    let marked: Vec<usize> = chunks
        .iter()
        .enumerate()
        .filter(|(_, c)| c.is_final)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        marked,
        vec![chunks.len() - 1],
        "exactly the last chunk must be terminal, got marks at {marked:?} of {} chunks",
        chunks.len()
    );
}

/// The bundled pricing catalogue in a hot-reload holder.
///
/// Every provider adapter that accounts cache writes takes this holder so it can snapshot one
/// pricing generation per request. Most tests are not about pricing and just need a real one.
pub fn bundled_pricing_holder()
-> std::sync::Arc<std::sync::RwLock<oxigate::domain::pricing::PricingDb>> {
    let db = oxigate::domain::pricing::PricingDb::load(
        oxigate::domain::pricing::BUNDLED_PRICING_JSON,
        &oxigate::config::PricingConfig::default(),
    )
    .expect("bundled pricing DB loads");
    std::sync::Arc::new(std::sync::RwLock::new(db))
}

/// Lazy (never-connecting) PG pool for tests that need an AppState but do not touch the DB.
pub fn lazy_pg_pool() -> oxigate::db::DbPool {
    sqlx::postgres::PgPoolOptions::new()
        .acquire_timeout(Duration::from_millis(100))
        .connect_lazy("postgres://postgres:postgres@127.0.0.1:1/postgres")
        .expect("lazy PG pool must build")
}

/// Lazy (never-connecting) Redis pool for tests that need an AppState but do not touch Redis.
pub fn lazy_redis_pool() -> oxigate::redis_pool::RedisPool {
    create_pool(&RedisConfig {
        url: SecretString::new("redis://127.0.0.1:1"),
        pool_size: Some(1),
        pool_timeout_secs: Some(1),
    })
    .expect("lazy Redis pool must build")
}

/// Returns the shared process-global Prometheus handle, installing the recorder on first call.
///
/// All integration test modules that need to assert on metric output must call this instead of
/// installing their own recorder — `metrics::set_global_recorder` is a one-shot operation per
/// process; duplicate installs panic.
pub fn test_prometheus_handle() -> PrometheusHandle {
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE
        .get_or_init(|| {
            metrics_exporter_prometheus::PrometheusBuilder::new()
                .install_recorder()
                .expect("prometheus recorder must install once in test process")
        })
        .clone()
}
