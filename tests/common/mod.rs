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
