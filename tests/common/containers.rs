// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Typed testcontainers helpers for PostgreSQL and Redis.
//!
//! Eliminates copy-paste across integration tests by wrapping container startup
//! and exposing connection URLs and pools.

use testcontainers::core::ContainerAsync;
use testcontainers_modules::postgres::Postgres;
use testcontainers_modules::redis::Redis;
use testcontainers_modules::testcontainers::runners::AsyncRunner;

use oxigate::config::{DatabaseConfig, RedisConfig, SecretString};
use oxigate::db::{connect, run_migrations};
use oxigate::redis_pool::create_pool;

const PG_PORT: u16 = 5432;
const REDIS_PORT: u16 = 6379;

/// PostgreSQL container with pre-configured pool and migrations applied.
pub struct PgContainer {
    #[allow(dead_code)]
    container: ContainerAsync<Postgres>,
    pub pool: oxigate::db::DbPool,
    #[allow(dead_code)]
    pub url: String,
}

impl PgContainer {
    /// Starts a Postgres container, connects, and runs migrations.
    pub async fn start() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let container: ContainerAsync<Postgres> = Postgres::default()
            .start()
            .await
            .map_err(|e| format!("postgres container start: {e}"))?;
        let host_port = container
            .get_host_port_ipv4(PG_PORT)
            .await
            .map_err(|e| format!("host port: {e}"))?;
        let url = format!("postgres://postgres:postgres@localhost:{host_port}/postgres");

        let config = DatabaseConfig {
            url: SecretString::new(url.clone()),
            max_connections: Some(2),
            pool_acquire_timeout_secs: Some(10),
        };

        let pool = connect(&config)
            .await
            .map_err(|e| format!("pool connect: {e}"))?;
        run_migrations(&pool)
            .await
            .map_err(|e| format!("migrations: {e}"))?;

        Ok(Self {
            container,
            pool,
            url,
        })
    }
}

/// Redis container with pre-configured pool.
pub struct RedisContainer {
    #[allow(dead_code)]
    container: ContainerAsync<Redis>,
    pub pool: oxigate::redis_pool::RedisPool,
    #[allow(dead_code)]
    pub url: String,
}

impl RedisContainer {
    /// Starts a Redis container and creates a connection pool.
    pub async fn start() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let container: ContainerAsync<Redis> = Redis::default()
            .start()
            .await
            .map_err(|e| format!("redis container start: {e}"))?;
        let host_port = container
            .get_host_port_ipv4(REDIS_PORT)
            .await
            .map_err(|e| format!("host port: {e}"))?;
        let url = format!("redis://127.0.0.1:{host_port}");

        let config = RedisConfig {
            url: SecretString::new(url.clone()),
            pool_size: Some(2),
            pool_timeout_secs: Some(2),
        };

        let pool = create_pool(&config).map_err(|e| format!("pool create: {e}"))?;

        Ok(Self {
            container,
            pool,
            url,
        })
    }
}
