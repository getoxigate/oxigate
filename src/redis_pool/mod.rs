// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Redis connection pool (deadpool-redis).
//!
//! `RedisPool` is a type alias for `deadpool_redis::Pool`. All Redis command code receives
//! a `RedisPool` (or a connection checked out from it) — never a raw connection directly.
//!
//! Module is named `redis_pool` to avoid crate name conflict with the `redis` crate.
//!
//! # Logging
//!
//! When adding tracing for Redis commands (e.g. debug spans with command names),
//! never log keys, values, or AUTH credentials. Mask or omit sensitive data.

use deadpool_redis::{Config as DeadpoolConfig, Pool, PoolConfig as DeadpoolPoolConfig, Runtime};
use secrecy::ExposeSecret;
use thiserror::Error;

use crate::config::RedisConfig;

/// Shareable Redis connection pool. Clone is cheap (Arc-backed).
pub type RedisPool = Pool;

/// Redis pool or command error.
#[derive(Debug, Error)]
pub enum RedisError {
    /// Pool could not be created (bad URL, invalid config, or build failure).
    #[error("redis pool creation failed: {0}")]
    Create(#[from] deadpool_redis::CreatePoolError),
    /// Pool is exhausted; connection could not be acquired within the configured timeout.
    #[error("redis pool timeout: {0}")]
    Pool(#[from] deadpool_redis::PoolError),
    /// A Redis command failed.
    #[error("redis command error: {0}")]
    Command(#[from] redis::RedisError),
}

/// Build a `deadpool-redis` pool from `config`.
///
/// `pool_size` defaults to 16 when not specified.
/// `pool_timeout_secs` is the maximum time to wait for a free connection; defaults to 5 s.
///
/// Uses `Config::create_pool(Some(Runtime::Tokio1))` — the convenience method that returns
/// `CreatePoolError` (wrapping both URL config and pool build errors).
///
/// Returns `RedisError::Create` if the URL is invalid or pool construction fails.
/// Does NOT verify connectivity — call `health_check()` after this to confirm Redis is reachable.
#[allow(clippy::module_name_repetitions)]
pub fn create_pool(config: &RedisConfig) -> Result<RedisPool, RedisError> {
    let mut cfg = DeadpoolConfig::from_url(config.url.expose_secret());
    cfg.pool = Some(DeadpoolPoolConfig {
        max_size: config.pool_size.unwrap_or(16) as usize,
        timeouts: deadpool_redis::Timeouts {
            wait: Some(std::time::Duration::from_secs(
                config.pool_timeout_secs.unwrap_or(5),
            )),
            create: None,
            recycle: None,
        },
        ..Default::default()
    });
    cfg.create_pool(Some(Runtime::Tokio1))
        .map_err(RedisError::Create)
}

/// Send a PING to Redis and verify a response is returned.
///
/// Returns `Ok(())` on success, or `RedisError` on pool exhaustion or command failure.
/// Callers should apply a 500 ms timeout (see startup wiring in `main.rs`).
pub async fn health_check(pool: &RedisPool) -> Result<(), RedisError> {
    let mut conn = pool.get().await?;
    redis::cmd("PING").query_async::<String>(&mut *conn).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SecretString;

    #[test]
    fn test_redis_error_display_create() {
        let config = RedisConfig {
            url: SecretString::new("not-a-valid-redis-url"),
            pool_size: Some(1),
            pool_timeout_secs: Some(1),
        };
        let result = create_pool(&config);
        let err = result.unwrap_err();
        let s = err.to_string();
        assert!(s.contains("redis pool creation failed"));
    }

    #[test]
    fn test_redis_error_display_command() {
        let err = RedisError::Command(redis::RedisError::from((
            redis::ErrorKind::TypeError,
            "command failed",
        )));
        let s = err.to_string();
        assert!(s.contains("redis command error"));
    }

    #[test]
    fn test_redis_error_display_pool() {
        let err = RedisError::Pool(deadpool_redis::PoolError::Timeout(
            deadpool::managed::TimeoutType::Wait,
        ));
        let s = err.to_string();
        assert!(s.contains("redis pool timeout"));
    }
}
