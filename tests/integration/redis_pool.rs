// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for Redis pool .
//!
//! Uses testcontainers via RedisContainer to spin up a real Redis instance.

use crate::common::containers::RedisContainer;
use oxigate::config::{RedisConfig, SecretString};
use oxigate::redis_pool::{create_pool, health_check};

#[tokio::test]
async fn test_pool_creates_and_health_check_passes() {
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    health_check(&redis.pool)
        .await
        .expect("health check should pass");
}

#[tokio::test]
async fn test_health_check_fails_on_unreachable_redis() {
    let config = RedisConfig {
        url: SecretString::new("redis://127.0.0.1:19999"),
        pool_size: Some(1),
        pool_timeout_secs: Some(1),
    };

    let pool = create_pool(&config).expect("pool struct creates even with bad URL");
    let result = health_check(&pool).await;
    assert!(
        result.is_err(),
        "health check should fail on unreachable Redis"
    );
}

#[tokio::test]
async fn test_pool_exhaustion_returns_timeout() {
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let config = RedisConfig {
        url: SecretString::new(redis.url.clone()),
        pool_size: Some(1),
        pool_timeout_secs: Some(1),
    };

    let pool = create_pool(&config).unwrap();
    let _conn = pool.get().await.unwrap();

    let result = tokio::time::timeout(std::time::Duration::from_secs(2), health_check(&pool)).await;

    assert!(
        result.is_err() || result.unwrap().is_err(),
        "should not deadlock; expected timeout or PoolError"
    );
}
