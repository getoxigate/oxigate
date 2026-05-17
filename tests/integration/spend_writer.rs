// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for spend_writer .
//!
//! Uses PgContainer + RedisContainer from tests/common/containers.rs.
//! Tests cover: DB row insertion, Redis counter increment, TTL, seeding, failure isolation.

use std::sync::Arc;

use axum::http::StatusCode;
use bytes::Bytes;
use chrono::{TimeZone, Utc};
use tokio::sync::RwLock;

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::StreamStubAdapter;
use oxigate::api::CHAT_COMPLETIONS_PATH;
use oxigate::config::BudgetDuration;
use oxigate::domain::chat::{StreamChunk, Usage};
use oxigate::domain::ports::NanoUsd;
use oxigate::domain::spend::SpendRecord;
use oxigate::redis_pool::create_pool;

use chrono_tz::UTC;

/// Build a SpendRecord directly (SpendRecord has no #[non_exhaustive], all fields public).
fn make_record(org: &str, id: &str, cost: i64) -> SpendRecord {
    SpendRecord {
        org_id: org.into(),
        identity_id: id.into(),
        model: "gpt-4.1".into(),
        provider: "openai".into(),
        prompt_tokens: 100,
        completion_tokens: 50,
        cache_read_tokens: 0,
        cache_write_5m_tokens: 0,
        cache_write_1h_tokens: 0,
        thinking_tokens: 0,
        cost_nano_usd: NanoUsd::from_i64(cost),
        latency_ms: 42,
        tags: serde_json::Value::Object(serde_json::Map::new()),
    }
}

/// write_spend inserts a row into spend_records with correct field values,
/// including the split cache-write columns, thinking_tokens, latency_ms,
/// and tags (JSONB round-trip).
#[tokio::test]
async fn test_write_spend_inserts_row() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    let pool = Arc::new(RwLock::new(pg.pool.clone()));
    let rp = Arc::new(RwLock::new(redis.pool.clone()));

    // Use a record with non-zero values for every schema column under test.
    let record = SpendRecord {
        org_id: "acme".into(),
        identity_id: "key-1".into(),
        model: "gpt-4.1".into(),
        provider: "openai".into(),
        prompt_tokens: 100,
        completion_tokens: 50,
        cache_read_tokens: 3,
        cache_write_5m_tokens: 15,
        cache_write_1h_tokens: 8,
        thinking_tokens: 12,
        cost_nano_usd: NanoUsd(1_500_000_000),
        latency_ms: 42,
        tags: serde_json::json!({"team": "ml"}),
    };
    oxigate::db::spend_writer::write_spend(record, pool, rp, BudgetDuration::None, UTC, Utc::now())
        .await;

    let row: (
        String,
        String,
        String,
        String,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i32,
    ) = sqlx::query_as(
        "SELECT org_id, identity_id, model, provider, \
             prompt_tokens, completion_tokens, cost_nano_usd, \
             cache_write_5m_tokens, cache_write_1h_tokens, \
             thinking_tokens, cache_read_tokens, latency_ms \
             FROM spend_records LIMIT 1",
    )
    .fetch_one(&pg.pool)
    .await
    .expect("row must exist");

    assert_eq!(row.0, "acme");
    assert_eq!(row.1, "key-1");
    assert_eq!(row.2, "gpt-4.1");
    assert_eq!(row.3, "openai");
    assert_eq!(row.4, 100); // prompt_tokens
    assert_eq!(row.5, 50); // completion_tokens
    assert_eq!(row.6, 1_500_000_000); // cost_nano_usd
    assert_eq!(row.7, 15); // cache_write_5m_tokens
    assert_eq!(row.8, 8); // cache_write_1h_tokens
    assert_eq!(row.9, 12); // thinking_tokens
    assert_eq!(row.10, 3); // cache_read_tokens
    assert_eq!(row.11, 42); // latency_ms

    let tags: serde_json::Value = sqlx::query_scalar("SELECT tags FROM spend_records LIMIT 1")
        .fetch_one(&pg.pool)
        .await
        .expect("tags query");
    assert_eq!(
        tags["team"], "ml",
        "tags JSONB must round-trip through Postgres",
    );
}

/// write_spend called twice accumulates the Redis counter (INCRBY semantics).
#[tokio::test]
async fn test_write_spend_increments_redis() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    let pool = Arc::new(RwLock::new(pg.pool.clone()));
    let rp = Arc::new(RwLock::new(redis.pool.clone()));

    oxigate::db::spend_writer::write_spend(
        make_record("org", "id1", 1_000),
        Arc::clone(&pool),
        Arc::clone(&rp),
        BudgetDuration::None,
        UTC,
        Utc::now(),
    )
    .await;
    oxigate::db::spend_writer::write_spend(
        make_record("org", "id1", 2_000),
        Arc::clone(&pool),
        Arc::clone(&rp),
        BudgetDuration::None,
        UTC,
        Utc::now(),
    )
    .await;

    let key = "oxigate:org:org:spend:id1";
    let mut conn = redis.pool.get().await.expect("redis conn");
    let val: i64 = redis::cmd("GET")
        .arg(key)
        .query_async(&mut *conn)
        .await
        .expect("GET");
    assert_eq!(val, 3_000, "Redis counter must be sum of both writes");
}

/// after write_spend, Redis key has a TTL close to 60 days (within 5 s tolerance).
#[tokio::test]
async fn test_redis_key_has_60d_ttl() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    let pool = Arc::new(RwLock::new(pg.pool.clone()));
    let rp = Arc::new(RwLock::new(redis.pool.clone()));

    oxigate::db::spend_writer::write_spend(
        make_record("org", "ttl-id", 500),
        pool,
        rp,
        BudgetDuration::None,
        UTC,
        Utc::now(),
    )
    .await;

    let key = "oxigate:org:org:spend:ttl-id";
    let mut conn = redis.pool.get().await.expect("redis conn");
    let ttl: i64 = redis::cmd("TTL")
        .arg(key)
        .query_async(&mut *conn)
        .await
        .expect("TTL");

    let expected: i64 = 60 * 24 * 3600;
    assert!(
        (ttl - expected).abs() <= 5,
        "TTL should be ~{expected}s, got {ttl}"
    );
}

/// seed_redis_from_db sets Redis key to the sum from DB rows and seeds the global counter.
#[tokio::test]
async fn test_seed_redis_from_db() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    // Insert a row directly (bypassing write_spend).
    sqlx::query(
        "INSERT INTO spend_records (org_id, identity_id, model, provider, cost_nano_usd) \
         VALUES ('s-org', 's-id', 'gpt-4', 'openai', 9000)",
    )
    .execute(&pg.pool)
    .await
    .expect("direct insert");

    let period_start = Utc::now() - chrono::Duration::hours(1);
    oxigate::db::spend_writer::seed_redis_from_db(
        &pg.pool,
        &redis.pool,
        period_start,
        BudgetDuration::None,
        UTC,
        false,
    )
    .await;

    let mut conn = redis.pool.get().await.expect("redis conn");

    let val: i64 = redis::cmd("GET")
        .arg("oxigate:org:s-org:spend:s-id")
        .query_async(&mut *conn)
        .await
        .expect("GET identity key");
    assert_eq!(
        val, 9_000,
        "Redis identity key must equal the seeded DB total"
    );

    let global: i64 = redis::cmd("GET")
        .arg("oxigate:global:spend")
        .query_async(&mut *conn)
        .await
        .expect("GET global key");
    assert_eq!(
        global, 9_000,
        "Global counter must equal sum of all spend records",
    );
}

/// monthly duration seeds the period-suffixed key, not the legacy unprefixed key.
#[tokio::test]
async fn test_seed_redis_period_keyed_monthly() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    sqlx::query(
        "INSERT INTO spend_records (org_id, identity_id, model, provider, cost_nano_usd) \
         VALUES ('s-org', 's-id', 'gpt-4', 'openai', 9000)",
    )
    .execute(&pg.pool)
    .await
    .expect("direct insert");

    let aggregate_since = Utc::now() - chrono::Duration::hours(1);
    oxigate::db::spend_writer::seed_redis_from_db(
        &pg.pool,
        &redis.pool,
        aggregate_since,
        BudgetDuration::Monthly,
        UTC,
        false,
    )
    .await;

    let period = oxigate::utils::period_key(BudgetDuration::Monthly, Utc::now(), UTC);
    let keyed = format!("oxigate:org:s-org:spend:s-id:{period}");
    let legacy = "oxigate:org:s-org:spend:s-id";

    let mut conn = redis.pool.get().await.expect("redis conn");
    let val: i64 = redis::cmd("GET")
        .arg(&keyed)
        .query_async(&mut *conn)
        .await
        .expect("GET period key");
    assert_eq!(val, 9_000);

    let legacy_exists: u8 = redis::cmd("EXISTS")
        .arg(legacy)
        .query_async(&mut *conn)
        .await
        .expect("EXISTS legacy");
    assert_eq!(
        legacy_exists, 0,
        "unprefixed key must not be set for period mode"
    );
}

/// spec #5 — `write_spend` uses period-suffixed key + monthly TTL (~62 days).
#[tokio::test]
async fn test_write_spend_monthly_period_key_and_ttl() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    let pool = Arc::new(RwLock::new(pg.pool.clone()));
    let rp = Arc::new(RwLock::new(redis.pool.clone()));

    let fixed_now = Utc.with_ymd_and_hms(2026, 3, 15, 12, 0, 0).unwrap();
    oxigate::db::spend_writer::write_spend(
        make_record("org-m", "id-m", 2_000),
        Arc::clone(&pool),
        Arc::clone(&rp),
        BudgetDuration::Monthly,
        UTC,
        fixed_now,
    )
    .await;

    let period = oxigate::utils::period_key(BudgetDuration::Monthly, fixed_now, UTC);
    assert_eq!(period, "2026-03");
    let key = format!("oxigate:org:org-m:spend:id-m:{period}");

    let mut conn = redis.pool.get().await.expect("redis conn");
    let val: i64 = redis::cmd("GET")
        .arg(&key)
        .query_async(&mut *conn)
        .await
        .expect("GET period key");
    assert_eq!(val, 2_000);

    let ttl: i64 = redis::cmd("TTL")
        .arg(&key)
        .query_async(&mut *conn)
        .await
        .expect("TTL");
    let expected: i64 = 62 * 24 * 3600;
    assert!(
        (ttl - expected).abs() <= 5,
        "monthly per-identity TTL should be ~{expected}s, got {ttl}"
    );
}

/// P1: explicit budget_reset_at uses unprefixed key during seed.
#[tokio::test]
async fn test_seed_redis_explicit_reset_at_unprefixed() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    sqlx::query(
        "INSERT INTO spend_records (org_id, identity_id, model, provider, cost_nano_usd) \
         VALUES ('s-org', 's-id', 'gpt-4', 'openai', 9000)",
    )
    .execute(&pg.pool)
    .await
    .expect("direct insert");

    let aggregate_since = Utc::now() - chrono::Duration::hours(1);
    oxigate::db::spend_writer::seed_redis_from_db(
        &pg.pool,
        &redis.pool,
        aggregate_since,
        BudgetDuration::Monthly,
        UTC,
        true,
    )
    .await;

    let legacy = "oxigate:org:s-org:spend:s-id";
    let mut conn = redis.pool.get().await.expect("redis conn");
    let val: i64 = redis::cmd("GET")
        .arg(legacy)
        .query_async(&mut *conn)
        .await
        .expect("GET legacy key");
    assert_eq!(val, 9_000);

    let period = oxigate::utils::period_key(BudgetDuration::Monthly, Utc::now(), UTC);
    let keyed = format!("oxigate:org:s-org:spend:s-id:{period}");
    let period_exists: u8 = redis::cmd("EXISTS")
        .arg(&keyed)
        .query_async(&mut *conn)
        .await
        .expect("EXISTS period key");
    assert_eq!(period_exists, 0);
}

/// when the global Redis key is corrupted (non-integer), write_spend still
/// increments the per-identity counter correctly and completes without panicking.
/// Verifies the split-pipeline best-effort contract for the global write.
#[tokio::test]
async fn test_write_spend_global_key_corruption_fails_open() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    // Corrupt the global key — INCRBY on a non-integer string returns a Redis error.
    {
        let mut conn = redis.pool.get().await.expect("redis conn");
        redis::cmd("SET")
            .arg("oxigate:global:spend")
            .arg("not-an-integer")
            .query_async::<()>(&mut *conn)
            .await
            .expect("SET corrupt global key");
    }

    let pool = Arc::new(RwLock::new(pg.pool.clone()));
    let rp = Arc::new(RwLock::new(redis.pool.clone()));
    oxigate::db::spend_writer::write_spend(
        make_record("p-org", "p-id", 5_000),
        pool,
        rp,
        BudgetDuration::None,
        UTC,
        Utc::now(),
    )
    .await;

    let mut conn = redis.pool.get().await.expect("redis conn");

    // Per-identity counter must be written correctly despite the global write failing.
    let identity_val: i64 = redis::cmd("GET")
        .arg("oxigate:org:p-org:spend:p-id")
        .query_async(&mut *conn)
        .await
        .expect("GET identity key");
    assert_eq!(
        identity_val, 5_000,
        "per-identity counter must be incremented even when global write fails",
    );

    // Global key must remain corrupted — our failed INCRBY must not have changed it.
    let global_raw: String = redis::cmd("GET")
        .arg("oxigate:global:spend")
        .query_async(&mut *conn)
        .await
        .expect("GET global key");
    assert_eq!(
        global_raw, "not-an-integer",
        "corrupted global key must be unchanged after failed INCRBY",
    );
}

/// write_spend with a broken Redis pool still inserts the DB row (no panic).
#[tokio::test]
async fn test_write_spend_redis_failure_still_inserts_db() {
    let pg = PgContainer::start().await.expect("pg container");

    // Bad Redis pool — unreachable URL.
    let bad_redis_cfg = oxigate::config::RedisConfig {
        url: oxigate::config::SecretString::new("redis://127.0.0.1:19999"),
        pool_size: Some(1),
        pool_timeout_secs: Some(1),
    };
    let bad_pool = create_pool(&bad_redis_cfg).expect("pool struct created");

    let pool = Arc::new(RwLock::new(pg.pool.clone()));
    let rp = Arc::new(RwLock::new(bad_pool));

    let record = make_record("fail-org", "fail-id", 7_777);
    // Must not panic even with broken Redis.
    oxigate::db::spend_writer::write_spend(record, pool, rp, BudgetDuration::None, UTC, Utc::now())
        .await;

    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM spend_records WHERE org_id = 'fail-org'")
            .fetch_one(&pg.pool)
            .await
            .expect("count query");
    assert_eq!(
        count, 1,
        "DB row must be inserted even when Redis is unavailable"
    );
}

/// spend_records.org_id matches identity.org_id after write_spend.
#[tokio::test]
async fn test_write_spend_includes_org_id() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    let pool = Arc::new(RwLock::new(pg.pool.clone()));
    let rp = Arc::new(RwLock::new(redis.pool.clone()));

    let record = make_record("my-org", "my-id", 999);
    oxigate::db::spend_writer::write_spend(record, pool, rp, BudgetDuration::None, UTC, Utc::now())
        .await;

    let org_id: String =
        sqlx::query_scalar("SELECT org_id FROM spend_records WHERE identity_id = 'my-id'")
            .fetch_one(&pg.pool)
            .await
            .expect("row");
    assert_eq!(org_id, "my-org");
}

/// write_spend with PG down must not panic; Redis counter IS written
/// because Redis is first in the execution order. Symmetric twin of
/// the Redis-down test (Redis down → DB row still written).
#[tokio::test]
async fn test_write_spend_pg_failure_redis_still_written() {
    let redis = RedisContainer::start().await.expect("redis container");

    // Lazy pool pointing at an unreachable PG — creation succeeds, first query fails.
    let bad_pool = sqlx::postgres::PgPoolOptions::new()
        .acquire_timeout(std::time::Duration::from_millis(100))
        .connect_lazy("postgresql://bad:bad@127.0.0.1:19999/bad")
        .expect("lazy pool must be constructable");

    let pool = Arc::new(RwLock::new(bad_pool));
    let rp = Arc::new(RwLock::new(redis.pool.clone()));

    let record = make_record("pg-fail-org", "pg-fail-id", 5_000);
    // Must not panic even with broken PG.
    oxigate::db::spend_writer::write_spend(record, pool, rp, BudgetDuration::None, UTC, Utc::now())
        .await;

    // Redis INCRBY ran before the PG attempt — counter must still be present.
    let key = "oxigate:org:pg-fail-org:spend:pg-fail-id";
    let mut conn = redis.pool.get().await.expect("redis conn");
    let val: i64 = redis::cmd("GET")
        .arg(key)
        .query_async(&mut *conn)
        .await
        .expect("GET");
    assert_eq!(
        val, 5_000,
        "Redis counter must be written even when PG is down",
    );
}

/// write_spend dual-writes to the global spend key `oxigate:global:spend`.
/// Verifies that the global counter is incremented alongside the per-identity counter.
#[tokio::test]
async fn test_write_spend_increments_global_key() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    let pool = Arc::new(RwLock::new(pg.pool.clone()));
    let rp = Arc::new(RwLock::new(redis.pool.clone()));

    let record = make_record("acme", "key-global", 1_000_000);
    oxigate::db::spend_writer::write_spend(record, pool, rp, BudgetDuration::None, UTC, Utc::now())
        .await;

    let mut conn = redis.pool.get().await.expect("redis conn");
    let val: i64 = redis::cmd("GET")
        .arg("oxigate:global:spend")
        .query_async(&mut *conn)
        .await
        .expect("GET global spend key");
    assert_eq!(
        val, 1_000_000,
        "global spend counter must equal the write cost",
    );
}

/// a streaming request whose provider emits two usage-bearing chunks
/// (Anthropic-style: message_start + message_delta both carry usage) must produce
/// exactly ONE spend_records row — not two.
///
/// Validates the EOF-only emit introduced to fix the streaming duplicate-write bug.
#[tokio::test]
async fn test_streaming_two_usage_chunks_produces_single_spend_record() {
    let pg = PgContainer::start().await.expect("pg container");
    let redis = RedisContainer::start().await.expect("redis container");

    // Simulate Anthropic: message_start carries input_tokens only; message_delta has final totals.
    let usage_start = Usage {
        prompt_tokens: 10,
        completion_tokens: 0,
        total_tokens: 10,
        ..Default::default()
    };
    let usage_final = Usage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
        ..Default::default()
    };
    let chunks = vec![
        Ok(StreamChunk::new(
            Bytes::from("data: {\"choices\":[]}\n\n"),
            Some(usage_start),
            Some("gpt-4.1".to_string()),
        )),
        Ok(StreamChunk::new(
            Bytes::from("data: {\"choices\":[]}\n\n"),
            Some(usage_final),
            Some("gpt-4.1".to_string()),
        )),
    ];

    let provider = Arc::new(StreamStubAdapter::new(chunks));
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::OK);

    // Allow the spawned write_spend task to complete before querying.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM spend_records")
        .fetch_one(&pg.pool)
        .await
        .expect("count query");
    assert_eq!(
        count, 1,
        "two usage-bearing chunks must produce exactly one spend_records row"
    );

    // Also verify the Redis counter was incremented exactly once.
    let key = "oxigate:org:default:spend:default";
    let mut conn = redis.pool.get().await.expect("redis conn");
    let redis_val: Option<i64> = redis::cmd("GET")
        .arg(key)
        .query_async(&mut *conn)
        .await
        .expect("GET");
    assert!(
        redis_val.is_some(),
        "Redis counter must exist after streaming spend write"
    );
    let db_cost: i64 = sqlx::query_scalar("SELECT cost_nano_usd FROM spend_records LIMIT 1")
        .fetch_one(&pg.pool)
        .await
        .expect("cost query");
    assert_eq!(
        redis_val.unwrap(),
        db_cost,
        "Redis counter must equal the single DB row's cost_nano_usd"
    );
}
