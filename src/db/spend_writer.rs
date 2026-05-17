// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Spend persistence: Redis INCRBY counter + Postgres audit row .
//!
//! Two public functions:
//! - `write_spend` — called after every completed provider request (spawn-and-forget).
//! - `seed_redis_from_db` — called at startup to recover Redis counters from Postgres.

use std::sync::Arc;

use tokio::sync::RwLock;

use chrono::{DateTime, Utc};
use chrono_tz::Tz;

use crate::config::BudgetDuration;
use crate::db::DbPool;
use crate::domain::spend::SpendRecord;
use crate::redis_pool::RedisPool;
use crate::utils::{
    identity_spend_key, period_key, spend_key_ttl_secs, tag_spend_key, team_spend_key,
};

/// Redis TTL for the **global** spend counter only​.
///
/// The per-identity key uses [`spend_key_ttl_secs`]. This key is a lifetime-cumulative
/// instance circuit breaker — it is **not** reset on `budget_duration` cadence.
const SPEND_KEY_TTL_SECS: u64 = 60 * 24 * 3600; // 5_184_000

/// Redis key for the instance-wide global spend counter.
///
/// Written by every `write_spend` call as part of the same pipeline as the per-identity
/// key. Global counter is best-effort — if this write fails, the per-identity counter
/// is still the source of truth.
///
/// P2: **not** period-keyed and **not** subject to `budget_duration` resets.
pub(crate) const GLOBAL_SPEND_KEY: &str = "oxigate:global:spend";

/// Write a spend record: Redis INCRBY first, then Postgres INSERT.
///
/// Both operations are best-effort. Errors are logged at ERROR level and swallowed —
/// spend writes must never fail in-band requests. The spawn-and-forget caller in
/// `chat.rs` expects this function to never panic.
///
/// Redis key format: `oxigate:org:{org_id}:spend:{identity_id}` or with `:{period}` .
pub async fn write_spend(
    record: SpendRecord,
    pool: Arc<RwLock<DbPool>>,
    redis_pool: Arc<RwLock<RedisPool>>,
    duration: BudgetDuration,
    tz: Tz,
    now: DateTime<Utc>,
) {
    let period = period_key(duration, now, tz);
    let redis_key = identity_spend_key(&record.org_id, &record.identity_id, &period);
    let identity_ttl = spend_key_ttl_secs(duration);

    // --- Redis: INCRBY + EXPIRE (atomic pipeline) ---
    {
        let rp = redis_pool.read().await.clone();
        match rp.get().await {
            Err(e) => {
                tracing::error!(
                    error = %e,
                    key = %redis_key,
                    "spend_writer: failed to acquire Redis connection; skipping Redis update"
                );
            }
            Ok(mut conn) => {
                // Pipeline 1: per-identity INCRBY + EXPIRE (critical path).
                let mut pipe = redis::pipe();
                pipe.cmd("INCRBY")
                    .arg(&redis_key)
                    .arg(record.cost_nano_usd.as_i64())
                    .ignore()
                    .cmd("EXPIRE")
                    .arg(&redis_key)
                    .arg(identity_ttl)
                    .ignore();
                if let Err(e) = pipe.query_async::<()>(&mut *conn).await {
                    tracing::error!(
                        error = %e,
                        key = %redis_key,
                        "spend_writer: identity Redis pipeline failed"
                    );
                }

                // Pipeline 2: global counter — best-effort; per-identity is source of truth.
                // Split from the identity pipeline so failures are distinguishable in logs and alerts.
                // GlobalSafetyLayer may temporarily under-count if this write fails — that is safe (fail-open).
                let mut global_pipe = redis::pipe();
                global_pipe
                    .cmd("INCRBY")
                    .arg(GLOBAL_SPEND_KEY)
                    .arg(record.cost_nano_usd.as_i64())
                    .ignore()
                    .cmd("EXPIRE")
                    .arg(GLOBAL_SPEND_KEY)
                    .arg(SPEND_KEY_TTL_SECS)
                    .ignore();
                if let Err(e) = global_pipe.query_async::<()>(&mut *conn).await {
                    tracing::warn!(
                        event = "global_spend_write_failed",
                        error = %e,
                        "spend_writer: global Redis counter write failed; GlobalSafetyLayer may under-count until recovery"
                    );
                }

                // Pipeline 3: team + tag INCRBYs — best-effort, fail-open .
                // Incremented unconditionally (even when no budget is configured) so spend
                // data is available for future query APIs. Each key is independent — partial
                // Redis failure is logged per-key and does not abort the other increments.
                // WARNING: a "team" tag (k="team", v="engineering") also produces a tag entry
                // tag:team:engineering:spend. Operators must not configure tag_budgets with
                // "team:*" keys alongside teams entries to avoid double-counting.
                if let Some(obj) = record.tags.as_object() {
                    let mut team_tag_pipe = redis::pipe();
                    let mut pipe_used = false;
                    let mut keys_written: Vec<String> = Vec::new();
                    if let Some(serde_json::Value::String(team)) = obj.get("team") {
                        let key = team_spend_key(&record.org_id, team, &period);
                        team_tag_pipe
                            .cmd("INCRBY")
                            .arg(&key)
                            .arg(record.cost_nano_usd.as_i64())
                            .ignore()
                            .cmd("EXPIRE")
                            .arg(&key)
                            .arg(identity_ttl)
                            .ignore();
                        keys_written.push(key);
                        pipe_used = true;
                    }
                    for (k, v) in obj {
                        if let Some(v_str) = v.as_str() {
                            let kv = format!("{k}:{v_str}");
                            let key = tag_spend_key(&record.org_id, &kv, &period);
                            team_tag_pipe
                                .cmd("INCRBY")
                                .arg(&key)
                                .arg(record.cost_nano_usd.as_i64())
                                .ignore()
                                .cmd("EXPIRE")
                                .arg(&key)
                                .arg(identity_ttl)
                                .ignore();
                            keys_written.push(key);
                            pipe_used = true;
                        }
                    }
                    if pipe_used && let Err(e) = team_tag_pipe.query_async::<()>(&mut *conn).await {
                        tracing::warn!(
                            event = "spend_writer_team_tag_redis_error",
                            error = %e,
                            keys = ?keys_written,
                            "spend_writer: team/tag Redis pipeline failed; counters may be stale"
                        );
                    }
                }
            }
        }
    }

    // --- Postgres: INSERT ---
    {
        let db = pool.read().await.clone();
        let result = sqlx::query(
            r#"
            INSERT INTO spend_records
                (org_id, identity_id, model, provider,
                 prompt_tokens, completion_tokens, cache_read_tokens,
                 cache_write_5m_tokens, cache_write_1h_tokens, thinking_tokens,
                 cost_nano_usd, latency_ms, tags)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            "#,
        )
        .bind(&record.org_id)
        .bind(&record.identity_id)
        .bind(&record.model)
        .bind(&record.provider)
        .bind(record.prompt_tokens)
        .bind(record.completion_tokens)
        .bind(record.cache_read_tokens)
        .bind(record.cache_write_5m_tokens)
        .bind(record.cache_write_1h_tokens)
        .bind(record.thinking_tokens)
        .bind(record.cost_nano_usd.as_i64())
        .bind(record.latency_ms)
        .bind(&record.tags)
        .execute(&db)
        .await;

        if let Err(e) = result {
            tracing::error!(
                error = %e,
                org_id = %record.org_id,
                identity_id = %record.identity_id,
                "spend_writer: Postgres INSERT failed"
            );
        }
    }
}

/// Seed Redis spend counters from Postgres aggregates on startup (crash recovery).
///
/// Queries the total spend per (org_id, identity_id) since `aggregate_since` and
/// writes each total to Redis with a TTL from [`spend_key_ttl_secs`]. This recovers
/// counters that were lost when the gateway restarted (Redis is ephemeral; Postgres is durable).
///
/// Both `pool` and `redis_pool` are borrowed directly (not `Arc`-wrapped). The
/// call site is expected to clone or read-lock the Arc before calling this function,
/// yielding an owned value that is then passed by reference here.
///
/// Errors are logged at WARN level; startup continues even if seeding fails.
pub async fn seed_redis_from_db(
    pool: &DbPool,
    redis_pool: &RedisPool,
    aggregate_since: chrono::DateTime<chrono::Utc>,
    duration: BudgetDuration,
    tz: Tz,
    budget_reset_at_is_explicit: bool,
) {
    // Step 1: collect distinct org_ids.
    let org_ids: Vec<String> =
        match sqlx::query_scalar::<_, String>("SELECT DISTINCT org_id FROM spend_records")
            .fetch_all(pool)
            .await
        {
            Ok(rows) => rows,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "seed_redis_from_db: failed to query distinct org_ids; skipping Redis seeding"
                );
                return;
            }
        };

    let period_suffix = if budget_reset_at_is_explicit {
        String::new()
    } else {
        period_key(duration, Utc::now(), tz)
    };
    let identity_ttl = spend_key_ttl_secs(duration);

    if org_ids.is_empty() {
        tracing::info!(
            aggregate_since = %aggregate_since,
            org_count = 0,
            "seed_redis_from_db: Redis spend counters seeded from Postgres"
        );
        return;
    }

    // Step 2: for each org, aggregate per-identity spend since aggregate_since.
    for org_id in &org_ids {
        let rows: Vec<(String, String, i64)> = match sqlx::query_as::<_, (String, String, i64)>(
            r#"
            SELECT org_id, identity_id, SUM(cost_nano_usd)::BIGINT AS total
            FROM spend_records
            WHERE org_id = $1 AND created_at >= $2
            GROUP BY org_id, identity_id
            "#,
        )
        .bind(org_id)
        .bind(aggregate_since)
        .fetch_all(pool)
        .await
        {
            Ok(rows) => rows,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    org_id = %org_id,
                    "seed_redis_from_db: failed to aggregate spend for org; skipping"
                );
                continue;
            }
        };

        for (org, identity, total) in rows {
            let key = identity_spend_key(&org, &identity, &period_suffix);
            match redis_pool.get().await {
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        key = %key,
                        "seed_redis_from_db: failed to acquire Redis connection"
                    );
                    continue;
                }
                Ok(mut conn) => {
                    // SET + EXPIRE in a single pipeline — consistent with write_spend and
                    // crash-safe: a process killed between SET and EXPIRE would leave a key
                    // with no TTL; the pipeline issues both commands atomically.
                    let mut pipe = redis::pipe();
                    pipe.cmd("SET")
                        .arg(&key)
                        .arg(total)
                        .ignore()
                        .cmd("EXPIRE")
                        .arg(&key)
                        .arg(identity_ttl)
                        .ignore();
                    if let Err(e) = pipe.query_async::<()>(&mut *conn).await {
                        tracing::warn!(
                            error = %e,
                            key = %key,
                            "seed_redis_from_db: SET+EXPIRE pipeline failed"
                        );
                    }
                }
            }
        }
    }

    // Step 3: seed the instance-wide global counter from a single aggregate query.
    // Best-effort — failure is logged at WARN and startup continues.
    let global_total: Option<i64> = match sqlx::query_scalar::<_, Option<i64>>(
        "SELECT SUM(cost_nano_usd)::BIGINT FROM spend_records WHERE created_at >= $1",
    )
    .bind(aggregate_since)
    .fetch_one(pool)
    .await
    {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "seed_redis_from_db: failed to aggregate global spend; global counter not seeded"
            );
            None
        }
    };

    if let Some(total) = global_total.filter(|&t| t > 0) {
        match redis_pool.get().await {
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "seed_redis_from_db: failed to acquire Redis connection for global counter seeding"
                );
            }
            Ok(mut conn) => {
                let mut pipe = redis::pipe();
                pipe.cmd("SET")
                    .arg(GLOBAL_SPEND_KEY)
                    .arg(total)
                    .ignore()
                    .cmd("EXPIRE")
                    .arg(GLOBAL_SPEND_KEY)
                    .arg(SPEND_KEY_TTL_SECS)
                    .ignore();
                if let Err(e) = pipe.query_async::<()>(&mut *conn).await {
                    tracing::warn!(
                        error = %e,
                        "seed_redis_from_db: global counter SET+EXPIRE pipeline failed"
                    );
                }
            }
        }
    }

    tracing::info!(
        aggregate_since = %aggregate_since,
        org_count = org_ids.len(),
        "seed_redis_from_db: Redis spend counters seeded from Postgres"
    );
}
