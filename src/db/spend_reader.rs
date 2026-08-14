// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Spend read queries: daily, by-provider, by-model aggregations .
//!
//! Each function accepts a time window as two exclusive `DateTime<Utc>` bounds
//! (`from_dt` inclusive, `to_dt` exclusive) so callers control the boundary math.
//! The handlers in `api::spend` convert user-supplied `NaiveDate` values into these
//! bounds before calling here.

use chrono::{DateTime, NaiveDate, Utc};
use sqlx::PgPool;

/// A single day's aggregated spend for an org.
#[derive(Debug, sqlx::FromRow)]
pub struct SpendDailyRow {
    pub date: NaiveDate,
    pub cost_nano_usd: i64,
}

/// Aggregated spend for a single named dimension (provider or model).
#[derive(Debug, sqlx::FromRow)]
pub struct SpendDimensionRow {
    pub dimension: String,
    pub cost_nano_usd: i64,
}

/// Aggregate spend per calendar day (UTC) for `org_id` in `[from_dt, to_dt)`.
///
/// Rows are ordered ascending by date. Days with zero spend are absent (no group).
pub async fn query_daily_spend(
    pool: &PgPool,
    org_id: &str,
    from_dt: DateTime<Utc>,
    to_dt: DateTime<Utc>,
) -> Result<Vec<SpendDailyRow>, sqlx::Error> {
    sqlx::query_as::<_, SpendDailyRow>(
        r#"
        SELECT DATE_TRUNC('day', created_at)::DATE AS date,
               SUM(cost_nano_usd)::BIGINT          AS cost_nano_usd
        FROM spend_records
        WHERE org_id    = $1
          AND created_at >= $2
          AND created_at <  $3
        GROUP BY 1
        ORDER BY 1
        "#,
    )
    .bind(org_id)
    .bind(from_dt)
    .bind(to_dt)
    .fetch_all(pool)
    .await
}

/// Aggregate spend per provider for `org_id` in `[from_dt, to_dt)`.
///
/// Rows are ordered ascending by provider name.
pub async fn query_spend_by_provider(
    pool: &PgPool,
    org_id: &str,
    from_dt: DateTime<Utc>,
    to_dt: DateTime<Utc>,
) -> Result<Vec<SpendDimensionRow>, sqlx::Error> {
    sqlx::query_as::<_, SpendDimensionRow>(
        r#"
        SELECT provider                  AS dimension,
               SUM(cost_nano_usd)::BIGINT AS cost_nano_usd
        FROM spend_records
        WHERE org_id    = $1
          AND created_at >= $2
          AND created_at <  $3
        GROUP BY 1
        ORDER BY 1
        "#,
    )
    .bind(org_id)
    .bind(from_dt)
    .bind(to_dt)
    .fetch_all(pool)
    .await
}

/// Aggregate spend per model for `org_id` in `[from_dt, to_dt)`.
///
/// Rows are ordered ascending by model name.
pub async fn query_spend_by_model(
    pool: &PgPool,
    org_id: &str,
    from_dt: DateTime<Utc>,
    to_dt: DateTime<Utc>,
) -> Result<Vec<SpendDimensionRow>, sqlx::Error> {
    sqlx::query_as::<_, SpendDimensionRow>(
        r#"
        SELECT model                     AS dimension,
               SUM(cost_nano_usd)::BIGINT AS cost_nano_usd
        FROM spend_records
        WHERE org_id    = $1
          AND created_at >= $2
          AND created_at <  $3
        GROUP BY 1
        ORDER BY 1
        "#,
    )
    .bind(org_id)
    .bind(from_dt)
    .bind(to_dt)
    .fetch_all(pool)
    .await
}
