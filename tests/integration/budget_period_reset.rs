// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! period-keyed budget behaviour and lazy reset (requires `pro` + `test-hooks`).

use std::sync::Arc;

use axum::http::StatusCode;
use chrono::{TimeZone, Utc};
use chrono_tz::America::New_York;
use chrono_tz::UTC;

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::StubAdapter;
use oxigate::config::{AuthConfig, BudgetConfig, BudgetDuration};
use oxigate::utils::{CostHeader, identity_spend_key, period_key, spend_key_ttl_secs};

/// Spec #1 — current period spend reads zero when only a *previous* monthly key is populated.
#[tokio::test]
async fn fresh_monthly_period_sees_zero_spend_without_clock_travel() {
    let pg = PgContainer::start().await.expect("pg");
    let redis = RedisContainer::start().await.expect("redis");
    let budget = BudgetConfig {
        budget_duration: Some("30d".into()),
        timezone: "UTC".into(),
        soft_cap_usd: None,
        hard_cap_usd: Some(10.0),
        ..BudgetConfig::default()
    };
    let (gw, _) = TestGateway::spawn_with_budget_and_clock(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        AuthConfig::default(),
        budget,
        None,
    )
    .await;

    let old_instant = Utc::now() - chrono::Duration::days(45);
    let old_period = period_key(BudgetDuration::Monthly, old_instant, UTC);
    let stale_key = identity_spend_key("default", "default", &old_period);
    let mut conn = redis.pool.get().await.expect("redis");
    redis::cmd("SET")
        .arg(&stale_key)
        .arg(9_000_000_000_i64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed stale period key");

    let response = gw.server.get("/v1/models").await;
    response.assert_status(StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("10.000000"),
        "must ignore spend under a non-current period suffix"
    );
}

/// Spec #2 — monthly lazy rollover via `now_override` (log shape: `budget::tests`).
#[tokio::test]
async fn monthly_lazy_reset_via_now_override() {
    let pg = PgContainer::start().await.expect("pg");
    let redis = RedisContainer::start().await.expect("redis");
    let t_jan = Utc.with_ymd_and_hms(2026, 1, 15, 12, 0, 0).unwrap();
    let budget = BudgetConfig {
        budget_duration: Some("30d".into()),
        timezone: "UTC".into(),
        soft_cap_usd: None,
        hard_cap_usd: Some(10.0),
        ..BudgetConfig::default()
    };
    let (gw, rt) = TestGateway::spawn_with_budget_and_clock(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        AuthConfig::default(),
        budget,
        Some(t_jan),
    )
    .await;
    {
        let mut w = rt.write().await;
        w.next_reset_at = Utc.with_ymd_and_hms(2026, 2, 1, 0, 0, 0).unwrap();
    }

    let p = period_key(BudgetDuration::Monthly, t_jan, UTC);
    assert_eq!(p, "2026-01");
    let sk = identity_spend_key("default", "default", &p);
    let mut conn = redis.pool.get().await.expect("redis");
    redis::cmd("SET")
        .arg(&sk)
        .arg(5_000_000_000_i64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed jan spend");

    let r1 = gw.server.get("/v1/models").await;
    r1.assert_status(StatusCode::OK);
    assert_eq!(
        r1.headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("5.000000")
    );

    rt.write().await.now_override = Some(Utc.with_ymd_and_hms(2026, 2, 2, 12, 0, 0).unwrap());

    let r2 = gw.server.get("/v1/models").await;
    r2.assert_status(StatusCode::OK);
    assert_eq!(
        r2.headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("10.000000")
    );
}

/// Spec #3 — weekly cadence: crossing Monday midnight starts a new week bucket.
#[tokio::test]
async fn weekly_lazy_reset_monday_boundary() {
    let pg = PgContainer::start().await.expect("pg");
    let redis = RedisContainer::start().await.expect("redis");
    let t_sun = Utc.with_ymd_and_hms(2026, 3, 22, 12, 0, 0).unwrap();
    let budget = BudgetConfig {
        budget_duration: Some("7d".into()),
        timezone: "UTC".into(),
        soft_cap_usd: None,
        hard_cap_usd: Some(10.0),
        ..BudgetConfig::default()
    };
    let (gw, rt) = TestGateway::spawn_with_budget_and_clock(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        AuthConfig::default(),
        budget,
        Some(t_sun),
    )
    .await;
    {
        let mut w = rt.write().await;
        w.next_reset_at = Utc.with_ymd_and_hms(2026, 3, 23, 0, 0, 0).unwrap();
    }

    let p = period_key(BudgetDuration::Weekly, t_sun, UTC);
    let sk = identity_spend_key("default", "default", &p);
    let mut conn = redis.pool.get().await.expect("redis");
    redis::cmd("SET")
        .arg(&sk)
        .arg(8_000_000_000_i64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed week spend");

    let r1 = gw.server.get("/v1/models").await;
    r1.assert_status(StatusCode::OK);
    assert_eq!(
        r1.headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("2.000000")
    );

    rt.write().await.now_override = Some(Utc.with_ymd_and_hms(2026, 3, 24, 12, 0, 0).unwrap());

    let r2 = gw.server.get("/v1/models").await;
    r2.assert_status(StatusCode::OK);
    assert_eq!(
        r2.headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("10.000000")
    );
}

/// Spec #4 — `US/Eastern` daily boundary (matches `period_key` unit scenario).
#[tokio::test]
async fn daily_lazy_reset_us_eastern_midnight() {
    let pg = PgContainer::start().await.expect("pg");
    let redis = RedisContainer::start().await.expect("redis");
    let t_before = Utc.with_ymd_and_hms(2026, 11, 20, 4, 59, 59).unwrap();
    let budget = BudgetConfig {
        budget_duration: Some("1d".into()),
        timezone: "America/New_York".into(),
        soft_cap_usd: None,
        hard_cap_usd: Some(10.0),
        ..BudgetConfig::default()
    };
    let (gw, rt) = TestGateway::spawn_with_budget_and_clock(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        AuthConfig::default(),
        budget,
        Some(t_before),
    )
    .await;
    {
        let mut w = rt.write().await;
        w.next_reset_at = Utc.with_ymd_and_hms(2026, 11, 20, 5, 0, 0).unwrap();
    }

    let p = period_key(BudgetDuration::Daily, t_before, New_York);
    assert_eq!(p, "2026-11-19");
    let sk = identity_spend_key("default", "default", &p);
    let mut conn = redis.pool.get().await.expect("redis");
    redis::cmd("SET")
        .arg(&sk)
        .arg(3_000_000_000_i64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed day spend");

    let r1 = gw.server.get("/v1/models").await;
    r1.assert_status(StatusCode::OK);
    assert_eq!(
        r1.headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("7.000000")
    );

    rt.write().await.now_override = Some(Utc.with_ymd_and_hms(2026, 11, 20, 5, 0, 1).unwrap());

    let r2 = gw.server.get("/v1/models").await;
    r2.assert_status(StatusCode::OK);
    assert_eq!(
        r2.headers()
            .get(CostHeader::BUDGET_REMAINING)
            .and_then(|v| v.to_str().ok()),
        Some("10.000000")
    );
}

/// Spec #9 — scheduler `SET NX` does not clobber spend if the new-period key already exists.
#[tokio::test]
async fn budget_scheduler_set_nx_does_not_zero_live_spend() {
    let redis = RedisContainer::start().await.expect("redis");
    let now = Utc.with_ymd_and_hms(2026, 4, 10, 12, 0, 0).unwrap();
    let budget = BudgetConfig {
        budget_duration: Some("30d".into()),
        timezone: "UTC".into(),
        soft_cap_usd: Some(10.0),
        ..BudgetConfig::default()
    };
    let runtime = Arc::new(tokio::sync::RwLock::new({
        let mut r =
            oxigate::middleware::budget::BudgetRuntimeConfig::from_budget_config(budget.clone());
        r.now_override = Some(now);
        r.next_reset_at = Utc.with_ymd_and_hms(2026, 4, 1, 0, 0, 0).unwrap();
        r
    }));
    let sched = oxigate::middleware::budget_scheduler::BudgetResetScheduler::new(
        Arc::clone(&runtime),
        Arc::new(tokio::sync::RwLock::new(redis.pool.clone())),
    );

    let period = period_key(BudgetDuration::Monthly, now, UTC);
    let key = identity_spend_key("acme", "k1", &period);
    let ttl = spend_key_ttl_secs(BudgetDuration::Monthly);
    let mut conn = redis.pool.get().await.expect("redis");
    redis::cmd("SET")
        .arg(&key)
        .arg(4_000_000_000_i64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("pre-seed new-period key as if lazy reset ran first");

    let failures = sched.wake_cycle_for_test().await;
    assert_eq!(failures, 0);

    let v: i64 = redis::cmd("GET")
        .arg(&key)
        .query_async(&mut *conn)
        .await
        .expect("GET");
    assert_eq!(
        v, 4_000_000_000,
        "SET NX must not overwrite existing counter"
    );
    let t_live: i64 = redis::cmd("TTL")
        .arg(&key)
        .query_async(&mut *conn)
        .await
        .expect("TTL");
    assert!(
        (t_live - ttl as i64).abs() <= 5,
        "TTL refreshed (~{ttl}s), got {t_live}"
    );
}
