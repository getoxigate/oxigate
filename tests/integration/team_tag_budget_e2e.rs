// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for per-team and per-tag budget enforcement.
//!
//! All tests require Redis + Postgres containers.
//! Identity tags are injected via `X-OxiGate-Team` and `X-OxiGate-Project` headers.
//! Redis keys seeded directly mirror what `spend_writer` and `TeamTagBudgetLayer` produce.

use std::collections::HashMap;
use std::sync::Arc;

use axum::http::StatusCode;

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::StubAdapter;
use oxigate::api::CHAT_COMPLETIONS_PATH;
use oxigate::config::{AuthConfig, BudgetCapEntry, BudgetConfig};

// Default org key prefix (auth-disabled → org_id="default", identity_id="default").
const DEFAULT_IDENTITY_SPEND_KEY: &str = "oxigate:org:default:spend:default";
const DEFAULT_TEAM_ENG_SPEND_KEY: &str = "oxigate:org:default:team:engineering:spend";
const DEFAULT_TAG_PROJECT_CHATBOT_SPEND_KEY: &str =
    "oxigate:org:default:tag:project:chat-bot:spend";
const DEFAULT_TAG_PROJECT_ANALYTICS_SPEND_KEY: &str =
    "oxigate:org:default:tag:project:analytics:spend";

// ─── Helpers ────────────────────────────────────────────────────────────────

fn budget_with_team(team: &str, soft: Option<f64>, hard: Option<f64>) -> BudgetConfig {
    let mut teams = HashMap::new();
    teams.insert(
        team.to_string(),
        BudgetCapEntry {
            soft_cap_usd: soft,
            hard_cap_usd: hard,
        },
    );
    BudgetConfig {
        teams,
        ..BudgetConfig::default()
    }
}

fn budget_with_tag(kv: &str, soft: Option<f64>, hard: Option<f64>) -> BudgetConfig {
    let mut tag_budgets = HashMap::new();
    tag_budgets.insert(
        kv.to_string(),
        BudgetCapEntry {
            soft_cap_usd: soft,
            hard_cap_usd: hard,
        },
    );
    BudgetConfig {
        tag_budgets,
        ..BudgetConfig::default()
    }
}

// ─── Team budget tests ───────────────────────────────────────────────────────

/// AC: team hard cap breached → 429 + X-Oxigate-Budget-Remaining: 0.000000 + JSON error.
#[tokio::test]
async fn team_hard_cap_rejects_429() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());
    let budget = budget_with_team("engineering", None, Some(10.0));

    let (gateway, _) = TestGateway::spawn_with_team_tag_budgets(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    // Seed spend at exactly the hard cap (10 USD = 10_000_000_000 nano USD).
    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_TEAM_ENG_SPEND_KEY)
        .arg(10_000_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed team spend key");

    let response = gateway
        .server
        .get("/v1/models")
        .add_header("X-OxiGate-Team", "engineering")
        .await;

    response.assert_status(StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get("X-Oxigate-Budget-Remaining")
            .and_then(|v| v.to_str().ok()),
        Some("0.000000")
    );
    let body: serde_json::Value = serde_json::from_str(&response.text()).expect("JSON body");
    assert_eq!(body["error"], "team_budget_exceeded");
    assert_eq!(body["team"], "engineering");
}

/// AC: team soft cap crossed → request proceeds (200) (soft cap is non-blocking).
#[tokio::test]
async fn team_soft_cap_warn_proceeds() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());
    let budget = budget_with_team("engineering", Some(10.0), None);

    let (gateway, _) = TestGateway::spawn_with_team_tag_budgets(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    // Seed spend above the 80% threshold (9 USD > 80% of $10).
    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_TEAM_ENG_SPEND_KEY)
        .arg(9_000_000_000_u64) // $9 spent
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed team spend key");

    let response = gateway
        .server
        .get("/v1/models")
        .add_header("X-OxiGate-Team", "engineering")
        .await;

    response.assert_status(StatusCode::OK);
    assert!(
        response
            .headers()
            .get("X-Oxigate-Budget-Remaining")
            .is_none(),
        "X-Oxigate-Budget-Remaining must be omitted when no hard cap is configured"
    );
}

/// AC: request carries a team tag not in config → no budget check, request passes through.
#[tokio::test]
async fn team_no_budget_configured_unlimited() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());
    // Budget configured for "platform" team, not "engineering".
    let budget = budget_with_team("platform", None, Some(5.0));

    let (gateway, _) = TestGateway::spawn_with_team_tag_budgets(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    // No spend seeded — team "engineering" has no entry in config.
    let response = gateway
        .server
        .get("/v1/models")
        .add_header("X-OxiGate-Team", "engineering")
        .await;

    response.assert_status(StatusCode::OK);
}

/// AC: request has no team tag → TeamTagBudgetLayer is a no-op, request passes.
#[tokio::test]
async fn team_missing_tag_is_noop() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());
    let budget = budget_with_team("engineering", None, Some(10.0));

    let (gateway, _) = TestGateway::spawn_with_team_tag_budgets(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    // Seed spend at hard cap — but no team header on request.
    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_TEAM_ENG_SPEND_KEY)
        .arg(10_000_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed team spend key");

    // No X-OxiGate-Team header → budget check never runs.
    let response = gateway.server.get("/v1/models").await;
    response.assert_status(StatusCode::OK);
}

// ─── Tag budget tests ────────────────────────────────────────────────────────

/// AC: tag hard cap breached → 429 + X-Oxigate-Budget-Remaining: 0.000000 + JSON error.
#[tokio::test]
async fn tag_hard_cap_rejects_429() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());
    let budget = budget_with_tag("project:chat-bot", None, Some(10.0));

    let (gateway, _) = TestGateway::spawn_with_team_tag_budgets(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_TAG_PROJECT_CHATBOT_SPEND_KEY)
        .arg(10_000_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed tag spend key");

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("X-OxiGate-Project", "chat-bot")
        .json(&serde_json::json!({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .await;

    response.assert_status(StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get("X-Oxigate-Budget-Remaining")
            .and_then(|v| v.to_str().ok()),
        Some("0.000000")
    );
    let body: serde_json::Value = serde_json::from_str(&response.text()).expect("JSON body");
    assert_eq!(body["error"], "tag_budget_exceeded");
    assert_eq!(body["tag"], "project:chat-bot");
}

/// AC: tag soft cap crossed → request proceeds (200) (soft cap is non-blocking).
#[tokio::test]
async fn tag_soft_cap_warn_proceeds() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());
    let budget = budget_with_tag("project:chat-bot", Some(10.0), None);

    let (gateway, _) = TestGateway::spawn_with_team_tag_budgets(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_TAG_PROJECT_CHATBOT_SPEND_KEY)
        .arg(8_500_000_000_u64) // $8.5 spent — above 80% threshold
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed tag spend key");

    let response = gateway
        .server
        .get("/v1/models")
        .add_header("X-OxiGate-Project", "chat-bot")
        .await;

    response.assert_status(StatusCode::OK);
    assert!(
        response
            .headers()
            .get("X-Oxigate-Budget-Remaining")
            .is_none(),
        "X-Oxigate-Budget-Remaining must be omitted when no hard cap is configured"
    );
}

/// AC: request with two tag dimensions, one exhausted → 429 names the exhausted tag.
///
/// Uses `X-OxiGate-Team: eng` + `X-OxiGate-Project: analytics` to inject two tags.
/// `tag_budgets` has `"project:analytics"` (exhausted) and `"team:eng"` (not exhausted).
/// Tags sort alphabetically: `"project:analytics" < "team:eng"` → analytics checked first → 429.
#[tokio::test]
async fn multi_tag_most_restrictive_hard_cap_wins() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());

    // Note: both entries are in tag_budgets (not teams) — they match via the tag kv iteration.
    let mut tag_budgets = HashMap::new();
    tag_budgets.insert(
        "project:analytics".to_string(),
        BudgetCapEntry {
            soft_cap_usd: None,
            hard_cap_usd: Some(5.0),
        },
    );
    tag_budgets.insert(
        "team:eng".to_string(),
        BudgetCapEntry {
            soft_cap_usd: None,
            hard_cap_usd: Some(100.0),
        },
    );
    let budget = BudgetConfig {
        tag_budgets,
        ..BudgetConfig::default()
    };

    let (gateway, _) = TestGateway::spawn_with_team_tag_budgets(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    let mut conn = redis.pool.get().await.expect("redis conn");
    // "project:analytics" exhausted; "team:eng" well below cap.
    redis::cmd("SET")
        .arg(DEFAULT_TAG_PROJECT_ANALYTICS_SPEND_KEY)
        .arg(5_000_000_000_u64) // at hard cap
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed analytics tag spend key");
    redis::cmd("SET")
        .arg("oxigate:org:default:tag:team:eng:spend")
        .arg(1_000_000_000_u64) // $1 of $100
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed team:eng tag spend key");

    let response = gateway
        .server
        .get("/v1/models")
        .add_header("X-OxiGate-Team", "eng")
        .add_header("X-OxiGate-Project", "analytics")
        .await;

    response.assert_status(StatusCode::TOO_MANY_REQUESTS);
    let body: serde_json::Value = serde_json::from_str(&response.text()).expect("JSON body");
    assert_eq!(body["error"], "tag_budget_exceeded");
    assert_eq!(body["tag"], "project:analytics");
}

/// AC: X-Oxigate-Budget-Remaining reflects the most-restrictive remaining across team and tag hard caps.
#[tokio::test]
async fn x_budget_remaining_most_restrictive_of_team_and_tag() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());

    // Team hard cap $100, tag hard cap $20 — tag is more restrictive.
    let mut teams = HashMap::new();
    teams.insert(
        "engineering".to_string(),
        BudgetCapEntry {
            soft_cap_usd: None,
            hard_cap_usd: Some(100.0),
        },
    );
    let mut tag_budgets = HashMap::new();
    tag_budgets.insert(
        "project:chat-bot".to_string(),
        BudgetCapEntry {
            soft_cap_usd: None,
            hard_cap_usd: Some(20.0),
        },
    );
    let budget = BudgetConfig {
        teams,
        tag_budgets,
        ..BudgetConfig::default()
    };

    let (gateway, _) = TestGateway::spawn_with_team_tag_budgets(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    let mut conn = redis.pool.get().await.expect("redis conn");
    // Team: $80 spent → $20 remaining.
    redis::cmd("SET")
        .arg(DEFAULT_TEAM_ENG_SPEND_KEY)
        .arg(80_000_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed team spend key");
    // Tag: $18 spent → $2 remaining (more restrictive).
    redis::cmd("SET")
        .arg(DEFAULT_TAG_PROJECT_CHATBOT_SPEND_KEY)
        .arg(18_000_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed tag spend key");

    let response = gateway
        .server
        .get("/v1/models")
        .add_header("X-OxiGate-Team", "engineering")
        .add_header("X-OxiGate-Project", "chat-bot")
        .await;

    response.assert_status(StatusCode::OK);
    let remaining = response
        .headers()
        .get("X-Oxigate-Budget-Remaining")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok())
        .expect("X-Oxigate-Budget-Remaining must parse");
    // Most restrictive: tag remaining = $2.0
    assert!(
        (remaining - 2.0).abs() < 0.01,
        "X-Oxigate-Budget-Remaining should be ~$2.0 (tag is most restrictive), got {remaining}"
    );
}

/// Team and tag both exhausted — verify team reported first (sort order: team < tags).
#[tokio::test]
async fn team_and_tag_exhausted_reports_team_first() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());

    let mut teams = HashMap::new();
    teams.insert(
        "engineering".to_string(),
        BudgetCapEntry {
            soft_cap_usd: None,
            hard_cap_usd: Some(10.0),
        },
    );
    let mut tag_budgets = HashMap::new();
    tag_budgets.insert(
        "project:chat-bot".to_string(),
        BudgetCapEntry {
            soft_cap_usd: None,
            hard_cap_usd: Some(10.0),
        },
    );
    let budget = BudgetConfig {
        teams,
        tag_budgets,
        ..BudgetConfig::default()
    };

    let (gateway, _) = TestGateway::spawn_with_team_tag_budgets(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    let mut conn = redis.pool.get().await.expect("redis conn");
    // Exhaust both team and tag budgets (10 USD = 10_000_000_000 nano USD).
    redis::cmd("SET")
        .arg(DEFAULT_TEAM_ENG_SPEND_KEY)
        .arg(10_000_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed team spend key");
    redis::cmd("SET")
        .arg(DEFAULT_TAG_PROJECT_CHATBOT_SPEND_KEY)
        .arg(10_000_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed tag spend key");

    let response = gateway
        .server
        .get("/v1/models")
        .add_header("X-OxiGate-Team", "engineering")
        .add_header("X-OxiGate-Project", "chat-bot")
        .await;

    response.assert_status(StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get("X-Oxigate-Budget-Remaining")
            .and_then(|v| v.to_str().ok()),
        Some("0.000000")
    );
    let body: serde_json::Value = serde_json::from_str(&response.text()).expect("JSON body");
    assert_eq!(body["error"], "team_budget_exceeded");
    assert_eq!(body["team"], "engineering");
}

/// AC: Redis unavailable → request passes through (fail-open).
#[tokio::test]
async fn redis_unavailable_fail_open() {
    use oxigate::config::{RedisConfig, SecretString};
    use oxigate::redis_pool::create_pool;

    let pg = PgContainer::start().await.expect("pg container must start");
    let bad_redis_cfg = RedisConfig {
        url: SecretString::new("redis://127.0.0.1:19999"),
        pool_size: Some(1),
        pool_timeout_secs: Some(1),
    };
    let bad_redis = create_pool(&bad_redis_cfg).expect("pool struct should still construct");
    let provider = Arc::new(StubAdapter::new());
    let budget = budget_with_team("engineering", None, Some(10.0));

    let (gateway, _) = TestGateway::spawn_with_team_tag_budgets(
        pg.pool.clone(),
        bad_redis,
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    // Even though this team is "over budget" in theory, Redis is unreachable → fail-open.
    let response = gateway
        .server
        .get("/v1/models")
        .add_header("X-OxiGate-Team", "engineering")
        .await;

    response.assert_status(StatusCode::OK);
}

/// AC: spend exactly at hard_cap_nano_usd boundary → 429 (>= check, not >).
#[tokio::test]
async fn team_hard_cap_exact_boundary_rejects() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());
    let budget = budget_with_team("engineering", None, Some(5.0));

    let (gateway, _) = TestGateway::spawn_with_team_tag_budgets(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    // Seed spend at exactly hard cap (5 USD = 5_000_000_000 nano USD).
    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_TEAM_ENG_SPEND_KEY)
        .arg(5_000_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed team spend key");

    let response = gateway
        .server
        .get("/v1/models")
        .add_header("X-OxiGate-Team", "engineering")
        .await;

    // At exactly hard_cap → rejected (spend >= hard_cap).
    response.assert_status(StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get("X-Oxigate-Budget-Remaining")
            .and_then(|v| v.to_str().ok()),
        Some("0.000000")
    );
}

/// AC: hot-reload via Arc swap — updated team budget is enforced without restart.
#[tokio::test]
async fn config_reload_team_tag_budgets() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());

    // Initial config: no budget for "engineering".
    let initial_budget = BudgetConfig::default();

    let (gateway, config_handle) = TestGateway::spawn_with_team_tag_budgets(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        AuthConfig::default(),
        initial_budget,
    )
    .await;

    // First request — no budget configured → passes regardless of spend.
    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_TEAM_ENG_SPEND_KEY)
        .arg(10_000_000_000_u64) // would be over a $10 hard cap
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed team spend key");

    let first = gateway
        .server
        .get("/v1/models")
        .add_header("X-OxiGate-Team", "engineering")
        .await;
    first.assert_status(StatusCode::OK);

    // Hot-reload: swap in a $10 hard cap for "engineering".
    {
        let mut teams = HashMap::new();
        teams.insert(
            "engineering".to_string(),
            BudgetCapEntry {
                soft_cap_usd: None,
                hard_cap_usd: Some(10.0),
            },
        );
        let updated = BudgetConfig {
            teams,
            ..BudgetConfig::default()
        };
        *config_handle.write().await = updated;
    }

    // Second request — new config enforced → 429.
    let second = gateway
        .server
        .get("/v1/models")
        .add_header("X-OxiGate-Team", "engineering")
        .await;
    second.assert_status(StatusCode::TOO_MANY_REQUESTS);
    let body: serde_json::Value = serde_json::from_str(&second.text()).expect("JSON body");
    assert_eq!(body["error"], "team_budget_exceeded");
}

/// N-4: Most-restrictive-wins when BOTH the per-identity BudgetLayer AND the
/// TeamTagBudgetLayer are active simultaneously.
///
/// Per-identity hard cap = $10 (→ $1 remaining after $9 seeded).
/// Team hard cap         = $100 (→ $20 remaining after $80 seeded).
/// Expected X-Oxigate-Budget-Remaining ≈ $1.0 — the identity cap is more restrictive and wins.
#[tokio::test]
async fn x_budget_remaining_most_restrictive_across_identity_and_team_layers() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());

    let mut teams = HashMap::new();
    teams.insert(
        "engineering".to_string(),
        BudgetCapEntry {
            soft_cap_usd: None,
            hard_cap_usd: Some(100.0),
        },
    );
    // spawn_with_budget sets both `budget_settings` (TeamTagBudgetLayer) and
    // `budget` / BudgetRuntimeConfig (per-identity BudgetLayer + HardCapLayer).
    let budget = BudgetConfig {
        soft_cap_usd: None,
        hard_cap_usd: Some(10.0), // per-identity hard cap — more restrictive
        teams,
        ..BudgetConfig::default()
    };
    let gateway = TestGateway::spawn_with_budget(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    let mut conn = redis.pool.get().await.expect("redis conn");
    // Per-identity spend: $9 → $1 remaining against $10 soft cap.
    redis::cmd("SET")
        .arg(DEFAULT_IDENTITY_SPEND_KEY)
        .arg(9_000_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed identity spend key");
    // Team spend: $80 → $20 remaining against $100 soft cap.
    redis::cmd("SET")
        .arg(DEFAULT_TEAM_ENG_SPEND_KEY)
        .arg(80_000_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed team spend key");

    let response = gateway
        .server
        .get("/v1/models")
        .add_header("X-OxiGate-Team", "engineering")
        .await;

    response.assert_status(StatusCode::OK);
    let remaining = response
        .headers()
        .get("X-Oxigate-Budget-Remaining")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok())
        .expect("X-Oxigate-Budget-Remaining must be present and parseable");
    // Per-identity remaining ($1) < team remaining ($20) → identity wins.
    assert!(
        (remaining - 1.0).abs() < 0.01,
        "X-Oxigate-Budget-Remaining should be ~$1.0 (identity is most restrictive), got {remaining}"
    );
}

/// AC: when both soft+hard are set, and spend is between them, request proceeds and header
/// reflects remaining to the hard cap.
#[tokio::test]
async fn team_soft_and_hard_warn_zone_proceeds_and_header_is_hard_remaining() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let provider = Arc::new(StubAdapter::new());
    let budget = budget_with_team("engineering", Some(10.0), Some(15.0));

    let (gateway, _) = TestGateway::spawn_with_team_tag_budgets(
        pg.pool.clone(),
        redis.pool.clone(),
        provider,
        AuthConfig::default(),
        budget,
    )
    .await;

    // Seed spend between soft and hard caps: $12.
    let mut conn = redis.pool.get().await.expect("redis conn");
    redis::cmd("SET")
        .arg(DEFAULT_TEAM_ENG_SPEND_KEY)
        .arg(12_000_000_000_u64)
        .query_async::<()>(&mut *conn)
        .await
        .expect("seed team spend key");

    let response = gateway
        .server
        .get("/v1/models")
        .add_header("X-OxiGate-Team", "engineering")
        .await;

    response.assert_status(StatusCode::OK);
    let remaining = response
        .headers()
        .get("X-Oxigate-Budget-Remaining")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok())
        .expect("X-Oxigate-Budget-Remaining must parse");
    // Remaining to hard cap: 15 - 12 = 3.
    assert!(
        (remaining - 3.0).abs() < 0.01,
        "expected ~$3.0 remaining to hard cap, got {remaining}"
    );
}
