// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! E2E smoke tests for gateway startup, routing, and JSON error shapes.
//!
//! Uses testcontainers Postgres + Redis and axum-test TestServer (no child process).
//! Single test shares one TestGateway instance to avoid duplicate container lifecycles.
//!
//! TODO: add wiremock-intercepted gateway routing test — AC "wiremock server
//! intercepts and responds" deferred to chat completions E2E.

use std::sync::Arc;

use axum::http::StatusCode;

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::StubAdapter;

#[tokio::test]
async fn test_gateway_routing() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let gateway = TestGateway::spawn(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
    )
    .await;

    // GET /health — 200 OK, {"status":"ok"}
    let health = gateway.server.get("/health").await;
    health.assert_status(StatusCode::OK);
    health.assert_json(&serde_json::json!({"status": "ok"}));

    // GET /v1/nonexistent — 404 with JSON error body
    let not_found = gateway.server.get("/v1/nonexistent").await;
    not_found.assert_status(StatusCode::NOT_FOUND);
    not_found.assert_json(&serde_json::json!({
        "error": "not_found",
        "message": "route does not exist"
    }));
}
