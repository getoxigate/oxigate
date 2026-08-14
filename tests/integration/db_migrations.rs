// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for database migrations .
//!
//! Uses testcontainers via PgContainer to spin up a real PostgreSQL instance.

use crate::common::containers::PgContainer;
use oxigate::config::{DatabaseConfig, SecretString};
use oxigate::db::{DbError, connect, run_migrations};

#[tokio::test]
async fn test_migrations_apply_on_fresh_db() {
    let pg = PgContainer::start().await.expect("pg container must start");

    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM _sqlx_migrations")
        .fetch_one(&pg.pool)
        .await
        .expect("query");
    assert_eq!(
        count, 1,
        "expected 1 migration recorded (0001_initial_schema squash,)"
    );
}

#[tokio::test]
async fn test_migrations_idempotent() {
    let pg = PgContainer::start().await.expect("pg container must start");

    run_migrations(&pg.pool)
        .await
        .expect("second migration run should be idempotent");
}

#[tokio::test]
async fn test_spend_records_table_exists_after_migration() {
    let pg = PgContainer::start().await.expect("pg container must start");

    sqlx::query("SELECT 1 FROM spend_records LIMIT 0")
        .execute(&pg.pool)
        .await
        .expect("spend_records table must exist after migrations");
}

#[tokio::test]
async fn test_spend_records_migration_idempotent() {
    let pg = PgContainer::start().await.expect("pg container must start");

    run_migrations(&pg.pool)
        .await
        .expect("init_db runs migrations; second run must be idempotent");
    run_migrations(&pg.pool)
        .await
        .expect("third run must also succeed");
}

#[tokio::test]
async fn test_connect_fails_on_unreachable_db() {
    let config = DatabaseConfig {
        url: SecretString::new("postgres://invalid:5432/nodb"),
        max_connections: Some(1),
        pool_acquire_timeout_secs: Some(1),
    };

    let result = connect(&config).await;
    assert!(result.is_err(), "should fail on unreachable DB");
    assert!(matches!(result.unwrap_err(), DbError::Connect(_)));
}
