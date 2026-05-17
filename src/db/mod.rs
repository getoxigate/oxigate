// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! PostgreSQL connection pool, migration runner, and spend persistence.
//!
//! `DbPool` is a type alias for `sqlx::PgPool`. All query code in other modules
//! receives a `&DbPool` (or `Pool<Postgres>`) — never a raw connection.

pub mod spend_reader;
pub mod spend_writer;

use std::time::Duration;

use secrecy::ExposeSecret;
use sqlx::postgres::{PgPool, PgPoolOptions};
use thiserror::Error;

use crate::config::{DatabaseConfig, GatewayConfig};

/// TCP connect timeout (seconds) for fail-fast when DB host is unreachable.
/// Appended to connection URL per libpq. Note: sqlx 0.8 may ignore this; see.
const TCP_CONNECT_TIMEOUT_SECS: u64 = 2;

/// Shareable PostgreSQL connection pool. Clone is cheap (Arc-backed).
pub type DbPool = PgPool;

/// Database connection or migration error.
#[derive(Debug, Error)]
pub enum DbError {
    #[error("database connection failed: {0}")]
    Connect(#[from] sqlx::Error),
    #[error("migration failed: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
}

/// Build a connection pool from `config`.
///
/// `max_connections` defaults to 10 when not specified.
/// Uses a short TCP connect timeout for fail-fast when the DB host is unreachable.
/// Returns `DbError::Connect` if the database is unreachable.
pub async fn connect(config: &DatabaseConfig) -> Result<DbPool, DbError> {
    let url = config.url.expose_secret();
    let url_with_timeout = if url.contains('?') {
        format!("{}&connect_timeout={}", url, TCP_CONNECT_TIMEOUT_SECS)
    } else {
        format!("{}?connect_timeout={}", url, TCP_CONNECT_TIMEOUT_SECS)
    };

    PgPoolOptions::new()
        .max_connections(config.max_connections.unwrap_or(10))
        .acquire_timeout(Duration::from_secs(
            config.pool_acquire_timeout_secs.unwrap_or(30),
        ))
        .connect(&url_with_timeout)
        .await
        .map_err(DbError::Connect)
}

/// Apply all pending migrations in `migrations/`.
///
/// Uses the compile-time `sqlx::migrate!()` macro — the migration files are
/// embedded in the binary at build time, so no filesystem access is needed at runtime.
pub async fn run_migrations(pool: &DbPool) -> Result<(), DbError> {
    sqlx::migrate!().run(pool).await.map_err(DbError::Migrate)
}

/// Ping the database with a lightweight query. Returns `Ok(())` on success.
///
/// Executes `SELECT 1` with the pool's own acquire timeout as the implicit deadline.
/// Callers should wrap this in `tokio::time::timeout` for predictable probe latency.
pub async fn health_check(pool: &DbPool) -> Result<(), DbError> {
    sqlx::query("SELECT 1")
        .execute(pool)
        .await
        .map(|_| ())
        .map_err(DbError::Connect)
}

/// Connect and run migrations. Shared by Serve and Migrate modes.
pub async fn init_db(config: &GatewayConfig) -> Result<DbPool, DbError> {
    let pool = connect(&config.database).await?;
    run_migrations(&pool).await?;
    Ok(pool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_db_error_display_connect() {
        let io_err = io::Error::new(io::ErrorKind::ConnectionRefused, "test connect");
        let err = DbError::Connect(sqlx::Error::Io(io_err));
        let s = err.to_string();
        assert!(s.contains("database connection failed"));
    }

    #[test]
    fn test_db_error_display_migrate() {
        let err = DbError::Migrate(sqlx::migrate::MigrateError::Source(Box::new(
            io::Error::new(io::ErrorKind::NotFound, "migration file missing"),
        )));
        let s = err.to_string();
        assert!(s.contains("migration failed"));
    }
}
