// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration test suite.
//!
//! Each submodule contains tests for a specific area (db_migrations, redis_pool, etc.).

mod auth;
mod budget_e2e;
#[cfg(feature = "test-hooks")]
mod budget_period_reset;
mod chat_completions_e2e;
#[path = "../common/mod.rs"]
mod common;
mod concurrency;
mod db_migrations;
mod embeddings_e2e;
mod fallback_e2e;
mod gateway_e2e;
mod global_safety_e2e;
mod health_checks;
mod models_e2e;
mod pricing_e2e;
mod prometheus_metrics;
mod proptest_invariants;
mod providers;
mod redis_pool;
mod request_size_limits;
mod routing;
mod spend_e2e;
mod spend_writer;
mod stream_timeout_e2e;
mod streaming;
mod tagger_e2e;
mod team_tag_budget_e2e;
