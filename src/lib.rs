// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
#[cfg(all(feature = "test-hooks", not(debug_assertions)))]
compile_error!("feature `test-hooks` is for debug tests only; do not enable it in release builds");

pub mod api;
pub mod config;
pub mod db;
pub mod domain;
pub mod lifecycle;
pub mod middleware;
pub mod observability;
pub mod plugins;
pub mod providers;
pub mod redis_pool;
pub mod utils;
