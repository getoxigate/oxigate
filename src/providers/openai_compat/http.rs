// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Shared HTTP client for the OpenAI-compat adapter family.
//!
//! One [`CompatHttpClient`] is created at startup and [`Arc`]-shared across all
//! `openai_compat[]` instances. HTTP-level policy (redirect hardening, connection pool
//! sizing) lives here. Per-instance config (`timeout_secs`) is applied per-request in
//! [`OpenAICompatAdapter::build_request`].

use std::time::Duration;

use crate::domain::ports::ProviderError;

/// Shared reqwest client for all OpenAI-compat adapter instances.
pub struct CompatHttpClient {
    pub(crate) inner: reqwest::Client,
}

impl CompatHttpClient {
    /// Builds the shared client. Call once at startup; clone the [`Arc`] for each adapter.
    pub fn new() -> Result<Self, ProviderError> {
        let inner = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none()) // compat providers publish stable URLs
            .pool_max_idle_per_host(8) // bounded FD ceiling for N compat instances
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .map_err(|e| ProviderError::Unreachable(format!("compat http client: {e}")))?;
        Ok(Self { inner })
    }
}
