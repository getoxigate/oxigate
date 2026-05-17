// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Prometheus scrape endpoint — GET /metrics .
//!
//! This route is mounted on the outermost router, outside all auth and budget middleware.
//! It requires NO authentication — protect at the network level (firewall, NetworkPolicy,
//! reverse proxy). See `docs/guides/prometheus-metrics.md` for operator guidance.

use axum::http::StatusCode;
use axum::response::IntoResponse;
use metrics_exporter_prometheus::PrometheusHandle;

/// Renders the current Prometheus text snapshot.
///
/// Returns 200 + text/plain when a [`PrometheusHandle`] extension is present (production).
/// Returns 503 when no handle is available (e.g. in test contexts without a recorder).
pub async fn metrics_handler(
    handle: Option<axum::extract::Extension<PrometheusHandle>>,
) -> impl IntoResponse {
    match handle {
        Some(axum::extract::Extension(h)) => (StatusCode::OK, h.render()).into_response(),
        None => StatusCode::SERVICE_UNAVAILABLE.into_response(),
    }
}
