// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Embeddings handler — POST /v1/embeddings.
//!
//! OpenAI-compatible embeddings with auth stub.
//! Cost headers, structured cost log, and spend write for embeddings.

use std::sync::Arc;

use axum::Extension;
use axum::Json;
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header::RETRY_AFTER};
use axum::response::IntoResponse;
use serde_json::json;
use thiserror::Error;

use crate::api::AppState;
use crate::domain::auth::RequestIdentity;
use crate::domain::embedding::EmbeddingRequest;
use crate::domain::ports::{AttemptedMeta, ProviderError};
use crate::domain::spend::SpendRecord;
use crate::middleware::request_metrics::ProviderLabel;
use crate::observability::metrics::COST_USD_TOTAL;
use crate::utils::cost_headers::{build_embedding_cost_headers, inject_zero_cost_headers};

/// Embeddings endpoint error.
#[derive(Debug, Error)]
pub enum EmbeddingsError {
    #[error("invalid request: {0}")]
    BadRequest(String),
    #[error("unknown model: {0}")]
    UnknownModel(String),
    #[error("authentication error: {0}")]
    Auth(String),
    #[error("rate limited")]
    RateLimited { retry_after: Option<u64> },
    #[error("provider unavailable: {0}")]
    Unavailable(String),
    #[error("provider timeout after {elapsed_ms}ms")]
    Timeout { elapsed_ms: u64 },
    #[error("provider error: {0}")]
    Provider(String),
    #[error("not implemented")]
    NotImplemented,
}

impl From<ProviderError> for EmbeddingsError {
    fn from(e: ProviderError) -> Self {
        match e {
            ProviderError::NotImplemented => EmbeddingsError::NotImplemented,
            ProviderError::InvalidRequest(s) => EmbeddingsError::BadRequest(s),
            ProviderError::UnknownModel(s) => EmbeddingsError::UnknownModel(s),
            ProviderError::Auth(s) => EmbeddingsError::Auth(s),
            ProviderError::RateLimited { retry_after } => {
                EmbeddingsError::RateLimited { retry_after }
            }
            ProviderError::AllProvidersRateLimited { retry_after } => {
                EmbeddingsError::RateLimited {
                    retry_after: Some(retry_after),
                }
            }
            ProviderError::ProviderUnavailable(s) => EmbeddingsError::Unavailable(s),
            ProviderError::Unreachable(s) => EmbeddingsError::Unavailable(s),
            ProviderError::Timeout { elapsed_ms } => EmbeddingsError::Timeout { elapsed_ms },
            _ => EmbeddingsError::Provider(e.to_string()),
        }
    }
}

impl IntoResponse for EmbeddingsError {
    fn into_response(self) -> axum::response::Response {
        let (status, code) = match &self {
            Self::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            Self::UnknownModel(_) => (StatusCode::NOT_FOUND, "invalid_request_error"),
            Self::Auth(_) => (StatusCode::UNAUTHORIZED, "authentication_error"),
            Self::RateLimited { .. } => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_exceeded"),
            Self::Unavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "provider_unavailable"),
            Self::Timeout { .. } => (StatusCode::GATEWAY_TIMEOUT, "provider_timeout"),
            Self::Provider(_) => (StatusCode::BAD_GATEWAY, "provider_error"),
            Self::NotImplemented => (StatusCode::NOT_IMPLEMENTED, "not_implemented"),
        };
        let body = Json(json!({
            "error": {
                "message": self.to_string(),
                "type": code,
                "param": null,
                "code": code
            }
        }));
        let mut response = (status, body).into_response();
        if matches!(self, Self::Auth(_)) {
            response
                .headers_mut()
                .insert("WWW-Authenticate", HeaderValue::from_static("Bearer"));
        }
        if let Self::RateLimited {
            retry_after: Some(secs),
        } = self
        {
            response.headers_mut().insert(
                RETRY_AFTER,
                HeaderValue::from_str(&secs.to_string())
                    .expect("u64 decimal is always a valid HeaderValue"),
            );
        }
        response
    }
}

/// Handles POST /v1/embeddings.
#[tracing::instrument(skip_all, fields(model = %req.model))]
pub async fn embeddings(
    State(state): State<AppState>,
    Extension(identity): Extension<RequestIdentity>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<impl IntoResponse, EmbeddingsError> {
    let request_start = std::time::Instant::now();
    let provider = state.provider.read().await.clone();

    // Validate before dispatch — consistent 400 regardless of provider.
    let inputs = req.input.as_slice();
    if inputs.is_empty() || inputs.iter().all(|s| s.is_empty()) {
        return Err(EmbeddingsError::BadRequest(
            "input must not be empty".into(),
        ));
    }

    let request_id = uuid::Uuid::new_v4().to_string();
    let provider_label = provider.metadata().name.to_string();

    // Destructure dispatch result into a uniform (embed_result, provider_name, meta) triple.
    // On error, meta defaults to empty; provider_label (pre-dispatch name) is the fallback.
    // For non-router adapters this is the correct name. For routers it falls back to "router"
    // because ProviderError does not carry AttemptedMeta — acceptable until the trait surface changes.
    let (embed_result, provider_name, meta) = match provider.embeddings_with_trace(&req).await {
        Ok((response, meta)) => {
            let name = meta
                .providers
                .last()
                .cloned()
                .unwrap_or_else(|| provider_label.clone());
            (Ok(response), name, meta)
        }
        Err(e) => (Err(e), provider_label, AttemptedMeta::default()),
    };

    match embed_result {
        Err(e) => {
            let mut resp = EmbeddingsError::from(e).into_response();
            inject_zero_cost_headers(&mut resp, &req.model);
            resp.extensions_mut().insert(ProviderLabel(provider_name));
            Ok(resp)
        }
        Ok(response) => {
            let model_used = meta
                .models
                .last()
                .cloned()
                .unwrap_or_else(|| req.model.clone());

            tracing::info!(
                vectors = %response.data.len(),
                "embeddings"
            );

            let latency_ms =
                i32::try_from(request_start.elapsed().as_millis()).unwrap_or_else(|_| {
                    tracing::warn!("embedding request latency overflows i32; recording -1");
                    -1
                });

            let mut resp = if let Some(ref usage) = response.usage {
                // /v1/embeddings is the sync endpoint — batch discount applies only to
                // /v1/batches (async Batch API). Always false here regardless of input shape.
                let (cost_headers, cost_breakdown, token_usage) = build_embedding_cost_headers(
                    &model_used,
                    usage,
                    Arc::clone(&state.pricing_db),
                    false,
                );
                let cost_usd = cost_breakdown.total_cost.to_display_string();

                let record = SpendRecord::build(
                    &identity,
                    &model_used,
                    &provider_name,
                    &token_usage,
                    &cost_breakdown,
                    latency_ms,
                );
                let budget = state.budget_settings.read().await.clone();
                crate::api::spawn_cost_log_and_spend(
                    "embedding_cost",
                    record,
                    &request_id,
                    &cost_usd,
                    Arc::clone(&state.pool),
                    Arc::clone(&state.redis_pool),
                    budget,
                );

                metrics::counter!(
                    COST_USD_TOTAL,
                    "provider" => provider_name.clone()
                )
                .increment(cost_breakdown.total_cost.as_u64());

                let mut r = (StatusCode::OK, Json(response)).into_response();
                r.headers_mut().extend(cost_headers);
                r
            } else {
                let mut r = (StatusCode::OK, Json(response)).into_response();
                inject_zero_cost_headers(&mut r, &model_used);
                r
            };

            resp.extensions_mut()
                .insert(ProviderLabel(provider_name.clone()));

            let expose_providers = state.security.read().await.expose_provider_names;
            if expose_providers {
                crate::api::inject_attempted_headers(
                    resp.headers_mut(),
                    &meta.providers,
                    &meta.models,
                    meta.fallback_trigger.as_deref(),
                    meta.fallback_dispatched,
                );
            }
            Ok(resp)
        }
    }
}
