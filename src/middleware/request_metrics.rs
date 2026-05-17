// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Request metrics Tower layer .
//!
//! Emits `oxigate_requests_total{method, status, provider}` and
//! `oxigate_request_duration_seconds{provider}` for every v1 LLM request.
//!
//! The `provider` label value is read from a [`ProviderLabel`] extension injected
//! into the response by the chat handler. If absent (e.g. for non-chat routes),
//! it falls back to `"unknown"`.

use std::task::{Context, Poll};
use std::time::Instant;

use axum::extract::Request;
use axum::response::Response;
use tower::{Layer, Service};

use crate::observability::metrics::{REQUEST_DURATION_SECONDS, REQUESTS_TOTAL};

/// Response extension set by the chat handler to propagate the final provider name.
///
/// Must be inserted with `response.extensions_mut().insert(ProviderLabel(name))` before
/// the handler returns; `RequestMetricsService` reads it to populate the `provider` label.
#[derive(Clone)]
pub struct ProviderLabel(pub String);

/// Tower layer that emits per-request counter and latency histogram.
#[derive(Clone, Default)]
pub struct RequestMetricsLayer;

impl RequestMetricsLayer {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for RequestMetricsLayer {
    type Service = RequestMetricsService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestMetricsService { inner }
    }
}

/// Inner service that times the request and emits metrics on response.
#[derive(Clone)]
pub struct RequestMetricsService<S> {
    inner: S,
}

impl<S, E> Service<Request> for RequestMetricsService<S>
where
    S: Service<Request, Response = Response, Error = E> + Clone + Send + 'static,
    S::Future: Send + 'static,
    E: Send + 'static,
{
    type Response = Response;
    type Error = E;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Response, E>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), E>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request) -> Self::Future {
        let mut inner = self.inner.clone();
        std::mem::swap(&mut inner, &mut self.inner);

        let method = req.method().as_str().to_owned();
        let start = Instant::now();

        Box::pin(async move {
            let response = inner.call(req).await?;
            let elapsed = start.elapsed().as_secs_f64();
            let status = response.status().as_u16().to_string();
            let provider = response
                .extensions()
                .get::<ProviderLabel>()
                .map(|l| l.0.clone())
                .unwrap_or_else(|| "unknown".to_owned());

            metrics::counter!(
                REQUESTS_TOTAL,
                "method"   => method,
                "status"   => status,
                "provider" => provider.clone()
            )
            .increment(1);
            metrics::histogram!(REQUEST_DURATION_SECONDS, "provider" => provider).record(elapsed);

            Ok(response)
        })
    }
}
