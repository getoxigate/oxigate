// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Active-connections gauge Tower layer .
//!
//! Tracks concurrent in-flight LLM requests via `oxigate_active_connections`.
//! The decrement fires when the **response body is dropped** — which happens at stream EOF or
//! client disconnect — not when headers are returned. This correctly accounts for long-lived
//! SSE streaming connections.
//!
//! ## Implementation note: per-instance counter
//! Each `layer()` call creates a fresh `Arc<AtomicI64>`. In production (single router) all
//! `ActiveConnectionsService` clones share the same `Arc`, so the counter is correct.
//! In tests, separate `router_with_metrics()` calls each create their own `Arc`, but all write
//! to the same global `oxigate_active_connections` gauge — the gauge value becomes meaningless
//! across concurrent test gateways. This is an accepted test-only limitation.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::task::{Context, Poll};

use axum::body::Body;
use axum::extract::Request;
use axum::response::Response;
use bytes::Bytes;
use http_body::Frame;
use tower::{Layer, Service};

use crate::observability::metrics::ACTIVE_CONNECTIONS;

/// Tower layer that maintains the `oxigate_active_connections` gauge.
#[derive(Clone, Default)]
pub struct ActiveConnectionsLayer;

impl ActiveConnectionsLayer {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for ActiveConnectionsLayer {
    type Service = ActiveConnectionsService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ActiveConnectionsService {
            inner,
            counter: Arc::new(AtomicI64::new(0)),
        }
    }
}

/// Inner service that increments the gauge on request and decrements via body drop guard.
#[derive(Clone)]
pub struct ActiveConnectionsService<S> {
    inner: S,
    counter: Arc<AtomicI64>,
}

/// Drop guard: decrements the gauge when dropped.
///
/// Stored inside `GuardedBody` so the decrement fires at body drop time (stream EOF or
/// client disconnect), not at response-headers time.
struct DecrementGuard(Arc<AtomicI64>);

impl Drop for DecrementGuard {
    fn drop(&mut self) {
        let new_val = self.0.fetch_sub(1, Ordering::Relaxed) - 1;
        metrics::gauge!(ACTIVE_CONNECTIONS).set(new_val as f64);
    }
}

/// Response body wrapper that holds the `DecrementGuard`.
///
/// When axum drops this body — at stream EOF or on client disconnect — the guard fires
/// and the gauge is decremented. `Body: Unpin` so projection is safe without `pin_project`.
struct GuardedBody {
    inner: Body,
    _guard: DecrementGuard,
}

impl http_body::Body for GuardedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // Body: Unpin, so get_mut() is safe on Pin<&mut Self>.
        Pin::new(&mut self.get_mut().inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

impl<S, E> Service<Request> for ActiveConnectionsService<S>
where
    S: Service<Request, Response = Response, Error = E> + Clone + Send + 'static,
    S::Future: Send + 'static,
    E: Send + 'static,
{
    type Response = Response;
    type Error = E;
    type Future = Pin<Box<dyn std::future::Future<Output = Result<Response, E>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), E>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request) -> Self::Future {
        let mut inner = self.inner.clone();
        std::mem::swap(&mut inner, &mut self.inner);

        let counter = Arc::clone(&self.counter);
        Box::pin(async move {
            let new_val = counter.fetch_add(1, Ordering::Relaxed) + 1;
            metrics::gauge!(ACTIVE_CONNECTIONS).set(new_val as f64);
            let guard = DecrementGuard(counter);

            let response = inner.call(req).await?;
            // Move the decrement guard into the response body so it fires at body-drop time,
            // not at future-completion time. This correctly tracks streaming connections.
            let (parts, body) = response.into_parts();
            let guarded = GuardedBody {
                inner: body,
                _guard: guard,
            };
            Ok(Response::from_parts(parts, Body::new(guarded)))
        })
    }
}
