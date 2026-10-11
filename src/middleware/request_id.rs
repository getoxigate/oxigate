// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Request-ID Tower layer.
//!
//! The single owner of a request's [`RequestId`]: it mints one per request, places it in the
//! request extensions for the handlers, and returns it in the [`REQUEST_ID_HEADER`] response
//! header. It sits outside every layer that can reject a request, so a rejection carries the
//! header as well. A client-sent header of the same name is never read: the value is always
//! minted here.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use axum::extract::Request;
use axum::http::HeaderValue;
use axum::response::Response;
use tower::{Layer, Service};

use crate::domain::request_id::RequestId;

/// Response header carrying the gateway-minted request ID.
pub const REQUEST_ID_HEADER: &str = "X-Oxigate-Request-Id";

/// Tower layer that mints a [`RequestId`] per request and returns it in [`REQUEST_ID_HEADER`].
#[derive(Clone, Copy, Default)]
pub struct RequestIdLayer;

impl RequestIdLayer {
    /// Creates the layer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for RequestIdLayer {
    type Service = RequestIdService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestIdService { inner }
    }
}

/// Service produced by [`RequestIdLayer`].
#[derive(Clone)]
pub struct RequestIdService<S> {
    inner: S,
}

impl<S> Service<Request> for RequestIdService<S>
where
    S: Service<Request, Response = Response>,
    S::Future: Send + 'static,
{
    type Response = Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Response, S::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), S::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request) -> Self::Future {
        let request_id = RequestId::mint();
        req.extensions_mut().insert(request_id);
        let response = self.inner.call(req);
        Box::pin(async move {
            // Set once the inner service returns: a streamed response's headers are sent before
            // its body, and a response a layer built itself never passed through a handler.
            let mut response = response.await?;
            // A hyphenated UUID is plain ASCII, so the conversion cannot fail.
            if let Ok(value) = HeaderValue::from_str(&request_id.to_string()) {
                response.headers_mut().insert(REQUEST_ID_HEADER, value);
            }
            Ok(response)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use axum::body::Body;
    use tower::ServiceExt;
    use uuid::Uuid;

    use super::*;

    /// Calls the layer around an inner service that echoes the ID it was handed in the body,
    /// returning `(ID the handler saw, ID in the response header)`.
    async fn round_trip() -> (Uuid, Uuid) {
        let service = RequestIdLayer::new().layer(tower::service_fn(|req: Request| async move {
            let id = *req
                .extensions()
                .get::<RequestId>()
                .expect("the layer places the ID in the request extensions");
            Ok::<_, Infallible>(Response::new(Body::from(id.to_string())))
        }));
        let response = service
            .oneshot(Request::new(Body::empty()))
            .await
            .expect("infallible");
        let header = response
            .headers()
            .get(REQUEST_ID_HEADER)
            .expect("the layer sets the response header")
            .to_str()
            .expect("ASCII header")
            .parse::<Uuid>()
            .expect("header is a UUID");
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body reads");
        let handed = std::str::from_utf8(&body)
            .expect("UTF-8 body")
            .parse::<Uuid>()
            .expect("handler saw a UUID");
        (handed, header)
    }

    #[tokio::test]
    async fn header_returns_the_id_the_handler_saw() {
        let (handed, header) = round_trip().await;
        assert_eq!(header, handed);
    }

    #[tokio::test]
    async fn each_request_gets_a_distinct_version_7_id() {
        let (first, _) = round_trip().await;
        let (second, _) = round_trip().await;
        assert_ne!(first, second, "two requests must not share an ID");
        assert_eq!(first.get_version_num(), 7);
        assert_eq!(second.get_version_num(), 7);
    }
}
