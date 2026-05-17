// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Auth Tower layer — Bearer token validation against config.
//!
//! Validates Authorization: Bearer token against auth.key from config.
//! When auth.key is absent, bypasses validation and injects RequestIdentity::default().

use std::task::{Context, Poll};

use axum::extract::Request;
use axum::http::header::{AUTHORIZATION, WWW_AUTHENTICATE};
use axum::response::{IntoResponse, Response};
use secrecy::ExposeSecret;
use serde_json::json;
use subtle::ConstantTimeEq;
use tower::Layer;
use tower::Service;

use crate::config::AuthConfig;
use crate::domain::auth::RequestIdentity;

/// Tower layer that validates Bearer tokens and injects RequestIdentity.
///
/// Holds `Arc<RwLock<AuthConfig>>` for hot-reload (Class A). Reads config on each request.
#[derive(Clone)]
pub struct AuthLayer {
    auth_config: std::sync::Arc<tokio::sync::RwLock<AuthConfig>>,
}

impl AuthLayer {
    /// Creates a new AuthLayer. Accepts Arc<RwLock<AuthConfig>> for SIGHUP hot-reload.
    #[must_use]
    pub fn new(auth_config: std::sync::Arc<tokio::sync::RwLock<AuthConfig>>) -> Self {
        Self { auth_config }
    }
}

impl<S> Layer<S> for AuthLayer {
    type Service = AuthService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuthService {
            inner,
            auth_config: std::sync::Arc::clone(&self.auth_config),
        }
    }
}

/// Inner service that performs Bearer validation and injects RequestIdentity.
#[derive(Clone)]
pub struct AuthService<S> {
    inner: S,
    auth_config: std::sync::Arc<tokio::sync::RwLock<AuthConfig>>,
}

fn inject_default_identity(req: &mut Request) {
    req.extensions_mut().insert(RequestIdentity::default());
}

fn unauthorized_response(message: &str) -> Response {
    (
        axum::http::StatusCode::UNAUTHORIZED,
        [
            (axum::http::header::CONTENT_TYPE, "application/json"),
            // RFC 6750 §3: realm auth-param is required.
            (WWW_AUTHENTICATE, "Bearer realm=\"oxigate\""),
        ],
        axum::Json(json!({
            "error": "unauthorized",
            "message": message
        })),
    )
        .into_response()
}

/// Constant-time comparison that avoids both content and length timing leaks.
///
/// Both inputs are copied into fixed-size 256-byte buffers (keys longer than 256 bytes are
/// rejected elsewhere in the auth flow, so truncation never occurs in practice). Length and
/// content are then compared with `subtle::ConstantTimeEq` so neither the presence of a
/// length difference nor its magnitude leaks via timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    const MAX_KEY_LEN: usize = 256;
    let mut padded_a = [0u8; MAX_KEY_LEN];
    let mut padded_b = [0u8; MAX_KEY_LEN];
    let len_a = a.len().min(MAX_KEY_LEN);
    let len_b = b.len().min(MAX_KEY_LEN);
    padded_a[..len_a].copy_from_slice(&a[..len_a]);
    padded_b[..len_b].copy_from_slice(&b[..len_b]);
    // Compare length (as u64 to avoid u8 truncation for len > 255) and content, both in
    // constant time; combine with & so a false length never short-circuits content check.
    let len_eq = (a.len() as u64).ct_eq(&(b.len() as u64));
    bool::from(len_eq & padded_a.ct_eq(&padded_b))
}

impl<S, E> Service<Request> for AuthService<S>
where
    S: Service<Request, Response = Response, Error = E> + Clone + Send + 'static,
    S::Future: Send + 'static,
    E: Send + 'static,
{
    type Response = Response;
    type Error = E;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request) -> Self::Future {
        let auth_config = std::sync::Arc::clone(&self.auth_config);

        // Tower contract: self.inner has been polled ready and must be the instance called.
        // Swap it into the future; leave a fresh clone in self.inner for the next poll_ready.
        let mut inner = self.inner.clone();
        std::mem::swap(&mut inner, &mut self.inner);

        Box::pin(async move {
            // Copy key bytes out before any .await so the RwLock read guard is dropped
            // before the inner service call. Holding it across inner.call() would stall
            // SIGHUP auth reloads for the full duration of every proxied LLM request.
            let key_bytes: Option<Vec<u8>> = {
                let config = auth_config.read().await;
                config
                    .key
                    .as_ref()
                    .map(|k| k.expose_secret().as_bytes().to_vec())
            }; // read guard dropped here

            if key_bytes.is_none() {
                tracing::debug!(
                    auth_bypassed = true,
                    "auth.key not configured — bypassing auth",
                );
                inject_default_identity(&mut req);
                return inner.call(req).await;
            }

            let expected = key_bytes.unwrap();

            // Validate the Bearer token. Use a block so the immutable borrow of req
            // (through auth_header) ends before req.extensions_mut() below.
            let auth_result: Result<(), Response> = {
                let auth_header = req
                    .headers()
                    .get(AUTHORIZATION)
                    .and_then(|v| v.to_str().ok());

                match auth_header {
                    None => Err(unauthorized_response("missing Authorization header")),
                    Some(h) if !h.starts_with("Bearer ") => Err(unauthorized_response(
                        "invalid authorization scheme; use Bearer",
                    )),
                    Some(h) => {
                        let supplied = &h.as_bytes()["Bearer ".len()..];
                        if constant_time_eq(supplied, &expected) {
                            Ok(())
                        } else {
                            Err(unauthorized_response("invalid API key"))
                        }
                    }
                }
            };

            if let Err(resp) = auth_result {
                return Ok(resp);
            }

            inject_default_identity(&mut req);
            inner.call(req).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SecretString;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_auth_bypass_when_key_absent() {
        let config = AuthConfig { key: None };
        let mut svc = AuthLayer::new(std::sync::Arc::new(tokio::sync::RwLock::new(config))).layer(
            tower::service_fn(|req: Request<Body>| async move {
                let ext = req.extensions().get::<RequestIdentity>().cloned();
                Ok::<_, std::convert::Infallible>((
                    axum::http::StatusCode::OK,
                    axum::Json(serde_json::json!({
                        "identity": ext.map(|i| serde_json::json!({"id": i.id, "org_id": i.org_id}))
                    })),
                ).into_response())
            }),
        );

        let req = Request::builder().uri("/").body(Body::empty()).unwrap();
        let resp = svc.ready().await.unwrap().call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["identity"]["id"], "default");
        assert_eq!(v["identity"]["org_id"], "default");
    }

    #[tokio::test]
    async fn test_auth_rejects_missing_header() {
        let config = AuthConfig {
            key: Some(SecretString::from("secret")),
        };
        let mut svc = AuthLayer::new(std::sync::Arc::new(tokio::sync::RwLock::new(config))).layer(
            tower::service_fn(|_req: Request<Body>| async {
                Ok::<_, std::convert::Infallible>(
                    axum::Json(serde_json::json!({"ok": true})).into_response(),
                )
            }),
        );

        let req = Request::builder().uri("/").body(Body::empty()).unwrap();
        let resp = svc.ready().await.unwrap().call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let www_auth = resp.headers().get(WWW_AUTHENTICATE).unwrap();
        assert_eq!(www_auth.to_str().unwrap(), "Bearer realm=\"oxigate\"");
    }

    #[tokio::test]
    async fn test_auth_rejects_non_bearer_scheme() {
        let config = AuthConfig {
            key: Some(SecretString::from("secret")),
        };
        let mut svc = AuthLayer::new(std::sync::Arc::new(tokio::sync::RwLock::new(config))).layer(
            tower::service_fn(|_req: Request<Body>| async {
                Ok::<_, std::convert::Infallible>(
                    axum::Json(serde_json::json!({"ok": true})).into_response(),
                )
            }),
        );

        let req = Request::builder()
            .uri("/")
            .header(AUTHORIZATION, "Basic dXNlcjpwYXNz")
            .body(Body::empty())
            .unwrap();
        let resp = svc.ready().await.unwrap().call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_rejects_wrong_token() {
        let config = AuthConfig {
            key: Some(SecretString::from("correct-token")),
        };
        let mut svc = AuthLayer::new(std::sync::Arc::new(tokio::sync::RwLock::new(config))).layer(
            tower::service_fn(|_req: Request<Body>| async {
                Ok::<_, std::convert::Infallible>(
                    axum::Json(serde_json::json!({"ok": true})).into_response(),
                )
            }),
        );

        let req = Request::builder()
            .uri("/")
            .header(AUTHORIZATION, "Bearer wrong-token")
            .body(Body::empty())
            .unwrap();
        let resp = svc.ready().await.unwrap().call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_accepts_correct_token() {
        let config = AuthConfig {
            key: Some(SecretString::from("my-secret")),
        };
        let mut svc = AuthLayer::new(std::sync::Arc::new(tokio::sync::RwLock::new(config))).layer(
            tower::service_fn(|req: Request<Body>| async move {
                let ext = req.extensions().get::<RequestIdentity>().cloned();
                Ok::<_, std::convert::Infallible>((
                    axum::http::StatusCode::OK,
                    axum::Json(serde_json::json!({
                        "identity": ext.map(|i| serde_json::json!({"id": i.id, "org_id": i.org_id}))
                    })),
                ).into_response())
            }),
        );

        let req = Request::builder()
            .uri("/")
            .header(AUTHORIZATION, "Bearer my-secret")
            .body(Body::empty())
            .unwrap();
        let resp = svc.ready().await.unwrap().call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["identity"]["id"], "default");
        assert_eq!(v["identity"]["org_id"], "default");
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"a", b"a"));
        assert!(!constant_time_eq(b"a", b"b"));
        assert!(!constant_time_eq(b"a", b"ab"));
        assert!(!constant_time_eq(b"ab", b"a"));
    }

    #[test]
    fn test_constant_time_eq_boundary() {
        // Keys at exactly MAX_KEY_LEN (256 bytes) compare correctly.
        let key_a = vec![b'x'; 256];
        let key_b = vec![b'x'; 256];
        assert!(constant_time_eq(&key_a, &key_b));

        // Keys that differ at byte 255 (last byte within the buffer) are rejected.
        let mut key_c = vec![b'x'; 256];
        key_c[255] = b'y';
        assert!(!constant_time_eq(&key_a, &key_c));

        // config.rs::validate() rejects keys that are empty or >256 bytes before they
        // reach this function, so we only need to prove correct behaviour at the boundary.
    }

    #[tokio::test]
    async fn test_auth_rejects_empty_bearer_token() {
        // "Bearer " with no trailing token — zero-length supplied key must be rejected.
        let config = AuthConfig {
            key: Some(SecretString::from("secret")),
        };
        let mut svc = AuthLayer::new(std::sync::Arc::new(tokio::sync::RwLock::new(config))).layer(
            tower::service_fn(|_req: Request<Body>| async {
                Ok::<_, std::convert::Infallible>(
                    axum::Json(serde_json::json!({"ok": true})).into_response(),
                )
            }),
        );

        let req = Request::builder()
            .uri("/")
            .header(AUTHORIZATION, "Bearer ")
            .body(Body::empty())
            .unwrap();
        let resp = svc.ready().await.unwrap().call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}
