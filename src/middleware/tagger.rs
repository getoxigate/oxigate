// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Tagger Tower layer — extracts attribution tags from request headers .
//!
//! Reads `X-OxiGate-Team` and `X-OxiGate-Project` headers, sanitizes values,
//! extends `RequestIdentity.tags`, and emits structured log events with the tag values.
//! Never modifies `id` or `org_id`.
//!
//! Observability: a named `"tagger"` span is created with `team`, `project`, and `org_id`
//! as pre-declared fields and then used to `.instrument()` the inner call, so all
//! downstream log lines automatically inherit the tag values. `Span::current().record()`
//! is NOT used — it is a no-op unless the field was pre-declared on the span at creation
//! time, and the handler's own `#[instrument]` span does not exist yet when `call()` runs.
//!
//! TODO: Prometheus label injection deferred; per-tag labels would cause
//! cardinality explosion without a bounded label allow-list.

use std::task::{Context, Poll};

use axum::extract::Request;
use axum::http::header::HeaderName;
use axum::response::Response;
use tower::Layer;
use tower::Service;
use tracing::Instrument as _;

use crate::domain::auth::RequestIdentity;

/// HTTP header for team attribution tag.
const HEADER_TEAM: HeaderName = HeaderName::from_static("x-oxigate-team");
/// HTTP header for project attribution tag.
const HEADER_PROJECT: HeaderName = HeaderName::from_static("x-oxigate-project");

/// Max tag value length in bytes (UTF-8). Truncation happens at char boundary.
const MAX_TAG_LEN: usize = 128;

/// Tower layer that extracts attribution tags from request headers.
///
/// Runs after the auth layer. Reads `RequestIdentity` from extensions, extends
/// its `tags` map with sanitized header values, re-inserts the identity, and
/// emits `tracing::info!` events with the extracted tag values.
#[derive(Clone, Default)]
pub struct TaggerLayer;

impl TaggerLayer {
    /// Creates a new TaggerLayer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for TaggerLayer {
    type Service = TaggerService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        TaggerService { inner }
    }
}

/// Inner service that performs tag extraction and log event emission.
#[derive(Clone)]
pub struct TaggerService<S> {
    inner: S,
}

/// Sanitizes a tag value: truncate to 128 bytes at UTF-8 char boundary, replace
/// ASCII control chars with `_`. Returns `None` if the result is empty.
fn sanitize(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }

    let mut s = String::with_capacity(value.len().min(MAX_TAG_LEN));
    for ch in value.chars() {
        if s.len() + ch.len_utf8() > MAX_TAG_LEN {
            break;
        }
        let c = if ch.is_ascii_control() { '_' } else { ch };
        s.push(c);
    }

    // No final trim needed: value.trim() already removed leading/trailing whitespace,
    // and control-char replacement only inserts `_` (never whitespace).
    if s.is_empty() { None } else { Some(s) }
}

impl<S, E> Service<Request> for TaggerService<S>
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
        // Tower contract: self.inner has been polled ready and must be the instance called.
        let mut inner = self.inner.clone();
        std::mem::swap(&mut inner, &mut self.inner);

        let mut identity = req
            .extensions()
            .get::<RequestIdentity>()
            .cloned()
            .unwrap_or_default();

        // Extract tags from headers (first value only; multi-value headers are not supported).
        // A named span is created here with team/project as pre-declared fields so that
        // all downstream log lines emitted within inner.call(req) automatically include
        // the tag values. Span::current().record() is NOT used because record() is a no-op
        // unless the field was pre-declared at span creation time — and the handler's own
        // #[instrument] span does not exist yet at this point.
        let span = tracing::info_span!(
            "tagger",
            org_id = %identity.org_id,
            team   = tracing::field::Empty,
            project = tracing::field::Empty,
        );

        if let Some(raw) = req
            .headers()
            .get(&HEADER_TEAM)
            .and_then(|v| v.to_str().ok())
            && let Some(sanitized) = sanitize(raw)
        {
            span.record("team", sanitized.as_str());
            tracing::info!(parent: &span, team = %sanitized, org_id = %identity.org_id, "request tag extracted");
            identity.tags.insert("team".into(), sanitized);
        }
        if let Some(raw) = req
            .headers()
            .get(&HEADER_PROJECT)
            .and_then(|v| v.to_str().ok())
            && let Some(sanitized) = sanitize(raw)
        {
            span.record("project", sanitized.as_str());
            tracing::info!(parent: &span, project = %sanitized, org_id = %identity.org_id, "request tag extracted");
            identity.tags.insert("project".into(), sanitized);
        }

        req.extensions_mut().insert(identity);

        Box::pin(inner.call(req).instrument(span))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AuthConfig;
    use crate::middleware::auth::AuthLayer;
    use axum::body::Body;
    use axum::http::Request;
    use axum::response::IntoResponse;
    use tower::ServiceExt;

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    /// Parses a `RequestIdentity` from the JSON body produced by the identity handler below.
    fn parse_identity(body: &[u8]) -> RequestIdentity {
        let v: serde_json::Value = serde_json::from_slice(body).unwrap();
        let id = v["identity"]["id"].as_str().unwrap_or("").to_string();
        let org_id = v["identity"]["org_id"].as_str().unwrap_or("").to_string();
        let label = v["identity"]["label"].as_str().map(String::from);
        let tags: std::collections::HashMap<String, String> = v["identity"]["tags"]
            .as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                    .collect()
            })
            .unwrap_or_default();
        RequestIdentity {
            id,
            org_id,
            label,
            tags,
        }
    }

    /// Drives the AuthLayer → TaggerLayer stack with the given `config` and `headers`,
    /// returning the `RequestIdentity` seen by the inner handler.
    async fn tagged_identity_with_config(
        config: AuthConfig,
        headers: &[(&str, &str)],
    ) -> RequestIdentity {
        let inner = tower::service_fn(|req: Request<Body>| async move {
            let ext = req.extensions().get::<RequestIdentity>().cloned().unwrap();
            Ok::<_, std::convert::Infallible>(
                axum::Json(serde_json::json!({
                    "identity": {"id": ext.id, "org_id": ext.org_id, "tags": ext.tags}
                }))
                .into_response(),
            )
        });
        let mut svc = AuthLayer::new(std::sync::Arc::new(tokio::sync::RwLock::new(config)))
            .layer(TaggerLayer::new().layer(inner));

        let mut builder = Request::builder().uri("/");
        for (name, value) in headers {
            builder = builder.header(*name, *value);
        }
        let req = builder.body(Body::empty()).unwrap();

        let resp = svc.ready().await.unwrap().call(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        parse_identity(&body)
    }

    /// Convenience wrapper for unauthenticated (key: None) requests.
    async fn tagged_identity(headers: &[(&str, &str)]) -> RequestIdentity {
        tagged_identity_with_config(AuthConfig { key: None }, headers).await
    }

    // ---------------------------------------------------------------------------
    // Async (service-level) tests
    // ---------------------------------------------------------------------------

    #[tokio::test]
    async fn test_both_headers_present() {
        let identity =
            tagged_identity(&[("X-OxiGate-Team", "eng"), ("X-OxiGate-Project", "chat")]).await;
        assert_eq!(identity.id, "default");
        assert_eq!(identity.tags.get("team"), Some(&"eng".to_string()));
        assert_eq!(identity.tags.get("project"), Some(&"chat".to_string()));
    }

    #[tokio::test]
    async fn test_only_team_header() {
        let identity = tagged_identity(&[("X-OxiGate-Team", "eng")]).await;
        assert_eq!(identity.tags.get("team"), Some(&"eng".to_string()));
        assert!(!identity.tags.contains_key("project"));
    }

    #[tokio::test]
    async fn test_only_project_header() {
        let identity = tagged_identity(&[("X-OxiGate-Project", "chat")]).await;
        assert_eq!(identity.tags.get("project"), Some(&"chat".to_string()));
        assert!(!identity.tags.contains_key("team"));
    }

    #[tokio::test]
    async fn test_no_headers() {
        let identity = tagged_identity(&[]).await;
        assert_eq!(identity.id, "default");
        assert!(identity.tags.is_empty());
    }

    #[tokio::test]
    async fn test_empty_header_value_not_inserted() {
        let identity = tagged_identity(&[("X-OxiGate-Team", "")]).await;
        assert!(!identity.tags.contains_key("team"));
    }

    #[tokio::test]
    async fn test_tagger_does_not_overwrite_existing_auth_identity() {
        use crate::config::SecretString;
        let config = AuthConfig {
            key: Some(SecretString::from("my-secret")),
        };
        let identity = tagged_identity_with_config(
            config,
            &[
                ("Authorization", "Bearer my-secret"),
                ("X-OxiGate-Team", "eng"),
            ],
        )
        .await;
        assert_eq!(identity.id, "default");
        assert_eq!(identity.tags.get("team"), Some(&"eng".to_string()));
    }

    // ---------------------------------------------------------------------------
    // Pure unit tests for sanitize()
    // ---------------------------------------------------------------------------

    #[test]
    fn test_sanitize_truncates_at_128_bytes() {
        let s = "a".repeat(200);
        let out = sanitize(&s).unwrap();
        assert_eq!(out.len(), 128);
        assert!(out.chars().all(|c| c == 'a'));
    }

    #[test]
    fn test_sanitize_utf8_char_boundary() {
        // "α" is 2 bytes; at 128 bytes we should end on a char boundary
        let s = "α".repeat(100); // 200 bytes
        let out = sanitize(&s).unwrap();
        assert!(out.len() <= 128);
        assert!(std::str::from_utf8(out.as_bytes()).is_ok());
    }

    #[test]
    fn test_sanitize_control_chars_replaced() {
        let out = sanitize("a\x01b\x7f").unwrap();
        assert_eq!(out, "a_b_");
    }

    #[test]
    fn test_sanitize_empty_after_strip() {
        assert!(sanitize("   \t  ").is_none());
        assert!(sanitize("").is_none());
    }
}
