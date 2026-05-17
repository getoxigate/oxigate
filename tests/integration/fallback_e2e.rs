// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Integration tests for fallback + retry engine.
//!
//! These tests exercise the full HTTP layer: a `ProviderRouter` is injected into
//! `TestGateway`, a real request goes through axum, and the response headers are
//! verified. Unit tests for the dispatch logic live in `src/providers/router.rs`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use axum::http::StatusCode;
use bytes::Bytes;
use oxigate::api::{CHAT_COMPLETIONS_PATH, EMBEDDINGS_PATH};
use oxigate::config::{
    FallbackRule, FallbackTarget, PricingConfig, RetryConfig, RoutingConfig, SecurityConfig,
};
use oxigate::domain::chat::{ChatRequest, ChatResponse, Choice, Message, Role, StreamChunk, Usage};
use oxigate::domain::embedding::{EmbeddingData, EmbeddingRequest, EmbeddingResponse};
use oxigate::domain::ports::{
    ChatCompletionStream, HealthStatus, ProviderAdapter, ProviderError, ProviderMetadata,
};
use oxigate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
use oxigate::domain::routing::WeightedRandom;
use oxigate::providers::{ProviderHealthTracker, ProviderRouter};

use crate::common::containers::{PgContainer, RedisContainer};

// ---------------------------------------------------------------------------
// Test adapters
// ---------------------------------------------------------------------------

/// Builds a minimal valid `ChatResponse` for testing.
fn ok_response(model: &str) -> ChatResponse {
    ChatResponse {
        id: "chatcmpl-test".into(),
        object: "chat.completion".into(),
        created: 1_700_000_000,
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: Role::Assistant,
                content: Some(oxigate::domain::chat::MessageContent::Text("ok".into())),
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: Some("stop".into()),
        }],
        usage: Usage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            ..Default::default()
        },
    }
}

/// Adapter that returns `Unreachable` for the first `fail_for` calls, then succeeds.
struct FailThenSucceedAdapter {
    meta: ProviderMetadata,
    fail_for: u32,
    call_count: Arc<AtomicU32>,
}

impl FailThenSucceedAdapter {
    fn new(name: &str, models: Vec<&str>, fail_for: u32) -> Self {
        Self {
            meta: ProviderMetadata {
                name: name.to_string(),
                supported_models: models.iter().map(|s| s.to_string()).collect(),
                supports_streaming: false,
                supports_tools: false,
                supports_vision: false,
                supports_embeddings: false,
                supports_thinking: false,
                kind: Default::default(),
                ..Default::default()
            },
            fail_for,
            call_count: Arc::new(AtomicU32::new(0)),
        }
    }
}

#[async_trait]
impl ProviderAdapter for FailThenSucceedAdapter {
    async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        let n = self.call_count.fetch_add(1, Ordering::Relaxed);
        if n < self.fail_for {
            Err(ProviderError::Unreachable(format!(
                "{} simulated failure #{n}",
                self.meta.name
            )))
        } else {
            Ok(ok_response(&req.model))
        }
    }

    fn metadata(&self) -> &ProviderMetadata {
        &self.meta
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }
}

/// Always-succeeding adapter.
struct AlwaysSucceedAdapter {
    meta: ProviderMetadata,
}

impl AlwaysSucceedAdapter {
    fn new(name: &str, models: Vec<&str>) -> Self {
        Self {
            meta: ProviderMetadata {
                name: name.to_string(),
                supported_models: models.iter().map(|s| s.to_string()).collect(),
                supports_streaming: false,
                supports_tools: false,
                supports_vision: false,
                supports_embeddings: false,
                supports_thinking: false,
                kind: Default::default(),
                ..Default::default()
            },
        }
    }
}

#[async_trait]
impl ProviderAdapter for AlwaysSucceedAdapter {
    async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        Ok(ok_response(&req.model))
    }

    fn metadata(&self) -> &ProviderMetadata {
        &self.meta
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_pricing_db() -> Arc<std::sync::RwLock<PricingDb>> {
    Arc::new(std::sync::RwLock::new(
        PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing must parse"),
    ))
}

fn make_router(
    providers: Vec<Arc<dyn ProviderAdapter>>,
    weights: HashMap<String, f64>,
    retry: RetryConfig,
    fallbacks: Vec<FallbackRule>,
    security: SecurityConfig,
) -> ProviderRouter {
    let names: Vec<String> = providers
        .iter()
        .map(|p| p.metadata().name.clone())
        .collect();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let health = ProviderHealthTracker::new_for_test(&name_refs);
    let routing_config = RoutingConfig {
        weights,
        ..Default::default()
    };
    ProviderRouter::new_with_resilience(
        providers,
        Arc::new(WeightedRandom),
        health,
        make_pricing_db(),
        routing_config,
        retry,
        fallbacks,
        security,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// After `max_retries` are exhausted on the primary provider, the fallback fires.
///
/// Verifies:
///   - HTTP 200 returned (fallback succeeded)
///   - `X-Oxigate-Attempted-Providers: anthropic,openai` header (expose_provider_names=true)
///   - `X-Oxigate-Attempted-Models: claude-sonnet-4-6,gpt-4o` header (model rewrite via Explicit target)
#[tokio::test]
async fn fallback_fires_after_retry_exhaustion_with_headers() {
    let pg = PgContainer::start().await.expect("pg must start");
    let redis = RedisContainer::start().await.expect("redis must start");

    // max_retries=2 → 3 total attempts before exhaustion.
    let primary: Arc<dyn ProviderAdapter> =
        Arc::new(FailThenSucceedAdapter::new("anthropic", vec!["*"], 3));
    let fallback_provider: Arc<dyn ProviderAdapter> =
        Arc::new(AlwaysSucceedAdapter::new("openai", vec!["gpt-4o"]));

    let mut weights = HashMap::new();
    weights.insert("anthropic".into(), 1.0_f64);
    // openai weight=0.0 → excluded from primary selection; reachable only via fallback rule.
    weights.insert("openai".into(), 0.0_f64);

    let router = make_router(
        vec![primary, fallback_provider],
        weights,
        RetryConfig {
            max_retries: 2,
            ..Default::default()
        },
        vec![FallbackRule {
            provider: Some("anthropic".into()),
            model: None,
            targets: vec![FallbackTarget::Explicit {
                provider: "openai".into(),
                model: Some("gpt-4o".into()),
            }],
            key: None,
            on: None,
        }],
        SecurityConfig {
            expose_provider_names: true,
        },
    );

    let gateway = crate::common::gateway::TestGateway::spawn_with_security(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(router),
        SecurityConfig {
            expose_provider_names: true,
        },
    )
    .await;

    let resp = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&serde_json::json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .await;

    resp.assert_status(StatusCode::OK);

    let providers_header = resp
        .headers()
        .get("x-oxigate-attempted-providers")
        .expect("X-Oxigate-Attempted-Providers must be present when expose_provider_names=true")
        .to_str()
        .expect("header must be ASCII");
    assert_eq!(
        providers_header, "anthropic,openai",
        "attempted providers must be anthropic then openai"
    );

    let models_header = resp
        .headers()
        .get("x-oxigate-attempted-models")
        .expect("X-Oxigate-Attempted-Models must be present when expose_provider_names=true")
        .to_str()
        .expect("header must be ASCII");
    assert_eq!(
        models_header, "claude-sonnet-4-6,gpt-4o",
        "attempted models must be original then override"
    );
}

/// When `expose_provider_names=false` (default), neither attempted header is emitted.
#[tokio::test]
async fn attempted_headers_absent_when_expose_disabled() {
    let pg = PgContainer::start().await.expect("pg must start");
    let redis = RedisContainer::start().await.expect("redis must start");

    let primary: Arc<dyn ProviderAdapter> =
        Arc::new(FailThenSucceedAdapter::new("anthropic", vec!["*"], 3));
    // Must support "*" — FallbackTarget::Provider preserves the original model "claude-sonnet-4-6".
    let fallback_provider: Arc<dyn ProviderAdapter> =
        Arc::new(AlwaysSucceedAdapter::new("openai", vec!["*"]));

    let mut weights = HashMap::new();
    weights.insert("anthropic".into(), 1.0_f64);
    weights.insert("openai".into(), 0.0_f64);

    let router = make_router(
        vec![primary, fallback_provider],
        weights,
        RetryConfig {
            max_retries: 2,
            ..Default::default()
        },
        vec![FallbackRule {
            provider: Some("anthropic".into()),
            model: None,
            targets: vec![FallbackTarget::Provider("openai".into())],
            key: None,
            on: None,
        }],
        SecurityConfig {
            expose_provider_names: false, // default
        },
    );

    let gateway = crate::common::gateway::TestGateway::spawn(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(router),
    )
    .await;

    let resp = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&serde_json::json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .await;

    resp.assert_status(StatusCode::OK);

    assert!(
        resp.headers()
            .get("x-oxigate-attempted-providers")
            .is_none(),
        "X-Oxigate-Attempted-Providers must not be present when expose_provider_names=false"
    );
    assert!(
        resp.headers().get("x-oxigate-attempted-models").is_none(),
        "X-Oxigate-Attempted-Models must not be present when expose_provider_names=false"
    );
}

/// A 401 Auth error from the primary provider must trigger the fallback cascade.
///
/// Each provider has its own API key in config — a 401 from Anthropic does NOT imply
/// the OpenAI key is invalid. Auth errors are non-retryable (no same-provider retry)
/// but are fallback-eligible ("any error" triggers the cascade).
///
/// Verifies:
///   - HTTP 200 returned (fallback to openai succeeded)
///   - `X-Oxigate-Attempted-Providers: anthropic,openai` (expose_provider_names=true)
#[tokio::test]
async fn auth_error_triggers_fallback_to_different_provider() {
    struct AuthFailAdapter {
        meta: ProviderMetadata,
    }

    impl AuthFailAdapter {
        fn new(name: &str) -> Self {
            Self {
                meta: ProviderMetadata {
                    name: name.to_string(),
                    supported_models: vec!["*".to_string()],
                    supports_streaming: false,
                    supports_tools: false,
                    supports_vision: false,
                    supports_embeddings: false,
                    supports_thinking: false,
                    kind: Default::default(),
                    ..Default::default()
                },
            }
        }
    }

    #[async_trait::async_trait]
    impl ProviderAdapter for AuthFailAdapter {
        async fn chat_completion(
            &self,
            _req: &oxigate::domain::chat::ChatRequest,
        ) -> Result<oxigate::domain::chat::ChatResponse, ProviderError> {
            Err(ProviderError::Auth("invalid API key".into()))
        }
        fn metadata(&self) -> &ProviderMetadata {
            &self.meta
        }
        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }

    let pg = PgContainer::start().await.expect("pg must start");
    let redis = RedisContainer::start().await.expect("redis must start");

    let primary: Arc<dyn ProviderAdapter> = Arc::new(AuthFailAdapter::new("anthropic"));
    let fallback_provider: Arc<dyn ProviderAdapter> =
        Arc::new(AlwaysSucceedAdapter::new("openai", vec!["*"]));

    let mut weights = HashMap::new();
    weights.insert("anthropic".into(), 1.0_f64);
    weights.insert("openai".into(), 0.0_f64); // reachable only via fallback rule

    let router = make_router(
        vec![primary, fallback_provider],
        weights,
        RetryConfig::default(), // no retries — Auth is non-retryable anyway
        vec![FallbackRule {
            provider: Some("anthropic".into()),
            model: None,
            targets: vec![FallbackTarget::Provider("openai".into())],
            key: None,
            on: None,
        }],
        SecurityConfig {
            expose_provider_names: true,
        },
    );

    let gateway = crate::common::gateway::TestGateway::spawn_with_security(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(router),
        SecurityConfig {
            expose_provider_names: true,
        },
    )
    .await;

    let resp = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .await;

    resp.assert_status(StatusCode::OK);

    let providers_header = resp
        .headers()
        .get("x-oxigate-attempted-providers")
        .expect("X-Oxigate-Attempted-Providers must be present when expose_provider_names=true")
        .to_str()
        .expect("header must be ASCII");
    assert_eq!(
        providers_header, "anthropic,openai",
        "Auth 401 from anthropic must trigger fallback to openai"
    );
}

/// When a compat provider is in the provider list with weight 0.0, it must never be selected.
/// All 20 requests must be served by the named provider, not the zero-weight compat instance.
///
/// This reproduces the pre-period-keyed passthrough opt-in bug scenario: a FallbackOnly provider
/// with explicit weight 0.0 must be excluded from normal routing.
#[tokio::test]
async fn compat_default_weight_zero_excludes_from_routing() {
    use crate::common::stub_adapter::StubAdapter;

    let pg = PgContainer::start().await.expect("pg must start");
    let redis = RedisContainer::start().await.expect("redis must start");

    // Stub with a name that would identify it if selected — any request routed to it will fail.
    let stub: Arc<dyn ProviderAdapter> = Arc::new(StubAdapter::with_name("compat-test", vec!["*"]));

    // Real provider that always succeeds.
    let real: Arc<dyn ProviderAdapter> = Arc::new(AlwaysSucceedAdapter::new("openai", vec!["*"]));

    let mut weights = HashMap::new();
    weights.insert("compat-test".into(), 0.0_f64);
    weights.insert("openai".into(), 1.0_f64);

    let router = make_router(
        vec![stub, real],
        weights,
        RetryConfig::default(),
        vec![],
        SecurityConfig::default(),
    );

    let gateway = crate::common::gateway::TestGateway::spawn(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(router),
    )
    .await;

    for _ in 0..20 {
        let resp = gateway
            .server
            .post(CHAT_COMPLETIONS_PATH)
            .add_header("Authorization", "Bearer sk-test-key")
            .json(&serde_json::json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .await;

        resp.assert_status(StatusCode::OK);
    }
}

/// Embeddings fallback fires after retry exhaustion — proves `dispatch_with_fallback`
/// is correctly wired for the embeddings path end-to-end under the real gateway stack.
///
/// Primary returns `Unreachable` for the first `fail_for` calls; fallback always succeeds.
/// Verifies HTTP 200 and `X-Oxigate-Attempted-Providers` header (expose_provider_names=true).
#[tokio::test]
async fn embeddings_fallback_fires_after_retry_exhaustion() {
    use oxigate::domain::ports::ProviderError;

    struct EmbeddingFailThenSucceed {
        meta: ProviderMetadata,
        fail_for: u32,
        call_count: Arc<AtomicU32>,
    }

    fn ok_embedding(model: &str) -> EmbeddingResponse {
        EmbeddingResponse {
            object: "list".into(),
            data: vec![EmbeddingData {
                object: "embedding".into(),
                embedding: vec![0.1, 0.2, 0.3],
                index: 0,
            }],
            model: model.to_string(),
            usage: None,
        }
    }

    #[async_trait::async_trait]
    impl ProviderAdapter for EmbeddingFailThenSucceed {
        async fn chat_completion(
            &self,
            _req: &oxigate::domain::chat::ChatRequest,
        ) -> Result<oxigate::domain::chat::ChatResponse, ProviderError> {
            Err(ProviderError::NotImplemented)
        }
        fn metadata(&self) -> &ProviderMetadata {
            &self.meta
        }
        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
        async fn embeddings(
            &self,
            req: &EmbeddingRequest,
        ) -> Result<EmbeddingResponse, ProviderError> {
            let n = self.call_count.fetch_add(1, Ordering::Relaxed);
            if n < self.fail_for {
                Err(ProviderError::Unreachable(format!(
                    "{} embedding failure #{n}",
                    self.meta.name
                )))
            } else {
                Ok(ok_embedding(&req.model))
            }
        }
    }

    struct AlwaysSucceedEmbeddingAdapter {
        meta: ProviderMetadata,
    }

    #[async_trait::async_trait]
    impl ProviderAdapter for AlwaysSucceedEmbeddingAdapter {
        async fn chat_completion(
            &self,
            _req: &oxigate::domain::chat::ChatRequest,
        ) -> Result<oxigate::domain::chat::ChatResponse, ProviderError> {
            Err(ProviderError::NotImplemented)
        }
        fn metadata(&self) -> &ProviderMetadata {
            &self.meta
        }
        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
        async fn embeddings(
            &self,
            req: &EmbeddingRequest,
        ) -> Result<EmbeddingResponse, ProviderError> {
            Ok(EmbeddingResponse {
                object: "list".into(),
                data: vec![EmbeddingData {
                    object: "embedding".into(),
                    embedding: vec![1.0],
                    index: 0,
                }],
                model: req.model.clone(),
                usage: None,
            })
        }
    }

    let pg = PgContainer::start().await.expect("pg must start");
    let redis = RedisContainer::start().await.expect("redis must start");

    // max_retries=2 → 3 total attempts on primary before exhaustion.
    let primary: Arc<dyn ProviderAdapter> = Arc::new(EmbeddingFailThenSucceed {
        meta: ProviderMetadata {
            name: "primary".to_string(),
            supported_models: vec!["*".to_string()],
            supports_streaming: false,
            supports_tools: false,
            supports_vision: false,
            supports_embeddings: true,
            supports_thinking: false,
            kind: Default::default(),
            ..Default::default()
        },
        fail_for: 3,
        call_count: Arc::new(AtomicU32::new(0)),
    });
    let fallback_provider: Arc<dyn ProviderAdapter> = Arc::new(AlwaysSucceedEmbeddingAdapter {
        meta: ProviderMetadata {
            name: "fallback".to_string(),
            supported_models: vec!["*".to_string()],
            supports_streaming: false,
            supports_tools: false,
            supports_vision: false,
            supports_embeddings: true,
            supports_thinking: false,
            kind: Default::default(),
            ..Default::default()
        },
    });

    let mut weights = HashMap::new();
    weights.insert("primary".into(), 1.0_f64);
    weights.insert("fallback".into(), 0.0_f64);

    let router = make_router(
        vec![primary, fallback_provider],
        weights,
        RetryConfig {
            max_retries: 2,
            ..Default::default()
        },
        vec![FallbackRule {
            provider: Some("primary".into()),
            model: None,
            targets: vec![FallbackTarget::Provider("fallback".into())],
            key: None,
            on: None,
        }],
        SecurityConfig {
            expose_provider_names: true,
        },
    );

    let gateway = crate::common::gateway::TestGateway::spawn_with_security(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(router),
        SecurityConfig {
            expose_provider_names: true,
        },
    )
    .await;

    let resp = gateway
        .server
        .post(EMBEDDINGS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&serde_json::json!({
            "model": "text-embedding-ada-002",
            "input": "hello world"
        }))
        .await;

    resp.assert_status(StatusCode::OK);

    let providers_header = resp
        .headers()
        .get("x-oxigate-attempted-providers")
        .expect("X-Oxigate-Attempted-Providers must be present when expose_provider_names=true")
        .to_str()
        .expect("header must be ASCII");
    assert_eq!(
        providers_header, "primary,fallback",
        "embeddings fallback must fire after primary retry exhaustion"
    );
}

/// HTTP-level streaming fallback: `stream: true` request where the primary provider fails
/// pre-stream (before yielding any chunks) is transparently rerouted to the fallback.
///
/// Verifies:
///   - HTTP 200 with `Content-Type: text/event-stream`
///   - `X-Oxigate-Attempted-Providers: primary,fallback` (set on response headers before
///     the body stream starts, so axum-test can read them after buffering)
///   - `X-Oxigate-Attempted-Models: gpt-4o,gpt-4o`
///   - Response body contains the SSE chunk from the fallback provider
#[tokio::test]
async fn streaming_fallback_prestream_fires_with_headers() {
    // ── Inline adapters ────────────────────────────────────────────────────

    /// Primary: fails `chat_completion_stream()` for `fail_for` calls, then returns empty.
    struct StreamPreFailAdapter {
        meta: ProviderMetadata,
        fail_for: u32,
        call_count: Arc<AtomicU32>,
    }

    #[async_trait]
    impl ProviderAdapter for StreamPreFailAdapter {
        async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
            Err(ProviderError::NotImplemented)
        }

        async fn chat_completion_stream(
            &self,
            _req: &ChatRequest,
        ) -> Result<ChatCompletionStream, ProviderError> {
            let n = self.call_count.fetch_add(1, Ordering::Relaxed);
            if n < self.fail_for {
                return Err(ProviderError::Unreachable(format!(
                    "primary simulated pre-stream failure #{}",
                    n
                )));
            }
            Ok(Box::pin(futures::stream::empty()))
        }

        fn metadata(&self) -> &ProviderMetadata {
            &self.meta
        }

        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }

    /// Fallback: always yields one `[DONE]` SSE chunk then terminates.
    struct StreamOkAdapter {
        meta: ProviderMetadata,
    }

    #[async_trait]
    impl ProviderAdapter for StreamOkAdapter {
        async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
            Err(ProviderError::NotImplemented)
        }

        async fn chat_completion_stream(
            &self,
            _req: &ChatRequest,
        ) -> Result<ChatCompletionStream, ProviderError> {
            let chunk = StreamChunk::new(Bytes::from_static(b"data: [DONE]\n\n"), None, None);
            Ok(Box::pin(futures::stream::once(async { Ok(chunk) })))
        }

        fn metadata(&self) -> &ProviderMetadata {
            &self.meta
        }

        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }

    // ── Setup ──────────────────────────────────────────────────────────────

    let pg = PgContainer::start().await.expect("pg must start");
    let redis = RedisContainer::start().await.expect("redis must start");

    // max_retries=0 — primary gets 1 attempt then fallback fires immediately.
    let primary: Arc<dyn ProviderAdapter> = Arc::new(StreamPreFailAdapter {
        meta: ProviderMetadata {
            name: "primary".into(),
            supported_models: vec!["gpt-4o".into()],
            supports_streaming: true,
            supports_tools: false,
            supports_vision: false,
            supports_embeddings: false,
            supports_thinking: false,
            kind: Default::default(),
            ..Default::default()
        },
        fail_for: 100, // always fails
        call_count: Arc::new(AtomicU32::new(0)),
    });

    let fallback_provider: Arc<dyn ProviderAdapter> = Arc::new(StreamOkAdapter {
        meta: ProviderMetadata {
            name: "fallback".into(),
            supported_models: vec!["gpt-4o".into()],
            supports_streaming: true,
            supports_tools: false,
            supports_vision: false,
            supports_embeddings: false,
            supports_thinking: false,
            kind: Default::default(),
            ..Default::default()
        },
    });

    let mut weights = HashMap::new();
    weights.insert("primary".into(), 1.0_f64);
    weights.insert("fallback".into(), 0.0_f64); // reachable only via fallback rule

    let router = make_router(
        vec![primary, fallback_provider],
        weights,
        RetryConfig {
            max_retries: 0,
            base_delay_ms: 0,
            jitter_ms: 0,
            ..Default::default()
        },
        vec![FallbackRule {
            provider: Some("primary".into()),
            model: None,
            targets: vec![FallbackTarget::Provider("fallback".into())],
            key: None,
            on: None,
        }],
        SecurityConfig {
            expose_provider_names: true,
        },
    );

    let gateway = crate::common::gateway::TestGateway::spawn_with_security(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(router),
        SecurityConfig {
            expose_provider_names: true,
        },
    )
    .await;

    // ── Exercise ───────────────────────────────────────────────────────────

    let resp = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        }))
        .await;

    // ── Assertions ─────────────────────────────────────────────────────────

    resp.assert_status(StatusCode::OK);

    let content_type = resp
        .headers()
        .get("content-type")
        .expect("Content-Type must be present")
        .to_str()
        .expect("Content-Type must be ASCII");
    assert!(
        content_type.starts_with("text/event-stream"),
        "Content-Type must be text/event-stream; got {content_type}"
    );

    let providers_header = resp
        .headers()
        .get("x-oxigate-attempted-providers")
        .expect("X-Oxigate-Attempted-Providers must be present when expose_provider_names=true")
        .to_str()
        .expect("header must be ASCII");
    assert_eq!(
        providers_header, "primary,fallback",
        "primary must fail and fallback must be selected"
    );

    let models_header = resp
        .headers()
        .get("x-oxigate-attempted-models")
        .expect("X-Oxigate-Attempted-Models must be present when expose_provider_names=true")
        .to_str()
        .expect("header must be ASCII");
    assert_eq!(
        models_header, "gpt-4o,gpt-4o",
        "attempted models must reflect the original model on both attempts"
    );

    // The fallback's [DONE] chunk must appear in the buffered SSE body.
    let body = resp.text();
    assert!(
        body.contains("[DONE]"),
        "response body must contain the SSE [DONE] chunk from the fallback stream; body: {body}"
    );
}
