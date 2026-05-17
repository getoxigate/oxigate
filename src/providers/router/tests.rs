// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Unit tests for ProviderRouter dispatch, retry, and fallback logic.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use futures::stream;

use crate::config::{FallbackRule, FallbackTarget, PricingConfig, RoutingConfig};
use crate::domain::chat::{Message, MessageContent, Role, StreamChunk};
use crate::domain::ports::{
    ChatCompletionStream, HealthStatus, ProviderAdapter, ProviderError, ProviderMetadata,
};
use crate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
use crate::domain::routing::WeightedRandom;
use crate::providers::health::ProviderHealthTracker;

use super::*;

fn make_pricing_db() -> Arc<std::sync::RwLock<PricingDb>> {
    Arc::new(std::sync::RwLock::new(
        PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing DB must parse"),
    ))
}

fn test_adapter(name: &str, models: Vec<&str>) -> Arc<dyn ProviderAdapter> {
    struct TestAdapter {
        name: String,
        metadata: ProviderMetadata,
    }
    impl TestAdapter {
        fn new(name: &str, models: Vec<&str>) -> Self {
            let supported_models: Vec<String> = models.iter().map(|s| (*s).to_string()).collect();
            let metadata = ProviderMetadata {
                name: name.to_string(),
                supported_models,
                supports_streaming: false,
                supports_tools: false,
                supports_vision: false,
                supports_embeddings: false,
                supports_thinking: false,
                kind: ProviderKind::Primary,
                ..Default::default()
            };
            Self {
                name: name.to_string(),
                metadata,
            }
        }
    }
    #[async_trait::async_trait]
    impl ProviderAdapter for TestAdapter {
        async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
            Ok(ChatResponse {
                id: "test".into(),
                object: "chat.completion".into(),
                created: 0,
                model: _req.model.clone(),
                choices: vec![crate::domain::chat::Choice {
                    index: 0,
                    message: Message {
                        role: Role::Assistant,
                        content: Some(MessageContent::Text(self.name.clone())),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    finish_reason: Some("stop".into()),
                }],
                usage: crate::domain::chat::Usage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                    completion_tokens_details: None,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                    prompt_tokens_details: None,
                    tier_threshold_override: None,
                    cache_accounting: crate::domain::chat::CacheAccounting::Inclusive,
                    image_units: None,
                    audio_seconds: None,
                    cache_creation_5m_tokens: 0,
                    cache_creation_1h_tokens: 0,
                },
            })
        }
        fn metadata(&self) -> &ProviderMetadata {
            &self.metadata
        }
        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }
    #[async_trait::async_trait]
    impl ProviderAdapterExt for TestAdapter {}
    Arc::new(TestAdapter::new(name, models))
}

/// Test adapter that fails with `RateLimited` for the first `fail_count` calls,
/// then succeeds.
fn failing_adapter(name: &str, models: Vec<&str>, fail_count: u32) -> Arc<dyn ProviderAdapter> {
    struct FailingAdapter {
        name: String,
        metadata: ProviderMetadata,
        calls: Arc<AtomicU32>,
        fail_until: u32,
    }
    #[async_trait::async_trait]
    impl ProviderAdapter for FailingAdapter {
        async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
            let n = self.calls.fetch_add(1, Ordering::Relaxed);
            if n < self.fail_until {
                return Err(ProviderError::RateLimited { retry_after: None });
            }
            Ok(ChatResponse {
                id: "test".into(),
                object: "chat.completion".into(),
                created: 0,
                model: req.model.clone(),
                choices: vec![crate::domain::chat::Choice {
                    index: 0,
                    message: Message {
                        role: Role::Assistant,
                        content: Some(MessageContent::Text(self.name.clone())),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    finish_reason: Some("stop".into()),
                }],
                usage: crate::domain::chat::Usage::default(),
            })
        }
        fn metadata(&self) -> &ProviderMetadata {
            &self.metadata
        }
        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }
    #[async_trait::async_trait]
    impl ProviderAdapterExt for FailingAdapter {}
    let supported_models: Vec<String> = models.iter().map(|s| s.to_string()).collect();
    Arc::new(FailingAdapter {
        name: name.to_string(),
        metadata: ProviderMetadata {
            name: name.to_string(),
            supported_models,
            supports_streaming: false,
            supports_tools: false,
            supports_vision: false,
            supports_embeddings: false,
            supports_thinking: false,
            kind: ProviderKind::Primary,
            ..Default::default()
        },
        calls: Arc::new(AtomicU32::new(0)),
        fail_until: fail_count,
    })
}

fn make_router(providers: Vec<Arc<dyn ProviderAdapter>>) -> ProviderRouter {
    let names: Vec<String> = providers
        .iter()
        .map(|p| p.metadata().name.clone())
        .collect();
    let tracker = ProviderHealthTracker::new(&names, None, 60, 0.1);
    ProviderRouter::new(
        providers,
        Arc::new(WeightedRandom),
        tracker,
        make_pricing_db(),
        RoutingConfig::default(),
    )
}

fn make_request(model: &str) -> ChatRequest {
    ChatRequest {
        model: model.into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("hi".into())),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        max_completion_tokens: None,
        stream: None,
        tools: None,
        parallel_tool_calls: None,
        request_id: None,
        extra: serde_json::Map::new(),
    }
}

#[tokio::test]
async fn test_router_resolves_gpt_to_openai() {
    let openai = test_adapter("openai", vec!["gpt-4o", "gpt-"]);
    let router = make_router(vec![openai]);

    let resp = router
        .chat_completion(&make_request("gpt-4o"))
        .await
        .unwrap();
    assert_eq!(
        resp.choices[0]
            .message
            .content
            .as_ref()
            .and_then(|c| match c {
                MessageContent::Text(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap(),
        "openai"
    );
}

#[tokio::test]
async fn test_router_resolves_gemini_to_google() {
    let gemini = test_adapter("gemini", vec!["gemini-2.0-flash"]);
    let router = make_router(vec![gemini]);

    let resp = router
        .chat_completion(&make_request("gemini-2.0-flash"))
        .await
        .unwrap();
    assert_eq!(
        resp.choices[0]
            .message
            .content
            .as_ref()
            .and_then(|c| match c {
                MessageContent::Text(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap(),
        "gemini"
    );
}

#[tokio::test]
async fn test_router_falls_back_to_compat() {
    // Compat instance has FallbackOnly by default (weight 0.0).
    // Set explicit weight 1.0 so it is reachable for unknown models.
    let openai = test_adapter("openai", vec!["gpt-4o"]);
    let compat = test_adapter("compat-test", vec!["*"]);
    let names = vec!["openai".to_string(), "compat-test".to_string()];
    let tracker = ProviderHealthTracker::new(&names, None, 60, 0.1);
    let routing = RoutingConfig {
        weights: std::collections::HashMap::from([("compat-test".to_string(), 1.0)]),
        ..RoutingConfig::default()
    };
    let router = ProviderRouter::new(
        vec![openai, compat],
        Arc::new(WeightedRandom),
        tracker,
        make_pricing_db(),
        routing,
    );

    let resp = router
        .chat_completion(&make_request("unknown-model-xyz"))
        .await
        .unwrap();
    assert_eq!(
        resp.choices[0]
            .message
            .content
            .as_ref()
            .and_then(|c| match c {
                MessageContent::Text(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap(),
        "compat-test"
    );
}

/// Retry loop: provider fails twice then succeeds; expect success after retry.
#[tokio::test]
async fn test_retry_loop_succeeds_on_third_attempt() {
    let adapter = failing_adapter("openai", vec!["gpt-4o"], 2);
    let names = vec!["openai".to_string()];
    let tracker = ProviderHealthTracker::new(&names, None, 60, 0.1);
    let retry = RetryConfig {
        max_retries: 3,
        base_delay_ms: 0,
        jitter_ms: 0,
        ..RetryConfig::default()
    };
    let router = ProviderRouter::new_with_resilience(
        vec![adapter],
        Arc::new(WeightedRandom),
        tracker,
        make_pricing_db(),
        RoutingConfig::default(),
        retry,
        vec![],
        SecurityConfig::default(),
    );
    let resp = router.chat_completion(&make_request("gpt-4o")).await;
    assert!(resp.is_ok(), "should succeed after retries: {resp:?}");
}

/// Retry loop: provider fails more times than max_retries; expect error.
#[tokio::test]
async fn test_retry_loop_exhausts_and_errors() {
    let adapter = failing_adapter("openai", vec!["gpt-4o"], 5);
    let names = vec!["openai".to_string()];
    let tracker = ProviderHealthTracker::new(&names, None, 60, 0.1);
    let retry = RetryConfig {
        max_retries: 2,
        base_delay_ms: 0,
        jitter_ms: 0,
        ..RetryConfig::default()
    };
    let router = ProviderRouter::new_with_resilience(
        vec![adapter],
        Arc::new(WeightedRandom),
        tracker,
        make_pricing_db(),
        RoutingConfig::default(),
        retry,
        vec![],
        SecurityConfig::default(),
    );
    let err = router
        .chat_completion(&make_request("gpt-4o"))
        .await
        .unwrap_err();
    assert!(
        matches!(err, ProviderError::RateLimited { .. }),
        "expected RateLimited, got {err:?}"
    );
}

/// Fallback cascade: primary always fails, fallback succeeds.
#[tokio::test]
async fn test_fallback_cascade_uses_secondary_on_primary_failure() {
    let primary = failing_adapter("openai", vec!["gpt-4o"], 100);
    let fallback_adp = test_adapter("anthropic", vec!["gpt-4o"]);
    let names = vec!["openai".to_string(), "anthropic".to_string()];
    let tracker = ProviderHealthTracker::new(&names, None, 60, 0.1);
    let retry = RetryConfig {
        max_retries: 0,
        base_delay_ms: 0,
        jitter_ms: 0,
        ..RetryConfig::default()
    };
    let fallbacks = vec![FallbackRule {
        provider: Some("openai".into()),
        model: None,
        targets: vec![FallbackTarget::Provider("anthropic".into())],
        key: None,
        on: None,
    }];
    // Anthropic weight=0.0 so it is never selected as primary — only reachable via fallback.
    // Without this, WeightedRandom might select anthropic directly (both support gpt-4o),
    // which would make the test non-deterministic.
    let routing = RoutingConfig {
        weights: std::collections::HashMap::from([("anthropic".to_string(), 0.0)]),
        ..RoutingConfig::default()
    };
    let router = ProviderRouter::new_with_resilience(
        vec![primary, fallback_adp],
        Arc::new(WeightedRandom),
        tracker,
        make_pricing_db(),
        routing,
        retry,
        fallbacks,
        SecurityConfig::default(),
    );
    let resp = router
        .chat_completion_with_trace(&make_request("gpt-4o"))
        .await;
    assert!(resp.is_ok(), "fallback should succeed: {resp:?}");
    let (_, meta) = resp.unwrap();
    assert_eq!(meta.providers, vec!["openai", "anthropic"]);
    assert_eq!(meta.providers.last().map(String::as_str), Some("anthropic"));
}

/// Helper: adapter that always fails with the given error.
fn const_error_adapter(
    name: &str,
    models: Vec<&str>,
    err: ProviderError,
) -> Arc<dyn ProviderAdapter> {
    struct ConstErrorAdapter {
        metadata: ProviderMetadata,
        err: std::sync::Mutex<ProviderError>,
    }
    #[async_trait::async_trait]
    impl ProviderAdapter for ConstErrorAdapter {
        async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
            // Clone the error by re-matching (ProviderError doesn't impl Clone trivially).
            let err = self.err.lock().unwrap();
            Err(match &*err {
                ProviderError::Auth(s) => ProviderError::Auth(s.clone()),
                ProviderError::ContentFiltered(s) => ProviderError::ContentFiltered(s.clone()),
                ProviderError::RateLimited { retry_after } => ProviderError::RateLimited {
                    retry_after: *retry_after,
                },
                ProviderError::ProviderUnavailable(s) => {
                    ProviderError::ProviderUnavailable(s.clone())
                }
                ProviderError::Timeout { elapsed_ms } => ProviderError::Timeout {
                    elapsed_ms: *elapsed_ms,
                },
                e => ProviderError::Internal(format!("const_error_adapter: {e}")),
            })
        }
        fn metadata(&self) -> &ProviderMetadata {
            &self.metadata
        }
        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }
    #[async_trait::async_trait]
    impl ProviderAdapterExt for ConstErrorAdapter {}
    let supported_models: Vec<String> = models.iter().map(|s| s.to_string()).collect();
    Arc::new(ConstErrorAdapter {
        metadata: ProviderMetadata {
            name: name.to_string(),
            supported_models,
            supports_streaming: false,
            supports_tools: false,
            supports_vision: false,
            supports_embeddings: false,
            supports_thinking: false,
            kind: ProviderKind::Primary,
            ..Default::default()
        },
        err: std::sync::Mutex::new(err),
    })
}

/// Auth error (non-retryable) triggers the fallback cascade .
#[tokio::test]
async fn test_auth_error_triggers_fallback() {
    let primary = const_error_adapter(
        "openai",
        vec!["gpt-4o"],
        ProviderError::Auth("bad key".into()),
    );
    let fallback_adp = test_adapter("anthropic", vec!["gpt-4o"]);
    let names = vec!["openai".to_string(), "anthropic".to_string()];
    let tracker = ProviderHealthTracker::new(&names, None, 60, 0.1);
    let fallbacks = vec![FallbackRule {
        provider: Some("openai".into()),
        model: None,
        targets: vec![FallbackTarget::Provider("anthropic".into())],
        key: None,
        on: None,
    }];
    let routing = RoutingConfig {
        weights: std::collections::HashMap::from([("anthropic".to_string(), 0.0)]),
        ..RoutingConfig::default()
    };
    let router = ProviderRouter::new_with_resilience(
        vec![primary, fallback_adp],
        Arc::new(WeightedRandom),
        tracker,
        make_pricing_db(),
        routing,
        RetryConfig {
            max_retries: 0,
            base_delay_ms: 0,
            jitter_ms: 0,
            ..RetryConfig::default()
        },
        fallbacks,
        SecurityConfig::default(),
    );
    let resp = router
        .chat_completion_with_trace(&make_request("gpt-4o"))
        .await;
    assert!(
        resp.is_ok(),
        "auth error should trigger fallback cascade: {resp:?}"
    );
    let (_, meta) = resp.unwrap();
    assert_eq!(meta.providers.last().map(String::as_str), Some("anthropic"));
    assert_eq!(meta.providers, vec!["openai", "anthropic"]);
}

/// Fallback with model_override populates `attempted_models` with the overridden model.
#[tokio::test]
async fn test_attempted_models_populated_with_model_override() {
    let primary = const_error_adapter(
        "openai",
        vec!["claude-sonnet-4-6"],
        ProviderError::RateLimited { retry_after: None },
    );
    let fallback_adp = test_adapter("anthropic", vec!["gpt-4o"]);
    let names = vec!["openai".to_string(), "anthropic".to_string()];
    let tracker = ProviderHealthTracker::new(&names, None, 60, 0.1);
    let fallbacks = vec![FallbackRule {
        provider: Some("openai".into()),
        model: None,
        targets: vec![FallbackTarget::Explicit {
            provider: "anthropic".into(),
            model: Some("gpt-4o".into()),
        }],
        key: None,
        on: None,
    }];
    let routing = RoutingConfig {
        weights: std::collections::HashMap::from([("anthropic".to_string(), 0.0)]),
        ..RoutingConfig::default()
    };
    let router = ProviderRouter::new_with_resilience(
        vec![primary, fallback_adp],
        Arc::new(WeightedRandom),
        tracker,
        make_pricing_db(),
        routing,
        RetryConfig {
            max_retries: 0,
            base_delay_ms: 0,
            jitter_ms: 0,
            ..RetryConfig::default()
        },
        fallbacks,
        SecurityConfig::default(),
    );
    let resp = router
        .chat_completion_with_trace(&make_request("claude-sonnet-4-6"))
        .await;
    assert!(
        resp.is_ok(),
        "fallback with model override should succeed: {resp:?}"
    );
    let (_, meta) = resp.unwrap();
    assert_eq!(meta.providers, vec!["openai", "anthropic"]);
    assert_eq!(
        meta.models,
        vec!["claude-sonnet-4-6", "gpt-4o"],
        "meta.models must reflect the model override"
    );
}

/// Runtime cycle detection: same provider+model already attempted is skipped with a warn.
#[tokio::test]
async fn test_fallback_same_provider_model_cycle_skipped() {
    // Primary fails; fallback target is the same provider+model. Should be skipped.
    // Because the visit key (provider:model) matches the primary, the router skips
    // the target and exhausts all targets → returns the primary error.
    let primary = const_error_adapter(
        "openai",
        vec!["gpt-4o"],
        ProviderError::RateLimited { retry_after: None },
    );
    let names = vec!["openai".to_string()];
    let tracker = ProviderHealthTracker::new(&names, None, 60, 0.1);
    let fallbacks = vec![FallbackRule {
        provider: Some("openai".into()),
        model: None,
        // Same provider+model as primary → visit-key collision → skipped.
        targets: vec![FallbackTarget::Provider("openai".into())],
        key: None,
        on: None,
    }];
    let router = ProviderRouter::new_with_resilience(
        vec![primary],
        Arc::new(WeightedRandom),
        tracker,
        make_pricing_db(),
        RoutingConfig::default(),
        RetryConfig {
            max_retries: 0,
            base_delay_ms: 0,
            jitter_ms: 0,
            ..RetryConfig::default()
        },
        fallbacks,
        SecurityConfig::default(),
    );
    let result = router.chat_completion(&make_request("gpt-4o")).await;
    assert!(result.is_err(), "cycle target skipped → error returned");
    assert!(matches!(
        result.unwrap_err(),
        ProviderError::RateLimited { .. }
    ));
}

// ─── Streaming helpers ──────────────────────────────────────────────────────

/// Returns an immediately-terminating (empty) `ChatCompletionStream`.
fn empty_stream() -> ChatCompletionStream {
    // Type annotation drives inference: empty stream item = Result<StreamChunk, ProviderError>.
    let _: Option<&StreamChunk> = None; // anchor type for inference
    Box::pin(stream::empty::<Result<StreamChunk, ProviderError>>())
}

/// Streaming adapter that always returns `empty_stream()` successfully.
fn stream_ok_adapter(name: &str, models: Vec<&str>) -> Arc<dyn ProviderAdapter> {
    struct StreamOkAdapter {
        metadata: ProviderMetadata,
    }
    #[async_trait::async_trait]
    impl ProviderAdapter for StreamOkAdapter {
        async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
            Err(ProviderError::NotImplemented)
        }
        async fn chat_completion_stream(
            &self,
            _req: &ChatRequest,
        ) -> Result<ChatCompletionStream, ProviderError> {
            Ok(empty_stream())
        }
        fn metadata(&self) -> &ProviderMetadata {
            &self.metadata
        }
        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }
    #[async_trait::async_trait]
    impl ProviderAdapterExt for StreamOkAdapter {}
    let supported_models: Vec<String> = models.iter().map(|s| s.to_string()).collect();
    Arc::new(StreamOkAdapter {
        metadata: ProviderMetadata {
            name: name.to_string(),
            supported_models,
            supports_streaming: true,
            supports_tools: false,
            supports_vision: false,
            supports_embeddings: false,
            supports_thinking: false,
            kind: ProviderKind::Primary,
            ..Default::default()
        },
    })
}

/// Streaming adapter that fails `chat_completion_stream` for the first `fail_count` calls
/// with `RateLimited`, then returns `empty_stream()`.
fn stream_failing_adapter(
    name: &str,
    models: Vec<&str>,
    fail_count: u32,
) -> Arc<dyn ProviderAdapter> {
    struct StreamFailingAdapter {
        metadata: ProviderMetadata,
        calls: Arc<AtomicU32>,
        fail_until: u32,
    }
    #[async_trait::async_trait]
    impl ProviderAdapter for StreamFailingAdapter {
        async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
            Err(ProviderError::NotImplemented)
        }
        async fn chat_completion_stream(
            &self,
            _req: &ChatRequest,
        ) -> Result<ChatCompletionStream, ProviderError> {
            let n = self.calls.fetch_add(1, Ordering::Relaxed);
            if n < self.fail_until {
                return Err(ProviderError::RateLimited { retry_after: None });
            }
            Ok(empty_stream())
        }
        fn metadata(&self) -> &ProviderMetadata {
            &self.metadata
        }
        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }
    #[async_trait::async_trait]
    impl ProviderAdapterExt for StreamFailingAdapter {}
    let supported_models: Vec<String> = models.iter().map(|s| s.to_string()).collect();
    Arc::new(StreamFailingAdapter {
        metadata: ProviderMetadata {
            name: name.to_string(),
            supported_models,
            supports_streaming: true,
            supports_tools: false,
            supports_vision: false,
            supports_embeddings: false,
            supports_thinking: false,
            kind: ProviderKind::Primary,
            ..Default::default()
        },
        calls: Arc::new(AtomicU32::new(0)),
        fail_until: fail_count,
    })
}

fn make_resilience_router(
    providers: Vec<Arc<dyn ProviderAdapter>>,
    retry: RetryConfig,
    fallbacks: Vec<FallbackRule>,
    routing: RoutingConfig,
) -> ProviderRouter {
    let names: Vec<String> = providers
        .iter()
        .map(|p| p.metadata().name.clone())
        .collect();
    let tracker = ProviderHealthTracker::new(&names, None, 60, 0.1);
    ProviderRouter::new_with_resilience(
        providers,
        Arc::new(WeightedRandom),
        tracker,
        make_pricing_db(),
        routing,
        retry,
        fallbacks,
        SecurityConfig::default(),
    )
}

fn zero_delay_retry(max_retries: u32) -> RetryConfig {
    RetryConfig {
        max_retries,
        base_delay_ms: 0,
        jitter_ms: 0,
        ..RetryConfig::default()
    }
}

// ─── dispatch_stream_with_fallback tests ────────────────────────────────────

/// Pre-stream failure on primary → fallback stream is selected; both providers in attempted.
#[tokio::test]
async fn test_stream_fallback_on_primary_pre_stream_failure() {
    let primary = stream_failing_adapter("openai", vec!["gpt-4o"], 100); // always fails
    let fallback_adp = stream_ok_adapter("anthropic", vec!["gpt-4o"]);
    let fallbacks = vec![FallbackRule {
        provider: Some("openai".into()),
        model: None,
        targets: vec![FallbackTarget::Provider("anthropic".into())],
        key: None,
        on: None,
    }];
    let routing = RoutingConfig {
        weights: std::collections::HashMap::from([("anthropic".to_string(), 0.0)]),
        ..RoutingConfig::default()
    };
    let router = make_resilience_router(
        vec![primary, fallback_adp],
        zero_delay_retry(0),
        fallbacks,
        routing,
    );

    let result = router
        .dispatch_stream_with_fallback(&make_request("gpt-4o"), None)
        .await;
    assert!(result.is_ok(), "fallback stream should succeed");
    let (_, trace) = result.unwrap();
    let dispatched: Vec<&str> = trace
        .attempts
        .iter()
        .filter(|a| a.attempted)
        .map(|a| a.provider.as_str())
        .collect();
    assert_eq!(dispatched, vec!["openai", "anthropic"]);
    let dispatched_models: Vec<&str> = trace
        .attempts
        .iter()
        .filter(|a| a.attempted)
        .map(|a| a.model.as_str())
        .collect();
    assert_eq!(dispatched_models, vec!["gpt-4o", "gpt-4o"]);
}

/// Primary fails twice (retried), then succeeds; no fallback provider appears in attempted.
#[tokio::test]
async fn test_stream_primary_retry_then_success_no_fallback() {
    let primary = stream_failing_adapter("openai", vec!["gpt-4o"], 2); // fails 2×, then ok
    let fallback_adp = stream_ok_adapter("anthropic", vec!["gpt-4o"]);
    let fallbacks = vec![FallbackRule {
        provider: Some("openai".into()),
        model: None,
        targets: vec![FallbackTarget::Provider("anthropic".into())],
        key: None,
        on: None,
    }];
    let routing = RoutingConfig {
        weights: std::collections::HashMap::from([("anthropic".to_string(), 0.0)]),
        ..RoutingConfig::default()
    };
    let router = make_resilience_router(
        vec![primary, fallback_adp],
        zero_delay_retry(3),
        fallbacks,
        routing,
    );

    let result = router
        .dispatch_stream_with_fallback(&make_request("gpt-4o"), None)
        .await;
    assert!(result.is_ok(), "primary should succeed after retries");
    let (_, trace) = result.unwrap();
    // With retries, trace now contains one entry per dispatch iteration (attempt 0 + 2 retries).
    assert!(
        trace.attempts.iter().all(|a| a.provider == "openai"),
        "fallback must not appear when primary retries succeed; got {:?}",
        trace
            .attempts
            .iter()
            .map(|a| &a.provider)
            .collect::<Vec<_>>()
    );
    assert!(
        trace.attempts.iter().any(|a| !a.is_retry),
        "first attempt must have is_retry=false"
    );
    assert!(
        trace.attempts.iter().any(|a| a.is_retry),
        "retry attempts must have is_retry=true"
    );
}

/// Fallback provider that does not support the model must NOT appear in X-Oxigate-Attempted-Providers.
/// Regression test for C1: attempted.push() must happen after the model-support check.
#[tokio::test]
async fn test_stream_skipped_unsupported_fallback_not_in_attempted() {
    // groq supports "other-model" only → model check fails → must NOT appear in attempted.
    // anthropic supports "gpt-4o" → dispatched and succeeds.
    let primary = stream_failing_adapter("openai", vec!["gpt-4o"], 100);
    let groq = stream_ok_adapter("groq", vec!["other-model"]);
    let anthropic = stream_ok_adapter("anthropic", vec!["gpt-4o"]);
    let fallbacks = vec![FallbackRule {
        provider: Some("openai".into()),
        model: None,
        targets: vec![
            FallbackTarget::Provider("groq".into()),
            FallbackTarget::Provider("anthropic".into()),
        ],
        key: None,
        on: None,
    }];
    let routing = RoutingConfig {
        weights: std::collections::HashMap::from([
            ("groq".to_string(), 0.0),
            ("anthropic".to_string(), 0.0),
        ]),
        ..RoutingConfig::default()
    };
    let router = make_resilience_router(
        vec![primary, groq, anthropic],
        zero_delay_retry(0),
        fallbacks,
        routing,
    );

    let result = router
        .dispatch_stream_with_fallback(&make_request("gpt-4o"), None)
        .await;
    assert!(result.is_ok(), "should succeed via anthropic");
    let (_, trace) = result.unwrap();
    let dispatched_providers: Vec<&str> = trace
        .attempts
        .iter()
        .filter(|a| a.attempted)
        .map(|a| a.provider.as_str())
        .collect();
    assert!(
        !dispatched_providers.contains(&"groq"),
        "groq was skipped (model unsupported) and must NOT be in attempted; got {:?}",
        dispatched_providers
    );
    assert_eq!(dispatched_providers, vec!["openai", "anthropic"]);
}

// ───: trigger-specific fallback + retry policy tests ─────────────────

/// `fallback.on = Some([RateLimit])`: primary fails with ProviderUnavailable (503) →
/// trigger is NOT in the on-list → all targets skipped → AbortedByPolicy.
#[tokio::test]
async fn test_fallback_on_filter_blocks_wrong_trigger() {
    use crate::config::FallbackTrigger;
    use crate::providers::router::fallback_trace::{DecisionOutcome, SkipReason};

    let primary = const_error_adapter(
        "openai",
        vec!["gpt-4o"],
        ProviderError::ProviderUnavailable("503".into()),
    );
    let fallback_adp = test_adapter("anthropic", vec!["gpt-4o"]);
    let fallbacks = vec![FallbackRule {
        provider: Some("openai".into()),
        model: None,
        targets: vec![FallbackTarget::Provider("anthropic".into())],
        key: Some("block-503".into()),
        // Only fire on RateLimit — ProviderUnavailable must be blocked.
        on: Some(vec![FallbackTrigger::RateLimit]),
    }];
    let routing = RoutingConfig {
        weights: std::collections::HashMap::from([("anthropic".to_string(), 0.0)]),
        ..RoutingConfig::default()
    };
    let router = make_resilience_router(
        vec![primary, fallback_adp],
        zero_delay_retry(0),
        fallbacks,
        routing,
    );

    let (result, trace) = router
        .dispatch_with_fallback(&make_request("gpt-4o"), |adapter, r| async move {
            adapter.chat_completion(&r).await
        })
        .await;

    // Request must fail — trigger filter blocked the fallback.
    assert!(result.is_err(), "should fail when trigger not in on-list");
    assert_eq!(trace.outcome, DecisionOutcome::AbortedByPolicy);
    // All targets recorded as skipped.
    assert!(
        trace
            .attempts
            .iter()
            .any(|a| !a.attempted && a.skip_reason == Some(SkipReason::TriggerNotAllowed)),
        "expected TriggerNotAllowed skip in attempts"
    );
}

/// `fallback.on = Some([RateLimit])`: primary fails with RateLimited → trigger IS in list →
/// fallback dispatched → outcome Success.
#[tokio::test]
async fn test_fallback_on_filter_allows_matching_trigger() {
    use crate::config::FallbackTrigger;
    use crate::providers::router::fallback_trace::DecisionOutcome;

    let primary = const_error_adapter(
        "openai",
        vec!["gpt-4o"],
        ProviderError::RateLimited { retry_after: None },
    );
    let fallback_adp = test_adapter("anthropic", vec!["gpt-4o"]);
    let fallbacks = vec![FallbackRule {
        provider: Some("openai".into()),
        model: None,
        targets: vec![FallbackTarget::Provider("anthropic".into())],
        key: None,
        on: Some(vec![FallbackTrigger::RateLimit]),
    }];
    let routing = RoutingConfig {
        weights: std::collections::HashMap::from([("anthropic".to_string(), 0.0)]),
        ..RoutingConfig::default()
    };
    let router = make_resilience_router(
        vec![primary, fallback_adp],
        zero_delay_retry(0),
        fallbacks,
        routing,
    );

    let (result, trace) = router
        .dispatch_with_fallback(&make_request("gpt-4o"), |adapter, r| async move {
            adapter.chat_completion(&r).await
        })
        .await;

    assert!(
        result.is_ok(),
        "fallback should succeed when trigger matches: {trace:?}"
    );
    assert_eq!(trace.outcome, DecisionOutcome::Success);
    let dispatched: Vec<&str> = trace
        .attempts
        .iter()
        .filter(|a| a.attempted)
        .map(|a| a.provider.as_str())
        .collect();
    assert_eq!(dispatched, vec!["openai", "anthropic"]);
}

/// `fallback.on = None` (default): any error triggers fallback (backward compat).
#[tokio::test]
async fn test_fallback_on_none_triggers_for_any_error() {
    use crate::providers::router::fallback_trace::DecisionOutcome;

    let primary = const_error_adapter(
        "openai",
        vec!["gpt-4o"],
        ProviderError::ProviderUnavailable("any".into()),
    );
    let fallback_adp = test_adapter("anthropic", vec!["gpt-4o"]);
    let fallbacks = vec![FallbackRule {
        provider: Some("openai".into()),
        model: None,
        targets: vec![FallbackTarget::Provider("anthropic".into())],
        key: None,
        on: None, // any-error
    }];
    let routing = RoutingConfig {
        weights: std::collections::HashMap::from([("anthropic".to_string(), 0.0)]),
        ..RoutingConfig::default()
    };
    let router = make_resilience_router(
        vec![primary, fallback_adp],
        zero_delay_retry(0),
        fallbacks,
        routing,
    );

    let (result, trace) = router
        .dispatch_with_fallback(&make_request("gpt-4o"), |adapter, r| async move {
            adapter.chat_completion(&r).await
        })
        .await;

    assert!(result.is_ok(), "any-error fallback should fire: {trace:?}");
    assert_eq!(trace.outcome, DecisionOutcome::Success);
}

/// Streaming path: primary fails with RateLimited, rule `on: [Timeout]` (non-matching)
/// → AbortedByPolicy on the streaming dispatch path.
#[tokio::test]
async fn test_stream_fallback_on_filter_aborts_by_policy() {
    use crate::config::FallbackTrigger;

    // stream_failing_adapter emits RateLimited; the rule only fires on Timeout → blocked.
    let primary = stream_failing_adapter("openai", vec!["gpt-4o"], 100);
    let fallback_adp = stream_ok_adapter("anthropic", vec!["gpt-4o"]);
    let fallbacks = vec![FallbackRule {
        provider: Some("openai".into()),
        model: None,
        targets: vec![FallbackTarget::Provider("anthropic".into())],
        key: Some("timeout-only".into()),
        on: Some(vec![FallbackTrigger::Timeout]), // blocks RateLimited
    }];
    let routing = RoutingConfig {
        weights: std::collections::HashMap::from([("anthropic".to_string(), 0.0)]),
        ..RoutingConfig::default()
    };
    let router = make_resilience_router(
        vec![primary, fallback_adp],
        zero_delay_retry(0),
        fallbacks,
        routing,
    );

    let result = router
        .dispatch_stream_with_fallback(&make_request("gpt-4o"), None)
        .await;

    // Must fail — trigger filter blocked the fallback cascade.
    assert!(
        result.is_err(),
        "streaming fallback should be blocked by trigger filter"
    );
}

/// Acceptance criterion for per-attempt FetchAttempt tracking:
/// max_retries=2, fails twice then succeeds → 3 FetchAttempt entries (attempt 0 + 2 retries).
#[tokio::test]
async fn test_retry_loop_produces_one_fetch_attempt_per_dispatch() {
    use crate::providers::router::fallback_trace::FetchAttempt;
    let primary = failing_adapter("openai", vec!["gpt-4o"], 2); // fails 2×, then succeeds
    let router = make_resilience_router(
        vec![primary],
        zero_delay_retry(2),
        vec![],
        RoutingConfig::default(),
    );

    let (result, trace) = router
        .dispatch_with_fallback(&make_request("gpt-4o"), |adapter, r| async move {
            adapter.chat_completion(&r).await
        })
        .await;

    assert!(result.is_ok(), "should succeed after 2 retries");
    // 3 dispatched attempts: attempt 0 (is_retry=false) + retry 1 + retry 2.
    let dispatched: Vec<&FetchAttempt> = trace.attempts.iter().filter(|a| a.attempted).collect();
    assert_eq!(
        dispatched.len(),
        3,
        "expected 3 FetchAttempts; got {:?}",
        trace.attempts
    );
    assert!(
        !dispatched[0].is_retry,
        "attempt 0 must have is_retry=false"
    );
    assert!(dispatched[1].is_retry, "attempt 1 must have is_retry=true");
    assert!(dispatched[2].is_retry, "attempt 2 must have is_retry=true");
    // The trigger on retry attempts should be set from the previous failure.
    assert!(
        dispatched[1].trigger.is_some(),
        "retry attempt 1 must carry the trigger from the preceding failure"
    );
}

/// max_retries=0: exactly 1 FetchAttempt with is_retry=false — no regression.
#[tokio::test]
async fn test_retry_loop_max_retries_zero_produces_one_attempt() {
    use crate::providers::router::fallback_trace::FetchAttempt;
    let primary = failing_adapter("openai", vec!["gpt-4o"], 100); // always fails
    let router = make_resilience_router(
        vec![primary],
        zero_delay_retry(0),
        vec![],
        RoutingConfig::default(),
    );

    let (result, trace) = router
        .dispatch_with_fallback(&make_request("gpt-4o"), |adapter, r| async move {
            adapter.chat_completion(&r).await
        })
        .await;

    assert!(result.is_err(), "should fail with no retries");
    let dispatched: Vec<&FetchAttempt> = trace.attempts.iter().filter(|a| a.attempted).collect();
    assert_eq!(
        dispatched.len(),
        1,
        "exactly 1 attempt expected with max_retries=0"
    );
    assert!(
        !dispatched[0].is_retry,
        "sole attempt must have is_retry=false"
    );
}

/// `retry.on = Some([RateLimit])`: primary fails with ProviderUnavailable (non-matching
/// trigger) → retry is suppressed → error propagates after 0 retries despite max_retries=3.
#[tokio::test]
async fn test_retry_on_filter_suppresses_non_matching_trigger() {
    use crate::config::FallbackTrigger;

    // fails_with_unavailable: always returns ProviderUnavailable
    let primary = const_error_adapter(
        "openai",
        vec!["gpt-4o"],
        ProviderError::ProviderUnavailable("503".into()),
    );
    let retry = RetryConfig {
        max_retries: 3,
        base_delay_ms: 0,
        jitter_ms: 0,
        on: Some(vec![FallbackTrigger::RateLimit]), // only retry RateLimit
        ..RetryConfig::default()
    };
    let router = make_resilience_router(vec![primary], retry, vec![], RoutingConfig::default());

    // dispatch_with_fallback drives retry_loop internally; check call count via
    // wrapping in a counting adapter is complex — verify the result instead.
    let (result, trace) = router
        .dispatch_with_fallback(&make_request("gpt-4o"), |adapter, r| async move {
            adapter.chat_completion(&r).await
        })
        .await;

    assert!(
        result.is_err(),
        "should fail: retry.on filter blocked retry"
    );
    // Primary attempt is the only dispatched attempt (no retries, no fallbacks).
    let dispatched_count = trace.attempts.iter().filter(|a| a.attempted).count();
    assert_eq!(
        dispatched_count, 1,
        "retry suppressed — only one attempt expected"
    );
}

// Config validation tests for `on` lists live in config.rs's #[cfg(test)] module
// (see test_fallback_rule_empty_on_list_is_rejected and test_retry_on_empty_list_is_rejected).

/// `best_matching_rule`: provider+model rule wins over provider-only rule.
#[tokio::test]
async fn test_best_matching_rule_provider_plus_model_wins() {
    let names = vec!["openai".to_string(), "a".to_string(), "b".to_string()];
    let tracker = ProviderHealthTracker::new(&names, None, 60, 0.1);
    let fallbacks = vec![
        FallbackRule {
            provider: Some("openai".into()),
            model: None,
            targets: vec![FallbackTarget::Provider("a".into())],
            key: None,
            on: None,
        },
        FallbackRule {
            provider: Some("openai".into()),
            model: Some("gpt-4o".into()),
            targets: vec![FallbackTarget::Provider("b".into())],
            key: None,
            on: None,
        },
    ];
    let router = ProviderRouter::new_with_resilience(
        vec![
            test_adapter("openai", vec!["gpt-4o"]),
            test_adapter("a", vec!["gpt-4o"]),
            test_adapter("b", vec!["gpt-4o"]),
        ],
        Arc::new(WeightedRandom),
        tracker,
        make_pricing_db(),
        RoutingConfig::default(),
        RetryConfig::default(),
        fallbacks,
        SecurityConfig::default(),
    );
    let matched = router.best_matching_rule_indexed("openai", "gpt-4o");
    let (_, rule) = matched.expect("should find a matching rule");
    assert_eq!(rule.targets.len(), 1);
    assert_eq!(
        rule.targets[0].provider_name(),
        "b",
        "provider+model rule should win"
    );
}

// -----------------------------------------------------------------------
// Bug regression tests (critical fixes)
// -----------------------------------------------------------------------

/// Bug 1 regression: `record_retry` must not be called on the last loop iteration
/// (when no retry will actually follow). With max_retries=0 and a single retryable
/// failure, zero retry metrics should be emitted (verified indirectly: the
/// FetchAttempt count stays at 1 and is_retry=false).
#[tokio::test]
async fn test_retry_metric_not_emitted_when_no_retry_follows() {
    let primary = const_error_adapter(
        "openai",
        vec!["gpt-4o"],
        ProviderError::RateLimited { retry_after: None },
    );
    let router = make_resilience_router(
        vec![primary],
        zero_delay_retry(0), // max_retries=0: the only attempt IS the last iteration
        vec![],
        RoutingConfig::default(),
    );

    let (result, trace) = router
        .dispatch_with_fallback(&make_request("gpt-4o"), |adapter, r| async move {
            adapter.chat_completion(&r).await
        })
        .await;

    assert!(result.is_err());
    let dispatched: Vec<_> = trace.attempts.iter().filter(|a| a.attempted).collect();
    // Exactly one attempt, not a retry — confirms the loop didn't double-count.
    assert_eq!(dispatched.len(), 1);
    assert!(
        !dispatched[0].is_retry,
        "sole attempt must not be marked is_retry"
    );
}

/// Bug 2 regression: primary retries must not be misclassified as fallback attempts.
/// After max_retries=2 on the primary with no fallback rule configured, the trace
/// must report 3 dispatched attempts (attempt 0 + 2 retries) and fallback_dispatched=false.
#[tokio::test]
async fn test_primary_retries_not_misclassified_as_fallback() {
    use crate::providers::router::fallback_trace::FetchAttempt;

    let primary = failing_adapter("openai", vec!["gpt-4o"], 10); // always fails
    let router = make_resilience_router(
        vec![primary],
        zero_delay_retry(2), // 1 initial + 2 retries = 3 attempts
        vec![],              // no fallback rules
        RoutingConfig::default(),
    );

    let (result, trace) = router
        .dispatch_with_fallback(&make_request("gpt-4o"), |adapter, r| async move {
            adapter.chat_completion(&r).await
        })
        .await;

    assert!(result.is_err());
    let dispatched: Vec<&FetchAttempt> = trace.attempts.iter().filter(|a| a.attempted).collect();
    assert_eq!(dispatched.len(), 3, "primary + 2 retries = 3 attempts");
    // None of the retry attempts should look like a fallback dispatch.
    // Verify by checking meta_from_trace produces no fallback_trigger.
    let meta = meta_from_trace(&trace);
    assert!(
        meta.fallback_trigger.is_none(),
        "primary retries must not produce a fallback_trigger; got {:?}",
        meta.fallback_trigger
    );
    assert!(
        !meta.fallback_dispatched,
        "fallback_dispatched must be false when only primary retries occurred"
    );
}

/// Bug 3 regression: dedup in meta_from_trace must be by (provider, model), not just provider.
/// When primary retries with a different model override, both must appear in AttemptedMeta.
/// Simulated by constructing a FallbackDecisionTrace directly.
#[test]
fn test_meta_from_trace_dedup_by_provider_and_model() {
    use crate::config::FallbackTrigger;
    use crate::providers::router::fallback_trace::{
        DecisionOutcome, FallbackDecisionTrace, FetchAttempt,
    };

    // Two consecutive attempts from "openai" but with different models.
    // Should NOT be collapsed: each (provider, model) pair is distinct.
    let trace = FallbackDecisionTrace {
        source_provider: "openai".into(),
        source_model: "gpt-4o".into(),
        trigger: Some(FallbackTrigger::RateLimit),
        matched_rule_index: Some(0),
        matched_rule_key: None,
        attempts: vec![
            FetchAttempt {
                provider: "openai".into(),
                model: "gpt-4o".into(),
                is_retry: false,
                trigger: None,
                attempted: true,
                skip_reason: None,
                error_class: Some("rate_limit".into()),
                latency_ms: Some(50.0),
            },
            // Same provider, different model (fallback target with model_override).
            FetchAttempt {
                provider: "openai".into(),
                model: "gpt-4o-mini".into(),
                is_retry: false,
                trigger: Some(FallbackTrigger::RateLimit),
                attempted: true,
                skip_reason: None,
                error_class: None,
                latency_ms: Some(40.0),
            },
        ],
        outcome: DecisionOutcome::Success,
    };

    let meta = meta_from_trace(&trace);
    assert_eq!(
        meta.providers,
        vec!["openai", "openai"],
        "both (openai,gpt-4o) and (openai,gpt-4o-mini) must appear"
    );
    assert_eq!(
        meta.models,
        vec!["gpt-4o", "gpt-4o-mini"],
        "both models must appear in the models header"
    );
}

/// Bug 3 regression (inverse): same (provider, model) pair repeated as retries
/// must be collapsed to one entry in AttemptedMeta.
#[test]
fn test_meta_from_trace_collapses_same_provider_model_retries() {
    use crate::providers::router::fallback_trace::{
        DecisionOutcome, FallbackDecisionTrace, FetchAttempt,
    };

    let trace = FallbackDecisionTrace {
        source_provider: "openai".into(),
        source_model: "gpt-4o".into(),
        trigger: None,
        matched_rule_index: None,
        matched_rule_key: None,
        attempts: vec![
            FetchAttempt {
                provider: "openai".into(),
                model: "gpt-4o".into(),
                is_retry: false,
                trigger: None,
                attempted: true,
                skip_reason: None,
                error_class: None,
                latency_ms: Some(50.0),
            },
            FetchAttempt {
                provider: "openai".into(),
                model: "gpt-4o".into(),
                is_retry: true,
                trigger: None,
                attempted: true,
                skip_reason: None,
                error_class: None,
                latency_ms: Some(45.0),
            },
        ],
        outcome: DecisionOutcome::Success,
    };

    let meta = meta_from_trace(&trace);
    assert_eq!(
        meta.providers,
        vec!["openai"],
        "retries on same (provider, model) must be collapsed to one entry"
    );
    assert_eq!(meta.models, vec!["gpt-4o"]);
}

/// Health cooldown fix regression: when retry.on blocks retrying a transient error,
/// on_rate_limit() must still be called so the provider enters cooldown and subsequent
/// routing decisions don't favour a known-bad provider.
///
/// Build the router with an explicit tracker reference so the test can call candidates()
/// after the dispatch to observe whether the provider was marked in cooldown.
#[tokio::test]
async fn test_health_cooldown_updated_even_when_retry_on_blocks_retry() {
    use crate::config::FallbackTrigger;
    use crate::domain::ports::ProviderCandidate;

    let provider = const_error_adapter(
        "openai",
        vec!["gpt-4o"],
        ProviderError::RateLimited { retry_after: None },
    );
    let names = vec!["openai".to_string()];
    // Long cooldown so the provider stays in cooldown within the test.
    let tracker = Arc::new(ProviderHealthTracker::new(&names, None, 3600, 0.1));
    let retry = RetryConfig {
        max_retries: 3,
        base_delay_ms: 0,
        jitter_ms: 0,
        on: Some(vec![FallbackTrigger::Timeout]), // RateLimit is NOT in the list
        ..RetryConfig::default()
    };
    let router = ProviderRouter::new_with_resilience(
        vec![Arc::clone(&provider)],
        Arc::new(WeightedRandom),
        Arc::clone(&tracker),
        make_pricing_db(),
        RoutingConfig::default(),
        retry,
        vec![],
        SecurityConfig::default(),
    );

    let (result, _trace) = router
        .dispatch_with_fallback(&make_request("gpt-4o"), |adapter, r| async move {
            adapter.chat_completion(&r).await
        })
        .await;

    assert!(result.is_err(), "should fail: retry blocked by retry.on");

    // After the dispatch the provider must be in cooldown despite no retry occurring.
    let candidates: Vec<ProviderCandidate> = tracker
        .candidates(
            &[Arc::clone(&provider)],
            &Default::default(),
            "gpt-4o",
            &make_pricing_db(),
        )
        .await;
    // In cooldown → candidates() may return the provider with is_cooling_down=true
    // (HalfOpen probe) or return an empty list (fully closed). Either way it must not
    // be offered as a healthy candidate (is_cooling_down must not be false).
    let healthy_candidate = candidates.iter().find(|c| !c.is_cooling_down);
    assert!(
        healthy_candidate.is_none(),
        "provider should be in cooldown after a RateLimited error, \
         even when retry.on blocked the retry"
    );
}

/// Timeout must trigger on_rate_limit() in retry_loop just like RateLimited/ProviderUnavailable,
/// consistent with fallback.rs and streaming.rs. A timing-out provider should accumulate
/// cooldown pressure so it is not aggressively re-selected in subsequent routing decisions.
#[tokio::test]
async fn test_health_cooldown_updated_on_timeout_in_retry_loop() {
    use crate::domain::ports::ProviderCandidate;

    let provider = const_error_adapter(
        "openai",
        vec!["gpt-4o"],
        ProviderError::Timeout { elapsed_ms: 5000 },
    );
    let names = vec!["openai".to_string()];
    let tracker = Arc::new(ProviderHealthTracker::new(&names, None, 3600, 0.1));
    let router = ProviderRouter::new_with_resilience(
        vec![Arc::clone(&provider)],
        Arc::new(WeightedRandom),
        Arc::clone(&tracker),
        make_pricing_db(),
        RoutingConfig::default(),
        zero_delay_retry(1), // one retry so the loop runs the health-update path
        vec![],
        SecurityConfig::default(),
    );

    let (result, _trace) = router
        .dispatch_with_fallback(&make_request("gpt-4o"), |adapter, r| async move {
            adapter.chat_completion(&r).await
        })
        .await;

    assert!(result.is_err());

    let candidates: Vec<ProviderCandidate> = tracker
        .candidates(
            &[Arc::clone(&provider)],
            &Default::default(),
            "gpt-4o",
            &make_pricing_db(),
        )
        .await;
    let healthy_candidate = candidates.iter().find(|c| !c.is_cooling_down);
    assert!(
        healthy_candidate.is_none(),
        "provider should be in cooldown after Timeout errors in retry_loop"
    );
}

/// Verifies that a `FallbackOnly` adapter with wildcard `supported_models: ["*"]`
/// is never selected by primary routing. Its weight defaults to 0.0 in `WeightedRandom`,
/// so the router must return an `Internal` error rather than dispatching to it.
#[tokio::test]
async fn test_fallback_only_wildcard_not_selected_for_primary_routing() {
    struct FallbackOnlyAdapter {
        metadata: ProviderMetadata,
    }
    #[async_trait::async_trait]
    impl ProviderAdapter for FallbackOnlyAdapter {
        async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
            Ok(ChatResponse {
                id: "fallback".into(),
                object: "chat.completion".into(),
                created: 0,
                model: req.model.clone(),
                choices: vec![],
                usage: crate::domain::chat::Usage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                    completion_tokens_details: None,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                    prompt_tokens_details: None,
                    tier_threshold_override: None,
                    cache_accounting: crate::domain::chat::CacheAccounting::Inclusive,
                    image_units: None,
                    audio_seconds: None,
                    cache_creation_5m_tokens: 0,
                    cache_creation_1h_tokens: 0,
                },
            })
        }
        fn metadata(&self) -> &ProviderMetadata {
            &self.metadata
        }
        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }
    #[async_trait::async_trait]
    impl ProviderAdapterExt for FallbackOnlyAdapter {}

    let adapter: Arc<dyn ProviderAdapter> = Arc::new(FallbackOnlyAdapter {
        metadata: ProviderMetadata {
            name: "compat-wildcard".to_string(),
            supported_models: vec!["*".to_string()],
            supports_streaming: false,
            supports_tools: false,
            supports_vision: false,
            supports_embeddings: false,
            supports_thinking: false,
            kind: ProviderKind::FallbackOnly,
            ..Default::default()
        },
    });

    let router = make_router(vec![adapter]);
    let err = router
        .chat_completion(&make_request("gpt-4o"))
        .await
        .unwrap_err();

    assert!(
        matches!(err, ProviderError::Internal(_)),
        "FallbackOnly wildcard adapter must not be selected for primary routing; got: {err:?}"
    );
}
