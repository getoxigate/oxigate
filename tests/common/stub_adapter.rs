// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Stub ProviderAdapter for tests that don't need chat functionality.

use async_trait::async_trait;

use oxigate::domain::chat::{ChatRequest, ChatResponse, StreamChunk};
use oxigate::domain::ports::{
    HealthStatus, ProviderAdapter, ProviderAdapterExt, ProviderError, ProviderMetadata,
};

/// Stub adapter that returns NotImplemented for chat_completion.
/// Used by tests that only exercise health, 404, etc.
pub struct StubAdapter {
    metadata: ProviderMetadata,
}

impl StubAdapter {
    /// Creates a stub adapter. Wrap in `Arc` when passing to `TestGateway::spawn`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: ProviderMetadata {
                name: "stub".to_string(),
                supported_models: vec![],
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

    /// Creates a stub adapter with a configurable name and supported_models list.
    ///
    /// Use this when the test needs a specific provider name in the router or health tracker
    /// but doesn't need real forwarding behaviour (e.g. auth, budget, tagger middleware tests).
    #[must_use]
    pub fn with_name(name: impl Into<String>, models: Vec<&str>) -> Self {
        Self {
            metadata: ProviderMetadata {
                name: name.into(),
                supported_models: models.iter().map(|s| (*s).to_string()).collect(),
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

impl Default for StubAdapter {
    fn default() -> Self {
        Self {
            metadata: ProviderMetadata {
                name: "stub".to_string(),
                supported_models: vec![],
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
impl ProviderAdapter for StubAdapter {
    async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        Err(ProviderError::NotImplemented)
    }

    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }
}

impl ProviderAdapterExt for StubAdapter {}

/// Stub adapter with configurable supported_models for /v1/models tests.
pub struct ModelsTestAdapter {
    metadata: ProviderMetadata,
}

impl ModelsTestAdapter {
    /// Creates an adapter with the given models and provider name.
    #[must_use]
    pub fn new(name: &str, models: Vec<&str>) -> Self {
        Self {
            metadata: ProviderMetadata {
                name: name.to_string(),
                supported_models: models.iter().map(|s| (*s).to_string()).collect(),
                supports_streaming: true,
                supports_tools: true,
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
impl ProviderAdapter for ModelsTestAdapter {
    async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        Err(ProviderError::NotImplemented)
    }

    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }
}

impl ProviderAdapterExt for ModelsTestAdapter {}

/// Stub adapter that yields a configurable stream of chunks.
/// Used for streaming E2E tests (model divergence, mid-stream failure).
pub struct StreamStubAdapter {
    metadata: ProviderMetadata,
    chunks: Vec<Result<StreamChunk, ProviderError>>,
}

impl StreamStubAdapter {
    /// Creates an adapter that yields the given chunks when chat_completion_stream is called.
    #[must_use]
    pub fn new(chunks: Vec<Result<StreamChunk, ProviderError>>) -> Self {
        Self {
            metadata: ProviderMetadata {
                name: "stream-stub".to_string(),
                supported_models: vec!["*".to_string()],
                supports_streaming: true,
                supports_tools: false,
                supports_vision: false,
                supports_embeddings: false,
                supports_thinking: false,
                kind: Default::default(),
                ..Default::default()
            },
            chunks,
        }
    }
}

#[async_trait]
impl ProviderAdapter for StreamStubAdapter {
    async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        Err(ProviderError::NotImplemented)
    }

    async fn chat_completion_stream(
        &self,
        _req: &ChatRequest,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::stream::Stream<Item = Result<StreamChunk, ProviderError>> + Send>,
        >,
        ProviderError,
    > {
        Ok(Box::pin(futures::stream::iter(self.chunks.clone())))
    }

    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }
}

impl ProviderAdapterExt for StreamStubAdapter {}

/// Stub adapter that returns Err from chat_completion_stream (pre-dispatch).
/// Used to test zero-cost headers on streaming error path before any stream starts.
pub struct FailingStreamStubAdapter {
    metadata: ProviderMetadata,
    error: ProviderError,
}

impl FailingStreamStubAdapter {
    /// Creates an adapter that returns the given error from chat_completion_stream.
    #[must_use]
    pub fn new(error: ProviderError) -> Self {
        Self {
            metadata: ProviderMetadata {
                name: "failing-stream-stub".to_string(),
                supported_models: vec!["*".to_string()],
                supports_streaming: true,
                supports_tools: false,
                supports_vision: false,
                supports_embeddings: false,
                supports_thinking: false,
                kind: Default::default(),
                ..Default::default()
            },
            error,
        }
    }
}

#[async_trait]
impl ProviderAdapter for FailingStreamStubAdapter {
    async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        Err(ProviderError::NotImplemented)
    }

    async fn chat_completion_stream(
        &self,
        _req: &ChatRequest,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::stream::Stream<Item = Result<StreamChunk, ProviderError>> + Send>,
        >,
        ProviderError,
    > {
        Err(self.error.clone())
    }

    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }
}

impl ProviderAdapterExt for FailingStreamStubAdapter {}

/// Stub adapter that always returns `AllProvidersRateLimited`.
///
/// Used to exercise the HTTP 503 + `Retry-After` response path end-to-end without
/// needing a real multi-provider cooldown scenario.
pub struct AllRateLimitedStubAdapter {
    metadata: ProviderMetadata,
    retry_after: u64,
}

impl AllRateLimitedStubAdapter {
    /// Creates an adapter that returns `AllProvidersRateLimited { retry_after }`.
    #[must_use]
    pub fn new(retry_after: u64) -> Self {
        Self {
            metadata: ProviderMetadata {
                name: "all-rate-limited-stub".to_string(),
                supported_models: vec!["*".to_string()],
                supports_streaming: false,
                supports_tools: false,
                supports_vision: false,
                supports_embeddings: false,
                supports_thinking: false,
                kind: Default::default(),
                ..Default::default()
            },
            retry_after,
        }
    }
}

#[async_trait]
impl ProviderAdapter for AllRateLimitedStubAdapter {
    async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        Err(ProviderError::AllProvidersRateLimited {
            retry_after: self.retry_after,
        })
    }

    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }
}

impl ProviderAdapterExt for AllRateLimitedStubAdapter {}
