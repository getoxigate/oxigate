// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Stub ProviderAdapter for tests that don't need chat functionality.

use std::time::Duration;

use async_stream::stream;
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
///
/// The two delay builders exist so a test can decide *where* the stream is parked when the
/// client goes away. Without them every chunk is ready on the first poll, the whole body is
/// written into the socket before the client can react, and a test that means to disconnect
/// part-way silently exercises the read-to-completion path instead.
pub struct StreamStubAdapter {
    metadata: ProviderMetadata,
    chunks: Vec<Result<StreamChunk, ProviderError>>,
    inter_chunk_delay: Duration,
    trailing_hold: Duration,
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
            inter_chunk_delay: Duration::ZERO,
            trailing_hold: Duration::ZERO,
        }
    }

    /// Overrides the provider name reported by `metadata()`.
    ///
    /// Metric assertions need it: cost counters are keyed by provider, with no request
    /// dimension, so tests that read a counter delta must not share a label with another test
    /// running in the same process.
    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.metadata.name = name.to_string();
        self
    }

    /// Waits this long before yielding each chunk after the first.
    #[must_use]
    pub fn with_inter_chunk_delay(mut self, delay: Duration) -> Self {
        self.inter_chunk_delay = delay;
        self
    }

    /// Waits this long after the last chunk before the stream ends.
    ///
    /// Keeps the stream open past its final chunk, so a consumer that finalizes only on
    /// end-of-stream stays parked and cannot reach finalization within a test's deadline.
    #[must_use]
    pub fn with_trailing_hold(mut self, hold: Duration) -> Self {
        self.trailing_hold = hold;
        self
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
        let chunks = self.chunks.clone();
        let inter_chunk_delay = self.inter_chunk_delay;
        let trailing_hold = self.trailing_hold;
        Ok(Box::pin(stream! {
            for (index, chunk) in chunks.into_iter().enumerate() {
                if index > 0 && !inter_chunk_delay.is_zero() {
                    tokio::time::sleep(inter_chunk_delay).await;
                }
                yield chunk;
            }
            if !trailing_hold.is_zero() {
                tokio::time::sleep(trailing_hold).await;
            }
        }))
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
