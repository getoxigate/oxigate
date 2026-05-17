// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Embedding domain types (OpenAI-compatible).
//!
//! Request/response shapes for POST /v1/embeddings.
//! No axum/reqwest imports — domain stays I/O-free.

use serde::{Deserialize, Serialize};

/// OpenAI-compatible embedding request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Model to use for embeddings.
    pub model: String,
    /// Input text(s) to embed. Single string or array of strings.
    pub input: EmbeddingInput,
    /// Output dimensionality (forwarded to OpenAI; ignored by Gemini).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// Encoding format (forwarded as-is; default omitted).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
}

/// Input for embedding: single string or array of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl Default for EmbeddingInput {
    fn default() -> Self {
        EmbeddingInput::Single(String::new())
    }
}

impl EmbeddingRequest {
    /// Returns a clone of this request with the model field replaced.
    /// Used by the fallback cascade to rewrite the model for a target provider.
    #[must_use]
    pub fn with_model(&self, model: &str) -> Self {
        Self {
            model: model.to_string(),
            ..self.clone()
        }
    }
}

impl EmbeddingInput {
    /// Returns the input(s) as a slice of strings.
    #[must_use]
    pub fn as_slice(&self) -> &[String] {
        match self {
            Self::Single(s) => std::slice::from_ref(s),
            Self::Batch(b) => b.as_slice(),
        }
    }
}

/// OpenAI-compatible embedding response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Top-level object type; always "list" per OpenAI spec.
    pub object: String,
    /// Embedding objects.
    pub data: Vec<EmbeddingData>,
    /// Model used (echoed from request).
    pub model: String,
    /// Token usage (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<EmbeddingUsage>,
}

/// A single embedding result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    /// Object type; always "embedding" per OpenAI spec.
    pub object: String,
    /// Embedding vector.
    pub embedding: Vec<f32>,
    /// Index in the input batch.
    pub index: u32,
}

/// Token usage for embedding request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbeddingUsage {
    /// Input tokens; populated from provider response.
    #[serde(default)]
    pub prompt_tokens: u64,
    /// Total tokens used.
    pub total_tokens: u64,
}
