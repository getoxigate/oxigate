// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! OpenAI-specific wire types.
//!
//! Minimal: ChatRequest/ChatResponse from domain/chat are OpenAI-shaped.
//! This module holds types needed for streaming usage extraction.

use serde::Deserialize;

use crate::domain::chat::Usage;

/// Parsed SSE chunk for usage extraction.
///
/// OpenAI sends `usage: null` on all streaming chunks except the final one
/// when `stream_options.include_usage: true`. We parse the final chunk to
/// extract token counts for cost tracking.
#[derive(Debug, Deserialize)]
pub(super) struct StreamChunkWithUsage {
    /// Resolved model name from the provider​.
    #[serde(default)]
    pub model: Option<String>,
    /// Present only in the final chunk when include_usage is true.
    #[serde(default)]
    pub usage: Option<Usage>,
}
