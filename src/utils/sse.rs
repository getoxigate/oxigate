// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Shared SSE (Server-Sent Events) helpers for OpenAI-compatible streaming.
//!
//! Ensures consistent envelope format (id, object, created, model) across adapters.

/// Builds the OpenAI-compatible SSE envelope for chat.completion.chunk events.
///
/// Returns a `serde_json::Map` that can be extended (e.g. with `usage` for Gemini)
/// and then serialized to `data: {...}\n\n`.
#[must_use]
pub fn openai_chat_completion_envelope(
    created: u64,
    model: &str,
    request_id: &str,
    choice: serde_json::Value,
) -> serde_json::Map<String, serde_json::Value> {
    let mut root = serde_json::Map::new();
    root.insert(
        "id".to_string(),
        serde_json::json!(format!("chatcmpl-{}", request_id)),
    );
    root.insert(
        "object".to_string(),
        serde_json::json!("chat.completion.chunk"),
    );
    root.insert("created".to_string(), serde_json::json!(created));
    root.insert("model".to_string(), serde_json::json!(model));
    root.insert("choices".to_string(), serde_json::json!([choice]));
    root
}
