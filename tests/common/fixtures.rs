// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! OpenAI-compatible response shape builders for integration tests.
//!
//! Used with wiremock to return valid fixture responses for chat completions, streaming,
//! and embeddings.

use serde_json::{Value, json};

/// Builds a non-streaming OpenAI `/v1/chat/completions` response.
pub fn openai_chat_response(model: &str, prompt_tokens: u64, completion_tokens: u64) -> Value {
    let total_tokens = prompt_tokens + completion_tokens;
    json!({
        "id": "chatcmpl-test-001",
        "object": "chat.completion",
        "created": 1677652288u64,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "test response"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    })
}

/// OpenAI response with prompt_tokens_details.cached_tokens (KV-cache).
pub fn openai_chat_response_with_cache(
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
) -> Value {
    let mut usage = serde_json::map::Map::new();
    usage.insert("prompt_tokens".into(), json!(prompt_tokens));
    usage.insert("completion_tokens".into(), json!(completion_tokens));
    usage.insert(
        "total_tokens".into(),
        json!(prompt_tokens + completion_tokens),
    );
    usage.insert(
        "prompt_tokens_details".into(),
        json!({ "cached_tokens": cached_tokens }),
    );
    json!({
        "id": "chatcmpl-test-001",
        "object": "chat.completion",
        "created": 1677652288u64,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "test" },
            "finish_reason": "stop"
        }],
        "usage": usage
    })
}

/// Response with Anthropic-style cache fields (cache_creation, cache_read).
/// For Anthropic semantics: `prompt_tokens` is input-only (plain tokens, excludes cached);
/// `cache_read_input_tokens` is additive.
pub fn openai_chat_response_with_anthropic_cache(
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
    cache_creation_input_tokens: u64,
    cache_read_input_tokens: u64,
) -> Value {
    let total_tokens = prompt_tokens + completion_tokens;
    json!({
        "id": "chatcmpl-test-001",
        "object": "chat.completion",
        "created": 1677652288u64,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "test" },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens
        }
    })
}

/// Returns an SSE data line string for a content delta chunk.
#[allow(dead_code)] // TODO: remove when used
pub fn openai_stream_chunk(model: &str, content_delta: &str) -> String {
    let chunk = json!({
        "id": "chatcmpl-test-001",
        "object": "chat.completion.chunk",
        "created": 1677652288u64,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "content": content_delta },
            "finish_reason": null
        }]
    });
    format!("data: {}\n\n", chunk)
}

/// Returns the final SSE chunk with usage.
#[allow(dead_code)] // TODO: remove when used
pub fn openai_usage_chunk(model: &str, prompt_tokens: u64, completion_tokens: u64) -> String {
    let total_tokens = prompt_tokens + completion_tokens;
    let chunk = json!({
        "id": "chatcmpl-test-001",
        "object": "chat.completion.chunk",
        "created": 1677652288u64,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    });
    format!("data: {}\n\n", chunk)
}

/// Builds an OpenAI-compatible `/v1/embeddings` response with a single 4-dim vector.
pub fn openai_embeddings_response(model: &str, prompt_tokens: u64) -> serde_json::Value {
    json!({
        "object": "list",
        "data": [{
            "object": "embedding",
            "index": 0,
            "embedding": [0.1, 0.2, 0.3, 0.4]
        }],
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens
        }
    })
}

/// Builds an OpenAI-compatible `/v1/embeddings` response for a batch of `n` inputs.
pub fn openai_embeddings_batch_response(
    model: &str,
    n: usize,
    prompt_tokens: u64,
) -> serde_json::Value {
    let data: Vec<serde_json::Value> = (0..n)
        .map(|i| {
            json!({
                "object": "embedding",
                "index": i,
                "embedding": [0.1, 0.2, 0.3, 0.4]
            })
        })
        .collect();
    json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens
        }
    })
}
