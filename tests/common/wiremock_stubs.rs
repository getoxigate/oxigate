// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Provider HTTP stub builders for wiremock.
//! will use these for full E2E chat completions tests.

use oxigate::api::{CHAT_COMPLETIONS_PATH, EMBEDDINGS_PATH};
use wiremock::matchers::{method, path};
use wiremock::{MockServer, ResponseTemplate};

use super::fixtures;

/// Registers a wiremock stub that responds with a valid OpenAI chat response.
/// Use `stub_openai_chat_once` when you need this to match only the first request.
pub async fn stub_openai_chat(
    server: &MockServer,
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
) {
    let body = fixtures::openai_chat_response(model, prompt_tokens, completion_tokens);
    let mock = wiremock::Mock::given(method("POST"))
        .and(path(CHAT_COMPLETIONS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(body));
    server.register(mock).await;
}

/// Registers a wiremock stub that matches at most once (for tests needing distinct responses).
pub async fn stub_openai_chat_once(
    server: &MockServer,
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
) {
    let body = fixtures::openai_chat_response(model, prompt_tokens, completion_tokens);
    let mock = wiremock::Mock::given(method("POST"))
        .and(path(CHAT_COMPLETIONS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .up_to_n_times(1);
    server.register(mock).await;
}

/// Registers a wiremock stub with OpenAI cached tokens in usage.
pub async fn stub_openai_chat_with_cache(
    server: &MockServer,
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
) {
    let body = fixtures::openai_chat_response_with_cache(
        model,
        prompt_tokens,
        completion_tokens,
        cached_tokens,
    );
    let mock = wiremock::Mock::given(method("POST"))
        .and(path(CHAT_COMPLETIONS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(body));
    server.register(mock).await;
}

/// Registers a wiremock stub with OpenAI cached tokens in usage.
pub async fn stub_openai_chat_with_cache_once(
    server: &MockServer,
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
) {
    let body = fixtures::openai_chat_response_with_cache(
        model,
        prompt_tokens,
        completion_tokens,
        cached_tokens,
    );
    let mock = wiremock::Mock::given(method("POST"))
        .and(path(CHAT_COMPLETIONS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .up_to_n_times(1);
    server.register(mock).await;
}

/// Registers a wiremock stub with Anthropic-style cache fields in usage.
pub async fn stub_openai_chat_with_anthropic_cache(
    server: &MockServer,
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
    cache_creation_input_tokens: u64,
    cache_read_input_tokens: u64,
) {
    let body = fixtures::openai_chat_response_with_anthropic_cache(
        model,
        prompt_tokens,
        completion_tokens,
        cache_creation_input_tokens,
        cache_read_input_tokens,
    );
    let mock = wiremock::Mock::given(method("POST"))
        .and(path(CHAT_COMPLETIONS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(body));
    server.register(mock).await;
}

/// Registers a wiremock stub that responds with a minimal valid OpenAI SSE stream.
///
/// The response body is a static sequence of `data:` chunks followed by `data: [DONE]`,
/// with a `text/event-stream` Content-Type. Suitable for testing raw-bytes forwarding on
/// the streaming path (stream_options_support: false, req.stream == Some(true)).
pub async fn stub_openai_stream(server: &MockServer, model: &str) {
    let body = format!(
        "data: {{\"id\":\"s1\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"delta\":{{\"content\":\"hi\"}},\"index\":0,\"finish_reason\":null}}],\"model\":\"{model}\"}}\n\n\
         data: {{\"id\":\"s2\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"delta\":{{}},\"index\":0,\"finish_reason\":\"stop\"}}],\"model\":\"{model}\",\"usage\":{{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}}}\n\n\
         data: [DONE]\n\n"
    );
    let mock = wiremock::Mock::given(method("POST"))
        .and(path(CHAT_COMPLETIONS_PATH))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Content-Type", "text/event-stream")
                .set_body_bytes(body.into_bytes()),
        );
    server.register(mock).await;
}

/// Registers a wiremock stub that returns an error status.
pub async fn stub_openai_error(server: &MockServer, status: u16) {
    let mock = wiremock::Mock::given(method("POST"))
        .and(path(CHAT_COMPLETIONS_PATH))
        .respond_with(ResponseTemplate::new(status).set_body_string("error"));
    server.register(mock).await;
}

/// Registers a wiremock stub for POST /v1/embeddings with an OpenAI-compatible response.
pub async fn stub_openai_embeddings(server: &MockServer, model: &str, prompt_tokens: u64) {
    let body = fixtures::openai_embeddings_response(model, prompt_tokens);
    let mock = wiremock::Mock::given(method("POST"))
        .and(path(EMBEDDINGS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(body));
    server.register(mock).await;
}

/// Registers a wiremock stub for POST /v1/embeddings returning `n` vectors.
pub async fn stub_openai_embeddings_batch(
    server: &MockServer,
    model: &str,
    n: usize,
    prompt_tokens: u64,
) {
    let body = fixtures::openai_embeddings_batch_response(model, n, prompt_tokens);
    let mock = wiremock::Mock::given(method("POST"))
        .and(path(EMBEDDINGS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(body));
    server.register(mock).await;
}
