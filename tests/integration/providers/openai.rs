// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! OpenAI adapter integration tests.
//!
//! Wiremock-based tests; no real OpenAI API calls.

use futures::StreamExt;
use oxigate::config::{OpenAIConfig, SecretString};
use oxigate::domain::chat::{ChatRequest, Message, MessageContent, Role};
use oxigate::domain::ports::{ProviderAdapter, ProviderError};
use oxigate::providers::OpenAiAdapter;
use wiremock::matchers::{body_partial_json, header, header_exists, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn openai_chat_response(
    prompt_tokens: u32,
    completion_tokens: u32,
    text: &str,
) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
}

fn openai_chat_response_with_reasoning(
    prompt_tokens: u32,
    completion_tokens: u32,
    reasoning_tokens: u32,
    text: &str,
) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "o3",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens + reasoning_tokens,
            "completion_tokens_details": {
                "reasoning_tokens": reasoning_tokens
            }
        }
    })
}

fn openai_config(mock_base: &str) -> OpenAIConfig {
    OpenAIConfig {
        api_key: Some(SecretString::new("sk-test-key")),
        default_model: Some("gpt-4o".into()),
        api_base_url: Some(mock_base.to_string()),
        timeout_secs: Some(10),
        supported_models: None,
        organization: None,
        project: None,
    }
}

#[tokio::test]
async fn test_openai_chat_completion_non_streaming() {
    let mock = MockServer::start().await;
    let body = openai_chat_response(5, 10, "Hello from OpenAI");
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("Authorization", "Bearer sk-test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = openai_config(mock.uri().trim_end_matches('/'));
    let adapter = OpenAiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gpt-4o".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("Hi".into())),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        max_completion_tokens: None,
        stream: Some(false),
        tools: None,
        parallel_tool_calls: None,
        request_id: Some("test-req-1".into()),
        extra: Default::default(),
    };

    let resp = adapter
        .chat_completion(&req)
        .await
        .expect("chat must succeed");
    assert_eq!(resp.choices.len(), 1);
    let msg = &resp.choices[0].message;
    let content_text = match &msg.content {
        Some(MessageContent::Text(s)) => s.as_str(),
        _ => "",
    };
    assert_eq!(content_text, "Hello from OpenAI");
    assert_eq!(resp.usage.prompt_tokens, 5);
    assert_eq!(resp.usage.completion_tokens, 10);
}

#[tokio::test]
async fn test_openai_chat_completion_streaming() {
    let mock = MockServer::start().await;
    let sse_body =
        "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: [DONE]\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header_exists("Authorization"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(sse_body.as_bytes(), "text/event-stream"),
        )
        .mount(&mock)
        .await;

    let config = openai_config(mock.uri().trim_end_matches('/'));
    let adapter = OpenAiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gpt-4o".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("Hi".into())),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        max_completion_tokens: None,
        stream: Some(true),
        tools: None,
        parallel_tool_calls: None,
        request_id: Some("test-req-2".into()),
        extra: Default::default(),
    };

    let mut stream = adapter
        .chat_completion_stream(&req)
        .await
        .expect("stream must start");
    let mut chunks: Vec<oxigate::domain::chat::StreamChunk> = Vec::new();
    while let Some(res) = stream.next().await {
        chunks.push(res.expect("chunk must be ok"));
    }
    assert!(!chunks.is_empty());
    let concat: Vec<u8> = chunks.iter().flat_map(|c| c.data.iter().copied()).collect();
    let concat_str = String::from_utf8_lossy(&concat);
    assert!(
        concat_str.contains("[DONE]"),
        "stream should contain [DONE], got: {concat_str}"
    );
}

#[tokio::test]
async fn test_openai_streaming_usage_extraction() {
    let mock = MockServer::start().await;
    let sse_with_usage = r#"data: {"choices":[{"delta":{}}]}
data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5,"completion_tokens_details":{"reasoning_tokens":1}}}
data: [DONE]
"#;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(sse_with_usage.as_bytes(), "text/event-stream"),
        )
        .mount(&mock)
        .await;

    let config = openai_config(mock.uri().trim_end_matches('/'));
    let adapter = OpenAiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gpt-4o".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("Hi".into())),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        max_completion_tokens: None,
        stream: Some(true),
        tools: None,
        parallel_tool_calls: None,
        request_id: None,
        extra: Default::default(),
    };

    let mut stream = adapter
        .chat_completion_stream(&req)
        .await
        .expect("stream must start");
    while let Some(res) = stream.next().await {
        res.expect("chunk must be ok");
    }
}

#[tokio::test]
async fn test_openai_429_preserves_retry_after() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("Retry-After", "60")
                .set_body_json(serde_json::json!({
                    "error": { "message": "Rate limit exceeded" }
                })),
        )
        .mount(&mock)
        .await;

    let config = openai_config(mock.uri().trim_end_matches('/'));
    let adapter = OpenAiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gpt-4o".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("Hi".into())),
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
        extra: Default::default(),
    };

    let err = adapter.chat_completion(&req).await.unwrap_err();
    match &err {
        ProviderError::RateLimited { retry_after } => {
            assert_eq!(*retry_after, Some(60));
        }
        _ => panic!("expected RateLimited, got {err:?}"),
    }
}

#[tokio::test]
async fn test_openai_auth_header_sent() {
    let mock = MockServer::start().await;
    let body = openai_chat_response(1, 1, "ok");
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("Authorization", "Bearer sk-test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = openai_config(mock.uri().trim_end_matches('/'));
    let adapter = OpenAiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gpt-4o".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("x".into())),
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
        extra: Default::default(),
    };

    adapter.chat_completion(&req).await.expect("must succeed");
    mock.verify().await;
}

#[tokio::test]
async fn test_openai_organization_header() {
    let mock = MockServer::start().await;
    let mut config = openai_config(mock.uri().trim_end_matches('/'));
    config.organization = Some("org-123".into());
    let body = openai_chat_response(1, 1, "ok");
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("Authorization", "Bearer sk-test-key"))
        .and(header("OpenAI-Organization", "org-123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let adapter = OpenAiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gpt-4o".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("x".into())),
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
        extra: Default::default(),
    };

    adapter.chat_completion(&req).await.expect("must succeed");
    mock.verify().await;
}

#[tokio::test]
async fn test_openai_reasoning_model_e2e() {
    let mock = MockServer::start().await;
    let body = openai_chat_response_with_reasoning(5, 10, 100, "Reasoned answer");
    // Verify upstream request has system→developer remapping and max_tokens→max_completion_tokens
    let expected_upstream = serde_json::json!({
        "messages": [{ "role": "developer", "content": "You are helpful." }],
        "max_completion_tokens": 500
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(&expected_upstream))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = openai_config(mock.uri().trim_end_matches('/'));
    let adapter = OpenAiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "o3".into(),
        messages: vec![
            Message {
                role: Role::System,
                content: Some(MessageContent::Text("You are helpful.".into())),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: Some(MessageContent::Text("Hi".into())),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.5),
        max_tokens: Some(500),
        max_completion_tokens: None,
        stream: None,
        tools: None,
        parallel_tool_calls: None,
        request_id: None,
        extra: Default::default(),
    };

    let resp = adapter
        .chat_completion(&req)
        .await
        .expect("chat must succeed");
    assert_eq!(resp.usage.completion_tokens, 10);
    assert_eq!(
        resp.usage
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens),
        Some(100)
    );
}
