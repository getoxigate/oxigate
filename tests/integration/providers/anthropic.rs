// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Anthropic adapter integration tests.
//!
//! Wiremock-based tests; no real Anthropic API calls.

use futures::StreamExt;
use oxigate::config::{AnthropicConfig, SecretString};
use oxigate::domain::chat::{ChatRequest, Message, MessageContent, Role};
use oxigate::domain::ports::{ProviderAdapter, ProviderError};
use oxigate::providers::AnthropicAdapter;
use wiremock::matchers::{header, header_exists, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn anthropic_chat_response(
    prompt_tokens: u32,
    completion_tokens: u32,
    text: &str,
) -> serde_json::Value {
    serde_json::json!({
        "id": "msg_01test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens
        }
    })
}

fn anthropic_config(mock_base: &str) -> AnthropicConfig {
    AnthropicConfig {
        api_key: Some(SecretString::new("sk-ant-test-key")),
        api_base_url: Some(mock_base.to_string()),
        anthropic_version: Some("2023-06-01".into()),
        default_model: Some("claude-sonnet-4-6".into()),
        default_max_tokens: Some(4096),
        timeout_secs: Some(10),
        supported_models: None,
        tool_call_buffer_cap_bytes: None,
    }
}

#[tokio::test]
async fn test_anthropic_chat_completion_non_streaming() {
    let mock = MockServer::start().await;
    let body = anthropic_chat_response(5, 10, "Hello from Claude");
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "sk-ant-test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = anthropic_config(mock.uri().trim_end_matches('/'));
    let adapter = AnthropicAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "claude-sonnet-4-6".into(),
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
    assert_eq!(content_text, "Hello from Claude");
    assert_eq!(resp.usage.prompt_tokens, 5);
    assert_eq!(resp.usage.completion_tokens, 10);
}

// Satisfies AC: "Final [DONE] is forwarded and stream closes cleanly" — Anthropic path.
#[tokio::test]
async fn test_anthropic_chat_completion_streaming() {
    let mock = MockServer::start().await;
    let sse_body = r#"data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","usage":{"input_tokens":2,"output_tokens":0}}}

data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

data: {"type":"content_block_stop","index":0}

data: {"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}}

data: {"type":"message_stop"}

"#;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header_exists("x-api-key"))
        .and(header_exists("anthropic-version"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Content-Type", "text/event-stream")
                .set_body_raw(sse_body.as_bytes(), "text/event-stream"),
        )
        .mount(&mock)
        .await;

    let config = anthropic_config(mock.uri().trim_end_matches('/'));
    let adapter = AnthropicAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "claude-sonnet-4-6".into(),
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
async fn test_anthropic_streaming_usage_in_final_chunk() {
    let mock = MockServer::start().await;
    let sse_body = r#"data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","usage":{"input_tokens":3,"output_tokens":0}}}

data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}

data: {"type":"content_block_stop","index":0}

data: {"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}}

data: {"type":"message_stop"}

"#;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Content-Type", "text/event-stream")
                .set_body_raw(sse_body.as_bytes(), "text/event-stream"),
        )
        .mount(&mock)
        .await;

    let config = anthropic_config(mock.uri().trim_end_matches('/'));
    let adapter = AnthropicAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "claude-sonnet-4-6".into(),
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

    let mut stream = adapter.chat_completion_stream(&req).await.unwrap();
    let mut last_usage = None;
    while let Some(res) = stream.next().await {
        let chunk = res.unwrap();
        if chunk.usage.is_some() {
            last_usage = chunk.usage.clone();
        }
    }
    let usage = last_usage.expect("final chunk must have usage");
    assert_eq!(usage.prompt_tokens, 3);
    assert_eq!(usage.completion_tokens, 2);
}

#[tokio::test]
async fn test_anthropic_429_preserves_retry_after() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("Retry-After", "42")
                .set_body_json(serde_json::json!({
                    "error": {"message": "rate limited", "type": "rate_limit_error"}
                })),
        )
        .mount(&mock)
        .await;

    let config = anthropic_config(mock.uri().trim_end_matches('/'));
    let adapter = AnthropicAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "claude-sonnet-4-6".into(),
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
            assert_eq!(*retry_after, Some(42));
        }
        _ => panic!("expected RateLimited, got {:?}", err),
    }
}

#[tokio::test]
async fn test_anthropic_529_is_retriable() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(529).set_body_json(serde_json::json!({
            "error": {"message": "overloaded", "type": "overloaded"}
        })))
        .mount(&mock)
        .await;

    let config = anthropic_config(mock.uri().trim_end_matches('/'));
    let adapter = AnthropicAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "claude-sonnet-4-6".into(),
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
        ProviderError::ProviderUnavailable(msg) => {
            assert!(msg.contains("overloaded") || msg.contains("anthropic"));
        }
        _ => panic!("expected ProviderUnavailable, got {:?}", err),
    }
}

#[tokio::test]
async fn test_anthropic_auth_headers_sent() {
    let mock = MockServer::start().await;
    let body = anthropic_chat_response(1, 2, "Hi");
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "sk-ant-test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = anthropic_config(mock.uri().trim_end_matches('/'));
    let adapter = AnthropicAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "claude-sonnet-4-6".into(),
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

    let _ = adapter.chat_completion(&req).await.unwrap();
    mock.verify().await;
}

#[tokio::test]
async fn test_anthropic_cache_tokens_surfaced() {
    let mock = MockServer::start().await;
    let body = serde_json::json!({
        "id": "msg_01test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_creation_input_tokens": 50,
            "cache_read_input_tokens": 30
        }
    });
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header_exists("x-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = anthropic_config(mock.uri().trim_end_matches('/'));
    let adapter = AnthropicAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "claude-sonnet-4-6".into(),
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

    let resp = adapter.chat_completion(&req).await.unwrap();
    assert_eq!(resp.usage.cache_creation_input_tokens, Some(50));
    assert_eq!(resp.usage.cache_read_input_tokens, Some(30));
}

#[tokio::test]
async fn test_anthropic_streaming_cache_tokens() {
    let mock = MockServer::start().await;
    let sse_body = r#"data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":10,"output_tokens":0,"cache_read_input_tokens":50}}}
data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}
data: {"type":"content_block_stop","index":0}
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":2,"cache_read_input_tokens":50}}}
data: {"type":"message_stop"}
"#;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header_exists("x-api-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Content-Type", "text/event-stream")
                .set_body_raw(sse_body.as_bytes(), "text/event-stream"),
        )
        .mount(&mock)
        .await;

    let config = anthropic_config(mock.uri().trim_end_matches('/'));
    let adapter = AnthropicAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "claude-sonnet-4-6".into(),
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

    let mut stream = adapter.chat_completion_stream(&req).await.unwrap();
    let mut last_usage = None;
    while let Some(res) = stream.next().await {
        let chunk = res.unwrap();
        if chunk.usage.is_some() {
            last_usage = chunk.usage.clone();
        }
    }
    let usage = last_usage.expect("final chunk must have usage");
    assert_eq!(usage.cache_read_input_tokens, Some(50));
}

#[tokio::test]
async fn test_anthropic_tool_use_non_streaming() {
    let mock = MockServer::start().await;
    let body = serde_json::json!({
        "id": "msg_01test",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll check the weather."},
            {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "NYC"}}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 5, "output_tokens": 15}
    });
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header_exists("x-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = anthropic_config(mock.uri().trim_end_matches('/'));
    let adapter = AnthropicAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "claude-sonnet-4-6".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("Weather in NYC?".into())),
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

    let resp = adapter.chat_completion(&req).await.unwrap();
    assert_eq!(resp.choices.len(), 1);
    let tcs = resp.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("tool_calls");
    assert_eq!(tcs.len(), 1);
    assert_eq!(tcs[0].id, "toolu_01");
    assert_eq!(tcs[0].function.name, "get_weather");
    assert_eq!(tcs[0].function.arguments, r#"{"city":"NYC"}"#);
    assert_eq!(resp.choices[0].finish_reason, Some("tool_calls".into()));
}

#[tokio::test]
async fn test_anthropic_thinking_tokens_non_streaming() {
    let mock = MockServer::start().await;
    let body = serde_json::json!({
        "id": "msg_01test",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Let me reason about this..."},
            {"type": "text", "text": "The answer is 42."}
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 5,
            "output_tokens": 25,
            "output_tokens_details": {"thinking_tokens": 15}
        }
    });
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header_exists("x-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = anthropic_config(mock.uri().trim_end_matches('/'));
    let adapter = AnthropicAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "claude-sonnet-4-6".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("What is 6*7?".into())),
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

    let resp = adapter.chat_completion(&req).await.unwrap();
    assert_eq!(
        resp.usage
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens),
        Some(15)
    );
    let content = resp.choices[0].message.content.as_ref().expect("content");
    let text = match content {
        MessageContent::Text(s) => s.as_str(),
        _ => "",
    };
    assert_eq!(text, "The answer is 42.");
    assert!(!text.contains("Let me reason"));
}

#[tokio::test]
async fn test_anthropic_streaming_tool_use() {
    // Carried-M5: streaming tool-use round-trip using real Anthropic SSE event sequence.
    let mock = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_t1\",\"type\":\"message\",\"role\":\"assistant\",\"usage\":{\"input_tokens\":8,\"output_tokens\":0}}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_01\",\"name\":\"get_weather\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\": \\\"NYC\\\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"}\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"usage\":{\"input_tokens\":8,\"output_tokens\":12}}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header_exists("x-api-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Content-Type", "text/event-stream")
                .set_body_raw(sse_body.as_bytes(), "text/event-stream"),
        )
        .mount(&mock)
        .await;

    let config = anthropic_config(mock.uri().trim_end_matches('/'));
    let adapter = AnthropicAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "claude-sonnet-4-6".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("What is the weather in NYC?".into())),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        max_completion_tokens: None,
        stream: Some(true),
        tools: None,
        parallel_tool_calls: None,
        request_id: Some("test-tool-stream".into()),
        extra: Default::default(),
    };

    let mut stream = adapter
        .chat_completion_stream(&req)
        .await
        .expect("stream must start");
    let mut chunks = Vec::new();
    while let Some(res) = stream.next().await {
        chunks.push(res.expect("chunk must be ok"));
    }

    assert!(!chunks.is_empty(), "stream must emit at least one chunk");
    let concat: Vec<u8> = chunks.iter().flat_map(|c| c.data.iter().copied()).collect();
    let concat_str = String::from_utf8_lossy(&concat);

    // 1. Final SSE event must include [DONE].
    assert!(
        concat_str.contains("[DONE]"),
        "stream must contain [DONE], got: {concat_str}"
    );

    // Parse every SSE data: frame (excluding [DONE]) as JSON.
    let frames: Vec<serde_json::Value> = concat_str
        .split("\n\n")
        .filter_map(|block| {
            let line = block.trim();
            if let Some(payload) = line.strip_prefix("data: ") {
                if payload == "[DONE]" {
                    return None;
                }
                serde_json::from_str(payload).ok()
            } else {
                None
            }
        })
        .collect();

    assert!(
        !frames.is_empty(),
        "must have at least one parseable SSE frame"
    );

    // 2. At least one frame must carry tool_calls with name="get_weather" and id="toolu_01".
    let tool_name_found = frames
        .iter()
        .any(|f| f["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather");
    assert!(
        tool_name_found,
        "no frame with tool_calls[0].function.name=get_weather; frames: {frames:?}"
    );

    let tool_id_found = frames
        .iter()
        .any(|f| f["choices"][0]["delta"]["tool_calls"][0]["id"] == "toolu_01");
    assert!(
        tool_id_found,
        "no frame with tool_calls[0].id=toolu_01; frames: {frames:?}"
    );

    // 3. Concatenate all tool_call argument fragments for index 0 and verify the full JSON.
    let mut accumulated_args = String::new();
    for frame in &frames {
        if let Some(tcs) = frame["choices"][0]["delta"]["tool_calls"].as_array() {
            for tc in tcs {
                if tc["index"] == 0 {
                    if let Some(args) = tc["function"]["arguments"].as_str() {
                        accumulated_args.push_str(args);
                    }
                }
            }
        }
    }
    let args_json: serde_json::Value =
        serde_json::from_str(&accumulated_args).unwrap_or_else(|e| {
            panic!("accumulated arguments must be valid JSON: {e}; got: {accumulated_args:?}")
        });
    assert_eq!(
        args_json["city"], "NYC",
        "accumulated tool arguments city mismatch"
    );

    // 4. Exactly one frame must carry finish_reason="tool_calls".
    let finish_reason_found = frames
        .iter()
        .any(|f| f["choices"][0]["finish_reason"] == "tool_calls");
    assert!(
        finish_reason_found,
        "no frame with finish_reason=tool_calls; frames: {frames:?}"
    );
}

// ── M4: buffer cap enforcement (integration) ──────────────────────────────────

#[tokio::test]
async fn test_anthropic_non_streaming_tool_buffer_overflow() {
    let mock = MockServer::start().await;
    let body = serde_json::json!({
        "id": "msg_01test",
        "type": "message",
        "role": "assistant",
        "content": [
            // input serializes to ~9 bytes '{"a":1}' > 3-byte cap
            {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"a": 1}}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 5, "output_tokens": 15}
    });
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header_exists("x-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let mut config = anthropic_config(mock.uri().trim_end_matches('/'));
    config.tool_call_buffer_cap_bytes = Some(3);
    let adapter = AnthropicAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "claude-sonnet-4-6".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("Weather?".into())),
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
    match err {
        ProviderError::ToolCallBufferOverflow {
            cap_bytes,
            tool_call_id,
            ..
        } => {
            assert_eq!(cap_bytes, 3);
            assert_eq!(tool_call_id, "toolu_01");
        }
        _ => panic!("expected ToolCallBufferOverflow, got {:?}", err),
    }
}

#[tokio::test]
async fn test_anthropic_streaming_tool_buffer_overflow() {
    let mock = MockServer::start().await;
    // "abcd" is 4 bytes, which exceeds the 3-byte cap.
    let sse_body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_t1\",\"type\":\"message\",\"role\":\"assistant\",\"usage\":{\"input_tokens\":8,\"output_tokens\":0}}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_01\",\"name\":\"get_weather\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"abcd\"}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header_exists("x-api-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Content-Type", "text/event-stream")
                .set_body_raw(sse_body.as_bytes(), "text/event-stream"),
        )
        .mount(&mock)
        .await;

    let mut config = anthropic_config(mock.uri().trim_end_matches('/'));
    config.tool_call_buffer_cap_bytes = Some(3);
    let adapter = AnthropicAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "claude-sonnet-4-6".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("Weather?".into())),
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
    let mut all_data = Vec::new();
    while let Some(res) = stream.next().await {
        // Buffer overflow is emitted as Ok(overflow_sse_event), not Err — headers already sent.
        let chunk = res.expect("chunk must be ok");
        all_data.extend_from_slice(&chunk.data);
    }

    let all_str = String::from_utf8_lossy(&all_data);
    assert!(
        all_str.contains("tool_call_buffer_overflow"),
        "stream must emit overflow SSE event, got: {all_str}"
    );
    assert!(
        all_str.contains("toolu_01"),
        "overflow event must identify the tool_call_id, got: {all_str}"
    );
}

#[tokio::test]
async fn test_anthropic_thinking_tokens_streaming() {
    let mock = MockServer::start().await;
    let sse_body = r#"data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","usage":{"input_tokens":5,"output_tokens":0}}}
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me reason..."}}
data: {"type":"content_block_stop","index":0}
data: {"type":"content_block_start","index":1,"content_block":{"type":"text"}}
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The answer is 42."}}
data: {"type":"content_block_stop","index":1}
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":25,"output_tokens_details":{"thinking_tokens":15}}}}
data: {"type":"message_stop"}
"#;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header_exists("x-api-key"))
        .and(header_exists("anthropic-beta"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Content-Type", "text/event-stream")
                .set_body_raw(sse_body.as_bytes(), "text/event-stream"),
        )
        .mount(&mock)
        .await;

    let config = anthropic_config(mock.uri().trim_end_matches('/'));
    let adapter = AnthropicAdapter::new(config)
        .await
        .expect("adapter must build");

    let mut extra = serde_json::Map::new();
    extra.insert("thinking".into(), serde_json::json!(1000));
    let req = ChatRequest {
        model: "claude-sonnet-4-6".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("What is 6*7?".into())),
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
        extra,
    };

    let mut stream = adapter.chat_completion_stream(&req).await.unwrap();
    let mut chunks = Vec::new();
    let mut last_usage = None;
    while let Some(res) = stream.next().await {
        let chunk = res.unwrap();
        chunks.push(chunk.data.clone());
        if let Some(ref u) = chunk.usage {
            last_usage = Some(u.clone());
        }
    }

    let usage = last_usage.expect("final chunk must have usage");
    assert_eq!(
        usage
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens),
        Some(15)
    );

    let concat = chunks
        .iter()
        .flat_map(|c| c.iter().copied())
        .collect::<Vec<_>>();
    let concat_str = String::from_utf8_lossy(&concat);
    assert!(!concat_str.contains("Let me reason"));
    assert!(concat_str.contains("The answer is 42"));
}
