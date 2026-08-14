// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Bedrock Converse adapter integration tests .
//!
//! Uses wiremock to mock the Bedrock endpoint. Tests verify:
//! - Request/response translation (non-streaming and streaming)
//! - SigV4 headers are present on every upstream request
//! - Error mapping (ThrottlingException → RateLimited)
//! - Config validation (missing region fails fast)

use futures::StreamExt;
use oxigate::config::{BedrockConfig, SecretString};
use oxigate::domain::chat::{ChatRequest, Message, MessageContent, Role, Tool, ToolFunction};
use oxigate::domain::ports::{ProviderAdapter, ProviderError};
use oxigate::providers::BedrockAdapter;
use oxigate::providers::bedrock::eventstream::build_frame;
use wiremock::matchers::{header_exists, method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn bedrock_config(mock_uri: &str) -> BedrockConfig {
    BedrockConfig {
        region: "us-east-1".to_string(),
        access_key_id: Some(SecretString::from("AKIDEXAMPLE")),
        secret_access_key: Some(SecretString::from(
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        )),
        session_token: None,
        endpoint_url: Some(mock_uri.trim_end_matches('/').to_string()),
        default_model: Some("anthropic.claude-3-5-sonnet-20241022-v2:0".to_string()),
        timeout_secs: Some(10),
        supported_models: None,
    }
}

fn user_request(model: &str, text: &str) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        max_completion_tokens: None,
        stream: None,
        tools: None,
        parallel_tool_calls: None,
        request_id: Some("test-req-001".to_string()),
        extra: serde_json::Map::new(),
    }
}

fn tool_request(model: &str, tool_name: &str) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("What is the weather?".to_string())),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        max_completion_tokens: None,
        stream: None,
        tools: Some(vec![Tool {
            type_: "function".to_string(),
            function: ToolFunction {
                name: tool_name.to_string(),
                description: Some("Get weather".to_string()),
                parameters: Some(serde_json::json!({"type":"object","properties":{}})),
            },
        }]),
        parallel_tool_calls: None,
        request_id: Some("test-tool-bedrock".to_string()),
        extra: serde_json::Map::new(),
    }
}

fn converse_response_body(text: &str, input_tokens: u64, output_tokens: u64) -> serde_json::Value {
    serde_json::json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}]
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens
        }
    })
}

/// Builds a minimal EventStream byte sequence with the given frames.
fn event_stream_bytes(frames: &[(&str, serde_json::Value)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (event_type, payload) in frames {
        let payload_bytes = serde_json::to_vec(payload).unwrap();
        bytes.extend_from_slice(&build_frame(event_type, &payload_bytes));
    }
    bytes
}

#[tokio::test]
async fn test_bedrock_chat_non_streaming() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse$"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(converse_response_body(
                "Hello from Bedrock",
                10,
                5,
            )),
        )
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()))
        .await
        .expect("adapter must build");

    let req = user_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "Say hi");
    let resp = adapter.chat_completion(&req).await.unwrap();

    assert_eq!(resp.choices.len(), 1);
    assert_eq!(resp.choices[0].message.role, Role::Assistant);
    if let Some(MessageContent::Text(t)) = &resp.choices[0].message.content {
        assert_eq!(t, "Hello from Bedrock");
    } else {
        panic!("expected text content");
    }
    assert_eq!(resp.usage.prompt_tokens, 10);
    assert_eq!(resp.usage.completion_tokens, 5);
    assert_eq!(resp.usage.total_tokens, 15);
}

#[tokio::test]
async fn test_bedrock_chat_streaming() {
    let mock = MockServer::start().await;

    let frames: &[(&str, serde_json::Value)] = &[
        (
            "contentBlockDelta",
            serde_json::json!({"contentBlockIndex":0,"delta":{"text":"Hello "}}),
        ),
        (
            "contentBlockDelta",
            serde_json::json!({"contentBlockIndex":0,"delta":{"text":"world"}}),
        ),
        ("messageStop", serde_json::json!({"stopReason":"end_turn"})),
    ];
    let stream_bytes = event_stream_bytes(frames);

    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse-stream$"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(stream_bytes)
                .append_header("content-type", "application/vnd.amazon.eventstream"),
        )
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()))
        .await
        .expect("adapter must build");

    let mut req = user_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "Count");
    req.stream = Some(true);

    let stream = adapter.chat_completion_stream(&req).await.unwrap();
    let chunks: Vec<_> = stream.collect().await;

    for chunk in &chunks {
        assert!(chunk.is_ok(), "no error chunks expected");
    }

    // Derive expected count from the input frames so this assertion stays in sync
    // automatically if the frame list is ever changed.
    // Formula: 1 preamble + N content deltas + 1 fallback (no metadata frame in this stream).
    let n_deltas = frames
        .iter()
        .filter(|(t, _)| *t == "contentBlockDelta")
        .count();
    assert_eq!(
        chunks.len(),
        1 + n_deltas + 1,
        "expected 1 preamble + {n_deltas} deltas + 1 fallback"
    );

    // First ok-chunk must carry the role preamble with empty content.
    let first_data = String::from_utf8_lossy(
        &chunks
            .iter()
            .filter_map(|c| c.as_ref().ok())
            .next()
            .unwrap()
            .data,
    )
    .to_string();
    assert!(
        first_data.contains("\"role\":\"assistant\""),
        "first chunk must carry role"
    );
    assert!(
        first_data.contains("\"content\":\"\""),
        "first chunk content must be empty string"
    );

    // Content-delta chunks (skip preamble at index 0) must not carry role.
    // Delta chunks have "content" in their delta; the fallback final chunk has "delta":{} so
    // no "content" key — filtering on "content" alone correctly isolates just the deltas.
    let content_chunks: Vec<_> = chunks
        .iter()
        .skip(1)
        .filter_map(|c| c.as_ref().ok())
        .filter(|c| String::from_utf8_lossy(&c.data).contains("\"content\""))
        .collect();
    assert!(
        !content_chunks.is_empty(),
        "expected at least one content-delta chunk after the preamble"
    );
    for c in &content_chunks {
        let s = String::from_utf8_lossy(&c.data).to_string();
        assert!(
            !s.contains("\"role\""),
            "content delta must not carry role: {s}"
        );
    }

    let all_data: String = chunks
        .iter()
        .filter_map(|c| c.as_ref().ok())
        .map(|c| String::from_utf8_lossy(&c.data).to_string())
        .collect();
    assert!(
        all_data.contains("Hello ") || all_data.contains("world"),
        "stream data must contain text deltas"
    );
    assert!(
        all_data.contains("[DONE]"),
        "stream must terminate with [DONE] even without a metadata frame"
    );
    assert!(
        all_data.contains("finish_reason"),
        "final chunk must carry finish_reason"
    );
}

#[tokio::test]
async fn test_bedrock_streaming_usage_in_final_chunk() {
    let mock = MockServer::start().await;

    // Real AWS order: contentBlockDelta(s) → messageStop → metadata.
    // metadata is the last event and carries billing token counts.
    let frames: &[(&str, serde_json::Value)] = &[
        (
            "contentBlockDelta",
            serde_json::json!({"contentBlockIndex":0,"delta":{"text":"hi"}}),
        ),
        ("messageStop", serde_json::json!({"stopReason":"end_turn"})),
        (
            "metadata",
            serde_json::json!({"usage":{"inputTokens":20,"outputTokens":8,"totalTokens":28}}),
        ),
    ];
    let stream_bytes = event_stream_bytes(frames);

    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse-stream$"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(stream_bytes)
                .append_header("content-type", "application/vnd.amazon.eventstream"),
        )
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()))
        .await
        .expect("adapter must build");

    let mut req = user_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "hi");
    req.stream = Some(true);

    let stream = adapter.chat_completion_stream(&req).await.unwrap();
    let chunks: Vec<_> = stream.collect().await;

    // Find the chunk that carries usage
    let usage_chunk = chunks
        .iter()
        .filter_map(|c| c.as_ref().ok())
        .find(|c| c.usage.is_some());

    assert!(
        usage_chunk.is_some(),
        "at least one stream chunk must carry usage data"
    );
    let usage = usage_chunk.unwrap().usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, 20);
    assert_eq!(usage.completion_tokens, 8);
}

#[tokio::test]
async fn test_bedrock_sigv4_headers_on_mock() {
    let mock = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse$"))
        .and(header_exists("authorization"))
        .and(header_exists("x-amz-date"))
        .and(header_exists("x-amz-content-sha256"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(converse_response_body("signed", 5, 3)),
        )
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()))
        .await
        .expect("adapter must build");

    let req = user_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "test signing");
    // If SigV4 headers are missing, wiremock returns 404 (no mock matches).
    let resp = adapter.chat_completion(&req).await;
    assert!(
        resp.is_ok(),
        "request with SigV4 headers must succeed: {resp:?}"
    );
}

#[tokio::test]
async fn test_bedrock_429_throttling() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse$"))
        .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
            "__type": "ThrottlingException",
            "message": "Rate exceeded"
        })))
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()))
        .await
        .expect("adapter must build");

    let req = user_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "hi");
    let err = adapter.chat_completion(&req).await.unwrap_err();
    assert!(
        matches!(err, ProviderError::RateLimited { retry_after: None }),
        "ThrottlingException must map to RateLimited{{retry_after: None}}, got: {err:?}"
    );
}

#[tokio::test]
async fn test_bedrock_streaming_fallback_no_metadata() {
    // Exercises the post-loop fallback: stream ends after messageStop with no metadata frame.
    // Validates that [DONE] is still emitted and no usage chunk is produced.
    let mock = MockServer::start().await;

    let frames: &[(&str, serde_json::Value)] = &[
        (
            "contentBlockDelta",
            serde_json::json!({"contentBlockIndex":0,"delta":{"text":"ok"}}),
        ),
        ("messageStop", serde_json::json!({"stopReason":"end_turn"})),
        // intentionally no metadata frame
    ];
    let stream_bytes = event_stream_bytes(frames);

    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse-stream$"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(stream_bytes)
                .append_header("content-type", "application/vnd.amazon.eventstream"),
        )
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()))
        .await
        .expect("adapter must build");

    let mut req = user_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "hi");
    req.stream = Some(true);

    let stream = adapter.chat_completion_stream(&req).await.unwrap();
    let chunks: Vec<_> = stream.collect().await;

    for chunk in &chunks {
        assert!(chunk.is_ok(), "no error chunks expected in fallback path");
    }

    let all_data: String = chunks
        .iter()
        .filter_map(|c| c.as_ref().ok())
        .map(|c| String::from_utf8_lossy(&c.data).to_string())
        .collect();
    assert!(
        all_data.contains("[DONE]"),
        "fallback path must still emit [DONE]"
    );

    // No metadata frame means no usage chunk.
    let has_usage = chunks
        .iter()
        .filter_map(|c| c.as_ref().ok())
        .any(|c| c.usage.is_some());
    assert!(
        !has_usage,
        "no usage chunk expected when metadata frame is absent"
    );
}

#[tokio::test]
async fn test_bedrock_config_missing_region_fails() {
    let config = BedrockConfig {
        region: "".to_string(), // intentionally empty
        access_key_id: Some(SecretString::from("AKID")),
        secret_access_key: Some(SecretString::from("SECRET")),
        session_token: None,
        endpoint_url: None,
        default_model: None,
        timeout_secs: None,
        supported_models: None,
    };

    let err = match BedrockAdapter::new(config).await {
        Ok(_) => panic!("missing region must fail at startup"),
        Err(e) => e,
    };
    let msg = format!("{err:?}");
    assert!(
        msg.contains("region") || matches!(err, ProviderError::InvalidRequest(_)),
        "missing region must produce an actionable error, got: {err:?}"
    );
}

// ── M3: Bedrock tool use integration tests ────────────────────────────────────────────────────

#[tokio::test]
async fn test_bedrock_tool_use_non_streaming_round_trip() {
    let mock = MockServer::start().await;
    let response_body = serde_json::json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_01",
                            "name": "get_weather",
                            "input": {"city": "London"}
                        }
                    }
                ]
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 20, "outputTokens": 10}
    });

    Mock::given(method("POST"))
        .and(path_regex(r"/model/.*/converse$"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_body))
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()))
        .await
        .expect("adapter must build");

    let req = tool_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "get_weather");
    let resp = adapter.chat_completion(&req).await.expect("must succeed");

    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("tool_calls"));
    let tool_calls = resp.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("tool_calls present");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "tooluse_01");
    assert_eq!(tool_calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "London");
}

#[tokio::test]
async fn test_bedrock_streaming_with_tools_returns_not_yet_supported() {
    let mock = MockServer::start().await;
    // Server should not be called; the error happens before dispatch.
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&mock)
        .await;

    let adapter = BedrockAdapter::new(bedrock_config(&mock.uri()))
        .await
        .expect("adapter must build");

    let mut req = tool_request("anthropic.claude-3-5-sonnet-20241022-v2:0", "get_weather");
    req.stream = Some(true);

    let err = match adapter.chat_completion_stream(&req).await {
        Err(e) => e,
        Ok(_) => panic!("expected NotYetSupported error but got Ok"),
    };
    assert!(
        matches!(err, ProviderError::NotYetSupported { .. }),
        "expected NotYetSupported, got: {err:?}"
    );
}
