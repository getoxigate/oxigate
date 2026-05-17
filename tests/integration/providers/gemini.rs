// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Gemini/Vertex adapter integration tests.
//!
//! Wiremock-based tests; no real Google API calls.
//! TODO: add wiremock fixture for Gemini inlineData usage.

use futures::StreamExt;
use oxigate::config::{GeminiConfig, GeminiMode, SecretString};
use oxigate::domain::chat::{ChatRequest, Message, MessageContent, Role};
use oxigate::domain::embedding::{EmbeddingInput, EmbeddingRequest};
use oxigate::domain::ports::{ProviderAdapter, ProviderError};
use oxigate::providers::GeminiAdapter;
use rsa::RsaPrivateKey;
use rsa::pkcs1::{EncodeRsaPrivateKey, LineEnding};
use rsa::rand_core::OsRng;
use wiremock::matchers::{header_exists, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Gemini API (non-streaming) response fixture.
fn gemini_chat_response(
    prompt_tokens: u32,
    completion_tokens: u32,
    text: &str,
    finish_reason: &str,
) -> serde_json::Value {
    gemini_chat_response_with_thinking(prompt_tokens, completion_tokens, None, text, finish_reason)
}

/// Gemini API response with optional thinking tokens.
fn gemini_chat_response_with_thinking(
    prompt_tokens: u32,
    completion_tokens: u32,
    thoughts_token_count: Option<u32>,
    text: &str,
    finish_reason: &str,
) -> serde_json::Value {
    let total = prompt_tokens + completion_tokens + thoughts_token_count.unwrap_or(0);
    let mut usage = serde_json::json!({
        "promptTokenCount": prompt_tokens,
        "candidatesTokenCount": completion_tokens,
        "totalTokenCount": total
    });
    if let Some(t) = thoughts_token_count {
        usage["thoughtsTokenCount"] = serde_json::json!(t);
    }
    serde_json::json!({
        "candidates": [{
            "content": {
                "parts": [{ "text": text }],
                "role": "model"
            },
            "finishReason": finish_reason
        }],
        "usageMetadata": usage
    })
}

/// Gemini streaming chunk (NDJSON) — single chunk with finish.
fn gemini_stream_chunk(text: &str, finish_reason: &str) -> String {
    let chunk = serde_json::json!({
        "candidates": [{
            "content": { "parts": [{ "text": text }], "role": "model" },
            "finishReason": finish_reason
        }],
        "usageMetadata": {
            "promptTokenCount": 2,
            "candidatesTokenCount": 3,
            "totalTokenCount": 5
        }
    });
    format!("{}\n", chunk)
}

/// Gemini embedding response (API format).
fn gemini_embed_response(values: &[f64]) -> serde_json::Value {
    serde_json::json!({
        "embedding": { "values": values }
    })
}

fn test_rsa_key_pem() -> String {
    let mut rng = OsRng;
    let key = RsaPrivateKey::new(&mut rng, 2048).expect("rsa key generation must succeed");
    key.to_pkcs1_pem(LineEnding::LF)
        .expect("pkcs1 pem conversion must succeed")
        .to_string()
}

async fn gemini_config_api(mock_base: &str) -> GeminiConfig {
    GeminiConfig {
        mode: GeminiMode::Api,
        api_key: Some(SecretString::new("test-key")),
        vertex_project: None,
        vertex_location: None,
        vertex_service_account_json: None,
        default_model: Some("gemini-2.0-flash".into()),
        timeout_secs: Some(10),
        api_base_url: Some(mock_base.to_string()),
        vertex_base_url_override: None,
        supported_models: None,
        default_thinking_budget: None,
        embed_api_version: None,
    }
}

#[tokio::test]
async fn test_gemini_api_chat_completion() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1beta/models/gemini-2.0-flash:generateContent";
    let body = gemini_chat_response(5, 10, "Hello from Gemini", "STOP");
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gemini-2.0-flash".into(),
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
    assert_eq!(content_text, "Hello from Gemini");
    assert_eq!(resp.usage.prompt_tokens, 5);
    assert_eq!(resp.usage.completion_tokens, 10);
}

#[tokio::test]
async fn test_gemini_streaming_chunk_order() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1beta/models/gemini-2.0-flash:streamGenerateContent";
    let ndjson = gemini_stream_chunk("Hello", "STOP");
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(ndjson.as_bytes().to_vec(), "application/x-ndjson"),
        )
        .mount(&mock)
        .await;

    let config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gemini-2.0-flash".into(),
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
    assert!(!chunks.is_empty(), "must have at least one chunk");
    let last = chunks.last().expect("last chunk");
    let last_str = String::from_utf8_lossy(&last.data);
    assert!(
        last_str.contains("[DONE]") || last_str.contains("data: [DONE]"),
        "last chunk should signal done, got: {last_str}"
    );
}

#[tokio::test]
async fn test_gemini_function_calling() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1beta/models/gemini-2.0-flash:generateContent";
    let body = serde_json::json!({
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "get_weather",
                        "args": { "location": "London" }
                    }
                }],
                "role": "model"
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15
        }
    });
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gemini-2.0-flash".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("Weather in London?".into())),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        max_completion_tokens: None,
        stream: Some(false),
        tools: Some(vec![oxigate::domain::chat::Tool {
            type_: "function".into(),
            function: oxigate::domain::chat::ToolFunction {
                name: "get_weather".into(),
                description: None,
                parameters: Some(serde_json::json!({"type":"object"})),
            },
        }]),
        parallel_tool_calls: None,
        request_id: Some("test-req-3".into()),
        extra: Default::default(),
    };

    let resp = adapter
        .chat_completion(&req)
        .await
        .expect("chat must succeed");
    let tool_calls = resp
        .choices
        .first()
        .and_then(|c| c.message.tool_calls.as_ref())
        .expect("must have tool_calls");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "get_weather");
}

#[tokio::test]
async fn test_gemini_429_is_retriable() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1beta/models/gemini-2.0-flash:generateContent";
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .respond_with(
            ResponseTemplate::new(429)
                .append_header("Retry-After", "5")
                .set_body_string(r#"{"error":{"message":"Rate limit"}}"#),
        )
        .mount(&mock)
        .await;

    let config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gemini-2.0-flash".into(),
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
        request_id: None,
        extra: Default::default(),
    };

    let err = adapter
        .chat_completion(&req)
        .await
        .expect_err("must fail with 429");
    match &err {
        ProviderError::RateLimited { retry_after } => {
            assert_eq!(*retry_after, Some(5));
        }
        _ => panic!("expected RateLimited, got {err:?}"),
    }
}

#[tokio::test]
async fn test_gemini_404_unknown_model() {
    let mock = MockServer::start().await;
    // Use a model that will produce a 404 path
    let path_pattern = "/v1beta/models/unknown-model-xyz:generateContent";
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .respond_with(
            ResponseTemplate::new(404)
                .set_body_string(r#"{"error":{"message":"Model not found"}}"#),
        )
        .mount(&mock)
        .await;

    let mut config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    config.default_model = Some("unknown-model-xyz".into());
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "unknown-model-xyz".into(),
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
        request_id: None,
        extra: Default::default(),
    };

    let err = adapter
        .chat_completion(&req)
        .await
        .expect_err("must fail with 404");
    match &err {
        ProviderError::UnknownModel(m) => assert!(m.contains("unknown") || m.contains("not found")),
        _ => panic!("expected UnknownModel, got {err:?}"),
    }
}

#[tokio::test]
async fn test_gemini_safety_block_mapped_to_content_filtered() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1beta/models/gemini-2.0-flash:generateContent";
    let body = serde_json::json!({
        "candidates": [{
            "content": null,
            "finishReason": "SAFETY"
        }],
        "usageMetadata": { "promptTokenCount": 1, "candidatesTokenCount": 0, "totalTokenCount": 1 }
    });
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gemini-2.0-flash".into(),
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
        request_id: None,
        extra: Default::default(),
    };

    let err = adapter
        .chat_completion(&req)
        .await
        .expect_err("must fail with ContentFiltered");
    match &err {
        ProviderError::ContentFiltered(_) => {}
        _ => panic!("expected ContentFiltered, got {err:?}"),
    }
}

#[tokio::test]
async fn test_gemini_embeddings() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1/models/text-embedding-004:embedContent";
    let body = gemini_embed_response(&[0.1, -0.2, 0.3]);
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = EmbeddingRequest {
        model: "text-embedding-004".into(),
        input: EmbeddingInput::Single("hello world".into()),
        ..Default::default()
    };

    let resp = adapter
        .embeddings(&req)
        .await
        .expect("embeddings must succeed");
    assert!(!resp.data.is_empty());
    assert!(!resp.data[0].embedding.is_empty());
    // OpenAI compatibility: object and model at top level, object in each data item
    assert_eq!(resp.object, "list");
    assert_eq!(resp.model, "text-embedding-004");
    assert_eq!(resp.data[0].object, "embedding");
}

#[tokio::test]
async fn test_vertex_chat_completion() {
    let mock = MockServer::start().await;
    let mock_uri = mock.uri();
    let mock_base = mock_uri.trim_end_matches('/');
    let token_uri = format!("{}/token", mock_base);
    let vertex_path = "/v1/projects/test-proj/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent";

    // OAuth token exchange
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": "fake-vertex-token",
            "expires_in": 3600
        })))
        .mount(&mock)
        .await;

    // Vertex generateContent — must receive Authorization: Bearer
    let body = gemini_chat_response(3, 7, "Hello from Vertex", "STOP");
    Mock::given(method("POST"))
        .and(path(vertex_path))
        .and(header_exists("Authorization"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let rsa_key = test_rsa_key_pem();
    let sa_json = serde_json::json!({
        "client_email": "test@test-proj.iam.gserviceaccount.com",
        "private_key": rsa_key,
        "token_uri": token_uri
    });

    let config = GeminiConfig {
        mode: GeminiMode::Vertex,
        api_key: None,
        vertex_project: Some("test-proj".into()),
        vertex_location: Some("us-central1".into()),
        vertex_service_account_json: Some(SecretString::new(sa_json.to_string())),
        default_model: Some("gemini-2.0-flash".into()),
        timeout_secs: Some(10),
        api_base_url: None,
        vertex_base_url_override: Some(mock_base.to_string()),
        supported_models: None,
        default_thinking_budget: None,
        embed_api_version: None,
    };

    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gemini-2.0-flash".into(),
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
        request_id: Some("test-vertex-1".into()),
        extra: Default::default(),
    };

    let resp = adapter
        .chat_completion(&req)
        .await
        .expect("chat must succeed");
    assert_eq!(resp.choices.len(), 1);
    let content_text = match &resp.choices[0].message.content {
        Some(oxigate::domain::chat::MessageContent::Text(s)) => s.as_str(),
        _ => "",
    };
    assert_eq!(content_text, "Hello from Vertex");
}

#[tokio::test]
async fn test_gemini_25_thinking_config_in_request() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1beta/models/gemini-2.5-pro:generateContent";
    let body = gemini_chat_response(5, 10, "Hello", "STOP");
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gemini-2.5-pro".into(),
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
        request_id: None,
        extra: Default::default(),
    };

    let _resp = adapter
        .chat_completion(&req)
        .await
        .expect("chat must succeed");

    let requests = mock
        .received_requests()
        .await
        .expect("must record requests");
    assert!(!requests.is_empty());
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let thinking_budget = body
        .get("generationConfig")
        .and_then(|g| g.get("thinkingConfig"))
        .and_then(|t| t.get("thinkingBudget"))
        .and_then(|v| v.as_i64());
    assert_eq!(
        thinking_budget,
        Some(0),
        "thinkingBudget must be 0 by default"
    );
}

#[tokio::test]
async fn test_gemini_25_thinking_config_dynamic_override() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1beta/models/gemini-2.5-pro:generateContent";
    let body = gemini_chat_response(5, 10, "Hello", "STOP");
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let mut config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    config.default_thinking_budget = Some(0);
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let mut extra = serde_json::Map::new();
    extra.insert("thinking_budget".into(), serde_json::json!(-1));
    let req = ChatRequest {
        model: "gemini-2.5-pro".into(),
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
        request_id: None,
        extra,
    };

    let _resp = adapter
        .chat_completion(&req)
        .await
        .expect("chat must succeed");

    let requests = mock
        .received_requests()
        .await
        .expect("must record requests");
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let thinking_budget = body
        .get("generationConfig")
        .and_then(|g| g.get("thinkingConfig"))
        .and_then(|t| t.get("thinkingBudget"))
        .and_then(|v| v.as_i64());
    assert_eq!(thinking_budget, Some(-1), "per-request override must win");
}

#[tokio::test]
async fn test_gemini_3x_thinking_level_in_request() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1beta/models/gemini-3.1-pro-preview:generateContent";
    let body = gemini_chat_response(5, 10, "Hello", "STOP");
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gemini-3.1-pro-preview".into(),
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
        request_id: None,
        extra: Default::default(),
    };

    let _resp = adapter
        .chat_completion(&req)
        .await
        .expect("chat must succeed");

    let requests = mock
        .received_requests()
        .await
        .expect("must record requests");
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let thinking_level = body
        .get("generationConfig")
        .and_then(|g| g.get("thinkingConfig"))
        .and_then(|t| t.get("thinkingLevel"))
        .and_then(|v| v.as_str());
    assert_eq!(
        thinking_level,
        Some("MEDIUM"),
        "default level must be MEDIUM"
    );
    assert!(
        body.get("generationConfig")
            .and_then(|g| g.get("thinkingConfig"))
            .and_then(|t| t.get("thinkingBudget"))
            .is_none(),
        "thinkingBudget must not be present for 3.x"
    );
}

#[tokio::test]
async fn test_gemini_20_no_thinking_config_in_request() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1beta/models/gemini-2.0-flash:generateContent";
    let body = gemini_chat_response(5, 10, "Hello", "STOP");
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gemini-2.0-flash".into(),
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
        request_id: None,
        extra: Default::default(),
    };

    let _resp = adapter
        .chat_completion(&req)
        .await
        .expect("chat must succeed");

    let requests = mock
        .received_requests()
        .await
        .expect("must record requests");
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let thinking_config = body
        .get("generationConfig")
        .and_then(|g| g.get("thinkingConfig"));
    assert!(
        thinking_config.is_none(),
        "gemini-2.0 must NOT receive thinkingConfig (Google returns 400)"
    );
}

#[tokio::test]
async fn test_gemini_25_streaming_returns_content_chunks() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1beta/models/gemini-2.5-flash:streamGenerateContent";
    let chunk1 = serde_json::json!({
        "candidates": [{
            "content": { "parts": [{ "thought": true, "text": "let me think..." }], "role": "model" }
        }]
    });
    let chunk2 = serde_json::json!({
        "candidates": [{
            "content": { "parts": [{ "text": "The answer is 391." }], "role": "model" },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 6,
            "thoughtsTokenCount": 120,
            "totalTokenCount": 136
        }
    });
    let ndjson = format!("{}\n{}\n", chunk1, chunk2);
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(ndjson.as_bytes().to_vec(), "application/x-ndjson"),
        )
        .mount(&mock)
        .await;

    let config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gemini-2.5-flash".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("Solve: 17 * 23".into())),
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
    let mut chunks: Vec<oxigate::domain::chat::StreamChunk> = Vec::new();
    while let Some(res) = stream.next().await {
        chunks.push(res.expect("chunk must be ok"));
    }

    let content_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| {
            let s = String::from_utf8_lossy(&c.data);
            s.contains("delta") && s.contains("content") && !s.contains("[DONE]")
        })
        .collect();
    assert!(
        !content_chunks.is_empty(),
        "must have at least one content delta chunk"
    );
    let last_content = content_chunks
        .last()
        .map(|c| String::from_utf8_lossy(&c.data));
    assert!(
        last_content.as_ref().is_some_and(|s| s.contains("391")),
        "content must include answer, got: {:?}",
        last_content
    );
    let has_reasoning = chunks
        .iter()
        .any(|c| String::from_utf8_lossy(&c.data).contains("reasoning_tokens"));
    assert!(has_reasoning, "final chunk must include reasoning_tokens");
}

#[tokio::test]
async fn test_invalid_thinking_level_returns_400() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1beta/models/gemini-3.1-pro-preview:generateContent";
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(gemini_chat_response(5, 10, "ok", "STOP")),
        )
        .mount(&mock)
        .await;

    let config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let mut extra = serde_json::Map::new();
    extra.insert("thinking_level".into(), serde_json::json!("EXTREME"));
    let req = ChatRequest {
        model: "gemini-3.1-pro-preview".into(),
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
        request_id: None,
        extra,
    };

    let err = adapter
        .chat_completion(&req)
        .await
        .expect_err("must fail before upstream");
    let msg = match &err {
        ProviderError::Translate(s) => s.as_str(),
        ProviderError::InvalidRequest(s) => s.as_str(),
        _ => panic!("expected Translate or InvalidRequest, got {err:?}"),
    };
    assert!(
        msg.contains("invalid") || msg.contains("thinking") || msg.contains("EXTREME"),
        "error must mention invalid thinking_level, got: {msg}"
    );
    let requests = mock
        .received_requests()
        .await
        .expect("must record requests");
    assert!(
        requests.is_empty(),
        "must not reach Google when invalid thinking_level"
    );
}

#[tokio::test]
async fn test_gemini_25_thinking_tokens_in_usage() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1beta/models/gemini-2.5-pro:generateContent";
    let body = gemini_chat_response_with_thinking(100, 50, Some(300), "Answer", "STOP");
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&mock)
        .await;

    let config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gemini-2.5-pro".into(),
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
        request_id: None,
        extra: Default::default(),
    };

    let resp = adapter
        .chat_completion(&req)
        .await
        .expect("chat must succeed");
    assert_eq!(resp.usage.completion_tokens, 50);
    assert_eq!(
        resp.usage
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens),
        Some(300)
    );
}

#[tokio::test]
async fn test_gemini_25_streaming_thinking_tokens_in_final_chunk() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1beta/models/gemini-2.5-flash:streamGenerateContent";
    let chunk = serde_json::json!({
        "candidates": [{
            "content": { "parts": [{ "text": "42" }], "role": "model" },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "thoughtsTokenCount": 150,
            "totalTokenCount": 165
        }
    });
    let ndjson = format!("{}\n", chunk);
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(ndjson.as_bytes().to_vec(), "application/x-ndjson"),
        )
        .mount(&mock)
        .await;

    let config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gemini-2.5-flash".into(),
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
    let mut chunks: Vec<oxigate::domain::chat::StreamChunk> = Vec::new();
    while let Some(res) = stream.next().await {
        chunks.push(res.expect("chunk must be ok"));
    }

    let last_with_usage = chunks
        .iter()
        .find(|c| String::from_utf8_lossy(&c.data).contains("reasoning_tokens"));
    assert!(
        last_with_usage.is_some(),
        "final chunk must include reasoning_tokens"
    );
    let data = String::from_utf8_lossy(&last_with_usage.unwrap().data);
    assert!(data.contains("\"reasoning_tokens\":150"));
}

/// Vertex AI sends usage_metadata in a separate chunk after the one with finish_reason.
/// OxiGate must read that extra chunk and merge usage into the final SSE.
#[tokio::test]
async fn test_gemini_25_streaming_usage_in_separate_chunk() {
    let mock = MockServer::start().await;
    let path_pattern = "/v1beta/models/gemini-2.5-flash:streamGenerateContent";
    let chunk1 = serde_json::json!({
        "candidates": [{
            "content": { "parts": [{ "text": "391" }], "role": "model" },
            "finishReason": "STOP"
        }]
    });
    let chunk2 = serde_json::json!({
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 10,
            "thoughtsTokenCount": 200,
            "totalTokenCount": 230
        }
    });
    let ndjson = format!("{}\n{}\n", chunk1, chunk2);
    Mock::given(method("POST"))
        .and(path(path_pattern))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(ndjson.as_bytes().to_vec(), "application/x-ndjson"),
        )
        .mount(&mock)
        .await;

    let config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");

    let req = ChatRequest {
        model: "gemini-2.5-flash".into(),
        messages: vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("17 * 23 = ?".into())),
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
    let mut chunks: Vec<oxigate::domain::chat::StreamChunk> = Vec::new();
    while let Some(res) = stream.next().await {
        chunks.push(res.expect("chunk must be ok"));
    }

    let last_with_usage = chunks
        .iter()
        .find(|c| String::from_utf8_lossy(&c.data).contains("reasoning_tokens"));
    assert!(
        last_with_usage.is_some(),
        "final chunk must include reasoning_tokens from separate usage chunk"
    );
    let data = String::from_utf8_lossy(&last_with_usage.unwrap().data);
    assert!(data.contains("\"reasoning_tokens\":200"));
    assert!(data.contains("\"391\""));
}

#[tokio::test]
async fn test_supported_models_config_override() {
    let mock = MockServer::start().await;
    let mut config = gemini_config_api(mock.uri().trim_end_matches('/')).await;
    config.supported_models = Some(vec!["my-vertex-fine-tune".into()]);
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");
    let meta = adapter.metadata();
    assert_eq!(meta.supported_models, vec!["my-vertex-fine-tune"]);
}

#[test]
fn test_vertex_config_missing_project_fails_validation() {
    use figment::{
        Figment,
        providers::{Format, Serialized, Yaml},
    };
    use oxigate::config::GatewayConfig;

    // Config with gemini in Vertex mode but missing vertex_project
    let yaml = r#"
server:
  host: "127.0.0.1"
  port: 8080
database:
  url: "postgres://localhost/test"
redis:
  url: "redis://localhost"
log_level: "info"
providers:
  gemini:
    mode: "vertex"
    vertex_location: "us-central1"
    vertex_service_account_json: "{\"client_email\":\"x@y.iam.gserviceaccount.com\",\"private_key\":\"test-private-key-placeholder\",\"token_uri\":\"https://oauth2.googleapis.com/token\"}"
"#;
    let cfg: GatewayConfig = Figment::from(Serialized::defaults(GatewayConfig::default()))
        .merge(Yaml::string(yaml))
        .extract()
        .expect("yaml must parse");
    let err = cfg.validate();
    assert!(
        err.is_err(),
        "validation must fail when vertex_project missing"
    );
    let err_msg = format!("{:?}", err.unwrap_err());
    assert!(
        err_msg.contains("vertex_project"),
        "error should mention vertex_project, got: {err_msg}"
    );
}
