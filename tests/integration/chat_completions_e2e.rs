// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! E2E tests for POST /v1/chat/completions.
//!
//! Wiremock-intercept tests; gateway receives request, compat adapter forwards
//! to mock, response + cost headers verified.

use std::sync::Arc;

use axum::http::StatusCode;
use bytes::Bytes;
use oxigate::domain::chat::{StreamChunk, Usage};
use oxigate::domain::ports::ProviderError;

use oxigate::api::CHAT_COMPLETIONS_PATH;
use oxigate::config::OpenAICompatConfig;
use oxigate::providers::{CompatHttpClient, OpenAICompatAdapter};
use oxigate::utils::CostHeader;

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::fixtures;
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::{
    AllRateLimitedStubAdapter, FailingStreamStubAdapter, StreamStubAdapter, StubAdapter,
};
use crate::common::wiremock_stubs;

/// Parses SSE response body into (event_type, data) pairs. Skips empty records.
fn parse_sse_events(body: &str) -> Vec<(String, String)> {
    let mut events = Vec::new();
    for record in body.split("\n\n") {
        let record = record.trim();
        if record.is_empty() {
            continue;
        }
        let mut event_type = String::new();
        let mut data = String::new();
        for line in record.lines() {
            if let Some(rest) = line.strip_prefix("event: ") {
                event_type = rest.to_string();
            } else if let Some(rest) = line.strip_prefix("data: ") {
                data = rest.to_string();
            }
        }
        if !event_type.is_empty() || !data.is_empty() {
            if event_type.is_empty() {
                event_type = "message".to_string();
            }
            events.push((event_type, data));
        }
    }
    events
}

#[tokio::test]
async fn test_chat_completions_e2e_with_wiremock() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_chat(&mock, "gpt-4.1", 10, 20).await;

    let provider = Arc::new(
        OpenAICompatAdapter::new(
            OpenAICompatConfig {
                name: "compat-test".to_string(),
                base_url: mock.uri().trim_end_matches('/').to_string(),
                api_key: None,
                supported_models: None,
                stream_options_support: false,
                supports_tools: false,
                timeout_secs: None,
            },
            Arc::new(CompatHttpClient::new().expect("compat http")),
        )
        .await
        .expect("compat adapter must build"),
    );
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::OK);
    assert!(
        response.content_type().contains("application/json"),
        "expected application/json, got {}",
        response.content_type()
    );
    let headers = response.headers();
    assert!(
        headers.contains_key(CostHeader::REQUEST_COST),
        "missing request cost header"
    );
    assert!(
        headers.contains_key(CostHeader::INPUT_TOKENS),
        "missing input tokens header"
    );
    assert!(
        headers.contains_key(CostHeader::OUTPUT_TOKENS),
        "missing output tokens header"
    );
    assert!(
        headers.contains_key(CostHeader::MODEL_USED),
        "missing model-used header"
    );
    let cost_val = headers
        .get(CostHeader::REQUEST_COST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("0.000000");
    assert_ne!(
        cost_val, "0.000000",
        "model gpt-4.1 in pricing DB must produce non-zero request cost"
    );

    let json: serde_json::Value = response.json();
    assert_eq!(json["id"].as_str(), Some("chatcmpl-test-001"));
    assert_eq!(json["model"].as_str(), Some("gpt-4.1"));
    assert!(json["choices"].is_array());
    assert_eq!(json["usage"]["prompt_tokens"].as_u64(), Some(10));
    assert_eq!(json["usage"]["completion_tokens"].as_u64(), Some(20));
}

#[tokio::test]
async fn test_chat_completions_provider_down_returns_503() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    // Unreachable URL — no server listening
    let provider = Arc::new(
        OpenAICompatAdapter::new(
            OpenAICompatConfig {
                name: "compat-test".to_string(),
                base_url: "http://127.0.0.1:1".to_string(),
                api_key: None,
                supported_models: None,
                stream_options_support: false,
                supports_tools: false,
                timeout_secs: Some(2),
            },
            Arc::new(CompatHttpClient::new().expect("compat http")),
        )
        .await
        .expect("compat adapter must build"),
    );
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::SERVICE_UNAVAILABLE);
    let headers = response.headers();
    assert_eq!(
        headers
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok()),
        Some("0.000000"),
        "​error path must have zero cost header"
    );
    assert_eq!(
        headers
            .get(CostHeader::INPUT_TOKENS)
            .and_then(|v| v.to_str().ok()),
        Some("0"),
        "​error path must have zero input tokens"
    );
    assert_eq!(
        headers
            .get(CostHeader::OUTPUT_TOKENS)
            .and_then(|v| v.to_str().ok()),
        Some("0"),
        "​error path must have zero output tokens"
    );
    assert_eq!(
        headers
            .get(CostHeader::MODEL_USED)
            .and_then(|v| v.to_str().ok()),
        Some("gpt-4"),
        "​error path must echo attempted model in model-used header"
    );
    let json: serde_json::Value = response.json();
    assert!(json["error"].is_object());
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("unreachable")
    );
}

#[tokio::test]
async fn test_chat_completions_provider_5xx_returns_503() {
    // Upstream 5xx is normalized into ProviderUnavailable → HTTP 503.
    // (Previously mapped to 502 BAD_GATEWAY via ProviderHttpError; changed when the adapter
    // adopted proper HTTP error normalization so that the cooldown path fires correctly.)
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_error(&mock, 500).await;

    let provider = Arc::new(
        OpenAICompatAdapter::new(
            OpenAICompatConfig {
                name: "compat-test".to_string(),
                base_url: mock.uri().trim_end_matches('/').to_string(),
                api_key: None,
                supported_models: None,
                stream_options_support: false,
                supports_tools: false,
                timeout_secs: None,
            },
            Arc::new(CompatHttpClient::new().expect("compat http")),
        )
        .await
        .expect("compat adapter must build"),
    );
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::SERVICE_UNAVAILABLE);
    let headers = response.headers();
    assert_eq!(
        headers
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok()),
        Some("0.000000"),
        "​error response must include request cost 0.000000"
    );
    assert_eq!(
        headers
            .get(CostHeader::INPUT_TOKENS)
            .and_then(|v| v.to_str().ok()),
        Some("0"),
        "​error response must have zero input tokens"
    );
    assert_eq!(
        headers
            .get(CostHeader::OUTPUT_TOKENS)
            .and_then(|v| v.to_str().ok()),
        Some("0"),
        "​error response must have zero output tokens"
    );
    assert_eq!(
        headers
            .get(CostHeader::MODEL_USED)
            .and_then(|v| v.to_str().ok()),
        Some("gpt-4"),
        "​error response must echo gpt-4 in model-used header"
    );
    let json: serde_json::Value = response.json();
    assert!(json["error"].is_object());
}

/// When provider returns resolved model (e.g. gpt-4-0613), `CostHeader::MODEL_USED` must be the
/// resolved name, not the request alias (gpt-4).
#[tokio::test]
async fn test_chat_completions_model_alias_resolution() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_chat(&mock, "gpt-4-0613", 10, 20).await;

    let provider = Arc::new(
        OpenAICompatAdapter::new(
            OpenAICompatConfig {
                name: "compat-test".to_string(),
                base_url: mock.uri().trim_end_matches('/').to_string(),
                api_key: None,
                supported_models: None,
                stream_options_support: false,
                supports_tools: false,
                timeout_secs: None,
            },
            Arc::new(CompatHttpClient::new().expect("compat http")),
        )
        .await
        .expect("compat adapter must build"),
    );
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::OK);
    let headers = response.headers();
    assert_eq!(
        headers
            .get(CostHeader::MODEL_USED)
            .and_then(|v| v.to_str().ok()),
        Some("gpt-4-0613"),
        "​model-used header must be resolved model from provider, not request alias"
    );
    let json: serde_json::Value = response.json();
    assert_eq!(json["model"].as_str(), Some("gpt-4-0613"));
}

/// Gherkin scenario 4: First chunk has model "gpt-4-0613", later chunk has "gpt-4-1";
/// `CostHeader::MODEL_USED` in oxigate.usage JSON must be "gpt-4-0613" (first-wins). WARN on model change is
/// implicit — the handler emits it when prev != m; tracing_test cannot capture it across the
/// TestServer HTTP boundary.
#[tokio::test]
async fn test_chat_completions_streaming_model_divergence() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let chunk1_data = fixtures::openai_stream_chunk("gpt-4-0613", "Hello");
    let chunk2_data = fixtures::openai_stream_chunk("gpt-4-1", " world");
    let usage = Usage {
        prompt_tokens: 5,
        completion_tokens: 2,
        total_tokens: 7,
        ..Default::default()
    };
    let chunk3_data = fixtures::openai_usage_chunk("gpt-4-1", 5, 2);

    let chunks = vec![
        Ok(StreamChunk::new(
            Bytes::from(chunk1_data.clone()),
            None,
            Some("gpt-4-0613".to_string()),
        )),
        Ok(StreamChunk::new(
            Bytes::from(chunk2_data.clone()),
            None,
            Some("gpt-4-1".to_string()),
        )),
        Ok(StreamChunk::new(
            Bytes::from(chunk3_data),
            Some(usage),
            Some("gpt-4-1".to_string()),
        )),
    ];

    let provider = Arc::new(StreamStubAdapter::new(chunks));
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::OK);
    assert!(
        response.content_type().contains("text/event-stream"),
        "expected text/event-stream, got {}",
        response.content_type()
    );

    let body_text = response.text();
    let events = parse_sse_events(&body_text);

    let usage_events: Vec<_> = events
        .iter()
        .filter(|(ev, _)| ev == "oxigate.usage")
        .collect();
    assert_eq!(usage_events.len(), 1, "expected one oxigate.usage event");
    let data: serde_json::Value =
        serde_json::from_str(&usage_events[0].1).expect("oxigate.usage data must be JSON");
    assert_eq!(
        data.get(CostHeader::MODEL_USED).and_then(|v| v.as_str()),
        Some("gpt-4-0613"),
        "​oxigate.usage model-used must be first chunk (gpt-4-0613), not later (gpt-4-1)"
    );
    // WARN on model change: handler emits it when prev != m; tracing_test cannot capture
    // logs across the TestServer HTTP boundary, so we assert the observable first-wins result.
}

/// Gherkin scenario 5: Stub sends 2 valid chunks then error; gateway must emit
/// oxigate.error and must NOT emit oxigate.usage.
#[tokio::test]
async fn test_chat_completions_mid_stream_failure_emits_error_event() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let chunk1_data = fixtures::openai_stream_chunk("gpt-4", "Hello");
    let chunk2_data = fixtures::openai_stream_chunk("gpt-4", "!");

    let chunks: Vec<Result<StreamChunk, ProviderError>> = vec![
        Ok(StreamChunk::new(
            Bytes::from(chunk1_data),
            None,
            Some("gpt-4".to_string()),
        )),
        Ok(StreamChunk::new(
            Bytes::from(chunk2_data),
            None,
            Some("gpt-4".to_string()),
        )),
        Err(ProviderError::ProviderUnavailable(
            "mid-stream failure".to_string(),
        )),
    ];

    let provider = Arc::new(StreamStubAdapter::new(chunks));
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::OK);
    assert!(
        response.content_type().contains("text/event-stream"),
        "expected text/event-stream, got {}",
        response.content_type()
    );

    let body_text = response.text();
    let events = parse_sse_events(&body_text);

    let error_events: Vec<_> = events
        .iter()
        .filter(|(ev, _)| ev == "oxigate.error")
        .collect();
    assert_eq!(error_events.len(), 1, "expected one oxigate.error event");
    let data: serde_json::Value =
        serde_json::from_str(&error_events[0].1).expect("oxigate.error data must be JSON");
    assert_eq!(
        data.get("error").and_then(|v| v.as_str()),
        Some("stream_interrupted"),
        "oxigate.error must contain error: stream_interrupted"
    );
    assert!(
        data.get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .contains("mid-stream failure"),
        "oxigate.error message must contain the provider error"
    );

    let usage_events: Vec<_> = events
        .iter()
        .filter(|(ev, _)| ev == "oxigate.usage")
        .collect();
    assert!(
        usage_events.is_empty(),
        "​must NOT emit oxigate.usage on mid-stream failure"
    );
}

/// Pre-dispatch streaming error — chat_completion_stream returns Err before stream starts.
/// Gateway must inject zero-cost headers (same as non-streaming error path).
#[tokio::test]
async fn test_chat_completions_pre_dispatch_streaming_error_injects_zero_cost_headers() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let provider = Arc::new(FailingStreamStubAdapter::new(
        ProviderError::ProviderUnavailable("pre-dispatch failure".to_string()),
    ));
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::SERVICE_UNAVAILABLE);
    let headers = response.headers();
    assert_eq!(
        headers
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.to_str().ok()),
        Some("0.000000"),
        "​pre-dispatch streaming error must have request cost 0.000000"
    );
    assert_eq!(
        headers
            .get(CostHeader::INPUT_TOKENS)
            .and_then(|v| v.to_str().ok()),
        Some("0"),
        "​pre-dispatch streaming error must have zero input tokens"
    );
    assert_eq!(
        headers
            .get(CostHeader::OUTPUT_TOKENS)
            .and_then(|v| v.to_str().ok()),
        Some("0"),
        "​pre-dispatch streaming error must have zero output tokens"
    );
    assert_eq!(
        headers
            .get(CostHeader::MODEL_USED)
            .and_then(|v| v.to_str().ok()),
        Some("gpt-4"),
        "​pre-dispatch streaming error must echo attempted model in model-used header"
    );
}

#[tokio::test]
async fn test_chat_completions_missing_auth_returns_401() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let auth = oxigate::config::AuthConfig {
        key: Some(oxigate::config::SecretString::from("required-token")),
    };
    let gateway = TestGateway::spawn_with_auth(
        pg.pool.clone(),
        redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        auth,
    )
    .await;

    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response = gateway.server.post(CHAT_COMPLETIONS_PATH).json(&body).await;

    response.assert_status(StatusCode::UNAUTHORIZED);
    let headers = response.headers();
    assert!(
        headers.contains_key("WWW-Authenticate"),
        "401 must include WWW-Authenticate per RFC 6750"
    );
}

/// When all providers are in 429 cooldown, the gateway must return HTTP 503 with a
/// `Retry-After` header whose value matches the `retry_after` from the strategy error.
///
/// This test exercises the full HTTP response mapping for `ChatError::AllProvidersRateLimited`
/// (src/api/chat.rs `IntoResponse`), including the header injection.
#[tokio::test]
async fn test_all_providers_rate_limited_returns_503_with_retry_after() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let provider = Arc::new(AllRateLimitedStubAdapter::new(42));
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::SERVICE_UNAVAILABLE);

    let retry_after = response
        .headers()
        .get("Retry-After")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .expect("Retry-After header must be present and numeric");
    assert_eq!(
        retry_after, 42,
        "Retry-After must match the retry_after from AllProvidersRateLimited"
    );

    let json: serde_json::Value = serde_json::from_slice(response.as_bytes()).unwrap();
    assert_eq!(
        json["error"]["type"].as_str(),
        Some("rate_limit_exceeded"),
        "error type must be rate_limit_exceeded"
    );
}

// ──: zero-copy raw-bytes forwarding ────────────────────────────────────────────────

/// the handler forwards the exact inbound bytes to the compat upstream.
///
/// Sends a JSON body with non-default whitespace/key ordering that serde would normalize
/// on re-serialization. Verifies wiremock received the original bytes verbatim.
#[tokio::test]
async fn raw_bytes_forwarded_verbatim_to_compat_upstream() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_chat(&mock, "deepseek-chat", 5, 3).await;

    let provider = Arc::new(
        OpenAICompatAdapter::new(
            OpenAICompatConfig {
                name: "raw-fwd-test".to_string(),
                base_url: mock.uri().trim_end_matches('/').to_string(),
                api_key: None,
                supported_models: None,
                stream_options_support: false,
                supports_tools: false,
                timeout_secs: None,
            },
            Arc::new(CompatHttpClient::new().expect("compat http")),
        )
        .await
        .expect("adapter must build"),
    );
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    // Use extra whitespace and a non-alphabetical key order that serde would normalize.
    // If the gateway re-serializes, the upstream body won't match this literal.
    let raw_body = b"{ \"model\" :  \"deepseek-chat\" ,  \"messages\" : [ { \"role\" : \"user\" , \"content\" : \"hi\" } ] }";

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .add_header("Content-Type", "application/json")
        .bytes(Bytes::from_static(raw_body))
        .await;

    response.assert_status(StatusCode::OK);

    // Verify upstream (wiremock) received the original bytes byte-for-byte.
    let received = mock.received_requests().await.unwrap_or_default();
    assert_eq!(received.len(), 1, "exactly one upstream request expected");
    assert_eq!(
        received[0].body,
        raw_body.as_slice(),
        "upstream must receive the original raw bytes, not a re-serialized form"
    );
}

/// for the streaming path (stream_options_support: false + stream: true), the
/// wiremock-captured upstream body is byte-for-byte identical to the original inbound body.
#[tokio::test]
async fn raw_bytes_forwarded_verbatim_to_compat_upstream_streaming() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    // stub_openai_stream matches on path/method only — no body matcher — so a
    // non-canonical raw body still reaches the stub and is captured by received_requests().
    wiremock_stubs::stub_openai_stream(&mock, "deepseek-chat").await;

    let provider = Arc::new(
        OpenAICompatAdapter::new(
            OpenAICompatConfig {
                name: "raw-stream-fwd-test".to_string(),
                base_url: mock.uri().trim_end_matches('/').to_string(),
                api_key: None,
                supported_models: None,
                stream_options_support: false, // enables raw-bytes path for streaming
                supports_tools: false,
                timeout_secs: None,
            },
            Arc::new(CompatHttpClient::new().expect("compat http")),
        )
        .await
        .expect("adapter must build"),
    );
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    // Extra whitespace + non-alphabetical key ordering that serde would normalise on
    // re-serialisation. If the gateway re-serialises, the upstream body won't match.
    let raw_body =
        b"{ \"model\" :  \"deepseek-chat\" ,  \"stream\" : true ,  \"messages\" : [ { \"role\" : \"user\" , \"content\" : \"hi\" } ] }";

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .add_header("Content-Type", "application/json")
        .bytes(Bytes::from_static(raw_body))
        .await;

    response.assert_status(StatusCode::OK);

    let received = mock.received_requests().await.unwrap_or_default();
    assert_eq!(received.len(), 1, "exactly one upstream request expected");
    assert_eq!(
        received[0].body,
        raw_body.as_slice(),
        "streaming upstream must receive the original raw bytes, not a re-serialized form"
    );
}

/// the body forwarded upstream parses cleanly as a standard OpenAI request
/// with no Oxigate-internal fields (e.g. `request_id` which is `#[serde(skip)]`).
#[tokio::test]
async fn forwarded_body_has_no_internal_oxigate_fields() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_chat(&mock, "deepseek-chat", 5, 3).await;

    let provider = Arc::new(
        OpenAICompatAdapter::new(
            OpenAICompatConfig {
                name: "no-internal-fields-test".to_string(),
                base_url: mock.uri().trim_end_matches('/').to_string(),
                api_key: None,
                supported_models: None,
                stream_options_support: false,
                supports_tools: false,
                timeout_secs: None,
            },
            Arc::new(CompatHttpClient::new().expect("compat http")),
        )
        .await
        .expect("adapter must build"),
    );
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "hello"}]
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::OK);

    let received = mock.received_requests().await.unwrap_or_default();
    assert_eq!(received.len(), 1);

    // Body must parse as standard OpenAI request — no Oxigate-internal fields.
    let parsed: serde_json::Value =
        serde_json::from_slice(&received[0].body).expect("upstream body must be valid JSON");
    assert!(
        parsed.get("request_id").is_none(),
        "request_id must not appear upstream (serde skip)"
    );
    assert_eq!(parsed["model"].as_str(), Some("deepseek-chat"));
}

// ── M2: Content-Type enforcement ─────────────────────────────────────────────────────────

/// M2: a request with no Content-Type header must be rejected with 400.
#[tokio::test]
async fn missing_content_type_returns_400() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let provider = Arc::new(StubAdapter::default());
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        // deliberately omit Content-Type
        .bytes(Bytes::from_static(b"{}"))
        .await;

    response.assert_status(StatusCode::BAD_REQUEST);
    let json: serde_json::Value = serde_json::from_slice(response.as_bytes()).unwrap();
    assert_eq!(
        json["error"]["type"].as_str(),
        Some("invalid_request_error"),
        "error type must be invalid_request_error"
    );
}

// ──: compat provider cost-tracking + streaming + cooldown ─────────────────────

/// Helper: asserts a compat provider routes a request and returns non-zero cost.
async fn assert_compat_cost_tracked(model: &str, provider_name: &str) {
    let pg = PgContainer::start().await.expect("pg must start");
    let redis = RedisContainer::start().await.expect("redis must start");

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_chat(&mock, model, 10, 20).await;

    let provider = Arc::new(
        OpenAICompatAdapter::new(
            OpenAICompatConfig {
                name: provider_name.to_string(),
                base_url: mock.uri().trim_end_matches('/').to_string(),
                api_key: None,
                supported_models: None,
                stream_options_support: false,
                supports_tools: true,
                timeout_secs: None,
            },
            Arc::new(CompatHttpClient::new().expect("compat http")),
        )
        .await
        .expect("compat adapter must build"),
    );
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::OK);
    let cost_val = response
        .headers()
        .get(CostHeader::REQUEST_COST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("0.000000");
    assert_ne!(
        cost_val, "0.000000",
        "model {model} (provider {provider_name}) must produce non-zero cost"
    );
}

/// AC1: Mistral request is forwarded and cost is tracked.
#[tokio::test]
async fn compat_mistral_routes_and_tracks_cost() {
    // real URL: https://api.mistral.ai — wiremock substitutes in the test
    assert_compat_cost_tracked("mistral-large-latest", "mistral").await;
}

/// AC1/AC2: Groq request is forwarded and cost is tracked at Groq per-token rate.
#[tokio::test]
async fn compat_groq_routes_and_tracks_cost() {
    // real URL: https://api.groq.com/openai — wiremock substitutes in the test
    assert_compat_cost_tracked("llama-3.3-70b-versatile", "groq").await;
}

/// AC1/AC2: Together AI request is forwarded and cost is tracked.
#[tokio::test]
async fn compat_together_ai_routes_and_tracks_cost() {
    // real URL: https://api.together.xyz — wiremock substitutes in the test
    assert_compat_cost_tracked("meta-llama/Llama-3.3-70B-Instruct-Turbo", "together-ai").await;
}

/// AC1: DeepSeek-V3 chat request is forwarded and cost is tracked.
#[tokio::test]
async fn compat_deepseek_chat_routes_and_tracks_cost() {
    // real URL: https://api.deepseek.com — wiremock substitutes in the test
    assert_compat_cost_tracked("deepseek-chat", "deepseek").await;
}

/// AC2: deepseek-reasoner cost reflects cache_read_multiplier 0.1 arithmetic.
#[tokio::test]
async fn compat_deepseek_reasoner_routes_and_tracks_cost() {
    let pg = PgContainer::start().await.expect("pg must start");
    let redis = RedisContainer::start().await.expect("redis must start");

    let mock = wiremock::MockServer::start().await;
    // 10 prompt tokens, 5 of which are cached; 20 completion tokens.
    wiremock_stubs::stub_openai_chat_with_cache(&mock, "deepseek-reasoner", 10, 20, 5).await;

    let provider = Arc::new(
        OpenAICompatAdapter::new(
            OpenAICompatConfig {
                name: "deepseek".to_string(),
                base_url: mock.uri().trim_end_matches('/').to_string(),
                api_key: None,
                supported_models: None,
                stream_options_support: false,
                supports_tools: true,
                timeout_secs: None,
            },
            Arc::new(CompatHttpClient::new().expect("compat http")),
        )
        .await
        .expect("compat adapter must build"),
    );
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "Reason about this"}]
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::OK);
    let cost_str = response
        .headers()
        .get(CostHeader::REQUEST_COST)
        .and_then(|v| v.to_str().ok())
        .expect("X-Oxigate-Request-Cost header must be present");
    let actual_usd: f64 = cost_str.parse().expect("cost header must be valid f64");

    // deepseek-reasoner: input=2.8e-7 USD/token, output=4.2e-7 USD/token, cache_read_multiplier=0.1
    // mock: 10 prompt (5 cached), 20 completion; Inclusive accounting → input_tokens=5
    // expected = 5 * 2.8e-7 * 0.1  (cached input at 10% rate)
    //          + 5 * 2.8e-7        (non-cached input)
    //          + 20 * 4.2e-7       (output)
    //          = 9.94e-6 USD
    // Tolerance: 5e-7 (half a micro-USD) accounts for 6-decimal-place header rounding.
    // Wrong (1.0×) case would produce 11.2e-6 USD — 1.26e-6 outside tolerance. ✗
    let expected_usd = 5.0 * 2.8e-7 * 0.1 + 5.0 * 2.8e-7 + 20.0 * 4.2e-7;
    assert!(
        (actual_usd - expected_usd).abs() < 5e-7,
        "deepseek-reasoner cost must reflect cache_read_multiplier 0.1: \
         expected {expected_usd:.9} USD, got {actual_usd:.9}"
    );
}

/// xAI (Grok) request is forwarded and cost is tracked.
#[tokio::test]
async fn compat_xai_routes_and_tracks_cost() {
    // real URL: https://api.x.ai/v1 — wiremock substitutes in the test
    assert_compat_cost_tracked("grok-3-latest", "xai").await;
}

/// Cerebras request is forwarded and cost is tracked.
#[tokio::test]
async fn compat_cerebras_routes_and_tracks_cost() {
    // real URL: https://api.cerebras.ai/v1 — wiremock substitutes in the test
    assert_compat_cost_tracked("llama-3.3-70b", "cerebras").await;
}

/// AC2 / / /: All compat providers share the same streaming path.
/// One representative test (Groq) verifies Content-Type, chunk order, and [DONE] termination.
#[tokio::test]
async fn compat_streaming_sse_chunks_forwarded() {
    let pg = PgContainer::start().await.expect("pg must start");
    let redis = RedisContainer::start().await.expect("redis must start");

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_stream(&mock, "llama-3.3-70b-versatile").await;

    let provider = Arc::new(
        OpenAICompatAdapter::new(
            OpenAICompatConfig {
                name: "groq".to_string(),
                base_url: mock.uri().trim_end_matches('/').to_string(),
                api_key: None,
                supported_models: None,
                stream_options_support: true,
                supports_tools: true,
                timeout_secs: None,
            },
            Arc::new(CompatHttpClient::new().expect("compat http")),
        )
        .await
        .expect("compat adapter must build"),
    );
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::OK);
    assert!(
        response.content_type().contains("text/event-stream"),
        "streaming response must be text/event-stream, got {}",
        response.content_type()
    );

    let body_text = response.text();

    // All upstream SSE data lines must be present in order.
    let chunk_pos = body_text
        .find("\"id\":\"s1\"")
        .expect("first chunk (s1) must be present in stream body");
    let done_pos = body_text
        .find("[DONE]")
        .expect("[DONE] must terminate the stream");
    assert!(
        chunk_pos < done_pos,
        "SSE chunks must appear before [DONE] sentinel"
    );

    // The stream must carry at least one data: line with content.
    assert!(
        body_text.contains("data: "),
        "stream body must contain data: lines"
    );
}

/// AC3: A 429 from the Groq mock sets the provider into cooldown in ProviderHealthTracker.
#[tokio::test]
async fn groq_429_updates_cooldown_state() {
    use oxigate::config::{PricingConfig, RetryConfig, RoutingConfig};
    use oxigate::domain::ports::ProviderAdapter;
    use oxigate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
    use oxigate::domain::routing::WeightedRandom;
    use oxigate::providers::{ProviderHealthTracker, ProviderRouter};
    use std::collections::HashMap;

    let pg = PgContainer::start().await.expect("pg must start");
    let redis = RedisContainer::start().await.expect("redis must start");

    let mock = wiremock::MockServer::start().await;
    wiremock_stubs::stub_openai_error(&mock, 429).await;

    let adapter: Arc<dyn ProviderAdapter> = Arc::new(
        OpenAICompatAdapter::new(
            OpenAICompatConfig {
                name: "groq".to_string(),
                base_url: mock.uri().trim_end_matches('/').to_string(),
                api_key: None,
                supported_models: Some(vec!["llama-3.3-70b-versatile".to_string()]),
                stream_options_support: true,
                supports_tools: true,
                timeout_secs: None,
            },
            Arc::new(CompatHttpClient::new().expect("compat http")),
        )
        .await
        .expect("compat adapter must build"),
    );

    let health = ProviderHealthTracker::new_for_test(&["groq"]);
    let pricing_db = Arc::new(std::sync::RwLock::new(
        PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("pricing DB must load"),
    ));

    // max_retries: 0 → single attempt; avoids multi-second backoff delays in tests.
    let router = ProviderRouter::new_with_resilience(
        vec![Arc::clone(&adapter)],
        Arc::new(WeightedRandom),
        Arc::clone(&health),
        Arc::clone(&pricing_db),
        RoutingConfig::default(),
        RetryConfig {
            max_retries: 0,
            ..Default::default()
        },
        vec![],
        oxigate::config::SecurityConfig::default(),
    );

    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), Arc::new(router)).await;

    let body = serde_json::json!({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    // 429 from upstream → RateLimited → gateway forwards 429 to client.
    response.assert_status(StatusCode::TOO_MANY_REQUESTS);

    // After a 429, the retry loop calls health.on_rate_limit("groq").
    // Verify via candidates(): is_cooling_down must be true.
    let candidates = health
        .candidates(
            &[adapter],
            &HashMap::new(),
            "llama-3.3-70b-versatile",
            &pricing_db,
        )
        .await;
    assert!(!candidates.is_empty(), "candidates must include groq");
    assert!(
        candidates[0].is_cooling_down,
        "groq must be in cooldown after a 429 response"
    );
}

/// M2: a request with Content-Type: text/plain must be rejected with 400.
#[tokio::test]
async fn wrong_content_type_returns_400() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let provider = Arc::new(StubAdapter::default());
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .add_header("Content-Type", "text/plain")
        .bytes(Bytes::from_static(
            b"{\"model\":\"gpt-4o\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        ))
        .await;

    response.assert_status(StatusCode::BAD_REQUEST);
    let json: serde_json::Value = serde_json::from_slice(response.as_bytes()).unwrap();
    assert_eq!(
        json["error"]["type"].as_str(),
        Some("invalid_request_error"),
        "error type must be invalid_request_error"
    );
}
