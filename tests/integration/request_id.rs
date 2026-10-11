// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! The gateway request ID: minted per request, returned in `X-Oxigate-Request-Id`, and stored as
//! the `spend_records.id` of the row the request produced.
//!
//! The header name is spelled out rather than imported: it is the public contract under test.
//! Every case that expects a row joins the header value to `spend_records.id`, so the assertion
//! is request-scoped by construction. The row is written by a spawned task, so it is polled for.

use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use bytes::Bytes;
use oxigate::api::{CHAT_COMPLETIONS_PATH, EMBEDDINGS_PATH};
use oxigate::config::{
    AnthropicConfig, AuthConfig, OpenAICompatConfig, OpenAIConfig, SecretString,
};
use oxigate::db::DbPool;
use oxigate::domain::chat::{StreamChunk, Usage};
use oxigate::domain::ports::ProviderAdapterExt;
use oxigate::providers::openai::OpenAiAdapter;
use oxigate::providers::{AnthropicAdapter, CompatHttpClient, OpenAICompatAdapter};
use uuid::Uuid;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::{InspectingStubAdapter, StreamStubAdapter, StubAdapter};
use crate::common::wiremock_stubs;
use crate::common::{bundled_pricing_holder, lazy_pg_pool, lazy_redis_pool};

const REQUEST_ID_HEADER: &str = "X-Oxigate-Request-Id";
/// Ceiling on the spawned spend write becoming visible.
const ROW_DEADLINE: Duration = Duration::from_secs(5);
/// Window a negative case waits before concluding nothing was written.
const NO_ROW_WINDOW: Duration = Duration::from_secs(1);

struct Env {
    pg: PgContainer,
    redis: RedisContainer,
}

impl Env {
    async fn start() -> Self {
        Self {
            pg: PgContainer::start().await.expect("pg container must start"),
            redis: RedisContainer::start()
                .await
                .expect("redis container must start"),
        }
    }

    async fn gateway(&self, provider: Arc<dyn ProviderAdapterExt>) -> TestGateway {
        TestGateway::spawn(self.pg.pool.clone(), self.redis.pool.clone(), provider).await
    }
}

/// A gateway on never-connecting pools, for cases that assert on the response alone.
async fn gateway_without_storage(provider: Arc<dyn ProviderAdapterExt>) -> TestGateway {
    TestGateway::spawn(lazy_pg_pool(), lazy_redis_pool(), provider).await
}

/// The request ID a response carries, parsed as a UUID.
fn request_id(response: &axum_test::TestResponse) -> Uuid {
    let value = response
        .headers()
        .get(REQUEST_ID_HEADER)
        .unwrap_or_else(|| panic!("response must carry {REQUEST_ID_HEADER}"))
        .to_str()
        .expect("request ID header is ASCII");
    Uuid::parse_str(value).expect("request ID header is a UUID")
}

/// Polls for the row stored under `id`, returning its `model` once it exists.
async fn await_row(pool: &DbPool, id: Uuid) -> Option<String> {
    let start = std::time::Instant::now();
    loop {
        let row: Option<String> =
            sqlx::query_scalar("SELECT model FROM spend_records WHERE id = $1")
                .bind(id)
                .fetch_optional(pool)
                .await
                .expect("spend_records is queryable");
        if row.is_some() || start.elapsed() >= ROW_DEADLINE {
            return row;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn row_count(pool: &DbPool) -> i64 {
    sqlx::query_scalar("SELECT COUNT(*) FROM spend_records")
        .fetch_one(pool)
        .await
        .expect("spend_records is countable")
}

async fn compat_adapter(mock: &MockServer) -> Arc<OpenAICompatAdapter> {
    Arc::new(
        OpenAICompatAdapter::new(
            OpenAICompatConfig {
                name: "compat-request-id".to_string(),
                base_url: mock.uri().trim_end_matches('/').to_string(),
                api_key: None,
                supported_models: None,
                stream_options_support: false,
                supports_tools: false,
                timeout_secs: Some(10),
            },
            Arc::new(CompatHttpClient::new().expect("compat http")),
        )
        .await
        .expect("compat adapter must build"),
    )
}

fn chat_body(stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": stream,
    })
}

#[tokio::test]
async fn buffered_chat_row_is_stored_under_the_returned_request_id() {
    let env = Env::start().await;
    let mock = MockServer::start().await;
    wiremock_stubs::stub_openai_chat(&mock, "gpt-4.1", 10, 20).await;
    let gateway = env.gateway(compat_adapter(&mock).await).await;

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .json(&chat_body(false))
        .await;

    response.assert_status(StatusCode::OK);
    let id = request_id(&response);
    assert_eq!(
        await_row(&env.pg.pool, id).await.as_deref(),
        Some("gpt-4.1"),
        "the spend row must be stored under the request ID the client was given"
    );
}

#[tokio::test]
async fn streamed_chat_row_is_stored_under_the_returned_request_id() {
    let env = Env::start().await;
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
        ..Default::default()
    };
    let provider = Arc::new(StreamStubAdapter::new(vec![Ok(StreamChunk::new(
        Bytes::from("data: {\"choices\":[]}\n\n"),
        Some(usage),
        Some("gpt-4.1".to_string()),
    ))]));
    let gateway = env.gateway(provider).await;

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .json(&chat_body(true))
        .await;

    response.assert_status(StatusCode::OK);
    let id = request_id(&response);
    assert_eq!(
        await_row(&env.pg.pool, id).await.as_deref(),
        Some("gpt-4.1"),
        "the streamed request's spend row must be stored under the returned request ID"
    );
}

#[tokio::test]
async fn embeddings_row_is_stored_under_the_returned_request_id() {
    let env = Env::start().await;
    let mock = MockServer::start().await;
    wiremock_stubs::stub_openai_embeddings(&mock, "text-embedding-3-small", 8).await;
    // The compat adapter has no embeddings support; the native OpenAI adapter does.
    let provider = OpenAiAdapter::new(
        OpenAIConfig {
            api_key: Some(SecretString::new("sk-request-id")),
            api_base_url: Some(mock.uri().trim_end_matches('/').to_string()),
            default_model: None,
            timeout_secs: Some(10),
            supported_models: None,
            organization: None,
            project: None,
        },
        bundled_pricing_holder(),
    )
    .await
    .expect("OpenAiAdapter must build");
    let gateway = env.gateway(Arc::new(provider)).await;

    let response = gateway
        .server
        .post(EMBEDDINGS_PATH)
        .json(&serde_json::json!({"model": "text-embedding-3-small", "input": "hello"}))
        .await;

    response.assert_status(StatusCode::OK);
    let id = request_id(&response);
    assert_eq!(
        await_row(&env.pg.pool, id).await.as_deref(),
        Some("text-embedding-3-small"),
        "the embeddings spend row must be stored under the returned request ID"
    );
}

/// On a translated lane the body ID is built from the request ID, so its UUID suffix is the
/// header value — buffered and streamed alike.
#[tokio::test]
async fn translated_lane_body_id_carries_the_request_id() {
    let mock = MockServer::start().await;
    let stream_body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":2,\"output_tokens\":1}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(serde_json::json!({"stream": true})))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(stream_body.as_bytes(), "text/event-stream"),
        )
        .mount(&mock)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 1}
        })))
        .mount(&mock)
        .await;
    let adapter = AnthropicAdapter::new(
        AnthropicConfig {
            api_key: Some(SecretString::new("sk-ant-request-id")),
            api_base_url: Some(mock.uri().trim_end_matches('/').to_string()),
            anthropic_version: None,
            default_model: None,
            default_max_tokens: None,
            timeout_secs: Some(10),
            supported_models: None,
            tool_call_buffer_cap_bytes: None,
        },
        bundled_pricing_holder(),
    )
    .await
    .expect("anthropic adapter must build");
    let gateway = gateway_without_storage(Arc::new(adapter)).await;
    let body = |stream: bool| {
        serde_json::json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": stream,
        })
    };

    let buffered = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .json(&body(false))
        .await;
    buffered.assert_status(StatusCode::OK);
    let expected = format!("chatcmpl-{}", request_id(&buffered));
    let json: serde_json::Value = buffered.json();
    assert_eq!(json["id"], expected.as_str(), "buffered body ID");

    let streamed = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .json(&body(true))
        .await;
    streamed.assert_status(StatusCode::OK);
    let expected = format!("chatcmpl-{}", request_id(&streamed));
    let ids: Vec<String> = streamed
        .text()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
        .filter_map(|chunk| chunk["id"].as_str().map(str::to_owned))
        .collect();
    assert!(!ids.is_empty(), "the stream must carry chunks with an ID");
    assert!(
        ids.iter().all(|id| *id == expected),
        "every streamed chunk ID must be {expected}, got {ids:?}"
    );
}

/// Rejections that never reach a handler still carry the header, and write nothing.
#[tokio::test]
async fn middleware_rejections_carry_the_request_id_and_write_no_row() {
    let env = Env::start().await;

    let authed = TestGateway::spawn_with_auth(
        env.pg.pool.clone(),
        env.redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        AuthConfig {
            key: Some(SecretString::new("sk-right")),
        },
    )
    .await;
    let unauthorized = authed
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-wrong")
        .json(&chat_body(false))
        .await;
    unauthorized.assert_status(StatusCode::UNAUTHORIZED);
    request_id(&unauthorized);

    let limited = TestGateway::spawn_with_body_limit(
        env.pg.pool.clone(),
        env.redis.pool.clone(),
        Arc::new(StubAdapter::new()),
        1024,
    )
    .await;
    let too_large = limited
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .json(&serde_json::json!({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "x".repeat(2048)}]
        }))
        .await;
    too_large.assert_status(StatusCode::PAYLOAD_TOO_LARGE);
    request_id(&too_large);

    tokio::time::sleep(NO_ROW_WINDOW).await;
    assert_eq!(
        row_count(&env.pg.pool).await,
        0,
        "a rejected request writes no row"
    );
}

#[tokio::test]
async fn provider_error_response_carries_the_request_id() {
    let mock = MockServer::start().await;
    wiremock_stubs::stub_openai_error(&mock, 500).await;
    let gateway = gateway_without_storage(compat_adapter(&mock).await).await;

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .json(&chat_body(false))
        .await;

    assert!(
        response.status_code().is_server_error(),
        "expected a provider error, got {}",
        response.status_code()
    );
    request_id(&response);
}

/// A client-sent request ID is ignored: the response carries a gateway-minted one, and the row
/// is stored under it.
#[tokio::test]
async fn client_cannot_set_the_request_id() {
    let env = Env::start().await;
    let mock = MockServer::start().await;
    wiremock_stubs::stub_openai_chat(&mock, "gpt-4.1", 10, 20).await;
    let gateway = env.gateway(compat_adapter(&mock).await).await;
    let spoofed = Uuid::parse_str("01890a5d-ac96-774b-bcce-b302099a8057").expect("valid UUID");

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header(REQUEST_ID_HEADER, spoofed.to_string())
        .json(&chat_body(false))
        .await;

    response.assert_status(StatusCode::OK);
    let id = request_id(&response);
    assert_ne!(
        id, spoofed,
        "the gateway must not echo a client-sent request ID"
    );
    assert!(
        await_row(&env.pg.pool, id).await.is_some(),
        "the row must be stored under the gateway's ID"
    );
    assert_eq!(
        row_count(&env.pg.pool).await,
        1,
        "exactly one row, under the gateway's ID"
    );
}

/// Every chat request the gateway hands to a provider carries the request ID the client was
/// given, buffered and streamed. Checked at the provider seam, so it holds for every lane.
#[tokio::test]
async fn provider_receives_the_request_id_the_client_was_given() {
    let adapter = Arc::new(InspectingStubAdapter::new());
    let gateway = gateway_without_storage(adapter.clone()).await;

    let mut returned = Vec::new();
    for stream in [false, true] {
        let response = gateway
            .server
            .post(CHAT_COMPLETIONS_PATH)
            .json(&chat_body(stream))
            .await;
        response.assert_status(StatusCode::OK);
        returned.push(Some(request_id(&response).to_string()));
    }

    assert_eq!(
        adapter.seen_request_ids(),
        returned,
        "the provider must receive each request's gateway request ID (buffered, then streamed)"
    );
}
