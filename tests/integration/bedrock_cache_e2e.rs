// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Handler-level coverage for Bedrock cache-write accounting reaching the request handler and
//! the budget layer, not just the finalized `Usage`.
//!
//! `test_bedrock_buffered_accounts_cache_writes_from_the_adapter`
//! (`tests/integration/providers/bedrock.rs`) proves the adapter accounts and prices cache
//! writes correctly; it stops at the finalized `Usage`. This file proves that finalized cost
//! actually reaches the two surfaces only the request handler and the budget layer can produce:
//! the terminal `oxigate.usage` SSE event on a streaming response, and the Redis budget counter.
//! Both require the real gateway wired to Postgres and Redis, which the wiremock-only adapter
//! tests do not stand up.

use std::sync::Arc;

use axum::http::StatusCode;

use crate::common::bundled_pricing_holder;
use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use oxigate::api::CHAT_COMPLETIONS_PATH;
use oxigate::config::{BedrockConfig, SecretString};
use oxigate::domain::usage_accounting::CostStatus;
use oxigate::providers::BedrockAdapter;
use oxigate::providers::bedrock::eventstream::build_frame;
use oxigate::utils::CostHeader;
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

// Matches RequestIdentity::default() key path used by auth-disabled test flows, as in
// budget_e2e.rs and embeddings_e2e.rs. Not shared centrally — each E2E file defines its own
// copy, the established pattern in this suite.
const DEFAULT_SPEND_KEY: &str = "oxigate:org:default:spend:default";

fn bedrock_config(mock_uri: &str) -> BedrockConfig {
    BedrockConfig {
        region: "us-east-1".to_string(),
        access_key_id: Some(SecretString::from("AKIDEXAMPLE")),
        secret_access_key: Some(SecretString::from(
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        )),
        session_token: None,
        endpoint_url: Some(mock_uri.trim_end_matches('/').to_string()),
        default_model: Some("anthropic.claude-sonnet-4-6".to_string()),
        timeout_secs: Some(10),
        supported_models: None,
    }
}

fn event_stream_bytes(frames: &[(&str, serde_json::Value)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (event_type, payload) in frames {
        let payload_bytes = serde_json::to_vec(payload).unwrap();
        bytes.extend_from_slice(&build_frame(event_type, &payload_bytes));
    }
    bytes
}

/// A streamed Bedrock cache write reaches the terminal `oxigate.usage` event and the Redis
/// budget counter with the cost the adapter's own accounting computed.
///
/// Fixture is the same one `converse_cache_write`'s exact-cost oracle in `translate.rs` uses,
/// against the bundled `anthropic.claude-sonnet-4-6` entry (input 3e-06 USD/token, output
/// 1.5e-05, cache_read_multiplier 0.1, cache_write_multipliers `{5m: 1.25, 1h: 2.0}`): prompt
/// 10,000 with 2,000 read, 1,000 written at `5m` and 500 at `1h`; completion 500. Expected total
/// is 44,850,000 nano-USD ("0.044850"), `exact` — reproduced here rather than re-derived, so a
/// regression in either surface shows against a number already established independently.
#[tokio::test]
async fn test_bedrock_streaming_cache_write_reaches_usage_event_and_budget_counter() {
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");

    let mock = MockServer::start().await;
    let frames: &[(&str, serde_json::Value)] = &[
        (
            "contentBlockDelta",
            serde_json::json!({"contentBlockIndex": 0, "delta": {"text": "cached"}}),
        ),
        ("messageStop", serde_json::json!({"stopReason": "end_turn"})),
        (
            "metadata",
            serde_json::json!({
                "usage": {
                    "inputTokens": 10_000,
                    "outputTokens": 500,
                    "totalTokens": 10_500,
                    "cacheReadInputTokens": 2_000,
                    "cacheWriteInputTokens": 1_500,
                    "cacheDetails": [
                        {"ttl": "5m", "inputTokens": 1_000},
                        {"ttl": "1h", "inputTokens": 500}
                    ]
                }
            }),
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

    let provider = Arc::new(
        BedrockAdapter::new(bedrock_config(&mock.uri()), bundled_pricing_holder())
            .await
            .expect("adapter must build"),
    );
    let gateway = TestGateway::spawn(pg.pool.clone(), redis.pool.clone(), provider).await;

    let body = serde_json::json!({
        "model": "anthropic.claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "Say hi"}],
        "stream": true
    });

    let response = gateway
        .server
        .post(CHAT_COMPLETIONS_PATH)
        .add_header("Authorization", "Bearer sk-test-key")
        .json(&body)
        .await;

    response.assert_status(StatusCode::OK);
    let body_text = response.text();

    let usage_event_data = body_text
        .split("\n\n")
        .find_map(|record| {
            let mut event_type = None;
            let mut data = None;
            for line in record.lines() {
                if let Some(rest) = line.strip_prefix("event: ") {
                    event_type = Some(rest);
                } else if let Some(rest) = line.strip_prefix("data: ") {
                    data = Some(rest);
                }
            }
            if event_type == Some("oxigate.usage") {
                data
            } else {
                None
            }
        })
        .expect("exactly one oxigate.usage event must be emitted");

    let usage_json: serde_json::Value =
        serde_json::from_str(usage_event_data).expect("oxigate.usage data must be JSON");
    assert_eq!(
        usage_json
            .get(CostHeader::REQUEST_COST)
            .and_then(|v| v.as_str()),
        Some("0.044850"),
        "the terminal event must carry the cost the adapter's accounting computed, not a \
         double-counted or fallback-priced amount"
    );
    assert_eq!(
        usage_json.get("cost_status").and_then(|v| v.as_str()),
        Some(CostStatus::Exact.as_str()),
        "both cache-write classes were credited and priced, so status must be exact"
    );

    // The response completing only proves the persistence task was spawned, not that it ran —
    // spend is written via tokio::spawn after the response body finishes streaming. Poll with a
    // bound rather than assert immediately, as budget_e2e.rs's retrospective-spend test does.
    let mut spend: Option<u64> = None;
    for _ in 0..20 {
        let mut conn = redis.pool.get().await.expect("redis conn");
        spend = redis::cmd("GET")
            .arg(DEFAULT_SPEND_KEY)
            .query_async(&mut *conn)
            .await
            .expect("read spend key");
        if spend.is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(
        spend,
        Some(44_850_000),
        "the Redis budget counter must be incremented by the same total the usage event reported"
    );
}
