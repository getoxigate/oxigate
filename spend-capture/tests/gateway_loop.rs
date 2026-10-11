// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Gateway in the loop: the gateway, in process with its real OpenAI adapter, sends a buffered
//! and a streaming request through the capture to a scripted upstream, then the same requests
//! through a replay of what was captured — and reports the same response and cost both times.

mod support;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use oxigate::api::{AppState, CHAT_COMPLETIONS_PATH, router};
use oxigate::config::{
    AuthConfig, BudgetConfig, OpenAIConfig, PricingConfig, RedisConfig, SecretString,
    SecurityConfig,
};
use oxigate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
use oxigate::middleware::budget::BudgetRuntimeConfig;
use oxigate::middleware::global_safety::GlobalSafetyRuntimeConfig;
use oxigate::providers::ProviderHealthTracker;
use oxigate::providers::openai::OpenAiAdapter;
use oxigate::utils::CostHeader;
use spend_capture::recording::{BodyKind, RecordingStore, Termination};
use support::{TlsUpstream, client, start_capture, start_replay};

const MODEL: &str = "gpt-4o-mini";
const USAGE: &str = r#"{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11}"#;

/// Starts the gateway on an ephemeral port with one OpenAI provider at `api_base_url`. Postgres
/// and Redis pools are built but never connect: cost is read off the response.
async fn start_gateway(api_base_url: String) -> SocketAddr {
    let pricing = || {
        let db = PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing");
        Arc::new(std::sync::RwLock::new(db))
    };
    let adapter = OpenAiAdapter::new(
        OpenAIConfig {
            api_key: Some(SecretString::new("sk-gateway-test")),
            api_base_url: Some(api_base_url),
            default_model: None,
            timeout_secs: Some(10),
            supported_models: None,
            organization: None,
            project: None,
        },
        pricing(),
    )
    .await
    .expect("adapter");
    let pool = sqlx::postgres::PgPoolOptions::new()
        .acquire_timeout(Duration::from_millis(100))
        .connect_lazy("postgres://postgres:postgres@127.0.0.1:1/postgres")
        .expect("lazy pool");
    let redis_pool = oxigate::redis_pool::create_pool(&RedisConfig {
        url: SecretString::new("redis://127.0.0.1:1"),
        pool_size: Some(1),
        pool_timeout_secs: Some(1),
    })
    .expect("lazy redis pool");
    let state = AppState {
        pool: Arc::new(tokio::sync::RwLock::new(pool)),
        redis_pool: Arc::new(tokio::sync::RwLock::new(redis_pool)),
        pricing_db: pricing(),
        provider: Arc::new(tokio::sync::RwLock::new(Arc::new(adapter))),
        auth: Arc::new(tokio::sync::RwLock::new(AuthConfig::default())),
        global_safety: Arc::new(tokio::sync::RwLock::new(
            GlobalSafetyRuntimeConfig::default(),
        )),
        budget_settings: Arc::new(tokio::sync::RwLock::new(BudgetConfig::default())),
        budget: Arc::new(tokio::sync::RwLock::new(BudgetRuntimeConfig::default())),
        startup_time: 1,
        health: ProviderHealthTracker::new(&[], None, 60, 0.1),
        security: Arc::new(tokio::sync::RwLock::new(SecurityConfig::default())),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        axum::serve(
            listener,
            router(state).into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .expect("gateway serves");
    });
    addr
}

/// What the gateway reported for one request.
#[derive(Debug, PartialEq)]
struct Reported {
    body: Vec<u8>,
    cost: String,
    cost_status: String,
}

async fn chat(gateway: SocketAddr, stream: bool) -> Reported {
    let response = client()
        .post(format!("http://{gateway}{CHAT_COMPLETIONS_PATH}"))
        .json(&serde_json::json!({
            "model": MODEL,
            "messages": [{ "role": "user", "content": "hi" }],
            "stream": stream,
        }))
        .send()
        .await
        .expect("send");
    assert_eq!(response.status(), 200);
    let header = |name: &str| {
        response
            .headers()
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    };
    let headers = (
        header(CostHeader::REQUEST_COST),
        header(CostHeader::COST_STATUS),
    );
    let body = response.bytes().await.expect("body").to_vec();
    let (cost, cost_status) = if stream {
        let text = String::from_utf8(body.clone()).expect("utf-8 stream");
        let data = text
            .split("\n\n")
            .find_map(|event| {
                event
                    .strip_prefix("event: oxigate.usage\ndata: ")
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| panic!("no oxigate.usage event in {text}"));
        let usage: serde_json::Value = serde_json::from_str(&data).expect("usage json");
        let member = |key: &str| usage[key].as_str().expect(key).to_owned();
        (member(CostHeader::REQUEST_COST), member("cost_status"))
    } else {
        (
            headers.0.expect("cost header"),
            headers.1.expect("cost status header"),
        )
    };
    Reported {
        body,
        cost,
        cost_status,
    }
}

fn chunk_event(fields: &str) -> String {
    format!(
        "data: {{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\
         \"model\":\"{MODEL}\",{fields}}}\n\n"
    )
}

#[tokio::test]
async fn the_gateway_reports_the_same_response_and_cost_on_capture_and_replay() {
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    let upstream = TlsUpstream::start().await;
    let capture = start_capture(
        &results,
        &[("openai", &upstream.origin)],
        &upstream.ca_pem,
        &[],
    )
    .await;

    let buffered_body = format!(
        "{{\"id\":\"chatcmpl-0\",\"object\":\"chat.completion\",\"created\":1,\"model\":\"{MODEL}\",\
         \"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"hi\"}},\
         \"finish_reason\":\"stop\"}}],\"usage\":{USAGE}}}"
    );
    let content = chunk_event(
        r#""choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]"#,
    );
    let (content_head, content_tail) = content.split_at(content.len() / 2);
    let stream_chunks = [
        content_head.to_owned(),
        content_tail.to_owned(),
        chunk_event(r#""choices":[{"index":0,"delta":{},"finish_reason":"stop"}]"#),
        chunk_event(&format!("\"choices\":[],\"usage\":{USAGE}")) + "data: [DONE]\n\n",
    ];
    let streamed: Vec<u8> = stream_chunks.concat().into_bytes();
    let upstream_task = tokio::spawn(async move {
        let mut conn = upstream.accept().await;
        conn.read_request().await;
        conn.respond(
            200,
            &[
                ("content-type", "application/json"),
                ("connection", "close"),
            ],
            buffered_body.as_bytes(),
        )
        .await;
        let mut conn = upstream.accept().await;
        conn.read_request().await;
        conn.start_stream(&[]).await;
        for chunk in &stream_chunks {
            conn.chunk(chunk.as_bytes()).await;
        }
        // The provider has said [DONE] and not yet ended its body. Returns only once the capture
        // closes this connection, which it does when the gateway stops reading at [DONE].
        conn.wait_closed().await;
    });

    let gateway = start_gateway(format!("http://{}/openai", capture.local_addr())).await;
    let captured = [chat(gateway, false).await, chat(gateway, true).await];
    upstream_task.await.expect("upstream closed by the capture");
    capture.wait_finished(2).await;
    capture.shutdown().await;

    let recordings = RecordingStore::open(&results)
        .expect("store")
        .load_all()
        .expect("load");
    assert_eq!(recordings.len(), 2);
    assert_eq!(
        recordings[0].meta.termination,
        Termination::UpstreamComplete
    );
    // The gateway stopped at the provider's terminator: every byte up to it is recorded, and
    // the recording says the downstream finished first.
    let stream = &recordings[1];
    assert_eq!(stream.meta.response_kind, BodyKind::EventStream);
    assert_eq!(stream.meta.termination, Termination::DownstreamClosedFirst);
    assert_eq!(stream.response_body, streamed);

    let replay = start_replay(&results, &[]).await;
    let gateway = start_gateway(format!("http://{}/openai", replay.local_addr())).await;
    let replayed = [chat(gateway, false).await, chat(gateway, true).await];
    assert_eq!(replay.mismatch(), None);
    assert_eq!(replay.served(), 2);
    replay.shutdown().await;

    for (captured, replayed) in captured.iter().zip(&replayed) {
        let cost: f64 = captured.cost.parse().expect("cost is a number");
        assert!(cost > 0.0, "the fixture is priced: {}", captured.cost);
        assert_eq!(captured.cost_status, "exact");
        assert_eq!(replayed, captured);
    }
}
