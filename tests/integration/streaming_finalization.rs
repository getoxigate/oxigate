// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! A streamed request whose provider reaches a clean terminal chunk is accounted regardless of
//! how far the HTTP client reads the response body.
//!
//! Every case drives a scripted provider stream through a real HTTP gateway and asserts the same
//! three effects, with the evidence ranked deliberately:
//!
//! - **Row (primary).** Each test owns its Postgres container, so `spend_records` is empty until
//!   this request writes to it and a plain count is already request-scoped. The write is spawned,
//!   never synchronous, so it is awaited with a bounded-retry poll rather than a fixed sleep.
//! - **Log (primary, and the only per-request cardinality check).** `chat_completion_cost` is
//!   captured process-wide and counted **only for this test's provider name**, so a test running
//!   concurrently in the same process cannot contribute a line. This is what separates "finalized
//!   once" from "finalized twice"; it is emitted synchronously, so it is readable as soon as the
//!   response ends. The line's `request_id` cannot serve as the filter: the handler generates it
//!   and a client cannot supply one, so a test has no way to know it in advance.
//! - **Metric (corroborating only).** `oxigate_cost_usd_total` is keyed by `(provider, endpoint)`
//!   with no request dimension, so the scrape is filtered by that same provider name. Even
//!   isolated, a counter records a sum and not a call count: a matching delta corroborates the row
//!   and the log, it does not replace them.
//!
//! **Every test must therefore give its adapter a provider name no other test uses.** It is the
//! only dimension carrying test identity through both the log and the metric, and a reused name
//! silently merges two tests' evidence.
//!
//! **Why the scripted stream holds itself open.** With every chunk ready on the first poll the
//! whole body reaches the socket before the client can react, and a test that means to disconnect
//! part-way exercises the read-to-completion path instead. `with_trailing_hold` keeps the provider
//! stream open past its terminal chunk, so end-of-stream is unreachable within the test's deadline
//! and the row can only come from the terminal chunk itself.

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use bytes::Bytes;
use oxigate::api::CHAT_COMPLETIONS_PATH;
use oxigate::db::DbPool;
use oxigate::domain::chat::{StreamChunk, Usage};
use oxigate::domain::ports::ProviderError;
use oxigate::observability::metrics::{COST_USD_TOTAL, ENDPOINT_CHAT};
use tracing_subscriber::Layer as _;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::common::containers::{PgContainer, RedisContainer};
use crate::common::gateway::TestGateway;
use crate::common::stub_adapter::StreamStubAdapter;

/// Model whose bundled pricing gives this fixture's usage a non-zero, exact cost.
const MODEL: &str = "gpt-4o-2024-11-20";
/// The structured cost log line finalization emits.
const COST_LOG_EVENT: &str = "chat_completion_cost";
/// The provider's own stream terminator, as forwarded to the client.
const DONE_MARKER: &[u8] = b"data: [DONE]";
/// Longer than any deadline below: the scripted stream must not reach end-of-stream on its own.
const HOLD_OPEN: Duration = Duration::from_secs(30);
/// Ceiling on reading the response body.
const READ_DEADLINE: Duration = Duration::from_secs(10);
/// Ceiling on the spawned spend write becoming visible.
const ROW_DEADLINE: Duration = Duration::from_secs(5);
/// Window a negative case waits before concluding nothing was written.
const NO_ROW_WINDOW: Duration = Duration::from_secs(1);

// ---------------------------------------------------------------------------
// Cost-log capture
// ---------------------------------------------------------------------------

/// Every `chat_completion_cost` line the process emits, as its `provider` field.
///
/// Installed once for the whole test binary: the gateway runs on the same runtime as the test but
/// not necessarily the same task, so a thread-local subscriber would miss lines depending on where
/// the handler happens to be polled.
fn captured_cost_logs() -> &'static Mutex<Vec<String>> {
    static STORE: OnceLock<&'static Mutex<Vec<String>>> = OnceLock::new();
    STORE.get_or_init(|| {
        let store: &'static Mutex<Vec<String>> = Box::leak(Box::new(Mutex::new(Vec::new())));
        // `try_init` rather than `init`: another module in this binary may legitimately want a
        // global subscriber one day, and losing that race must not take this test down. When it
        // is lost, the capture is empty and the assertions below fail with their own message
        // instead of a panic whose text depends on which test ran first.
        if tracing_subscriber::registry()
            .with(CostLogLayer { store }.with_filter(LevelFilter::INFO))
            .try_init()
            .is_err()
        {
            eprintln!(
                "streaming_finalization: a global tracing subscriber was already installed; \
                 cost-log assertions in this module cannot observe anything"
            );
        }
        store
    })
}

/// How many `chat_completion_cost` lines this test's provider produced.
fn cost_log_lines(provider: &str) -> usize {
    captured_cost_logs()
        .lock()
        .expect("cost log store is not poisoned")
        .iter()
        .filter(|captured| *captured == provider)
        .count()
}

struct CostLogLayer {
    store: &'static Mutex<Vec<String>>,
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CostLogLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut fields = CostLogFields::default();
        event.record(&mut fields);
        if fields.message.as_deref() != Some(COST_LOG_EVENT) {
            return;
        }
        if let Some(provider) = fields.provider {
            self.store
                .lock()
                .expect("cost log store is not poisoned")
                .push(provider);
        }
    }
}

#[derive(Default)]
struct CostLogFields {
    message: Option<String>,
    provider: Option<String>,
}

impl CostLogFields {
    fn set(&mut self, name: &str, value: String) {
        match name {
            "message" => self.message = Some(value),
            "provider" => self.provider = Some(value),
            _ => {}
        }
    }
}

impl tracing::field::Visit for CostLogFields {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.set(field.name(), value.to_string());
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.set(field.name(), format!("{value:?}"));
    }
}

// ---------------------------------------------------------------------------
// Metric scrape
// ---------------------------------------------------------------------------

/// The chat cost counter for one provider label, in nano-USD.
///
/// Zero when the label has not been recorded yet, which is what a "before" reading wants.
fn chat_cost_counter(scrape: &str, provider: &str) -> u64 {
    let provider_label = format!("provider=\"{provider}\"");
    let endpoint_label = format!("endpoint=\"{ENDPOINT_CHAT}\"");
    for line in scrape.lines() {
        if line.starts_with('#') || !line.starts_with(COST_USD_TOTAL) {
            continue;
        }
        if line.contains(&provider_label) && line.contains(&endpoint_label) {
            return line
                .split_whitespace()
                .next_back()
                .and_then(|value| value.parse::<f64>().ok())
                .map_or(0, |value| value as u64);
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn usage(prompt_tokens: u64, completion_tokens: u64) -> Usage {
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        ..Default::default()
    }
}

fn content_chunk(usage: Option<Usage>) -> Result<StreamChunk, ProviderError> {
    Ok(StreamChunk::new(
        Bytes::from_static(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"),
        usage,
        Some(MODEL.to_string()),
    ))
}

/// The clean terminal chunk an adapter honouring the completion contract emits.
fn terminal_chunk(usage: Option<Usage>) -> Result<StreamChunk, ProviderError> {
    let mut chunk = StreamChunk::new(
        Bytes::from_static(b"data: [DONE]\n\n"),
        usage,
        Some(MODEL.to_string()),
    );
    chunk.is_final = true;
    Ok(chunk)
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// Where the test client stops reading and drops the response.
#[derive(Clone, Copy)]
enum StopAt {
    /// The first body frame — the provider has not produced its terminal chunk yet.
    FirstFrame,
    /// As soon as the forwarded `data: [DONE]` has been seen. The stock-SDK stopping point.
    Terminator,
    /// As soon as the terminal `oxigate.usage` event has been seen.
    UsageEvent,
    /// Natural end of the response body.
    EndOfStream,
}

impl StopAt {
    fn marker(self) -> Option<&'static [u8]> {
        match self {
            Self::FirstFrame | Self::EndOfStream => None,
            Self::Terminator => Some(DONE_MARKER),
            Self::UsageEvent => Some(b"event: oxigate.usage"),
        }
    }
}

/// A gateway with its own Postgres and Redis, fronting a scripted provider stream.
struct Harness {
    _pg: PgContainer,
    _redis: RedisContainer,
    _gateway: TestGateway,
    pool: DbPool,
    url: String,
}

async fn harness(adapter: StreamStubAdapter) -> Harness {
    // Install the capture before the request runs — the cost log line is emitted once and is
    // gone if no subscriber is listening when it fires.
    captured_cost_logs();
    let pg = PgContainer::start().await.expect("pg container must start");
    let redis = RedisContainer::start()
        .await
        .expect("redis container must start");
    let pool = pg.pool.clone();
    let gateway = TestGateway::spawn_random_http_port(
        pool.clone(),
        redis.pool.clone(),
        std::sync::Arc::new(adapter),
    )
    .await;
    let url = gateway
        .server
        .server_url(CHAT_COMPLETIONS_PATH)
        .expect("TestServer must expose an HTTP URL")
        .to_string();
    Harness {
        _pg: pg,
        _redis: redis,
        _gateway: gateway,
        pool,
        url,
    }
}

/// Streams the request and drops the response at `stop_at`, returning the bytes read.
async fn read_until(harness: &Harness, stop_at: StopAt) -> Vec<u8> {
    let client = reqwest::Client::new();
    let mut response = client
        .post(&harness.url)
        .header("Authorization", "Bearer sk-test-key")
        .json(&serde_json::json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "x"}],
            "stream": true,
        }))
        .send()
        .await
        .expect("gateway accepts the streaming request");
    assert!(
        response.status().is_success(),
        "expected 200 from the streaming endpoint, got {}",
        response.status()
    );

    let mut seen: Vec<u8> = Vec::new();
    let read = tokio::time::timeout(READ_DEADLINE, async {
        while let Some(frame) = response.chunk().await.expect("read response body") {
            seen.extend_from_slice(&frame);
            match stop_at {
                StopAt::FirstFrame => break,
                StopAt::EndOfStream => continue,
                _ => {
                    let marker = stop_at.marker().expect("non-terminal stop has a marker");
                    if seen.windows(marker.len()).any(|w| w == marker) {
                        break;
                    }
                }
            }
        }
        seen
    })
    .await
    .expect("response body must reach the client's stopping point within the deadline");
    drop(response);
    read
}

/// One `spend_records` row as `(cost_nano_usd, cost_status)`.
type SpendRow = (i64, String);

async fn spend_rows(pool: &DbPool) -> Vec<SpendRow> {
    sqlx::query_as::<_, SpendRow>("SELECT cost_nano_usd, cost_status FROM spend_records")
        .fetch_all(pool)
        .await
        .expect("spend_records is queryable")
}

/// Polls until at least one row exists, or the deadline passes. The spend write is spawned, so
/// its durability is eventual by construction — there is no synchronous moment to read instead.
async fn await_spend_rows(pool: &DbPool, deadline: Duration) -> Vec<SpendRow> {
    let start = std::time::Instant::now();
    loop {
        let rows = spend_rows(pool).await;
        if !rows.is_empty() || start.elapsed() >= deadline {
            return rows;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/// A client that stops at the forwarded `data: [DONE]` — where the stock SDKs stop — is still
/// accounted. The provider stream is held open past its terminal chunk, so end-of-stream cannot
/// supply the row: it can only come from the terminal chunk itself.
#[tokio::test]
async fn finalizes_a_priced_stream_when_the_client_stops_at_the_terminator() {
    let provider = "finalization-terminator-priced";
    let handle = crate::common::test_prometheus_handle();

    let harness = harness(
        StreamStubAdapter::new(vec![
            content_chunk(None),
            terminal_chunk(Some(usage(120, 40))),
        ])
        .with_name(provider)
        .with_trailing_hold(HOLD_OPEN),
    )
    .await;

    let before = chat_cost_counter(&handle.render(), provider);
    let body = read_until(&harness, StopAt::Terminator).await;
    assert!(
        body.windows(DONE_MARKER.len()).any(|w| w == DONE_MARKER),
        "the client must have received the provider's terminal chunk"
    );

    let rows = await_spend_rows(&harness.pool, ROW_DEADLINE).await;
    assert_eq!(
        rows.len(),
        1,
        "a client that stops at the terminal chunk must still leave exactly one spend row"
    );
    assert_eq!(rows[0].1, "exact", "reported usage must be priced exactly");
    assert!(
        rows[0].0 > 0,
        "a priced request must record a positive cost"
    );
    assert_eq!(
        cost_log_lines(provider),
        1,
        "finalization must emit exactly one cost log line for this request"
    );

    let delta = chat_cost_counter(&handle.render(), provider) - before;
    assert_eq!(
        delta,
        u64::try_from(rows[0].0).expect("a spend row's cost is non-negative"),
        "the cost counter must move by this request's own cost"
    );
}

/// The no-usage branch of finalization under the same disconnect. Nothing in the priced case
/// exercises it, and it is the branch that keeps a cost-unavailable request from vanishing.
#[tokio::test]
async fn finalizes_a_cost_unavailable_stream_when_the_client_stops_at_the_terminator() {
    let provider = "finalization-terminator-unpriced";
    let handle = crate::common::test_prometheus_handle();

    let harness = harness(
        StreamStubAdapter::new(vec![content_chunk(None), terminal_chunk(None)])
            .with_name(provider)
            .with_trailing_hold(HOLD_OPEN),
    )
    .await;

    let before = chat_cost_counter(&handle.render(), provider);
    read_until(&harness, StopAt::Terminator).await;

    // Row and log are the whole proof here. A zero counter delta cannot distinguish a zero-cost
    // increment from no increment at all, so it is read below only as "no cost was added".
    let rows = await_spend_rows(&harness.pool, ROW_DEADLINE).await;
    assert_eq!(
        rows.len(),
        1,
        "a stream that reported no usage must still leave exactly one spend row"
    );
    assert_eq!(rows[0].1, "cost-unavailable");
    assert_eq!(rows[0].0, 0, "an unpriceable request records zero cost");
    assert_eq!(
        cost_log_lines(provider),
        1,
        "finalization must emit exactly one cost log line for this request"
    );

    assert_eq!(
        chat_cost_counter(&handle.render(), provider),
        before,
        "an unpriceable request must add no cost to the counter"
    );
}

/// A client that reads one event further — through the terminal `oxigate.usage` event — and then
/// drops. Accounting is already complete by the time either event is on the wire.
#[tokio::test]
async fn finalizes_when_the_client_stops_after_the_usage_event() {
    let provider = "finalization-usage-event";
    let handle = crate::common::test_prometheus_handle();

    let harness = harness(
        StreamStubAdapter::new(vec![
            content_chunk(None),
            terminal_chunk(Some(usage(90, 30))),
        ])
        .with_name(provider)
        .with_trailing_hold(HOLD_OPEN),
    )
    .await;

    let before = chat_cost_counter(&handle.render(), provider);
    read_until(&harness, StopAt::UsageEvent).await;

    let rows = await_spend_rows(&harness.pool, ROW_DEADLINE).await;
    assert_eq!(
        rows.len(),
        1,
        "a client that stops after the usage event must still leave exactly one spend row"
    );
    assert!(rows[0].0 > 0);
    assert_eq!(cost_log_lines(provider), 1);

    let delta = chat_cost_counter(&handle.render(), provider) - before;
    assert_eq!(
        delta,
        u64::try_from(rows[0].0).expect("a spend row's cost is non-negative")
    );
}

/// Usage reported on an earlier chunk survives a terminal chunk that carries none — the shape
/// several adapters produce. Guards the accumulator against being overwritten with `None`, which
/// would silently finalize a fully-priced request as cost-unavailable.
#[tokio::test]
async fn prices_a_stream_whose_terminal_chunk_carries_no_usage() {
    let provider = "finalization-usage-before-terminator";

    let harness = harness(
        StreamStubAdapter::new(vec![
            content_chunk(Some(usage(200, 60))),
            terminal_chunk(None),
        ])
        .with_name(provider),
    )
    .await;

    read_until(&harness, StopAt::EndOfStream).await;

    let rows = await_spend_rows(&harness.pool, ROW_DEADLINE).await;
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].1, "exact",
        "usage seen before the terminal chunk must still price the request"
    );
    assert!(rows[0].0 > 0, "the request must not be recorded as free");
}

/// Reading the stream to its natural end must not finalize twice: the terminal chunk and the
/// end-of-stream fallback are both reachable in this run, and only one of them may fire.
#[tokio::test]
async fn finalizes_exactly_once_when_the_stream_is_read_to_completion() {
    let provider = "finalization-read-to-completion";
    let handle = crate::common::test_prometheus_handle();

    let harness = harness(
        StreamStubAdapter::new(vec![
            content_chunk(None),
            terminal_chunk(Some(usage(150, 50))),
        ])
        .with_name(provider),
    )
    .await;

    let before = chat_cost_counter(&handle.render(), provider);
    let body = read_until(&harness, StopAt::EndOfStream).await;
    let text = String::from_utf8_lossy(&body);
    assert_eq!(
        text.matches("event: oxigate.usage").count(),
        1,
        "the terminal usage event must be emitted exactly once"
    );

    // The log line is the cardinality proof: it is emitted synchronously, so a second
    // finalization would already have been captured by the time the response ended.
    assert_eq!(
        cost_log_lines(provider),
        1,
        "finalization must run exactly once for a fully-read stream"
    );
    let rows = await_spend_rows(&harness.pool, ROW_DEADLINE).await;
    assert_eq!(rows.len(), 1, "one request must leave one spend row");

    let delta = chat_cost_counter(&handle.render(), provider) - before;
    assert_eq!(
        delta,
        u64::try_from(rows[0].0).expect("a spend row's cost is non-negative"),
        "the counter must carry one request's cost, not two"
    );
}

/// The guarantee stops where the provider's completion does. A client that leaves before the
/// stream reaches its terminal chunk is still charged nothing — the accepted trade-off that the
/// pull-coupled body preserves, and the one this change must not regress.
///
/// That no *further upstream chunk* is requested after such a disconnect is proved separately, by
/// `streaming::streaming_client_disconnect_releases_upstream_before_slow_chunk`.
#[tokio::test]
async fn does_not_finalize_when_the_client_leaves_before_the_terminal_chunk() {
    let provider = "finalization-pre-terminal-disconnect";
    let handle = crate::common::test_prometheus_handle();

    let harness = harness(
        StreamStubAdapter::new(vec![
            content_chunk(Some(usage(300, 100))),
            terminal_chunk(None),
        ])
        .with_name(provider)
        .with_inter_chunk_delay(HOLD_OPEN),
    )
    .await;

    let before = chat_cost_counter(&handle.render(), provider);
    read_until(&harness, StopAt::FirstFrame).await;

    let rows = await_spend_rows(&harness.pool, NO_ROW_WINDOW).await;
    assert!(
        rows.is_empty(),
        "an interrupted stream must not be accounted, got {rows:?}"
    );
    assert_eq!(
        cost_log_lines(provider),
        0,
        "an interrupted stream must emit no cost log line"
    );
    assert_eq!(
        chat_cost_counter(&handle.render(), provider),
        before,
        "an interrupted stream must add no cost to the counter"
    );
}
