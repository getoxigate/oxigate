// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! No credential value reaches the results directory, whichever carrier brought it.

mod support;

use oxigate::config::{GeminiConfig, GeminiMode, SecretString};
use oxigate::domain::chat::{ChatRequest, Message, MessageContent, Role};
use oxigate::domain::ports::ProviderAdapter;
use oxigate::providers::GeminiAdapter;
use spend_capture::recording::{FieldValue, RecordingStore};
use support::{TlsUpstream, client, files_containing, files_under, start_capture};

/// Every credential value the exchanges below carry, each distinct so a leak names its carrier.
const AUTHORIZATION: &str = "sk-authorization-value-01";
const PROXY_AUTHORIZATION: &str = "proxy-authorization-value-02";
const X_API_KEY: &str = "x-api-key-value-03";
const API_KEY: &str = "api-key-value-04";
const X_GOOG_API_KEY: &str = "x-goog-api-key-value-05";
const COOKIE: &str = "cookie-value-06";
const SET_COOKIE: &str = "set-cookie-value-07";
const AMZ_TOKEN: &str = "amz-security-token-value-08";
const SESSION_TOKEN: &str = "session-token-value-09";
const LOCATION_PASSWORD: &str = "location-password-value-10";
const LOCATION_TOKEN: &str = "location-token-value-11";
const GEMINI_KEY: &str = "gemini-query-key-value-12";

#[tokio::test]
async fn no_credential_reaches_the_results_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    let results = dir.path().join("results");
    let upstream = TlsUpstream::start().await;
    let capture = start_capture(
        &results,
        &[("openai", &upstream.origin), ("gemini", &upstream.origin)],
        &upstream.ca_pem,
        &[],
    )
    .await;

    let location =
        format!("https://user:{LOCATION_PASSWORD}@api.example.com/v1/next?token={LOCATION_TOKEN}");
    let gemini_response = serde_json::json!({
        "candidates": [{
            "content": { "parts": [{ "text": "hi" }], "role": "model" },
            "finishReason": "STOP"
        }],
        "usageMetadata": { "promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2 }
    })
    .to_string();
    let upstream_task = tokio::spawn(async move {
        let mut first = upstream.accept().await;
        let headers_request = first.read_request().await;
        first
            .respond(
                302,
                &[
                    ("connection", "close"),
                    (
                        "Set-Cookie",
                        &format!("sess={SET_COOKIE}; Path=/; HttpOnly"),
                    ),
                    ("Location", &location),
                    ("x-ratelimit-remaining-tokens", "149984"),
                    ("anthropic-ratelimit-input-tokens-remaining", "39000"),
                ],
                b"{}",
            )
            .await;
        let mut second = upstream.accept().await;
        let gemini_request = second.read_request().await;
        second
            .respond(
                200,
                &[
                    ("content-type", "application/json"),
                    ("connection", "close"),
                ],
                gemini_response.as_bytes(),
            )
            .await;
        (headers_request, gemini_request)
    });

    // Named carriers in mixed case, and one caught only by the name pattern.
    let response = client()
        .post(format!(
            "http://{}/openai/v1/chat/completions",
            capture.local_addr()
        ))
        .header("Authorization", format!("Bearer {AUTHORIZATION}"))
        .header(
            "PROXY-AUTHORIZATION",
            format!("Basic {PROXY_AUTHORIZATION}"),
        )
        .header("X-Api-Key", X_API_KEY)
        .header("api-key", API_KEY)
        .header("X-GOOG-API-KEY", X_GOOG_API_KEY)
        .header("Cookie", format!("theme=dark; session={COOKIE}"))
        .header("x-amz-security-token", AMZ_TOKEN)
        .header("X-Session-Token", SESSION_TOKEN)
        .body("{}")
        .send()
        .await
        .expect("send");
    assert_eq!(
        response.status(),
        302,
        "redirects are passed on, not followed"
    );

    // The gateway's own Gemini adapter, which puts its key in the URL query.
    let adapter = GeminiAdapter::new(GeminiConfig {
        mode: GeminiMode::Api,
        api_key: Some(SecretString::new(GEMINI_KEY)),
        vertex_project: None,
        vertex_location: None,
        vertex_service_account_json: None,
        default_model: None,
        timeout_secs: Some(10),
        api_base_url: Some(format!("http://{}/gemini", capture.local_addr())),
        vertex_base_url_override: None,
        supported_models: None,
        default_thinking_budget: None,
        embed_api_version: None,
    })
    .await
    .expect("gemini adapter");
    adapter
        .chat_completion(&ChatRequest {
            model: "gemini-3.8-flash".into(),
            messages: vec![Message {
                role: Role::User,
                content: Some(MessageContent::Text("hi".into())),
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
        })
        .await
        .expect("gemini call through the capture");

    let (headers_request, gemini_request) = upstream_task.await.expect("upstream");
    // Forwarding is untouched: the upstream got every end-to-end credential as sent.
    assert_eq!(headers_request.header("x-api-key"), [X_API_KEY]);
    assert_eq!(
        headers_request.header("authorization"),
        [format!("Bearer {AUTHORIZATION}")]
    );
    assert!(
        gemini_request
            .target()
            .contains(&format!("key={GEMINI_KEY}")),
        "{}",
        gemini_request.target()
    );

    capture.wait_finished(2).await;
    capture.shutdown().await;
    assert!(
        files_under(&results)
            .iter()
            .any(|p| p.ends_with("capture.log")),
        "the harness log is part of the scanned tree"
    );
    for secret in [
        AUTHORIZATION,
        PROXY_AUTHORIZATION,
        X_API_KEY,
        API_KEY,
        X_GOOG_API_KEY,
        COOKIE,
        SET_COOKIE,
        AMZ_TOKEN,
        SESSION_TOKEN,
        LOCATION_PASSWORD,
        LOCATION_TOKEN,
        GEMINI_KEY,
    ] {
        assert!(
            files_containing(&results, secret.as_bytes()).is_empty(),
            "`{secret}` reached the results directory"
        );
    }

    let recordings = RecordingStore::open(&results)
        .expect("store")
        .load_all()
        .expect("load");
    assert_eq!(recordings.len(), 2, "both exchanges are recorded, redacted");
    let kept = |name: &str| {
        recordings[0]
            .meta
            .response_headers
            .iter()
            .find(|h| h.name == name)
            .map(|h| h.value.clone())
    };
    assert_eq!(
        kept("x-ratelimit-remaining-tokens"),
        Some(FieldValue::Text("149984".into()))
    );
    assert_eq!(
        kept("anthropic-ratelimit-input-tokens-remaining"),
        Some(FieldValue::Text("39000".into()))
    );
    assert!(
        recordings[1].meta.uri.ends_with("key=[REDACTED]"),
        "{}",
        recordings[1].meta.uri
    );
}
