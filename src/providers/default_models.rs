// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! The native lanes' default model lists, checked against the bundled pricing snapshot.
//!
//! With no `supported_models` in config, each native lane's built-in list is what the router
//! matches a request against and what `GET /v1/models` enumerates. A model the snapshot prices
//! but no list names is invisible, and on the Gemini lane, which has no wildcard, unroutable.
//! A model a list names but the snapshot does not price is costed at zero, so its requests
//! accumulate no spend and no hard cap stops them. For the Anthropic, OpenAI and Gemini lanes both
//! directions are checked, so a refresh that prices a model and forgets the lane list fails a test.
//! A priced model may stay off a list only where [`NOT_LISTED`] or the Gemini lane's
//! [`NOT_DEFAULT`] names it. Bedrock is checked in the priced direction only (see its test).

use std::sync::{Arc, RwLock};

use serde_json::Value;

use crate::config::{
    AnthropicConfig, BedrockConfig, GeminiConfig, GeminiMode, OpenAIConfig, PricingConfig,
    SecretString,
};
use crate::domain::ports::ProviderAdapter;
use crate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};
use crate::providers::anthropic::AnthropicAdapter;
use crate::providers::bedrock::BedrockAdapter;
use crate::providers::gemini::{GeminiAdapter, ModelFlags, NOT_DEFAULT, model_flags};
use crate::providers::health::model_matches_provider;
use crate::providers::openai::OpenAiAdapter;

/// Bundled entries a lane deliberately does not list, each with the reason.
const NOT_LISTED: &[(&str, &str)] = &[
    (
        "gemini-embedding-2",
        "the Gemini embeddings path cannot drive this model yet, so a listed route would fail \
         or record zero cost",
    ),
    (
        "o1-pro-2025-03-19",
        "OpenAI serves it on the Responses API only, and this lane calls Chat Completions",
    ),
    (
        "o3-pro-2025-06-10",
        "OpenAI serves it on the Responses API only, and this lane calls Chat Completions",
    ),
    (
        "text-embedding-3-small",
        "embedding model; the OpenAI lane routes it by the `text-embedding-` prefix",
    ),
    (
        "text-embedding-3-large",
        "embedding model; the OpenAI lane routes it by the `text-embedding-` prefix",
    ),
    (
        "text-embedding-ada-002",
        "embedding model; the OpenAI lane routes it by the `text-embedding-` prefix",
    ),
];

/// Gemini models with the thinking and preview flags Google documents for them. The flags decide
/// which thinking parameter a request is translated with.
const GEMINI_FLAGS: &[(&str, ModelFlags)] = &[
    ("gemini-3.8-flash", ModelFlags::THINKING_LEVEL),
    ("gemini-3.5-flash-lite", ModelFlags::THINKING_LEVEL),
    ("gemini-3.1-flash-lite", ModelFlags::THINKING_LEVEL),
];

fn bundled_pricing() -> PricingDb {
    PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
        .expect("bundled pricing must load")
}

async fn anthropic_defaults() -> Vec<String> {
    let config = AnthropicConfig {
        api_key: Some(SecretString::new("sk-ant-test")),
        api_base_url: None,
        anthropic_version: None,
        default_model: None,
        default_max_tokens: None,
        timeout_secs: None,
        supported_models: None,
        tool_call_buffer_cap_bytes: None,
    };
    let adapter = AnthropicAdapter::new(config, Arc::new(RwLock::new(bundled_pricing())))
        .await
        .expect("adapter must build");
    adapter.metadata().supported_models.clone()
}

async fn openai_defaults() -> Vec<String> {
    let config = OpenAIConfig {
        api_key: Some(SecretString::new("sk-test")),
        default_model: None,
        api_base_url: None,
        timeout_secs: None,
        supported_models: None,
        organization: None,
        project: None,
    };
    let adapter = OpenAiAdapter::new(config, Arc::new(RwLock::new(bundled_pricing())))
        .await
        .expect("adapter must build");
    adapter.metadata().supported_models.clone()
}

async fn gemini_defaults() -> Vec<String> {
    let config = GeminiConfig {
        mode: GeminiMode::Api,
        api_key: Some(SecretString::new("test-key")),
        vertex_project: None,
        vertex_location: None,
        vertex_service_account_json: None,
        default_model: None,
        timeout_secs: None,
        api_base_url: None,
        vertex_base_url_override: None,
        supported_models: None,
        default_thinking_budget: None,
        embed_api_version: None,
    };
    let adapter = GeminiAdapter::new(config)
        .await
        .expect("adapter must build");
    adapter.metadata().supported_models.clone()
}

/// A routing pattern rather than a model ID: a wildcard or a prefix.
fn is_pattern(entry: &str) -> bool {
    entry.ends_with('*') || entry.ends_with('-')
}

fn bundled_models() -> serde_json::Map<String, Value> {
    let asset: Value = serde_json::from_slice(BUNDLED_PRICING_JSON).expect("asset is JSON");
    asset["models"]
        .as_object()
        .expect("asset has a models map")
        .clone()
}

/// Every bundled `provider` entry is listed, under its own ID or an alias, and routes — unless
/// [`NOT_LISTED`] or the lane's `not_default` names it. An announced shutdown does not exempt an
/// entry: a lane lists a model until the refresh that removes its price.
fn assert_every_priced_model_is_listed(provider: &str, defaults: &[String], not_default: &[&str]) {
    let mut checked = 0usize;
    for (id, entry) in &bundled_models() {
        if entry["provider"] != provider
            || not_default.contains(&id.as_str())
            || NOT_LISTED.iter().any(|(skip, _)| skip == id)
        {
            continue;
        }
        let aliases = entry["aliases"].as_array().expect("aliases is an array");
        let names: Vec<&str> = std::iter::once(id.as_str())
            .chain(aliases.iter().filter_map(Value::as_str))
            .collect();
        assert!(
            names
                .iter()
                .any(|name| defaults.iter().any(|listed| listed == name)),
            "`{id}` is priced for `{provider}`, but its default list names neither it nor an \
             alias, so it is missing from /v1/models"
        );
        assert!(
            names
                .iter()
                .any(|name| model_matches_provider(name, defaults)),
            "`{id}` is priced for `{provider}`, but its default list does not route it"
        );
        checked += 1;
    }
    assert!(checked > 0, "no bundled `{provider}` entry was checked");
}

/// Every model ID a default list names resolves in the bundled snapshot.
fn assert_every_listed_model_is_priced(lane: &str, defaults: &[String]) {
    let db = bundled_pricing();
    let inner = db.read();
    for entry in defaults.iter().filter(|entry| !is_pattern(entry)) {
        assert!(
            inner.lookup(entry, None).is_some(),
            "{lane} default list names `{entry}`, which the bundled snapshot does not price"
        );
    }
}

#[tokio::test]
async fn anthropic_default_list_matches_the_priced_anthropic_models() {
    let defaults = anthropic_defaults().await;
    assert_every_priced_model_is_listed("anthropic", &defaults, &[]);
    assert_every_listed_model_is_priced("anthropic", &defaults);
}

#[tokio::test]
async fn openai_default_list_matches_the_priced_openai_models() {
    let defaults = openai_defaults().await;
    assert_every_priced_model_is_listed("openai", &defaults, &[]);
    assert_every_listed_model_is_priced("openai", &defaults);
}

#[tokio::test]
async fn gemini_default_list_matches_the_priced_gemini_models() {
    let defaults = gemini_defaults().await;
    assert_every_priced_model_is_listed("gemini", &defaults, NOT_DEFAULT);
    assert_every_listed_model_is_priced("gemini", &defaults);
    let mut deprecated: Vec<String> = bundled_models()
        .into_iter()
        .filter(|(_, entry)| {
            entry["provider"] == "gemini" && entry.get("deprecation_notice").is_some()
        })
        .map(|(id, _)| id)
        .collect();
    let mut not_default: Vec<String> = NOT_DEFAULT.iter().map(|id| (*id).to_string()).collect();
    deprecated.sort();
    not_default.sort();
    assert_eq!(
        not_default, deprecated,
        "the Gemini lane's NOT_DEFAULT must name exactly the Gemini entries with a shutdown notice"
    );
    for (model, flags) in GEMINI_FLAGS {
        assert_eq!(
            model_flags(model).bits(),
            flags.bits(),
            "`{model}` carries the wrong capability flags"
        );
    }
}

async fn bedrock_defaults() -> Vec<String> {
    let config = BedrockConfig {
        region: "us-east-1".into(),
        access_key_id: Some(SecretString::new("AKIDEXAMPLE")),
        secret_access_key: Some(SecretString::new("secret")),
        session_token: None,
        endpoint_url: None,
        default_model: None,
        timeout_secs: None,
        supported_models: None,
    };
    let adapter = BedrockAdapter::new(config, Arc::new(RwLock::new(bundled_pricing())))
        .await
        .expect("adapter must build");
    adapter.metadata().supported_models.clone()
}

/// Bedrock routes every `anthropic.*` ID, so its list only advertises; it is held to the priced
/// direction alone. Bedrock IDs on the Messages-API endpoint are priced but not listed: this
/// adapter calls the Converse API, which Anthropic documents only for the ARN-versioned IDs.
#[tokio::test]
async fn bedrock_default_list_names_only_priced_models() {
    let defaults = bedrock_defaults().await;
    assert!(
        defaults.iter().any(|m| !is_pattern(m)),
        "bedrock list advertises no model"
    );
    assert_every_listed_model_is_priced("bedrock", &defaults);
}
