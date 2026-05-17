// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! GET /v1/models — OpenAI-compatible model list with OxiGate extensions.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;

use crate::api::AppState;
use crate::domain::ports::{HealthStatus, ProviderAdapter};
use crate::domain::pricing::PricingDbInner;
use crate::providers::leaf_adapters;

/// Response envelope for GET /v1/models. OpenAI-compatible; `oxigate` extension in each entry.
#[derive(Debug, serde::Serialize)]
pub struct ModelListResponse {
    /// Always "list" for OpenAI compatibility.
    pub object: &'static str,
    /// Model entries; excludes wildcard "*".
    pub data: Vec<ModelEntry>,
}

/// Single model entry. Standard OpenAI fields + `oxigate` extension.
#[derive(Debug, serde::Serialize)]
pub struct ModelEntry {
    pub id: String,
    /// Always "model" for OpenAI compatibility.
    pub object: &'static str,
    /// Gateway startup timestamp (Unix seconds).
    pub created: u64,
    /// Provider name (e.g. "openai", "deepseek").
    pub owned_by: String,
    /// OxiGate-specific metadata.
    pub oxigate: OxigateModelMeta,
}

/// OxiGate extension per model. Pricing, capabilities, health.
#[derive(Debug, serde::Serialize)]
pub struct OxigateModelMeta {
    pub provider: String,
    pub context_window: Option<u32>,
    pub supports_streaming: bool,
    pub supports_tools: bool,
    pub supports_vision: bool,
    pub supports_embeddings: bool,
    pub supports_thinking: bool,
    pub cost_per_input_token_usd: Option<f64>,
    pub cost_per_output_token_usd: Option<f64>,
    /// "available" | "degraded" | "unavailable" | "unknown" from health tracker.
    pub health_status: String,
}

fn health_status_str(status: Option<&HealthStatus>) -> String {
    match status {
        Some(HealthStatus::Healthy) => "available".to_string(),
        Some(HealthStatus::Degraded) => "degraded".to_string(),
        Some(HealthStatus::Unhealthy) => "unavailable".to_string(),
        Some(HealthStatus::Unknown) | None => "unknown".to_string(),
    }
}

/// Pure model-entry builder; no I/O, no locks.
///
/// Extracted from the handler so unit tests can exercise the mapping logic
/// without needing a real `AppState` (which requires live DB/Redis pools).
pub(crate) fn build_model_entries(
    providers: &[Arc<dyn ProviderAdapter>],
    pricing: &PricingDbInner,
    health_statuses: &[(String, HealthStatus)],
    startup_time: u64,
) -> Vec<ModelEntry> {
    let mut data = Vec::new();
    for adapter in providers {
        let meta = adapter.metadata();
        let health = health_status_str(
            health_statuses
                .iter()
                .find(|(name, _)| name == &meta.name)
                .map(|(_, s)| s),
        );

        for model_id in &meta.supported_models {
            if model_id == "*" {
                continue;
            }
            let entry = pricing.lookup(model_id, None);
            let (context_window, cost_in, cost_out) = match entry {
                Some(e) => {
                    let tier = e.tiers.first();
                    (
                        Some(e.context_window),
                        tier.map(|t| t.input_per_token),
                        tier.map(|t| t.output_per_token),
                    )
                }
                None => (None, None, None),
            };

            data.push(ModelEntry {
                id: model_id.to_string(),
                object: "model",
                created: startup_time,
                owned_by: meta.name.clone(),
                oxigate: OxigateModelMeta {
                    provider: meta.name.clone(),
                    context_window,
                    supports_streaming: meta.supports_streaming,
                    supports_tools: meta.supports_tools,
                    supports_vision: meta.supports_vision,
                    supports_embeddings: meta.supports_embeddings,
                    supports_thinking: meta.supports_thinking,
                    cost_per_input_token_usd: cost_in,
                    cost_per_output_token_usd: cost_out,
                    health_status: health.clone(),
                },
            });
        }
    }
    data
}

/// Lists all models routable through configured providers.
///
/// Excludes wildcard "*" entries. Models not in the pricing DB are included
/// with null context_window and cost fields.
pub async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    // Fetch provider list first (async) and drop the guard before acquiring
    // the pricing lock — RwLockReadGuard is not Send, must not be held across await.
    let providers = {
        let provider = state.provider.read().await;
        leaf_adapters(&provider)
    };

    // Fetch health statuses asynchronously before acquiring any sync locks.
    let health_statuses = state.health.provider_statuses().await;

    // Two-level lock: AppState.pricing_db is Arc<std::sync::RwLock<PricingDb>>;
    // PricingDb itself wraps a second Arc<RwLock<PricingDbInner>> for its own
    // internal reload boundary. Acquire the outer guard first, then the inner.
    // Both are read-only here; no write path holds these in the opposite order,
    // so there is no deadlock risk. Both guards are released at end of scope.
    // No .await after this point — safe to hold !Send guards.
    let db_guard = state
        .pricing_db
        .read()
        .expect("pricing DB lock poisoned — process should restart");
    let pricing_inner = db_guard.read();

    let data = build_model_entries(
        &providers,
        &pricing_inner,
        &health_statuses,
        state.startup_time,
    );

    Json(ModelListResponse {
        object: "list",
        data,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;

    use crate::config::PricingConfig;
    use crate::domain::chat::{ChatRequest, ChatResponse};
    use crate::domain::ports::{HealthStatus, ProviderAdapter, ProviderError, ProviderMetadata};
    use crate::domain::pricing::{BUNDLED_PRICING_JSON, PricingDb};

    use super::build_model_entries;

    // ---------------------------------------------------------------------------
    // Minimal stub adapter for unit tests
    // ---------------------------------------------------------------------------

    struct StubProvider {
        metadata: ProviderMetadata,
    }

    impl StubProvider {
        fn new(name: &str, models: &[&str]) -> Arc<dyn ProviderAdapter> {
            Arc::new(Self {
                metadata: ProviderMetadata {
                    name: name.to_string(),
                    supported_models: models.iter().map(|s| (*s).to_string()).collect(),
                    supports_streaming: true,
                    supports_tools: true,
                    supports_vision: false,
                    supports_embeddings: false,
                    supports_thinking: false,
                    kind: Default::default(),
                    ..Default::default()
                },
            })
        }
    }

    #[async_trait]
    impl ProviderAdapter for StubProvider {
        async fn chat_completion(&self, _req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
            Err(ProviderError::NotImplemented)
        }
        fn metadata(&self) -> &ProviderMetadata {
            &self.metadata
        }
        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }

    fn bundled_pricing() -> PricingDb {
        PricingDb::load(BUNDLED_PRICING_JSON, &PricingConfig::default())
            .expect("bundled pricing must parse")
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_list_models_excludes_wildcard() {
        let providers = vec![StubProvider::new("compat-test", &["*"])];
        let db = bundled_pricing();
        let pricing = db.read();
        let entries = build_model_entries(&providers, &pricing, &[], 1);
        assert!(
            entries.is_empty(),
            "wildcard-only provider must contribute zero entries; got: {:?}",
            entries.iter().map(|e| &e.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_list_models_populates_pricing() {
        // gpt-4o is in the bundled pricing DB.
        let providers = vec![StubProvider::new("openai", &["gpt-4o"])];
        let db = bundled_pricing();
        let pricing = db.read();
        let entries = build_model_entries(&providers, &pricing, &[], 1);
        assert_eq!(entries.len(), 1);
        let oxi = &entries[0].oxigate;
        assert!(
            oxi.context_window.is_some(),
            "context_window must be non-null for a model in the pricing DB"
        );
        assert!(
            oxi.cost_per_input_token_usd.is_some(),
            "cost_per_input_token_usd must be non-null for a model in the pricing DB"
        );
        assert!(
            oxi.cost_per_output_token_usd.is_some(),
            "cost_per_output_token_usd must be non-null for a model in the pricing DB"
        );
    }

    #[test]
    fn test_list_models_null_pricing_when_missing() {
        // This model id is intentionally not in the pricing DB.
        let providers = vec![StubProvider::new("custom", &["custom-model-xyz-not-in-db"])];
        let db = bundled_pricing();
        let pricing = db.read();
        let entries = build_model_entries(&providers, &pricing, &[], 1);
        assert_eq!(
            entries.len(),
            1,
            "model absent from pricing DB must still appear in response"
        );
        let oxi = &entries[0].oxigate;
        assert!(
            oxi.context_window.is_none(),
            "context_window must be null when model not in pricing DB"
        );
        assert!(
            oxi.cost_per_input_token_usd.is_none(),
            "cost_per_input_token_usd must be null"
        );
        assert!(
            oxi.cost_per_output_token_usd.is_none(),
            "cost_per_output_token_usd must be null"
        );
    }

    #[test]
    fn test_list_models_owned_by_derived() {
        let providers = vec![StubProvider::new("openai", &["some-model"])];
        let db = bundled_pricing();
        let pricing = db.read();
        let entries = build_model_entries(&providers, &pricing, &[], 42);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].owned_by, "openai");
        assert_eq!(entries[0].oxigate.provider, "openai");
        assert_eq!(entries[0].created, 42);
    }

    #[test]
    fn test_list_models_health_available() {
        let providers = vec![StubProvider::new("anthropic", &["claude-3-5-sonnet"])];
        let db = bundled_pricing();
        let pricing = db.read();
        let health_statuses = vec![("anthropic".to_string(), HealthStatus::Healthy)];
        let entries = build_model_entries(&providers, &pricing, &health_statuses, 1);
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].oxigate.health_status, "available",
            "health_status must be \"available\" when provider is Healthy"
        );
    }
}
