// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Provider adapters — outbound adapters implementing ProviderAdapter.
//!
//! Use [build_from_config] to construct the active provider from gateway config.
//! ProviderRouter dispatches by model name when multiple providers are configured.

use std::sync::Arc;

use tokio::sync::RwLock;

use crate::config::{GatewayConfig, RoutingStrategyKind};
use crate::domain::ports::{ProviderAdapter, ProviderAdapterExt, ProviderError};
use crate::domain::pricing::PricingDb;
use crate::domain::routing::{LowestCost, RateLimitAware, WeightedRandom};
use crate::redis_pool::RedisPool;

pub mod anthropic;
pub mod azure;
pub mod bedrock;
pub mod gemini;
pub mod health;
pub mod openai;
pub mod openai_compat;
pub mod router;
pub mod tool_limits;

pub use anthropic::AnthropicAdapter;
pub use azure::AzureAdapter;
pub use bedrock::BedrockAdapter;
pub use gemini::GeminiAdapter;
pub use health::{InFlightGuard, ProviderHealthTracker};
pub use openai::OpenAiAdapter;
pub use openai_compat::{CompatHttpClient, OpenAICompatAdapter};
pub use router::ProviderRouter;

/// Returns the leaf adapters from a (potentially composite) root adapter.
///
/// When `root` is a `ProviderRouter`, returns its underlying per-provider slice.
/// When `root` is a single adapter (no composite), wraps it in a one-element `Vec`.
///
/// Use this wherever code needs to iterate individual adapters — e.g. health checks
/// at startup and the `/v1/models` handler — to avoid duplicating the
/// `as_providers_slice().map(...).unwrap_or_else(|| vec![root])` pattern.
pub fn leaf_adapters(root: &Arc<dyn ProviderAdapterExt>) -> Vec<Arc<dyn ProviderAdapter>> {
    root.as_providers_slice()
        .map(|s| s.to_vec())
        .unwrap_or_else(|| vec![Arc::clone(root) as Arc<dyn ProviderAdapter>])
}

/// Builds the active provider adapter and health tracker from gateway config.
///
/// `existing_tracker`: pass `Some(tracker)` on SIGHUP reload to preserve cooldown/EWMA state.
///                     Pass `None` at startup to create a fresh tracker.
///
/// Returns `(provider_router, health_tracker)`. The same Arc<ProviderHealthTracker>
/// remains in `AppState.health` on SIGHUP — it is updated in-place, never swapped.
pub async fn build_from_config(
    config: &GatewayConfig,
    pricing_db: Arc<std::sync::RwLock<PricingDb>>,
    redis: Option<Arc<RwLock<RedisPool>>>,
    existing_tracker: Option<Arc<ProviderHealthTracker>>,
) -> Result<(Arc<dyn ProviderAdapterExt>, Arc<ProviderHealthTracker>), ProviderError> {
    let mut providers: Vec<Arc<dyn ProviderAdapter>> = Vec::new();
    let mut provider_names: Vec<String> = Vec::new();

    if let Some(ref openai_cfg) = config.providers.openai {
        let adapter = Arc::new(OpenAiAdapter::new(openai_cfg.clone()).await?);
        provider_names.push(adapter.metadata().name.clone());
        providers.push(adapter as Arc<dyn ProviderAdapter>);
    }
    if let Some(ref anthropic_config) = config.providers.anthropic {
        let adapter = Arc::new(AnthropicAdapter::new(anthropic_config.clone()).await?);
        provider_names.push(adapter.metadata().name.clone());
        providers.push(adapter as Arc<dyn ProviderAdapter>);
    }
    if let Some(ref gemini_config) = config.providers.gemini {
        let adapter = Arc::new(GeminiAdapter::new(gemini_config.clone()).await?);
        provider_names.push(adapter.metadata().name.clone());
        providers.push(adapter as Arc<dyn ProviderAdapter>);
    }
    if let Some(ref bedrock_cfg) = config.providers.bedrock {
        let adapter = Arc::new(BedrockAdapter::new(bedrock_cfg.clone()).await?);
        provider_names.push(adapter.metadata().name.clone());
        providers.push(adapter as Arc<dyn ProviderAdapter>);
    }

    if !config.providers.openai_compat.is_empty() {
        let compat_http = Arc::new(CompatHttpClient::new()?);
        for compat_cfg in &config.providers.openai_compat {
            let adapter = Arc::new(
                OpenAICompatAdapter::new(compat_cfg.clone(), Arc::clone(&compat_http)).await?,
            );
            provider_names.push(adapter.metadata().name.clone());
            providers.push(adapter as Arc<dyn ProviderAdapter>);
        }
    }

    if !config.providers.azure.is_empty() {
        let azure_http = Arc::new(CompatHttpClient::new()?);
        for azure_cfg in &config.providers.azure {
            let adapter =
                Arc::new(AzureAdapter::new(azure_cfg.clone(), Arc::clone(&azure_http)).await?);
            provider_names.push(adapter.metadata().name.clone());
            providers.push(adapter as Arc<dyn ProviderAdapter>);
        }
    }

    // Resolve or create the health tracker.
    let tracker = if let Some(t) = existing_tracker {
        // SIGHUP: update tracker in-place; cooldown + EWMA state preserved for survivors.
        // Also update routing params so operator changes to cooldown_secs / latency_ewma_alpha
        // in YAML take effect without a full restart.
        t.update_routing_params(
            config.routing.cooldown_secs,
            config.routing.latency_ewma_alpha,
        );
        t.sync_providers(&provider_names).await;
        t
    } else {
        // Startup: create fresh tracker.
        ProviderHealthTracker::new(
            &provider_names,
            redis,
            config.routing.cooldown_secs,
            config.routing.latency_ewma_alpha,
        )
    };

    // Build strategy from config.
    let strategy: Arc<dyn crate::domain::ports::RoutingStrategy> = match config.routing.strategy {
        RoutingStrategyKind::WeightedRandom => Arc::new(WeightedRandom),
        RoutingStrategyKind::RateLimitAware => Arc::new(RateLimitAware),
        RoutingStrategyKind::LowestCost => Arc::new(LowestCost),
    };

    let router = Arc::new(ProviderRouter::new_with_resilience(
        providers,
        strategy,
        Arc::clone(&tracker),
        pricing_db,
        config.routing.clone(),
        config.retry.clone(),
        config.fallbacks.clone(),
        config.security.clone(),
    ));

    Ok((router, tracker))
}
