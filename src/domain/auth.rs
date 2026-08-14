// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Request identity type — canonical identity for spend, budget, and tagging.
//!
//! Injected into axum request extensions by the auth Tower layer.
//! All downstream middleware and handlers read from extensions — never re-derive identity.

use std::collections::HashMap;

/// Identifies the authenticated caller. Injected into axum request extensions
/// by the auth Tower layer. All downstream middleware and handlers read from
/// extensions — never re-derive identity.
///
/// `id` is the stable budget/spend key. MVP uses the "default"
/// sentinel.
///
/// Both fields use "default" sentinel in MVP; Pro/Enterprise tiers populate
/// real values.
#[derive(Debug, Clone)]
pub struct RequestIdentity {
    /// Stable identity key. "default" for config-auth MVP.
    pub id: String,
    /// Organisation scope. "default" for MVP; real org ID in Pro/Enterprise.
    /// Must scope all spend queries and budget checks.
    pub org_id: String,
    /// Human-readable label for logs/metrics (e.g. "dev-laptop-key").
    /// Identifies keys in logs without exposing the raw key value.
    pub label: Option<String>,
    /// Attribution tags extracted from request headers.
    /// Keys: "team", "project".
    /// NOTE: tags are attribution metadata only. `id` and `org_id` remain unchanged
    /// — they are auth/billing identity, not tag values.
    pub tags: HashMap<String, String>,
}

impl Default for RequestIdentity {
    fn default() -> Self {
        Self {
            id: "default".into(),
            org_id: "default".into(),
            label: None,
            tags: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_identity_default() {
        let identity = RequestIdentity::default();
        assert_eq!(identity.id, "default");
        assert_eq!(identity.org_id, "default");
        assert!(identity.label.is_none());
        assert!(identity.tags.is_empty());
    }
}
