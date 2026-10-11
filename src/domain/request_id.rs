// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! The gateway-minted identifier of one request.

use std::fmt;

use uuid::Uuid;

/// The identifier the gateway mints for one request.
///
/// Minted once per request at the edge of the `/v1` stack and never taken from the client. It is
/// returned to the client in a response header and stored as the `spend_records.id` of the row
/// the request produced, if any. Treat the value as an opaque UUID: it is unique, and nothing
/// else about it — version, ordering — is part of its contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(Uuid);

impl RequestId {
    /// Mints a new request ID, unique across gateway instances without coordination.
    #[must_use]
    pub fn mint() -> Self {
        Self(Uuid::now_v7())
    }

    /// The ID as a UUID, for binding it to the `spend_records.id` column.
    #[must_use]
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
