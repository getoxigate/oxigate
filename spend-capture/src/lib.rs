// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Capture and replay of the gateway's upstream provider traffic.
//!
//! The harness sits between the gateway and a provider, at the HTTP boundary and before any
//! parsing, so a recording holds what the provider actually sent rather than the gateway's
//! interpretation of it. A recorded exchange can then be replayed to the gateway locally.
//!
//! - [`capture`] — the recording proxy.
//! - [`recording`] — the versioned on-disk recording format and its atomic commit.
//! - [`redact`] — credential redaction applied before anything is written.
//! - [`replay`] — the server that plays recordings back to the gateway.

pub mod capture;
pub mod recording;
pub mod redact;
pub mod replay;
mod serve;
