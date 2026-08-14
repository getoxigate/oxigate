# Contributing to OxiGate

Thank you for your interest in contributing!

## Contributor License Agreement (CLA)

Before your first pull request can be merged, you must sign the OxiGate CLA.
[CLA Assistant](https://cla-assistant.io) will post a comment on your PR with
a one-click sign link — it takes about 60 seconds. You only need to sign once.

**Why a CLA?** OxiGate is dual-licensed: the community edition is AGPLv3; Pro
and Enterprise editions are sold under a commercial license. The CLA gives the
maintainer the right to include your contribution in both. You retain full
copyright over your work.

## Getting Started

```bash
git clone https://github.com/getoxigate/oxigate
cd oxigate
cargo build
cargo xtask check   # fmt + clippy + nextest
```

## Pull Requests

- One feature per PR; keep diffs focused.
- All PRs must pass `cargo xtask check` (fmt, clippy `-D warnings`, nextest).
- Add or update tests for any behaviour change.
- Update `docs/api.md` if you change a public endpoint.

## Source file headers

Every Rust source file must begin with:

```rust
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
```

"OxiGate contributors" is attribution shorthand for the individuals who
hold copyright in their commits — collective attribution, not a legal
entity.

## License

Contributions are licensed to the public under AGPLv3 per [LICENSE](LICENSE).
The CLA additionally grants the maintainer rights to include your contribution
in commercial editions.
