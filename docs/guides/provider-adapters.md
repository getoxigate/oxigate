# Provider Adapter Guide: Usage Accounting

This guide is for contributors adding or changing a bundled provider adapter under
`src/providers/`. It covers the one part of an adapter that feeds billing: turning the upstream
response's usage report into the gateway's domain `Usage`. Everything downstream of that — tier
selection, cost, cost status, response headers, the terminal `oxigate.usage` event, the spend row
and the budget counter — is computed once, from that `Usage`, outside the adapter.

A usage mistake in an adapter is a billing mistake. The rules below exist so that one cannot hide
on one path while the other stays correct.

---

## Extraction, then one projection

An adapter reaches usage in two steps:

1. **Extraction** — read the provider's usage members off the wire. This may differ by path: a
   buffered response is one complete object, while a stream restates usage across several events.
2. **Projection** — turn that wire state into a domain `Usage`. There is **exactly one projection
   per wire shape**, and both the buffered and the streaming path go through it.

Extraction may be path-specific; projection may not. If a bucket is mapped in a projection, it is
mapped for both paths at once. Two hand-written `Usage` literals — one per path — are how a stream
ends up billed differently from the same response buffered, with nothing failing.

Keep the wire state small and local to the provider module. Do not introduce one shared carrier
type full of optional fields for every provider; each wire shape reports different things.

## The accounting contract is a parameter

Providers disagree about what their counts contain. `UsageAccounting` records that per provider
contract on two axes:

| Axis | Value | Meaning |
|---|---|---|
| `cache` | `Inclusive` | `prompt_tokens` already contains the cached tokens; accounted cache buckets are carved out of it |
| `cache` | `Additive` | `prompt_tokens` is the plain portion only; cache buckets are charged beside it |
| `reasoning` | `IncludedInOutput` | `completion_tokens` already contains the reasoning tokens; they are carved out before the standard output rate applies |
| `reasoning` | `Additive` | reasoning tokens are reported outside `completion_tokens` and charged beside it |

Rules:

- Declare **one constant per provider contract**, beside the projection that uses it.
- **Every axis the wire shape reports is cited.** Its doc comment quotes the provider's first-party
  documentation and gives the date it was accessed. The Anthropic, Gemini and OpenAI constants, and
  Bedrock's cache axis, show the form.
- **An axis the wire shape does not report is declared inert.** Where no count arrives for an
  axis — Bedrock Converse reports no reasoning tokens — the constant still sets a value, and its doc
  comment says the value is the neutral one, that nothing is charged on that axis because nothing
  populates it, that no source is being relied on, and that evidence must be captured before the
  axis is used.
- **A contract no provider documents is declared an unverified fallback.** A generic contract —
  the OpenAI-compatible default, which covers any backend that accepts the schema — has axes that
  *are* populated and *do* move billing, so it is not neutral. Its doc comment says the values are
  an unverified assumption, states each axis's billing consequence for a backend that counts the
  other way, and requires captured evidence from a specific backend before either value changes.
- Never invent a citation to fill a gap.
- The projection **takes the constant as a parameter**. It never reads a constant itself and never
  infers the contract from an adapter name, a provider string or the payload's shape. Two contracts
  can share a wire format — OpenAI, Azure and generic OpenAI-compatible backends all do — and a
  projection that guessed would give every one of them the same answer.
- Do not rely on `UsageAccounting::default()`. It is the historical default, not a declaration.

## Wire shapes today

| Wire shape | Wire state | Projection | Contracts it serves |
|---|---|---|---|
| Anthropic Messages | `AnthropicUsageState` | `AnthropicUsageState::project` | Anthropic |
| Bedrock Converse | `ConverseUsage` | `converse_usage_to_usage` | Bedrock |
| Gemini `generateContent` | `UsageMetadata` | `usage_metadata_to_usage` | Gemini (AI Studio and Vertex) |
| OpenAI-compatible | the domain `Usage` itself | `normalize_openai_usage` | OpenAI, Azure, generic compat |

A new provider that speaks one of these shapes reuses its projection with its own constant. A
provider with a new shape adds its own wire state and one projection.

**Cumulative stream snapshots are replaced, not summed.** Some providers restate a cumulative usage
snapshot on later stream events. A later snapshot replaces the earlier one; an omitted member leaves
the earlier value standing. If your projection can run more than once per stream, it must read its
state without consuming it.

### The OpenAI-compatible shape

The domain `Usage` is the OpenAI-compatible usage schema, so lanes on that shape deserialize
straight into it; there is no separate wire state. For that shape the rule is:

> Every successfully parsed OpenAI-shaped usage passes through `normalize_openai_usage` exactly once
> before leaving the provider boundary, with its accounting constant supplied explicitly.

`normalize_openai_usage` maps `prompt_tokens_details.cached_tokens` onto the cache-read bucket and
stamps the constant. Its `pricing_context` argument is a lane's declared position on cache writes:
`Some` credits `prompt_tokens_details.cache_write_tokens` as a priced cache write; `None` echoes the
field and accounts nothing, which is correct for a backend with no first-party cache-write contract.

## The usage-site scan

`cargo xtask usage-scan` runs first in `cargo xtask check`. It parses every file under
`src/providers/` and holds each file's production code to an **exact** count of:

- **constructions** — a `Usage { .. }` literal or `Usage::default`; and
- **deserializations** — `Usage` or `ChatResponse` in a parse position: a `let` ascribed with
  either (unless initialized to `None`), a turbofish naming either, or a field naming either on a
  locally declared type that derives `Deserialize`, directly or through a `cfg_attr` that is not
  test-only.

Inside an impl whose self type is `Usage` or `ChatResponse`, `Self` counts as that type, so
`Self { .. }` and `Self::default()` in `impl From<Wire> for Usage` are constructions.

The allowed counts, and what each allowed site is for, are in `ALLOWANCES` in
`xtask/src/usage_scan.rs`. Every file not listed must have none. A count above its allowance is a
new site that should have gone through the projection. A count below it fails too, for one of two
reasons: a site was removed, and the allowance is lowered in the same change; or a site was
rewritten into a form the scan does not count (below), and it is still there — restore a counted
form rather than lowering the allowance.

Code the compiler can only build under `cfg(test)` is not counted. Any other `cfg` — including
`cfg(any(test, feature = "x"))` and `cfg(not(feature = "x"))` — is production code and is scanned.

Renaming `Usage`, `ChatResponse` or `Deserialize` under `src/providers/` — `use … as …` or a type
alias — always fails the scan. The scanner reads names without resolving them, so a rename would
hide every site behind it.

The scan polices **sites**, not behaviour, and only the forms listed above are sites. It reads
syntax, not types, so it cannot see through `serde` into a struct field and cannot prove that a
parsed usage was normalized. Nor does it see anything that reaches `Usage` only through inference —
and some of that carries real token counts:

- a parse whose target type is inferred rather than written at the parse: from the function's
  return type, from the field or argument the result is assigned or passed to
  (`chunk.usage = serde_json::from_value(v).ok()`,
  `StreamChunk::new(d, Some(serde_json::from_value(v)?), m)`), or from a closure parameter's
  annotation (`|u: Usage| …`);
- a `Usage` produced without naming it: `Default::default()` or `.unwrap_or_default()` in a
  `Usage` position, `std::mem::take(&mut r.usage)`, or a call to any function that returns one;
- a type whose `Deserialize` is implemented by hand rather than derived;
- a parse inside a macro body that is not ordinary Rust (a `tracing` field list, `json!`), where
  only constructions are found, by token shape. `stream! { .. }` bodies are ordinary Rust and are
  scanned in full.

The reachability tests below are the backstop, but **only for the entry points they already
test**. A new entry point written in one of these forms passes the scan; it is protected only once
it has its own reachability test, which is why the checklist requires one.

## Checklist for a new adapter

- [ ] **One projection per wire shape.** The adapter's wire shape has exactly one function that
      produces a `Usage`, or reuses an existing shape's projection.
- [ ] **Both paths through it.** The buffered response and every streaming event that carries
      usage reach that same projection. Neither path builds its own `Usage`.
- [ ] **The accounting citation at the projection.** The contract constant is declared beside the
      projection and passed in as a parameter. Every axis the wire shape reports cites the
      provider's first-party documentation with an access date. An axis the wire shape does not
      report is declared inert: neutral value, no source relied on, evidence required before use.
      A contract no provider documents is declared an unverified fallback, with each axis's
      billing consequence stated and backend-specific evidence required before either value
      changes.
- [ ] **OpenAI-shaped lanes normalize once.** Every parsed usage goes through
      `normalize_openai_usage` exactly once, with the lane's own constant.
- [ ] **Parity fixtures.** A test states the same usage once as a buffered body and once as the
      stream's events, and compares the two resulting `Usage` values field by field with the shared
      `assert_usage_parity` helper in `src/providers/usage_parity.rs`. Cover every bucket the shape
      reports — plain input and output, cache reads, cache writes with and without a per-class
      breakdown, reasoning — plus absent usage and, where the provider restates snapshots, a
      restated snapshot.
- [ ] **Exact cost assertions, against synthetic prices.** A test runs a fully populated response
      through the adapter, `build_cost_headers` and `SpendRecord::build`, and asserts the exact
      nano-USD of every cost component, the total and the cost status. Price it against a
      **synthetic, fixed** `PricingDb` entry — an invented model name with arbitrary, permanent
      rates, loaded with `PricingDb::load` from an inline JSON fixture (the `tiered-fixture` model in
      `src/domain/pricing.rs` is the pattern) — and derive the expected figures by hand from those
      fixture rates before running the test. Never use a bundled catalogue entry as the oracle: a
      legitimate price refresh would turn the test red for a reason unrelated to what it checks.
      The payload shape and accounting semantics still come from first-party evidence; only the
      rates are synthetic.
- [ ] **Reachability at every production entry point.** Each entry point that can return usage —
      buffered, streaming, and any raw-forward path — has a test through a mocked upstream that
      asserts the mapped buckets and `usage.accounting == <the lane's constant>`. A projection that
      is correct in a unit test but bypassed on a real request path is the failure this catches.
- [ ] **The scan is green, with allowances exact.** `cargo xtask usage-scan` passes. If the change
      legitimately adds or removes a site, `ALLOWANCES` changes in the same commit, with a technical
      reason for the entry.
- [ ] **Provider doc.** `docs/providers/<name>.md` documents the provider's token accounting on
      both axes and cites the same sources as the constant.
- [ ] **Accounting contract matrix rows.** A new contract adds a buffered and a streamed row to
      `tests/integration/accounting_contract.rs`, priced against that file's synthetic catalogue,
      with a hand-derived oracle in the row's doc comment and a primary fixture that reaches the
      upper tier only through its cache tokens. The two rows state the same usage and must agree on
      cost, status and persisted columns.
- [ ] **Usage-field audit table.** The provider doc lists **every** member, recursively, of the
      usage object(s) the adapter parses — read from a pinned source (a repository commit, or a
      captured revision where the provider serves only its latest), recorded with that revision
      and the access date — and gives each member its disposition: the `Usage` field it maps to,
      or why it is not read. A member that can move the bill and is not read is a gap to be
      tracked, not a row to leave as is.
