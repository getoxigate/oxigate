# OpenAI Provider Implementation Notes

**Provider:** OpenAI  
**Adapter:** `src/providers/openai/`  

---

## Embeddings

The OpenAI adapter supports `POST /v1/embeddings` via the `embeddings()` method.

### Supported models

| Model | Dimensions | Input token limit |
|-------|-----------|-------------------|
| `text-embedding-3-small` | 512, 1536 | 8191 |
| `text-embedding-3-large` | 256, 1024, 3072 | 8191 |
| `text-embedding-ada-002` | 1536 | 8191 |

### Configuration

No extra config required beyond `providers.openai.api_key`. The `supported_models` list in YAML also governs which embedding models the adapter declares (operators may extend it).

### Request forwarding

The request body (`model`, `input`, `dimensions`, `encoding_format`) is forwarded verbatim to `https://api.openai.com/v1/embeddings` (or `api_base_url/v1/embeddings` when overridden).

### Token normalisation

OpenAI's embedding response may return `prompt_tokens: 0` with `total_tokens > 0` on some API versions. The adapter backfills `prompt_tokens` from `total_tokens` in that case so downstream cost tracking is accurate.

### Cost headers

Cost headers (`X-Oxigate-Request-Cost`, `X-Oxigate-Input-Tokens`, `X-Oxigate-Output-Tokens`) are injected on every successful response. `X-Oxigate-Output-Tokens` is always `0` for embeddings.

---

## Token accounting

Both of OpenAI's usage axes report a **subset** of a larger figure rather than a separate quantity
beside it. The gateway declares this once, in `src/providers/openai/utils.rs`, and applies it on
both the streaming and non-streaming paths.

| Axis | Semantics | Source |
|---|---|---|
| Cache | `prompt_tokens` **contains** `prompt_tokens_details.cached_tokens` | `developers.openai.com/api/docs/guides/prompt-caching` |
| Reasoning | `completion_tokens` **contains** `completion_tokens_details.reasoning_tokens` | `developers.openai.com/api/docs/guides/reasoning` — reasoning tokens "still occupy space in the model's context window and are billed as output tokens" |

Billing therefore subtracts the cached tokens from the reported prompt before charging the
remainder at the full input rate, and subtracts the reasoning tokens from the reported completion
before charging the remainder at the standard output rate. Each subset is then charged once more at
its own rate.

`X-Oxigate-Input-Tokens` and `X-Oxigate-Output-Tokens` continue to report the **provider's** figures
unchanged; only what is charged at each rate is affected.

### Cache writes

`prompt_tokens_details.cache_write_tokens` — "the unadjusted number of prompt tokens written to
cache" — is parsed and billed on both the streaming and non-streaming paths.

OpenAI's prompt-caching guide documents exactly one cache duration: "The only supported value,
`30m`, is also the default." A closed one-value vocabulary is what licenses crediting the whole
reported quantity to a class, so the field is billed as a `30m` cache write at the selected tier's
`30m` entry in `cache_write_multipliers`.

**The rate is per model, not adapter-wide.** In the bundled pricing snapshot only the GPT-5.6
family — `gpt-5.6-sol`, `gpt-5.6-terra` and `gpt-5.6-luna` — carries a `30m` multiplier, at
**1.25×** the input rate, matching what OpenAI publishes on those model pages. Every other bundled
OpenAI entry carries no cache-write multiplier at all; see [Models the snapshot does not price for
cache writes](#models-the-snapshot-does-not-price-for-cache-writes) below.

**Where the multiplier applies, it is the total, not a surcharge.** `cache_write_tokens` is
reported *inside* `prompt_tokens`, so the written tokens are carved out of the prompt total before
the remainder is charged at the plain input rate, then charged once at 1.25×. Leaving them inside
the prompt total as well would bill OpenAI's documented 1.25× write at 2.25×.

#### Models the snapshot does not price for cache writes

When the selected tier configures no multiplier for the class, the written tokens are priced at that
tier's cache-write fallback rate — the highest cache-write multiplier the tier itself configures,
and never below the full input rate. For a tier that configures none, that is **1.0×**: the written
tokens are carved out of the prompt total and charged straight back at the plain input rate, so no
cache-write premium is applied and the billed total is unchanged.

A **positive** quantity priced that way pulls the request's cost status to `rate-fallback`, so a
write priced without a contractual rate is visible rather than silent. A reported zero does not
degrade the status — there is nothing that was mispriced. Operators can set
`cache_write_multipliers` on the tier to price it exactly.

An **absent** field and a reported **zero** are different statements and stay different: an absent
field publishes no quantity at all, while `cache_write_tokens: 0` is published as zero. The
quantity the gateway billed is republished on `cache_creation_input_tokens`, so the response body,
the spend row and the budget counter cannot disagree.

### OpenAI-compatible backends are not covered by this

A third-party backend served through the `openai_compat` adapter speaks the same wire format, which
proves what fields it emits — not how it counts them. Those instances keep charging the reported
completion total whole **and** the reasoning breakdown beside it, so a compat backend that reports
reasoning inside its completion total is charged twice for that subset.

This is deliberate. Declaring OpenAI's accounting for arbitrary third-party backends would be an
unverified billing change; the declaration will be made per backend once a captured payload
establishes what that backend actually reports. See `openai-compat.md`.

---

## Reasoning models (o-series)

| Feature | Behaviour |
|---------|-----------|
| `max_tokens` | Converted to `max_completion_tokens` |
| `system` role | Converted to `developer` role |
| `temperature` / `top_p` | Stripped for o1-series; forwarded for o3/o4-series |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-09-02 | `prompt_tokens_details.cache_write_tokens` billed as a `30m` cache write, net of the input charge. Cost rises on cache-writing requests to the bundled GPT-5.6 models (1.25× input); other bundled entries price no multiplier, so their billed total is unchanged and the request reports `rate-fallback` |
| 2026-08-23 | Reasoning tokens no longer charged twice — declared as contained in `completion_tokens`. Cost drops on reasoning-heavy requests |
| 2026-05-09 | embeddings() impl, EmbeddingCapabilities, cost headers for /v1/embeddings |
| 2026-05-05 | Initial OpenAI adapter |
