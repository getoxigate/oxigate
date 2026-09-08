# Azure OpenAI adapter

OxiGate's Azure OpenAI adapter forwards chat completions to Azure-hosted OpenAI deployments.
It handles deployment-based URL construction, `api-key` header auth, and always injects
`stream_options.include_usage: true` so streaming responses carry non-zero cost data.

## Quick start

```yaml
providers:
  azure:
    # Two deployments — OxiGate rotates across both per the active RoutingStrategy.
    - name: azure-gpt4o-prod
      endpoint: "https://my-resource.openai.azure.com"
      deployment_name: "gpt-4o"
      api_version: "2024-10-21"
      api_key: "${AZURE_API_KEY_PROD}"
      supported_models:
        - "gpt-4o"
        - "gpt-4o-2024-11-20"

    - name: azure-gpt4o-fallback
      endpoint: "https://my-resource-eu.openai.azure.com"
      deployment_name: "gpt-4o"
      api_version: "2024-10-21"
      api_key: "${AZURE_API_KEY_EU}"
      # supported_models omitted → FallbackOnly (excluded from weighted routing)
```

`name` must be unique across all providers. Convention: `azure-{deployment}-{env}`.
`api_version: "2024-10-21"` is the current GA stable version.

## Feature / behaviour table

| Feature | Status | Notes |
|---------|--------|-------|
| Chat completions (non-streaming) | Supported | Standard OpenAI wire format |
| Chat completions (streaming) | Supported | `stream_options.include_usage: true` always injected |
| Cost tracking (non-streaming) | Supported | Usage in response body; normalized via `normalize_openai_usage` |
| Cost tracking (streaming) | Supported | Usage extracted from final SSE chunk; non-zero when `include_usage` is injected |
| Budget enforcement | Supported | Community — HardCapLayer and SoftCapLayer apply |
| Cache token tracking | Supported | `prompt_tokens_details.cached_tokens` → `cache_read_input_tokens`; `cache_write_tokens` billed as a `30m` write on Standard deployments that report it |
| Tool use / function calling | Not supported | Planned |
| Vision (image inputs) | Not supported | Planned |
| Embeddings | Not supported | Planned |
| APIM auth / managed identity | Not supported | Planned |
| Zero-copy forwarding | Not applicable | Body must be re-serialized to inject `stream_options` |

## Token accounting

Azure OpenAI Service returns OpenAI's `usage` schema, and both of its axes report a **subset** of a
larger figure rather than a separate quantity beside it. Azure carries its own declaration in
`src/providers/openai/utils.rs` — separate from OpenAI's, because it is a separate vendor contract
with its own documentation.

| Axis | Semantics | Source |
|---|---|---|
| Cache | `prompt_tokens` **contains** `prompt_tokens_details.cached_tokens` | Microsoft's prompt-caching guide: "cache hits show up as `cached_tokens` under `prompt_tokens_details`". Its worked response carries `prompt_tokens: 1566` against `cached_tokens: 1408` |
| Reasoning | `completion_tokens` **contains** `completion_tokens_details.reasoning_tokens` | Microsoft's reasoning-models guide: reasoning tokens "occupy space in the context window and are billed as output tokens". Its worked response carries `completion_tokens: 1843` against `reasoning_tokens: 448` |

Billing therefore subtracts the cached tokens from the reported prompt before charging the
remainder at the full input rate, and subtracts the reasoning tokens from the reported completion
before charging the remainder at the standard output rate. Each subset is then charged once more at
its own rate.

`X-Oxigate-Input-Tokens` and `X-Oxigate-Output-Tokens` continue to report Azure's own figures
unchanged; only what is charged at each rate is affected.

### Cache writes

Azure inherits the OpenAI cache-write behaviour described in `providers/openai.md`:
`prompt_tokens_details.cache_write_tokens` is billed as a `30m` cache write at the selected tier's
`30m` multiplier, carved out of the prompt total rather than added to it. Microsoft documents the
same single supported duration.

This applies on **Standard pay-as-you-go** GPT-5.6+ deployments, which are the ones that report the
field. The rate is per model: a tier that configures no `30m` multiplier prices the write at that
tier's cache-write fallback rate — the highest cache-write multiplier the tier itself configures,
never below the full input rate, so 1.0× when it configures none. A **positive** quantity priced
that way reports `rate-fallback`; a reported zero does not degrade the status. See [Models the
snapshot does not price for cache
writes](openai.md#models-the-snapshot-does-not-price-for-cache-writes).

**Provisioned (PTU-M) deployments do not expose `cache_write_tokens` at all**, and nothing changes
for them. An absent field is not a zero: no cache-write quantity is published and no cache-write
charge is made. A deployment that does report `cache_write_tokens: 0` is saying something
different, and that zero is published as a zero.

### Streaming carries the Azure declaration, not the compat default

Azure streams through the SSE reader shared with the OpenAI-compatible adapter. That reader takes
the accounting declaration as a **required parameter** rather than inferring it from the payload,
because every contract using it emits the same SSE shape. The Azure adapter passes Azure's
declaration; a generic compat instance passes its own. Streaming and non-streaming Azure requests
are therefore billed under the same semantics.

---

## URL construction

The adapter constructs the upstream URL from config at startup:

```
{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}
```

Example: `https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21`

A trailing slash on `endpoint` is trimmed defensively. `deployment_name` and `api_version` are
validated at startup — they must not contain `/`, `?`, `#`, `&`, `%`, or whitespace (OWASP A03).

## `api-key` vs `Authorization`

Azure OpenAI uses the `api-key` header, not `Authorization: Bearer`:

```
api-key: <your-key>
```

OxiGate never sets `Authorization` for Azure requests. Some Azure deployments reject requests
that include both headers simultaneously.

## API version compatibility

- `"2024-10-21"` is the minimum tested GA version. OxiGate forwards `api_version` verbatim to
  Azure, so newer preview versions (e.g. `"2025-02-01-preview"`) work if your deployment supports them.
  Operators migrating from LiteLLM may be on `"2025-02-01-preview"` — both work.
- `response_format` (JSON mode / structured outputs) requires `api_version >= "2024-08-*"`. Older
  deployments will receive a 400 from Azure. OxiGate forwards the field as-is without
  version-gating; omit `response_format` for deployments on older API versions.

## Deferred capabilities

| Capability | Deferred to |
|-----------|------------|
| Tool use / function calling | |
| Vision (image inputs) | |
| Embeddings (`/v1/embeddings`) | |
| APIM auth / managed identity | Planned |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-09-02 | `prompt_tokens_details.cache_write_tokens` billed as a `30m` cache write, net of the input charge, on Standard deployments that report it. PTU-M deployments are unaffected |
| 2026-08-23 | Reasoning tokens no longer charged twice — declared as contained in `completion_tokens`. Cost drops on reasoning-heavy requests |
