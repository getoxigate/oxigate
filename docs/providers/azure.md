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
| Cache token tracking | Supported | `prompt_tokens_details.cached_tokens` → `cache_read_input_tokens` |
| Tool use / function calling | Not supported | Planned |
| Vision (image inputs) | Not supported | Planned |
| Embeddings | Not supported | Planned |
| APIM auth / managed identity | Not supported | Planned |
| Zero-copy forwarding | Not applicable | Body must be re-serialized to inject `stream_options` |

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
