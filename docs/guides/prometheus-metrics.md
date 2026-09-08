# OxiGate Prometheus Metrics Guide

## Endpoint

| Property | Value |
|---|---|
| Path | `GET /metrics` |
| Auth | **None** — no Authorization header required |
| Format | Prometheus text exposition format (v0.0.4) |

> **Security note:** The `/metrics` endpoint is intentionally unauthenticated.
> Protect it at the network level — firewall, Kubernetes `NetworkPolicy`, or reverse-proxy
> allow-list — to prevent exposing operational data to untrusted clients.

---

## Metric Reference

### Baseline request metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `oxigate_requests_total` | Counter | `method`, `status`, `provider`, `endpoint` | Total LLM requests dispatched. Incremented once per request. |
| `oxigate_request_duration_seconds` | Histogram | `provider`, `endpoint` | End-to-end request latency in seconds (time-to-first-byte for streaming). Explicit buckets: `0.001 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1.0 2.5 5.0 10.0`. Every other histogram on this page uses the exporter's default buckets. |
| `oxigate_cost_usd_total` | Counter | `provider`, `endpoint` | Accumulated request cost in **nano-USD** (divide by 1e9 in PromQL for USD). |
| `oxigate_active_connections` | Gauge | _(none)_ | Current number of in-flight LLM requests (decremented on client disconnect). |

**Label values:**
- `method` — HTTP method string, e.g. `POST`
- `status` — HTTP status code string, e.g. `200`, `401`, `429`
- `provider` — stable lowercase provider name, e.g. `openai`, `anthropic`, `gemini`;
  `unknown` when a request failed before a provider was selected
- `endpoint` — which route served the request: `chat`, `embeddings`, or `other`.
  `oxigate_cost_usd_total` only ever carries `chat` or `embeddings`, since no other route
  has a cost.

**Labels that are never emitted:**
`key_id`, `user_id`, `model`, `model_family` — high cardinality; spend attribution by key is available via the `/spend` API.

---

### Fallback + retry metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `oxigate_fallback_trigger_total` | Counter | `trigger` | Incremented once per fallback dispatch. `trigger` is the snake_case trigger type (e.g. `rate_limit`, `timeout`). |
| `oxigate_retry_attempt_total` | Counter | `provider`, `trigger` | Incremented once per same-provider retry. |
| `oxigate_fallback_skip_total` | Counter | `reason` | Incremented once per skipped fallback target. `reason` is the skip reason (e.g. `trigger_not_allowed`, `in_cooldown`, `any`). |
| `oxigate_fallback_resolution_seconds` | Histogram | _(none)_ | Start-to-terminal latency for the full fallback resolution pipeline (seconds). |
| `oxigate_fallback_resolution_attempts` | Histogram | _(none)_ | Total dispatched attempts (retries + fallback targets) per request. |

---

### Embeddings metrics

Scoped to `POST /v1/embeddings`. The baseline metrics above also fire for this route
(with `endpoint="embeddings"`); these add the breakdown that is specific to embeddings.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `oxigate_embeddings_total` | Counter | `provider`, `status` | Embedding requests dispatched. `status` here is a semantic outcome — `success` or `error` — not the HTTP status code that `oxigate_requests_total` carries. |
| `oxigate_embeddings_duration_seconds` | Histogram | `provider` | Embedding request latency in seconds. |
| `oxigate_embeddings_vectors_total` | Counter | `provider` | Cumulative embedding vectors returned. Incremented by the batch size of each successful response, so it grows faster than `oxigate_embeddings_total`. |

---

### Usage-accounting integrity

| Metric | Type | Labels | Description |
|---|---|---|---|
| `oxigate_usage_invariant_violation_total` | Counter | `kind` | Incremented once per provider usage payload whose own reported numbers contradict the accounting that provider's contract declares. |

**`kind` label values** (a closed set — no other value is ever emitted):

| Value | Meaning |
|---|---|
| `cache_exceeds_prompt` | The provider reported more cached tokens than the prompt total those cached tokens are documented to be part of. |
| `reasoning_exceeds_completion` | The provider reported more reasoning tokens than the completion total those reasoning tokens are documented to sit inside. |

This counter means *the provider's own numbers do not add up*. It is not a gateway error and
it does not mean a cost is wrong — the request is still served and still priced. Treat a
non-zero rate as a signal to check that provider's usage reporting before trusting
fine-grained cost attribution for it.

```promql
sum(rate(oxigate_usage_invariant_violation_total[1h])) by (kind)
```

---

## PromQL Examples

### Request rate (requests/second, last 5 min)

```promql
sum(rate(oxigate_requests_total[5m])) by (provider)
```

### P99 latency by provider

```promql
histogram_quantile(0.99,
  sum(rate(oxigate_request_duration_seconds_bucket[5m])) by (le, provider)
)
```

### Cost per provider per minute (USD)

```promql
sum(rate(oxigate_cost_usd_total[1m])) by (provider) / 1e9
```

### Error rate (non-2xx responses)

```promql
sum(rate(oxigate_requests_total{status=~"4..|5.."}[5m])) by (provider, status)
  /
sum(rate(oxigate_requests_total[5m])) by (provider, status)
```

### Active connections

```promql
oxigate_active_connections
```

---

## Why there is no `model` label

Per-model labels are deliberately not emitted. A `model` label is unbounded in practice —
every provider ships new model IDs continuously, and each dated revision
(`claude-3-5-sonnet-20241022`) is its own series — so adding it multiplies the cardinality of
every request and cost metric by a number that only grows.

Per-model spend is available without that cost: query `GET /v1/spend/models`, which reads the
`spend_records` table rather than the metrics registry. See `docs/api.md`.

---

## Prometheus Scrape Configuration

```yaml
scrape_configs:
  - job_name: oxigate
    static_configs:
      - targets: ['oxigate:8080']
    metrics_path: /metrics
```

Or with Kubernetes annotations:

```yaml
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
    prometheus.io/path: "/metrics"
```
