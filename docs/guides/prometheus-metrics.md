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
| `oxigate_requests_total` | Counter | `method`, `status`, `provider` | Total LLM requests dispatched. Incremented once per request. |
| `oxigate_request_duration_seconds` | Histogram | `provider` | End-to-end request latency in seconds (time-to-first-byte for streaming). Explicit buckets: `0.001 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1.0 2.5 5.0 10.0`. |
| `oxigate_cost_usd_total` | Counter | `provider` | Accumulated request cost in **nano-USD** (divide by 1e9 in PromQL for USD). |
| `oxigate_active_connections` | Gauge | _(none)_ | Current number of in-flight LLM requests (decremented on client disconnect). |

**Label values:**
- `method` — HTTP method string, e.g. `POST`
- `status` — HTTP status code string, e.g. `200`, `401`, `429`
- `provider` — stable lowercase provider name, e.g. `openai`, `anthropic`, `gemini`

**Forbidden labels (never emitted in Community tier):**
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

## Pro-Tier Model Labels (deferred)

Per-model labels are intentionally out of scope for Community tier.

- **Pro (opt-in): expose the raw `model` label behind a config flag
  (`metrics.include_model_label: true`). Operator opts in knowing the cardinality implications.
- **No automatic normalisation:** `claude-3-5-sonnet-20241022` is not automatically grouped to
  `claude-3-5`; operator-defined model groupings are planned for the Pro tier.

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
