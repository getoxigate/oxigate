# OxiGate

**Cloud FinOps for LLMs.** A high-performance, open-source LLM gateway written in Rust — with hard budget enforcement, cost attribution, and spend analytics built into the request path.

Drop-in OpenAI-compatible proxy. Route to any provider, track every dollar, enforce budgets before they blow.

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              OxiGate Gateway            │
                        │                                         │
 OpenAI-compatible ────►│  Auth → Budget → Cost → Tag → Route     │────► OpenAI
 client (any SDK)       │                                         │────► Anthropic
                        │  GlobalSafetyLayer (instance-wide cap)  │────► Gemini
                        │  HardCapLayer     (per-identity cap)    │────► Bedrock
                        │  BudgetLayer      (soft alerts)         │────► Azure OpenAI
                        │  TaggerLayer      (team / project)      │────► any OpenAI-compat
                        │  CostLayer        (nano-USD accounting) │
                        │  RouterLayer      (strategy + fallback) │
                        │                                         │
                        │  PostgreSQL ◄── spend_records           │
                        │  Redis      ◄── atomic spend counters   │
                        └─────────────────────────────────────────┘
```

Tower middleware pipeline. Every request passes through auth, budget check, cost
accounting, tagging, and routing — in microseconds.

---

## Provider status

| Provider | Chat | Streaming | Tool use | Embeddings | Cost tracking |
|---|---|---|---|---|---|
| OpenAI | ✅ | ✅ | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | ✅ | — | ✅ (cache pricing) |
| Google Gemini / Vertex | ✅ | ✅ | ✅ | ✅ | ✅ (tiered + thinking) |
| AWS Bedrock (Converse) | ✅ | ✅ | ✅ | — | ✅ (per-region) |
| Azure OpenAI | ✅ | ✅ | planned | planned | ✅ |
| OpenAI-compatible (DeepSeek, Groq, Together AI, Mistral, …) | ✅ | ✅ | varies | — | ✅ |

## Feature status

| Feature | Status |
|---|---|
| Hard + soft budget caps (per-identity) | ✅ |
| Instance-wide safety cap (GlobalSafetyLayer) | ✅ |
| Per-team and per-tag budget enforcement | ✅ |
| Budget reset — daily / weekly / monthly | ✅ |
| Progressive alerts (80 / 90 / 100%) | ✅ |
| Response cost headers (`X-Oxigate-*`) | ✅ |
| Spend tracking — Redis counters + PostgreSQL audit | ✅ |
| Spend query API (`GET /v1/spend/*`) | ✅ |
| Request tagging (`X-OxiGate-Team`, `X-OxiGate-Project`) | ✅ |
| Prometheus metrics (`/metrics`) | ✅ |
| Structured JSON logging | ✅ |
| Load balancing — weighted-random, rate-limit-aware, lowest-cost | ✅ |
| Fallback + retry with exponential backoff | ✅ |
| Config-based auth (single Bearer token) | ✅ |
| YAML config + env var overrides + SIGHUP hot-reload | ✅ |
| Health checks (`/health/ready`, `/health/live`) | ✅ |
| Per-key API key management | planned |
| Rate limiting (RPM / TPM) | planned |
| Plugin system (Rust `.so` + Python) | planned |
| OpenTelemetry OTLP export | planned (Pro) |

---

## Quickstart

### Docker Compose (recommended)

Starts the gateway, PostgreSQL, and Redis in one command.

```bash
git clone https://github.com/getoxigate/oxigate.git
cd oxigate

# Copy the example env and add at least one provider key
cp .env.example .env
# edit .env — set OPENAI_API_KEY or any other provider key

docker compose up -d
```

The gateway is now listening on `http://localhost:8080`.

Test it:

```bash
curl http://localhost:8080/health/ready
# {"status":"ok"}

curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OXIGATE_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

Cost headers appear on every response:

```
X-Oxigate-Request-Cost: 0.000003
X-Oxigate-Input-Tokens: 10
X-Oxigate-Output-Tokens: 25
X-Oxigate-Model-Used: gpt-4o-mini
```

### Build from source

Requires Rust stable, PostgreSQL, and Redis.

```bash
git clone https://github.com/getoxigate/oxigate.git
cd oxigate

cp .env.example .env
# edit .env

# Run database migrations
export DATABASE_URL="postgres://oxigate:secret@localhost:5432/oxigate"
cargo run --bin oxigate -- migrate

# Start the gateway
cargo run --release --bin oxigate -- start --config config/oxigate.yaml
```

---

## Configuration

The gateway is configured via `config/oxigate.yaml` with environment variable overrides.

```yaml
server:
  port: 8080

auth:
  bearer_token: "your-secret-token"   # or set OXIGATE__AUTH__BEARER_TOKEN

providers:
  openai:
    - name: openai-main
      api_key: "${OPENAI_API_KEY}"

budget:
  global_safety_cap_usd: 100.0        # instance-wide hard stop
```

See [`config/oxigate.yaml`](config/oxigate.yaml) for the full annotated reference and
[`docs/`](docs/) for provider-specific guides.

---

## License

Licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).

---

[oxigate.com](https://oxigate.com) · [info@oxigate.com](mailto:info@oxigate.com)
