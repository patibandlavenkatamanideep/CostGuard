# CostGuard

> **LLM reliability and cost optimization proxy. Route agent calls through CostGuard to get validity scoring, cost tracking, automatic fallbacks, and alerting — without building your own evaluation infrastructure.**

[![CI/CD](https://github.com/patibandlavenkatamanideep/CostGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/patibandlavenkatamanideep/CostGuard/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Powered by RDAB](https://img.shields.io/badge/Evaluation-RealDataAgentBench-7c3aed)](https://github.com/patibandlavenkatamanideep/RealDataAgentBench)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20Now%20%E2%86%92-brightgreen)](https://costguard-production-3afa.up.railway.app)

---

## What Is CostGuard?

CostGuard is a self-hostable reliability layer for LLM-powered agents. It sits between your code and the LLM provider and gives you:

- **Real-time response filtering** — every LLM response is scored with a RDAB-calibrated heuristic validator before being returned. Responses below your quality threshold are rejected automatically.
- **Automatic fallback** — on rejection, CostGuard retries with the next model in your fallback chain. No changes to your application code required.
- **Exact cost tracking** — per-call token accounting at $0.000001 precision across 12 models and 5 providers.
- **Dataset benchmarking** — powered by [RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench) (1,180+ evaluation runs, 39 tasks, 12 models). Upload any CSV/Parquet and get a statistically grounded model recommendation.
- **Alerting** — validity drops, cost spikes, high failure rates, circuit breaker events, and consecutive rejections routed to Slack or any webhook.
- **Per-provider circuit breakers** — automatically stops hammering a failing provider during an outage.

**Who this is for:** teams running LangGraph, CrewAI, or custom LLM agents who need reliability guarantees and cost control without building their own evaluation infrastructure.

---

## How Validity Scoring Works (Be Honest With Yourself)

CostGuard has **two validity modes** — understanding the difference matters:

| Mode | Endpoint | How It Works | Latency |
|------|----------|-------------|---------|
| **Heuristic** | `POST /proxy` | RDAB-calibrated keyword scorer (~1ms) | ~1ms overhead |
| **Full RDAB** | `POST /evaluate` | Actual RDAB agent evaluation with dataset-grounded questions | 15s–3min |

The `/proxy` endpoint uses a fast heuristic scorer: it checks for statistical markers (p-values, confidence intervals, uncertainty quantification) and penalizes failure-mode phrases ("I cannot", "I don't know", error tracebacks). It is **not** a full LLM evaluation — it's a practical pre-filter you can run synchronously on every call without adding meaningful latency.

The `/evaluate` endpoint runs actual [RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench) evaluations grounded in your uploaded dataset, returning four-dimensional RDAB scorecards from 1,180+ benchmark runs across 12 models.

If you need true response quality assurance, use `/evaluate` for batch benchmarking and `/proxy` as a fast sanity filter in your hot path.

---

## Live Demo

**[costguard-production-3afa.up.railway.app](https://costguard-production-3afa.up.railway.app)**

No account. No setup. Upload a CSV or Parquet file and get a model recommendation in under 15 seconds (Simulation Mode) or 1–3 minutes (Live Mode with your API keys).

---

## Self-Host in One Command

```bash
git clone https://github.com/patibandlavenkatamanideep/CostGuard.git && cd CostGuard
cp .env.example .env   # add at least one provider API key
docker compose up
```

- Dashboard → **http://localhost:8501**
- API + Proxy → **http://localhost:8000**
- API Docs → **http://localhost:8000/docs**
- Prometheus Metrics → **http://localhost:8000/metrics**

With Grafana monitoring:
```bash
docker compose --profile monitoring up
# Grafana → http://localhost:3000 (admin / costguard)
```

---

## Architecture

```
Your Agent / LangGraph / CrewAI
        │
        ▼
┌─────────────────────────────────────────────┐
│          CostGuard Middleware Stack          │
│                                             │
│  RequestID → RateLimit → Security →         │
│  Prometheus (every request)                 │
└──────────────┬──────────────────────────────┘
               │
      ┌────────▼────────────────────────────┐
      │         POST /proxy                  │
      │  1. Circuit breaker check            │
      │  2. LLM call (30s timeout)           │
      │  3. Heuristic validity score (~1ms)  │
      │  4. Reject + fallback if score < T   │
      │  5. Cost calculation                 │
      │  6. Async alert checks               │
      │  7. SQLite log (thread pool)         │
      └────────────────────────────────────-─┘
               │
      ┌────────▼────────────────────────────┐
      │      Per-Provider Circuit Breaker   │
      │  CLOSED → (5 failures) → OPEN       │
      │  OPEN   → (60s timeout) → HALF_OPEN │
      │  HALF_OPEN → (2 successes) → CLOSED │
      └────────────────────────────────────-┘
               │
      ┌────────▼────────────────────────────┐
      │     LLM Providers                   │
      │  anthropic | openai | groq |        │
      │  google | xai                       │
      └────────────────────────────────────-┘

      ┌─────────────────────────────────────┐
      │         POST /evaluate              │
      │  Full RDAB benchmarking pipeline    │
      │  (dataset upload → question gen →   │
      │   agent eval → cost-weighted rank)  │
      └────────────────────────────────────-┘
```

---

## The Proxy — Drop-In LLM Guard

Replace your direct LLM call with a POST to `/proxy`.

### Before (no reliability layer)
```python
import anthropic
client = anthropic.Anthropic(api_key="sk-ant-...")
response = client.messages.create(model="claude-sonnet-4-6", ...)
# No validity check. No cost tracking. No fallback.
```

### After (with CostGuard)
```python
import httpx

response = httpx.post("http://costguard:8000/proxy", json={
    "model_id": "claude-sonnet-4-6",
    "prompt": "Analyze Q3 revenue trends and compute 95% confidence intervals.",
    "reject_threshold": 0.30,
    "fallback_models": ["gpt-4.1", "gemini-2.5-flash"],
}).json()

print(response["content"])           # the LLM's response
print(response["accepted"])          # True / False
print(response["validity_score"])    # heuristic scorecard
print(response["cost_usd"])          # exact cost for this call
print(response["fallback_used"])     # True if primary was rejected
```

### Proxy Response Schema
```json
{
  "call_id": "a3f9e1b2c4d5",
  "model_id": "claude-sonnet-4-6",
  "provider": "anthropic",
  "content": "The 95% confidence interval for Q3 revenue is...",
  "accepted": true,
  "rejection_reason": null,
  "fallback_used": false,
  "validity_score": {
    "rdab_score": 0.742,
    "correctness": 0.75,
    "code_quality": 0.70,
    "efficiency": 0.85,
    "stat_validity": 0.45,
    "simulated": true
  },
  "latency_ms": 843.2,
  "input_tokens": 1247,
  "output_tokens": 312,
  "cost_usd": 0.00000851,
  "attempts": 1,
  "circuit_breaker_state": "closed"
}
```

> **Note:** `validity_score.simulated: true` indicates the score came from the heuristic proxy scorer, not a full RDAB evaluation. This is expected for `/proxy` — the fast path.

---

## Dataset Benchmarking (POST /evaluate)

Upload any CSV or Parquet file. CostGuard generates dataset-grounded questions, runs them through all available models using RealDataAgentBench, and returns a ranked recommendation with exact cost estimates.

**Two modes:**
- **Simulation mode** (no API keys): returns calibrated scores from 1,180+ RDAB benchmark runs. Deterministic — same file always produces the same ranking.
- **Live mode** (with API keys): runs real RDAB agent evaluations against your actual dataset.

```bash
curl -X POST http://localhost:8000/evaluate \
  -F "file=@my_data.csv" \
  -F "task_description=Analyze customer churn patterns" \
  -F "num_questions=5"
```

---

## Alerting

Six alert types, all configurable via environment variables.

| Alert Type | Trigger | Default Threshold |
|-----------|---------|------------------|
| `ValidityThreshold` | Response validity below threshold | 0.25 |
| `CostSpike` | Single call cost > N× rolling average | 3× |
| `HighFailureRate` | >N% of recent calls failed | 20% |
| `ConsecutiveLowValidity` | N consecutive rejections from same model | 3 |
| `CircuitBreakerOpen` | Provider circuit breaker opened | — |
| `RateLimit` | 429 response from provider | — |

Alerts fire to console always. Add channels via environment variables:

```bash
# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../...

# Generic webhook (PagerDuty, OpsGenie, custom)
COSTGUARD_ALERT_WEBHOOK_URL=https://your-webhook.example.com/alerts
```

---

## Monitoring

Prometheus `/metrics` endpoint with Grafana dashboard included.

| Metric | Type | Description |
|--------|------|-------------|
| `costguard_proxy_requests_total` | Counter | Proxy calls by model, provider, status |
| `costguard_proxy_latency_seconds` | Histogram | LLM call latency per model |
| `costguard_proxy_rejections_total` | Counter | Responses rejected below threshold |
| `costguard_proxy_fallbacks_total` | Counter | Fallbacks triggered |
| `costguard_eval_requests_total` | Counter | Dataset evaluations by mode and status |
| `costguard_circuit_breaker_open` | Gauge | 1 if circuit breaker open for a provider |
| `costguard_alerts_fired_total` | Counter | Alerts fired by type and channel |
| `costguard_api_request_duration_seconds` | Histogram | API request latency |

---

## RDAB Scoring Methodology

CostGuard uses [RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench) — **1,180+ evaluation runs across 39 tasks and 12 models** — for the `/evaluate` endpoint.

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| **Correctness** | 50% | Answer accuracy vs ground truth (fuzzy-matched, ±15% tolerance) |
| **Code Quality** | 20% | Vectorised operations, naming conventions, no magic numbers |
| **Efficiency** | 15% | Token + step budget adherence |
| **Stat Validity** | 15% | Reports p-values, confidence intervals, avoids overconfident claims |

**Key RDAB Benchmark Findings (1,180+ runs · 39 tasks · 12 models):**
- **GPT-4.1** = top composite score at $0.013/task — best quality-per-dollar
- **Gemini 2.5 Flash** = cheapest at $0.0015/task; only 20.6% below top score
- **Stat validity gap**: model average 55.8% vs human expert baseline 81.3%

### Ranking Formula
```
composite = rdab_score × 0.75 + cost_score × 0.25
cost_score = 1 − sqrt(model_cost / max_cost_in_cohort)
```

---

## Supported Models (12 RDAB-benchmarked)

| Model | Provider | Tier | Input $/1K | Context |
|-------|----------|------|-----------|---------|
| Claude Sonnet 4.6 | Anthropic | Premium | $0.003 | 200K |
| Claude Opus 4.6 | Anthropic | Premium | $0.015 | 200K |
| Claude Haiku 4.5 | Anthropic | Economy | $0.00025 | 200K |
| **GPT-4.1** | OpenAI | Premium | **$0.002** | 1M |
| GPT-4.1 mini | OpenAI | Balanced | $0.0004 | 1M |
| GPT-4.1 nano | OpenAI | Economy | $0.0001 | 1M |
| GPT-4o | OpenAI | Premium | $0.0025 | 128K |
| GPT-4o mini | OpenAI | Balanced | $0.00015 | 128K |
| GPT-5 | OpenAI | Premium | $0.015 | 128K |
| **Gemini 2.5 Flash** | Google | Economy | **$0.000075** | 1M |
| Llama 3.3 70B (Groq) | Groq | Balanced | $0.00059 | 128K |
| Grok-3 mini | xAI | Balanced | $0.0003 | 131K |

---

## Quickstart (Local, No Docker)

```bash
git clone https://github.com/patibandlavenkatamanideep/CostGuard.git
cd CostGuard
cp .env.example .env    # add at least one provider API key
pip install -e .
./scripts/dev.sh
```

- Dashboard: **http://localhost:8501**
- API Docs: **http://localhost:8000/docs**

---

## Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/patibandlavenkatamanideep/CostGuard)

**Required environment variables:**

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Generate with `openssl rand -hex 32` |
| `COSTGUARD_DB_PATH` | SQLite path — use a persistent volume mount |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | At least one provider for Live Mode |
| `SLACK_WEBHOOK_URL` | Optional — enables Slack alerting |

---

## Project Structure

```
costguard/
├── backend/
│   ├── main.py            # FastAPI app — routes, middleware wiring
│   ├── proxy.py           # LLM proxy + heuristic validator + auto-fallback
│   ├── alerting.py        # Alert engine (6 types, Slack + webhook channels)
│   ├── metrics.py         # Prometheus metrics + OpenTelemetry setup
│   ├── middleware.py      # RequestID + LRU-bounded RateLimit + SecurityHeaders
│   ├── circuit_breaker.py # Per-provider circuit breaker (CLOSED/OPEN/HALF_OPEN)
│   ├── config.py          # Pydantic settings (env var management)
│   ├── models.py          # Request/response schemas
│   └── logger.py          # Structured logging (loguru)
├── evaluation/
│   ├── engine.py          # RDAB evaluation orchestrator (live + simulation)
│   ├── observability.py   # SQLite logging + drift detection (WAL mode)
│   ├── data_loader.py     # CSV/Parquet ingestion (multi-encoding robust)
│   ├── pricing.py         # 12-model pricing catalogue
│   ├── question_generator.py
│   └── token_counter.py
├── frontend/
│   └── app.py             # Streamlit dashboard
├── deploy/
│   ├── prometheus.yml     # Prometheus scrape config
│   └── grafana/           # Grafana dashboard + datasource provisioning
├── tests/
│   ├── test_evaluation.py # Evaluation + pricing + data loader tests
│   └── test_proxy.py      # Proxy + circuit breaker + alerting + middleware tests
├── scripts/
│   ├── dev.sh
│   └── start.sh
├── .dockerignore
├── docker-compose.yml     # Named volume + optional monitoring profile
├── Dockerfile             # Multi-stage build (builder + non-root runtime)
└── pyproject.toml
```

---

## API Reference

Full docs at `/docs` (Swagger) and `/redoc`.

### POST `/proxy` — Real-time LLM guard
```bash
curl -X POST http://localhost:8000/proxy \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt-4.1",
    "prompt": "Analyze revenue trends",
    "reject_threshold": 0.30,
    "fallback_models": ["claude-sonnet-4-6"]
  }'
```

### POST `/evaluate` — Full RDAB dataset benchmarking
```bash
curl -X POST http://localhost:8000/evaluate \
  -F "file=@my_data.csv" \
  -F "task_description=Analyze churn patterns"
```

### GET `/health` — Deep health check
```bash
curl http://localhost:8000/health
# {"status":"ok","db_ok":true,"rdab_available":true,"circuit_breakers":{...}}
```

### GET `/metrics` — Prometheus scrape
```bash
curl http://localhost:8000/metrics
```

### GET `/proxy/status` — Circuit breaker states
```bash
curl http://localhost:8000/proxy/status
```

---

## Development

```bash
pytest tests/ -v
ruff check . && ruff format .
mypy backend/ evaluation/
```

---

## Known Limitations

- **Proxy validity scoring is heuristic-only** — the fast path uses keyword matching, not a real LLM evaluation. Use `/evaluate` for statistical validity assessment.
- **Rate limit state is in-memory** — IP-based rate limit buckets reset on server restart. Fine for most deployments.
- **SQLite for single-node persistence** — appropriate for self-hosted single-node deployments. Circuit breaker and alerting state survives process restarts via the `runtime_state` SQLite table. For multi-node or high-throughput deployments, migrate the state store to Redis and replace `observability.py` with PostgreSQL + asyncpg connection pool. Set `COSTGUARD_STATE_BACKEND=none` to disable persistence entirely.

---

## Production Readiness Checklist

| Component | Status | Notes |
|-----------|--------|-------|
| LLM proxy with auto-reject + fallback | ✅ Production-Ready | `POST /proxy` |
| Per-call timeout (30s, per-attempt) | ✅ Production-Ready | `asyncio.timeout` inside retry loop |
| Same-model retry + exponential backoff | ✅ Production-Ready | tenacity: 3 attempts, 1–8s backoff, 429/503/connection errors |
| Per-IP rate limiting (LRU-bounded) | ✅ Production-Ready | Token bucket, capped at 10K IPs |
| Per-provider circuit breaker | ✅ Production-Ready | CLOSED/OPEN/HALF_OPEN state machine |
| Circuit breaker state persistence | ✅ Production-Ready (single-node) | SQLite `runtime_state` table; survives restarts |
| 6 alert types with cooldown | ✅ Production-Ready | Slack + generic webhook; cooldowns persist across restarts |
| Gemini SDK (google-genai ≥ 1.0) | ✅ Production-Ready | Per-call client objects; no global state; full concurrency |
| Prometheus metrics (13 metrics) | ✅ Production-Ready | `/metrics` endpoint |
| Grafana dashboard | ✅ Production-Ready | Auto-provisioned in `--profile monitoring` |
| SQLite WAL mode (observability) | ✅ Production-Ready (single-node) | Concurrent read/write safe |
| Named Docker volume | ✅ Production-Ready | DB survives container restarts |
| Request ID propagation | ✅ Production-Ready | `X-Request-ID` header + log correlation |
| Security headers | ✅ Production-Ready | X-Frame-Options, X-Content-Type-Options, etc. |
| OpenTelemetry traces | ✅ Production-Ready | Opt-in via `OTEL_EXPORTER_OTLP_ENDPOINT` |
| Non-root Docker user | ✅ Production-Ready | UID 1001 |
| Multi-stage Dockerfile | ✅ Production-Ready | Builder + minimal runtime |
| `.dockerignore` | ✅ Production-Ready | Prevents `.env` from baking into image |
| CI security scanning | ✅ Production-Ready | Bandit + Trivy + pip-audit |
| Proxy unit tests (73 total) | ✅ Production-Ready | CB, alerting, persistence, retry, middleware, scorer |
| Load test | ✅ Available | `locust -f tests/locustfile.py` — finds RPS ceiling |
| LLM client connection pooling | ⚠️ Still Demo | Default-key clients cached; per-request keys get fresh clients |
| Proxy heuristic scorer | ⚠️ Still Demo | Fast pre-filter (~1ms), not full RDAB evaluation |
| Multi-replica CB/alert state | ⚠️ Still Demo | Requires Redis for shared state across replicas |
| Observability store (multi-node) | ⚠️ Still Demo | SQLite is single-node only; replace with PostgreSQL + asyncpg |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md). Report vulnerabilities to security@costguard.dev.

## License

MIT — see [LICENSE](LICENSE).

---

*Built with FastAPI, Streamlit, and [RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench). The /evaluate endpoint gives you statistical grounding for model selection. The /proxy endpoint gives you a fast reliability filter in your hot path.*
