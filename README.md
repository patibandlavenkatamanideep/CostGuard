# CostGuard

> **Production-grade LLM reliability and cost optimization platform. Route every agent call through CostGuard — it evaluates validity, tracks cost, fires alerts, and auto-rejects low-quality responses before they reach your users.**

[![CI/CD](https://github.com/patibandlavenkatamanideep/CostGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/patibandlavenkatamanideep/CostGuard/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Powered by RDAB](https://img.shields.io/badge/Evaluation-RealDataAgentBench-7c3aed)](https://github.com/patibandlavenkatamanideep/RealDataAgentBench)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20Now%20%E2%86%92-brightgreen)](https://costguard-production-3afa.up.railway.app)

---

## What Is CostGuard?

CostGuard is a self-hostable reliability and cost-optimization layer for LLM-powered agents and applications. It sits between your code and the LLM provider, giving you:

- **Real-time response validation** — every LLM response is scored for statistical validity before being returned to your application
- **Auto-reject + automatic fallback** — if a response scores below your quality threshold, CostGuard rejects it and retries with the next best model automatically
- **Exact cost tracking** — per-call token accounting at $0.000001 precision across 12 models and 5 providers
- **Comprehensive alerting** — validity drops, cost spikes, high failure rates, circuit breaker events, and consecutive low-quality responses
- **Model benchmarking** — powered by [RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench) across 1,180+ evaluation runs, 39 tasks, 12 models

**Who this is for:** teams running LangGraph, CrewAI, or any custom LLM agent in production who need reliability guarantees and cost control without building their own evaluation infrastructure.

---

## Live Demo

**[costguard-production-3afa.up.railway.app](https://costguard-production-3afa.up.railway.app)**

No account. No setup. Upload a CSV or Parquet file and get a model recommendation in under 15 seconds (Simulation Mode) or 1–3 minutes (Live Mode with your API keys).

---

## Self-Host in One Command

```bash
git clone https://github.com/patibandlavenkatamanideep/CostGuard.git && cd CostGuard
cp .env.example .env   # add at least one API key for proxy mode
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
┌───────────────────────────────────────────┐
│         CostGuard Proxy Layer             │
│                                           │
│  POST /proxy   ← real-time intercept      │
│  POST /evaluate ← batch benchmarking      │
│  GET  /metrics  ← Prometheus scrape       │
│  GET  /health   ← deep health check       │
│                                           │
│  Stack: RequestID → RateLimit →           │
│         Security → Prometheus             │
└────────────┬─────────────┬────────────────┘
             │             │
      ┌──────▼──────┐ ┌────▼──────────────┐
      │ RDAB        │ │ Alerting Engine    │
      │ Evaluator   │ │ Validity | Cost    │
      │ (fast mode) │ │ FailureRate | CB   │
      └──────┬──────┘ └────────────────────┘
             │
      ┌──────▼──────────────────────────────┐
      │  Circuit Breaker (per provider)     │
      │  anthropic | openai | groq | google │
      └──────┬──────────────────────────────┘
             │
      ┌──────▼──────────────────────────────┐
      │  LLM Providers (with backoff)       │
      └─────────────────────────────────────┘
```

---

## The Proxy — Drop-In LLM Guard

Replace your direct LLM call with a POST to `/proxy`. Every response is evaluated before it reaches your code.

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
    "reject_threshold": 0.30,           # reject if validity < 30%
    "fallback_models": ["gpt-4.1", "gemini-2.5-flash"],  # try these on rejection
}).json()

print(response["content"])           # the LLM's response
print(response["accepted"])          # True / False
print(response["validity_score"])    # RDAB scorecard
print(response["cost_usd"])          # exact cost for this call
print(response["fallback_used"])     # True if primary was rejected
```

### Proxy Response Schema
```json
{
  "call_id": "a3f9e1b2",
  "model_id": "claude-sonnet-4-6",
  "provider": "anthropic",
  "content": "The 95% confidence interval for Q3 revenue is...",
  "accepted": true,
  "rejection_reason": null,
  "fallback_used": false,
  "validity_score": {
    "rdab_score": 0.742,
    "correctness": 0.85,
    "code_quality": 0.70,
    "efficiency": 0.80,
    "stat_validity": 0.55
  },
  "latency_ms": 843.2,
  "input_tokens": 1247,
  "output_tokens": 312,
  "cost_usd": 0.00000851,
  "attempts": 1,
  "circuit_breaker_state": "closed"
}
```

---

## Dataset Benchmarking (POST /evaluate)

Upload any CSV or Parquet file. CostGuard runs it through RealDataAgentBench across 12 models and returns the best recommendation with exact cost estimates.

```bash
curl -X POST https://costguard-production-3afa.up.railway.app/evaluate \
  -F "file=@my_data.csv" \
  -F "task_description=Analyze customer churn patterns" \
  -F "num_questions=5"
```

---

## Alerting

Set environment variables to configure alert channels and thresholds.

| Alert Type | Trigger | Default Threshold |
|-----------|---------|------------------|
| `ValidityThreshold` | Response validity below threshold | 0.25 |
| `CostSpike` | Single call cost > N× rolling average | 3× |
| `HighFailureRate` | >N% of recent calls failed | 20% |
| `ConsecutiveLowValidity` | N consecutive low-validity responses | 3 |
| `CircuitBreakerOpen` | Provider circuit breaker opened | — |
| `RateLimit` | 429 response from provider | — |

**Slack:**
```bash
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../...
```

**Generic webhook** (PagerDuty, OpsGenie, etc.):
```bash
export COSTGUARD_ALERT_WEBHOOK_URL=https://your-webhook.example.com/alerts
```

---

## Monitoring

CostGuard exposes a Prometheus `/metrics` endpoint. Key metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `costguard_proxy_requests_total` | Counter | Proxy calls by model, provider, status |
| `costguard_proxy_latency_seconds` | Histogram | LLM call latency per model |
| `costguard_proxy_rejections_total` | Counter | Responses rejected below threshold |
| `costguard_proxy_fallbacks_total` | Counter | Fallbacks triggered (primary → fallback) |
| `costguard_eval_requests_total` | Counter | Dataset evaluations by mode and status |
| `costguard_circuit_breaker_open` | Gauge | 1 if circuit breaker open for a provider |
| `costguard_alerts_fired_total` | Counter | Alerts fired by type and channel |
| `costguard_api_request_duration_seconds` | Histogram | API request latency |

---

## RDAB Scoring Methodology

CostGuard uses [RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench) — **1,180+ evaluation runs across 39 tasks and 12 models** — to score every model on four dimensions:

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| **Correctness** | 50% | Answer accuracy vs ground truth (fuzzy-matched, ±15% tolerance) |
| **Code Quality** | 20% | Vectorised operations, naming conventions, no magic numbers |
| **Efficiency** | 15% | Token + step budget adherence |
| **Stat Validity** | 15% | Reports p-values, confidence intervals, avoids overconfident claims |

**Key RDAB Benchmark Findings (1,180+ runs · 39 tasks · 12 models):**
- **GPT-4.1** = top composite score at $0.013/task — best quality-per-dollar
- **Gemini 2.5 Flash** = cheapest at $0.0015/task; only 20.6% below top score
- **Stat validity gap**: model average 55.8% vs human expert baseline 81.3% — a real capability gap that CostGuard exposes

### Ranking Formula
```
composite = rdab_score × 0.75 + cost_score × 0.25
cost_score = 1 − sqrt(model_cost / max_cost_in_cohort)
```

---

## Supported Models

| Model | Provider | Tier | Input $/1K | RDAB |
|-------|----------|------|-----------|:----:|
| Claude Sonnet 4.6 | Anthropic | Premium | $0.003 | ✓ |
| Claude Opus 4.6 | Anthropic | Premium | $0.015 | ✓ |
| Claude Haiku 4.5 | Anthropic | Economy | $0.00025 | ✓ |
| **GPT-4.1** | OpenAI | Premium | **$0.002** | ✓ **Cost leader** |
| GPT-4.1 mini | OpenAI | Balanced | $0.0004 | ✓ |
| GPT-4.1 nano | OpenAI | Economy | $0.0001 | ✓ |
| GPT-4o | OpenAI | Premium | $0.0025 | ✓ |
| GPT-4o mini | OpenAI | Balanced | $0.00015 | ✓ |
| GPT-5 | OpenAI | Premium | $0.015 | ✓ |
| **Gemini 2.5 Flash** | Google | Economy | **$0.000075** | ✓ **Cheapest** |
| Gemini 2.5 Pro | Google | Premium | $0.00125 | ✓ |
| Llama 3.3 70B (Groq) | Groq | Balanced | $0.00059 | ✓ |
| Mixtral 8x7B (Groq) | Groq | Economy | $0.00024 | ✓ |
| Grok-3 | xAI | Premium | $0.003 | ✓ |
| Grok-3 mini | xAI | Balanced | $0.0003 | ✓ |

---

## Quickstart (Local, No Docker)

```bash
git clone https://github.com/patibandlavenkatamanideep/CostGuard.git
cd CostGuard
cp .env.example .env
pip install -e .
./scripts/dev.sh
```

- Dashboard: **http://localhost:8501**
- API Docs: **http://localhost:8000/docs**

---

## Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/patibandlavenkatamanideep/CostGuard)

```bash
npm install -g @railway/cli
railway login && railway init && railway up
```

**Required environment variables for production:**

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Signing key — generate with `openssl rand -hex 32` |
| `COSTGUARD_DB_PATH` | SQLite path — use a persistent volume mount |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | At least one provider for Live Mode |
| `SLACK_WEBHOOK_URL` | Optional — enables Slack alerting |

---

## Project Structure

```
costguard/
├── backend/
│   ├── main.py            # FastAPI app — routes, middleware wiring
│   ├── proxy.py           # Real-time LLM proxy + auto-reject/fallback  ← NEW
│   ├── alerting.py        # Comprehensive alert engine (6 alert types)  ← NEW
│   ├── metrics.py         # Prometheus metrics + OTEL setup             ← NEW
│   ├── middleware.py      # RequestID + RateLimit + SecurityHeaders      ← NEW
│   ├── circuit_breaker.py # Per-provider circuit breaker                ← NEW
│   ├── config.py          # Pydantic settings
│   ├── models.py          # Request/response schemas
│   └── logger.py          # Structured logging (loguru)
├── evaluation/
│   ├── engine.py          # RDAB evaluation orchestrator
│   ├── observability.py   # SQLite logging + drift detection (WAL mode) ← FIXED
│   ├── data_loader.py     # CSV/Parquet ingestion
│   ├── pricing.py         # 15-model pricing catalogue
│   ├── question_generator.py
│   └── token_counter.py
├── frontend/
│   └── app.py             # Streamlit dashboard
├── deploy/
│   ├── prometheus.yml     # Prometheus scrape config                    ← NEW
│   └── grafana/           # Grafana dashboard + datasource provisioning ← NEW
├── tests/
│   └── test_evaluation.py
├── scripts/
│   ├── dev.sh
│   └── start.sh
├── .github/workflows/ci.yml  # CI + security scanning (Bandit + Trivy) ← UPDATED
├── Dockerfile
├── docker-compose.yml        # With named volume + monitoring profile   ← FIXED
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

### POST `/evaluate` — Dataset benchmarking
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

## Production Readiness Checklist

- [x] Real-time proxy with auto-reject + fallback (`POST /proxy`)
- [x] Per-IP rate limiting (configurable per endpoint)
- [x] Per-provider circuit breaker
- [x] Prometheus metrics endpoint + Grafana docker-compose profile
- [x] 6 alert types: validity, cost spike, failure rate, consecutive low validity, circuit breaker, rate limit
- [x] Slack + generic webhook alert channels
- [x] SQLite WAL mode (concurrent read/write safe)
- [x] Named Docker volume (history DB survives container restarts)
- [x] Request ID propagation through all logs
- [x] Security headers (X-Frame-Options, X-Content-Type-Options, etc.)
- [x] OpenTelemetry traces (activate via OTEL_EXPORTER_OTLP_ENDPOINT)
- [x] Non-root Docker user
- [x] Multi-stage Dockerfile build
- [x] Deep health check (DB + circuit breakers + RDAB availability)
- [x] CI security scanning (Bandit + Trivy + pip-audit)
- [x] Deployment health verification in CI

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md). Report vulnerabilities to security@costguard.dev.

## License

MIT — see [LICENSE](LICENSE).

---

*Built with FastAPI, Streamlit, and RealDataAgentBench. Every LLM call deserves a reliability check before reaching production.*
