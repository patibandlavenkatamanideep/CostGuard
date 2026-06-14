# Changelog

All notable changes to CostGuard are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- **API-key authentication** — `Authorization: Bearer <COSTGUARD_API_KEY>` required on all non-public routes (`/proxy`, `/v1/chat/completions`, `/evaluate`, `/replay`, `/models`). The server refuses to start in production without a key unless `COSTGUARD_ALLOW_UNAUTHENTICATED=true` is set for an intentional public demo.
- **OpenAI-compatible endpoint** — `POST /v1/chat/completions` is a drop-in for the OpenAI SDK (change only `base_url`). Returns the standard ChatCompletion schema with CostGuard data in `x-costguard-*` headers / a `costguard` field. **SSE streaming** supported (`stream=true`); text is scored and logged after the stream completes.
- **Opt-in validity blocking** — new `enforce` flag on `/proxy` (default `false` = score-and-log). Blocking + validity-based fallback only when `enforce=true`.
- **Pricing freshness** — `PRICING_AS_OF` / `pricing_age_days()` in `evaluation/pricing.py`; `/models` now returns `pricing_as_of` and a CI test fails once the catalogue exceeds `PRICING_MAX_AGE_DAYS`.
- **Tether schema contract** — `tether_reader` validates the Tether `steps` schema before replay and raises a clear `TetherSchemaError` (→ HTTP 400) instead of a cryptic SQL failure; covered by a contract test.
- **README "Integrate in 5 minutes"** conversion section and `ui` healthcheck in `docker-compose.yml`.

### Changed
- Repositioned the `/proxy` scorer honestly as a **fast lexical pre-filter (inspired by RDAB findings)** — not "RDAB-calibrated" — across code and docs. Full RDAB scoring remains on `/evaluate`.
- Corrected a stale RDAB statistical-validity claim in the README (≤0.25 → the real ~0.56 vs 0.81 human baseline) and pinned figures with an "as-of" note.
- Clarified `RDABScoreCard.simulated` to mean "not a live RDAB eval" (true for the proxy pre-filter and simulation-mode eval).

### Security
- Added a threat model and documented the Streamlit dashboard as an **admin-only** surface (it bypasses API auth in-process) in `SECURITY.md`.
- Documented the persistent writable data path (history + circuit-breaker/alert state) and per-platform persistence in `DEPLOYMENT.md`.

## [0.1.0] — 2025-01-01

### Added
- Initial release of CostGuard
- CSV and Parquet file upload support
- 10 LLM models across OpenAI, Anthropic, Google, and Groq
- Accuracy scoring via RealDataAgentBench-style evaluation
- Exact cost estimation per model per run
- One-click copyable config output
- Streamlit dashboard with interactive Plotly charts
- FastAPI backend with full OpenAPI documentation
- Docker + docker-compose support
- GitHub Actions CI/CD pipeline
- Railway and Render deployment configurations
- Structured logging with loguru
- Token counting with tiktoken
- Automatic question generation from dataset schema
