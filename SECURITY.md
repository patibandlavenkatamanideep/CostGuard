# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

To report a security vulnerability, please open a private security advisory at https://github.com/patibandlavenkatamanideep/CostGuard/security/advisories/new. Do not file public GitHub issues for security-sensitive reports.

We will respond within 48 hours and aim to release a fix within 7 days of confirmation.

## Security Design

- **Authentication**: All non-public routes (`/proxy`, `/v1/chat/completions`, `/evaluate`, `/replay`, `/models`) require an `Authorization: Bearer <COSTGUARD_API_KEY>` header. `/health`, `/metrics`, and the docs routes are intentionally public.
- **No data persistence**: Uploaded files are processed in memory and never written to permanent storage.
- **No user accounts**: CostGuard collects no personally identifiable information.
- **API keys**: Never logged, never stored beyond the runtime process.
- **File validation**: Uploaded files are type-checked and size-limited before processing.
- **Non-root Docker**: The container runs as a non-root user (UID 1001).
- **CORS**: Strict origin allowlist configured via `CORS_ORIGINS` env var.

## Threat Model

CostGuard is a self-hosted proxy that **handles LLM provider API keys**. Two kinds of keys exist:

1. **Server keys** — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc., set in the environment. Used by `/proxy`, `/v1/chat/completions`, and `/replay` when the caller does not supply their own.
2. **Caller keys** — provider keys passed per-request in the body (`api_key` on `/proxy`, the `*_api_key` form fields on `/evaluate`). Used only for that request; never persisted, never logged.

**The risk:** an instance reachable on a public URL with **no `COSTGUARD_API_KEY`** is an open relay. Anyone who finds it can spend your server keys or route arbitrary traffic through your instance (rate-limited only per-IP). An exposed instance leaks your provider spend, not your data.

**Mitigations:**

- Set `COSTGUARD_API_KEY` (`openssl rand -hex 32`) and send it as `Authorization: Bearer <key>` on every request. In `production` the app **refuses to start** without it.
- To run an intentionally public, read-only demo (no server keys configured), set `COSTGUARD_ALLOW_UNAUTHENTICATED=true`. This is the only supported way to run open, and it logs a startup warning. Do **not** combine it with server provider keys.
- Caller-supplied provider keys are never written to logs or the observability store. Do not add `api_key` / `*_api_key` fields to any log line.

### The Streamlit dashboard is admin-only

The Streamlit UI (`frontend/app.py`) calls the evaluation engine **in-process** and reads the observability DB directly — it does **not** go through the authenticated HTTP API, and Streamlit has no built-in auth. In the single-container deploy (`scripts/start.sh`) only Streamlit is exposed publicly; FastAPI is bound to localhost. Treat the dashboard as a privileged, local-admin surface:

- Do not expose it publicly except as an intentional **simulation-mode** demo (no server provider keys configured).
- For a secured deployment, put the UI behind your own network controls (VPN, reverse-proxy auth, IP allowlist) or run only the FastAPI service and integrate via the API.
- The HTTP API (`/proxy`, `/v1/chat/completions`, `/evaluate`, `/replay`) is the authenticated integration surface; prefer it for anything programmatic.
