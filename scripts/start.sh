#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# CostGuard — single-container startup script
#
# Railway / Render expose exactly ONE public port ($PORT).
# Strategy:
#   1. Start FastAPI (uvicorn) on localhost:8000 in the background.
#   2. Start Streamlit on $PORT in the foreground (this becomes the public URL).
#   3. Streamlit talks to FastAPI on http://localhost:8000.
#
# Usage (Railway / Render sets $PORT automatically):
#   ./scripts/start.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Allow overriding ports via env vars; Railway injects $PORT for the public face
PUBLIC_PORT="${PORT:-8501}"
API_PORT="${API_PORT:-8000}"

echo "[start.sh] Launching CostGuard..."
echo "[start.sh]   FastAPI  → http://localhost:${API_PORT}"
echo "[start.sh]   Streamlit → http://0.0.0.0:${PUBLIC_PORT}  (public)"

# Export so Streamlit's app.py picks it up
export API_BASE_URL="http://localhost:${API_PORT}"

# ── 1. Start FastAPI in the background ──────────────────────────────────────
uvicorn backend.main:app \
  --host 127.0.0.1 \
  --port "${API_PORT}" \
  --log-level "${LOG_LEVEL:-info}" &

UVICORN_PID=$!
echo "[start.sh] uvicorn started (PID ${UVICORN_PID})"

# ── 2. Wait until FastAPI is healthy (up to 30 s) ────────────────────────────
echo "[start.sh] Waiting for FastAPI /health..."
for i in $(seq 1 30); do
  if curl -sf "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
    echo "[start.sh] FastAPI is healthy (${i}s)"
    break
  fi
  sleep 1
done

# ── 3. Start Streamlit in the foreground ────────────────────────────────────
exec streamlit run frontend/app.py \
  --server.port "${PUBLIC_PORT}" \
  --server.address "0.0.0.0" \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
