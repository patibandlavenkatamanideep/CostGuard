#!/usr/bin/env bash
# CostGuard — single-container startup (Railway / Render)
#
# Railway/Render expose exactly ONE public port ($PORT).
# Strategy:
#   1. Start FastAPI (uvicorn) on localhost:8000 in the background.
#   2. Start Streamlit on $PORT in the background immediately (no wait).
#      The Streamlit UI imports evaluation.engine directly — it does NOT
#      depend on the FastAPI process being alive.
#   3. Wait for either process to exit — exit with its code.
#
# NOTE: We used to block Streamlit on FastAPI health, but that caused
# Railway healthchecks (/_stcore/health) to time out if uvicorn was slow
# to start. Now both launch in parallel so Railway can probe Streamlit
# as soon as it's up.

set -euo pipefail

PUBLIC_PORT="${PORT:-8501}"
API_PORT="${API_PORT:-8000}"

echo "[start.sh] Launching CostGuard..."
echo "[start.sh]   FastAPI   → http://localhost:${API_PORT} (internal)"
echo "[start.sh]   Streamlit → http://0.0.0.0:${PUBLIC_PORT} (public)"

export API_BASE_URL="http://localhost:${API_PORT}"

# ── 1. Start FastAPI in the background ──────────────────────────────────────
uvicorn backend.main:app \
  --host 127.0.0.1 \
  --port "${API_PORT}" \
  --log-level "${LOG_LEVEL:-info}" &
UVICORN_PID=$!
echo "[start.sh] uvicorn PID=${UVICORN_PID}"

# ── 2. Start Streamlit immediately (parallel to FastAPI) ─────────────────────
streamlit run frontend/app.py \
  --server.port "${PUBLIC_PORT}" \
  --server.address "0.0.0.0" \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false &
STREAMLIT_PID=$!
echo "[start.sh] streamlit PID=${STREAMLIT_PID}"

# ── 3. Cleanup handler — kill both on exit ───────────────────────────────────
cleanup() {
  echo "[start.sh] Shutting down..."
  kill "${UVICORN_PID}" "${STREAMLIT_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── 4. Wait for either process to exit ──────────────────────────────────────
wait -n "${UVICORN_PID}" "${STREAMLIT_PID}"
EXIT_CODE=$?
echo "[start.sh] A child process exited (code=${EXIT_CODE}). Shutting down."
exit "${EXIT_CODE}"
