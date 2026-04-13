#!/usr/bin/env bash
# CostGuard — single-container startup (Railway / Render)
#
# Railway/Render expose exactly ONE public port ($PORT).
# Strategy:
#   1. Start FastAPI (uvicorn) on localhost:8000 in the background.
#   2. Wait until FastAPI is healthy (fail hard if it never starts).
#   3. Start Streamlit on $PORT in the background.
#   4. Wait for either process to exit — exit with its code.

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

# ── 2. Wait until FastAPI is healthy (fail hard after 60 s) ──────────────────
echo "[start.sh] Waiting for FastAPI /health..."
READY=0
for i in $(seq 1 30); do
  if curl -sf "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
    echo "[start.sh] FastAPI healthy after ${i}s"
    READY=1
    break
  fi
  # Bail early if uvicorn already died
  if ! kill -0 "${UVICORN_PID}" 2>/dev/null; then
    echo "[start.sh] ERROR: uvicorn exited unexpectedly" >&2
    exit 1
  fi
  sleep 1
done

if [ "${READY}" -eq 0 ]; then
  echo "[start.sh] ERROR: FastAPI did not become healthy within 30 s" >&2
  kill "${UVICORN_PID}" 2>/dev/null || true
  exit 1
fi

# ── 3. Start Streamlit in the background ────────────────────────────────────
# NOTE: Do NOT use 'exec' here — that would orphan uvicorn in some container
# runtimes. Run both as background children of this script so they share the
# same process group and are cleaned up together.
streamlit run frontend/app.py \
  --server.port "${PUBLIC_PORT}" \
  --server.address "0.0.0.0" \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false &
STREAMLIT_PID=$!
echo "[start.sh] streamlit PID=${STREAMLIT_PID}"

# ── 4. Cleanup handler — kill both on exit ───────────────────────────────────
cleanup() {
  echo "[start.sh] Shutting down..."
  kill "${UVICORN_PID}" "${STREAMLIT_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── 5. Wait for either process to exit ──────────────────────────────────────
# Re-export PIDs so wait -n works (bash 4.3+ — Debian/Ubuntu default)
wait -n "${UVICORN_PID}" "${STREAMLIT_PID}"
EXIT_CODE=$?
echo "[start.sh] A child process exited (code=${EXIT_CODE}). Shutting down."
exit "${EXIT_CODE}"
