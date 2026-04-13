#!/usr/bin/env bash
# CostGuard — single-container startup (Railway / Render)
#
# Railway injects $PORT for the public-facing port.
# Streamlit binds to $PORT.  FastAPI binds to a FIXED internal port (9000)
# so it can never clash with whatever value Railway assigns to $PORT.
#
# Startup order:
#   1. Python import check — fail fast with a useful error in deploy logs
#   2. FastAPI on 127.0.0.1:9000 in the background
#   3. exec Streamlit on $PORT — becomes the foreground process Railway monitors

set -euo pipefail

PUBLIC_PORT="${PORT:-8501}"
API_PORT="9000"          # fixed internal port; never equals $PORT
export API_BASE_URL="http://localhost:${API_PORT}"

echo "[start.sh] Python: $(python3 --version)"
echo "[start.sh] Streamlit → 0.0.0.0:${PUBLIC_PORT}  |  FastAPI → 127.0.0.1:${API_PORT}"

# ── 1. Fail-fast import check (output visible in Railway Deploy Logs) ─────────
echo "[start.sh] Checking imports..."
python3 -c "
from backend.config import get_settings
from backend.models import EvalMode, SessionKeys
from evaluation.engine import run_evaluation
print('[start.sh] All imports OK')
"

# ── 2. FastAPI in the background — Streamlit does NOT depend on it ────────────
uvicorn backend.main:app \
  --host 127.0.0.1 \
  --port "${API_PORT}" \
  --log-level "${LOG_LEVEL:-info}" &

# ── 3. Streamlit in the foreground — this is what Railway health-checks ───────
exec streamlit run frontend/app.py \
  --server.port "${PUBLIC_PORT}" \
  --server.address "0.0.0.0" \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
