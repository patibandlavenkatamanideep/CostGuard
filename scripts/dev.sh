#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# CostGuard — Local development startup script
# Usage: ./scripts/dev.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

# Check for .env
if [[ ! -f ".env" ]]; then
    echo "[WARNING] No .env file found. Copying .env.example → .env"
    cp .env.example .env
    echo "[ACTION REQUIRED] Edit .env and add your API keys, then rerun this script."
    exit 1
fi

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found. Install Python 3.11+."
    exit 1
fi

# Install dependencies if needed
if ! python3 -c "import fastapi" &>/dev/null; then
    echo "[INFO] Installing dependencies..."
    pip install -e ".[dev]"
fi

echo ""
echo "════════════════════════════════════════"
echo "  CostGuard — Starting Development Mode"
echo "════════════════════════════════════════"
echo ""

# Start FastAPI in background
echo "[1/2] Starting FastAPI backend on http://localhost:8000 ..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

sleep 2

# Start Streamlit
echo "[2/2] Starting Streamlit dashboard on http://localhost:8501 ..."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 &
UI_PID=$!

echo ""
echo "Services running:"
echo "  API:       http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "  Dashboard: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop."

# Wait and cleanup
trap "kill $API_PID $UI_PID 2>/dev/null; echo 'Stopped.'" EXIT INT TERM
wait
