# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder — install deps with cache
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies for pyarrow / pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir -e ".[dev]" || true

COPY . .
RUN pip install --no-cache-dir -e .

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Create upload temp directory
RUN mkdir -p /tmp/costguard

# Non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app /tmp/costguard
USER appuser

EXPOSE 8000 8501

# Health check for the API
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: start the API. Override CMD in docker-compose for Streamlit.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
