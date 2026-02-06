# syntax=docker/dockerfile:1

# ============ BUILD STAGE ============
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (with cache mount for faster rebuilds)
COPY requirements.txt .
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Create appuser home directory structure for Prisma cache
RUN mkdir -p /home/appuser/.cache
ENV HOME="/home/appuser"

# Generate Prisma client (with HOME set so binaries go to /home/appuser/.cache)
COPY prisma ./prisma
RUN prisma generate

# Pre-download embedding models to avoid runtime network dependency
# Cache mount speeds up rebuilds - downloads go to cache, then copy to image
# Primary model: bge-large (1024 dims) — pgvector indexing, memory, chunk search
# Light model: bge-small (384 dims) — on-the-fly fallback path (~10x faster on CPU)
RUN --mount=type=cache,id=hf-cache,target=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HF_HOME=/tmp/hf_cache \
    python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-large-en-v1.5', device='cpu'); \
SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')" && \
    mkdir -p /home/appuser/.cache/huggingface && \
    cp -r /tmp/hf_cache/* /home/appuser/.cache/huggingface/


# ============ RUNTIME STAGE ============
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime dependencies required by Prisma's bundled Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy Prisma binaries cache from builder (already at /home/appuser/.cache)
COPY --from=builder /home/appuser/.cache /home/appuser/.cache
RUN chown -R appuser:appgroup /home/appuser

# Set HOME for appuser (must match build stage HOME)
ENV HOME="/home/appuser"

# Copy application code
COPY src ./src

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check - use /ready for full readiness verification
# start-period=120s accounts for embedding model preload + DB connection
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/ready', timeout=10); exit(0 if r.status_code == 200 else 1)" || exit 1

# Run the server with Gunicorn + Uvicorn workers for better concurrency
# Workers = 4 (optimized for 32GB RAM — each worker ~2GB with embedding model)
CMD ["gunicorn", "src.server:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-b", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30"]
