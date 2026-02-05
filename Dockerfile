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

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create appuser home directory structure for Prisma cache
RUN mkdir -p /home/appuser/.cache
ENV HOME="/home/appuser"

# Generate Prisma client (with HOME set so binaries go to /home/appuser/.cache)
COPY prisma ./prisma
RUN prisma generate

# Pre-download embedding model to avoid runtime network dependency
# Model is cached in /home/appuser/.cache/huggingface/
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5', device='cpu')"


# ============ RUNTIME STAGE ============
FROM python:3.12-slim AS runtime

WORKDIR /app

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

# Copy application code and scripts
COPY src ./src
COPY scripts ./scripts
COPY prisma ./prisma

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check - use /ready for full readiness verification
# start-period=120s accounts for embedding model preload + DB init
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/ready', timeout=10); exit(0 if r.status_code == 200 else 1)" || exit 1

# Run database initialization then start the server
# Workers = 4 (optimized for 32GB RAM — each worker ~2GB with embedding model)
CMD ["bash", "-c", "bash scripts/init-db.sh && gunicorn src.server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --timeout 120 --graceful-timeout 30"]
