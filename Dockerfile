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

# Copy Prisma schema (needed by client at runtime)
COPY --from=builder /app/prisma ./prisma

# Copy application code
COPY src ./src

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Run the server with Gunicorn + Uvicorn workers for better concurrency
# Workers = 2 (optimized for 512MB RAM limit)
CMD ["gunicorn", "src.server:app", \
     "-w", "2", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-b", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30"]
# Build cache bust: Tue Jan 27 00:45:12 CET 2026
