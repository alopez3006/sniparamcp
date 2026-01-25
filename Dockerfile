# syntax=docker/dockerfile:1

FROM python:3.12-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Generate Prisma client - force rebuild v1.7.1 with OAuth models
COPY prisma ./prisma
RUN prisma generate

# Copy application code
COPY src ./src

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Run the server
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
# Force rebuild Sun Jan 25 06:49:14 CET 2026
