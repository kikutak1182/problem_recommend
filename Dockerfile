# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install only curl for health checks (gcc/g++ not needed for manylinux wheels)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip first
RUN pip install --upgrade pip

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install dependencies (no PyTorch needed for lightweight implementation)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (self-contained in app/ directory)
COPY app/ ./app/

# Create non-root user for security and set permissions
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

USER appuser

# Cloud Run will set PORT environment variable dynamically
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check for Cloud Run
HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Run the application (PORT will be set by Cloud Run) - JSON format to prevent signal issues
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
