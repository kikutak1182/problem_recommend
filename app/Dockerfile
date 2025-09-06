# Lightweight ONNX-based Sentence Embedding API
# Target size: 100-200MB (vs 1.7GB with PyTorch)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies and clean up in same layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY tokenizer/ ./tokenizer/
COPY model_int8.onnx .

# Optional: Copy FP32 model as fallback (comment out if not needed)
# COPY model_fp32.onnx .

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app
USER apiuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MODEL_PATH=model_int8.onnx
ENV TOKENIZER_PATH=tokenizer/
ENV MAX_LENGTH=256

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/healthz || exit 1

# Expose port (Cloud Run will set PORT environment variable)
EXPOSE 8080

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --log-level info