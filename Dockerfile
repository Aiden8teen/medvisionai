# Multi-stage or lightweight Python 3.11 image
FROM python:3.11-slim

# Set environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Install system dependencies (OpenCV headless & basic tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application directories
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY medvisionai/models/ ./medvisionai/models/
COPY medvisionai/examples/ ./medvisionai/examples/
COPY braintumor/models/ ./braintumor/models/
COPY braintumor/sample_mris/ ./braintumor/sample_mris/
COPY braintumor/submissionwithnomodels/ ./braintumor/submissionwithnomodels/
COPY run_app.sh .

EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/api/health || exit 1

CMD ["./run_app.sh"]
