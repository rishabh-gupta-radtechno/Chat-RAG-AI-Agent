# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    g++ \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# torch wheel index. CPU-only by default so the image builds and runs on any host.
# For a GPU image, override at build time with a CUDA index matching the host
# driver, e.g.:
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 .
# (docker-compose.gpu.yml sets this for you.)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ccache \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libpq5 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local

ENV PATH=/root/.local/bin:/usr/local/bin:$PATH

COPY . .

RUN mkdir -p static/uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health/')" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
