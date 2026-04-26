# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 and uv
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && apt-get clean

WORKDIR /app

# Install dependencies (CPU torch first; then patch with CUDA wheel)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ ./src/
COPY graphrag_project/ ./graphrag_project/
COPY config/ ./config/

# Download and install CUDA torch into venv site-packages
RUN mkdir -p /tmp/torch-cu121 && \
    curl -L -o /tmp/torch-cu121/torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl \
        https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp311-cp311-linux_x86_64.whl && \
    python3.11 -m pip install --no-deps --target=/app/.venv/Lib/site-packages \
        /tmp/torch-cu121/torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl && \
    rm -rf /tmp/torch-cu121

ENV PYTHONPATH=/app

# Default: run full pipeline (user can override with docker run <cmd>)
ENTRYPOINT ["uv", "run", "python", "src/preprocess.py"]