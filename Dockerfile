# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1-cudnn-runtime-ubuntu22.04

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

# Install dependencies (CPU torch first; setup_cuda_torch.sh patches in CUDA wheel)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ ./src/
COPY graphrag_project/ ./graphrag_project/
COPY setup_cuda_torch.sh ./
COPY config/ ./config/

# Patch in CUDA torch
RUN bash setup_cuda_torch.sh

ENV PYTHONPATH=/app

# Default: run full pipeline (user can override with docker run <cmd>)
ENTRYPOINT ["uv", "run", "python", "src/preprocess.py"]