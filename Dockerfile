# Dockerfile for Qwen ASR OpenAI-Compatible API Server
# Multi-stage build using uv's official Python image

# Build argument to control China mirror usage
ARG USE_CN_MIRROR=false

# ============= Builder stage =============
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

ARG USE_CN_MIRROR

# Set working directory
WORKDIR /app

# Use Tsinghua mirror for faster apt downloads (if enabled)
RUN if [ "$USE_CN_MIRROR" = "true" ]; then \
      sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources || \
      sed -i 's|http://deb.debian.org|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list; \
    fi

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    make \
    libc-dev \
    libopenblas-dev \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy only necessary files for building libqwen
COPY Makefile *.h *.c /app/
COPY qwen_asr_safetensors.c /app/

# Copy Python project files for dependency installation
COPY openai-compact-server/pyproject.toml /app/openai-compact-server/

# Build shared library for Python bindings
RUN echo "Building libqwen_asr.so with -fPIC..." && \
    make clean && \
    make qwen_asr.o qwen_asr_kernels.o qwen_asr_kernels_generic.o qwen_asr_kernels_neon.o qwen_asr_kernels_avx.o qwen_asr_audio.o qwen_asr_encoder.o qwen_asr_decoder.o qwen_asr_tokenizer.o qwen_asr_safetensors.o \
    CFLAGS="-Wall -Wextra -O3 -march=native -ffast-math -fPIC -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas" \
    LDFLAGS="-lm -lpthread -lopenblas" && \
    gcc -shared -fPIC -o libqwen_asr.so \
    qwen_asr.o qwen_asr_kernels.o qwen_asr_kernels_generic.o qwen_asr_kernels_neon.o qwen_asr_kernels_avx.o \
    qwen_asr_audio.o qwen_asr_encoder.o qwen_asr_decoder.o qwen_asr_tokenizer.o qwen_asr_safetensors.o \
    -lm -lpthread -lopenblas && \
    make clean && \
    ls -la libqwen_asr.so

# Install Python dependencies using uv
# Use system Python (already provided by uv base image)
# Conditionally use China PyPI mirrors
RUN if [ "$USE_CN_MIRROR" = "true" ]; then \
      export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple && \
      export UV_EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple; \
    fi && \
    echo "Installing Python dependencies..." && \
    cd /app/openai-compact-server && \
    uv venv .venv && \
    uv sync --no-editable

# ============= Final stage =============
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ARG USE_CN_MIRROR

# Use Tsinghua mirror for faster apt downloads (if enabled)
RUN if [ "$USE_CN_MIRROR" = "true" ]; then \
      sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources || \
      sed -i 's|http://deb.debian.org|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list; \
    fi

# Install runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    ffmpeg \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copy shared library to /app (libqwen.py searches parent directory)
COPY --from=builder --chown=app:app /app/libqwen_asr.so /app/

# Copy server files
COPY openai-compact-server/*.py /app/openai-compact-server/
COPY openai-compact-server/pyproject.toml /app/openai-compact-server/

# Copy virtual environment from builder
COPY --from=builder --chown=app:app /app/openai-compact-server/.venv /app/openai-compact-server/.venv

# Create model directory (empty, will be mounted or downloaded)
RUN mkdir -p /app/qwen3-asr-0.6b

# Environment variables (venv is in openai-compact-server subdir)
ENV PATH="/app/openai-compact-server/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV QWEN_HOST=0.0.0.0
ENV QWEN_PORT=8011
ENV QWEN_MODEL_POOL_SIZE=2
ENV QWEN_API_TOKEN=sk-docker-key

# Expose API port
EXPOSE 8011

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8011/ || exit 1

# Set working directory
WORKDIR /app/openai-compact-server

# Run server
CMD ["python", "main.py"]
