# Justfile for qwen-asr fork - custom build commands
# This file contains custom commands not in upstream Makefile
# To use: install just (https://github.com/casey/just)

# Default: list available commands
default:
    @just --list

# Build shared library for Python bindings (OpenAI-compatible server)
libqwen:
    #!/bin/bash
    set -e
    echo "Building libqwen_asr.so with -fPIC..."
    make clean
    make qwen_asr.o qwen_asr_kernels.o qwen_asr_kernels_generic.o qwen_asr_kernels_neon.o qwen_asr_kernels_avx.o qwen_asr_audio.o qwen_asr_encoder.o qwen_asr_decoder.o qwen_asr_tokenizer.o qwen_asr_safetensors.o CFLAGS="-Wall -Wextra -O3 -march=native -ffast-math -fPIC -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas" LDFLAGS="-lm -lpthread -lopenblas"
    gcc -shared -fPIC -o libqwen_asr.so qwen_asr.o qwen_asr_kernels.o qwen_asr_kernels_generic.o qwen_asr_kernels_neon.o qwen_asr_kernels_avx.o qwen_asr_audio.o qwen_asr_encoder.o qwen_asr_decoder.o qwen_asr_tokenizer.o qwen_asr_safetensors.o -lm -lpthread -lopenblas
    make clean
    echo "✓ Built libqwen_asr.so"
    ls -la libqwen_asr.so

# Install Python dependencies for the OpenAI server
install-deps:
    cd openai-compact-server && uv sync

# Start the OpenAI-compatible API server (with custom host/port)
serve HOST="0.0.0.0" PORT="8000":
    cd openai-compact-server && QWEN_HOST="{{HOST}}" QWEN_PORT="{{PORT}}" uv run python main.py

# Run a test transcription
test-transcribe audio_file="samples/jfk.wav":
    cd openai-compact-server && uv run python tests/test_client.py "{{audio_file}}"

# Run a streaming test
test-stream audio_file="samples/jfk.wav":
    cd openai-compact-server && uv run python tests/test_client.py --stream "{{audio_file}}"

# List available models
test-models:
    cd openai-compact-server && uv run python tests/test_client.py --models

# Quick test with curl
test-curl TOKEN="sk-test-key" MODEL="qwen-asr-0.6b" audio_file="samples/jfk.wav":
    curl -X POST http://localhost:8000/v1/audio/transcriptions \
      -H "Authorization: Bearer {{TOKEN}}" \
      -F "file=@{{audio_file}}" \
      -F "model={{MODEL}}"

# Development workflow: build lib + install deps + start server
dev:
    just libqwen
    just install-deps
    just serve
