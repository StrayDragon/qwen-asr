# Fork Notes

This repository is a fork of `antirez/qwen-asr`.

It tracks upstream `main` and adds an OpenAI-compatible HTTP server plus Docker packaging.

## Upstream updates (rebased in 2026-02)

Upstream commits pulled in this rebase:

- `d96fcb9` (2026-02-14): incremental stdin streaming for `--stdin --stream`
  - `--stdin --stream` no longer blocks until EOF; a reader thread fills a shared buffer and the streaming loop waits for new data.
  - Introduces `qwen_live_audio_t` and `qwen_transcribe_stream_live()` in `qwen_asr.h`.
- `9d533fe` (2026-02-14): harden long-run live streaming + text-level commit tweaks
- `06827dc` (2026-02-15): streaming emits token deltas (less monotonic reconciliation work)
- `b00b789` (2026-02-15): restore rollback/cached streaming path + sync docs/help
  - Adds overlap suppression and tail-repeat detection to reduce duplicate emissions in long streams.
  - Adds recovery + periodic reset hooks to keep encoder cache + text conditioning state stable over long runs.
  - Docs/`--help`: clarify `--stream --silent` non-interactive final refinement applies to file input; live stdin streaming remains chunked.

Files touched upstream include: `qwen_asr.c`, `qwen_asr_audio.c/h`, `qwen_asr.h`, `main.c`, `README.md`.

## Fork changes (on top of upstream)

Fork-only additions (high level):

- `openai-compact-server/`: FastAPI server implementing OpenAI-compatible `/v1/audio/transcriptions`
  - JSON + SSE streaming responses
  - Bearer-token auth
  - Model pool for concurrency
  - Lazy load + idle auto-unload to release memory when unused
- Docker packaging + CI:
  - `Dockerfile`, `docker-compose.yml`, `docker-compose.dev.yml`
  - `.github/workflows/docker-publish.yml` for GHCR builds
- Dev ergonomics:
  - `justfile` helpers (build shared lib, install deps, serve, smoke tests)

Notes:
- The server uses `libqwen_asr.so` built from this repo (see `just libqwen`).
- Current deployment defaults to the `0.6B` model in `openai-compact-server/config.py`.

## Rebase record

- 2026-02-17: rebased `main` onto `upstream/main` at `b00b789` (clean rebase, no conflicts).
- Fork feature commits re-applied on top (new SHAs after rebase):
  - `d3d6934` feat: add openai-compact server & docker images ship
  - `b86cbdd` conf: add ghcr version docker-compose
  - `61d830c` feat: openai-compact-server add auto unload model timeout

Because this was a rebase, `origin/main` will now diverge from local `main`.
To publish the rebased history, use `git push --force-with-lease origin main`.
If you want tags like `v0.0.2` to follow the rebased commits, they need to be moved/recreated.

## Verification (0.6B only)

CLI build + smoke:

```bash
make blas
./qwen_asr -d qwen3-asr-0.6b -i samples/jfk.wav --silent
cat samples/jfk.wav | ./qwen_asr -d qwen3-asr-0.6b --stdin --stream --silent
```

Focused regressions:

```bash
python asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-0.6b --segment-check-only
python asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-0.6b --stream-check-only
python asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-0.6b --stream-cache-check-only
```

OpenAI-compatible server (local):

```bash
XDG_RUNTIME_DIR=/tmp just libqwen
cd openai-compact-server
UV_CACHE_DIR=/tmp/uv-cache uv sync --frozen --no-managed-python --no-python-downloads
UV_CACHE_DIR=/tmp/uv-cache uv run python -m uvicorn main:app --host 127.0.0.1 --port 8011
```
