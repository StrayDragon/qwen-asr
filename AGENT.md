This file is the practical guide for agents working on this repository.
It is intentionally implementation-oriented: what to change, where, how to test,
and which behaviors are considered contractually stable.

## Project Scope

Pure C inference engine for Qwen3-ASR speech-to-text models:
- `Qwen3-ASR-0.6B`
- `Qwen3-ASR-1.7B`

Primary target is CPU inference (BLAS + architecture-specific SIMD paths).

## Source Of Truth

When docs and code disagree, trust these files first:
- CLI behavior and options: `main.c`
- Public API and runtime state: `qwen_asr.h`
- Offline + segmented + streaming orchestration: `qwen_asr.c`
- Encoder math + load path: `qwen_asr_encoder.c`
- Decoder math + KV cache path: `qwen_asr_decoder.c`
- Kernel dispatch and hot loops: `qwen_asr_kernels*.c`, `qwen_asr_kernels_impl.h`
- Test harness: `asr_regression.py`
- Build targets: `Makefile`

Architecture/background references:
- `MODEL.md`
- `MODEL_CARD_OFFICIAL.md`

## Supported Runtime Modes

- Offline full-context (default): `-S 0`
- Offline segmented: `-S <secs>`
- Streaming: `--stream`
- Input from file: `-i file.wav`
- Input from stdin: `--stdin` (WAV or raw s16le 16k mono)

## User-Facing Behavior Contract (Do Not Break)

- `--silent` must still print transcription to stdout.
- `--silent` suppresses status/debug noise (stderr), not the text output.
- Without `--debug`, stderr should be concise:
  - model loading info
  - final inference summary lines
- `--debug` enables verbose internal diagnostics.
- `--language` is the only language forcing flag (no `--force-language`).
- `--past-text` accepted values are exactly `yes|no|auto`.
- `--past-text auto` means:
  - `yes` for `--stream`
  - `no` for non-stream modes

## Model + Inference Facts

- Model variant is auto-detected from weights (0.6B vs 1.7B).
- Encoder uses per-chunk Conv2D + windowed attention.
- Decoder uses causal Qwen3 with KV cache and prefill reuse.
- Encoder weights are loaded as f32 (converted at load where needed).
- Decoder large weights are bf16 mmapped and consumed via bf16 kernels.

## Important Defaults

From `qwen_load()` and CLI:
- Segment mode default: `-S 0` (full-audio decode)
- Segment cut search window: `-W 3.0`
- Stream chunk: `2.0s`
- Stream rollback: `5` tokens
- Stream unfixed chunks: `2`
- Stream max new tokens/chunk: `32`
- Encoder infer attention window: `8s` (`--enc-window-sec` in `[1,8]`)

## Repository Map

- `main.c`
  - CLI parsing, defaults, reporting, callback wiring
- `qwen_asr.c`
  - high-level transcription flows
  - segmented logic + optional past-text cleanup path
  - streaming chunk loop, encoder-window cache, rollback commit logic
- `qwen_asr_encoder.c`
  - audio tower load + forward
- `qwen_asr_decoder.c`
  - decoder load + prefill + token step + KV cache
- `qwen_asr_audio.c`
  - WAV/stdin decoding, resampling, mel prep helpers
- `qwen_asr_tokenizer.c`
  - tokenizer encode/decode
- `qwen_asr_safetensors.c`
  - safetensors loading and mmap
- `qwen_asr_kernels.c`
  - common math, threading, BLAS paths
- `qwen_asr_kernels_generic.c`
  - generic hot kernels
- `qwen_asr_kernels_neon.c`
  - ARM NEON hot kernels
- `qwen_asr_kernels_avx.c`
  - x86 AVX hot kernels
- `qwen_asr_kernels_impl.h`
  - architecture dispatch macros
- `asr_regression.py`
  - quality + focused regression checks
- `download_model.sh`
  - interactive small/large model downloader (**user-run only**; agents must not download weights automatically)

## Model Files Policy (Do Not Auto-Download)

Model weights are large (GBs) and may be subject to network / licensing / access constraints.
To keep CI and agent runs reproducible + safe, agents must **never** download weights automatically.

### Hard rules (agents)

- Do **not** run `download_model.sh`, `curl`, `wget`, `huggingface-cli`, `git lfs pull`, or any other command that fetches model weights.
- Do **not** “helpfully” fix missing models by downloading them.
- If a build/test/run requires a model that is missing locally:
  1) clearly report which model dir is missing or incomplete
  2) provide the exact user-run command(s) to obtain it
  3) stop and wait (do not proceed with weight downloads)

Suggested report template (copy/paste to users):
```text
Model weights are missing or incomplete: qwen3-asr-____/
Please download the model manually (I won't fetch weights automatically):

  ./download_model.sh --model small   # 0.6B -> ./qwen3-asr-0.6b
  ./download_model.sh --model large   # 1.7B -> ./qwen3-asr-1.7b

Then verify locally:

  test -f qwen3-asr-0.6b/model.safetensors && echo OK
```

### What “model present” means (easy checklist)

This repo expects model directories under the repo root:

- **0.6B**: `qwen3-asr-0.6b/`
- **1.7B**: `qwen3-asr-1.7b/`

Minimal file set (what `download_model.sh` fetches):

- 0.6B:
  - `qwen3-asr-0.6b/model.safetensors`
  - `qwen3-asr-0.6b/config.json`
  - `qwen3-asr-0.6b/generation_config.json`
  - `qwen3-asr-0.6b/vocab.json`
  - `qwen3-asr-0.6b/merges.txt`
- 1.7B:
  - `qwen3-asr-1.7b/model.safetensors.index.json`
  - `qwen3-asr-1.7b/model-00001-of-00002.safetensors`
  - `qwen3-asr-1.7b/model-00002-of-00002.safetensors`
  - `qwen3-asr-1.7b/config.json`
  - `qwen3-asr-1.7b/generation_config.json`
  - `qwen3-asr-1.7b/vocab.json`
  - `qwen3-asr-1.7b/merges.txt`

Quick verify (no downloads, just local checks):
```bash
ls -lh qwen3-asr-0.6b/ | head
ls -lh qwen3-asr-1.7b/ | head
test -f qwen3-asr-0.6b/model.safetensors && echo "0.6B OK" || echo "0.6B MISSING"
test -f qwen3-asr-1.7b/model.safetensors.index.json && echo "1.7B OK" || echo "1.7B MISSING"
```

### Common “model missing” symptoms (and how to recognize them)

If the directory exists but is empty/incomplete, the CLI typically errors like:
```text
multi_safetensors_open: no safetensors files in qwen3-asr-1.7b
qwen_load: cannot open safetensors in qwen3-asr-1.7b
Failed to load model from qwen3-asr-1.7b
```

This is not a code bug: it means weights are not present locally.

### User-run examples (manual model download)

Agents may only **suggest** these commands; users must run them themselves:

```bash
# Download 0.6B (small) into ./qwen3-asr-0.6b
./download_model.sh --model small

# Download 1.7B (large) into ./qwen3-asr-1.7b
./download_model.sh --model large

# Download into a custom directory
./download_model.sh --model small --dir /path/to/qwen3-asr-0.6b
```

## Build + Run

Build:
```bash
make blas
```

Smoke run (offline/file input, 0.6B):
```bash
./qwen_asr -d qwen3-asr-0.6b -i samples/jfk.wav --silent

# Expected: a single transcript line on stdout, starting with:
# "And so, my fellow Americans, ask not what your country can do for you; ..."
```

Stdin path (offline/stdin input, 0.6B):
```bash
cat samples/jfk.wav | ./qwen_asr -d qwen3-asr-0.6b --stdin --silent
```

Stdin streaming (0.6B):
```bash
cat samples/jfk.wav | ./qwen_asr -d qwen3-asr-0.6b --stdin --stream --silent

# Expected: same final transcript as the offline command above.
```

Beginner copy/paste (0.6B end-to-end sanity):
```bash
set -e

# 0) Confirm model files exist (no downloads)
test -f qwen3-asr-0.6b/model.safetensors || { echo "Missing: qwen3-asr-0.6b/model.safetensors"; exit 1; }

# 1) Build
make blas

# 2) Offline smoke (should print the JFK quote)
./qwen_asr -d qwen3-asr-0.6b -i samples/jfk.wav --silent

# 3) Stdin streaming smoke (should match the quote above)
cat samples/jfk.wav | ./qwen_asr -d qwen3-asr-0.6b --stdin --stream --silent
```

## Regression Workflow

Primary suite:
```bash
make test
# equivalent to:
./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-1.7b
```

Beginner-friendly: 0.6B-only verification (no 1.7B needed):
```bash
# Focused checks (fast, stable, easiest to interpret)
./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-0.6b --segment-check-only
./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-0.6b --stream-check-only
./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-0.6b --stream-cache-check-only

# Optional: full sample regression on 0.6B (may be stricter/noisier than 1.7B)
./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-0.6b
```

What “success” looks like:
- Focused checks end with: `Focused regression checks PASSED`
- Full regression ends with: `Regression PASSED: ... samples within threshold`

Focused checks:
```bash
./asr_regression.py --segment-check-only --binary ./qwen_asr --model-dir qwen3-asr-1.7b
./asr_regression.py --stream-check-only --binary ./qwen_asr --model-dir qwen3-asr-1.7b
./asr_regression.py --stream-cache-check-only --binary ./qwen_asr --stream-cache-model-dir qwen3-asr-0.6b
```

Notes:
- Quality regression only runs on WAVs that already have sibling `.txt` refs.
- `make test` includes stream-cache equivalence check by default.
- This means both model dirs are typically required:
  - main model (`--model-dir`, default `qwen3-asr-1.7b`)
  - stream-cache model (`--stream-cache-model-dir`, default `qwen3-asr-0.6b`)
- If `make test` fails with “no safetensors files”, the model directory is missing/incomplete. See: **Model Files Policy (Do Not Auto-Download)**.

How to debug a single failing sample:
1. Re-run the reported WAV directly (to see raw transcript):
   - `./qwen_asr -d qwen3-asr-1.7b -i samples/path/to/sample.wav --silent`
2. Compare with its reference file next to it:
   - `cat samples/path/to/sample.txt`
3. If you want the regression harness on a smaller subset, point `--samples-root` at a subfolder:
   - `./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-1.7b --samples-root samples/night_of_the_living_dead_1968`
4. If you want to isolate only the quality regression loop (skip focused checks):
   - `./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-1.7b --samples-root samples/night_of_the_living_dead_1968 --skip-segment-check --skip-stream-check --skip-stream-cache-check`
5. If you need extra CLI flags, forward them via `--arg` (repeatable):
   - example: force English + 4 threads
     - `./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-1.7b --arg --language --arg en --arg -t --arg 4`

Reference management:
```bash
./asr_regression.py --generate-missing --binary ./qwen_asr --model-dir qwen3-asr-1.7b
./asr_regression.py --refresh-refs --binary ./qwen_asr --model-dir qwen3-asr-1.7b
```

## Streaming Implementation Notes

Current streaming behavior in `qwen_transcribe_stream()`:
- Chunk-by-chunk audio growth (default 2s)
- Encoder cache for completed local-attention windows
- Re-encode only current partial tail window
- Decoder prefill reuse by longest unchanged embedding prefix
- Prefix rollback policy for token stability
- Monotonic commit frontier (no retracting already-emitted text)

Debug/env switch:
- `QWEN_STREAM_NO_ENC_CACHE=1` disables encoder window cache (debug/regression only)

Important caveat:
- In streaming mode, if no token callback is installed (for example CLI `--silent`),
  the code uses direct final refinement instead of interactive chunk emission.
  This path is not representative of interactive stream throughput.

## Segmented Mode Notes

When `-S > 0`:
- split points are chosen near low-energy regions inside `-W`
- default emission is token-by-token ASAP

When `--past-text yes` in segmented mode:
- boundary cleanup/post-processing path is enabled
- output is buffered per segment before emission
- collapse guardrails can retry segments unconditioned and disable conditioning after repeated collapses

## Performance Reporting Contract

Final stderr summary line format is:
```text
Inference: <ms> ms, <tokens> text tokens (<tok/s> tok/s, encoding: <ms>ms, decoding: <ms>ms)
Audio: <audio_s> s processed in <infer_s> s (<x>x realtime)
```

`encoding` = mel + encoder time
`decoding` = decoder prefill + autoregressive decode

## Kernel/Optimization Rules

- Architecture dispatch is centralized in `qwen_asr_kernels_impl.h`.
- Keep generic/NEON/AVX variants functionally equivalent.
- If you optimize one path, verify no regression on others.
- Favor meaningful speedups; avoid complexity for tiny wins.

## Change Checklist For Agents

Before editing:
1. Identify behavioral contract impacted (CLI, output, speed, quality, memory).
2. Read corresponding source-of-truth file(s).

After editing:
1. Build: `make blas`
2. Run focused sanity command(s) for changed area.
3. Run regression:
   - at minimum relevant focused checks
   - ideally full `make test` for non-trivial changes
4. Update `README.md` if CLI/runtime behavior changed.
5. Keep `AGENT.md` aligned if workflow/test defaults changed.

## Local-Only Artifacts (Do Not Depend On In Commits)

Common local directories/files are intentionally ignored:
- `qwen3-asr-0.6b/`, `qwen3-asr-1.7b/`
- `Qwen3-ASR/`
- `samples/extra/`
- `TODO.md`
- virtualenv folders

Do not make code rely on these being present unless guarded by checks.
