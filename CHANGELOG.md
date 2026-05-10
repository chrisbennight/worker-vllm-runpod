# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — pod-mode pivot

This branch rebases the fork as a **pod** image rather than a serverless worker.

### Removed (serverless surface)

- `src/handler.py` — `runpod.serverless.start({"handler": handler, ...})` entry point + `concurrency_modifier` wiring. Pods don't pull from the Runpod job queue.
- `src/engine.py` — `OpenAIvLLMEngine` wrapper that translated runpod-job-format ↔ vLLM serving classes. vLLM 0.20.2 ships its own OpenAI-compatible HTTP server with every route (chat/completion/responses/messages/embeddings/rerank/score/classify) built in.
- `src/engine_args.py` — env → `AsyncEngineArgs` translator. With `handler.py` gone there's no consumer; the new `src/start.sh` translates env vars to CLI flags directly.
- `src/utils.py` — `JobInput` (runpod job parser), `BatchSize` (growth-factor token batching to amortize runpod's per-message HTTP overhead), `DummyRequest` / `DummyState` (faked a fastapi `Request` for the serverless handler).
- `src/tokenizer.py` — chat-template wrapper, only used by `OpenAIvLLMEngine`. vLLM's own server handles chat templating.
- `src/constants.py` — `DEFAULT_BATCH_SIZE` / `DEFAULT_MAX_CONCURRENCY` / `DEFAULT_BATCH_SIZE_GROWTH_FACTOR` / `DEFAULT_MIN_BATCH_SIZE`. All runpod-handler batching defaults.
- `.runpod/hub.json` — RunPod Hub UI surface for serverless deployments only.
- `.runpod/` directory (Hub readme + serverless test config).
- `runpod==1.9.0` — the serverless SDK, dropped from `requirements.in`.
- `docs/conventions.md` — architecture notes that described the serverless dispatch layer.

### Added (pod mode)

- `src/start.sh` rewritten to build a `vllm serve` CLI from env vars and `exec` it. Covers the common vLLM knobs (model/tokenizer/revision, max-model-len, dtype/kv-cache-dtype, GPU memory utilization, parallelism, quantization, attention backend, reasoning/tool-call parsers, multimodal limit, LoRA, prefix caching, etc). `VLLM_EXTRA_ARGS` is an escape hatch for the long tail of vLLM CLI flags.
- Auto-derive `--max-num-batched-tokens` from `MAX_MODEL_LEN` when the user doesn't set it explicitly. Preserves the same long-context-prefill fix that the deleted `engine_args.get_engine_args()` used to provide.
- `EXPOSE 8000` in the Dockerfile; `HOST` / `PORT` defaults of `0.0.0.0` / `8000` overridable via env.
- Documentation rewritten throughout for pod usage.

## [Unreleased prior] — fork as `worker-vllm-runpod` (pre-pivot)

The original divergence from upstream `runpod-workers/worker-vllm` covered the
hygiene work the pod-mode rewrite is built on top of:

### Hygiene fork (kept from the pre-pivot work)

- **Image renamed** from `runpod/worker-v1-vllm` (Docker Hub) to `ghcr.io/chrisbennight/worker-vllm-runpod` (GHCR). No Docker Hub credentials needed; uses `GITHUB_TOKEN`.
- **Tag scheme**: `:cu128`, `:cu130`, `:vX.Y.Z-cu128`, `:vX.Y.Z-cu130`, `:latest` (alias for `:cu128`).
- **CUDA-variant matrix**. The default `:cu128` image works on Ampere/Hopper/Blackwell SM100 (with FA4); `:cu130` is a Blackwell-only variant carrying CUDA 13.0 wheels for FP4 block-scaled cuBLAS.
- **Multi-stage Dockerfile**. Builder stage installs Python 3.12 + uv, generates a hashed lockfile via `pip-compile --generate-hashes`, and installs vLLM. Runtime stage copies the installed packages and drops the toolchain.
- **HF cache layout standardised** to `<volume>/.cache/huggingface/{hub,datasets}` (matching the HuggingFace default and the `comfyui-base-runpod` layout). Both stacks can share one Network Volume without duplicating model weights.
- **Volume mount auto-detection**. `start.sh` checks for `/workspace` (pod mount) first, then `/runpod-volume` (legacy/serverless mount), and exports `BASE_PATH` accordingly.
- **Multi-model bake** via `MODELS_MANIFEST` build arg. Accepts a JSON list and pre-downloads each entry into the standard HF hub cache. Runtime `MODEL_NAME` selects which baked model to load.
- **Provenance OCI labels** on every published image.
- **Tag verification gate** in `release.yml` — refuses to publish unless the requested tag exists, points at HEAD, and the working tree is clean.
- **PR build gate** (`.github/workflows/pr.yml`) — every push to a PR branch builds both CUDA variants in parallel and pushes them as `:pr-<num>-cu128` / `:pr-<num>-cu130`. `.github/workflows/pr-cleanup.yml` removes them from GHCR when the PR closes.
- **`AGENTS.md`** with branch / PR / release policy.
- **`.github/pull_request_template.md`** required for every PR.

### Pre-fork history

The fork branched from upstream commit `87d7365` (Merge PR #292, vLLM 0.19.1). The upstream release history is at <https://github.com/runpod-workers/worker-vllm/releases>.
