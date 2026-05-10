# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — fork as `worker-vllm-runpod`

### Breaking Changes (from upstream `runpod-workers/worker-vllm`)

- **Image renamed** from `runpod/worker-v1-vllm` (Docker Hub) to `ghcr.io/chrisbennight/worker-vllm-runpod` (GHCR). No Docker Hub credentials needed; uses `GITHUB_TOKEN`.
- **New tag scheme**: `:cu128`, `:cu130`, `:vX.Y.Z-cu128`, `:vX.Y.Z-cu130`, `:latest` (alias for `:cu128`). The old `runpod/worker-v1-vllm:vX.Y.Z` Docker Hub tags are not produced.
- **CUDA-variant matrix introduced**. The default `:cu128` image works on Ampere/Hopper/Blackwell SM100 (with FA4); `:cu130` is a Blackwell-only variant carrying CUDA 13.0 wheels for FP4 block-scaled cuBLAS. Mirrors the [`comfyui-base-runpod`](https://github.com/chrisbennight/comfyui-base-runpod) build matrix.
- **Multi-stage Dockerfile**. Builder stage installs Python 3.12 + uv, generates a hashed lockfile via `pip-compile --generate-hashes`, and installs vLLM. Runtime stage copies the installed packages and drops the toolchain.
- **`TRANSFORMERS_VERSION` runtime install hook removed**. Pin `transformers` in the lockfile instead. Runtime `pip install` was unsafe on serverless cold starts and inconsistent with `--require-hashes`.
- **Upstream Slack notification workflows removed** (`slack-pr-issue-notify.yml`, `slack-vllm-monitor.yml`) — RunPod-internal automation that doesn't apply to this fork.
- **HF cache layout standardised** to `<volume>/.cache/huggingface/{hub,datasets}` (matching the HuggingFace default and the `comfyui-base-runpod` layout). Old: `<BASE_PATH>/huggingface-cache/{hub,datasets}`. Both stacks can now share one Network Volume without duplicating model weights.
- **Volume mount auto-detection**. `start.sh` checks for `/runpod-volume` (serverless mount) first, then `/workspace` (pod mount), and exports `BASE_PATH` accordingly. The Dockerfile no longer hard-codes `BASE_PATH=/runpod-volume` at build time.
- **Tensorizer scaffolding removed** from `download_model.py` and `engine_args.py`. The TODO has been outstanding since the fork; not shipping dead code.

### Added

- **`/v1/embeddings`, `/v1/rerank`, `/rerank`, `/v2/rerank`, `/v1/score`, `/v1/classify`** endpoints. The same image now serves embedding, reranker, cross-encoder, and classification models — selected via `MODEL_TASK=generate|embed|score|classify` (default `generate` preserves prior behaviour).
- **Multi-model bake** via `MODELS_MANIFEST` build arg. Accepts a JSON list `[{"model": "...", "revision": "...", "quantization": "..."}, ...]` and pre-downloads each into the standard HF hub cache. Runtime `MODEL_NAME` selects which baked model to load.
- **Qwen3-VL turnkey defaults**. `LIMIT_MM_PER_PROMPT=image=4,video=0` is sane for image-only multimodal serving and is documented as the recommended preset; `decord` is available for video work behind a build arg.
- **Provenance OCI labels** on every published image (`org.opencontainers.image.{title,source,url,licenses,version,revision,created}`).
- **Tag verification gate** in `release.yml` — refuses to publish unless the requested tag exists, points at HEAD, and the working tree is clean.
- **`AGENTS.md`** with branch / PR / release policy.
- **`.github/pull_request_template.md`** required for every PR.
- **`CHANGELOG.md`** with this fork-divergence section.

### Pre-fork history

Below is the upstream changelog inherited from `runpod-workers/worker-vllm`. The full release history is at <https://github.com/runpod-workers/worker-vllm/releases>.

The fork branched from upstream commit `87d7365` (Merge PR #292, vLLM 0.19.1).
