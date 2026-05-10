# Configuration Reference

All behaviour is controlled through environment variables. `src/start.sh` reads them, builds a `vllm serve` CLI invocation, and `exec`s it. The set surfaced below is the curated common case; anything not surfaced can still be passed via `VLLM_EXTRA_ARGS` (see end of this file).

## Required

| Variable     | Description                                                                                                                                                  |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MODEL_NAME` | HuggingFace repo ID or local path. Resolved via `HF_HUB_CACHE` so a baked-in or volume-cached model loads without an HTTP round-trip. Refuses to start if unset. |

## Server (host / port)

| Variable | Default   | Description                                                                                                            |
| -------- | --------- | ---------------------------------------------------------------------------------------------------------------------- |
| `HOST`   | `0.0.0.0` | Bind address for the vLLM HTTP server.                                                                                 |
| `PORT`   | `8000`    | TCP port. `EXPOSE 8000` is set in the Dockerfile; if you change this, also update the pod template's exposed port set. |

## Model + tokenizer

| Variable             | vLLM CLI flag           | Description                                                                                                |
| -------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| `MODEL_REVISION`     | `--revision`            | Model git revision (branch/tag/SHA).                                                                       |
| `TOKENIZER`          | `--tokenizer`           | Tokenizer repo/path if it differs from the model's default.                                                |
| `TOKENIZER_NAME`     | `--tokenizer`           | Alias for `TOKENIZER`.                                                                                     |
| `TOKENIZER_REVISION` | `--tokenizer-revision`  | Tokenizer git revision.                                                                                    |
| `TOKENIZER_MODE`     | `--tokenizer-mode`      | `auto` (default) / `slow` / `mistral`.                                                                     |
| `SERVED_MODEL_NAME`  | `--served-model-name`   | Override the model name returned by `GET /v1/models` and accepted as the `model` parameter.                |
| `TRUST_REMOTE_CODE`  | `--trust-remote-code`   | Boolean. Required by some HF models (Qwen3.5, etc).                                                        |

## Memory + parallelism

| Variable                  | vLLM CLI flag                  | Description                                                                                                                                                 |
| ------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MAX_MODEL_LEN`           | `--max-model-len`              | Maximum context length the engine will serve. `start.sh` auto-derives `--max-num-batched-tokens=$MAX_MODEL_LEN` when the latter isn't set explicitly.       |
| `MAX_NUM_SEQS`            | `--max-num-seqs`               | Cap on concurrent sequences (≈ KV-cache budget divider). For 256k context on a 96 GB card, 4-8 is sane.                                                     |
| `MAX_NUM_BATCHED_TOKENS`  | `--max-num-batched-tokens`     | Prefill chunk size. Defaults to `MAX_MODEL_LEN` so prefill happens in one chunk.                                                                            |
| `MAX_SEQ_LEN_TO_CAPTURE`  | `--max-seq-len-to-capture`     | Above this sequence length CUDA graphs disable and the engine falls back to eager.                                                                          |
| `DTYPE`                   | `--dtype`                      | `bfloat16` / `float16` / `auto`. BF16 is the right default on Blackwell.                                                                                    |
| `KV_CACHE_DTYPE`          | `--kv-cache-dtype`             | `auto` / `fp8` / `fp8_e5m2`. For 256k context, `fp8` halves the KV footprint.                                                                               |
| `GPU_MEMORY_UTILIZATION`  | `--gpu-memory-utilization`     | Fraction of GPU memory vLLM reserves. `0.9-0.95` is normal; set lower if you need headroom for other CUDA processes.                                        |
| `SWAP_SPACE`              | `--swap-space`                 | CPU swap (GiB) per GPU.                                                                                                                                     |
| `CPU_OFFLOAD_GB`          | `--cpu-offload-gb`             | Bytes of model weights to offload to CPU RAM.                                                                                                               |
| `BLOCK_SIZE`              | `--block-size`                 | KV-cache block size (16 / 32 / 64).                                                                                                                         |
| `TENSOR_PARALLEL_SIZE`    | `--tensor-parallel-size`       | TP across GPUs (1 unless multi-GPU pod).                                                                                                                    |
| `PIPELINE_PARALLEL_SIZE`  | `--pipeline-parallel-size`     | Pipeline parallel stages.                                                                                                                                   |
| `ENFORCE_EAGER`           | `--enforce-eager`              | Disable CUDA graphs (debug knob).                                                                                                                           |
| `DEVICE`                  | `--device`                     | `cuda` / `cpu` / `auto`. Leave on `auto` unless you know better.                                                                                            |

## Performance knobs

| Variable                  | vLLM CLI flag                | Description                                                                                                                                  |
| ------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `ENABLE_PREFIX_CACHING`   | `--enable-prefix-caching`    | Boolean. Big win on multi-turn or RAG workloads.                                                                                              |
| `ENABLE_CHUNKED_PREFILL`  | `--enable-chunked-prefill`   | Boolean. Interleaves prefill with decode for better TTFT under load.                                                                          |
| `ENABLE_EXPERT_PARALLEL`  | `--enable-expert-parallel`   | Boolean. For MoE models (e.g. Qwen3-VL-30B-A3B).                                                                                              |
| `ATTENTION_BACKEND`       | `--attention-backend`        | Override vLLM's default. Defaults are sane (FA4 on SM100/SM103 Blackwell, FA3 on Hopper, FA2 elsewhere). Use to force `FLASH_ATTN` if FlashInfer has a head-size bug. |

## Quantization + loading

| Variable        | vLLM CLI flag       | Description                                                                                                       |
| --------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `QUANTIZATION`  | `--quantization`    | `awq` / `gptq` / `fp8` / `bitsandbytes` / `compressed-tensors` etc.                                               |
| `LOAD_FORMAT`   | `--load-format`     | `auto` / `safetensors` / `pt` / `bitsandbytes` etc.                                                               |
| `DOWNLOAD_DIR`  | `--download-dir`    | Where to download weights if not already cached. Honoured by HF Hub on first load.                                |

## Chat templating, reasoning, tool calls

| Variable                            | vLLM CLI flag                       | Description                                                                                                                                                                                                                |
| ----------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CHAT_TEMPLATE`                     | `--chat-template`                   | Jinja chat template string or path.                                                                                                                                                                                        |
| `CHAT_TEMPLATE_CONTENT_FORMAT`      | `--chat-template-content-format`    | `auto` / `string` / `openai`.                                                                                                                                                                                              |
| `REASONING_PARSER`                  | `--reasoning-parser`                | `deepseek_r1` / `qwen3` / `granite` / `hunyuan_a13b`. Enables reasoning-mode parsing (`<think>` block extraction etc).                                                                                                     |
| `TOOL_CALL_PARSER`                  | `--tool-call-parser`                | `hermes` / `mistral` / `llama3_json` / `llama4_pythonic` / `granite` / `deepseek_v3` / `internlm` / `pythonic` etc.                                                                                                        |
| `ENABLE_AUTO_TOOL_CHOICE`           | `--enable-auto-tool-choice`         | Boolean. Required alongside `TOOL_CALL_PARSER` for OpenAI-style function calling.                                                                                                                                          |

## Multimodal (Qwen3-VL preset)

| Variable               | vLLM CLI flag             | Description                                                                                                                                                                  |
| ---------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LIMIT_MM_PER_PROMPT`  | `--limit-mm-per-prompt`   | Per-modality limit. Highly recommended to set `image=N,video=0` for image-only Qwen3-VL serving — saves several GB of reserved VRAM from the video embedding reservation. |

Recommended preset for `Qwen/Qwen3-VL-7B-Instruct` (image-only):

```bash
MODEL_NAME=Qwen/Qwen3-VL-7B-Instruct
TRUST_REMOTE_CODE=true
LIMIT_MM_PER_PROMPT=image=4,video=0
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.9
```

For video work, build the image with `--build-arg INSTALL_VIDEO_EXTRAS=true` (or `INSTALL_VIDEO_EXTRAS=true` in the bake target). This pulls in `decord` and `opencv-python-headless`. The default image is image- and text-only to stay slim.

Open vLLM bug: `LIMIT_MM_PER_PROMPT` was reported as ineffective on some Qwen3-VL revisions ([vllm-project/vllm#38459](https://github.com/vllm-project/vllm/issues/38459)) — verify behaviour against your specific model revision before relying on it for capacity planning.

## LoRA adapters

| Variable        | vLLM CLI flag      | Description                                          |
| --------------- | ------------------ | ---------------------------------------------------- |
| `ENABLE_LORA`   | `--enable-lora`    | Boolean.                                             |
| `MAX_LORAS`     | `--max-loras`      | Concurrent LoRAs in a batch.                         |
| `MAX_LORA_RANK` | `--max-lora-rank`  | Maximum rank across registered LoRAs.                |
| `LORA_MODULES`  | `--lora-modules`   | vLLM's CLI form: `name1=path1 name2=path2 ...`.      |

## Logging + safety

| Variable                       | vLLM CLI flag                   | Description                                                                              |
| ------------------------------ | ------------------------------- | ---------------------------------------------------------------------------------------- |
| `DISABLE_LOG_STATS`            | `--disable-log-stats`           | Boolean. Suppress periodic stats lines.                                                  |
| `DISABLE_LOG_REQUESTS`         | `--disable-log-requests`        | Boolean. Suppress per-request logs.                                                      |
| `DISABLE_CUSTOM_ALL_REDUCE`    | `--disable-custom-all-reduce`   | Boolean. Falls back to NCCL all-reduce.                                                  |
| `DISABLE_SLIDING_WINDOW`       | `--disable-sliding-window`      | Boolean.                                                                                 |
| `API_KEY`                      | `--api-key`                     | Require clients to send `Authorization: Bearer <api_key>` to access the server.          |
| `SEED`                         | `--seed`                        | RNG seed.                                                                                |

## Escape hatch

| Variable           | Description                                                                                                                                                                              |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `VLLM_EXTRA_ARGS`  | Free-form pass-through. The string is split on whitespace and appended to the `vllm serve` invocation. Useful for niche flags not surfaced above (e.g. `--max-log-len 1000`, hf_overrides). |

vLLM has its own set of `VLLM_*` env vars (not surfaced here) that the binary reads directly — those still work as long as they're set in the pod's environment.

## Cache layout (set by `start.sh`)

`start.sh` auto-detects the volume mount and exports the standard HuggingFace + vLLM cache paths. User-provided values always win.

| Variable              | Default                              | Notes                                                                          |
| --------------------- | ------------------------------------ | ------------------------------------------------------------------------------ |
| `BASE_PATH`           | auto-detect `/workspace` or `/runpod-volume` | Root for all caches. Set to `/models` for in-image bakes that shouldn't be shadowed by a mounted volume. |
| `HF_HOME`             | `$BASE_PATH/.cache/huggingface`      | Parent of `hub/` and `datasets/`.                                              |
| `HF_HUB_CACHE`        | `$HF_HOME/hub`                       | Model weight cache. Matches the comfyui-base-runpod layout for volume sharing. |
| `HF_DATASETS_CACHE`   | `$HF_HOME/datasets`                  | Datasets cache.                                                                |
| `TORCH_HOME`          | `$BASE_PATH/.cache/torch`            | `torch.hub` and similar.                                                       |
| `VLLM_CACHE_ROOT`     | `$BASE_PATH/.cache/vllm`             | vLLM-private artifacts (torch.compile, CUDA graphs, LMCache disk tier).        |

## `HF_HUB_OFFLINE=1` for fully-baked images

If every model the pod will ever need is in the image (or pre-populated on the volume), set `HF_HUB_OFFLINE=1` (or `TRANSFORMERS_OFFLINE=1`) at runtime to skip every HuggingFace round-trip during boot. This removes one source of cold-start latency and prevents transient HF outages from delaying startup.

| Variable               | Description                                                                                                                       |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `HF_HUB_OFFLINE`       | Skip all HF Hub network calls; only resolve from cache. Set to `1` for fully-baked or pre-populated-volume deployments.           |
| `TRANSFORMERS_OFFLINE` | Skip all transformers HF calls. Pair with `HF_HUB_OFFLINE=1` for the strictest offline mode.                                      |

## Bake-time model fetch (`MODEL_NAME` + `MODELS_MANIFEST`)

Models can be downloaded at build time (faster cold starts, larger images) or at runtime (smaller images, slower first request). Two bake modes are supported.

### Single-model bake

Set `MODEL_NAME` (and optionally `MODEL_REVISION` / `TOKENIZER_NAME` / `TOKENIZER_REVISION` / `QUANTIZATION`) as Docker build args:

```bash
docker buildx bake -f docker-bake.hcl cu130 \
    --set "*.args.MODEL_NAME=Qwen/Qwen3-VL-7B-Instruct" \
    --set "*.args.BASE_PATH=/models"   # avoid /workspace so a network volume mount doesn't shadow the bake
```

For private/gated models, pass `HF_TOKEN` as a BuildKit secret:

```bash
docker buildx bake -f docker-bake.hcl cu130 \
    --set "*.args.MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct" \
    --set "*.args.BASE_PATH=/models" \
    --set "*.secrets.HF_TOKEN=$HF_TOKEN"
```

### Multi-model bake

Set `MODELS_MANIFEST` to a JSON array. Each entry can carry `model`, `revision`, `tokenizer`, `tokenizer_revision`, `quantization`. All entries are downloaded into the same HF cache; runtime `MODEL_NAME` selects which one to load:

```bash
docker buildx bake -f docker-bake.hcl cu130 \
    --set "*.args.BASE_PATH=/models" \
    --set "*.args.MODELS_MANIFEST=$(cat <<'JSON'
[
  {"model": "Qwen/Qwen3-VL-7B-Instruct"},
  {"model": "BAAI/bge-large-en-v1.5"},
  {"model": "BAAI/bge-reranker-v2-m3"}
]
JSON
)"
```

`MODELS_MANIFEST` takes precedence over `MODEL_NAME` when both are set.

### Verifying the bake at startup

`start.sh` lists every cached model in `HF_HUB_CACHE` at boot, so you can confirm what's actually available before the engine tries to load it:

```text
[start] Cached models in HF_HUB_CACHE:
  - Qwen/Qwen3-VL-7B-Instruct
  - BAAI/bge-large-en-v1.5
  - BAAI/bge-reranker-v2-m3
```

## Docker build arguments

These variables are used when building the image. They live in `docker-bake.hcl` as `variable` blocks and are passed to the Dockerfile via `ARG`:

| Variable              | Default                                | Type   | Description                                                                                              |
| --------------------- | -------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------- |
| `VLLM_VERSION`        | `0.20.2`                               | `str`  | vLLM version installed in the builder stage.                                                             |
| `TORCH_INDEX_SUFFIX`  | `cu128`                                | `str`  | PyTorch wheel index suffix (`cu128` default, `cu130` for the Blackwell variant).                         |
| `CUDA_VERSION_DASH`   | `12-8`                                 | `str`  | CUDA APT package suffix (`12-8` or `13-0`). Selects the `cuda-minimal-build-*` package in the builder.   |
| `CUDA_BASE_IMAGE`     | `nvidia/cuda:12.8.1-base-ubuntu24.04`  | `str`  | NVIDIA CUDA base image.                                                                                  |
| `INSTALL_VIDEO_EXTRAS`| `false`                                | `bool` | When `true`, appends `decord` + `opencv-python-headless` to the lockfile for video multimodal workloads. |
| `BASE_PATH`           | *(empty)*                              | `str`  | Bake-time cache root. Empty → `start.sh` auto-detects at runtime. Set to `/models` for in-image bakes.   |
| `MODEL_NAME`          | *(empty)*                              | `str`  | Single-model bake.                                                                                       |
| `MODELS_MANIFEST`     | *(empty)*                              | `JSON` | Multi-model bake (JSON array). Takes precedence over `MODEL_NAME`.                                       |
