<div align="center">

# worker-vllm-runpod (pod mode)

OpenAI-compatible vLLM image for **RunPod pods** — Blackwell-ready (CUDA 12.8 + 13.0), opinionated, slim.

</div>

> This is a [`runpod-workers/worker-vllm`](https://github.com/runpod-workers/worker-vllm) fork that pivots from **serverless** to **pod** deployment. It runs `vllm serve` directly and speaks the standard OpenAI-compatible HTTP API on port 8000 — no Runpod job-queue plumbing. If you want serverless, use [the upstream image](https://github.com/runpod-workers/worker-vllm).

Sibling project: [`chrisbennight/comfyui-base-runpod`](https://github.com/chrisbennight/comfyui-base-runpod). The two images share a cache layout so a single RunPod Network Volume can host model weights for both stacks.

## Images

| Tag                                                            | Audience                                                                                                        |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `ghcr.io/chrisbennight/worker-vllm-runpod:cu128`               | Default. Covers Ampere / Hopper / Blackwell SM100 with FA4 attention.                                            |
| `ghcr.io/chrisbennight/worker-vllm-runpod:cu130`               | Blackwell B200 / RTX 5090 / RTX PRO 6000 — carries CUDA 13.0 wheels (FP4 block-scaled cuBLAS).                  |
| `ghcr.io/chrisbennight/worker-vllm-runpod:latest`              | Alias for `:cu128`.                                                                                              |
| `ghcr.io/chrisbennight/worker-vllm-runpod:vX.Y.Z-cu{128,130}`  | Pinned releases. See [GitHub Releases](https://github.com/chrisbennight/worker-vllm-runpod/releases).            |

## Quick start (RunPod pod)

Spin up a pod with:

- **Image**: `ghcr.io/chrisbennight/worker-vllm-runpod:cu130` (or `:cu128` on non-Blackwell GPUs)
- **GPU**: whatever fits your model — e.g. `RTX PRO 6000 Blackwell Server Edition` (96 GB) for a 27B BF16 model at 256k context.
- **Container disk**: 50-100 GB.
- **Network volume**: mounted at `/workspace`. Highly recommended so model weights survive pod restarts.
- **Exposed ports**: TCP `8000` (vLLM). Optional `22` if you bake SSH in (the image doesn't ship SSH out of the box).
- **Env vars**: at minimum `MODEL_NAME`. See [docs/configuration.md](docs/configuration.md) for the full surface.

Once the pod is `RUNNING`, RunPod's HTTP proxy maps `https://<pod-id>-8000.proxy.runpod.net` → port 8000 inside the container. Hit it like any other OpenAI-compatible endpoint:

```bash
curl https://<pod-id>-8000.proxy.runpod.net/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen3.5-27B-uncensored-heretic",
      "messages": [{"role": "user", "content": "hello"}]
    }'
```

## Example: Qwen3.5-27B at 256k context on Blackwell

Env vars on the pod (covers the [`llmfan46/Qwen3.5-27B-uncensored-heretic`](https://huggingface.co/llmfan46/Qwen3.5-27B-uncensored-heretic) preset from the model card):

```bash
MODEL_NAME=llmfan46/Qwen3.5-27B-uncensored-heretic
MAX_MODEL_LEN=262144
TRUST_REMOTE_CODE=true
DTYPE=bfloat16
KV_CACHE_DTYPE=fp8                   # halves KV cache so 256k fits at 96 GB
GPU_MEMORY_UTILIZATION=0.92
MAX_NUM_SEQS=8                       # cap concurrent sequences for long context
REASONING_PARSER=qwen3
TOOL_CALL_PARSER=hermes
ENABLE_AUTO_TOOL_CHOICE=true
HF_HUB_ENABLE_HF_TRANSFER=1          # faster first-time HF download
```

That's it — `start.sh` translates each variable into the equivalent `vllm serve` CLI flag and execs the server. The actual invocation is logged on boot.

## Building a custom image with a model baked in

Two modes — single-model or a JSON manifest. See [docs/configuration.md](docs/configuration.md#bake-time-model-fetch-model_name--models_manifest) for the full reference. Short version:

```bash
docker buildx bake -f docker-bake.hcl cu130 \
    --set "*.args.MODEL_NAME=Qwen/Qwen3.5-27B-uncensored-heretic" \
    --set "*.args.BASE_PATH=/models"
```

For gated/private models, pass `HF_TOKEN` as a BuildKit secret (never baked into a layer):

```bash
docker buildx bake -f docker-bake.hcl cu130 \
    --set "*.args.MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct" \
    --set "*.args.BASE_PATH=/models" \
    --set "*.secrets.HF_TOKEN=$HF_TOKEN"
```

Multi-model bake (JSON array via build arg):

```bash
docker buildx bake -f docker-bake.hcl cu130 \
    --set "*.args.BASE_PATH=/models" \
    --set "*.args.MODELS_MANIFEST=$(cat <<'JSON'
[
  {"model": "Qwen/Qwen3.5-27B-uncensored-heretic"},
  {"model": "BAAI/bge-large-en-v1.5"}
]
JSON
)"
```

At runtime, set `MODEL_NAME` to whichever baked model the pod should serve.

## Network volume layout

The image expects (and auto-detects) the volume to be mounted at `/workspace` (RunPod pod convention) or `/runpod-volume` (legacy/serverless). Underneath either:

```
<volume>/
├── ComfyUI/                  # only if you share the volume with comfyui-base-runpod
├── .cache/
│   ├── huggingface/
│   │   ├── hub/              # shared HF model cache (single source of truth across stacks)
│   │   └── datasets/
│   └── torch/                # shared torch cache
└── .cache/vllm/              # vLLM-private (torch.compile, CUDA graphs, LMCache disk tier)
```

vLLM-private caches stay under `.cache/vllm/` so they never collide with comfy state when the same volume hosts both.

## Configuration

Full env-var reference (server, model, parallelism, multimodal, LoRA, etc.) lives in [`docs/configuration.md`](docs/configuration.md). Anything not surfaced by name can still be passed via `VLLM_EXTRA_ARGS` — that env var is split on whitespace and appended to `vllm serve`.

## Contributing

See [`AGENTS.md`](AGENTS.md) for the branch / PR / release policy. Every change lands via a PR using [`.github/pull_request_template.md`](.github/pull_request_template.md). The `pr.yml` workflow builds both CUDA variants automatically on every push to a PR branch; both must pass for the PR to merge.

## License

MIT (inherited from upstream).
