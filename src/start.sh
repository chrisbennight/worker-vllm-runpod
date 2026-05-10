#!/bin/bash
set -e

# Detect the network volume mount and export BASE_PATH so the rest of the
# stack can derive cache paths from it. Pods mount the volume at /workspace,
# serverless workers at /runpod-volume — same volume, different mount points.
# A user-provided BASE_PATH always wins (e.g. "/models" for in-image bakes).
if [ -z "${BASE_PATH:-}" ]; then
    if [ -d /runpod-volume ]; then
        export BASE_PATH=/runpod-volume
    elif [ -d /workspace ]; then
        export BASE_PATH=/workspace
    else
        export BASE_PATH=/runpod-volume
        mkdir -p "$BASE_PATH"
    fi
fi

# Honour user-provided HF_* / TORCH_HOME, otherwise derive standard layout.
# Matches the HuggingFace default and comfyui-base-runpod, so a single network
# volume can host both stacks without duplicating model weights.
export HF_HOME="${HF_HOME:-${BASE_PATH}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${BASE_PATH}/.cache/torch}"
# vLLM-private artifacts (torch.compile cache, CUDA graph cache, LMCache disk
# tier). Kept under .cache/vllm/ so they never collide with comfy state.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${BASE_PATH}/.cache/vllm}"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT"

echo "[start] BASE_PATH=$BASE_PATH"
echo "[start] HF_HOME=$HF_HOME"
echo "[start] HF_HUB_CACHE=$HF_HUB_CACHE"
echo "[start] VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"

# List models present in the HF Hub cache so the operator can confirm what's
# actually available before vLLM tries to load MODEL_NAME. Useful when the
# image was built with MODELS_MANIFEST and you want to see which entries
# survived the bake (or what's already on the volume).
if [ -d "$HF_HUB_CACHE" ]; then
    cached=$(find "$HF_HUB_CACHE" -maxdepth 1 -type d -name "models--*" 2>/dev/null \
              | sed -E 's|.*models--||; s|--|/|' | sort)
    if [ -n "$cached" ]; then
        echo "[start] Cached models in HF_HUB_CACHE:"
        echo "$cached" | sed 's/^/  - /'
    fi
fi

exec python3 /src/handler.py
