#!/bin/bash
set -e

# ----------------------------------------------------------------------------
# Volume mount + cache layout
# ----------------------------------------------------------------------------
# Detect the network volume mount and export BASE_PATH so the rest of the
# stack can derive cache paths from it. Pods mount the volume at /workspace,
# serverless workers at /runpod-volume — same volume, different mount points.
# A user-provided BASE_PATH always wins (e.g. "/models" for in-image bakes).
if [ -z "${BASE_PATH:-}" ]; then
    if [ -d /workspace ]; then
        export BASE_PATH=/workspace
    elif [ -d /runpod-volume ]; then
        export BASE_PATH=/runpod-volume
    else
        export BASE_PATH=/workspace
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

# flashinfer JIT-compiles attention kernels for SMs not covered by its
# pre-compiled cubin set (e.g. SM120 on RTX PRO 6000). Default cache is
# /root/.cache/flashinfer which is gone on container restart; redirect to
# the volume so a per-(model, dtype, head_dim) compile happens at most once.
export FLASHINFER_JIT_CACHE_DIR="${FLASHINFER_JIT_CACHE_DIR:-${BASE_PATH}/.cache/flashinfer}"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$FLASHINFER_JIT_CACHE_DIR"

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

# ----------------------------------------------------------------------------
# Build `vllm serve` CLI from env
# ----------------------------------------------------------------------------
if [ -z "${MODEL_NAME:-}" ]; then
    echo "[start] error: MODEL_NAME is required" >&2
    exit 1
fi

ARGS=( "$MODEL_NAME" )
ARGS+=( --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" )

# String/numeric flags — emit `--key value` when the env var is non-empty.
# Listed in roughly the order users most often tune.
_kv() {
    local env_name="$1" cli_flag="$2"
    local val="${!env_name:-}"
    if [ -n "$val" ]; then
        ARGS+=( "$cli_flag" "$val" )
    fi
}
_kv MODEL_REVISION              --revision
_kv TOKENIZER                   --tokenizer
_kv TOKENIZER_NAME              --tokenizer
_kv TOKENIZER_REVISION          --tokenizer-revision
_kv TOKENIZER_MODE              --tokenizer-mode
_kv SERVED_MODEL_NAME           --served-model-name
_kv MAX_MODEL_LEN               --max-model-len
_kv MAX_NUM_BATCHED_TOKENS      --max-num-batched-tokens
_kv MAX_NUM_SEQS                --max-num-seqs
_kv MAX_SEQ_LEN_TO_CAPTURE      --max-seq-len-to-capture
_kv DTYPE                       --dtype
_kv KV_CACHE_DTYPE              --kv-cache-dtype
_kv GPU_MEMORY_UTILIZATION      --gpu-memory-utilization
_kv SWAP_SPACE                  --swap-space
_kv CPU_OFFLOAD_GB              --cpu-offload-gb
_kv BLOCK_SIZE                  --block-size
_kv TENSOR_PARALLEL_SIZE        --tensor-parallel-size
_kv PIPELINE_PARALLEL_SIZE      --pipeline-parallel-size
_kv QUANTIZATION                --quantization
_kv LOAD_FORMAT                 --load-format
_kv DOWNLOAD_DIR                --download-dir
_kv DEVICE                      --device
_kv ATTENTION_BACKEND           --attention-backend
_kv REASONING_PARSER            --reasoning-parser
_kv TOOL_CALL_PARSER            --tool-call-parser
_kv CHAT_TEMPLATE               --chat-template
_kv CHAT_TEMPLATE_CONTENT_FORMAT --chat-template-content-format
_kv LIMIT_MM_PER_PROMPT         --limit-mm-per-prompt
_kv MAX_LORAS                   --max-loras
_kv MAX_LORA_RANK               --max-lora-rank
_kv LORA_MODULES                --lora-modules
_kv HF_OVERRIDES                --hf-overrides
_kv API_KEY                     --api-key
_kv SEED                        --seed

# Boolean flags — emit the flag iff the env var is "true" / "1" / "yes" / "on".
# Lowercase via `tr` so this also runs on macOS bash 3.2 during local dev.
_flag() {
    local env_name="$1" cli_flag="$2"
    local val
    val=$(printf '%s' "${!env_name:-}" | tr '[:upper:]' '[:lower:]')
    case "$val" in
        true|1|yes|on) ARGS+=( "$cli_flag" ) ;;
    esac
}
_flag TRUST_REMOTE_CODE         --trust-remote-code
_flag ENFORCE_EAGER             --enforce-eager
_flag ENABLE_PREFIX_CACHING     --enable-prefix-caching
_flag ENABLE_CHUNKED_PREFILL    --enable-chunked-prefill
_flag ENABLE_AUTO_TOOL_CHOICE   --enable-auto-tool-choice
_flag ENABLE_LORA               --enable-lora
_flag ENABLE_EXPERT_PARALLEL    --enable-expert-parallel
_flag DISABLE_LOG_STATS         --disable-log-stats
_flag DISABLE_LOG_REQUESTS      --disable-log-requests
_flag DISABLE_CUSTOM_ALL_REDUCE --disable-custom-all-reduce
_flag DISABLE_SLIDING_WINDOW    --disable-sliding-window

# Auto-derive max-num-batched-tokens from max-model-len when the user didn't
# set it explicitly. vLLM defaults to 2048 if absent, which is far too small
# for long-context serving (256k context → prefill in ~128 chunks otherwise).
if [ -z "${MAX_NUM_BATCHED_TOKENS:-}" ] && [ -n "${MAX_MODEL_LEN:-}" ]; then
    ARGS+=( --max-num-batched-tokens "$MAX_MODEL_LEN" )
    echo "[start] auto-derived --max-num-batched-tokens=$MAX_MODEL_LEN from MAX_MODEL_LEN"
fi

# Pass-through escape hatch for anything not surfaced above. Unquoted so the
# user-provided string is split into individual CLI words.
if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    ARGS+=( ${VLLM_EXTRA_ARGS} )
fi

echo "[start] exec: vllm serve ${ARGS[*]}"
exec vllm serve "${ARGS[@]}"
