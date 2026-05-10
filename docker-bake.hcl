variable "IMAGE_REF" {
  default = "ghcr.io/chrisbennight/worker-vllm-runpod"
}

variable "TAG" {
  default = "latest"
}

# === vLLM core ===
variable "VLLM_VERSION" {
  default = "0.20.2"
}

# === Build provenance (populated by CI from env; empty for local builds) ===
variable "GIT_SHA" {
  default = ""
}
variable "BUILD_DATE" {
  default = ""
}

# === PyTorch / CUDA ===
# Default image: CUDA 12.8 (cu128 wheels). Covers Ampere/Hopper/Blackwell SM100
# (with FA4 attention). Matches the comfyui-base-runpod default.
variable "TORCH_INDEX_SUFFIX" {
  default = "cu128"
}
variable "CUDA_VERSION_DASH" {
  default = "12-8"
}
variable "CUDA_BASE_IMAGE" {
  default = "nvidia/cuda:12.8.1-base-ubuntu24.04"
}

# Blackwell (B200 / RTX 5090) image: CUDA 13.0 (cu130 wheels). Carries FP4
# block-scaled cuBLAS for NVFP4 quantisation paths.
variable "TORCH_INDEX_SUFFIX_CU130" {
  default = "cu130"
}
variable "CUDA_VERSION_DASH_CU130" {
  default = "13-0"
}
variable "CUDA_BASE_IMAGE_CU130" {
  default = "nvidia/cuda:13.0.0-base-ubuntu24.04"
}

# === Optional model bake ===
# Single-model bake (legacy upstream path)
variable "MODEL_NAME" {
  default = ""
}
variable "MODEL_REVISION" {
  default = ""
}
variable "TOKENIZER_NAME" {
  default = ""
}
variable "TOKENIZER_REVISION" {
  default = ""
}
variable "QUANTIZATION" {
  default = ""
}

# Multi-model bake — JSON array, e.g.
#   '[{"model":"Qwen/Qwen3-VL-7B-Instruct"},{"model":"BAAI/bge-reranker-v2-m3"}]'
# Takes precedence over MODEL_NAME when set.
variable "MODELS_MANIFEST" {
  default = ""
}

# Pull in decord + opencv-python-headless for video multimodal workloads
# (Qwen3-VL with video=N>0). Off by default to keep the image slim for
# image-only and text-only deployments.
variable "INSTALL_VIDEO_EXTRAS" {
  default = "false"
}

# Storage location used at bake time. Default empty → start.sh auto-detects at
# runtime. Set to e.g. "/models" if you intend to bake the model into the image
# and don't want it shadowed by a network volume mounted at /runpod-volume.
variable "BASE_PATH" {
  default = ""
}

group "default" {
  targets = ["dev"]
}

# Common settings shared by all targets (defaults to CUDA 12.8 / cu128).
target "common" {
  context    = "."
  dockerfile = "Dockerfile"
  platforms  = ["linux/amd64"]
  labels = {
    "org.opencontainers.image.title"       = "worker-vllm-runpod"
    "org.opencontainers.image.description" = "Slim vLLM serverless worker for RunPod (Blackwell-first)"
    "org.opencontainers.image.source"      = "https://github.com/chrisbennight/worker-vllm-runpod"
    "org.opencontainers.image.url"         = "https://github.com/chrisbennight/worker-vllm-runpod"
    "org.opencontainers.image.licenses"    = "MIT"
    "org.opencontainers.image.version"     = TAG
    "org.opencontainers.image.revision"    = GIT_SHA
    "org.opencontainers.image.created"     = BUILD_DATE
  }
  args = {
    VLLM_VERSION       = VLLM_VERSION
    CUDA_BASE_IMAGE    = CUDA_BASE_IMAGE
    CUDA_VERSION_DASH  = CUDA_VERSION_DASH
    TORCH_INDEX_SUFFIX = TORCH_INDEX_SUFFIX
    MODEL_NAME         = MODEL_NAME
    MODEL_REVISION     = MODEL_REVISION
    TOKENIZER_NAME     = TOKENIZER_NAME
    TOKENIZER_REVISION = TOKENIZER_REVISION
    QUANTIZATION       = QUANTIZATION
    MODELS_MANIFEST      = MODELS_MANIFEST
    INSTALL_VIDEO_EXTRAS = INSTALL_VIDEO_EXTRAS
    BASE_PATH            = BASE_PATH
  }
}

# Production image, CUDA 12.8 (covers RTX 30/40, A100, H100, L40S, B200 with FA4).
target "cu128" {
  inherits = ["common"]
  tags = [
    "${IMAGE_REF}:${TAG}-cu128",
    "${IMAGE_REF}:cu128",
    "${IMAGE_REF}:latest",
  ]
}

# Production image, CUDA 13.0 (Blackwell B200 / RTX 5090 with FP4 paths).
target "cu130" {
  inherits = ["common"]
  tags = [
    "${IMAGE_REF}:${TAG}-cu130",
    "${IMAGE_REF}:cu130",
  ]
  args = {
    CUDA_BASE_IMAGE    = CUDA_BASE_IMAGE_CU130
    CUDA_VERSION_DASH  = CUDA_VERSION_DASH_CU130
    TORCH_INDEX_SUFFIX = TORCH_INDEX_SUFFIX_CU130
  }
}

# Local dev build (loaded into the local docker daemon, not pushed).
target "dev" {
  inherits = ["common"]
  tags     = ["${IMAGE_REF}:dev"]
  output   = ["type=docker"]
}

# CI-pushed dev tags (separate from :latest so manual testing doesn't override prod).
target "devpush-cu128" {
  inherits = ["common"]
  tags     = ["${IMAGE_REF}:dev-cu128"]
}

target "devpush-cu130" {
  inherits = ["cu130"]
  tags     = ["${IMAGE_REF}:dev-cu130"]
}
