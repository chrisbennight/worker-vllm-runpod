# ============================================================================
# Stage 1: Builder — generate hashed lockfile, install all Python packages.
# ============================================================================
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-base-ubuntu24.04

FROM ${CUDA_BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive

ARG VLLM_VERSION=0.20.2
ARG CUDA_VERSION_DASH=12-8
ARG TORCH_INDEX_SUFFIX=cu128

# Install Python 3.12 + build essentials + CUDA build packages (only in
# builder so we don't pay the size in the runtime layer).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        build-essential \
        ca-certificates \
        curl \
        wget \
        gnupg \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-minimal-build-${CUDA_VERSION_DASH} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-keyring_1.1-1_all.deb \
    && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

# Bootstrap pip + pip-tools for hashed lockfile generation.
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.12 /tmp/get-pip.py && \
    python3.12 -m pip install --no-cache-dir pip-tools && \
    rm /tmp/get-pip.py

# Set CUDA env for any wheels that need to compile.
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# Build the requirements.in we feed to pip-compile. We append vLLM here (rather
# than committing it in builder/requirements.in) so the lockfile resolves
# against the right CUDA-tagged PyTorch wheel for this build target.
COPY builder/requirements.in /tmp/build/requirements.in
WORKDIR /tmp/build
RUN { cat requirements.in; \
      echo ""; \
      echo "vllm[flashinfer]==${VLLM_VERSION}"; \
    } > requirements.full.in && \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/${TORCH_INDEX_SUFFIX}" \
    pip-compile --generate-hashes \
        --output-file=requirements.lock \
        --strip-extras \
        --allow-unsafe \
        requirements.full.in && \
    python3.12 -m pip install \
        --no-cache-dir \
        --ignore-installed \
        --require-hashes \
        --index-url https://pypi.org/simple \
        --extra-index-url "https://download.pytorch.org/whl/${TORCH_INDEX_SUFFIX}" \
        -r requirements.lock

# ============================================================================
# Stage 2: Runtime — slim image with the installed packages copied over.
# ============================================================================
FROM ${CUDA_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ARG CUDA_VERSION_DASH=12-8

# Runtime deps only. No build-essentials, no CUDA toolkit (just the base
# runtime that comes with the CUDA base image).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        ca-certificates \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

# Install python alternatives so `python3` and `python` resolve to 3.12.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Copy installed Python packages and entry-point scripts from the builder.
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Allow the container to start on hosts whose driver is just below the toolkit
# minor version (matches comfyui-base-runpod's posture). Also load CUDA compat
# libs if present.
ENV NVIDIA_REQUIRE_CUDA=""
ENV NVIDIA_DISABLE_REQUIRE=true
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
RUN if [ -d /usr/local/cuda/compat ]; then ldconfig /usr/local/cuda/compat; fi

# vLLM-specific runtime tunables.
ENV PYTHONPATH=/:/vllm-workspace \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4

# Bake-time model fetch args. BASE_PATH is intentionally empty so start.sh
# auto-detects the volume mount at runtime; users baking a model into the image
# should override with e.g. BASE_PATH=/models so the bake isn't shadowed by a
# network volume mounted at /runpod-volume.
ARG MODEL_NAME=""
ARG MODEL_REVISION=""
ARG TOKENIZER_NAME=""
ARG TOKENIZER_REVISION=""
ARG QUANTIZATION=""
ARG MODELS_MANIFEST=""
ARG BASE_PATH=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    QUANTIZATION=$QUANTIZATION \
    MODELS_MANIFEST=$MODELS_MANIFEST \
    BASE_PATH=$BASE_PATH

COPY src /src
RUN chmod +x /src/start.sh

# Optional bake-time download. Honours either MODEL_NAME (single) or
# MODELS_MANIFEST (JSON array). Uses an explicit BAKE_HF_HOME so the cache
# lands in the image filesystem rather than wherever HF_HOME might be set
# at build time. HF_TOKEN is read from a BuildKit secret and never baked.
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    BAKE_BASE_PATH="${BASE_PATH:-/runpod-volume}" && \
    export HF_HOME="${BAKE_BASE_PATH}/.cache/huggingface" && \
    export HF_HUB_CACHE="${HF_HOME}/hub" && \
    export HF_DATASETS_CACHE="${HF_HOME}/datasets" && \
    if [ -n "$MODELS_MANIFEST" ] || [ -n "$MODEL_NAME" ]; then \
        mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" && \
        python3 /src/download_model.py; \
    fi

CMD ["/bin/bash", "/src/start.sh"]
