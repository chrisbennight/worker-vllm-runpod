# ============================================================================
# Stage 1: Builder — generate hashed lockfile, install all Python packages.
# ============================================================================
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-base-ubuntu24.04

FROM ${CUDA_BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive

ARG VLLM_VERSION=0.20.2
ARG CUDA_VERSION_DASH=12-8
ARG TORCH_INDEX_SUFFIX=cu128
# Pull in decord + opencv-python-headless when set, for video multimodal
# workloads (Qwen3-VL with video=N>0). Off by default to keep the image slim
# for image-only and text-only deployments.
ARG INSTALL_VIDEO_EXTRAS=false

# Install Python 3.12 + build essentials + CUDA build packages (only in
# builder so we don't pay the size in the runtime layer).
#
# Note: the nvidia/cuda:*-base-* image already configures the NVIDIA APT
# repo with /usr/share/keyrings/cuda-archive-keyring.gpg as Signed-By.
# Re-running `dpkg -i cuda-keyring_*.deb` writes a second source entry with
# a conflicting Signed-By value and apt refuses to read the source list
# ("Conflicting values set for option Signed-By"). Don't add it.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        build-essential \
        ca-certificates \
        curl \
        cuda-minimal-build-${CUDA_VERSION_DASH} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
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
      if [ "${INSTALL_VIDEO_EXTRAS}" = "true" ]; then \
          echo "decord"; \
          echo "opencv-python-headless"; \
      fi; \
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

# Runtime deps + a working C/C++/CUDA toolchain.
#
# Why build-essential at runtime: PyTorch's torch.compile/inductor backend
# triggers Triton's JIT, which compiles CUDA driver utilities on first use
# via a gcc invocation. Without a C compiler on PATH the engine crashes
# during profile_run with:
#   torch._inductor.exc.InductorError: Failed to find C compiler.
# The builder stage has build-essential, but multi-stage copy only grabs
# /usr/local/lib/python3.12 + /usr/local/bin, so gcc (in /usr/bin/) is
# stripped. Re-install it here. Adds ~300 MB.
#
# Why cuda-minimal-build-${CUDA_VERSION_DASH}: vLLM 0.20.2's flashinfer
# (0.6.8.post1) ships pre-compiled cubins for some SMs but JIT-compiles
# missing ones at first request — notably SM120 (RTX PRO 6000 / RTX 5090).
# The JIT shells out to /usr/local/cuda/bin/nvcc. Without it, startup
# crashes during cudagraph profiling with:
#   RuntimeError: Ninja build failed.
#     /bin/sh: 1: /usr/local/cuda/bin/nvcc: not found
# We install the toolchain that matches the *image's* CUDA version so
# /usr/local/cuda → /usr/local/cuda-${CUDA_VERSION_DOT} → has nvcc.
# Adds ~600 MB.
#
# Why cuda-cudart-12-8: vLLM 0.20.2 also ships a precompiled nixl_ep
# extension (NIXL all-to-all helper for MoE) that links against
# libcudart.so.12 *regardless* of which CUDA wheel suffix vLLM was
# installed with — it's a vLLM packaging assumption, not a base-image
# choice. On the cu130 base image (which only has libcudart.so.13),
# startup crashes:
#   ImportError: libcudart.so.12: cannot open shared object file
# On the cu128 base image this package is already present from the base,
# so re-declaring it is a no-op there. Kept hardcoded to 12-8 because
# that's the SONAME nixl_ep needs.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        build-essential \
        ca-certificates \
        curl \
        cuda-minimal-build-${CUDA_VERSION_DASH} \
        cuda-cudart-12-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

# Make the libcudart 12 shim discoverable by the dynamic linker. On cu128
# this just re-confirms /usr/local/cuda-12.8/lib64; on cu130 it adds the
# back-compat path alongside the cu130 libs.
RUN echo "/usr/local/cuda-12.8/lib64" > /etc/ld.so.conf.d/cuda-12-compat.conf && \
    ldconfig

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
ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false

# vLLM OpenAI-compatible HTTP server defaults — overridable via env on the pod.
ENV HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000

# Bake-time model fetch args. BASE_PATH is intentionally empty so start.sh
# auto-detects the volume mount at runtime; users baking a model into the image
# should override with e.g. BASE_PATH=/models so the bake isn't shadowed by a
# network volume mounted at /workspace.
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
# MODELS_MANIFEST (JSON array). Writes into the standard HF Hub cache rooted
# at BASE_PATH (default /workspace for pods). HF_TOKEN is read from a
# BuildKit secret and never baked into a layer.
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    BAKE_BASE_PATH="${BASE_PATH:-/workspace}" && \
    export HF_HOME="${BAKE_BASE_PATH}/.cache/huggingface" && \
    export HF_HUB_CACHE="${HF_HOME}/hub" && \
    export HF_DATASETS_CACHE="${HF_HOME}/datasets" && \
    if [ -n "$MODELS_MANIFEST" ] || [ -n "$MODEL_NAME" ]; then \
        mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" && \
        python3 /src/download_model.py; \
    fi

CMD ["/bin/bash", "/src/start.sh"]
