"""Optional bake-time HuggingFace model download.

Two modes:

1. Single-model (legacy upstream behaviour): set MODEL_NAME (and optionally
   MODEL_REVISION / TOKENIZER_NAME / TOKENIZER_REVISION / QUANTIZATION).
2. Multi-model manifest: set MODELS_MANIFEST to a JSON array, e.g.
       [{"model": "Qwen/Qwen3-VL-7B-Instruct"},
        {"model": "BAAI/bge-reranker-v2-m3", "revision": "main"}]
   Each entry may carry "model", "revision", "tokenizer", "tokenizer_revision",
   "quantization". When set, MODELS_MANIFEST takes precedence over MODEL_NAME.

Models land in the standard HF Hub cache at $HF_HUB_CACHE (set by the
Dockerfile). For single-model mode we also write /local_model_args.json so
`engine_args.get_local_args` can pick up the cached path; multi-model mode
skips that file because the runtime decides which model to load.
"""

import glob
import json
import logging
import os
import sys
from huggingface_hub import snapshot_download
from utils import timer_decorator

# Patterns we accept for "the model is downloaded". Each tuple is OR-ed within;
# the first one that matches wins.
TOKENIZER_PATTERNS = ["*.json", "tokenizer*"]
MODEL_PATTERN_SETS = [
    ["*.safetensors"] + TOKENIZER_PATTERNS,
    ["*.bin"] + TOKENIZER_PATTERNS,
    ["*.pt"] + TOKENIZER_PATTERNS,
]


@timer_decorator
def _download(name, revision, kind, cache_dir):
    if kind == "model":
        pattern_sets = MODEL_PATTERN_SETS
    elif kind == "tokenizer":
        pattern_sets = [TOKENIZER_PATTERNS]
    else:
        raise ValueError(f"Invalid kind: {kind}")

    for patterns in pattern_sets:
        path = snapshot_download(
            name,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=patterns,
        )
        for pattern in patterns:
            if glob.glob(os.path.join(path, pattern)):
                logging.info("Downloaded %s for %s", pattern, name)
                return path

    raise RuntimeError(f"No matching files for {name} (kind={kind})")


def _resolve_cache_dir() -> str:
    """The HF Hub cache directory the build is targeting."""
    cache_dir = (
        os.getenv("HF_HUB_CACHE")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or (os.path.join(os.getenv("HF_HOME"), "hub") if os.getenv("HF_HOME") else None)
    )
    if not cache_dir:
        raise RuntimeError(
            "HF_HUB_CACHE / HF_HOME not set; refusing to download into an "
            "unknown cache location."
        )
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _download_one(spec: dict, cache_dir: str) -> dict:
    """Download a single model+tokenizer pair. Returns local-args metadata."""
    model_name = spec.get("model") or spec.get("MODEL_NAME")
    if not model_name:
        raise ValueError(f"Manifest entry missing 'model': {spec!r}")

    revision = spec.get("revision") or spec.get("MODEL_REVISION") or None
    tokenizer = spec.get("tokenizer") or spec.get("TOKENIZER_NAME") or model_name
    tokenizer_revision = (
        spec.get("tokenizer_revision") or spec.get("TOKENIZER_REVISION") or revision
    )
    quantization = spec.get("quantization") or spec.get("QUANTIZATION")

    model_path = _download(model_name, revision, "model", cache_dir)
    tokenizer_path = _download(tokenizer, tokenizer_revision, "tokenizer", cache_dir)

    return {
        "MODEL_NAME": model_path,
        "MODEL_REVISION": revision,
        "TOKENIZER_NAME": tokenizer_path,
        "TOKENIZER_REVISION": tokenizer_revision,
        "QUANTIZATION": quantization,
    }


def _parse_manifest() -> list[dict]:
    raw = os.getenv("MODELS_MANIFEST", "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"MODELS_MANIFEST is not valid JSON: {e}")
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise SystemExit(
            f"MODELS_MANIFEST must be a JSON array of objects (got {type(parsed).__name__})"
        )
    return parsed


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[download_model] %(message)s")

    cache_dir = _resolve_cache_dir()
    manifest = _parse_manifest()

    if manifest:
        logging.info("Multi-model bake: %d entries", len(manifest))
        for i, entry in enumerate(manifest):
            logging.info("Entry %d/%d: %s", i + 1, len(manifest), entry)
            _download_one(entry, cache_dir)
        # Multi-model mode does not write /local_model_args.json — runtime
        # selects which model to load via MODEL_NAME.
        return 0

    if not os.getenv("MODEL_NAME"):
        logging.info("No MODEL_NAME or MODELS_MANIFEST set; nothing to download.")
        return 0

    spec = {
        "model": os.getenv("MODEL_NAME"),
        "revision": os.getenv("MODEL_REVISION") or None,
        "tokenizer": os.getenv("TOKENIZER_NAME") or None,
        "tokenizer_revision": os.getenv("TOKENIZER_REVISION") or None,
        "quantization": os.getenv("QUANTIZATION") or None,
    }
    metadata = _download_one(spec, cache_dir)
    metadata = {k: v for k, v in metadata.items() if v not in (None, "")}

    with open("/local_model_args.json", "w") as f:
        json.dump(metadata, f)
    logging.info("Wrote /local_model_args.json: %s", metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
