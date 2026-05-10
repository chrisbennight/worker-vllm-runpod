# AGENTS.md

Guidance for AI agents (and humans) working in this repo.

## Intent

This is a slim, opinionated [vLLM](https://github.com/vllm-project/vllm) serverless worker for **RunPod Serverless**, forked from [`runpod-workers/worker-vllm`](https://github.com/runpod-workers/worker-vllm). The fork makes three deliberate changes:

1. **Owned image lineage**: published to GHCR as `ghcr.io/chrisbennight/worker-vllm-runpod` instead of DockerHub `runpod/worker-v1-vllm`. CI uses `GITHUB_TOKEN`; no DockerHub credentials.
2. **Blackwell-first build matrix**: two CUDA variants — `:cu128` (default, covers Ampere/Hopper/Blackwell SM100) and `:cu130` (Blackwell-only, FP4 block-scaled cuBLAS). Single Dockerfile, controlled by build args from `docker-bake.hcl`. Mirrors the [`comfyui-base-runpod`](https://github.com/chrisbennight/comfyui-base-runpod) layout so both projects can share one Network Volume.
3. **Broader endpoint surface**: in addition to `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, and `/v1/messages` (Anthropic), the worker also serves `/v1/embeddings`, `/v1/rerank` (+ `/rerank`, `/v2/rerank`), `/v1/score`, and `/v1/classify` so a single image can serve generation, embedding, and reranker models.

Side effects of the fork:
- Multi-stage build with a hash-verified Python lockfile generated at build time (`pip-compile --generate-hashes`).
- Standard HF cache layout (`<volume>/.cache/huggingface/{hub,datasets}`), aligning with `comfyui-base-runpod` so the same volume can host both stacks without duplicating model weights.
- Auto-detects whether the volume is mounted at `/workspace` (pod) or `/runpod-volume` (serverless) and exports `BASE_PATH` accordingly.
- Drops dead Tensorizer scaffolding, the runtime `pip install transformers==X` shim, and the upstream Slack notification workflows.

## File map

| Path | What it owns |
|---|---|
| `Dockerfile` | Multi-stage build. Stage 1 (builder) installs Python 3.12, generates a hashed lockfile (`pip-compile --generate-hashes`), and installs all Python deps including `vllm[flashinfer]`. Stage 2 (runtime) is a slim runtime image with the Python deps copied over. Same Dockerfile for both CUDA variants — controlled by `CUDA_VERSION_DASH` and `TORCH_INDEX_SUFFIX` build args. |
| `docker-bake.hcl` | **Single source of truth** for every pinned version (vLLM, PyTorch, CUDA), the GHCR image reference, and the build target matrix. Targets: `cu128`, `cu130`, `dev`, `devpush-cu128`, `devpush-cu130`. |
| `builder/requirements.in` | Top-level Python deps that get fed into `pip-compile`. Kept small — vLLM and its transitive set carry most of the dependency tree. |
| `src/start.sh` | Runtime entrypoint. Detects whether `/runpod-volume` or `/workspace` is mounted, sets `BASE_PATH` and `HF_HOME`/`HF_HUB_CACHE`/`TORCH_HOME`/`VLLM_CACHE_ROOT` accordingly, then `exec`s the handler. |
| `src/handler.py` | Runpod serverless `runpod.serverless.start(...)` entry point and request dispatch. |
| `src/engine.py` | `vLLMEngine` (raw streaming) + `OpenAIvLLMEngine` (chat / completion / responses / messages / embeddings / rerank / score / classify). |
| `src/engine_args.py` | Auto-discovers any `AsyncEngineArgs` field from upper-cased env vars. Bespoke handlers for speculative decoding, LMCache, HF cache lowercase fallback, `hf_overrides` rope_scaling sanitization. |
| `src/download_model.py` | Optional bake-time model fetch. Accepts a single `MODEL_NAME` *or* a `MODELS_MANIFEST` JSON list, and downloads each into the standard HF hub cache. |
| `.runpod/hub.json` | RunPod Hub UI surface — env-var presets and choices. |
| `.github/workflows/release.yml` | Tag-driven release pushing `:vX.Y.Z-cu128`, `:cu128`, `:latest`, `:vX.Y.Z-cu130`, `:cu130` to GHCR. Auths with `GITHUB_TOKEN`. |
| `.github/workflows/dev.yml` | Manual dev workflow that pushes `:dev-cu128` and `:dev-cu130` without touching `:latest`. |
| `.github/workflows/pr.yml` | Automatic PR build. Runs on every push to a PR branch, builds both CUDA variants in parallel, pushes them to GHCR as `:pr-<num>-cu128` and `:pr-<num>-cu130`. The build is the merge gate — see "Branch and PR policy" below. |
| `.github/workflows/pr-cleanup.yml` | Deletes the `:pr-<num>-*` tags from GHCR when the PR closes. |
| `.github/pull_request_template.md` | Required PR template — see "Branch and PR policy". |
| `CHANGELOG.md` | Has a "fork divergence" section at the top documenting how this image differs from upstream `runpod-workers/worker-vllm`. |
| `docs/configuration.md` | All env vars, defaults, and choices for runtime configuration. |

## Architecture invariants

These are load-bearing — break them and the image stops working as designed.

- **`docker-bake.hcl` owns every version pin.** The Dockerfile declares `ARG` names but their defaults come from bake. Don't hard-code versions in the Dockerfile.
- **No runtime installs.** `start.sh` must never call `pip install`. All Python deps are baked at build time with hash verification. The legacy `TRANSFORMERS_VERSION` runtime install hook has been removed; pin it in the lockfile instead.
- **One Dockerfile, both CUDA variants.** Don't fork into `Dockerfile.cu130`. The build args (`CUDA_VERSION_DASH`, `TORCH_INDEX_SUFFIX`, `TORCH_INDEX_URL`) handle it.
- **Cache paths align with `comfyui-base-runpod`.** Models share `<volume>/.cache/huggingface/hub` so both stacks reuse the same weights. vLLM-private artifacts (torch.compile cache, CUDA graphs, LMCache disk tier) live under `<volume>/.cache/vllm/` so they never collide with comfy's state.
- **`BASE_PATH` auto-detects the mount point.** Pods mount the volume at `/workspace`, serverless workers at `/runpod-volume`. `start.sh` picks whichever exists. Don't hard-code either path in the application code.
- **GHCR-only publishing.** No DockerHub. The release workflow uses `secrets.GITHUB_TOKEN`; no manual secret setup is needed beyond `HF_TOKEN` for gated-model bakes.
- **Tag scheme**: `:cu128`, `:cu130`, `:vX.Y.Z-cu128`, `:vX.Y.Z-cu130`, `:latest` (alias for `:cu128`). The old `runpod/worker-v1-vllm:vX.Y.Z` DockerHub tags are not produced by this fork.

## Common change recipes

### Bump vLLM
- Edit `VLLM_VERSION` in `docker-bake.hcl`. Lockfile regenerates at build time so the new vLLM dependency tree is picked up automatically.
- Verify the `attention_backend` defaults still match expectations on Blackwell — vLLM's defaults change occasionally (e.g. FA4 → TRTLLM on SM100).

### Bump PyTorch / CUDA
- Edit `TORCH_VERSION` (cu128 default) and `TORCH_VERSION_CU130` blocks in `docker-bake.hcl`.
- For new minor CUDA versions: also update `CUDA_VERSION_DASH` and the base image tag in the Dockerfile. The base image moves rarely; flag it in the PR.

### Add a new top-level Python dep
- Add to `builder/requirements.in`. Lockfile regenerates on next build. Don't commit `requirements.lock` — it's per-CUDA-variant and built fresh in CI.

### Add a new endpoint or change request shape
- Plumb through `src/engine.py` (`OpenAIvLLMEngine.generate` route dispatch) and `src/handler.py` if needed.
- Update `docs/configuration.md` and `.runpod/hub.json` if there's a new env var.
- Update this file under "Architecture invariants" if it changes the surface.

## Branch and PR policy

**Never push directly to `main`.** Every change lands via a pull request, regardless of size.

- Create a branch (`feat/short-name`, `fix/short-name`, `chore/short-name` — pick whichever fits).
- Open a PR against `main`.
- Use the PR template at [`.github/pull_request_template.md`](.github/pull_request_template.md). It's required, not aspirational — agents and humans both fill out:
  - **Intent**: what the PR is trying to achieve.
  - **High-level approach**: the shape of the solution in 2-5 sentences.
  - **Design considerations**: alternatives weighed, options not taken, why this direction.
  - **Flags / concerns / follow-ups**: anything not fully settled, anything that becomes more important if this lands.
  - **Validation**: what was actually run to confirm it works.
  - **References**: external docs, upstream PRs, model cards.

The template exists because this repo is small but published — context has to live somewhere durable. Commit messages aren't enough; the PR description is the canonical record of *why*.

If a PR is genuinely trivial (typo, comment fix), say so in the Intent section and leave the rest minimal — but use the template structure.

### Merge gate: PR build must pass

`pr.yml` triggers on every push to a PR branch and builds both CUDA variants
(`cu128` + `cu130`) in parallel. **The PR cannot merge unless both build jobs
succeed.** Each push cancels the in-flight build for that PR and starts a new
one — so a rapid-fire push streak only spends runner time on the last commit.

To enforce this server-side, branch protection on `main` should require:

- `Build cu128` (job name from `pr.yml`)
- `Build cu130` (job name from `pr.yml`)

Configure under **repo Settings → Branches → Branch protection rules → main →
Require status checks to pass before merging**. The workflow can't set this
itself; it has to be flipped on once per repo by an admin.

Each successful PR build leaves an artifact at:

```
ghcr.io/chrisbennight/worker-vllm-runpod:pr-<num>-cu128
ghcr.io/chrisbennight/worker-vllm-runpod:pr-<num>-cu130
```

Pull either to test on a RunPod endpoint before merging. `pr-cleanup.yml`
deletes both tags when the PR closes.

## Validate before pushing

vLLM images are large (~12 GB) and pulls multiple GB of wheels — local builds are slow. Use these gates instead of full builds when iterating:

```bash
# 1. Bake-file validates and resolves args
docker buildx bake --print -f docker-bake.hcl cu128 cu130 dev

# 2. Shell scripts parse cleanly
bash -n src/start.sh

# 3. Python compiles (no runtime, just syntax)
python3 -m compileall -q src/

# 4. Dockerfile syntax (cheap)
docker buildx build --check -f Dockerfile .
```

For real validation, push a branch and trigger the **Dev Build** workflow with `push=true`. It builds both variants on GHA and pushes `:dev-cu128` / `:dev-cu130`. That's the cheapest "did this actually work" loop.

Avoid running `docker buildx bake dev` locally unless you actually need to iterate on something the bake/syntax checks can't catch — pulling all the wheels is multi-GB.

## Releasing

Tag-driven. `git tag vX.Y.Z && git push origin vX.Y.Z` triggers `release.yml` which builds and pushes both variants to GHCR. The workflow uses `GITHUB_TOKEN` — no manual secret setup.

`docker-bake.hcl` reads the version from the `TAG` env var (set in the workflow). Don't override `tags` via `--set` in the workflow; tag construction lives in HCL.

**Tag verification gate.** `release.yml` refuses to publish unless:
- The `version` resolves to a real git tag (`git rev-parse "$VERSION"` succeeds), AND
- That tag points at the same commit as `HEAD`, AND
- The working tree is clean.

This makes `workflow_dispatch` from `main` with a non-existent version fail loud instead of silently producing a `:vX.Y.Z-cu128` image with no matching tag. To dispatch manually, push the tag first and select that tag as the workflow ref.

**Provenance labels.** Every published image carries OCI standard labels populated from the build environment:

| Label | Source |
|---|---|
| `org.opencontainers.image.title` | hard-coded `worker-vllm-runpod` |
| `org.opencontainers.image.source` / `.url` | hard-coded GitHub repo URL |
| `org.opencontainers.image.licenses` | hard-coded `MIT` |
| `org.opencontainers.image.version` | `TAG` env (the git tag for releases, `dev` for dev builds) |
| `org.opencontainers.image.revision` | `GIT_SHA` env (`git rev-parse HEAD` from the build) |
| `org.opencontainers.image.created` | `BUILD_DATE` env (RFC3339 timestamp from the build) |

Inspect with `docker buildx imagetools inspect <image>` or `docker inspect <image> --format '{{json .Config.Labels}}'`. For local builds the provenance labels are empty (intentional — local builds shouldn't claim provenance).

## Public repo conventions

This repo is **public**. Anything committed is visible on GitHub. Therefore:

- **No tokens, no secrets, no private URLs.** Not in code, not in scripts, not in workflow YAML (use `${{ secrets.* }}` references). The bake-time model download reads `HF_TOKEN` via a BuildKit secret mount and never bakes it into a layer.
- **No internal hostnames or coworker handles.** This is a personal-use fork; treat anything beyond `chrisbennight/worker-vllm-runpod`, `runpod-workers/worker-vllm`, and the upstream model/library repos as out of scope.
- **License is MIT** (inherited from upstream).

## What "done" looks like for a typical PR

- Lives on a branch. Opened against `main` via a PR — never pushed directly to `main`.
- PR description follows [`.github/pull_request_template.md`](.github/pull_request_template.md): Intent, High-level approach, Design considerations, Flags / concerns / follow-ups, Validation, References. Sections that genuinely don't apply can be marked "n/a" but not deleted.
- Touched files form a coherent change.
- `docker buildx bake --print` resolves cleanly with the new args.
- Shell scripts pass `bash -n`.
- Python files pass `python3 -m compileall -q src/`.
- `docs/configuration.md` and `.runpod/hub.json` are updated when the user-visible env-var surface moves.
- `CHANGELOG.md` has an entry under the relevant section.
- No commented-out code, no scratch files, no version drift between `docker-bake.hcl` and Dockerfile defaults.
