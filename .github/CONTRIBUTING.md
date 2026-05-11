# Contributing

Thanks for your interest. This repo builds a slim vLLM serverless worker for use on RunPod, published to GitHub Container Registry as `ghcr.io/chrisbennight/worker-vllm-runpod`.

## Releases

Releases are driven by tags. The `Release` workflow builds and pushes both CUDA variants to GHCR using the repo's `GITHUB_TOKEN` — no Docker Hub secrets required.

To cut a release:

1. Pick a semver tag, e.g. `v0.2.0`.
2. Trigger the workflow in any of three ways:
   - Push the tag: `git tag v0.2.0 && git push origin v0.2.0`
   - Publish a GitHub Release with that tag.
   - Run the **Release** workflow manually with `version = v0.2.0`.
3. The workflow builds and pushes:
   - `ghcr.io/chrisbennight/worker-vllm-runpod:v0.2.0-cu128`
   - `ghcr.io/chrisbennight/worker-vllm-runpod:cu128`
   - `ghcr.io/chrisbennight/worker-vllm-runpod:latest`
   - `ghcr.io/chrisbennight/worker-vllm-runpod:v0.2.0-cu130`
   - `ghcr.io/chrisbennight/worker-vllm-runpod:cu130`

Tags are defined in `docker-bake.hcl`; do not override via `--set` in the workflow.

## Dev builds

Use the **Dev Build** workflow (manually triggered). It pushes `ghcr.io/chrisbennight/worker-vllm-runpod:dev-cu128` and `:dev-cu130` without touching `:latest`.

For local builds:

```bash
docker buildx bake -f docker-bake.hcl dev
```

## Bumping pinned versions

- vLLM, PyTorch, CUDA suffix, image reference — all live in `docker-bake.hcl` as `variable` blocks.
- The Python lockfile is regenerated at build time from `builder/requirements.in` (and the pinned vLLM version), so dependency bumps are usually just an `.in` or HCL edit.

## PRs

- Keep changes focused and explain the rationale in the description.
- Use `.github/pull_request_template.md` (required).
- Update `docs/configuration.md` and `.runpod/hub.json` when changing env vars or runtime behavior.
- Update `CHANGELOG.md` for any user-visible change.

## License

MIT (inherited from upstream).
