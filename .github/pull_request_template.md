<!--
PR template — please fill in every section. The goal is for a future reader (or
agent) opening `git log` six months from now to understand not just *what*
changed, but *why* this approach, and what the rejected alternatives were.

Delete the comments and any sections that genuinely don't apply, but don't
delete a section just because it's awkward to fill in — that's usually where
the load-bearing context lives.
-->

## Intent

<!--
What is this PR trying to achieve? State the goal, not the diff. A reader
should be able to tell whether this PR succeeds without reading the code.

Examples:
- "Make the cu130 image bootable on RunPod's B200 pods with FA4 attention."
- "Cut Qwen3-VL cold-start time on a fresh Network Volume from ~6 min to ~90 s."
- "Stop the OpenAI engines from re-initializing on every request after a transient CUDA error."
-->

## High-level approach

<!--
The shape of the solution in 2-5 sentences. Which files/systems were touched,
and how do they fit together? Skip the line-by-line — this is the mental model
a reviewer needs before reading the diff.
-->

## Design considerations

<!--
What was the interesting decision in this PR, and what alternatives did you
weigh? Include the option(s) you didn't pick and why. Even one paragraph is
fine; the goal is to surface the hidden assumptions.

Useful prompts:
- Did you consider doing this at build time vs runtime?
- Did you consider a smaller diff that would have left some debt?
- Was there a "more correct" approach that was out of scope?
- What constraint forced the chosen direction (RunPod cold-start budget? vLLM
  upstream API churn? Blackwell kernel coverage?)
-->

## Flags / concerns / follow-ups

<!--
Anything you're not 100% confident about, anything you noticed but didn't fix,
anything that becomes more important if this PR lands. List them as bullets so
they can be turned into issues.

Examples:
- "FlashInfer head-size bug on SM120 (#40677) is still open; falls back to
  FLASH_ATTN cleanly but flagging in case Qwen3-VL hits it."
- "Bumping vLLM to 0.20.x dropped the legacy `MAX_CONTEXT_LEN_TO_CAPTURE` env
  var; not handling the migration in this PR."
- "Follow-up: pre-warm `torch.compile` cache to the volume on first boot."
-->

## Validation

<!--
What did you actually run to convince yourself this works? Be specific.

- [ ] `docker buildx bake --print -f docker-bake.hcl cu128 cu130 dev`
- [ ] `bash -n src/start.sh`
- [ ] `python3 -m compileall -q src/`
- [ ] Triggered the `Dev Build` workflow and pulled `:dev-cu128` on a RunPod endpoint
- [ ] Manually ran a request through `/v1/chat/completions` with the new model
- [ ] (other)
-->

## References

<!--
Links and external context. Anything a reviewer might need to follow along, or
that future-you will want when revisiting:

- vLLM PRs / issues / release notes
- Hugging Face model cards
- RunPod docs
- Discussion threads, RFCs, related PRs in this repo
-->
