# Mechanistic-understanding validation sprint — `pb_mechanistic-understanding_1780066611`

**Paper:** A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity (Lee et al., 2024)
**Bundle:** `third_party/paperbench/mechanistic-understanding` (96 rubric leaves, mostly Code Development + Code Execution)
**Scope:** GPT2-medium DPO only (Llama2 explicitly out of scope per the bundle's addendum)
**Goal:** ONE clean end-to-end pipeline success on a real ML paper — decoupling SDAR-specific complexity from general harness brittleness.

**Run mode:** RLM, `claude-oauth` root + sub-agents, `--sandbox runpod`
**Branch:** `pipeline-validation-mech-understanding`
**Spec:** `docs/superpowers/specs/2026-05-29-pipeline-validation-mech-understanding-design.md`
**Retry rule:** same-failure-twice halts the sprint (see [P4 root_cause_signature](../../backend/agents/diagnostics/root_cause.py))
**Per-attempt cap:** 90 min wall-clock; ~$5 USD

## Attempt 1 (launched 2026-05-29 ~14:xx UTC, project `pb_mechanistic-understanding_1780066611`)

**Launch command:**
```bash
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  REPROLAB_BASELINE_EXTRA_GUIDANCE="Llama2 is out of scope per bundle addendum (third_party/paperbench/mechanistic-understanding/addendum.md). Reproduce GPT2-medium DPO only (~350MB model, fits any GPU). Use unitary/unbiased-toxic-roberta for toxicity scoring and thesofakillers/jigsaw-toxic-comment-classification-challenge for the Jigsaw dataset per the addendum's Useful details section." \
  .venv/bin/python -m backend.cli reproduce mechanistic-understanding \
    --mode rlm --model claude-oauth --sandbox runpod --provider anthropic \
    --max-wall-clock 7200 --max-pod-seconds 7200 --max-usd 5 \
    > /tmp/mech-understanding-attempt1.log 2>&1 &
```

**Process:** PID 51305, log `/tmp/mech-understanding-attempt1.log`.
**Backend:** PID running locally on :8000 (verified `/health` returns ok).
**Monitors armed:** loops 1 (launch log) + 2 (dashboard events for `pb_mechanistic-understanding_1780066611`) + 3 (backend /health).

**Validation signals this attempt is meant to surface:**
- BUG-NEW-038 fix held end-to-end (no OAuth refusal loop)
- BUG-NEW-041 SIGTERM handler is irrelevant unless the user kills the run mid-flight
- BUG-NEW-042 shape guard either silent (sub-agent wrote clean Dockerfile) or fires with `restored: true` (sub-agent stomped but guard recovered)
- P1 commands.json manifest check either silent or fires repairable (sub-agent claimed success without writing referenced files)
- P3 preflight-sanity is a no-op (runpod path skips by design)
- P4 root_cause_signature: only used post-mortem if attempt fails

**Success criteria:**
- `runs/pb_mechanistic-understanding_1780066611/final_report.json` with `rubric_score > 0.3`
- `verify_against_rubric` called ≥ 2 times (forced_iteration policy fires)
- No unhandled `dockerfile_shape_guard` warnings (if present, all `restored: true`)
- No `commands_missing_file` failures
- Wall-clock < 90 min, cost < $5

**Status:** running.

---

## Sprint log

(Bug entries appended as the monitor loops emit unexpected events. Each
entry: BUG-NEW-NNN | symptom | root_cause_signature | fix-or-defer | validation path. New-bug numbering continues from 042 — start at 043.)
