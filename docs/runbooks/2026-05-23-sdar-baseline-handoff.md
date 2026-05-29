# SDAR (arxiv 2605.15155) baseline + next-session handoff

**Status:** active baseline as of 2026-05-23. The most recent multi-fix run
is in flight on `feat/efficiency-tier1-2`.

## Why SDAR is the baseline

[SDAR — Self-Distilled Agentic Reinforcement Learning](https://arxiv.org/abs/2605.15155)
is the canonical "hard" test paper for this pipeline because it stresses
every dimension of the system simultaneously:

| Dimension | What SDAR forces |
|---|---|
| Real LLM weights | Qwen3-1.7B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct (3 sizes) |
| Real datasets | ALFWorld + WebShop + Search-QA (text-game + web + multi-hop QA — three distinct environments) |
| Real algorithm | GRPO RL + sigmoid-gated On-Policy Self-Distillation (OPSD) — both losses required |
| Multi-seed | seeds=[42, 43, 44] |
| Compute | paper uses 8× H800 GPUs, 150 training steps |
| Comparison surface | 5 baselines (GRPO, OPSD, Skill-SD, GRPO+OPSD, RLSD) |

A surrogate cannot pass the rubric — the rubric's fine-grained leaves
inspect for `g_t = σ(β · Δ_t)`, `stop_grad` on the gate, λ=0.1, β=10, real
Qwen weights, real ALFWorld episodes, etc.

**Default test invocation** (smallest-two scope, capped for cost):

```
REPROLAB_RUNPOD_CLOUD_TYPE=COMMUNITY \
REPROLAB_BASELINE_EXTRA_GUIDANCE="SCOPE: reproduce SDAR using ONLY the two SMALLEST model variants the paper tests — Qwen3-1.7B-Instruct and Qwen2.5-3B-Instruct. SKIP Qwen2.5-7B entirely. Use the real pretrained weights from HuggingFace (no surrogate) and the real ALFWorld + Search-QA + WebShop datasets, but evaluate on a small representative slice (e.g. 32 tasks per env) to keep wall-clock practical on a single 24–48GB GPU. Report results for both 1.7B and 3B." \
.venv/bin/python -m backend.cli reproduce 2605.15155 \
  --mode rlm --sandbox runpod --model claude-oauth \
  --vram-gb 38 --max-wall-clock 5400 --max-pod-seconds 5400 --max-usd 20
```

## What we learned during the 2026-05-23 debug cycle

| Observation | Fix landed | Commit |
|---|---|---|
| React duplicate-key warnings (`baseline-candidate-path_1/2/3`) on every multi-iteration run — root cause: foldCandidateProposed blindly appends when the LLM root re-emits candidate IDs across iterations | dedupe by node.id in foldCandidateProposed + dedupe edges in layoutConstellation | `c4860e3` |
| Agent wrote a TinyLM surrogate against synthetic ALFWorld-like data (rubric ceiling 0.13 even on a real A6000) | NO STUB block in baseline-implementation prompt + RUNTIME COMPUTE DETECTION (popped from stash) | `4b6798f` |
| No clean way to scope a paper to a subset of its model sweep (e.g. "only Qwen 1.7B + 3B") | `REPROLAB_BASELINE_EXTRA_GUIDANCE` per-run env-var hook (paper-agnostic) | `9f5233c` |
| RunPod returns HTTP 500 "no instances currently available" — the resolver's ladder existed but didn't auto-advance on capacity errors | Capacity-error escalation in `runpod_backend._request_json` (`RUNPOD_CAPACITY_EXHAUSTED:` sentinel) + run_experiment escalation loop now matches it | `aae89ad` |
| Elapsed-clock badge kept ticking on a dead/failed run, showing "elapsed: 5h" misleadingly | runMeta.completedAt forwarded only when status is terminal; elapsedMs uses completedAtMs as reference when present | `d3151d5` |
| `pip install -q -r requirements.txt && python train.py` silently masked bitsandbytes-install failures on the cuda-runtime image (no dev headers), causing downstream "ModuleNotFoundError: transformers" | POD SETUP block in prompt + reverted default `runpod_image` to cuda-devel-ubuntu22.04 | `88c45b0` |

## Known infrastructure quirks (2026-05-23)

- **RunPod A6000 ($0.49/hr COMMUNITY)** — frequently no capacity; our resolver
  auto-advances to L40S → A100_40 → A100_80 → H100_80.
- **RunPod L40S ($0.86/hr COMMUNITY)** — provisioning succeeds but pods
  sometimes never expose SSH within 900s. Treat as flaky; the capacity-error
  escalation does NOT yet catch SSH-wait timeouts (deferred — task #44 if
  needed).
- **cuda-runtime image broke bitsandbytes / flash-attn / deepspeed** —
  reverted default to cuda-devel.
- **claude-oauth root model** — works fine on RunPod but is "not paper-validated"
  per CLAUDE.md (warning is informational; quality may be lower than
  Featherless Qwen3-Coder).

## Open follow-ups (not done this cycle)

1. **Run isolation per attempt** (task #42): each new `reproduce` invocation on
   the same project should archive prior `final_report.json`,
   `experiment_runs.jsonl`, `cost_ledger.jsonl` into `runs/<id>/attempts/<ts>/`
   so UI shows ONLY the current attempt's data.
2. **SSH-wait-timeout escalation** — extend the capacity-error pattern to also
   advance the ladder when a pod boots but never exposes SSH within the
   timeout (currently surfaces as an opaque "did not become SSH-ready").
3. **Per-model rubric breakdown in UI** — when a paper tests multiple model
   sizes, the lab UI should render a column per model (Qwen 1.7B vs 3B) so
   the user can see which size actually reproduced and which didn't.
4. **Codex review** of dynamic-GPU branch — never received notification; if it
   eventually lands, fold findings as a separate commit.

---

## Next-session prompt

Paste the block below to start the next session with full context. It points
at this runbook for detail.

```
You are continuing work on OpenResearch / ReproLab on branch
feat/efficiency-tier1-2. The previous session built the dynamic-GPU resolver,
the no-stub baseline-implementation prompt, the per-run extra-guidance hook,
RunPod capacity-error escalation, the lab UI dedupe + elapsed-clock-freeze,
and the cuda-devel pod-image revert. Full debug history:
docs/runbooks/2026-05-23-sdar-baseline-handoff.md.

The canonical baseline paper is SDAR (arxiv 2605.15155) — Self-Distilled
Agentic Reinforcement Learning. It tests three Qwen sizes (1.7B / 3B / 7B)
on ALFWorld + WebShop + Search-QA. Our scope guidance pins reproductions to
the two smallest models on a single 24–48GB GPU.

Goals for this session, in priority order:

1. PIPELINE / ORCHESTRATION HARDENING (highest priority)
   a. Implement run-isolation per attempt (task #42): on a new reproduce of
      an existing project, archive prior final_report.json,
      experiment_runs.jsonl, cost_ledger.jsonl into
      runs/<id>/attempts/<ISO-ts>/ before starting the new attempt. Keep
      project_id stable so the paper isn't re-ingested. UI should show ONLY
      the current attempt's data.
   b. Add SSH-wait-timeout escalation in run_experiment alongside
      RUNPOD_CAPACITY_EXHAUSTED and CUDA OOM. When a pod creates but never
      exposes SSH within boot_timeout_seconds, treat it as the same kind of
      infra failure: advance the ladder, retry, emit gpu_escalated
      reason=runpod_ssh_timeout.

2. UI / UX FOR MULTI-MODEL PAPERS
   a. SDAR is the edge case: paper tests 3 model sizes. The agent currently
      produces ONE metrics.json with all model rows mixed in. The lab UI's
      rubric-breakdown panel and final-report card should render a column
      (or tab) per model so the user can see qwen3_1.7b vs qwen2.5_3b
      results side-by-side.
   b. Extend the agent's prompt to write metrics in a per-model structure:
      metrics.json should have a `per_model` dict keyed by short name
      (qwen3_1.7b, qwen2.5_3b) with task-specific scores nested under each.
      Generic enough for any multi-scale paper, not SDAR-specific.
   c. The NodeDetailSidebar should show per-model rows when a candidate's
      metrics carry a per_model breakdown.

3. SDAR-SPECIFIC RUBRIC ENRICHMENT
   a. SDAR's fine-grained rubric leaves include exact algorithm invariants
      (sigmoid gate g_t = σ(β · Δ_t), stop-gradient on gate, λ=0.1, β=10).
      The agent's baseline-implementation prompt could include a SDAR-
      specific algorithmic-fidelity hint that surfaces these invariants
      explicitly when the paper is detected as SDAR. Use a generic mechanism:
      a paper-id → extra-guidance table, NOT a hardcoded if-statement.
   b. Add a CLI flag --paper-hint <id> that loads the right extra guidance
      automatically.

4. FINAL-RESULT REPORTING
   a. The current final_report.json conflates "model scope" (which models
      did we run) with "rubric" (which leaves did we satisfy). Separate
      these: add a `scope` section to the report with what the user
      requested, what we actually ran, and gaps. UI surfaces this so a
      partial scope doesn't look like a partial reproduction.
   b. The rubric scorer should clearly distinguish "method-fidelity-leaf
      passed because the SAME code passed for all models" from "method-
      fidelity passed only for the smallest model" — currently it's flat.

5. CODE QUALITY
   a. Strip the `aclose() asynchronous generator is already running` noise
      from claude-agent-sdk usage — known SDK quirk per CLAUDE.md, but
      pollutes logs. Either suppress at the call site or document as
      accepted.
   b. Polish the prompt-cache wrapper to confirm cache_hit telemetry surfaces
      so we can validate the Lane A token savings empirically.

Process:
- Use the feat/efficiency-tier1-2 branch.
- Same Opus-plans-Sonnet-executes pattern as before; fan out Sonnets for
  independent lanes.
- Each lane finishes with a focused commit.
- The SDAR run command is preserved in
  docs/runbooks/2026-05-23-sdar-baseline-handoff.md — re-run after each
  pipeline-hardening change to verify the rubric climbs.
- Frontend dev server on :3000, backend on :8000, both already up. Lab UI
  at http://localhost:3000/lab.
```

---

## 2026-05-28 attempt — blocked on REPL safe-builtins + auth precedence

Project: `prj_09047604e591d969` (same arxiv 2605.15155).

CLI: `python -m backend.cli reproduce 2605.15155 --provider openai --sandbox runpod --max-usd 15 --max-wall-clock 7200`

Outcome: `verdict: "partial"`, `iterations: 5`, `rubric.overall_score: 0.0`, `cost_usd: 0.0`. Wall-clock 7 min. Zero domain primitives executed (only `check_user_messages` x2).

Root cause: BUG-LR-011 — `rlm._SAFE_BUILTINS["globals"] = None` made `globals().get("report_state", {...})` crash with a bare `TypeError: 'NoneType' object is not callable` and no traceback. The model spiraled over iters 2-5, concluded "primitives unavailable" (false — all 15 were callable), and shipped a `partial`. See `runs/prj_09047604e591d969/rlm_state/iterations.jsonl` iter 0 `stderr_meta`.

Secondary: BUG-LR-014 — the first CLI attempt died at iter 0 with a 401 because a stale shell `OPENAI_API_KEY=sk-svcacct-…` shadowed the valid `sk-proj-…` in `.env`. Resubmitted with `env -u OPENAI_API_KEY` prefix.

Full forensics + fix designs: `docs/superpowers/specs/2026-05-28-rlm-stability-remediation-design.md`.

**BUG-LR-011 + BUG-LR-012 + BUG-LR-013 + BUG-LR-014 + BUG-LR-015 resolved in `271df91`.** The rerun command above is now safe — same env vars, same flags. No `env -u` prefix needed (the boot-time validator will warn if a shadow is detected).

