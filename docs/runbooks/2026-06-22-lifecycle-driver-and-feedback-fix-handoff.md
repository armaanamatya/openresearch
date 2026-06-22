# Lifecycle Driver + Forced-Iteration Feedback Fix — Handoff (2026-06-22)

> Branch `feat/grounded-self-improvement-harness-reliability`. All new code is
> **flag-gated, default-OFF → byte-identical when off.** Unit-tested; the
> lifecycle driver is **live-validated on GCP**. One gap remains (repair-handling).

## TL;DR

The keyless `claude-oauth` "SDAR degenerates on GCP" failure is now fully
diagnosed and the harness fix is built + validated. It was **never** the model,
auth, the funded key, the smoke, or the transport. It was **two layered harness
defects**:

1. **The forced-iteration recovery loop was broken** — the harness's "call
   `implement_baseline`" guidance *never reached the root model*. FIXED +
   unit/integration-proven.
2. **The root ignores the guidance even once it reaches it** — so the harness
   must *drive* the lifecycle, not instruct it. The **lifecycle driver** is built
   and **live-validated** (it drove `plan → implement → run → verify` itself
   after the root degenerated, keyless).

**Remaining gap (next session):** the driver drives the chain **once, without
repair**. The driven `run_experiment` failed `preflight_blocked / repairable`
(normal first-try code violation a healthy root would fix). Add **bounded
repair-handling** for a green reproduction. Precise spec below.

## How we got here (the diagnosis arc)

- The RL-aware **pre-GPU smoke fix** (commit `bfd86d52`) unblocked the GPU path —
  a real but separate fix. See `2026-06-22-sdar-gcp-e2e-and-rl-smoke-fix-handoff.md`.
- `--effort` (reasoning) and `--system-prompt` (replace Claude Code's agent
  prompt) were tried on the OAuth root. **Both failed** to stop the degeneration
  — they addressed the wrong root cause. (They're kept; `--effort` gives the
  reasoning-starved root real reasoning, `--system-prompt` makes it an RLM root
  not a Claude Code agent — both reasonable, neither the fix.)
- The user's key insight: **foundry/gpt-chat degenerate the same way** → it's
  model-independent → the harness/lifecycle.

### Root cause 1 — the broken feedback loop (FIXED)

`forced_iteration.py::_intercepted_final_var` returns a block string carrying the
recovery guidance on refusal. **But the rlm REPL runs the root's code via
`exec()`, which discards the return value of a bare `FINAL_VAR(...)` statement.**
The interceptor never `print()`ed it, the final-answer side-channel
(`_last_final_answer`) is deliberately unset on refusal, and the rlm core builds
the root's next-turn prompt **only from `REPLResult.stdout`**
(`rlm/utils/parsing.py::format_iteration` → `rlm/core/rlm.py:398`
`message_history.extend`). So the guidance reached the **SSE/UI** (via
`on_refusal`) but **never the root**. Every model degenerated because the entire
recovery mechanism was shouting into a void. The docstring's claim "the root sees
it as the FINAL_VAR return value" was false for the whole life of the feature.

**Fix:** `print(f"[forced-iteration] {message}")` at the two refusal return sites
in `_intercepted_final_var`. The returned block string is unchanged (so
`find_final_answer`'s "no answer" detection still works). Unit + integration
tested: `tests/rlm/test_forced_iteration_feedback.py` (guidance now lands in
`REPLResult.stdout` **and** survives `format_iteration` into the next prompt).

### Root cause 2 — the root ignores the guidance → drive the lifecycle (BUILT + VALIDATED)

With the feedback fixed, the root *receives* "call `implement_baseline`" and
**still** reads the paper then loops `FINAL_VAR` → `DEG`. (2-refusals-per-turn
confirm it calls FINAL_VAR inside a ```repl block, so the fixed chain delivers
the guidance.) The control flow is delegated to the root and the root won't drive
it. So the harness must.

## The lifecycle driver (the build)

- **`backend/agents/rlm/lifecycle_driver.py::drive_lifecycle_chain(*, tools, ctx,
  paper_text, rubric_spec, start_stage, emit, min_remaining_s=300.0)`** — executes
  `understand_section → detect_environment → plan_reproduction →
  implement_baseline → run_experiment → verify_against_rubric`, starting at the
  stage the root stalled at (`root_progress.infer_required_stage`). The
  `implement_baseline` `plan` is the documented 3-key assembly
  `{paper_claim_map, environment_spec, reproduction_contract}` of prior outputs.
  Fail-soft (only explicit `ok=False` stops; exceptions caught), wall-clock-aware,
  calls the **wrapped** tools (so the ledger + `_total_run_experiments` side
  effects fire — a driven `run_experiment` un-degenerates the policy and satisfies
  the evidence-gate exactly like a root call). 17 unit tests
  (`tests/rlm/test_lifecycle_driver.py`).
- **Feasibility (why a fresh chain works):** the known caveat ("can't reconstruct
  root-assembled REPL args") only blocks *resuming* the root's half-built vars. A
  *fresh* chain never needs them — every arg comes from the paper text the harness
  holds (`context_dict["paper_text"]`, `["rubric_spec"]`) or the prior step's
  output the harness itself produced.
- **Wiring:** `run.py::_make_degenerate_loop_callback` `_on_degenerate` gained a
  drive branch, gated by **`OPENRESEARCH_LIFECYCLE_DRIVE`** (default OFF).
  Drives **at most once per run** (`ctx._lifecycle_drive_count`). **Hands back only
  if `run_experiment` actually ran** (trust signals updated → root's next
  `FINAL_VAR` accepted, existing best-of-run/finalize ships the driven score);
  otherwise falls through to today's honest early-abort. All existing guards
  (wall-clock floor, already-terminal precedence) preserved verbatim. 7 wiring
  tests (`tests/rlm/test_run_lifecycle_drive.py`); all pre-existing
  degenerate/autodrive tests still pass (byte-identical-off).

### Live validation (2026-06-22, `runs/sdar_drive_v1`)

```
11:09  root early-FINAL_VAR refused (×2)
11:14  DEG=1 (degenerated)  → harness drove: detect_environment → plan_reproduction   (lifecycle_drive event, DRIVE=3)
11:16  → implement_baseline (executor wrote the SDAR code, ~25 min)                    (DRIVE=4)
11:41  → run_experiment (XR=1) → verify_against_rubric                                 (DRIVE=6)
>>> DRIVE WORKED — harness drove run_experiment (XR=1, DRIVE=6) after degeneration
```

Decisive signal (`lifecycle_drive` event + `XR>0` after `DEG`) **confirmed**, keyless `claude-oauth`.

## THE GAP — bounded repair-handling — IMPLEMENTED (increment 1, unit-tested; GPU validation pending)

> **STATUS 2026-06-22 (this session):** implemented + unit-tested (37 driver/policy/wiring
> tests, full suite 3072 green, ruff clean). `drive_lifecycle_chain` now drives a **bounded
> repair loop** (`max_repair_iterations`, default `OPENRESEARCH_MIN_REPAIR_ITERATIONS`=2), and
> the run.py handback calls a new `ForcedIterationPolicy.reset_repair_state()` so a
> harness-driven repair is **accepted** instead of bouncing the root's next `FINAL_VAR` off the
> policy's repair floor (the subtlety the original spec missed). All flag-gated behind
> `LIFECYCLE_DRIVE` → byte-identical off. **Next:** GPU re-validate `DRIVE=1` (real score, not 0),
> then the increment-2 full inversion (harness-owned `LifecycleController` as the proactive
> primary path). Code: `lifecycle_driver.py`, `forced_iteration.py`, `run.py`.

The driven `run_experiment` returned `success=false,
failure_class=preflight_blocked, outcome=repairable` — the executor's first-try
code had a contract violation (the *normal* "needs one repair pass" a healthy
root handles via the existing repair loop). The driver v1 drives once with no
repair, so it went to `verify` (no metrics → no real score) instead of repairing.

**Spec for v2 (contained, reuses existing machinery):**
- In `drive_lifecycle_chain`, after `run_experiment`, if the result is repairable
  (`outcome == "repairable"` / `success is False` with a `failure_class` in the
  repairable set), drive a **bounded repair**: re-call `implement_baseline` with
  `plan["repair_context"]` set to the failed run's `{error, failure_class,
  contract_violations}` (the exact shape `implement_baseline` already consumes),
  then re-call `run_experiment`. Cap the repair attempts (mirror
  `OPENRESEARCH_MIN_REPAIR_ITERATIONS`, e.g. 2) + re-check `remaining_s` each loop.
- Stop the repair loop on a successful `run_experiment` (real metrics) or the cap;
  then `verify`. Keep fail-soft.
- Re-validate on GCP: `ROOT=claude-oauth SMOKE=0 DRIVE=1 PROV=spot
  PROJECT_ID=sdar_drive_v2 scripts/sdar_gcp_e2e.sh run`; success = the driven
  chain produces a real rubric score (not a degenerate 0).

## Run + monitor procedure (deterministic)

- **Runner** `scripts/sdar_gcp_e2e.sh` — params `ROOT PROV SMOKE EFFORT ROOT_MODEL
  DRIVE USE_REPO PROJECT_ID`. Lifecycle-driver run:
  `ROOT=claude-oauth SMOKE=0 DRIVE=1 PROV=spot PROJECT_ID=sdar_drive_<id>
  scripts/sdar_gcp_e2e.sh run`. New actions `logs` (live `tail -f`) and `events`
  (readable dashboard-event dump via `scripts/render_run_events.py`).
- **Drive-aware monitoring:** the runner's `monitor` returns on `DEG`, but with
  `DRIVE=1` a `DEG` is *expected* (it triggers the drive). Watch instead for the
  `lifecycle_drive` event + `XR>0` (the harness drove `run_experiment`). A custom
  loop that polls `grep -c lifecycle_drive` + `wc -l experiment_runs.jsonl` and
  breaks on `DRIVE>=1 && XR>=1` is in this session's history.
- **Verbose logs:** `scripts/render_run_events.py --tail N <dashboard_events.jsonl>`
  renders the event stream readable (highlights `run_warning`/errors).

## Gotchas (verified this session)

- **Spot preempts often** — the a2 VM `TERMINATED` ~4× this session; the degenerate
  run can wedge `sshd` (kill `pkill -f "project-id <pid>"`, or `down`→`up` for a
  control-plane reset). On-demand a2 was stocked out in `us-central1-b`.
- **Local can't validate this:** no CUDA-usable GPU on the WSL host, and the local
  `claude` subscription collides with the run's `claude-oauth` CLI (the Claude
  Code session and the run fight over the same OAuth) — use GCP (separate headless
  token, no collision).
- The OAuth root records `reasoning_tokens: 0` (`claude --print` had no effort
  flag) — `--effort` adds reasoning but is **not** the degeneration fix.

## Files

| File | Change |
|---|---|
| `backend/agents/rlm/forced_iteration.py` | feedback fix (2 `print` lines) |
| `backend/agents/rlm/claude_oauth_client.py` | `--effort` + `--system-prompt` |
| `backend/agents/rlm/lifecycle_driver.py` | **NEW** — `drive_lifecycle_chain` |
| `backend/agents/rlm/run.py` | wiring (`_lifecycle_drive_enabled`, drive branch) |
| `scripts/sdar_gcp_e2e.sh` | `DRIVE`/`EFFORT`/`ROOT_MODEL` params + `logs`/`events` |
| `scripts/render_run_events.py` | **NEW** — readable event renderer |
| `tests/rlm/test_{forced_iteration_feedback,lifecycle_driver,run_lifecycle_drive,claude_oauth_root_effort}.py` | **NEW** tests |

Flags (all default-OFF, byte-identical off): `OPENRESEARCH_LIFECYCLE_DRIVE`,
`OPENRESEARCH_ROOT_EFFORT`.
