# SDAR on on-demand 4×A100 (GCP) — SSH-VM orchestration hardening + grok-4.3 retry

> Design note. Date 2026-06-18. **Status: APPROVED** — root=`grok-4.3` (Foundry, OAuth-free),
> scope=harden-core-then-retry, provisioning=fresh on-demand `a2-highgpu-4g` (spot VM left as
> fallback), detector fix #1 in-scope. Push target: **deepinvent only**.

## 1. Incident (what we're fixing)

Run `sdar_gcp_20260618` (Azure Foundry root `gpt-chat-latest`) **FAILED 13:16 UTC**: the
chat-aligned model refused the RLM REPL premise — it emitted prose with **empty code blocks for
all ~20 iterations** (`code_blocks: []`, `sub_calls: 0`), called **zero primitives**, the 8 A100s
never engaged → rlm hit its 20-iteration cap → "could not parse RLM response" → `verdict: failed`.
The spot VM `sdar-a100-8g` was then preempted (TERMINATED); IAP SSH returned raw `4033 not
authorized` tracebacks.

Four gaps, each fixed below:

1. The degenerate-loop detector only catches **`FINAL_VAR`-refusal** loops (it hooks
   `forced_iteration._final_var`). A pure-prose root never calls `FINAL_VAR`, so nothing aborts it —
   it burns the full iteration cap on a billing GPU VM.
2. `AZURE_FOUNDRY_DEPLOYMENT=grok-4.3 … launch` is a **silent no-op** (env not forwarded to the
   detached remote bash).
3. The preflight hardcodes `--min-gpus 8` (and instance/spot), blocking a 4-GPU on-demand VM.
4. The handoff doc's monitor commands SSH **blind**; on a non-RUNNING VM they emit raw IAP
   tracebacks instead of a clear message.

## 2. Fix 1 — empty-iteration degenerate-abort (harness, cloud-agnostic)

**Hook:** `backend/agents/rlm/run.py`, class `_FatalBackendGateLogger(OpenResearchRLMLogger)`
(~line 1199) — the per-iteration `RLMLogger` subclass rlm calls every iteration. Its `log()`
already does `super().log(iteration)` then raises `_FatalPrimitiveAbort` (run.py:800) on a fatal
primitive; the raise is caught at run.py:2909 → `_finalize_fatal_primitive_abort` ships a clean
report. The raw `RLMIteration` exposes `.code_blocks`.

**Logic:** extract a small pure helper on `_FatalBackendGateLogger`:
`_register_iteration_progress(self, has_code_block: bool) -> None`. Maintain a consecutive
no-progress counter (lazy `getattr(self, "_empty_iter_streak", 0)`, no `__init__` override).
- `has_code_block` True → reset streak to 0 (the root is doing real work).
- False → increment.
- When streak `>= OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD` (default 3; reuse
  `forced_iteration._default_degenerate_threshold()`): best-effort fire the policy's degenerate
  callback if present (`_current_policy()` → if `policy.on_degenerate_refusal_loop`, call with
  `{"signature": "empty_code_block", "count": streak, "required_stage": None}` in try/except so it
  emits the existing `root_degenerate_refusal_loop` run_warning), then **raise**
  `_FatalPrimitiveAbort(primitive_name="root_degenerate_loop", result={"error": "...N consecutive
  iterations with no code block (pure prose) — root is not driving the RLM REPL loop",
  "failure_class": "root_degenerate_loop", "suggested_fix": "Use a validated agentic root (gpt-5 /
  claude / grok-4.3); chat-only models (e.g. ChatGPT-latest) refuse the RLM REPL premise."})`.

`log()` calls the helper AFTER `super().log()` (iteration still streamed/checkpointed) with
`has_code_block=bool(getattr(iteration, "code_blocks", None))`. The existing fatal-primitive check
is untouched. A code-block iteration resets the streak → **healthy runs byte-identical**.

**Test** `tests/rlm/test_empty_iteration_abort.py`: drive `_register_iteration_progress` directly
(no need to fake the full `RLMIteration`/`sanitize_iteration`). With threshold=3: two `False` calls
don't raise; the third consecutive `False` raises `_FatalPrimitiveAbort` with
`failure_class="root_degenerate_loop"`; a `True` in the middle (F,F,T,F,F) resets so no raise.
Monkeypatch `OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD=3` for determinism. Confirm
`_FatalBackendGateLogger` is the logger actually wired into `RLM(...)` (grep its instantiation).

## 3. Fix 2 — launch env-passthrough (`scripts/gcp_sdar_preflight.sh::launch_run`)

The detached remote command (`setsid nohup bash scripts/sdar_gcp_run.sh …`) never carries the
run-shaping env. Build a `printf %q`-quoted, set-only whitelist locally and interpolate it as
`env <whitelist> bash scripts/sdar_gcp_run.sh`:
```bash
REMOTE_ENV=""
for v in AZURE_FOUNDRY_DEPLOYMENT OPENRESEARCH_SDAR_PROJECT_ID OPENRESEARCH_GRADER_SAMPLES; do
  [ -n "${!v:-}" ] && REMOTE_ENV+="$v=$(printf %q "${!v}") "
done
```
`sdar_gcp_run.sh` already honors these via `${VAR:-default}`. After this, the documented
`AZURE_FOUNDRY_DEPLOYMENT=grok-4.3 … launch` works.

## 4. Fix 3 — GPU-count parameterization (`scripts/gcp_sdar_preflight.sh`)

Add `MIN_GPUS="${OPENRESEARCH_SDAR_MIN_GPUS:-8}"` to the env block (~lines 20–27); replace the
literal `--min-gpus 8` with `--min-gpus "$MIN_GPUS"` in `remote_check` (~151) and `remote_prepare`
(~176). Instance + spot are already env-overridable (`OPENRESEARCH_GCP_INSTANCE`,
`OPENRESEARCH_REQUIRE_SPOT`). This run sets `OPENRESEARCH_GCP_INSTANCE=<4g name>`,
`OPENRESEARCH_REQUIRE_SPOT=false`, `OPENRESEARCH_SDAR_MIN_GPUS=4`.

## 5. Fix 4 — read-only `monitor` action + VM-state guard (`scripts/gcp_sdar_preflight.sh`)

Add `assert_running()` (reads `status_only`; exits 1 with `ERROR: VM <inst> is <state>; run
'start'/'prepare' first` when not `RUNNING` — does NOT start the VM). Add a new **read-only**
`monitor` action: `assert_running` then SSH-tail `runs/sdar_gcp_run.out`,
`runs/<project_id>/dashboard_events.jsonl`, and `nvidia-smi`. Wire `monitor)` into the `case`. This
replaces the handoff doc's raw `gcloud compute ssh` monitor commands (which IAP-tracebacked on the
preempted VM) with one clean entry point. Mutating actions (`check`/`prepare`/`launch`) keep their
existing `start_vm`.

## 6. Run config + sequence

On-demand `a2-highgpu-4g` (4×A100-40GB, `us-central1-c`), **fresh VM** (spot VM untouched as
fallback), `OPENRESEARCH_REQUIRE_SPOT=false`, `--sandbox local`, root+sub-roles `grok-4.3`
(`AZURE_FOUNDRY_DEPLOYMENT=grok-4.3`), full-3-model scope (1.7B/3B/7B-sharded-over-2-cards),
`--max-wall-clock 86400`, `--project-id sdar_gcp_ondemand_20260618`, `OPENRESEARCH_GRADER_SAMPLES=3`.
WebShop best-effort (ALFWorld + Search-QA expected). **Fail-fast via Fix 1.**

**Gates.** (1) Local: `ruff` on changed py + `pytest tests/rlm/test_empty_iteration_abort.py` +
existing forced-iteration / sdar-asset tests + `bash -n scripts/gcp_sdar_preflight.sh`. (2) Cloud:
create VM (confirm cost first) → `sync` hardened harness+scripts → `prepare`→**[GREEN]**
(`OPENRESEARCH_SDAR_MIN_GPUS=4`, re-warms ~25 GB caches) → `launch` grok → `monitor` (first ~3
iters MUST show code blocks/primitives, else Fix 1 aborts) → e2e → **stop VM** when `final_report`
lands.

## 7. Out of scope (follow-ups)

Mirror Fixes 2/4 to `azure_sdar_preflight.sh` (Azure parity — Fix 1 already covers Azure runs);
auto-stop VM on completion; the full bash→Python operator refactor (handoff doc's architectural
direction).

## 8. Constraints

deepinvent push only; no AI-attribution trailer; author `lolout1 / appradhann@gmail.com`; stage
specific files; never commit `.env` / `runs/` / project-ids.
