# OAuth Root Reliability + Harness Backstop Plan

> **For agentic workers:** implement in TDD order. Do not relaunch A1 or any
> reproduction loop with `claude-oauth` until Tasks 1-4 are complete. This plan
> is explicitly allowed to push back on old RLM infra assumptions where the
> failure proves the old contract is too weak.

**Goal:** Make `--model claude-oauth` usable for zero-dollar local/dev runs
without letting a weak root model burn minutes in a refusal loop or ship a
meaningless report. The fix is layered: steer by default, detect degenerate
loops quickly, then optionally let the harness auto-drive mandatory progress
behind a flag.

**Observed failure:** A1 run `bes_a1_1412_6980_1781710336` called only
`understand_section` and `extract_hyperparameters`, then attempted `FINAL_VAR`
16 times at iteration 0. The forced-iteration policy correctly refused every
attempt (`run_experiment has never been called`) but only returned a text error.
The root ignored that text, retried `FINAL_VAR`, then hit the defensive refusal
cap and failed with no `implement_baseline`, no candidates, and no score.

**Pushback on old infra:** The old contract is "the harness refuses and hopes the
root reads the refusal text." That is not an infrastructure guarantee. It is a
prompting convention. For paid or long-running workflows, required lifecycle
progress must be represented as state-machine gates the harness can observe and
act on. The model may choose *how* to solve a step; the harness must enforce
*that* mandatory steps actually occur or abort cheaply.

---

## Design Position

Use rollout option **1: split rollout**.

- **Steering layer default-ON.** Escalating refusal messages + early repeated
  refusal detection are low-risk. They are inert for healthy roots and cut the
  observed 13-minute churn down to 2-3 refusals.
- **Auto-drive default-OFF.** Harness-driven primitive execution changes control
  flow. It should be opt-in via `OPENRESEARCH_OAUTH_AUTODRIVE=1` until tested on
  several papers and compared against the existing root-only path.

Non-goals:

- Do not validate BES SELECT with `claude-oauth` until this lands.
- Do not use paid GPU to diagnose this. All tests are unit/integration replay
  tests over local run artifacts.
- Do not paper over the issue by permanently switching A1 to `gpt-5`; that is a
  temporary operator workaround, not the infrastructure fix.

---

## File Structure

| File | Responsibility | New/Mod |
|---|---|---|
| `backend/agents/rlm/forced_iteration.py` | Track repeated refusal signatures; emit escalating actionable messages; expose a degenerate-loop decision. | Modify |
| `backend/agents/rlm/run.py` | Wire an `on_degenerate_refusal_loop` callback; convert repeated no-progress loops into early abort or optional auto-drive. | Modify |
| `backend/agents/rlm/root_progress.py` | Pure state helper: derive required lifecycle stage from observed primitive calls / ctx state. | Create |
| `scripts/bes_a1_safe_capture.py` | Stop early on degenerate root loop; report `root_degenerate` instead of waiting for timeout. | Modify |
| `tests/rlm/test_forced_iteration_degenerate_loop.py` | Unit tests for repeated-refusal detection and escalation. | Create |
| `tests/rlm/test_root_progress.py` | Pure tests for lifecycle-stage inference. | Create |
| `tests/scripts/test_bes_a1_safe_capture.py` | Add fixture proving wrapper exits early on `root_degenerate` event. | Modify |
| `docs/runbooks/2026-06-17-bes-conversion-runpod-validation.md` | Update Tier 3 once OAuth root is safe again. | Modify |

---

## Task 1: Detect repeated refusal loops cheaply

**Problem:** `_MAX_REFUSALS_PER_RUN = 16` is a defensive termination cap, not a
progress guarantee. It allowed ~13 minutes of churn in the observed run.

- [ ] Add fields to `ForcedIterationPolicy`:
  - `_last_refusal_signature: str | None`
  - `_same_refusal_count: int`
  - `degenerate_refusal_threshold: int = int(env OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD or 3)`
  - `on_degenerate_refusal_loop: Callable[[str], None] | None`
- [ ] Compute a stable signature for refusal classes, e.g.:
  - `no_experiment`
  - `no_rubric`
  - `below_target`
  - `repair_floor`
  - `two_experiment_same_iteration`
- [ ] On each refusal, update same-signature count.
- [ ] When `signature == "no_experiment"` repeats at threshold and no progress
  primitive has occurred, call `on_degenerate_refusal_loop(message)`.
- [ ] Keep the existing 16-refusal cap as a last-ditch escape, but do not rely on
  it for normal safety.

Tests:

- [ ] Repeated `no_experiment` refusals fire the callback on the third refusal.
- [ ] Mixed refusal signatures reset the counter.
- [ ] Healthy progression (`record_run_experiment`) resets the counter.
- [ ] The default behavior remains accept-after-16 if no callback aborts.

---

## Task 2: Make refusal text operational, not advisory

**Problem:** The current message says "call build_environment... then
run_experiment", but in the no-code state that is incomplete: `run_experiment`
needs code, and `implement_baseline` is the real missing step.

- [ ] Replace the no-experiment refusal message with stage-specific text:
  - If no `implement_baseline` / no `code_path`: "call `plan_reproduction`, then
    `implement_baseline(plan)`; do not call `FINAL_VAR` again."
  - If code exists but no environment: "call `build_environment`."
  - If code and env exist but no experiment: "call `run_experiment(code_path, env_id)`."
- [ ] For `claude-oauth` roots, include a one-line hard command skeleton in the
  refusal text. This is not pretty, but the failure proves Sonnet needs fewer
  degrees of freedom under recovery.
- [ ] Emit the stage in the run warning payload:
  `payload.stage = "need_baseline" | "need_env" | "need_experiment"`.

Tests:

- [ ] No-code no-experiment refusal mentions `implement_baseline`, not only
  `run_experiment`.
- [ ] Existing-code refusal does not tell the root to re-implement unnecessarily.

---

## Task 3: Add root progress state helper

**Problem:** Old infra infers progress indirectly from text and score. We need a
small observable state machine.

Create `backend/agents/rlm/root_progress.py`:

```python
def infer_required_stage(*, primitives: list[str], code_path_exists: bool,
                         env_built: bool, total_run_experiments: int,
                         total_verifications: int) -> str:
    ...
```

Stages:

- `need_baseline`
- `need_environment`
- `need_experiment`
- `need_verification`
- `can_finalize`

Use this helper in forced-iteration messages and wrapper diagnostics.

Tests:

- [ ] Empty run -> `need_baseline`.
- [ ] Baseline code exists -> `need_environment` or `need_experiment`.
- [ ] Experiment exists but no score -> `need_verification`.
- [ ] Verification exists -> `can_finalize`.

---

## Task 4: Early abort for safe-capture / local dev

**Problem:** A1 safe-capture should never wait for timeout when the root is
provably degenerate.

- [ ] In `run.py`, emit a distinct warning/event:
  - `code="root_degenerate_refusal_loop"`
  - includes `signature`, `same_count`, and `required_stage`
- [ ] In `scripts/bes_a1_safe_capture.py`, poll `dashboard_events.jsonl` for that
  event.
- [ ] If seen before the candidate pool exists:
  - terminate the child process group
  - write `a1_result.json` with:
    `{"ok": false, "verdict": "root_degenerate", "required_stage": "..."}`
  - exit cleanly without waiting for timeout

Tests:

- [ ] Wrapper fixture with `root_degenerate_refusal_loop` exits early.
- [ ] Wrapper fixture with ordinary warnings continues polling.

---

## Task 5: Flag-gated auto-drive backstop

**Problem:** Steering still trusts the model. The harness needs a real guarantee
for OAuth if the model ignores recovery.

Flag: `OPENRESEARCH_OAUTH_AUTODRIVE=1`, default OFF.

Behavior:

- Trigger only when:
  - root backend is `anthropic-oauth` or model starts `claude-oauth`
  - repeated refusal signature is `no_experiment`
  - required stage is `need_baseline` / `need_environment` / `need_experiment`
  - no terminal failure or near-timeout condition is active
- Gate first:
  - emit `root_autodrive_gate` event
  - write a state marker under `rlm_state/root_autodrive.json`
- Auto-drive one step only:
  - `need_baseline`: call the existing baseline implementation route with the
    current plan/context if available; if no plan, ask the root one final
    structured `recommend_next_tool`/planning prompt.
  - `need_environment`: call `build_environment`.
  - `need_experiment`: call `run_experiment`.
- Return control to the root after the step. Do not fully run the paper from the
  harness in v1.

Tests:

- [ ] Flag OFF: event emits but no primitive is auto-called.
- [ ] Flag ON + `need_baseline`: one implementation step is attempted.
- [ ] Terminal failure / wall-clock floor disables auto-drive.
- [ ] Auto-drive writes an event and marker for postmortem.

Pushback: if this task grows large, split it. Do not hide invasive harness
execution inside `forced_iteration.py`; put orchestration in `run.py` or a new
module where side effects are visible.

---

## Task 6: OAuth root validation gate

**Problem:** `root_model_unvalidated` was a warning. We treated it like an
operator annoyance, but it predicted this failure.

- [ ] Add an explicit root validation result in `demo_status.json`:
  `root_model_validated: true|false`, `root_model_risk: "none"|"unvalidated"|"degenerate_loop_seen"`.
- [ ] For `claude-oauth`, show a loud warning in CLI output and run status:
  "OAuth root is supported for local dev but may require recovery steering."
- [ ] If `OPENRESEARCH_REQUIRE_VALIDATED_ROOT=1`, fail fast before any long run
  when the root is unvalidated.

Tests:

- [ ] `claude-oauth` marks `root_model_validated=false`.
- [ ] `OPENRESEARCH_REQUIRE_VALIDATED_ROOT=1` refuses unvalidated roots before
  expensive steps.

---

## Task 7: Documentation + operator runbook

- [ ] Update `CHANGELOG.md`.
- [ ] Update `CLAUDE.md` reference docs with the OAuth reliability plan.
- [ ] Update A1 runbook:
  - local OAuth A1 is allowed only after Tasks 1-4
  - `OPENRESEARCH_OAUTH_AUTODRIVE=1` is experimental
  - gpt-5 remains the recommended reliable-root path for the decisive A1 verdict
    until OAuth has a green replay test.

---

## Acceptance Criteria

- [ ] Replay/unit tests reproduce the exact failure shape from
  `bes_a1_1412_6980_1781710336`: repeated `FINAL_VAR` no-experiment refusal with
  zero `implement_baseline`.
- [ ] The run no longer waits for 16 refusals or a wall-clock timeout.
- [ ] Safe-capture exits in under 5 minutes on the degenerate loop.
- [ ] Healthy runs are behavior-identical unless a repeated refusal loop occurs.
- [ ] Auto-drive remains default-OFF.
- [ ] No paid GPU is used to validate this fix.

---

## Follow-On: Retire More Old Infra

This failure is a symptom of a broader old-infra issue: the RLM root is asked to
drive a mandatory lifecycle entirely through free-form Python decisions. That is
too loose for expensive systems.

Follow-on architectural direction:

1. Convert the paper reproduction lifecycle into an explicit harness state
   machine: `understand -> plan -> implement -> environment -> experiment ->
   verify -> repair/finalize`.
2. Let the root choose content and strategy inside each state.
3. Let the harness own state transitions, missing-stage detection, and cheap
   aborts.
4. Move special workflows like A1 candidate capture to direct engine entry
   points (`bes_rlm.compete()` / candidate generator) rather than relying on a
   full root reproduction loop when no full reproduction is desired.

That is the real pushback: for paid compute and long-running agent systems, "ask
the model to remember the workflow" is old infra. The harness should enforce the
workflow.
