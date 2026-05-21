# Phase 2 Implementation Plan Audit — RESOLVED

Verdict: **ADDRESSED (2026-05-21).** Every audit finding was rechecked against
source; the valid ones are fixed in `docs/design/phase2-implementation-plan.md`
(and the brief / execution brief). Two findings were assessed as over-reach and
not actioned, with reasons below. The original audit's claim-verification table
independently **confirmed** ~30 of the plan's API/schema claims (D0, the runtime
signatures, the Pydantic schemas, the title-agnostic helpers, the event schema) —
that corroboration stands.

## Blocking — all fixed

1. **Task 11 `HYP_JSON` missing required fields — FIXED.** `ImprovementHypothesis`
   requires `path_id, hypothesis, rationale, expected_outcome`; the test data had
   only `path_id, hypothesis, category`, so the fail-soft drop would have made the
   test `0 == 2`. `HYP_JSON` now supplies all four required fields on both items.

2. **`implement_baseline` aggregate-contract crash — FIXED.** It did
   `PaperClaimMap(**plan.get("paper_claim_map", {}))` — `PaperClaimMap(**{})`
   raises (`core_contribution` required), which broke Task 8's own test *and* a
   real `plan_reproduction → implement_baseline` handoff. Now
   `PaperClaimMap(**{"core_contribution": "", **plan.get("paper_claim_map", {})})`,
   matching `detect_environment`. The `implement_baseline` docstring and its
   `PRIMITIVE_DESCRIPTIONS` entry now spell out `plan`'s aggregate shape so the
   root model knows what to assemble.

3. **`set_final` contradicts brief §7 — FIXED.** Brief §7 lists nine primitives;
   `set_final` is a vestigial Phase-1 skeleton stub (`rlm` terminates via
   `FINAL_VAR`, no helper needed). Task 12 is repurposed to *remove* `set_final`;
   the file structure, Task 13, and Task 14 now reference nine primitives.

## High

4. **Task 6 not fully fail-soft — FIXED.** `build_environment`'s body is wrapped:
   any non-`SandboxRuntimeError` exception returns `{"ok": False, ...}`;
   `SandboxRuntimeError` still propagates. D3's claim now holds.

5. **Rubric backstop vs the prompt's graduated caps — NOT ACTIONED (over-reach).**
   `RUBRIC_VERIFIER_PROMPT` does carry graduated caps (≤0.20/≤0.35/≤0.40), but
   `verify_against_rubric`'s mechanical `min(score, 0.35)` is the *same* single
   backstop `orchestrator._run_rubric_verifier` enforces — the prompt drives the
   finer caps LLM-side, and `min()` only lowers a score, so a prompt-following
   LLM's stricter score survives. Mechanically encoding 0.20/0.40 would diverge
   from the canonical orchestrator and needs run-state Phase 2's `run_experiment`
   does not produce. A clarifying note was added to Task 10 instead.

6. **D0 not reconciled across all upstream docs — FIXED.** Task 0 corrected
   issue #60 + `rlms-spike-report.md`; brief §7 was the remaining gap. Brief §7
   now states the root REPL is `environment="local"`, and its stale
   `answer`-variable termination claim (and reserved-built-ins list) is corrected
   to `FINAL_VAR`.

7. **Task 14 proved only one primitive — FIXED.** Task 14 gains a second test
   that binds all nine primitives via `build_custom_tools` and runs the three
   no-dependency heuristic primitives through the wrapper. With the REPL
   end-to-end test + the Task 6–11 unit tests, #59's "every primitive callable"
   is covered.

## Medium

8. **Task 8 fake parameter `ai` — FIXED.** Renamed to `artifact_index`.
9. **`detect_environment` drops `method_spec` extras — NOT ACTIONED.**
   `PaperClaimMap.extra="ignore"` is deliberate schema behavior; `method_spec` is
   contractually a (partial) `PaperClaimMap`. Not a defect.
10. **`environment_spec` vs build-result confusion — FIXED** via precise
    `PRIMITIVE_DESCRIPTIONS`: `build_environment` returns a build result
    (`{ok, image_tag, ...}`), not an `EnvironmentSpec`; `implement_baseline`'s
    `plan` shape is spelled out.
11. **`REPRODUCTION_PLANNER_PROMPT` file-write conflict — FIXED.**
    `plan_reproduction` now uses a primitive-specific system prompt
    (`_PLAN_REPRODUCTION_SYSTEM`), not the orchestrator's file-writing prompt.
12. **`iteration=None` — FIXED (clarified).** Task 2's `primitive_call` docstring
    now names the Phase-3 mechanism: a custom `RLMLogger` subclass supplies the
    iteration index (`rlm` calls `logger.log(iteration)` per loop).

## Low

13. **Task 10 expected count — FIXED.** "PASS (2 tests)" → "(3 tests)".
14. **D0 citation — FIXED.** `_build_exec_script` is named as the module-level
    function in `rlm/environments/docker_repl.py`, not a `DockerREPL` method.

## Existing-suite note

The audit's `pytest tests/` run (878 passed, 2 pre-existing local-process
failures, 2 skipped) is unrelated to Phase 2.

`pip check` reports `litellm 1.83.7 has requirement python-dotenv==1.0.1, but
you have python-dotenv 1.2.2`. This is **not** a `backend/requirements.txt`
defect: `pip install --dry-run --ignore-installed -r backend/requirements.txt`
resolves cleanly and pulls **no `litellm`** (`deepeval` does not depend on it).
`litellm` is in the working `.venv` only as a transitive dependency of `dspy`,
which the RLM-engine spike (`tools/rlms_spike.py`) `pip install`ed to evaluate
`dspy.RLM` — rejected; see `rlms-spike-report.md` and issue #66. Neither `dspy`
nor `litellm` appears in `backend/requirements.txt`, `backend/requirements-dev.txt`
or `pyproject.toml`. Phase 2 raised `python-dotenv` to `>=1.2.1` (a real `rlms`
requirement), which is what makes the spike-leftover `litellm`'s exact `==1.0.1`
pin surface in `pip check`. `pip uninstall dspy litellm` clears the warning; the
RLM code path never imports `litellm`. (Codex review finding #1.)
