# Terminal prompt — execute the dark-switches plan

> Paste the block below into a fresh Claude Code session at the repo root to drive the
> work. It is self-contained.

---

```
Execute the dark-switches plan: docs/superpowers/plans/2026-06-21-dark-switches-plan.md
(research menu: docs/superpowers/specs/2026-06-21-system-improvement-opportunities.md;
operator handoff: docs/runbooks/2026-06-21-dark-switches-handoff.md).

GOAL: turn the built-but-OFF cost/reliability machinery into wins. Phase 1 = pure fixes,
ship hermetically (default-ON where it's a flag). Phase 2 = behavior-changing switches,
wire the mechanism + hermetic test but KEEP THE DEFAULT OFF (operator A/B-validates before
any default flip — repo rule: ≥3 paired A/B runs before flipping a default).

HARD CONSTRAINTS (non-negotiable):
- Work on branch feat/dark-switches (off feat/bes-conversion-correctness). Do NOT merge to
  main; do NOT touch main. Keep the branch separate from main.
- Zero GPU / zero live-API spend. Every test hermetic (suite is socket-hermetic). Real-pod
  validation is the operator's, not yours.
- TDD: red → green → refactor, one behaviour per test. Bind tasks to code ANCHORS (function
  names), verify the real signature with grep before editing — line numbers drift.
- Phase 1 flips: unset must behave as the NEW safe default; explicit opt-out must work; test
  both. Phase 2: unset must be byte-identical to today; test the mechanism ON; do NOT flip
  the default in code.
- Gate every task on the FULL hermetic suite (OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python -m
  pytest tests/ -q) + ruff (uvx ruff@0.15.16 check .). Commit per task with a clear message.

USE: superpowers:subagent-driven-development — fresh subagent per task, spec-compliance
review then code-quality review after each. The plan's tasks are already bite-sized with
real code; follow them in order (Phase 1 Tasks 1-4, then Phase 2 Tasks 5-10).

WHEN DONE: push feat/dark-switches, open a PR to the TRUNK (feat/bes-conversion-correctness),
NOT main. Update the handoff doc's Phase-1 table to "shipped" and leave the Phase-2 A/B
commands for the operator. Report: tests added, suite result, and the exact per-switch A/B
commands the operator still needs to run.

START by reading the plan, confirming the Phase-1 code anchors still exist (grep
orphan_guard_enabled, preflight_smoke, the patch-mode _os.replace in primitives.py), then
implement Task 1.
```

---

## Quick reference — what the prompt drives
- **Phase 1 (ship now):** re-preflight-after-patch · orphan-guard default-ON · preflight-smoke
  default-ON on cost sandboxes.
- **Phase 2 (wire, operator A/B before default flip):** cell-resume · dead-training early-stop
  · OOM hard-memcap · HF/dataset cache persistence · spot/interruptible GPUs.
- **Out of scope:** scoring default-flips (need the labeled honest/fab corpus — theme T7).
