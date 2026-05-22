# RLM Phase 5/6 — Progress

_Updated: 2026-05-22 — debug-and-harden session._

## Objective

Debug-and-harden pass: fix the issue catalogue (I1–I13 + I4), drive PaperBench
papers end-to-end (plus one harder recent paper), and complete GitHub issue #62.

## Status

The session's seven commits (`2630a77` P0, `4e7b4a4` catalogue I5–I13,
`52625d6` run_experiment Bug A/B/C + I3, `d656c7d` I4 + idempotency deflake,
`c22feb7` I3 revert, `652f842` rdr design spec + impl prompt, `bfe9e3d` run-3
outcome) are **squashed into one commit on `origin/main`**, whose `Closes #62`
keyword closed issue #62. The `merge` branch keeps the un-squashed history.

Test suite green (I3's 4 tests removed with the revert): 1252 passed, 3 skipped.

## run_experiment Bug A/B/C — fixed and verified

Run 2's `run_experiment` failed in 6 s. Three compounding bugs in
`backend/agents/rlm/primitives.py`, all fixed and **verified live by run 2b**:

- **Bug A** — `_execute_in_sandbox` logged stdout only; a failed command's
  stderr traceback was discarded. Fix: `_combine_command_output` joins both.
- **Bug B** — the experiment ran the image `detect_environment` built before
  any code existed (missing deps). Fix: `run_experiment` rebuilds from
  `ctx.project_dir/Dockerfile` via `build_environment`.
- **Bug C** — the sandbox ran `network_disabled`. Fix: `_execute_in_sandbox`
  enables network for the experiment container (user-approved).

## I3 — reverted (root-prompt change backfired)

The `_PAPER_GROUNDING` section anchored the `qwen3-coder-featherless` root on
the understanding phase — run 3 looped on `understand_section` for 21
iterations and never reproduced. Reverted; the known-good prompt is restored.
The `ftrl` acronym-collision I3 targeted is unaddressed — a robust fix needs
more than a prompt nudge on this root model. See `learn.md` 2026-05-22.

## I4 — fixed

Workspace `paper_text` now loads from the parser's full-text blob (located via
the `ParsingCompleted` event), not a lossy chunk-reassembly. Guard:
`test_paper_text_equals_parser_full_text`.

## Runs

See `runlog.md`.

- **Run 1** — sequential-neural-score-estimation: leaf **0.366**, partial.
- **Run 2b** — mechanistic-understanding (`pb_..._1779457326`): leaf **0.079**,
  failed — `run_experiment` succeeded (Fix A/B/C verified live); weak baseline.
- **Run 3** — GoRL (arXiv 2512.02581): first attempt looped under I3; re-run
  (I3 reverted) progressed through the full pipeline but crashed on the
  Featherless Qwen3-Coder 49 152-token context cap. `failed`, leaf 0.0.

## Remaining

Debug-and-harden and #62 are complete. Next phase: implement the rubric-driven
reproduction harness (`rdr`) — design spec and executable implementation prompt
in `docs/superpowers/specs/2026-05-22-rubric-driven-harness-{design,
implementation}.md`. That work happens on the `rlm_rubric_orchestration` branch.
