# Reproduction Campaign implementation — new-session handoff (2026-07-01)

> **Doc status:** Implementation handoff · 2026-07-01 · written for a fresh
> Claude (Fable 5) session implementing the Codex-reviewed spec
> `docs/history/specs/2026-07-01-reproduction-campaign-and-self-improving-harness-design.md`.

## What this is

OpenResearch reproduces research papers end-to-end, but the "repeat until
reproduced" outer brain is still a human babysitting
`scripts/loops/kill_and_restart.sh`. The approved spec replaces that with
`ReproductionCampaign` — a deterministic, checkpointed, budget-fail-closed
outer state machine (UNDERSTAND → per-attempt loop → honest terminal) over an
`AttemptDriver` seam (live CLI path default, unified `ReproductionRun` path
behind `OPENRESEARCH_UNIFIED_RUN`) — plus, as Phase C, a staged harness
self-edit tier gated by a dedicated `HarnessEditGate` with a constitutionally
frozen evaluator tier.

The spec is **v2**: adversarially reviewed by Codex (16 findings — 6 BLOCKERs
— all resolved; §20 maps each). The design was approved section-by-section by
the operator; the locked decisions in spec §2 are not up for relitigation.

## Where the repo stands

- **Branch:** `reconcile/grounded-self-improvement-on-main` (push ONLY to the
  `deepinvent` remote, and only when asked — never `origin`).
- **Phases 1a–1f are code-complete and merged** (env adapters, feasibility
  triage/`RunPlan`/`RunBudget`, `ComputeProvider`+`VmComputeProvider`+
  `ReproductionRun`, `CredentialBroker`/`AssetResolver`, `ExperienceMemory`/
  `FailureAttribution`/`held_out_gate`, `unified_run` composition root). All
  flag-gated default-OFF; the live path is byte-identical today.
- **Everything the campaign composes already exists** — spec §4 is the
  grounded inventory. The campaign is new orchestration over existing parts;
  it adds NO new LLM surface of its own.
- Live GPU runs may be in flight on GCP (e.g. `sdar_2model_v2` monitors from
  another session). Campaign work is additive and opt-in; touch nothing under
  `runs/` belonging to live projects.

## Required reading, in order

1. `docs/history/specs/2026-07-01-reproduction-campaign-and-self-improving-harness-design.md` — THE spec (v2, Codex-resolved). §20's resolutions are load-bearing implementation requirements, not commentary.
2. `docs/history/specs/2026-07-01-paper-agnostic-multicloud-reproduction-and-self-improvement-design.md` — the Phase-1 substrate (esp. §5.2–§5.5 compute tiering, §7 memory red line).
3. `docs/runbooks/2026-07-01-sdar-unified-run-cutover.md` — Phase-1f A/B discipline the campaign's paired mode conducts.
4. `CLAUDE.md` (auto-loaded) — invariants, flags, test commands.

## Implementation order (Phase B first; Phase C only after B is green)

Dependency-ordered units, each: tests first (hermetic, socket-safe), then
implementation, then `ruff` + full suite. Spec §15 is the file map; §14 is the
test catalog — both are binding.

1. **Campaign state + spend ledger** (`reproduction_campaign.py` skeleton):
   fail-CLOSED atomic+fsync ledger, write-ahead intent rows, halt-on-unwritable,
   resume protocol (`in_flight` + liveness probe → re-attach | assess-from-disk),
   idempotent replay. (Codex F1, F7 — the money-safety core.)
2. **`AttemptEnvelope` + enforceability** (`campaign_policy.py` part 1): split
   meters (LLM vs GPU USD vs GPU-hours vs wall-clock), mapping to the REAL
   knobs (`--max-usd` is LLM-only; GPU = `--max-run-gpu-usd`/hours; explicit
   `max-run-duration`, never the 28h default), `stage_on_gpu` accounting,
   fail-closed refuse-unattended. (F2, F3.)
3. **`AttemptDriver` + `LiveCliDriver`** (`attempt_driver.py`): run-spec
   round-trip validation (exact `OPENRESEARCH_*` keys, fail at $0 on any
   rejected key), force-quarantine of incomplete residue before every launch
   (new explicit-archive entry point on `attempt_isolation` — the warm-retry
   heuristic must never fire under a campaign). (F15, F6.)
4. **`AttemptAssessment`** (`attempt_assessment.py`): deterministic read of
   report/audit/guards/validator/attribution/cost; validator
   absence/staleness = quarantine, exactly like a tripped guard. (F4.)
5. **DECIDE policy** (`campaign_policy.py` part 2): terminal rules (incl.
   validator-clean requirement for `REPRODUCED`), guard-filtered
   campaign-owned champion selection + explicit seed pointer (score-ranked
   rails never choose the seed), lineage rule table, scope ladder, typed
   prose-free novelty fingerprint, plateau. (F5, F10.)
6. **Directive synthesis** (`campaign_directives.py`): structured artifacts
   only; transcript paths fail the build.
7. **UNDERSTAND gate** (`understanding_gate.py`): double-extraction
   deterministic diff; tiered blocking (span-grounded/probe-confirmed only;
   LLM-only fields advisory forever). (F9.)
8. **Campaign report + plan-only writer** (`campaign_report.py`): the campaign
   never terminates report-less. (F14.)
9. **CLI `campaign` subcommand + steering channel** (`cli.py`, thin route):
   `campaign/user_messages.jsonl` (the top-level one is archived per attempt),
   checkpointed cursor, `--resume`/`--driver`/`--mode`. (F13, F8: paired mode
   *conducts* the 1f A/B under operator sign-off.)
10. **`UnifiedRunDriver` + width minting** (`--project-id <campaign>_w<k>`),
    doomed-run comparator (flagged), evaluator lockdown + canary leaf
    (flagged). (F16, F12-adjacent.)
11. **Phase C (only after B green):** `harness_self_edit.py` +
    `self_edit_surface.json` whitelist + dedicated `HarnessEditGate`
    (executable `HarnessReplayCase`s; `held_out_gate` stays lesson-only) +
    strengthened canary (≥2 papers × ≥2 seeds + σ bound + negative control) +
    operator-confirmed default flip. (F11, F12.)

## Non-negotiable invariants (from the spec + repo law)

- **Evidence, not grade** (spec §2 row 11): the campaign layer adds no LLM
  judgment; grades count only inside a clean guard envelope.
- **Fail-closed money**: no LAUNCH without a durable intent row; no attempt
  whose meters can't be enforced; explicit cloud ceilings always.
- **Clean context**: attempt N+1 sees structured artifacts only — never
  transcripts; force-quarantine before relaunch.
- **All flags default-OFF; campaign path opt-in via subcommand; zero behavior
  change to `reproduce`/batch/UI paths.** Off-state tests prove it.
- **Frozen tier is structural** (Phase C): guards, evidence predicates,
  rubric(+gen), validator, budget enforcement, gates, whitelist file, the
  self-edit module itself. A test proves frozen-tier proposals are rejected.
- **No live GPU spend from the implementation session.** Hermetic tests only
  (`pytest-socket` blocks non-loopback). The cheap-paper + SDAR ≥3-paired
  validation campaigns are operator-run afterwards (~$300 budget).

## Verification commands

```bash
.venv/bin/python -m pytest tests/ -n auto        # full suite (must stay green)
.venv/bin/python -m pytest tests/rlm/test_reproduction_campaign*.py -q
uvx ruff@0.15.16 check .                         # lint (config in pyproject)
```

Off-state proof: run the existing suites with no campaign env set — byte-identical
behavior everywhere outside the new subcommand.

## Process contract (operator-locked)

- **Fable designs and reviews. Sonnet executes.** All non-trivial code is
  written by Sonnet subagents against tight per-unit specs authored by Fable;
  Fable reviews every diff closely (correctness, spec fidelity, invariants,
  test honesty) before the next unit starts. Codex is NOT used for
  implementation (it already did the spec review).
- Commit style: infrequent, substantial, milestone commits; descriptive
  present-tense headline (what + symptom + resolution); **no Conventional
  Commits prefixes; no Co-Authored-By / AI attribution trailers**; author =
  local git config (lolout1). Push only to `deepinvent`, only when asked.
- Keep `CLAUDE.md` + `system_overview.md` updated when adding the subcommand,
  SSE events, flags (both docs list them).
