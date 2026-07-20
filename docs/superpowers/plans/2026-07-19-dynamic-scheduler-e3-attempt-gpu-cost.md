# Enrichment 3 — true per-attempt GPU-$

## Goal

Replace the shadow scheduler's equal-share GPU-cost estimate with the observed
`AttemptAssessment.cost.gpu_usd` for each assessed attempt.  Preserve the two
meters: `scope_rung` remains fidelity-only and the observed cost is used only to
calculate width.

## Test-first steps

1. Add an ON test in `tests/rlm/test_asha_advisory_shadow.py` with two assessments
   whose `cost.gpu_usd` differs.  Assert the advisory's ranked branch decisions
   use those two exact costs, and that a zero/absent cost remains a zero cost.
2. Add an OFF test that compares the `Decision.to_dict()` output before and after
   invoking `_maybe_attach_asha_advisory` with the tree flag absent; assert exact
   equality and no `asha_advisory` key.
3. In `_maybe_attach_asha_advisory`, retain the environment gate as the first
   executable behavior.  After it, build `{attempt_n: assessment.cost.gpu_usd}`
   using only finite non-negative numeric values; omit malformed values so the
   adapter's conservative zero fallback applies.
4. Remove the uniform `gpu_usd_spent / n` mapping, but keep remaining campaign
   budget and A100 cap inputs unchanged.  Keep the whole enrichment inside the
   existing fail-soft `try` block.
5. Update the ASHA section of `backend/agents/rlm/CLAUDE.md` to say that the
   advisory consumes each assessment's observed GPU spend.
6. Run the focused shadow/adaptor tests and then all `tests/rlm/test_campaign_*.py`
   with the socket-hermetic command.

## Acceptance

No code path reads assessment cost while the flag is OFF.  A failure in cost
extraction produces no live-decision change and cannot influence evidence verdicts.
