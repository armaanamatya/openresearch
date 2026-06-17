# BES — evidence-first remediation + conversion/archival correctness (design)

> **Doc status:** Proposed · design spec · authored 2026-06-17 · **revised after 4
> external review rounds** (verdict stable across all four: *revise-then-ship*, cut
> evolution, make conversion + archival correctness a hard prerequisite).
> Supersedes the in-session "BES v2: 3-phase evolutionary redesign" proposal.
> Policy: [`docs/policies/documentation.md`](../../policies/documentation.md).

## 1. Summary

The request was "fix BES and incorporate it properly — a massive, creative,
effective plan." Recent literature (AlphaEvolve / ShinkaEvolve / AIDE / Large
Language Monkeys / Hyperband / MAP-Elites) pointed at an ambitious **3-phase
evolutionary BES**. A four-round advisor review **rejected that build** and
converged on a smaller, higher-confidence plan. This spec is that plan.

**Thesis (precise):** **conversion / report-projection is a *proven* blocker;
generation quality *and* selection quality remain *unproven* until conversion and
archive provenance are pinned.** So we spend effort on *measurement, conversion,
and archival correctness* — not on more *generation machinery* — and we gate every
BES efficacy claim behind a falsification experiment and a complete archive.

## 2. Why the evolutionary redesign was cut (evidence)

All numbers verified against repo records.

- **Selection sits at the edge of grader noise.** The All-CNN candidate pool
  scored **0.5488 vs 0.5567** (spread **0.00793**;
  `best_runs/allcnn_ab/bes/bes_candidates.json`). The only grader-repeatability
  estimate is σ ≈ **0.00667** — but that is a *single run's* N=3 stdev
  (`data/grader_calibration.json`, run `prj_4627097f8362928c`), **too thin to be
  evidence**. The spread is below pairwise noise √2·σ ≈ 0.0094 as a *sanity check
  only*; the honest conclusion is "the one observed static-SELECT margin is on the
  order of measured grader repeatability, and that estimate is too thin — so
  measure SELECT stability directly" (→ A1).
- **The proven failure is conversion, not generation.** Adam: the pool
  discriminated *cleanly* (0.546 vs 0.643), selected the better candidate, and the
  arm **still shipped 0.533** (`best_runs/adam_ab/`). Two claims of unequal
  strength, kept separate:
  - **(a) Demonstrable (intra-archive):** the final report's authoritative
    provenance/projection fields say *no measured result* — `baseline_metrics={}`,
    `experiment_run_id=null`, `primitive_trace={}` — **while the archived
    `rubric_evaluation` in the same bundle repeatedly cites `metrics.json` /
    `outputs/`.** The grader demonstrably saw measured artifacts the report's
    provenance fields deny. This holds regardless of archive curation.
  - **(b) Folklore (README-only):** the "best in-run verify 0.5686 banked →
    0.533 shipped" figure appears in **no** data artifact (no
    `dashboard_events.jsonl` was archived); it is unverified and must not anchor a
    claim. Exact causality requires reconstruction (→ P0).
- **Adam is not a valid stamped A/B pair.** `best_runs/adam_ab/ab_report.json`:
  control arm is **`unstamped`**, only BES is stamped — `arms_found:{bes:1,
  unstamped:1}`. **All-CNN is the only valid stamped pair** (n=1 clean).
- **Cost economics are fatal for a GPU cascade / evolution.** A discard-the-losers
  short GPU slice is ~1.9× GPU for a pool of 3 (5–15× under full evolution); for
  $/rubric-point to hold flat needs ~+90% more points; best clean evidence is +13%
  from a pool that did not discriminate.
- **Phase-3 mechanisms are statistically inert at population 4–8** (MAP-Elites bins
  ~1 individual/bin; novelty rejection discards 12–25% of a tiny pool; τ over a
  2-candidate ranking is degenerate ±1).

**Cut entirely:** generational evolution, source-level `cross` recombination,
program-database/genealogy, MAP-Elites, novelty rejection, CodeT consensus.

## 3. What already exists — and where it is incoherent

The "Phase-1" machinery is ~90% shipped, default-OFF — but the conversion rails
have a verified incoherence that *blocks any default flip*:

- **CPU smoke gate:** `select_best_gated` + `smoke_check_candidate`
  (`candidates.py:203`), behind `OPENRESEARCH_BES_SMOKE_SELECT`; pass-through when off.
- **Short-slice → select → full-GPU-on-winner:** `staged_search.py`
  (`run_staged_search:458`), wired at `primitives.py:5604`. Honesty invariant holds
  (only full cells reach `aggregate_cell_metrics`, `primitives.py:5644`) — **but
  budget-dropped full cells are only warned (`staged_search.py:526`), not folded
  into structured `scope.gaps`** (`cell_matrix.py:731` builds gaps from
  capacity+dataset only). Must fix before C3.
- **Champion-artifact is INCOHERENT (verified) — the default-flip blocker.**
  `_apply_champion_artifact` (`report.py:822`) restores **source only**
  (`restore_snapshot`, `champion_artifact.py:68-80`, which also never prunes extra
  files) and sets **only** `overall_score` + `champion_restored` (`report.py:852-855`).
  Downstream, `leaf_scores` backfill from the eval file **only if currently `None`**
  (`report.py:1663-1667`) and the eval scalar wins **only if `>=` current**
  (`report.py:1676-1687`). Net: a *higher* champion `overall_score` ships **paired
  with the stale, lower latest-verify `leaf_scores` / `rubric_evaluation`** — a
  top-line detached from its leaf evidence. (`meets_target` is NOT stale — it is
  recomputed from the final score at `report.py:1176` and repaired at `:1690`,
  after champion restore at `:990`.)
- **`median_score` is a single sample at default.** `record_champion(...,
  median_score=float(score))` (`binding.py:853`, `champion_artifact.py:116`) is a
  true median only when `OPENRESEARCH_GRADER_SAMPLES≥3`; the σ-gate default is **1**.
  The field name overclaims → record `sample_count`.
- **Conversion rails partly exist:** `finalize_regrade.py:131` (evidence-freshness
  gate), `:214` (avoids the degraded 0.35 auto-cap); best-of-run floor
  (`report.py:986`, `:1730`). Default-ON. Champion-artifact default-OFF
  (`report.py:813`). BES default-OFF/parity (`config.py:474`).

## 4. Methodology: P0-gated falsification

No BES **efficacy claim** and no GPU-cascade feature code (A2/C3) is produced until
**P0** (conversion + archival correctness) is pinned. **A1** (SELECT-stability
instrumentation) may run **in parallel** with P0 — it is a measurement experiment,
not an efficacy claim — but its archive must be complete and it cannot be marketed
as BES efficacy. The "P0 gates *everything*" framing is wrong: P0 gates **A2, C3,
and any efficacy conclusion**, not A1.

## 5. P0 — conversion + archival correctness (hard prerequisite)

The load-bearing prerequisite. Without it, every BES experiment risks measuring
report plumbing instead of candidate quality, and "the most important number
becomes folklore" (the Adam lesson).

- **P0-1 — deterministic conversion replay/regression.** Reconstruct the Adam-class
  case with **full** artifacts and add a deterministic test proving the *best
  honestly-banked evidence state ships*: the final report's provenance/projection
  fields (`baseline_metrics`, `experiment_run_id`, `primitive_trace`) must reflect
  the evidence the grader actually scored; a populated `metrics.json` cited by the
  rubric block can never coexist with an empty-provenance report. This is the
  acceptance test for the whole conversion surface.
- **P0-2 — coherent champion bundle (champion stays flag-only until this lands).**
  A champion entry must carry `score_sample_count`, a source-snapshot hash, and a
  metrics/scope hash, and restore must **either** re-materialize the full evidence
  bundle (source + the metrics/scope that earned the grade + a matching rubric
  block) **or** re-grade after restore and write a fresh rubric block — plus
  source pruning / manifest restore semantics so a restore can't leave stale extra
  files. Source-only restore is insufficient (§3). **No default flip.** Record
  `sample_count`; stop calling a single score a median.
- **P0-3 — archival-completeness gate.** No BES efficacy claim counts unless the
  run archive preserves **all** of: candidate source snapshots, `bes_candidates.json`,
  `dashboard_events.jsonl`, `experiment_runs.jsonl`, `rubric_evaluation.json`,
  `final_report.json`, `metrics.json`, rubric-tree / `generated_rubric.json`, and
  **exact env flags + stamp metadata + rubric-tree hash**. No complete archive, no
  claim.

## 6. A1 — SELECT-stability instrumentation (parallel to P0)

- **Procedure:** because the historical All-CNN pool snapshots are not retrievable
  (the `dir` paths point to a remote host, absent from local `runs/`), do a **fresh
  CPU-only candidate capture** on one cheap paper: generate **N≥3** candidates (so
  rank stability is more than a coin flip), **archive the snapshots** (P0-3), then
  re-grade K times at temperature=0 via `scripts/calibrate_grader.py`.
- **Report:** **top-1 flip rate, P(candidate i beats candidate j) under repeated
  grades, and the score-margin distribution** — *not* Kendall's τ (degenerate at
  small N).
- **Cost:** zero GPU, **bounded LLM spend** (K full-rubric re-grades; not "cents" —
  state it as measured against the actual rubric token usage).
- **Kill criterion:** if top-1 flips across re-grades at margins ≤ the measured
  repeatability, static LLM SELECT is noise → any selection-based BES is falsified;
  keep only the binary runnable/not smoke gate.

## 7. A2 — budget-matched predictiveness (only after P0)

- **Gate:** runs only after **P0** (else the final grade is plumbing-polluted).
- **Two endpoints:** (1) deterministic short-slice metric vs deterministic full
  metric *where available* (insulated from grader noise); (2) the rubric grade
  *after P0 is fixed*. Reporting both prevents confusing "short-slice bad" with
  "grader/reporting noisy."
- **Procedure (two-stage):** one cheap history paper first; proceed to a
  first-attempt paper and then SDAR only if short-slice ranking is stable. Run a
  stamped **refine-one** arm alongside.
- **Budget-matched:** compare BES at budget B vs refine-one at the **same token +
  GPU-wall budget** (current `cost_usd` is LLM-only and literally $0 for Adam BES on
  OAuth — unusable as written).
- **Bar to proceed to C3:** short-slice rank predicts full-run final on ≥2/3 papers
  including the history paper, and BES beats refine-one on **$/rubric-point** beyond
  the ±0.05 stochastic band.

## 8. C — minimal BES (only what survives the evidence)

- **C1 — CPU smoke gate (keep, validate):** turn on `OPENRESEARCH_BES_SMOKE_SELECT`
  and A/B-validate the runnable/not + missing-artifact gate. Noise-free (binary),
  cheap, no new code.
- **C2 — adaptive static SELECT for first-attempt papers (keep):** retain BES v1's
  CPU-cheap static SELECT, engaged only via the existing adaptive flag on
  first-attempt / weak-history papers — the one regime BES earns its keep, CPU-cheap
  so economics survive.
- **C3 — GPU short-slice select (CONDITIONAL):** build/enable the
  `staged_search`-backed select-on-*completed-short-run* (validation metric, not
  loss slope) **only if A2 clears its bar AND staged-search dropped groups are folded
  into structured `scope.gaps`**.

## 9. Invariants (non-negotiable)

- Flag-gated, **default-OFF, byte-for-byte parity when off.**
- **Honesty invariant:** full attempted-cell manifest → `aggregate_cell_metrics`;
  provisional candidate/slice metrics never enter the shipped score.
- **Conversion coherence:** the shipped `overall_score`, its leaf evidence,
  `meets_target`, and the provenance fields must describe one and the same evidence
  state (the P0 contract).
- **Archival completeness (P0-3)** gates every efficacy claim.
- **Fail-soft:** any error → fall back to one normal implementation.
- **Parity safety:** no change leaks into the default non-BES path.
- **A/B-gated:** ≥3 paired *stamped* budget-matched runs before any default flip;
  adaptive OFF on A/B arms.

## 10. Sequencing

```
P0  conversion replay + coherent champion bundle + archival gate
        │  (gates A2, C3, and every efficacy claim)
A1  SELECT-stability capture+regrade  ──┘ runs in parallel (no efficacy claim)
        ▼
A2  budget-matched predictiveness (deterministic metric + rubric-after-P0)
        ▼
C3  GPU short-slice cascade   (also blocked on staged-search dropped-groups → scope.gaps)

C1 smoke gate + C2 adaptive static SELECT — cheap, validate alongside (flag-only)
```

## 11. Success criteria

- **P0-1:** the Adam-class conversion case is reconstructed with full artifacts and
  a deterministic regression proves the best honestly-banked evidence state ships.
- **P0-2:** champion restore yields a coherent bundle (score ≡ leaf evidence ≡
  provenance), `sample_count` recorded; still flag-only.
- **P0-3:** every BES efficacy claim carries the complete archive.
- **A1:** top-1 flip-rate / pairwise-win / margin reported on an N≥3 fresh capture.
- **A2 (only if reached):** short-slice rank ρ on the deterministic endpoint ≥ bar,
  budget-matched, beats refine-one on $/rubric-point on ≥2/3 incl. the history paper.
- **Global guardrail:** no workstream regresses a non-BES run (parity).

## 12. References

Repo: `2026-06-16-grader-fidelity-and-harness-remediation-design.md`,
`2026-06-16-grader-noise-and-harness-remediation-design.md`,
`docs/superpowers/specs/2026-06-07-bes-integration/`, `bes_integration.md`,
`best_runs/{adam_ab,allcnn_ab}/`, `data/grader_calibration.json`.
Code (verified): `backend/agents/rdr/candidates.py:203`,
`backend/agents/rlm/staged_search.py:458,526,545`,
`backend/agents/rlm/champion_artifact.py:68-80,116`,
`backend/agents/rlm/report.py:813,822,852,990,1176,1663,1676,1690`,
`backend/agents/rlm/binding.py:853`, `backend/agents/rlm/finalize_regrade.py:131,214`,
`backend/agents/rlm/primitives.py:5604,5644`, `backend/config.py:474`,
`scripts/{calibrate_grader,ab_compare}.py`.
Literature (informed the *cut*): AIDE (arXiv:2502.13138), AlphaEvolve (2506.13131),
ShinkaEvolve (2509.19349), Large Language Monkeys (2407.21787), Hyperband,
MAP-Elites, CodeT (2207.10397).
