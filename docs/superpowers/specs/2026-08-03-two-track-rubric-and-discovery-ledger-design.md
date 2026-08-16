<!-- doc-meta: status=proposed; last-verified=2026-08-03 -->
# Two-track reproduction rubric + discovery claims ledger — design

**Date:** 2026-08-03 · **Author:** operator + Claude · **Scope:** proposal — rubric
restructure (repo-track vs scratch-track) + the open-ended discovery/ideation scoring layer.

> **STATUS 2026-08-03 — PROPOSAL ONLY. Nothing here is implemented or wired.** No flags
> exist yet; every flag named below follows the canonical default-OFF convention and must be
> byte-identical when unset. This spec is the design record for the operator ask of
> 2026-08-03: (1) split reproduction into official-repo and from-scratch tracks, and
> (2) credit what a run *discovers beyond the paper* — missing baselines, uncited or
> superseding work, contradictions, improvements — without breaking the evidence-not-grade
> red line.

## Relationship to prior work

Builds directly on, and reuses rather than replaces:

- **Two-axis verdict** (`two_axis_report.py`, `OPENRESEARCH_TWO_AXIS_VERDICT`) — extended
  here from a 2-axis pair to a per-track matrix; the upgrade clamp is unchanged.
- **`OPENRESEARCH_USE_AUTHOR_REPO`** (`repo/` pristine clone + `rlm_state/repo_spec.json`)
  — today reference-only; Track A makes execution of it an explicit, provenance-tagged path.
- **`literature_claim_gate`** (advisory rubric-input grounding) — the seed of the citation
  validators in §4.
- **A/B harness + evidence fingerprints** (`experiment_arm`, `evidence_key`,
  `record_candidate_outcome`) — the improvement-claim validator is a re-application.
- **Canary + leaf-evidence-gate patterns** (`rubric_gen.append_canary_leaf`,
  `evidence_gate.py`) — ported to discovery claims (§4.4, D6).
- **Campaign width + `branch_type`** (`attempt_assessment.py`) — confirmatory replication
  (§5.2) and the contrarian branch (§5.3).
- **Campaign spec** `docs/periods/2026-07.md` (v2, F1–F16) — ledger discipline (atomic +
  fsync, write-ahead intent) is copied verbatim for the new ledgers.

## The core design decision: closed-world rubric, open-world ledger

Reproduction is **closed-world** — the paper's claims are the answer key, so the pre-run,
pinned `generated_rubric.json` remains the right instrument and the trust anchor.
Discovery is **open-world** — an answer key cannot be written before the run for things the
paper does not contain. Encoding "novelty" as rubric leaves would recreate the
score-laundering incentive the harness exists to kill (the SDAR 0.188 lesson): an agent
that cannot reproduce compensates with cheap "discoveries."

Therefore:

- **D1 (separation).** Discovery output NEVER enters the reproduction score's numerator or
  denominator. Two scores, reported side by side, never blended into one scalar.
- **D2 (gating).** Discovery credit requires the reproduction side to hold: no claim of any
  type is `validated` unless the attempt's implementation verdict is at least faithful.
  A contradiction asserted by unfaithful code is noise.
- **D3 (fitness).** For discovery, exactly as for reproduction: the fitness signal is the
  deterministic validator outcome keyed on measured artifacts / resolvable external
  records. The LLM's significance/novelty judgment is **advisory ranking for operator
  review, forever** — mirroring the scheduler-authority rule (its A/B gate is evidence for
  operator review, never an automatic flip).

## §1 Two-track reproduction rubric

Restructure the rubric top level (rubric_gen emits tracks as top-level categories; leaf
grammar below them unchanged):

- **Track A — official-repo replication** (present only when a repo resolves): environment
  builds, author code runs, results match paper tables within tolerance. Anchored on the
  pinned `repo_spec.json` commit SHA.
- **Track B — from-scratch reimplementation**: today's pipeline, unchanged. The stronger
  claim (PaperBench forbids author code precisely because Track A is weaker evidence of
  understanding).
- **Concordance** — scratch-vs-repo delta leaves. A disagreement is a *finding* (does the
  official repo reproduce its own paper?), not an averaging problem.

**Rules:**

- **D4 (no blind averaging).** Tracks are scored and verdict-ed separately —
  `{repo, scratch} × {implementation, replication}` — and Track A acts as corroboration for
  Track B, not an equal-weight contributor. `final_report.reproducibility` carries the
  matrix; the legacy scalar verdict projects from the scratch track (compatibility).
- **D5 (provenance tag, fail-closed).** Track A requires executing author code, which
  today's posture forbids (`repo/` is orchestrator-host-only, never shipped to a GPU
  backend). Flag `OPENRESEARCH_REPO_TRACK` opens a dedicated staging path; EVERY
  `experiment_runs.jsonl` row gains `source: "repo"|"scratch"` and the leaf evidence gate
  matches on source exactly as it matches model×dataset — a repo-run result can never
  substantiate a scratch-track leaf, and vice versa. A row missing `source` under the flag
  is treated as unsubstantiating (fail-closed).
- Track A weighting in rubric-gen must reflect claim strength (ops ≪ understanding);
  concordance leaves are deterministic (numeric compare of two on-disk metrics under the
  paper tolerance), never LLM-graded.

## §2 The discovery claims ledger

New per-run ledger `runs/<id>/discovery_claims.jsonl` (atomic append + fsync + torn-tail
repair, same discipline as `campaign/attempts.jsonl`). Written via a new
`record_discovery_claim` primitive (NB: bumps the bound-primitive count — currently **21**
— and must sync `PRIMITIVE_REGISTRY`, `tests/rlm/test_registry.py::EXPECTED`, and
`tests/test_claude_md_fidelity.py`).

### 2.1 Claim schema (v1)

```json
{
  "claim_id": "dc_0007",
  "type": "improvement | contradiction | missing_citation | superseding_work | novel_ablation | limitation",
  "statement": "<=400 chars, falsifiable prose",
  "evidence_refs": [
    {"kind": "experiment_row", "path": "experiment_runs.jsonl", "row_sha256": "..."},
    {"kind": "metrics", "path": "code/metrics.json", "json_path": "per_model.resnet56.cifar10"},
    {"kind": "citation", "id": "arXiv:2401.01234", "claimed_value": 94.1}
  ],
  "protocol_key": "<evidence_key(metrics, scope)>",
  "prereg_id": "pr_0003 | null",
  "validation": {"status": "claimed | validated | rejected | confirmed",
                  "validator": "deterministic", "reasons": []},
  "significance": {"score": null, "note": null}
}
```

`significance` is filled by the advisory LLM ranking pass and is display/sort metadata
ONLY (D3). `validation.status` is written exclusively by the harness validators, never by
the agent.

### 2.2 Validator contracts (deterministic, per type)

| Type | `validated` requires (all of) |
|---|---|
| `improvement` | Paired baseline+candidate rows on disk, SAME `protocol_key` (evidence-fingerprint match), delta > grader σ, seed variance when `OPENRESEARCH_LEAF_ACTUATE_SEEDS` plans exist |
| `contradiction` | Faithful-implementation certificate + measured result outside the paper tolerance + multi-seed rows ruling out noise |
| `missing_citation` (prior work) | Citation resolvable (arXiv/DOI) + absent from the parsed bibliography + published strictly before the paper |
| `superseding_work` (posterior) | Citation resolvable + claimed value verifiably present in the cited work (literature-claim-gate machinery); GOLD path: the competing method actually run → rows on our own evidence ledger |
| `novel_ablation` / `limitation` | Every stated number cites a resolving `evidence_ref`; the "paper doesn't cover X" predicate checked by harness scan of the pinned paper text |

Any claim with a non-resolving `evidence_ref` is `rejected` — the leaf-evidence-gate
pattern, ported (a claim's enthusiasm cannot outrun the ledger).

### 2.3 Lifecycle

`claimed → validated → confirmed` (or `rejected`). `confirmed` requires independent
replication by a different lineage (§5.2). Mirrors candidate → champion.

### 2.4 Discovery canary (D6)

`OPENRESEARCH_DISCOVERY_CANARY` seeds the literature task with one decoy — a
plausible-sounding citation that does not exist. Any validator credit on it is a
fabrication signal wired into campaign ASSESS `guard_flags` (same slot family as
`canary_tripped`; hard quarantine).

## §3 Campaign wiring

- ASSESS ingests the claims ledger read-only: counts by type/status land on the attempt
  row; **discovery never influences `grade_usable_for_terminal`** except via the canary
  guard (D6). DECIDE may use `validated` discovery as tie-break input for exploration
  direction, never as a substitute for the reproduction gate to any terminal.
- DISTILL treats `rejected` claims as lesson input (why did the validator refuse?) —
  fabricated claims teach nothing about the paper but plenty about the agent.

## §4 Additional eval axes (ranked by value-per-effort)

1. **Pre-registration / calibration** (`OPENRESEARCH_PREREG`) — before any NEW experiment,
   the agent writes a falsifiable prediction (direction + magnitude bucket) to
   `rlm_state/prereg.jsonl`, hashed, write-ahead (the campaign intent-row pattern).
   Calibration (Brier-style on quantized buckets) is scored deterministically and stamped
   into `final_report`. Converts opinion into bets settled by the evidence ledger; a
   well-calibrated agent earns advisory trust for its unverifiable prose.
2. **Confirmatory replication** — campaign `--width` child lineages (`<id>_w<k>`) run
   confirmation attempts for `validated` claims → `confirmed` (§2.3).
3. **Contrarian branch** (`OPENRESEARCH_CONTRARIAN_BRANCH`) — a `branch_type="contrarian"`
   whose prompt-level job is to find the regime where the paper's claim fails
   (distribution shift, seed sensitivity, hyperparameter cliffs). Asymmetry lives ONLY in
   the prompt; scoring/validators identical. `AttemptAssessment.branch_type` vocabulary
   grows by one value (validation in `_validated_branch_type`).
4. **Minimal reproducing example** — smallest config that still shows the paper's effect
   within tolerance; deterministically checkable (config size + effect present). A
   product-grade artifact for deepinvent.
5. **Robustness rung** — tolerance-band + seed-variance reporting (activates the L5 seed
   plans beyond plan-only).
6. **Efficiency frontier** — reproduction cost vs paper-claimed compute. BLOCKED on honest
   accounting: the cost ledger is blind to Foundry spend and idle GPU time (root
   `CLAUDE.md` cost-visibility rule); do not ship this axis on `cost_ledger.jsonl` alone.
7. **External SOTA grounding** — leaderboard numbers as a harness-owned catalog (the
   `_INFEASIBLE_DATASET_TOKENS` pattern), operator-refreshed, never agent-asserted.

## §5 Flags (all default-OFF, byte-identical when unset)

| Flag | Gates |
|---|---|
| `OPENRESEARCH_REPO_TRACK` | Track A execution path + `source` provenance tag + concordance leaves |
| `OPENRESEARCH_DISCOVERY_LEDGER` | `record_discovery_claim` primitive + validators + report section |
| `OPENRESEARCH_DISCOVERY_CANARY` | Decoy-citation trap → ASSESS hard-quarantine guard |
| `OPENRESEARCH_PREREG` | Pre-registration ledger + calibration stamp |
| `OPENRESEARCH_CONTRARIAN_BRANCH` | `branch_type="contrarian"` in campaign PLAN |

Default-ON flips follow the standing gate: ≥3 paired A/B runs + grader-σ + operator
sign-off.

## Non-goals

- **No blended scalar.** Reproduction and discovery are never merged into one number (D1).
- **No autonomous significance.** LLM novelty/significance judgment never becomes fitness
  or gates anything (D3).
- **No network at grade time beyond citation resolution**, which must route through a
  harness-owned resolver (tests stay socket-hermetic; the resolver is mocked).
- **No change to the reproduction pipeline with all flags unset** — byte-identical.
- **No GKE implications** — nothing here touches sandbox routing; GKE posture unchanged.

## Open questions (for the implementation spec)

1. Track A sandbox staging: reuse the scratch `code/` staging path with a separate
   workdir, or a dedicated `repo_run/` tree? (Leaning dedicated tree — keeps `code/`
   meaning "the agent's project" everywhere downstream.)
2. Citation resolver: local snapshot (operator-refreshed dump) vs live arXiv/Crossref
   call — live conflicts with the socket-hermetic test posture and adds a availability
   dependency to grading.
3. Does `missing_citation` need a comparability sub-validator (same dataset/metric) before
   `validated`, or is that permanently the advisory layer's job?
4. Where the discovery score surfaces in the UI/leaderboard without inviting Goodharting
   (proposal: counts by status, no scalar).
