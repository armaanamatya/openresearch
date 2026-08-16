# Parallel Feature-Ablation Campaign — Design & Execution Plan

<!-- doc-meta: status=plan; authored=2026-07-31; last-verified=2026-08-04; owner=operator; state=paused -->

> **STATUS 2026-08-04 — PAUSED.** Executed on GCP: baseline `base_rn3` credited (0.466); `all_on_rn5`
> FAILED (agent code-bug variance); the first Tree-B GPU authority run (`treeb_rn_live`) FAILED on
> branch-provisioning (each branch times out before training). Root causes + next steps tracked in
> [`docs/open-issues.md`](../../open-issues.md) and the results run-log
> ([`docs/2026-08-01-feature-ablation-results.md`](../../2026-08-01-feature-ablation-results.md)).

**Goal.** Measure the *isolated* and *combined* score contribution of each quality
feature (Tree-A within-run + Tree-B scheduler authority), via paired A/B runs, so a
default-flip decision rests on measured evidence — not intuition and not a single
confounded run.

**North-star invariant preserved.** Every decision below keys on the **deterministic
evidence layer** (on-disk `metrics.json`, receipts, `meets_target`), corroborated by the
rubric score — never the LLM grade alone. No arm auto-flips a default; the campaign
produces evidence for operator sign-off (mirrors `asha_authority_gate.py`).

---

## 0. Reality gate (why this is a *design*, not a live launch)

- Live A100 runs need operator GPU budget + auth; they are not launchable from a dev
  session. This plan is executable by an operator with `python -m backend.cli campaign`.
- **Tree-B features (freeze / branch / revive / true-kill) are UNMEASURABLE today.** The
  2026-07-23 attempts (`runs/adam_treat2_grok_20260723/`, `runs_logs/treat3_gpt_20260723/`)
  spawned 4 branches and produced **0 receipts** → `scheduler_tree_state.json.applied_seq=5`
  (4× `branch_spawned` + 1× `launch_claimed`, then dead) → `0.203 inconclusive`. Freeze/
  promote/revive/kill **never fired**. They cannot produce a score until the checkpoint
  prerequisite (§8) lands. Tree-B is therefore a *gated sub-campaign*, not part of Phase 1-2.

---

## 1. Feature → flag → arm registry

| # | Feature | Tree | Toggle (confirm in flag registry before run) | Prior evidence | Status |
|---|---|---|---|---|---|
| F1 | **BES** (best-of-N candidate pool) | A | `OPENRESEARCH_BES_ENABLED=1`, `OPENRESEARCH_BES_CANDIDATES_PER_CLUSTER=2`, `OPENRESEARCH_BES_SELECT_METRIC=cluster_score` (see `configs/ablation/arms.json` → `bes`) | All-CNN **+0.085**, Adam **−0.183** (confounded) | proven, sign-ambiguous |
| F2 | **Cross-run evidence / champion rails** (seeded best-ancestor + prior-attempt measured cells → implementer prompt + pinned rubric) | A | `OPENRESEARCH_CHAMPION_ARTIFACT=1`, `OPENRESEARCH_EVIDENCE_FINGERPRINT=1` (see `configs/ablation/arms.json` → `champion`) | All-CNN result-axis **0.000→0.470**, →0.739 | proven, big, 1 paper only |
| F3 | **Positive recipes** (cross-run) | A | `OPENRESEARCH_POSITIVE_RECIPES=1` | none | wired, **unproven** |
| F4 | **Experience memory** | A | `OPENRESEARCH_EXPERIENCE_MEMORY=1` | none | **dormant** (not wired into run/report) |
| F5 | **Negative lessons** | A | `OPENRESEARCH_NEGATIVE_LESSONS=1` | none | **dormant** |
| F6 | **Evidence-audit critic** (deterministic veto) | A | `OPENRESEARCH_EVIDENCE_AUDIT=1` | none | wired, unproven |
| F7 | **Leaf evidence gate** (per-leaf veto) | A | `OPENRESEARCH_LEAF_EVIDENCE_GATE=1` (default-OFF) | none | anti-fabrication; expect **flat-or-down** on honest papers, **catches** fabrication |
| — | Verdict evidence gate | A | `OPENRESEARCH_EVIDENCE_GATE` (default-ON) | — | **held fixed ON** (baseline, not ablated) |
| — | Deterministic finalize rail (floor-after-rescore) | A | harness-side, always-on | recovered 0.694→0.739 | **held fixed** (baseline) |
| G1 | **Branching** | B | `OPENRESEARCH_SCHEDULER_TREE=1` + `OPENRESEARCH_SCHEDULER_AUTHORITATIVE=1` + authority spec | only op that ever fired | **blocked** (§8) |
| G2 | **Freezing** | B | (same gate) | never fired | **blocked** |
| G3 | **Promote** | B | (same gate) | never fired | **blocked** |
| G4 | **Revive / backtracking** | B | (same gate) | never fired | **blocked** |
| G5 | **True-kill** (only on `training_diverged`) | B | (same gate) | never fired | **blocked** |

---

## 2. Papers (fixed test set)

Chosen for: known-achievable reproduction, existing rubric + prior runs (priors), and
**contrasting BES behavior** (so we can detect paper×feature interaction, not just main
effects).

| Paper | id | Baseline score | Why included | ~cost/run |
|---|---|---|---|---|
| **All-CNN** | 1412.6806 | 0.744 | richest feature history; BES **helped** | ~7 h, $5, 3 GPU |
| **Adam** | 1412.6980 | 0.831 | BES **hurt** (−0.183) — the confound stress case | ~2.5 h, $3 |
| **ResNet** | 1512.03385 | 0.62 | fast, cheap third paper for the ≥3-paper generalization check | ~2 h, $3 |

Optional stress paper (exclude from main stats, run once): **SDAR** (2605.15155) — never
exceeds 0.363; useful only to see whether any feature moves a *hard* paper off the floor.

---

## 3. Design: tiered OFAT + a targeted interaction cell

Full factorial (2⁷ = 128 arms) is infeasible. Use **screen → confirm → interaction**:

### Phase 1 — Screening (find what moves the needle), 1 seed
- Arms per paper: **Baseline** (all F3-F7 OFF, F1/F2 OFF) + **7 single-feature-ON** (F1..F7)
  + **All-ON** = **9 arms**.
- Papers: All-CNN + Adam (the two anchors). ResNet held for Phase 2.
- Runs: 9 arms × 2 papers × **1 seed** = **18 runs**, launched **in parallel** (§6).
  *(Execution deviation 2026-08-01: ResNet was promoted into the screen — 9 arms × 3 papers
  = 27 runs; see `docs/runbooks/2026-08-01-feature-ablation-gcp-runbook.md` STEP 3 and
  `docs/2026-08-01-feature-ablation-results.md`.)*
- Output: a per-feature Δ vs Baseline on each paper. Keep only features with |Δ| ≥ 2·grader-σ
  (§5) as *candidates*. This cheaply kills the dormant no-effect features (likely F4/F5).

### Phase 2 — Confirmation (clear the statistical gate), ≥3 paired seeds
- Arms: Baseline + surviving candidates (expect F1, F2, F6, F7, maybe F3) + **All-ON**.
- Papers: All 3 (All-CNN, Adam, ResNet) → generalization.
- Replication: **≥3 paired runs** per (feature, paper), seeds `{1,2,3}`, control shares seed
  with treatment (`ab_pair_id`). This is the `asha_authority_gate` minimum.
- Runs: ~(1 baseline + 5 candidates + 1 all-on) × 3 papers × 3 seeds ≈ **~63 runs**.

### Phase 3 — Interaction cell (does "combined" compose or interfere?)
The 0.744 All-CNN run already hinted **BES × cross-run-evidence is redundant** ("with the
champion evidence already in-prompt the extra angle bought nothing"). Measure it directly
with a 2×2 on All-CNN, 3 seeds:

| | evidence OFF (F2) | evidence ON (F2) |
|---|---|---|
| **BES OFF (F1)** | Baseline | F2-only |
| **BES ON (F1)** | F1-only | F1+F2 |

- Interaction Δ = (F1+F2) − F1-only − F2-only + Baseline. If < 0 → redundancy (expected).
- 4 cells × 3 seeds = **12 runs**.
- **The "all-combined into one" question** = All-ON arm vs **Σ(individual Δ)**. If
  All-ON Δ < Σ individual Δ → features are sub-additive/redundant (design implication: ship
  the cheapest sufficient subset, not everything). If ≈ Σ → additive. If > Σ → synergy.

---

## 4. The concrete arm matrix (Phase 1)

```
arm_id            F1 BES  F2 evid  F3 recipe  F4 expmem  F5 lessons  F6 audit  F7 leafgate
------------------------------------------------------------------------------------------
baseline           .       .        .          .          .           .          .
bes                X       .        .          .          .           .          .
champion           .       X        .          .          .           .          .
recipes            .       .        X          .          .           .          .
expmem             .       .        .          X          .           .          .
lessons            .       .        .          .          X           .          .
audit              .       .        .          .          .           X          .
leafgate           .       .        .          .          .           .          X
all_on             X       X        X          X          X           X          X
```
Each row runs on {All-CNN, Adam}. `experiment_arm={arm:<id>, ab_pair_id:"ablate-p1-<paper>-<seed>"}`.

---

## 5. Metrics & decision rule (evidence-not-grade)

**Per run, record:**
- Primary (grade axis): `rubric.overall_score`, `compute_adjusted_score`, `meets_target`.
- **Decisive (evidence axis):** deterministic on-disk metrics — `test_error_pct` per cell,
  cells-converged count, `metrics_sha256`; for Tree-B: `#receipts`, applied actions.
- Cost axis: `gpu_usd` (deterministic assessment, **not** `cost_ledger.jsonl`), wall-clock,
  GPU-hours → **score-per-dollar**.

**Grader-σ calibration (gate prerequisite).** For each arm's final evidence, re-grade the
**identical** `metrics.json` **K≥5** times → σ. A feature Δ counts only if grader-σ ≤ 0.02
(the `asha_authority_gate` bar). This separates real gains from grader noise (the G1
truncation fix moved scores 0.02-0.06 with *zero* evidence change — that magnitude is the
noise floor to beat).

**A feature "wins" iff ALL hold** (fail-closed, like the authority gate):
1. mean paired Δ(rubric) > 0 across ≥3 pairs, **and** |Δ| > 2·grader-σ;
2. sign-consistent across pairs (paired sign-test / t-test p<0.05), **and** not driven by one paper;
3. **deterministic evidence corroborates** (e.g. more cells converged, lower test-error) —
   a grade gain with flat evidence is *rejected* as grader drift;
4. cost-benefit non-negative (Δscore per Δ$ acceptable) or explicitly accepted by operator.

**Never** auto-flip a default. Output = a manifest for operator review.

---

## 6. Parallel execution mechanics

"In parallel" = **one independent VM per arm**, launched concurrently.

```bash
# One arm = one GCP single-VM reproduction (or one AKS cell campaign).
launch_arm() {  # $1=paper $2=arm_id $3=seed $4="ENV1=1 ENV2=1"
  env $4 OPENRESEARCH_EXPERIMENT_ARM="$2" OPENRESEARCH_AB_PAIR_ID="ablate-p1-$1-$3" \
    python -m backend.cli campaign "$1" \
      --sandbox gcp --seed "$3" \
      --max-gpu-usd 15 --max-gpu-hours 10 \
      --pin-rubric "configs/papers/$1/rubric.json"   # hold rubric fixed across arms
}
# Fan out: 18 Phase-1 arms across N VMs (N = GPU quota). Stagger launches to avoid the
# 409 provision-collision; each writes runs/<arm>/ with experiment_arm + gpu_plan.json.
```

- **Isolation:** distinct `project_id`/run-dir per arm (the cohort loop already does this).
- **Concurrency cap** = GPU quota (e.g. 8-10 VMs). Queue the rest.
- **Stray-billing guard:** run `scripts/gcp_vm_audit.py` (Armaan's PR #12) on a timer;
  teardown stops-not-deletes, so orphans bill. Enforce the 28 h ceiling.
- **Collect:** each run's `final_report.json` + `rubric_evaluation.json` → a campaign
  manifest; feed to `python -m backend.agents.rlm.asha_authority_gate <manifest>`.

---

## 7. Confound controls (held byte-identical across arms)

Paper text · pinned rubric · seed (paired) · root+executor model · `--max-gpu-usd/hours`
· GPU type · base image · `OPENRESEARCH_EVIDENCE_GATE=ON` · finalize rail. **Only the one
ablated flag differs.** Randomize arm launch order (guards against provider price/model
drift over the campaign window). Grade blind to arm label. Log every flag into the run dir
(`env_snapshot.json`) so the ablation is auditable.

---

## 8. Tree-B sub-campaign (BLOCKED — prerequisite plan)

Freeze/branch/revive/kill cannot be ablated until cells emit receipts. Unblock =
**two changes**, then a scheduler-tree ablation becomes possible:

1. **Harness-forced checkpoint emission** — pre-scaffold `train_cell.py` wired to
   `cell_checkpoint.write_checkpoint(model, optimizer, lr_scheduler, rng, data_order)` at
   rung steps (or wire `OPENRESEARCH_GKE_SYNTH_CELL`). Without it, `build_raw_receipt` fails
   closed → the 0.203/0-receipt outcome repeats.
2. **Per-branch fail isolation** — today one branch's missing checkpoint raises
   `SchedulerRuntimeError` through `_cohort_loop` and aborts the whole campaign (branches
   2-4 never run). Fix so a bad branch is skipped, not fatal.

**Then** the Tree-B ablation (separate campaign, authority default-OFF, operator-gated):
- Arms: `serial` (authority off) vs `authority` (freeze+promote+revive+kill on). This is the
  ADAM A/B the `asha_authority_gate` was built for (≥3 pairs, σ≤0.02, kill-only-on-diverged
  audit, corroborated GPU-savings). Validate the full loop **on local transport first**
  (cherry-pick `gke-local-transport`) for $0 before any A100 spend.

---

## 9. Budget & sequencing

| Phase | Runs | ~GPU-hours | ~$ | Wall-clock @10 VMs |
|---|---:|---:|---:|---|
| 1 Screen | 18 | ~80 | ~$150 | ~1 day |
| 2 Confirm | ~63 | ~250 | ~$500 | ~2-3 days |
| 3 Interaction | 12 | ~60 | ~$120 | ~1 day |
| **Tree-A total** | **~93** | **~390** | **~$770** | **~1 week** |
| 8 Tree-B (after fix) | ~20 pairs | tbd | operator-gated | — |

Sequence: pin flags → Phase 1 screen → drop no-effect features → Phase 2 confirm on 3 papers
→ Phase 3 interaction + all-combined analysis → `asha_authority_gate` manifest → operator
review. Tree-B only after §8 lands + local-transport validation.

## 10. Deliverables

1. Per-feature Δ table (rubric + evidence + $/point), per paper, with grader-σ and CI.
2. Combined-vs-Σ-individual verdict (additive / redundant / synergistic).
3. Interaction map (at minimum BES×evidence).
4. A gate manifest + pass/fail per feature (operator-review, no auto-flip).
5. Recommended default-flip set (the cheapest subset that captures the gains).
