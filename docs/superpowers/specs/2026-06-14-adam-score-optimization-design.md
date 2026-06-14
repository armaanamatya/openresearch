# Adam (1412.6980) score-optimization plan — push 0.78 → ~0.90 and generalize the levers

**Status:** Phase 1 implemented (committed c8649800 + the optimization round); Phases 2–3 proposed. This doc is written FOR adversarial review — the goal is to find where it fails to reach ~0.90, not to defend it.

## 1. Grounded bottleneck analysis (the 0.7759, 24-leaf run)

Per-area headroom (every area → 0.95), highest-leverage first:

| Area | Score | Weight | Recoverable | Root cause |
|---|---|---|---|---|
| Result match | 0.510 | 0.15 | **+0.066** | optimizer orderings invert (mnist_mlp 0.0, imdb/cifar/vae 0.4) |
| Experiment execution | 0.625 | 0.18 | **+0.059** | same inversions + missing convergence-speed history |
| Eval protocol | 0.700 | 0.15 | +0.038 | per-epoch curves / metric-format gaps |
| Data fidelity | 0.895 | 0.12 | +0.007 | CIFAR GCN+ZCA whitening absent |
| Method fidelity | 0.940 | 0.35 | +0.004 | near-maxed |
| Artifact | 0.910 | 0.05 | +0.002 | minor |

**Single dominant root cause (~0.09):** the optimizer comparison does not reproduce the paper's claimed orderings because each optimizer is not run at *its own* tuned learning rate. This is a measured ARTIFACT (mnist_mlp adam final_train_loss 0.0087 vs sgd_momentum 1.6e-6), not an honest negative. The 0.4 leaves are partial matches ("supports half the claim, but AdaGrad inverts") — every ordering must hold for full credit.

**Meta-bottleneck:** a run that wedges (model-transport), dies on the wall-clock watchdog, or ships a stale partial grade scores ~0 regardless of training quality — so the score levers are worthless unless the run reliably completes AND the complete grid gets graded.

## 2. Phase 1 — implemented this session

- **P0 model-transport wedge fix** — `default_oauth_model()` pins an explicit model on every OAuth call; boot preflight fails fast (`failure_class=model_unavailable`); loud mid-run warning. (Fixes the 2026-06-14 Fable-5 wedge that zeroed both runs.)
- **LR tune-then-run protocol** (Adam hint): start from paper defaults (Adam α=1e-3/β1=0.9/β2=0.999; AdaGrad 1e-2; RMSProp 1e-3; SGD-Nesterov/momentum 1e-2; AdaDelta ρ=0.95; Adamax 2e-3) → SHORT 2-3 epoch per-(family,optimizer) sweep over {0.3×,1×,3×} → select by lowest final train loss → run the FULL comparison ONCE per optimizer at its selected lr. **The comparison grid stays the same size**; only a cheap tuning phase is prepended.
- **CIFAR ZCA whitening** directive (GCN per-image → ZCA fit on train covariance → apply train+test → record flag in provenance.json).
- **General fair-comparison** in `_AREA_REPAIR_HINTS["Result match…"]` (all comparative papers, reactive when the area is weak) + the honest-negative path (a fair sweep that still inverts is recorded truthfully, scored on the two-axis replication axis, not a fidelity penalty).
- **finalize_regrade** (already merged): re-grades the COMPLETE on-disk grid if it lands after the last verify and the recorded grade is below target (best-of-run MAX), across all finalize paths.

## 3. Phase 2 — proposed (push ~0.78 → ~0.90)

- **P2.1 Execution structure for tune-then-run.** Encode the tuning phase as a SEPARATE small `cells.json` sub-grid (family × optimizer × {0.3×,1×,3×}, 2-3 epochs) whose winner lr feeds the full comparison `cells.json`. Risk to kill: the agent naively cross-products lr × the full grid → wall-clock death (the exact thing Phase 1 tried to prevent in prose only).
- **P2.2 Convergence-speed evidence.** The paper's headline is SPEED. Emit per-epoch `history.<family>.<optimizer>` on a common x-axis with identical init; the result-match leaves credit "Adam reaches a given train-cost in fewer steps", not just final scalars.
- **P2.3 Eval-protocol format.** test-error-% at fixed checkpoints, the rubric's exact metric key names, mean over the paper's seed count.

## 4. Phase 3 — proposed (push ~0.90 → ~0.95)

- **P3.1 VAE families** (bias-correction + LR-sweep): the 0.4 vae leaves; show the bias-correction ELBO advantage and the high-lr instability the paper claims.
- **P3.2 Multi-seed averaging** (paper averages; also unlocks two-axis `contradicted` which requires ≥2 seeds). Cost-multiplies — gate behind a wall-clock check.

## 5. Open risks / where this could fail (review targets)

1. **tune-then-run is prose, not enforced.** Nothing MAKES the agent keep the grid bounded; a bad implementation still cross-products → wall-clock death. Should the harness *generate* the tuning sub-grid deterministically instead of trusting the hint?
2. **Does the LR fix actually move the leaves?** The grader scores result-match by claimed-vs-measured ordering. If the agent tunes lr but the orderings still don't perfectly match (e.g., AdaGrad vs SGD on IMDB), the leaf stays 0.4. Is 0.85 per-area realistic, or optimistic?
3. **Wall-clock math.** Adam = 6 families; tuning adds (≤3 lr × 6 optimizers × 2-3 epochs) per family BEFORE the full run. Does that fit the 14h cap with the cells route + finalize_regrade, or does it blow the budget on the VAE long-pole?
4. **All-papers blast radius.** The general `_AREA_REPAIR_HINTS` fair-comparison text fires for EVERY paper whose Result-match is weak — could it mislead a non-comparative paper (single-method) into a pointless lr sweep?
5. **Grading reliability.** finalize_regrade only re-grades if the grade is stale AND below target. If the agent verifies once on the COMPLETE grid (not partial), no regrade fires — is the best-of-run floor enough?
6. **Prediction honesty.** Is ~0.90 a real expectation or motivated reasoning from per-area-to-0.95 headroom that no single run will hit across all six areas at once?

## 5b. Codex adversarial-review outcome (2026-06-14) — recalibration

Codex tore down the plan; verified against the real run, the corrections stick:

- **Honest prediction is ~0.83, NOT 0.90.** Per-leaf (not per-area-to-0.95): Result-match cannot exceed ~0.79 while two ordering leaves stay at 0.4, and the run ALREADY tried LR-tuning on mnist_mlp_dropout with zero leaf movement. My 0.85-per-area was motivated reasoning.
- **VERIFIED CRITICAL BUG (fixed):** `leaf_triage._classify` matched "contradict"/"wrong ordering" but NOT "invert/inverting", and the word "dropout" hijacked classification to `protocol_gap`. The three real inversion leaves classified as `review`/`protocol_gap` — so the sharpened `result_quality` recourse **never fired on the inversions it was built for**. Fixed: added inversion morphology + result-match priority; verified all three now route to `result_quality`.
- **"tune-then-run" is unenforceable prose.** The cells route (`gpu_cell_runner`/`cell_scheduler`) runs one subprocess per `cells.json` cell with NO phase dependency or winner-propagation. A static manifest can only hardcode LRs or blind-cross-product → wall-clock death. **This is the core flaw: the LR protocol needs harness mechanism, not a hint.**
- **Tuning objective is wrong:** selecting LR by epoch-2/3 final loss rewards the early-fast/late-slow behavior the CIFAR claim needs to REVERSE. Must select by the paper's curve metric (time-to-threshold / AUC / epoch-45 loss), and never on the scored test evidence.
- **VERIFIED real gap:** `finalize_regrade.should_regrade` returns `already_meets_target` early → a maximization run never re-grades a grown grid past the floored target. Fix (deferred): regrade when evidence grew even at target; best-of-run MAX still adopts only-if-higher.
- **Likely-hallucinated specifics (Codex had 1 tool-use):** the "VAE 400-ReLU/20-dim vs 500-softplus/50-dim" numbers and the exact per-cell wall-clock seconds did NOT verify — `models.py` shows the MLP correctly at 1000-ReLU. Treat the VAE-arch and budget *arithmetic* as unconfirmed; the *general* concerns (budget fit, arch fidelity) stand.

**The ONE change Codex recommends (and I agree):** a **harness-owned staged-search manifest** — a structured `cells.json` schema with a bounded candidate phase, deterministic winner selection by the paper's curve metric, exactly one materialized full cell per (family, optimizer), and a **budget preflight** that refuses to launch unless `measured_throughput × total_cells + reserve ≤ remaining_wall_clock`. This converts the unenforceable prose into harness-guaranteed behavior and kills the wall-clock blocker.

**IMPLEMENTED 2026-06-14:** `backend/agents/rlm/staged_search.py` — pure core (`parse_search_spec` with hard candidate caps, `select_winner` by the declared metric, `materialize_full_cells`, `candidate_rate`/`estimate_full_seconds`/`affordable_full_cells` for the budget, `budget_feasible`) + `run_staged_search` orchestration (two `run_matrix` phases, measured candidate wall-clock, greedy cheapest-first reduction so a tight budget yields partial breadth not a wall-clock death). Wired into `primitives.py::_execute_cell_matrix` — **shape-gated** (no `search` key → legacy single-phase path byte-for-byte unchanged) + **local/docker only** (azure falls through). The Adam hint now emits the `search` schema. 25 module tests + 130-test cells-route regression green. v1 limitations (documented): the staged full cells skip the resume-fingerprint + capacity re-gate (the candidate phase proves VRAM); azure (k8s) staged path is future work.

## 6. Success check

A relaunch on the fixed harness + this hint produces: a complete 6-family grid (no wall-clock death), with per-optimizer selected lrs in provenance.json, result-match orderings matching the paper, and an overall ≥ 0.85 (target ~0.90) on the 24-leaf rubric — exceeding the 0.8308 record re-graded on the same 24-leaf instrument.
