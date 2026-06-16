# Grader-Fidelity & Harness Remediation — New-Session Handoff

**Date:** 2026-06-16 · **Branch:** `feat/azure-aks-gpu` · **Status:** design locked, implementation not started

**Purpose.** Self-contained kickoff for a fresh session to fix **every** issue found in the 2026-06-16 four-agent audit of orchestration, the execution harness, grading/self-improvement, and BES. The detailed design lives in `docs/superpowers/specs/2026-06-16-grader-fidelity-and-harness-remediation-design.md` (grilled to ground, Q1–Q6). This handoff repeats every issue inline so nothing is lost if you only read this file.

**How to use:** paste the *Kickoff Prompt* block below into a new session (or just tell the session "follow `docs/runbooks/2026-06-16-grader-fidelity-remediation-handoff.md`"). The *Issue Register* is the work-list; the *Rollout / Validation / Done* sections are the process.

---

## Kickoff Prompt (paste this)

```
Implement the 2026-06-16 Grader-Fidelity & Harness Remediation. Fix ALL issues in the register at
docs/runbooks/2026-06-16-grader-fidelity-remediation-handoff.md. Detailed design (LOCKED, grilled Q1–Q6):
docs/superpowers/specs/2026-06-16-grader-fidelity-and-harness-remediation-design.md — implement it, do not
re-design.

REPO: /home/sww35/openresearch. Branch off feat/azure-aks-gpu (e.g. feat/grader-fidelity).

THE INVARIANT YOU'RE BUYING: shipped score = median-of-N grade of the BEST artifact the run produced, at
its final evidence state — no MAX-over-noise anywhere. Root cause: the leaf grade is a non-deterministic LLM
call (no temp/seed) and the recovery stack compensates with an upward-biased best-of-run MAX.

HARD CONSTRAINTS (do not violate):
- Commit as lolout1, NO "Co-Authored-By: Claude" trailer.
- Flag-gate ALL new behavior; default = current behavior until the calibration gate passes, then flip
  default-ON. Flag =0/off must restore prior behavior byte-for-byte.
- 0-REGRESSION: full suite (.venv/bin/python -m pytest tests/, ~4850 green) after each workstream.
- Do NOT restore eval/exec/compile/input in the REPL safe-builtins patch.
- VERIFY every file:line anchor against the actual code before editing (some came from a fast audit).
- When you add a flag/primitive/SSE event, update CLAUDE.md + system_overview.md.
- Live validation runs cap at 14h (--max-wall-clock 50400).

ENV BLOCKER (fix before any claude-oauth e2e; unit tests don't need it): the global claude CLI is a broken
500-byte stub. Run: node /home/sww35/.nvm/versions/node/v24.12.0/lib/node_modules/@anthropic-ai/claude-code/install.cjs
(or reinstall @anthropic-ai/claude-code WITHOUT --ignore-scripts/--omit=optional), then verify `claude --version`.

ORDER:
0. Build the calibration harness FIRST (re-grade a fixed saved run dir K=5× through score_reproduction;
   record per-leaf + overall σ; extend data/calibration.json). Pure measurement; it validates every later step.
1. Workstream A (grader) in order A5+A1 → A6+A7 → A3+A4 → A2; re-run calibration after each, confirm σ drops.
2. C1 next (cheap/high-leverage), then B / rest of C / D / E. B,C,E items are independent — may fan out.
3. Q6: write scripts/regrade_backfill.py, regrade ALL historical scored runs to grader_version v1.

APPROACH (parallel + efficient, WITHOUT sacrificing quality): fan out independent-file items in ISOLATED
WORKTREES; merge sequentially through the full-suite 0-regression gate + calibration σ gate after EACH
merge. Keep the grader CORE (the leaf_scorer.py / report.py / run.py wiring: A5→A1→A6→A7→A3→A4→A2) a SINGLE
sequential owner — never two agents in the same hot file at once. Build new standalone modules in parallel
(calibration harness, deterministic_leaf_checker, grader digest, transport clients, evidence_key/champion
utils) with their own tests, THEN the core lane wires them in. Every correctness-critical fix gets an
independent adversarial review before merge.

DONE = every register item implemented+flag-gated+tested; calibration σ ≤ ~0.02 with A-flags on; suite green
0-regression; backfill run; committed as lolout1 on the feature branch; docs updated.

Start by reading the spec, then build the calibration harness.
```

---

## Issue Register (ALL issues)

Format: `ID — Problem (where) → Fix [flag · effort]`. Workstream A is the centerpiece; B–F are the rest of the audit. Effort: S/M/L.

### A. Grader fidelity (root cause)

- **A1 — Single noisy sample per leaf.** `_grade_batch` calls `llm_client.complete()` once at temp≈1.0 (`leaf_scorer.py:1646/1669`); a transient LLM/parse failure zeroes a whole 15-leaf batch (`:1678`). → Call `complete_samples(n=N)`, take the **per-leaf median**, `roll_up` once. Median (not mean) shrugs off the all-0.0 outlier. N=3. [`OPENRESEARCH_GRADER_SAMPLES` · S]
- **A2 — Mechanically-checkable leaves go to the noisy LLM.** ~half of ~20 leaves are checkable (execution/hparams, artifact existence, data prep, result-match trend) but are LLM-graded. → Route by `check_kind`: `deterministic:hparam` vs `provenance.json`, `deterministic:artifact` vs filesystem, `deterministic:numeric` vs `metrics.json` target; only method-fidelity/subtle-protocol (~7) stay LLM. Build on `run_invariant_checks` (`leaf_scorer.py:1214`), generalize from overall-cap to per-leaf. Additive: un-annotated leaves fall back to LLM. [`OPENRESEARCH_DETERMINISTIC_LEAVES` · L]
- **A3 — best-of-run MAX banks noise + ships score≠artifact.** `_best_recorded_rubric_score` (`report.py:637`) takes max over every verify event (0.712 over 0.694 same evidence, `report.py:980`); upward-biased. `finalize_regrade` gates on 120s mtime not content (`finalize_regrade.py:143`). → `evidence_key = hash(canonical metrics + scope)` (reuse `_compact_metrics_for_grader`); median-**within**-state; **strip the global max()**; demote the disk-reader to salvage-only (`median-at-latest-key`). [`OPENRESEARCH_EVIDENCE_FINGERPRINT` · M]
- **A4 — No within-run artifact champion.** `best_attempt.seed_reference_code` (`best_attempt.py:161`) only seeds a *reference* dir + advises the agent in prose; it's cross-attempt, not within-run. The floor faked anti-regression by banking a better *score* onto worse *code*. → Snapshot `code/` per verify keyed by `evidence_key` (reuse BES `_snapshot_code`); at finalize restore the highest-median-graded snapshot and ship *that* grade → score ≡ best artifact. [`OPENRESEARCH_CHAMPION_ARTIFACT` · M]
- **A5 — Grader rides the root client; SDK exposes no temp/seed.** `ctx.llm_client` shared (`run.py:186`) → a root/CLI wedge kills grading (the OmniZip mode). `ClaudeLlmClient.complete` (`rlm_query.py:584`) has no sampler control; `OpenAILlmClient` already does temp=0 but no seed/n. → Decouple grader transport; add optional `complete_samples(*, system, user, n, temperature, seed)` protocol method (mixin seq-fallback; OpenAI native n+seed+temp0; raw `AnthropicMessagesClient` temp0; SDK→median-of-N). Graceful per-backend degradation; Sonnet stays the grader. [`OPENRESEARCH_GRADER_BACKEND`/`_MODEL` · M]
- **A6 — Evidence the harness wrote is invisible to the grader.** Post-compaction scalar volume is unbounded → a wide grid hits a raw `[:96KB]` slice (32KB on the exception fallback) and trailing **headline** cells vanish. `_latest_metrics_path` ranks on `has_results` *truthiness* (a placeholder outranks measured data). → Count-based deterministic per-cell digest ({status, headline metric, n_epochs}); rank on `_per_model_has_measured_value`. [`OPENRESEARCH_GRADER_DIGEST` · S–M]
- **A7 — `degraded`/`meets_target`/EVIDENCE_GATE.** `degraded` auto-detect reads a stale `final_report.json` and caps a complete grid at 0.35× (`finalize_regrade.py:219` already dodges it twice). `meets_target=True` while `adjusted<target` on both All-CNN arms. `OPENRESEARCH_EVIDENCE_GATE` is documented as "the backstop" but has **zero `.py` refs**. → Make `degraded` an explicit required arg; compute `meets_target` from the authoritative post-rescore score; implement EVIDENCE_GATE (verify each cited `per_model` leaf exists on disk before writing) — share the A2 checker. [`OPENRESEARCH_EVIDENCE_GATE` · S each]

### B. Aggregation & lifecycle honesty

- **B1 — Hybrid Phase-2 clobbers a better Phase-1.** `run_pipeline_hybrid` (`controller.py:373`) restores Phase 1 only when Phase 2 score is `None`; a *worse-but-non-None* Phase 2 wins. → Apply the A3/A4 champion comparison at the hybrid layer. [S]
- **B2 — Graded-but-warned verify reads as "never verified."** `ctx.latest_rubric_score` set only on a non-failed verify (`binding.py:739`); a real score with any truthy `error` key never sets it → forced-iteration can ship a premature partial. → Set it from inside `verify_against_rubric` whenever a real `overall_score` exists. [S]

### C. Execution correctness

- **C1 — `--execution-mode max` silently half-dropped.** `resolve_experiment_timeout_s` reads `getattr(ctx, "execution_mode", …)` 3× but **`RunContext` has no such field** → the 6h cap applies only if also exported as `OPENRESEARCH_EXECUTION_MODE`; operators get 2h on long papers. → Add `execution_mode` to `RunContext` (thread at `run.py:1603`) or delete the dead branch. **Cheap, high-leverage — do early.** [S]
- **C2 — Mid-run orphan resources.** The per-primitive daemon-thread timeout (`binding.py:524`) returns `retryable` but the abandoned `run_experiment` thread keeps holding GPU/VRAM. → Process-group-kill the experiment subprocess on abandonment (reuse the cell runner's kill). [M]
- **C3 — OOM mitigation advisory, not enforced.** Harness only sets `OPENRESEARCH_CELL_BATCH_SCALE`; a non-cooperating `train_cell.py` OOMs identically 3×. OOM detection is stderr substring-match (`_OOM_SIGNATURES`) → non-matching message misclassifies terminal→repairable. → Inject a `set_per_process_memory_fraction` shim on retry; broaden classification. [M]
- **C4 — env_pin docker coverage gap.** `base_tag_for` claims cu121 for docker, but the `.pth`-following `LD_LIBRARY_PATH` prepend is in `LocalProcessBackend` only → docker exec misses it. → Route docker exec through the same prepend, or stop claiming cu121 for docker. [S]
- **C5 — Explicit duplicate-triple last-writer-wins (silent).** Two cells with identical explicit `model_key/env/baseline` → one leaf silently overwrites the other (`cell_matrix.py:230`); only *derived* dups are id-suffixed. → id-suffix + warn on explicit dups too. [S]
- **C6 — Runpod/azure rough edges.** Runpod has **no stall detection** (the 2026-06-08 redesign is local-scoped); every runpod run pays a **discarded** local `docker build` and hard-requires a daemon; the azure k8s cell branch (`primitives.py:5252`) is unreachable (the `("local","docker")` gate at `:5645` excludes azure first). → Add a remote heartbeat (or document loudly); short-circuit `build_environment` under runpod; add `"azure"` to the route gate or remove the dead branch. [runpod M, rest S]

### D. BES & A/B validity

- **D1 — `ab_compare.py` is a reporter, not a validator.** Accepts unstamped-as-control (`:222`), picks `latest` not `best` (`:170`), never asserts rubric/scope/seed equality. → Refuse a Δ unless both arms stamped + `rubric_tree.json` sha256 matches + scope matches; default `select=best`; `--require-stamped`. **Unblocks the BES bar.** [`OPENRESEARCH_REQUIRE_STAMPED_AB` · S]
- **D2 — BES SELECT is runtime-blind.** SELECT is code-only static grade (`degraded=False`, no metrics) — blind to OOM/crash/divergence; an all-failed pool still "selects"; tiny spreads (<0.07) are coin-flips. → Wire `execution_smoke`/`preflight_smoke` into the SELECT loop; on spread < σ_grader break ties deterministically (import-smoke/AST-completeness); on all-failed emit `degenerate_pool` + fall through to single-shot repair. [M]
- **D3 — BES posture.** 1 clean pair (+0.085 < within-paper variance), N always 2 (weakest useful), repo's ≥3-paired-SDAR bar unmet. → **Keep BES default-OFF** until A (noise) + D1 + D2 land and a proper ≥3-seed paired SDAR run clears the bar. [posture]

### E. Integration & posture

- **E1 — Self-improvement loops stranded off-branch.** `context_map.py` + `lesson_distiller.py` (PEEK-lite context-map + MUSE-lite negative-lessons) exist on `m2`/`m4`/`m9`/`origin/bes`/`origin/feat/rlm-wedge-hardening` but **not on `feat/azure-aks-gpu`** → `OPENRESEARCH_NEGATIVE_LESSONS=1`/`OPENRESEARCH_CONTEXT_MAP=on` are silent no-ops here; **nothing learns across runs.** Flag prefix drifted (`origin/bes` ships `OPENRESEARCH_*`). Audit exists: `docs/audits/2026-06-07-bes-doc-alignment-audit.md` (on `m2`). → Merge both modules in under ONE canonical flag prefix (recommended — biggest missing self-improvement lever), or strike the CLAUDE.md paragraphs. [M]
- **E2 — Proven guards default-OFF.** `best_attempt`, `dead_training_guard`, `execution_smoke`, `preflight_smoke` are real/tested/credited with recoveries but OFF, while unproven BES got an A/B harness. → Flip ON (or one-line rationale per flag). [S]
- **E3 — Loud-fail-soft sweep.** "Fail-soft everywhere" = silent degradation (cells→monolithic fallthrough, env_pin `|| true`, duplicate-triple overwrite, graded-but-warned verify). → Every degrade emits a coded `run_warning` + appends to `degradations_taken[]` in the report (the `cells_manifest_restored` pattern, made universal). [M]
- **E4 — Doc/code reconciliation.** `OPENRESEARCH_EVIDENCE_GATE` (A7), "12 primitives" (actually 16, `primitives.py:7182`), "azure = NotImplementedError stub" (false — returns a populated `GpuCapacity`). → Reconcile CLAUDE.md to checkout-vs-union reality. [S]

### F. Long-tail (smaller, but in scope of "all")

- **F1 — Per-iteration cost is fiction.** `_compute_cost_summary` (`run.py:1240`) equal-slices a flat ledger and surfaces `usd_this_iter`/`p50` as if measured. → Tag each `CostLedgerEntry` with `ctx.current_iteration`; aggregate by tag. [S]
- **F2 — No per-iteration sub-RLM fan-out cap.** `rlm_query`/`llm_query` bounded only by `max_depth=2` + global budget; an un-batched loop over slices is caught too late. → Per-iteration sub-call counter + soft refusal past a threshold. [`OPENRESEARCH_MAX_SUBCALLS_PER_ITER` · S]
- **F3 — `roll_up` None-sentinel typed as float.** `# type: ignore` overloads `None` as `float` (`leaf_scorer.py:99`) — latent crash/mis-weight for future callers. → Dedicated sentinel or `(score, eligible)` tuples. [S]
- **F4 — Dataset-synonym whack-a-mole.** Hand-maintained alias map (`leaf_scorer.py:530`); the generalized token-overlap tier it replaced *gamed the score* (caught only by a re-grade). → Prefer operator inclusion-scope (`_detect_out_of_inclusion_scope_leaves`) over fuzzy matching; reserve the map for un-inferrable synonyms. [S]
- **F5 — verifier/grader stamp wrong under `ACCELERATOR_SCOPE=all`.** `_finalize` stamps both = `llm_model` unconditionally (`run.py:2192`). → Stamp the actual grader transport. [S]
- **F6 — Two-experiment-guard reset target.** Boundary reset acts on `ctx._forced_iteration_policy` while the interceptor reads the thread-local stack (`forced_iteration.py`/`sse_bridge.py:389`) — latent divergence. → Reset via `forced_iteration._current_policy()`. [S]
- **F7 — Surface grader-noise spread.** The shipped score is a median over draws; record min/max/n_grades in the rubric block so a stable 0.75 is distinguishable from a floored one. [S]

---

## Rollout order (each step independently flag-gated + validated)

1. **A5 + A1** — universal denoiser, no producer deps (eat the transient 3× grader cost).
2. **A6 + A7** — make the grader see/treat evidence faithfully.
3. **A3 + A4** — retire the MAX floor honestly (fingerprint + champion-artifact).
4. **A2** — deterministic routing; brings cost back to flat; needs reliable `provenance.json` (C6) + rubric-gen annotations.
5. **C1** early (cheap), then **D1 + D2** (now BES is measurable → run the ≥3-seed paired SDAR A/B).
6. **B / rest of C / E / F** in parallel (independent of A).

## Parallelization plan (efficient, quality-preserved)

The rollout above is the **dependency truth**; this is the **execution strategy**. Goal: maximize concurrency without letting two agents fight over a hot file or letting an unreviewed fix reach the integration branch.

**Concurrency lanes** (run lanes in parallel; respect intra-lane order):

| Lane | Items | Concurrency | Owns these files |
|---|---|---|---|
| **0 · Calibration** (blocking) | calibration harness | 1 agent, finishes first | `scripts/calibrate_grader.py`, `data/calibration.json` |
| **A-core** (sequential spine) | A5→A1→A6→A7→A3→A4→A2 *wiring* | **1 owner, never parallel** | `leaf_scorer.py`, `report.py`, `run.py`, `finalize_regrade.py`, `binding.py` |
| **A-modules** (parallel) | new standalone files for A | N agents, isolated | `deterministic_leaf_checker.py`, grader-digest util, `complete_samples` mixin + `AnthropicMessagesClient`, `evidence_key`/champion-artifact utils |
| **Harness** (parallel) | C1, C4, C5, C6, F5, F6 | N agents, **worktree-isolated** | `context.py`, `primitives.py`, `cell_matrix.py`, `local_process.py`, `runpod_backend.py`, `gpu_capacity.py`, `forced_iteration.py` |
| **Lifecycle** (parallel) | B1, B2, F1 | N agents, worktree-isolated | `controller.py` (hybrid), `binding.py`†, `run.py`† |
| **BES/AB** (parallel; D2 after A) | D1, D2, D3 | N agents | `scripts/ab_compare.py`, `bes_rlm.py`, `candidates.py` |
| **Integration** (own worktree) | E1 git merge, E2, E3, E4, F3, F4, F7 | E1 isolated; rest small | merge from `origin/bes`; `config.py`/defaults; docs |

† `binding.py`/`run.py` overlap A-core and Lifecycle — serialize those edits behind A-core (same-owner) or stage them as separate, non-overlapping hunks.

**Quality rails (non-negotiable — this is the "without sacrificing quality" half):**
1. **Worktree isolation** for every parallel agent (`isolation: worktree`) → no shared-file clobber; merge back one at a time.
2. **Gate every merge** to the integration branch on the **full suite (0-regression)** AND, for any A-step, the **calibration σ gate** — not just at the end.
3. **Single owner per hot file.** `leaf_scorer.py`, `report.py`, `run.py`, `primitives.py` are correctness spines: exactly one lane touches each at a time. New logic lands as standalone modules first (A-modules lane), then the owning lane wires it in.
4. **Independent adversarial review** of every correctness-critical fix (all of A, plus A3/A4 aggregation and D2 SELECT) before merge — a second agent tries to *break* it (find the evidence-loss / bias / regression the author missed).
5. **Flag-gated by construction** → a half-finished parallel item is inert (default-off) until its lane completes and validates.
6. **Order the unavoidable couplings:** A5 before A1 (needs `complete_samples`); A3 before A4 (shared `evidence_key`); A1 before A2 (routes residual to median-of-N); C6's provenance reliability before A2's deterministic path flips default-ON.

**Net:** ~5 lanes run concurrently; the grader correctness spine stays single-threaded and reviewed; nothing reaches `main`-bound history without passing both gates.

## Validation gate (how to *prove* it worked) — the Q6 decisions

- **Calibration harness (build first):** re-grade a fixed saved run dir **K=5×** through `score_reproduction`; report per-leaf + overall **σ before/after** each A-step. Extend `data/calibration.json`. Promotion criterion: overall σ ≤ ~0.02 before any A-flag flips default-ON.
- **Regrade everything (decided):** `scripts/regrade_backfill.py` regrades all ~29 historical scored runs to `grader_version: v1` for a uniform leaderboard; stamp `grader_version`/`grader_samples`/`grader_temperature` on every rubric block; preserve `v0` in an archive sidecar (don't mutate in place).
- **A/B sanity:** after A lands, re-run the All-CNN pair; the +0.085 must survive *with a stated σ band* — the first honest BES data point.

## Definition of done

- Every register item (A–F) implemented + flag-gated + unit-tested.
- Calibration harness shows overall grader σ ≤ ~0.02 with A-flags on.
- Full suite green, 0-regression (~4850 baseline).
- `regrade_backfill.py` written and run; leaderboard uniform at `v1`.
- Each logical unit committed as **lolout1** (no Claude trailer) on the feature branch; CLAUDE.md + system_overview.md updated.

## References

- Design spec (detail): `docs/superpowers/specs/2026-06-16-grader-fidelity-and-harness-remediation-design.md`
- Memories: `grader-fidelity-design`, `feat-azure-aks-gpu-merge-debt`; related `grading-evidence-budget-fixes`, `model-wedge-fable5-fix`, `scoring-fairness-spec`, `recurring-failures-remediation`
- Prior alignment audit (on `m2`): `docs/audits/2026-06-07-bes-doc-alignment-audit.md`
