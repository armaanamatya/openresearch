# Grader-Fidelity & Harness Remediation — Design Spec

**Date:** 2026-06-16
**Status:** DESIGN LOCKED (grader-noise core grilled Q1–Q6; remaining workstreams scoped from the 2026-06-16 four-agent audit)
**Branch context:** authored on `feat/azure-aks-gpu`; several fixes interact with merge-debt from `origin/bes` / `m2` / `feat/rlm-wedge-hardening` (see Workstream E).

---

## 0. Thesis

A four-agent code-grounded audit of orchestration, the execution harness, grading/self-improvement, and BES converged on a single root weakness: **the grade is a non-deterministic LLM call, and almost every other mechanism is compensating for that.**

- The leaf score comes from one line — `llm_client.complete(system, user)` (`leaf_scorer.py:1669`) — with **no temperature/seed plumbed anywhere** (the protocol is `complete(*, system, user) -> str`; grep across all 5 client impls confirms zero sampler control). Default temp ≈1.0 on the OAuth path → **~2.5% drift on identical evidence** (`report.py:980`: All-CNN v3 verify#1 0.712 vs verify#2 0.694) and **±0.09–0.18 run-to-run swings**.
- The recovery stack — best-of-run floor (applied 3×), `finalize_rescore`, `finalize_regrade` — exists mostly to paper over that noise. Worse, **best-of-run is MAX-over-noisy-grades, which is upward-biased** (it banks the luckiest draw) and can ship a *score* detached from the *artifact*.
- BES is **unprovable** because its +0.085 lone signal and 0.0079 SELECT margins are *inside* the noise band.

**Fix the grade at root and a pile of machinery becomes deletable, BES becomes measurable, and the leaderboard becomes trustworthy.** That is Workstream A — the centerpiece. Workstreams B–E are the correctness, honesty, and integration debt the same audit surfaced.

**Invariant we are buying:** *the shipped score is a median-of-N grade of the best artifact the run actually produced, at that artifact's final evidence state — no MAX-over-noise survives anywhere.*

---

## Workstream A — Grader fidelity (centerpiece; design grilled Q1–Q6)

### A1. Median-of-N at the leaf (denoise the signal) — Q2/Q4

**Problem:** every grade is a single temp≈1.0 sample; one transient LLM/parse failure zeroes a whole 15-leaf batch (`leaf_scorer.py:1678`).

**Fix:** in `_grade_batch` (`leaf_scorer.py:1646`), call `complete_samples(n=N)` and take the **per-leaf median** of the N scores, then `roll_up` **once**. Per-*leaf* median is the right granularity; roll-up stays deterministic.

- **Median, not mean** — deliberately. The all-`0.0`-on-failure fallback is an outlier a median over N≥3 shrugs off; this *also fixes* the "one failure zeroes 15 leaves" bug for free.
- **N=3 default**, flag `REPROLAB_GRADER_SAMPLES` (`=1` → today's exact behavior). Odd for a clean median; √3≈1.7× variance cut.
- **Concurrency:** reuse the existing ≤8 ThreadPool; submit the N copies as extra futures.
- **Cost:** transient 3× grader calls; **cost-neutral at steady state** once A2 peels ~12 mechanical leaves off the LLM (≈7 judgment leaves × 3 ≈ today's 20 × 1).

**Effort:** S. **Flag:** `REPROLAB_GRADER_SAMPLES=3`.

### A2. Deterministic-by-construction routing (shrink the noisy surface) — Q2

**Problem:** the rubric census (3 papers, ~20–24 leaves each) shows ~half the leaves are **mechanically checkable** but still go to the noisy LLM:

| Category | ~/paper | Checkability | Ground truth |
|---|---|---|---|
| Experiment execution & reproducibility | 3–4 | mechanical (epochs, momentum, wd, LR sched) | `provenance.json` |
| Artifact completeness & provenance | 2–3 | mechanical (scripts/arch defs exist) | filesystem |
| Data & preprocessing fidelity | 3 | mostly mechanical (batch=16, whitening) | config/provenance |
| Result match vs targets | 4 | numeric compare *if claim is structured* | `metrics.json` vs target |
| Evaluation protocol & metric correctness | 3–4 | mixed | code + metric keys |
| **Method & code fidelity** (largest weight) | **5** | **judgment** / per-paper invariant | code semantics |

**Fix:** route each leaf to the cheapest sufficient grader. At rubric-gen, attach an optional `check_kind` + structured assertion to each leaf:
- `deterministic:hparam` → compare `provenance.json` fields vs `{field, op, value}`.
- `deterministic:artifact` → file/glob existence vs expected patterns.
- `deterministic:numeric` → `metrics.json[metric_key]` vs `{target, tolerance, direction}` (trend, not magnitude).
- `judgment` (method-fidelity + subtle protocol) → A1 median-of-N LLM.

A new pure-Python `deterministic_leaf_checker` (built on the existing `run_invariant_checks`/`_apply_invariant_gate`, `leaf_scorer.py:1214–1347` — generalized from a blunt *overall cap* into *per-leaf scores*) evaluates the deterministic leaves; the router in `score_reproduction` merges deterministic + LLM leaves before roll-up.

**Modular/additive guarantee:** a leaf with **no** structured assertion (old rubric) falls back to LLM median-of-N. Deterministic routing only *adds* where annotations exist; it can never break an un-annotated rubric. Producer dependency: rubric-gen emits the assertions; reliable `provenance.json` (Workstream C / scoring-fairness D2).

**Effort:** L (checker M + rubric-gen annotation M). **Flag:** `REPROLAB_DETERMINISTIC_LEAVES=1`.

### A3. Evidence fingerprint + median-within-state; retire the MAX floor — Q3

**Problem:** `_best_recorded_rubric_score` (`report.py:637`) takes **max** over *every* `rubric_score` event — banking same-evidence noise (0.712 over 0.694) and shipping a score that may not match the shipped code. `finalize_regrade` gates on a 120s **mtime** proxy (`finalize_regrade.py:143`), not content.

**Fix:** define `evidence_key = hash(canonical measured metrics + scope/leaf-set)`, reusing `_compact_metrics_for_grader` (series→{len,first,last,min,max}) as the canonical form (stable against epoch churn, no new machinery). Stamp it on every `rubric_score` event and on `rubric_evaluation.json`. Then the three jobs the floor conflated split cleanly:

1. **Same-evidence noise → median, not max.** Group grades by `evidence_key`; the estimate per key is the median (with A1, a key usually already holds one robust estimate).
2. **Evidence growth → `finalize_regrade` keyed on a *new* `evidence_key`** (genuinely ungraded growth), replacing the mtime heuristic.
3. **Repair regression → champion-*artifact*, not champion-*score*** (A4).

`_apply_best_of_run_floor`'s global `max()` is **stripped**. The disk-reader survives **demoted**: salvage/fallback **only** (run killed, or fresh grading impossible), returning `median-at-latest-evidence-key`, never global max.

**Effort:** M. **Flag:** `REPROLAB_EVIDENCE_FINGERPRINT=1` (gates the new aggregation; off → legacy floor).

### A4. Champion-artifact aggregation (honest anti-regression) — Q3

**Problem:** `best_attempt.seed_reference_code` (`best_attempt.py:161`) copies the best prior attempt's code to a *reference* dir `code/_best_attempt/` and **advises the agent in prose** to restore regressed leaves — it does **not** deterministically restore, and it's **cross-attempt**, not within-run. The within-run "a repair regressed the code" case has **no artifact home** today; the floor faked it by banking the better *score* onto worse *code*.

**Fix:** snapshot `code/` at each `verify_against_rubric`, content-addressed by `evidence_key` (reuse BES `_snapshot_code` + `_SNAPSHOT_IGNORE` — heavy artifacts excluded, so it's just source). At finalize, restore the snapshot whose **median-of-N grade is highest** and ship *that* grade → `score ≡ best artifact actually produced`. Unifies with BES (parallel-candidate champion) and best_attempt (cross-attempt champion) under one snapshot-grade-restore primitive.

**Effort:** M. **Flag:** `REPROLAB_CHAMPION_ARTIFACT=1`.

### A5. Decoupled, sampler-capable grader transport — Q5

**Problem:** the grader rides the *root model's* `ctx.llm_client` (`run.py:186`) — so a root/CLI wedge takes grading down with it (the OmniZip failure). The OAuth SDK path exposes no temp/seed (`rlm_query.py:584`); the OpenAI path already sets `temperature=0` (`openai_client.py:~115`) but no `seed`/`n`.

**Fix:**
1. **Decouple** grader transport via `REPROLAB_GRADER_BACKEND`/`REPROLAB_GRADER_MODEL` (default = Sonnet, honoring the CLAUDE.md "grader stays Sonnet-quality" rule). Robustness: grading survives a root/CLI wedge.
2. **One optional protocol method, backwards-compatible:**
   ```python
   # mixin default — every existing client works unchanged
   def complete_samples(self, *, system, user, n=1, temperature=None, seed=None) -> list[str]:
       return [self.complete(system=system, user=user) for _ in range(n)]
   ```
   - `OpenAILlmClient` overrides → native `n` + `seed` + `temperature=0` in **one round-trip** (scalable).
   - new raw `AnthropicMessagesClient` overrides → `temperature=0` + N calls (same Sonnet, now pinned).
   - `ClaudeLlmClient` (SDK) keeps the sequential fallback → median-of-N is its denoiser.
3. **Graceful degradation:** grader always calls `complete_samples(n=N, temperature=0, seed=fixed)`; each backend honors what it can; median-of-N is the universal floor. OAuth-only → SDK Sonnet + median-of-3 ($0, rate-bounded); `ANTHROPIC_API_KEY` → raw temp=0 Sonnet; OpenAI-root → temp=0 + seed + native-n (near-deterministic, cheapest).

**Effort:** M. **Flags:** `REPROLAB_GRADER_BACKEND`, `REPROLAB_GRADER_MODEL`.

### A6. Evidence *visibility* (the grader must see what the harness wrote) — folds in Tier-2 #4

**Problem:** the harness reliably *preserves* evidence on disk, but the grader can't always *see* it: `_compact_metrics_for_grader` fixed 14 models, but post-compaction scalar volume is unbounded — a wider grid hits a raw `[:96KB]` slice (32KB on the exception fallback), dropping trailing **headline** cells. Also `_latest_metrics_path` ranks on `has_results` *truthiness*, so a placeholder `per_model:{m:{}}` can outrank genuinely-measured older data.

**Fix:**
- Replace the byte-slice with a **count-based deterministic per-cell digest** (every cell: `{status, headline metric, n_epochs}`) so no cell silently vanishes from the grader prompt regardless of grid width.
- `_latest_metrics_path` ranks on `_per_model_has_measured_value`, not truthiness.

**Effort:** S–M. **Flag:** `REPROLAB_GRADER_DIGEST=1`.

### A7. `degraded` foot-gun + `meets_target` bug + EVIDENCE_GATE reconciliation

- **`degraded` auto-detect** reads a possibly-stale `final_report.json` and can cap a *complete* grid at 0.35× (the regrade paths already pass `degraded=False` twice to dodge it, `finalize_regrade.py:219`). **Fix:** make `degraded` an explicit required arg; remove the `None` auto-detect default.
- **`meets_target` bug:** both All-CNN arms show `meets_target=True` while `adjusted < target`. **Fix:** compute `meets_target` from the authoritative post-rescore score consistently.
- **`REPROLAB_EVIDENCE_GATE`** is documented in CLAUDE.md as "the backstop" but has **zero `.py` references** (confirmed). **Fix:** either implement it (verify every cited `per_model` leaf exists on disk before the report is written) or strike the doc claim and document the 0.35 degraded cap as the actual (weaker) backstop. **Recommend: implement** — it's the honest backstop the deterministic checker (A2) can share.

**Effort:** S (each).

---

## Workstream B — Aggregation & lifecycle honesty (beyond the grader)

### B1. Hybrid Phase-1/Phase-2 best-of guard (orchestration W6)
`run_pipeline_hybrid` (`controller.py:373`) restores Phase 1 only when Phase 2's score is `None` — a *worse-but-non-None* Phase 2 silently discards a better Phase 1. **Fix:** compare scores and restore Phase 1 when Phase 2 is worse (apply the A3/A4 champion logic at the hybrid layer). **Effort:** S.

### B2. Rubric→policy seam: graded-but-warned reads as "never verified" (orchestration W2)
`ctx.latest_rubric_score` is set only by `binding._emit_supplemental` on a *non-failed* verify (`binding.py:739`); a verify that computes a real score but carries any truthy `error` key never sets it → the forced-iteration guard sees "never verified" and can ship a premature partial. **Fix:** set `ctx.latest_rubric_score` from inside `verify_against_rubric` whenever a real `overall_score` exists, independent of advisory warnings. **Effort:** S.

---

## Workstream C — Execution correctness (harness)

### C1. `--execution-mode max` is silently half-dropped (orchestration W1) — Tier-1 #2
`resolve_experiment_timeout_s` reads `getattr(ctx, "execution_mode", None)` 3× but **`RunContext` has no `execution_mode` field** (confirmed) → the 6h max cap only applies if *also* exported as `REPROLAB_EXECUTION_MODE`. Operators get 2h on long papers and then the `partial_timeout` salvage everyone built. **Fix:** add `execution_mode` to `RunContext` and thread `execution_profile.execution_mode` at `run.py:1603` (or delete the dead ctx branch and make the env var the single source). **Effort:** S. **High leverage.**

### C2. Mid-run orphan-resource guard (orchestration W7)
The per-primitive daemon-thread timeout (`binding.py:524`) returns `retryable` to the caller but the abandoned `run_experiment` thread keeps holding GPU/VRAM. **Fix:** signal the experiment subprocess's process-group for termination on abandonment (reuse the cell runner's process-group-kill). **Effort:** M.

### C3. OOM mitigation: advisory → enforced
The harness only *sets* `REPROLAB_CELL_BATCH_SCALE`; a non-cooperating `train_cell.py` OOMs identically 3×. OOM detection is stderr substring-match (`_OOM_SIGNATURES`) — a non-matching message misclassifies terminal→repairable. **Fix:** inject a `torch.cuda.set_per_process_memory_fraction` shim on retry; broaden OOM classification. **Effort:** M.

### C4. env_pin docker coverage gap
`base_tag_for` claims `cu121` for docker, but the `.pth`-following `LD_LIBRARY_PATH` prepend lives only in `LocalProcessBackend` — docker exec doesn't get it. **Fix:** route docker exec through the same prepend, or stop claiming cu121 for docker so strip + lib-fix agree on scope. **Effort:** S.

### C5. Explicit duplicate-triple last-writer-wins (silent)
Two cells with identical explicit `model_key/env/baseline` → one leaf silently overwrites the other (`cell_matrix.py:230`); only *derived* dups are id-suffixed. **Fix:** id-suffix + warn on explicit dups too. **Effort:** S.

### C6. Runpod observability + wasted-build + azure routing
- Runpod has **no stall detection** (the 2026-06-08 reliability redesign is local-scoped) — add a remote heartbeat or document loudly.
- Every runpod run pays a **discarded** local `docker build` and hard-requires a daemon — short-circuit `build_environment` under runpod.
- The **azure k8s cell branch** (`primitives.py:5252`) is unreachable from the normal `run_experiment` entry (the `("local","docker")` gate at `:5645` excludes azure first) — add `"azure"` to the gate or remove the dead branch.

**Effort:** runpod-stall M, others S.

---

## Workstream D — BES & A/B validity

### D1. Make `ab_compare.py` a validator, not a reporter — Tier-1 #3
Today it accepts unstamped runs as control (`:222`), picks `latest` not `best` (`:170`), and **never asserts rubric/scope/seed equality**. **Fix:** refuse to emit a Δ unless both arms are explicitly stamped, `rubric_tree.json` sha256 matches, and scope matches; default `select=best`; add `--require-stamped` for the launch gate. **Effort:** S. **Unblocks the BES bar.**

### D2. Smoke-gate the BES SELECT signal — Tier-2 #6
SELECT is code-cosmetics only (`degraded=False`, no metrics) — blind to the runtime axis (torch-repin, VAE-crash) where this repo's failures live. **Fix:** wire `execution_smoke`/`preflight_smoke` (already exist, default-OFF) into the BES loop so a statically-faithful-but-non-runnable candidate can't win; on a top-2 spread < measured σ_grader, break ties on a deterministic signal (import-smoke, AST-completeness); on an all-failed pool emit `degenerate_pool` and fall through to single-shot repair. **Effort:** M.

### D3. BES posture
With 1 clean pair (+0.085 < within-paper variance) and the repo's own ≥3-paired-SDAR bar unmet, **keep BES default-OFF** until A (noise) + D1 (validator) + D2 (smoke SELECT) land and a proper ≥3-seed paired SDAR run clears the bar. The grader fix is what makes "does BES help?" answerable at all.

---

## Workstream E — Integration & posture

### E1. Resolve the negative-lessons / context-map merge debt — Tier-3 #7
**Confirmed:** `context_map.py` + `lesson_distiller.py` exist on `m2`/`m4`/`m9`/`origin/bes`/`origin/feat/rlm-wedge-hardening` but **not on `feat/azure-aks-gpu`**; CLAUDE.md here documents `REPROLAB_*` flags while `origin/bes` ships them as **`OPENRESEARCH_*`** (prefix drifted). An audit already exists: `docs/audits/2026-06-07-bes-doc-alignment-audit.md` (on `m2`). Today `REPROLAB_NEGATIVE_LESSONS=1` is a **silent no-op** on this branch → operators believe a cross-run learning loop is active when nothing learns across runs. **Fix:** merge the two modules into this branch under **one canonical flag prefix**, or strike the CLAUDE.md paragraphs. **Recommend merge** — cross-run failure memory is the single biggest *missing* self-improvement lever. **Effort:** M (merge + flag reconciliation).

### E2. Flip proven default-OFF guards ON — Tier-3 #8
`best_attempt`, `dead_training_guard`, `execution_smoke`, `preflight_smoke` are real, tested, and credited with recoveries — yet default-OFF, while unproven BES got an A/B harness. **Fix:** flip them ON (or add a one-line rationale per flag). `dead_training_guard`'s false-positive design is conservative (4 simultaneous conditions); the cost it prevents (~19min/dead cell, fake-ok scoring) is high. **Effort:** S.

### E3. Loud-fail-soft sweep (Theme B) — Tier-3 #9
"Fail-soft everywhere" = silent degradation everywhere (cells→monolithic fallthrough, env_pin `|| true`, duplicate-triple overwrite, graded-but-warned verify). **Fix:** every degrade emits a coded `run_warning` and appends to a `degradations_taken[]` list in the report (the pattern already exists for `cells_manifest_restored` — make it universal). **Effort:** M.

### E4. Doc/code reconciliation
Beyond E1: `REPROLAB_EVIDENCE_GATE` (A7), "12 primitives" (actually 16, `primitives.py:7182`), the "azure = NotImplementedError stub" claim (code returns a populated `GpuCapacity`). **Fix:** reconcile CLAUDE.md to the union-vs-checkout reality, or annotate per-branch. **Effort:** S.

---

## Consolidated flag table

| Flag | Default | Workstream | Effect |
|---|---|---|---|
| `REPROLAB_GRADER_SAMPLES` | `3` (was `1`) | A1 | median-of-N per leaf |
| `REPROLAB_DETERMINISTIC_LEAVES` | `1` | A2 | route mechanical leaves to Python checker |
| `REPROLAB_EVIDENCE_FINGERPRINT` | `1` | A3 | median-within-state; strip MAX floor |
| `REPROLAB_CHAMPION_ARTIFACT` | `1` | A4 | ship best artifact, graded fresh |
| `REPROLAB_GRADER_BACKEND` / `_MODEL` | Sonnet | A5 | decoupled, sampler-capable transport |
| `REPROLAB_GRADER_DIGEST` | `1` | A6 | count-based per-cell grader digest |
| `REPROLAB_EVIDENCE_GATE` | `1` | A7 | verify cited leaves exist on disk |
| `REPROLAB_REQUIRE_STAMPED_AB` | `1` | D1 | ab_compare validator mode |
| (flip ON) `best_attempt`, `dead_training_guard`, `execution_smoke`, `preflight_smoke` | ON | E2 | proven guards on by default |

All new behavior flag-gated; `=0`/`off` restores prior behavior byte-for-byte. Default-ON only after the validation gate (below) passes.

---

## Rollout sequence (each step independently flag-gated and validated)

1. **A5 transport + A1 median-of-N** — universal denoiser, no producer deps. Ship first. *(Eat the transient 3× grader cost.)*
2. **A6 visibility + A7 honesty** — make the grader see/treat evidence faithfully.
3. **A3 fingerprint + A4 champion-artifact** — retire the MAX floor honestly.
4. **A2 deterministic routing** — peels mechanical leaves off the LLM; brings cost back to flat; needs reliable `provenance.json` (C6) + rubric-gen annotations.
5. **D1 + D2** — now BES is measurable; run the ≥3-seed paired SDAR A/B.
6. **B/C/E** correctness + integration in parallel (independent of A).

---

## Validation gate (how we *prove* the noise dropped) — Q6

- **Calibration harness:** re-grade a fixed run dir **K=5** times through `score_reproduction`; report per-leaf and overall **σ before/after** each A-step. Extend the existing `data/calibration.json`. Promotion criterion: overall σ drops below a target band (e.g. ≤0.02) before any A-flag flips default-ON.
- **Regrade-everything (Q6 decision):** `scripts/regrade_backfill.py` re-grades all ~29 historical scored runs under `grader_version: v1` for a **uniform leaderboard**. Every rubric block stamps `grader_version` / `grader_samples` / `grader_temperature` for provenance. Historical `v0` scores are preserved in an archive sidecar, not mutated in place.
- **A/B sanity:** after A lands, re-run the All-CNN pair; the +0.085 must survive (or not) *with a stated σ band* — that's the first honest BES data point.

---

## Test plan

- **Unit (deterministic, no LLM):** `evidence_key` canonicalization stability (epoch churn → same key); `deterministic_leaf_checker` against on-disk `provenance.json`/`metrics.json` fixtures; median-of-N aggregation incl. the all-`0.0`-outlier case; `complete_samples` mixin fallback == N× `complete`; champion-artifact restore picks the highest median; demoted floor returns `median-at-latest-key` not global max.
- **Integration:** A-pipeline on a saved All-CNN run dir reproduces a stable score across 5 invocations (σ ≤ target); `ab_compare` rejects unstamped/rubric-mismatched pairs.
- **Regression:** every flag `=0` reproduces current behavior on the existing suite (target: 0-regression against the ~4850-test green baseline).

---

## Files of record

- Grader: `backend/evals/paperbench/leaf_scorer.py` (`_grade_batch:1646`, `_SYSTEM_PROMPT:1145`, `run_invariant_checks:1214`, `_latest_metrics_path`, `_compact_metrics_for_grader`)
- Aggregation: `backend/agents/rlm/report.py` (`_best_recorded_rubric_score:637`, `_apply_best_of_run_floor:673`, finalize_rescore wiring `:925`), `backend/agents/rlm/finalize_regrade.py`
- Artifact/champion: `backend/agents/rlm/best_attempt.py`, `backend/agents/rlm/bes_rlm.py` (`_snapshot_code`)
- Transport: `backend/agents/rlm/run.py:186` (`_build_llm_client`), `backend/services/context/workspace/tools/{rlm_query,openai_client,azure_openai_client}.py`
- Execution: `backend/agents/rlm/primitives.py`, `binding.py`, `cell_matrix.py`, `gpu_cell_runner.py`, `env_pin.py`, `context.py` (add `execution_mode`)
- BES/AB: `scripts/ab_compare.py`, `backend/agents/rdr/candidates.py`
- Integration: merge `context_map.py` + `lesson_distiller.py` from `origin/bes`; `docs/audits/2026-06-07-bes-doc-alignment-audit.md` (on `m2`)

---

## Appendix S — June-2026 SOTA grounding & addenda (companion)

A literature-grounding companion — **`2026-06-16-grader-noise-and-harness-remediation-design.md`** — annotates each workstream above against ~25 page-verified June-2026 papers (confirm/temper/contradict), with a SOTA→workstream map and an adversarial stress-test. It does **not** re-design anything locked; it grounds and challenges. Headline verdict: **the locked design is on the right side of the evidence; the additions below are narrow.** (Both the companion and this appendix were themselves adversarially reviewed.)

**Confirmed by the literature — no change needed:** A2 deterministic routing (DeCE arXiv:2509.16093, decomposed r=0.78 vs 0.35 — the biggest variance/bias lever); A3 strip-the-MAX-floor (max-over-noise is upward-biased); A4 champion-artifact (ground the verdict in the executed artifact — BadScientist arXiv:2510.18003); **A7 EVIDENCE_GATE is the single highest-value correctness lever in the corpus** (MLR-Bench arXiv:2505.19955 ~80% fabricated results; ImpossibleBench arXiv:2510.20270 76% cheat; RewardHacking arXiv:2605.02964 env-hardening −87.7% exploits) — treat it as more than an "S" cleanup; D1/D3 (single-agent ≥ MAS at equal token budget, arXiv:2604.02460 — budget-match arms; keep BES off until measurable).

**Addenda (non-binding):**
- **A1** — make median-of-N a *cascade* (escalate only judgment / first-two-samples-*disagree* leaves; cost-saving cascades arXiv:2502.09054; panel benefit plateaus at n_eff≈2.2, arXiv:2605.29800 → cap N≈3) and gate its default-ON on the **Lane-0 calibration σ-drop** — Rating Roulette (arXiv:2510.27106) warns same-model resampling may not converge at fixed settings, so *measure* σ-before/after, don't assume √N. Escalate on cross-sample disagreement, **not** verbalized confidence (judge ECE up to 39%, arXiv:2508.06225).
- **A5** — add an *optional cross-family* judgment-leaf grader, gated on a **measured** self-preference gap: Sonnet grades Sonnet-authored code, and cross-family > self-verification (arXiv:2512.02304, arXiv:2402.11436). A5's `REPROLAB_GRADER_BACKEND`/`_MODEL` mechanism already supports it — it is just never *used* for diversity. (Bias fix, not variance — low priority, off by default.)
- **B2** — surface prior grades/failures to the root as *tool/memory observations*, not its own prior prose (+23–93pp correction, Self-Correction Illusion arXiv:2606.05976). The harness already does this structurally; lean into the framing.
- **D1** — report every Δ relative to the **measured σ_grader band** (Lane-0); gate "significant" on Δ > kσ or a conformal bound (arXiv:2602.03814).

**New levers from the companion (§3.5):**
- **IMPLEMENTED 2026-06-16 (this branch):** a **decline-aware convergence advisory.** The existing flatline detector (`_rubric_plateaued`) misses a *declining* trajectory — the overthinking / inverse-scaling signal (arXiv:2604.10739, arXiv:2507.14417): the score peaked and recent changes made it worse, yet the run loops to the cap degrading further. Added `_rubric_declining` + `_decline_advisory_note` + a flag-gated (`REPROLAB_RUBRIC_DECLINE_ADVISORY`, **default-off**) regression `convergence_note`. Purely advisory (a tool-result note, never a forced stop); fail-soft; 0-regress when off. (Budget surfacing intentionally omitted for now — the note carries no wall-clock.) `backend/agents/rlm/primitives.py`; 15 unit tests in `tests/agents/rlm/test_scope_self_heal.py` (green). *Stash-verified isolated from the in-flight `leaf_scorer.py` refactor.*
- **Design-only (deferred):** broader per-iteration budget surfacing + capped / oscillation-aware forced-iteration (BATS / AgentStop arXiv:2605.15206 / overthinking) — needs a proactive per-iteration injection surface (e.g. piggyback `check_user_messages`), so deferred to avoid an invasive change to a hot file.

Full grounding, the SOTA→workstream map, the efficiency ledger, and the "where this could fail" stress-test live in the companion.
