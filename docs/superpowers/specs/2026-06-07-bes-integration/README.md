# BES Integration — Phased Design (2026-06-07)

**Status:** 🟡 PROPOSED — brainstorm-grounded, **not implemented**. Awaiting manual approval. Every code change below is design intent; treat the code as truth and confirm each seam before building.
**Provenance:** team whiteboard walkthrough `IMG_5104.MOV` (2026-06-05) → reconstructed docs (`origin/bes:paper-repro-bes-docs.zip` — `01_TRANSCRIPT_AV.md`, `02_ARCHITECTURE_SPEC.md`, `03_BES_PROPOSAL.md`, `BES_INTEGRATION.md`) → grilling + recon session 2026-06-07. The recon was grounded by four parallel codebase sub-agents (preflight, scorer, coverage, RDR-mechanics); their `file:line` findings are cited inline.
**One-line:** Integrate **BES (Bidirectional Evolutionary Search)** by **extending the existing RDR controller** (not a new `--mode`, not a greenfield engine), sequenced **after** the cheap honest score/retry wins so we learn whether BES's exploration is even the bottleneck.

> **Codex review (2026-06-07) — applied.** Codex verified the core `file:line` claims and flagged corrections now folded in: (1) **BES v1 = competing candidates only**; the **evolve/splice** step is **deferred to v2** (it requires per-candidate cell execution or a cell-granular redesign — Phase 3 §4). (2) The **pre-run preflight gate is a mode-agnostic Phase 2 item, not a BES delta**. (3) The **finalize re-score (Phase 0) depends on the Phase 4 env-guard** and must be deterministic (no re-grading, no `max()` across exclusion policies). (4) Several `file:line`/signature fixes (Phase 3 §2/§4, Phase 2 §3/§8). Per-doc "Codex review" callouts carry the detail. Rollback: each non-flag change (env merge, scorer change, `ScopeSpec` field, finalize re-score, dependency synthesis) needs a per-phase kill switch + rollback procedure — see each phase's Definition of done.

---

## 1. The problem — what the 0.36 SDAR run actually proves

Motivating run: `runs/prj_09047604e591d969` (SDAR, arXiv 2605.15155, `--sandbox local`, 8×A5000). `rubric_evaluation.json` → **overall 0.3556** vs target 0.6.

| Rubric area | Score | Weight | Where the points went |
|---|---|---|---|
| Method & code fidelity | **0.60** | 0.38 | ✅ Core SDAR correct: `gate=σ(β·Δ).detach()`→0.9, stop-grad→0.9, λ=0.1→0.85 |
| Data & preprocessing | 0.09 | 0.14 | ❌ No E5 retriever / real Search-R1; closed-book prompts |
| Experiment execution | 0.145 | 0.18 | ❌ Only Search-QA cells; ALFWorld/WebShop absent |
| Evaluation protocol | 0.285 | 0.15 | ⚠️ Missing baselines (Skill-SD, RLSD, OPSD) |
| Result match | 0.14 | 0.10 | ⚠️ Directionally right (SDAR>GRPO) but misses the +7.0% number |
| Artifact completeness | 0.42 | 0.05 | ⚠️ Weak provenance + scalar-only metrics (no curves) |

**The reframe:** the agent already writes SDAR correctly. This run fails on **breadth** (2 of 3 environments, 2 of 5 baselines missing) and **accounting** (8 zero-leaves, all ALFWorld/WebShop, were *declared out of scope but counted as failures* — see §6). So BES's payoff here is **"cover the env × baseline matrix in parallel and count the score honestly,"** not "discover the algorithm."

## 2. What the grilling established (locked decisions)

- **D1 — Embodiment: extend RDR, do not add `--mode bes`.** RDR (`backend/agents/rdr/controller.py`) is already a deterministic `decompose → cluster-dispatch → score → repair → report` controller — ~80% of the BES skeleton. A new mode would duplicate ~1,200 lines of hardened control flow. **BES v1 adds one delta — competing candidates (forward-search + select)** — to RDR behind flags; the **evolve/splice** step is **deferred to v2** (Phase 3 §4), and the **pre-run gate is mode-agnostic (Phase 2)**, not a BES delta.
- **D2 — TDD is severed from BES.** The "catch errors before the run" idea is a **mode-agnostic pre-run preflight** layer, not part of BES. It attacks a different failure class (crash/dep/schema) than BES (fidelity/exploration). (Phase 2.)
- **D3 — Cheap honest wins first.** Merge `full-scope-envs`, fix the accounting, add baselines via guidance — these likely reach ~0.50 **before any BES code**, and tell us whether exploration is the bottleneck. (Phases 0–1.)
- **D4 — Splice is cell-metric (2b), not cluster-file (2a).** True candidate recombination at the file level is blocked on missing leaf→file provenance; the clean, safe operation is unioning surviving matrix cells via `aggregate_cell_metrics`. (Phase 3.)

## 3. Recon map — BES design → real seams

| BES / pipeline concept | Lands on | File:line |
|---|---|---|
| RLM orchestrator entry | `RLM(...).completion()`, paper as REPL `context` | `backend/agents/rlm/run.py` |
| Baseline builder (one-shot today) | `implement_baseline(plan) → {ok,code_path,files}` | `primitives.py:1322` |
| Experiment runner / cell matrix | `run_experiment` → `_execute_cell_matrix` → `gpu_cell_runner.run_matrix` | `primitives.py:3706`, `:3582`; `gpu_cell_runner.py:300` |
| Rubric grader | `verify_against_rubric` → `score_reproduction` | `primitives.py:4489`; `backend/evals/paperbench/leaf_scorer.py` |
| Candidate pool (latent) | `propose_improvements`, `record_candidate_outcome(parent_id)` | `primitives.py:4784`, `:4899` |
| Deterministic controller (BES host) | `run_rdr`: decompose→cluster→score→repair | `backend/agents/rdr/controller.py` |
| Pre-run gate | `validate_code_pre_flight` ← `scan_code_dir` | `pre_flight_validator.py:1344`; `preflight_ast.py:756` |
| Feature flags | `Settings(BaseSettings)`, `REPROLAB_*` | `backend/config.py:12` |

## 4. The five phases

| Phase | Doc | Theme | Effort | Needs BES? | Expected effect |
|---|---|---|---|---|---|
| **0** | [phase-0-foundation-and-accounting.md](phase-0-foundation-and-accounting.md) | Merge `full-scope-envs` · finalize re-score · `cell_execution_error` routing | Low | No | Unblocks Data; recovers 0.3556→**0.431** for free; floor-enforces code-bug repairs |
| **1** | [phase-1-coverage-completion.md](phase-1-coverage-completion.md) | 3 missing baselines · eval templates · provenance + curves | Med (agent-side) | No | Heaviest leaf 0.15→~0.75; → ~**0.48–0.52** honestly |
| **2** | [phase-2-preflight-retry-reduction.md](phase-2-preflight-retry-reduction.md) | `requirements.txt`+import-verify on local · sandboxed env-construct smoke · swallowed-OOM check · **mode-agnostic RDR pre-run gate wiring** | XS–L | No (mode-agnostic) | Fewer GPU-burning retries |
| **3** | [phase-3-bes-on-rdr.md](phase-3-bes-on-rdr.md) | Competing candidates (v1) · evolve/splice (**v2, deferred**) | L | **Yes** | Parallel matrix coverage; honest assembly (v2) |
| **4** | [phase-4-frontier-and-honesty-guard.md](phase-4-frontier-and-honesty-guard.md) | ALFWorld learning (warm-start/shaping/env-once) · env-axis exclusion guard | High | No | The unlock past 0.6 + closes the gaming hole |

## 5. Corrections the grounding made to the brainstorm (honesty log)

The four sub-agents overturned several mid-brainstorm claims. Recorded so the design rests on verified facts, not the brainstorm's first draft:

1. **The recompute is 0.431, not 0.448.** Excluding *all 8* weak-leaves over-counts — leaf `d2c1a0a8` (SkillBank) is genuinely-unimplemented in-scope work, not an env de-scope. Excluding only the 7 ALFWorld/WebShop env leaves → **0.4315** (arithmetic in Phase 0; reproduces 0.3556 exactly pre-exclusion).
2. **`cell_execution_error` does not cause restarts.** It's typed `partial_evidence` (metrics-first short-circuit `primitives.py:224-226`), so the root keeps iterating in-place — the real bug is it *skips the repair-iteration floor* (gated on `==repairable`), not that it over-restarts.
3. **Swallowed-OOM is not the most frequent failure** — `cell_execution_error` (6×) and `compute_scope_invalid` (4×) tie or exceed `silent_oom` (1 distinct run).
4. **`compute_scope_invalid` is a warning, not a failure** (`primitives.py:1129-1166`, coerced to `None`, run proceeds) — drop it from the retry story.
5. **`full-scope-envs` makes ALFWorld real but not learnable** — ships sparse `float(won)` reward (`alfworld_env.py:458`), zero warm-start/shaping. B4 is genuinely unbuilt.

## 6. Cross-cutting: the sequencing trap and the honesty guard

- **⚠ Sequencing trap.** `full-scope-envs` adds a *verified-exclusion gate* (`leaf_scorer.py`): once an env is real-and-attempted, an *agent-declared* skip **stays scored at 0** instead of being excluded. So bringing ALFWorld in-scope **without making it learn moves the score backwards** (excluded 0-weight leaves → counted 0-score leaves). Only flip an env in-scope when you can (a) make it learn (Phase 4) or (b) keep it a *verified operator* exclusion.
- **⚠ Honesty/credibility.** The env axis has **no operator-sanction guard** (`leaf_scorer.py:494-497` excludes any self-declared `environments_skipped` unconditionally), while the model axis does (`b5e36dd`). Until Phase 4 closes this, **0.431 mixes principled (WebShop) and non-principled (ALFWorld-crashed) exclusion** — do not report it as "the number" without the caveat.

## 7. Open decisions (resolve with maintainer before building each phase)

- Phase 1: which baseline to add first (OPSD-standalone is a flag flip; Skill-SD needs the SkillBank feed).
- Phase 3: `bes_candidates_per_cluster` default cap (token-cost driver) and whether v1 ships splice at all.
- Phase 4: env-axis guard shape — operator `skip_environments` allow-list vs corroborated never-attempted evidence.

## 8. Source docs

The four reconstructed design docs live on `origin/bes` as `paper-repro-bes-docs.zip` (not copied here — the transcript carries peripheral GTM/business notes out of scope for this spec). Read order: `03_BES_PROPOSAL.md` → `02_ARCHITECTURE_SPEC.md` → `01_TRANSCRIPT_AV.md` → `BES_INTEGRATION.md`.
