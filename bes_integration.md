# BES Integration — Summary of Changes

**Bidirectional Evolutionary Search**, folded into the paper-reproduction pipeline by
**extending the existing RDR controller** — not a new `--mode`, not a greenfield engine.
Every behavior below ships behind a **feature flag, default OFF**; with the flags off the
pipeline runs **bit-for-bit as before**.

> **Provenance.** Whiteboard walkthrough `IMG_5104.MOV` (2026-06-05) → reconstructed brief
> (`origin/bes:paper-repro-bes-docs.zip`) → grilling + 4-agent codebase recon → the phased
> design at `docs/superpowers/specs/2026-06-07-bes-integration/`. This file is the executive
> summary of what actually landed.

---

## The reframe — what the 0.36 SDAR run proved

Motivating run `runs/prj_09047604e591d969` (SDAR, `--sandbox local`, 8×A5000) scored **0.3556**
vs a 0.6 target. The breakdown said the agent **already writes SDAR correctly** (method &
code fidelity 0.60). It failed on two things BES is built to fix:

| Failure class | Evidence |
|---|---|
| **Breadth** | 2 of 3 environments and 2 of 5 baselines were never run |
| **Accounting** | 8 zero-leaves (all ALFWorld/WebShop) were *declared out of scope but counted as failures* |

So BES's payoff here is **"cover the env × baseline matrix in parallel and count the score
honestly,"** not "discover the algorithm." That insight drove the sequencing below.

---

## Four locked decisions

- **D1 — Extend RDR, don't add a mode.** `run_rdr` is already a deterministic
  `decompose → cluster → score → repair → report` loop — ~80% of the BES skeleton. BES v1 adds
  **one delta: competing candidates** (forward-search + select).
- **D2 — TDD is severed from BES.** "Catch errors before the GPU run" is a **mode-agnostic
  preflight** layer (Phase 2), attacking a different failure class (crash/dep/schema) than BES
  (fidelity/exploration).
- **D3 — Cheap honest wins first.** Coverage + accounting fixes (Phases 0–1) likely reach ~0.50
  **before any BES code** — and tell us whether exploration is even the bottleneck.
- **D4 — Splice is deferred to v2.** File-level recombination is blocked on missing leaf→file
  provenance; v1 ships competing-candidates only.

---

## The five phases

| Phase | Theme | Needs BES? | Expected effect |
|:---:|---|:---:|---|
| **0** | Merge `full-scope-envs` · finalize re-score · `cell_execution_error` repair routing | No | Recovers 0.3556 → **0.431** honestly; floor-enforces code-bug repairs |
| **1** | 3 missing baselines · eval templates · provenance + curves guidance | No | Heaviest leaf 0.15 → ~0.75; → ~**0.48–0.52** |
| **2** | Local `requirements.txt` + import smoke · env-construct smoke · swallowed-OOM AST check · mode-agnostic RDR pre-run gate | No | Fewer GPU-burning retries |
| **3** | **Competing candidates (v1)** · evolve/splice (v2, deferred) | **Yes** | Parallel matrix coverage |
| **4** | ALFWorld env-once + reward shaping · env-axis exclusion honesty guard | No | The unlock past 0.6 + closes the score-gaming hole |

---

## Feature flags — all default OFF

| Flag | Default | Gate |
|---|:---:|---|
| `OPENRESEARCH_BES_ENABLED` | `False` | **Master** gate for BES-on-RDR; off ⇒ today's RDR path, inert children |
| `OPENRESEARCH_BES_CANDIDATES_PER_CLUSTER` | `1` | N competing candidates per cluster (`1` = parity) |
| `OPENRESEARCH_BES_SELECT_METRIC` | `cluster_score` | Candidate SELECT metric (`cluster_score` \| `failed_leaves`) |
| `OPENRESEARCH_BES_SPLICE_ENABLED` | `False` | Evolve/splice (v2) — no-op in v1 |
| `OPENRESEARCH_RDR_PREFLIGHT_GATE` | `False` | `scan_code_dir` before `run_experiment` on the RDR path |
| `OPENRESEARCH_PREFLIGHT_SMOKE` | `False` | Import-smoke probe of generated code (opt-in) |

**Turn on the BES delta:**
```bash
OPENRESEARCH_BES_ENABLED=true OPENRESEARCH_BES_CANDIDATES_PER_CLUSTER=3 \
  python -m backend.cli reproduce 2605.15155 --mode rdr --sandbox local
```

---

## Change map

Definitions in `backend/config.py`; design in `docs/superpowers/specs/2026-06-07-bes-integration/`.

| Phase | Source | Tests |
|:---:|---|---|
| 0 | `rlm/report.py` (finalize re-roll-up) · `rlm/failure_classifier.py` (`cell_execution_error` routing) · `evals/paperbench/leaf_scorer.py` (env-axis guard) | `tests/evals/test_finalize_rescore.py` |
| 1 | `rlm/primitives.py` (SDAR baselines + curves/provenance guidance) | `tests/agents/test_sdar_baselines_guidance.py` |
| 2 | `rlm/preflight_smoke.py` (import smoke) · `rlm/preflight_ast.py` (swallowed-OOM check) · `agents/baseline_implementation.py` (local `requirements.txt` synthesis) | `test_preflight_smoke.py` · `test_preflight_swallowed_oom.py` · `test_local_requirements_synthesis.py` |
| 3 | `rdr/controller.py` (competing candidates) · `rdr/candidates.py` · `rdr/agent.py` · `rdr/models.py` · `config.py` (flags) | `tests/rdr/test_bes_on_rdr.py` · `test_best_of_run_rubric.py` |
| 4 | `rlm/alfworld_env.py` (env-once + reward shaping) · `rlm/sdar_env_base.py` · `rlm/agentic_rollout.py` · `services/runtime/env_cache.py` | `test_alfworld_shaping.py` · `test_alfworld_env.py` |

**Follow-up fixes (2026-06-07):**

- **Preflight smoke → module-level imports only.** `ast.walk` probed lazy/`try`-guarded imports,
  so a non-SDAR paper got flagged for `alfworld`/`faiss` it never calls. Narrowed to
  `tree.body` direct children; a missed lazy import degrades to repairable run-time behavior
  (far cheaper than a false positive). `rlm/preflight_smoke.py` + 3 regression tests.
- **Scope accounting respects `models_skipped`.** `_validate_scope_metrics` now treats
  capacity-gated (VRAM-budget) model exclusions as accounted-for, not missing. `rlm/primitives.py`.
- **Batch venv base-inherit `.pth`.** Works around the uv `--system-site-packages` empty-base
  gotcha so per-run venvs see the repo `.venv`'s proven GPU ML stack instead of re-downloading
  ~2 GB of torch. `scripts/batch_reproduce.py`.

---

## Status

All five phases implemented across 12 commits (`e9a3ab5` → `f2b11d3`) plus the three follow-up
fixes above. New behavior is **default-OFF**; the flag-off path is regression-tested for parity.
**v2 (evolve/splice) is deferred** — see Phase 3 §4 of the design spec.
