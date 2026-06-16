# Harness Breakdown & Recent Changelog

A working map of the OpenResearch / ReproLab reproduction **harness** — the layers
that take a paper to a scored reproduction — and a changelog of what was
added/changed in the 2026-05-31 → 2026-06-01 work (the OOM/GPU cell-runner
remediation, env robustness, the lab-UI clarity fixes, and the fidelity-scoring
redesign). For the "why it fits together" narrative see `system_overview.md` and
`docs/design/rlm-pivot-brief.md`; for day-to-day commands see `CLAUDE.md`.

---

## Part 1 — Architecture: the four layers

```
 paper ──► (1) ORCHESTRATION ──► (2) EXECUTION ──► (3) SCORING ──► final_report
              RLM root + REPL        cell runner /      leaf scorer +
              + 12 primitives        sandboxes          rubric areas
                         └────────── (4) RUN STATE: runs/<id>/ (file-backed) ──────────┘
                                            └── SSE ──► (5) LAB UI
```

### (1) Orchestration — the RLM root + primitives
- `backend/agents/rlm/run.py` — run entry. Builds `rlm.RLM(...)` (the `rlms` PyPI lib), offloads the paper as the REPL `context` variable (the root sees only constant-size metadata, never the corpus), runs `.completion()` on a worker thread.
- The root model writes Python that calls **domain primitives** exposed in the REPL (`backend/agents/rlm/primitives.py`): `understand_section`, `extract_hyperparameters`, `detect_environment`, `build_environment`, `plan_reproduction`, `implement_baseline`, `run_experiment`, `verify_against_rubric`, `propose_improvements`, `record_candidate_outcome`, `check_user_messages`, `respond_to_user`.
- System prompt: `backend/agents/rlm/system_prompt.py`. Forced-iteration policy (refuse premature `FINAL_VAR`): `backend/agents/rlm/forced_iteration.py`.
- `implement_baseline` dispatches a Sonnet code-writing sub-agent (`backend/agents/baseline_implementation.py::run_with_sdk`); guidance is assembled in `_compute_constraint_guidance`.

### (2) Execution — sandboxes + the one-GPU-per-cell cell runner
- `run_experiment` (`primitives.py`) executes the baseline. Two paths:
  - **Cell route (preferred, OOM-safe):** when the backend exposes ≥1 GPU (local/docker) AND the agent emitted `code/cells.json` + `code/train_cell.py`, the matrix runs **one training cell per GPU** via `backend/agents/rlm/gpu_cell_runner.py::run_matrix` (`min(free_gpus, cells)` in parallel, per-cell OOM shrink-retry). Capacity/dataset gating + leaf-shaped aggregation: `backend/agents/rlm/cell_matrix.py`. Capacity descriptor: `backend/services/runtime/gpu_capacity.py`.
  - **Legacy route (fallback):** `python train.py` via `commands.json`, dispatched to a sandbox backend (`local` / `docker` / `runpod`) by `_execute_in_sandbox`. Used when no `cells.json`.
- Sandboxes: `OPENRESEARCH_DEFAULT_SANDBOX` selects `local` (host subprocesses), `docker`, or `runpod` (remote GPU pods). Local multi-GPU leasing: `backend/services/runtime/local_gpu_allocator.py`. Launcher: `scripts/reserve_and_run_sdar.py` → `scripts/batch_reproduce.py` (leases GPUs, per-run venv, monitors).

### (3) Scoring — leaf scorer + rubric areas + invariants
- `backend/evals/paperbench/leaf_scorer.py::score_reproduction` — the authority. Flattens the rubric tree to leaves, gathers evidence (code + metrics, bounded), and **LLM-grades each leaf 0..1 with a justification**. Rolls leaves up into 6 weighted **areas** (method fidelity, data fidelity, execution, eval protocol, result match, artifacts).
- **Invariant gate (the brittle layer):** optional regex `InvariantSpec` patterns (`backend/agents/prompts/paper_hints.py`, hand-coded per paper — SDAR only) line-by-line `re.search` the agent's source; a `must_match` miss **soft-caps to 0.5**, a `must_not_match` hit **hard-caps to 0.0**. See the redesign in Part 2.
- Rubric for papers without a curated `docs/papers/<id>.yaml`: auto-generated from the paper text (`backend/agents/rlm/rubric_gen.py::generate_rubric_tree`).
- `verify_against_rubric` (primitive) returns `{overall_score, areas, weak_leaves, leaf_scores (per-leaf id/score/justification), ...}`; persisted to `runs/<id>/rlm_state/rubric_evaluation.json`.

### (4) Run state — file-backed, resume-safe (`runs/<id>/`)
`demo_status.json` (UI snapshot, atomic) · `rlm_state/` (per-iteration checkpoints, `iterations.jsonl`, `rubric_evaluation.json`) · `dashboard_events.jsonl` (append-only SSE log) · `experiment_runs.jsonl` (every `run_experiment` result) · `cost_ledger.jsonl` · `code/` (the reproduced project + `cells.json`/`train_cell.py`/`outputs/<run_id>/metrics.json`) · `final_report.{json,md}`.

### (5) Lab UI ↔ backend
- Next.js frontend (`frontend/`) talks to FastAPI (`backend/app.py`) **server-side only** via `/api/demo/*` proxy. SSE stream drives the rubric strip, exploration graph (`ConstellationCanvas`), and chat. Events route through `backend/agents/rlm/sse_bridge.py` (the egress chokepoint that strips REPL locals + bounds output).

---

## Part 2 — What we added/changed (2026-05-31 → 2026-06-01)

### A. OOM + GPU-capacity remediation — the one-GPU-per-cell route (the headline)
**Why:** an SDAR baseline run (`prj_09047604e591d969`) died 11×: every cell stacked on `cuda:0` and OOM'd, fp32 full-vocab log-probs blew ~20 GB even on the 1.7B, and a re-OOM forced-iteration loop burned ~$10. Spec: `docs/superpowers/specs/2026-05-31-oom-gpu-capacity-remediation-design.md`; handoff: `docs/runbooks/2026-05-31-oom-gpu-remediation-handoff.md`.

| Commit | What |
|---|---|
| `feat(rlm): GPU capacity descriptor + cell runner + forced-iteration terminal-OOM bypass` | `gpu_capacity.describe_capacity` (backend-agnostic budget), `gpu_cell_runner.run_matrix` (one GPU per cell + OOM shrink-retry), `forced_iteration` terminal classes `oom_shrink_exhausted`/`capacity_exhausted` that accept `FINAL_VAR` (stop, don't re-OOM). |
| `feat(rlm): SDAR env interface ABC + preflight backstop` | `sdar_env_base.BaseEnv` (a missing `build_student_prompt` becomes a construction-time `TypeError`, not a mid-grid `AttributeError`) + recursive self-scoping AST check. |
| `feat(rlm): single-cell trainer contract + GPU-budget guidance` | guidance: GPU-budget brief + `train_cell.py`/`cells.json` contract + memory discipline (forbid fp32 full-vocab `log_softmax`); torchrun suppressed on the cell path. |
| `feat(rlm): route run_experiment through the GPU cell runner` | `cell_matrix.py` (capacity gate, dataset preflight, `per_model[model][env][baseline]` aggregation, verified vs a real metrics sample) + the `_execute_cell_matrix` route, fail-soft to legacy. |
| `feat(rlm): terminal-stop wiring — stop + report, never re-OOM` | `run.py` calls `note_terminal_failure`; `final_report.json.stop_reason`. |
| `feat(scripts): reconcile SDAR launcher flags` / `support uncapped SDAR runs via none/0 cap sentinels` | dropped torchrun-wrap + force-single-GPU; `SDAR_MAX_WALL_CLOCK=none` etc. omit caps → truly uncapped runs. |
| `test(rlm): run_experiment cell-route branch wiring` | end-to-end branch + manifest-gate tests. |

### B. Environment robustness
| Commit | What |
|---|---|
| `fix(env): per-run venv inherits the repo base stack + fail-soft optional imports` | per-run venvs created with `--system-site-packages` (inherit torch/matplotlib/… so the legacy path can't hit `No module named matplotlib`); guidance: wrap optional/viz imports (matplotlib/seaborn/wandb) in try/except — a missing viz lib skips a figure, never aborts training; cell contract is sticky across iterations (don't collapse back to a monolith). |
| `fix(rlm): don't false-drop HuggingFace/Kaggle dataset URLs in cell preflight` | the dataset HEAD-probe no longer confirms-dead `huggingface.co`/`kaggle.com` URLs (they resolve via the datasets lib, not a page GET) — fixes an nq_open false-drop → `capacity_exhausted`. |

### C. Lab-UI clarity
| Commit | What |
|---|---|
| `feat(cli): parse the real paper title from the paper text; reject paper_text placeholder` | titles like "Self-Distilled Agentic Reinforcement Learning" instead of `paper_text`/`Untitled`. |
| `fix(lab-ui): unique run keys + real paper title + run_N + graph always visible` / `truly-unique run key via backend runDir` | fixed the duplicate-React-key crash (was keying on the paper-locked `projectId`; now the unique run-dir name from `_list_runs`), real titles + `run N` numbering, and score panels capped so they never occlude the `ConstellationCanvas`. |

### D. Fidelity-scoring redesign (designed; build in a worktree)
- `docs: semantic invariant verification design` — diagnosis that the **regex invariant gate** (Part 1 §3) soft-caps correct-but-differently-named code (Goodhart, false-negatives, per-paper hand-coded, doesn't generalize). Proposes a **semantic, paper-derived, runtime-corroborated** verifier: invariants as data (yaml or LLM-extracted), an LLM `SemanticInvariantVerifier` (naming-agnostic), runtime-derived anti-surrogate, regex demoted to a positive fast-path. Drop-in behind `score_reproduction`. Spec: `docs/superpowers/specs/2026-06-01-semantic-invariant-verification-design.md`.
- **In flight (worktree `feat/rubric-ui-clarity`, not on this branch yet):** enrich the `rubric_score` SSE event with per-area `leaves` (`label, score, status, why`) + `weak_leaves` + `recent_errors`, and a frontend best-of-run headline + expandable, clearly-labeled rubric breakdown (which leaves fail, why, the specific errors).

### E. Live validation (the proof)
Uncapped SDAR rerun on 6×A5000 (smallest-two, Search-QA): cells **trained one-per-GPU** (no `cuda:0` stacking, no OOM), real metrics with **SDAR 0.114 > GRPO 0.065** (the paper's ordering), `run_experiment success=True`, **rubric climbed 0.176 → 0.32** (vs 0.0 when everything failed). The agent then self-diagnosed its own fidelity gaps (gate formula, grpo_opsd reward collapse, config scale) — i.e. the iteration loop works. Remaining ceiling ≈ 0.32 is the regex invariant gate (§D) + result-match training depth.

---

## Known ceilings / next levers
1. **Regex invariant gate** caps the highest-weight area (method fidelity) on correct code → the §D semantic verifier is the biggest unlock.
2. **`implement_baseline` ~15 min/iter** (full-rewrite, not patch) is the wall-clock bottleneck.
3. **Per-run venv torch** historically missing (now mitigated by `--system-site-packages`).
4. **Lossy paper parse** (`parsed_full_text.txt` missing) is mitigated by `OPENRESEARCH_PAPER_TEXT_PATH` but tracked out-of-scope.
