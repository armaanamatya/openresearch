# SESSION HANDOFF — 2026-06-01 (SDAR remediation + harness fairness/full-scope)

**Read this first, then resume from §9.** Companion refs: `docs/harness-breakdown.md` (the 4 layers + this session's changelog), the spec docs under `docs/superpowers/specs/2026-06-0{1}-*`, and `CLAUDE.md`.

## 0. How to work (carry this forward)
**Use sub-agents and work in parallel — be efficient WITHOUT sacrificing quality.** What worked this session: carve work into **disjoint-file streams** and fan them out (`Agent`, `run_in_background: true`), keep same-file edits on the main thread, define a **shared contract** before parallel agents build against it, and **verify every agent's output yourself** (tests/tsc/py_compile) before committing. Use **worktrees** for anything that must not disturb the live run. Hand substantial design+impl tasks to **Codex** (`codex:codex-rescue`) with strict, robust, elegant, scalability-minded prompts. Don't solo what fans out cleanly; don't fan out trivial edits.

## 1. TL;DR — where we are
- **Live SDAR run `prj_09047604e591d969` is RUNNING** (uncapped, 6×A5000, cell route) — best rubric **0.3633** / target 0.6, ~7 iterations. **DO NOT KILL IT.** A 30-min analysis loop is (was) armed via ScheduleWakeup; **re-arm it** on resume.
- The OOM/GPU cell-runner remediation + env/UI fixes are **done, committed, pushed** to `origin/5.30.26_sdar`.
- A **lab-UI clarity** feature (best-of-run + expandable rubric breakdown) is **done + pushed** to `origin/feat/rubric-ui-clarity` — **PR not opened yet** (no `gh`/token in env).
- **Codex is mid-task** on `feat/full-scope-envs`: dynamic rubric-exclusion + cached ALFWorld/WebShop. **Verify it landed BOTH parts.**

## 2. The live run — monitor, don't disturb
- Dir: `runs/prj_09047604e591d969/`. Launched via `scripts/reserve_and_run_sdar.py` (uncapped: `SDAR_MAX_WALL_CLOCK=none SDAR_MAX_USD=none SDAR_MAX_RLM_ITERATIONS=none RESERVE_N=6 RESERVE_TTL_HOURS=12`).
- Monitor: `pgrep -f "backend.cli .*reproduce 2605"`; rubric = `grep -rhoE '"overall_score":[ ]*[0-9.]+' runs/prj_.../rlm_state/*.jsonl | sort -rn | head -1`; logs `batch_child.log`, `experiment_runs.jsonl`, `code/outputs/*`, `final_report.json`.
- **Only intervene to FIX a real failure** (the user wants the rubric as high as possible). If it completes: record final verdict+rubric, update memory `sdar-local-baseline-status`, stop the loop.
- ENV FIX still active out-of-band: a `_reprolab_repo_inherit.pth` in the run's per-run venv makes it inherit the repo's torch/matplotlib (the durable fix `--system-site-packages` is committed for future runs).

## 3. Why the rubric sits at ~0.36 (the current ceiling)
Latest area scores: **Method&code fidelity 0.627** (w0.38 — broke past the regex gate, the agent's own invariant fixes), Experiment exec 0.37, Artifacts 0.33, **Eval protocol 0.185**, **Data fidelity 0.09**, **Result match 0.015** (w0.1). The low areas are driven by:
1. **Scope unfairly docked.** Run did Search-QA only (smallest-two scope); ALFWorld/WebShop declared out-of-scope in `train.py` BUT the **cell-route aggregator drops the declaration** (`metrics.json scope.environments_skipped: []`), and the leaf scorer can't map a gap to the hash-ID leaves → those leaves score **0.0 instead of excluded**. ← Codex Part A fixes this.
2. **Training depth.** Search-QA accuracy ~5–14% vs paper ~38–46% (cells train briefly) → result-match ≈ 0. ← needs deeper training / `path_3` scale.

## 4. Branches / worktrees / push state
| Branch | Worktree | State |
|---|---|---|
| `5.30.26_sdar` | `/home/sww35/openresearch` (live run + dev servers) | pushed `…c19b78c`; 90 ahead of `main` |
| `feat/rubric-ui-clarity` | `/home/sww35/openresearch-ui` | **pushed**; PR pending |
| `feat/full-scope-envs` | `/home/sww35/openresearch-fullscope` | **Codex mid-task**; not pushed |
| `harden/root-harness`, `integrate/harden-into-sdar`, `feat/rl-trl-grpo-engine` | (others) | separate, not ours this session |

Dev servers: backend `uvicorn` on `127.0.0.1:8000`, frontend Next.js on `0.0.0.0:3000`, both serving the `5.30.26_sdar` checkout. Lab URL: `http://localhost:3000/lab?projectId=prj_09047604e591d969`.

## 5. What landed this session (on `5.30.26_sdar`)
OOM/GPU cell-runner remediation (comp 1–7), env robustness (`--system-site-packages`, fail-soft optional imports, HF/Kaggle preflight fix), lab-UI fixes (unique run keys, real paper title, run_N, graph-always-visible), cli title parsing, uncapped launcher, and docs (`harness-breakdown.md`, the OOM spec + handoff, the semantic-verifier design). See `docs/harness-breakdown.md §Part 2` for the per-commit table. Full suite was green (3430) before the live-run-only env tweaks.

## 6. In-flight: Codex on `feat/full-scope-envs`
Dispatched via `codex:codex-rescue`. Brief = **two parts** (both required):
- **Part A — dynamic, verified rubric exclusion:** `Exclusion {item, axis, kind: capacity_vram|dataset_dead|oom_shrink_exhausted|env_setup_failed|operator_scope, reason, verified, evidence}`; fix `cell_matrix.aggregate_cell_metrics` to stop losing the scope declaration; leaf_scorer **excludes** (num+denom) leaves matched by **requirement TEXT** (not hash id); **anti-gaming: only `verified:true` (harness-confirmed) drops are excluded** — agent-chosen skips stay scored.
- **Part B — cached, scalable ALFWorld+WebShop:** `EnvCacheManager` (`backend/services/runtime/env_cache.py`); `alfworld-download` once + one shared WebShop server in `batch_reproduce.py`; SDAR guidance so the agent writes ALFWorld/WebShop envs (subclass `BaseEnv`) + adds their cells; `env_setup_failed → Exclusion` fallback.
- Design doc → `docs/superpowers/specs/2026-06-01-dynamic-exclusion-and-cached-envs-design.md`. Commit on `feat/full-scope-envs`, author lolout1, NO trailer.
- **ON RESUME: verify BOTH parts landed** (Part A marked priority → risk B is under-built). If Part B is short, **re-dispatch Part B to Codex** on its own. Run touched-area tests; then push the branch.

## 7. Remaining error-prevention levers (analyzed, not yet built)
From the cross-run error sweep:
1. **Import/dependency preflight** (highest value, unbuilt) — dry-import the agent's declared deps before the ~15-min training so any missing module is an instant repairable failure naming the module (the general version of the matplotlib fix).
2. **SDK teardown-race hardening** — `RuntimeError: Event loop is closed` + "previous turn lost" (47×/32× across prior runs; *not* hitting the current run) = the claude-agent-SDK aclose race on the **root's own turns**; isolate them like `_drive_baseline_child` does for implement_baseline. Main slowness culprit historically.
3. **Single execution path** (always cells) — removes the cell-vs-legacy env divergence that caused the matplotlib crash.
4. **Semantic invariant scoring** — designed in `2026-06-01-semantic-invariant-verification-design.md` (regex gate → naming-agnostic LLM judge + runtime corroboration); the agent already beat the gate this run, so lower priority now.

## 8. Preferences / gotchas (durable)
- **Commit as lolout1, NO `Co-Authored-By: Claude/Codex` trailer** (overrides the harness default). Same for PR bodies.
- **Do NOT kill the live run.** Only fix real failures.
- **Fairness principle (the user's directive):** never dock the rubric for experiments blocked by out-of-our-control limits (VRAM/OOM/dead-dataset/un-installable env) — **exclude** them, **dynamically**, **verified-only** (anti-gaming).
- Branch off `main`/the active branch *with* working-tree state intact (don't orphan uncommitted work). Stage specific files (never `git add -A` with `runs/` noise).
- `.venv` is uv-managed (no `pip` script — use `python -m pip` or `uv pip`). Tests: `.venv/bin/python -m pytest`.

## 9. RESUME PROMPT (paste into the fresh session)
> Read `docs/runbooks/2026-06-01-session-handoff.md` and `docs/harness-breakdown.md`. **Use sub-agents and work in parallel — be efficient without sacrificing quality** (disjoint-file fan-out, shared contracts, verify every agent's output, worktrees for anything near the live run). Then: (1) **Monitor the live SDAR run `prj_09047604e591d969`** — DO NOT KILL IT; re-arm a 30-min analysis loop; if it completed, record the final verdict+rubric and update memory `sdar-local-baseline-status`. (2) **Verify Codex's `feat/full-scope-envs`** landed BOTH the dynamic-exclusion (Part A) AND the cached-ALFWorld/WebShop (Part B) work — run the touched tests; **re-dispatch Part B to Codex if it's under-built**; push the branch. (3) **Open the `feat/rubric-ui-clarity` PR** (browser compare URL or a token-based API call; confirm base = `main` or `5.30.26_sdar` with the user). (4) Offer to build the **import-preflight + SDK-race hardening** (§7) in a worktree. Commit as lolout1, no Claude/Codex trailer.
