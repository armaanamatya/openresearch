# SDAR execute-mode seams — SESSION HANDOFF (2026-07-05)

> **Purpose:** seamless continuation in a fresh session. Wave 1 of the 6-seam plan is
> **implemented, self-tested green, and UNCOMMITTED on disk**. This doc is the single
> entry point: exact state, what remains, the discoveries that must not be re-derived,
> the run plan + money gate, and the working/style discipline. Continuation of
> [`docs/runbooks/2026-07-04-sdar-execute-mode-seams-and-repro-handoff.md`](2026-07-04-sdar-execute-mode-seams-and-repro-handoff.md).
>
> **Read order:** (1) this doc; (2) the **2026-07-04 handoff** (coordinates §2, run commands §8, gotchas §10);
> (3) [`docs/audits/2026-07-04-sdar-gcp-runs-log-analysis.md`](../audits/2026-07-04-sdar-gcp-runs-log-analysis.md) (evidence);
> (4) [`docs/history/specs/2026-07-04-execute-mode-verl-seams-and-sdar-repro-design.md`](../history/specs/2026-07-04-execute-mode-verl-seams-and-sdar-repro-design.md) (seam interfaces §3).
> Memory: `project_sdar_execute_mode_reproduction` (updated with this state).

---

## 0. Status at a glance

| # | Item | Status |
|---|---|---|
| **#1** | Cell command/launcher seam (`cell["command"]`) | ✅ **DONE + green (uncommitted)** |
| **#3** | verl→metrics adapter (`metrics_source.kind="verl"`) | ✅ **DONE + green (uncommitted)** |
| **#4** | Local-repo reuse + commit pin (`OPENRESEARCH_REPO_LOCAL_PATH/_COMMIT`) | ✅ **DONE + green (uncommitted)** |
| **#5** | Execute owns deps (skip cu121 bootstrap) | ✅ **DONE + green (uncommitted)** |
| **#6** | Fail-loud on execute clone failure | ✅ **DONE + green (uncommitted)** |
| **#2** | **Staged-env passthrough + HF_HOME clobber guard** | ⏳ **NOT STARTED** (fully specced, ready to dispatch) |
| **#7** | **NEW — `cells.json` pre-seed seam** (recommended; see §3.C) | ⏳ **NOT STARTED** (needed for a robust run; decision pending) |
| — | STEP-2 verl config fixes on the VM (ALFWorld/WebShop OOM) | ⏳ pending (VM-up, read-then-edit) |
| — | Driver execute-mode wiring (see §3.D) | ⏳ pending |
| — | Config artifacts (`configs/sdar_execute_*.json`) | ✅ written (uncommitted) |
| — | **The live GPU run** (VM restart + ~$400) | ⛔ gated on money checkpoint |

**First action in the new session:** Opus deep-hunk review of the 6 Wave-1 source diffs (they were Sonnet-implemented + self-tested, but the standing discipline is *Opus reviews every diff*). Then Wave 2 (#2 + #7), then STEP 2, then the run.

---

## 1. What landed (Wave 1) — DONE, green, uncommitted

**Combined verification (this session):** `177 passed` across the 6 new/extended test files + OFF-parity smoke
(`test_execute_mode`, `test_build_context_repo`, `test_env_pin`, `test_gpu_cell_runner`). `ruff` clean on all touched files.
Every change is **default-OFF / byte-identical** when its new field/flag is absent.

Touched source (`git diff --stat`, mine only):
```
backend/config.py                              | 11 ++   (repo_local_path, repo_commit fields)
backend/services/ingestion/repo/provisioner.py | 74 ++   (_reuse_local: copytree + commit checkout)
backend/agents/rlm/run.py                      | 23 ++   (#6 execute fail-loud branch)
backend/agents/rlm/primitives.py               | 36 ++   (#5 _execute_owns_deps helper + gate)
backend/agents/rlm/gpu_cell_runner.py          | 75 ++   (#1 command seam + #3 adapter post-step hook)
backend/agents/baseline_implementation.py      | 10 ++   (#3 _HARNESS_CODE_HELPERS + execute_repo_note)
```
New module: `backend/agents/rlm/verl_metrics_adapter.py` (232L, stdlib-only, vendored into `code/`).
New tests: `tests/rlm/{test_execute_clone_failure,test_execute_owns_deps,test_cell_command_seam,test_verl_metrics_adapter}.py`,
`tests/services/ingestion/repo/test_local_repo_reuse.py`, plus extensions to `tests/config/test_repo_flags.py` and
`tests/agents/rlm/test_cell_matrix.py`.

**Per-change notes for the reviewer:**
- **#4:** local-path logic lives inside `RepoProvisioner.clone` (`_reuse_local`): wipe `dest` → `shutil.copytree(local, dest, dirs_exist_ok=True)` incl. `.git` → best-effort `git checkout <pin>` → `rev-parse HEAD` → `build_manifest`. github path byte-identical when `repo_local_path` unset. `spec.url` still names *which* repo; `repo_local_path` is a local copy of it.
- **#6:** in `run.py::_resolve_and_clone_repo`, an `elif mode_override == "execute":` branch **before** the scratch-downgrade `else` keeps `spec.mode="execute"` + emits `run_warning code="repo_execute_unavailable"`. adapt/reference still downgrade to scratch (byte-identical).
- **#5:** `_execute_owns_deps(code_path)` — explicit `OPENRESEARCH_EXECUTE_OWNS_DEPS` truthy/falsy wins either way; unset ⇒ True iff `rlm_state/repo_spec.json` mode==`execute`. Gate at `_execute_in_sandbox` adds `and not _execute_owns_deps(code_path)`, skipping the whole cu121 bootstrap block. (Reason mode isn't a param: it's not threaded into `_execute_in_sandbox`; the helper reads `repo_spec.json` from `Path(code_path).parent/rlm_state`.)
- **#1:** in `_run_cell_subprocess`, non-blank `cell["command"]` → `["bash","-lc",command]`, `cwd=Path(cell_script).parent` (the `code/` root, so relative `examples/…/run_*.sh` resolves), + exports `OUTPUT_DIR`/`OPENRESEARCH_CELL_ID`. `cwd` kwarg only added on the command branch (keeps fixed-signature Popen mocks passing). No-command path byte-identical.
- **#3:** `write_cell_metrics_from_verl(...)` prefers a `*val*.json`/`*summary*.json`, else regex `(?<![\w/]){key}(?![\w/])[:=\s]+<num>`, **last** match (skips per-dataset sub-keys like `val/success_rate/nq`, matches aggregate `val/success_rate`). **Value-preserving**, **fail-honest** (no match ⇒ `{"status":"failed"}`, no fabricated number). Runner fires it lazily only when `metrics_source.kind=="verl"` AND no `metrics.json` exists after rc==0.
- **⚠ Reviewer flag (from agent C):** the adapter's `eval_provenance.json` sidecar has a *simpler* schema than `backend/agents/rlm/eval_provenance.py` (no per-example `records` — verl exposes only an aggregate). If `OPENRESEARCH_EVAL_PROVENANCE_GUARD` and the verl seam are BOTH on for the same cell, that guard could flag the sidecar as missing `records`. Both flags are independently default-OFF; decide whether the run enables both (the run-spec turns EVAL_PROVENANCE_GUARD on) — **reconcile before Phase 1** (either exempt verl-sourced cells or write a `records`-shaped sidecar).

**⚠ Working-tree entanglement:** `backend/config.py` and `backend/app.py` also carry *pre-existing, unrelated* uncommitted changes (the untracked **external-runs monitor** subsystem — `backend/services/external_monitor/`, `backend/routes/external_runs.py`, `frontend/.../external-runs/`, `configs/external_runs.json`). My #4 config fields are *added to* the already-`M` `config.py`. When committing, isolate the seam work; do not sweep the external-monitor changes into a seam commit.

**Pre-existing test failures to IGNORE (not mine, verified):** `tests/config/test_gcp_orchestrator_settings.py::test_claude_code_oauth_token_prefixed_env_override` (shell-env-shadows-.env), and `tests/{...}/test_accelerator.py`, `test_external_validator.py`, `test_report_validation_stamp.py`.

---

## 2. What remains (ordered)

### 2.1 Wave 2 — Change #2: staged-env passthrough + HF_HOME clobber guard
Spec §3.2. `OPENRESEARCH_CELL_ENV_PASSTHROUGH="HF_HOME,HF_DATASETS_CACHE,…"` forwards named orchestrator env vars into (a) each cell's `child_env` in `gpu_cell_runner._run_cell_subprocess` (re-assert **after** the `child_env={**os.environ}` at ~L631) and (b) the monolithic `SandboxConfig.environment` literal (`primitives.py` ~L3829-3839, so it crosses the docker/runpod boundary). **Clobber guard:** `asset_provisioning.py:340` **unconditionally** overwrites `os.environ["HF_HOME"]` — guard it so a var present in the passthrough allowlist is NOT overwritten. `_sandbox_contract.py:58` is prompt text (agent-authored shell, cells path never sees it) — a guard there is low value; skip or make it a one-line note. Test `tests/rlm/test_cell_env_passthrough.py`: listed vars reach `child_env` with precedence; a pre-set `HF_HOME` survives `asset_provisioning`; unset allowlist forwards nothing (byte-identical). **Files overlap Wave 1** (`gpu_cell_runner.py`, `primitives.py`) — run it AFTER Wave 1 is reviewed/settled, single agent.

### 2.2 Change #7 (recommended NEW seam) — `cells.json` pre-seed
**Why load-bearing:** recon is **definitive — no mechanism pre-seeds `code/cells.json` today**; the executor must author it, and the Foundry executor is documented to stub on exactly that (audit GAP 9). Without this, the run likely fails at manifest authoring. Minimal, paper-agnostic seam: `OPENRESEARCH_CELLS_SEED_PATH=<file.json>` → in `implement_baseline` (right where `_seed_code_from_repo` runs, `primitives.py` ~L2515), if set and `code/cells.json` absent, copy the operator file to `code/cells.json` before the executor runs. Default-OFF/byte-identical when unset. Test ON+OFF. This is the production-grade "operator seeds a manifest, harness guarantees it, executor only verifies" path — works for **any** paper, not just SDAR. *(Alternative if declined: pin `--models executor=sonnet` for the shim — $0 via OAuth but violates the locked "all-Foundry" decision. Recommend #7.)*

### 2.3 STEP 2 — authors' verl config fixes (VM-up; read-then-edit, do NOT sed blind)
On `/mnt/sdar-cache/SDAR/examples/sdar_trainer/` (host-side, pinned by commit):
- `run_alfworld_3b_4gpu.sh`: `ppo_micro_batch_size_per_gpu 32→16`, `rollout.gpu_memory_utilization 0.6→0.5`, add `actor.fsdp_config.optimizer_offload=True`, **REMOVE `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (vLLM `CuMemAllocator` asserts incompatible — proven dead end).
- `run_webshop_3b_patched.sh`: same knobs; strip `expandable_segments`; this is its first-ever training launch.
- `run_search_3b.sh`: unchanged (proven 0.456).
- **Parameterize all three to read `$SDAR_MODEL`** (default the current 3B for back-compat) — the grid needs Qwen3-1.7B *and* Qwen2.5-3B; scripts currently hardcode 3B. Confirm 1.7B fits.
- **⚠ Also inside `PYTORCH_CUDA_ALLOC_CONF`:** `gpu_cell_runner._run_cell_subprocess` sets `child_env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"` (~L641) for EVERY cell — this would re-introduce the exact vLLM-incompatible setting into the verl cells. **Must neutralize for the SDAR execute cells** (e.g. let the passthrough/cell_env override it, or drop it under a command-cell). Reconcile in Wave 2 / STEP 2. **This is a real edge case the command seam introduces.**

### 2.4 Driver execute-mode wiring (`scripts/`) — see §3.D for the exact gaps.

### 2.5 The run — §5 (money-gated).

---

## 3. Discoveries this session (do NOT re-derive)

**A. `env` key collision → `cell_env` (FIXED in the seam + artifacts).** `cell_matrix._AXIS_SYNONYMS` reserves `env` as the *environment axis* (`env`/`environment`/`dataset`/`task`/`benchmark`). The 07-04 runbook §6 cells.json used `env` for BOTH the axis and the env-vars dict (invalid duplicate key). **Per-cell env vars are `cell_env`.** `normalize_cell_axes` shallow-copies unknown keys, so `command`/`metrics_source`/`cell_env` pass through untouched (test-confirmed).

**B. Reproduction mode is NOT visible in `_execute_in_sandbox`** (grep=0). It lives in `rlm_state/repo_spec.json`; #5 reads it from there. Any future execution-path change needing mode must load it similarly (or thread a param).

**C. NO `cells.json` pre-seed hook exists (definitive).** Only `_seed_code_from_repo` (copies the authors' repo verbatim) + `_copy_harness_helpers_to_code_root` (copies stdlib `.py` helpers). `OPENRESEARCH_RESUME_CELLS` re-runs an *already-authored* manifest; it does not seed one. → motivates #7 (§2.2).

**D. Driver wiring gaps (`scripts/sdar_gcp_e2e.sh` → `gcp_sdar_preflight.sh` → `sdar_gcp_run.sh`):**
- The **preflight builds its OWN `runs/.cache/run_spec.json`** via `_spec_add` from *local* env, then passes `--run-spec runs/.cache/run_spec.json`. It **hardcodes `OPENRESEARCH_REPRODUCTION_MODE=adapt`** (preflight ~L619) and `USE_AUTHOR_REPO=1`/`REPO_URL=ZJU-REAL/SDAR`. → **To run execute mode, make that mode line respect an override** (e.g. `${OPENRESEARCH_REPRODUCTION_MODE:-adapt}`) and export the execute vars (from `configs/sdar_execute_run_spec.json`) locally before `run`, OR point the driver at that run-spec file directly. `_spec_add` forwards local env, so the cleanest is: `source`/export the execute vars locally, drop the hardcoded `adapt`.
- **`--scope-spec` defaults to `{"models":["Qwen3-1.7B"]}`** (single model) in `sdar_gcp_run.sh` (~L219) — override `OPENRESEARCH_SDAR_SCOPE_SPEC` for the 2-model grid, and note **cells.json (not scope-spec) is the baseline/env matrix**; with #7 the seeded cells.json IS the grid.
- **No `--max-run-gpu-usd` flag** in any driver script — but `OPENRESEARCH_MAX_RUN_GPU_USD=400` (in the run-spec, via `RunBudget`) is the enforced equivalent. OK.
- **Driver does NOT set `HF_HOME=/mnt/sdar-cache/hf`** — preflight only *relays* local `$HF_HOME` (~L649). → set `HF_HOME=/mnt/sdar-cache/hf` in the run-spec (or export on the VM) so #2's passthrough forwards it. Preflight already defaults `OPENRESEARCH_SDAR_CACHE_ROOT=/mnt/sdar-cache` and relays `ALFWORLD_DATA`.
- `--run-spec` **is** honored by `sdar_gcp_run.sh` (L228-231, and preflight passes it at ~L691). Autostop = `self_stop` EXIT-trap → GCS upload (`OPENRESEARCH_SDAR_REPORT_GCS`) → `shutdown`, gated by `OPENRESEARCH_SDAR_NO_AUTOSTOP` (keep unset/0 = autostop ON). Code sync = rsync→`gcloud compute scp` (NOT git).

**E. Operational reliability lesson:** background subagents launched with `run_in_background:true` were **lost** when the Claude Code process exited mid-turn (no completion record; nothing had hit disk). Re-dispatched **synchronously** (`run_in_background:false`) and they completed reliably in-turn. → **For critical implementation, prefer synchronous subagents (or commit checkpoints frequently).**

---

## 4. Coordinates (essentials; full table = 07-04 handoff §2)

- Repo: `/home/abheekp/openresearch`, branch `reconcile/grounded-self-improvement-on-main`. Push **`deepinvent` only**. Git identity `lolout1`. **No Co-Authored-By trailer.**
- GCP: project `deepinvent-ext-ut`, acct `abheek@deepinvent.ai`, `export CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud`.
- VM **`sdar-2model-a`**, zone **`us-central1-a`**, `a2-ultragpu-4g` (4×A100-80GB), **STOPPED** (restart to use). Cache disk `sdar-cache-a` → `/mnt/sdar-cache` (persists).
- Restart: `CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud gcloud compute instances start sdar-2model-a --zone us-central1-a --project deepinvent-ext-ut`
- Stop (halt cost): `… gcloud compute instances stop sdar-2model-a --zone us-central1-a --project deepinvent-ext-ut --discard-local-ssd=false`
- Staged cache: SDAR repo `@9f2ce6a82a90cc5a334d73f188c16df2c4107d80`; conda envs `sdar`/`retriever`/`verl-webshop`; `HF_HOME=/mnt/sdar-cache/hf` (Qwen2.5-3B-Instruct + Qwen3-1.7B + Qwen2.5-7B). GCS `gs://deepinvent-ext-ut-sdar-runs/`.
- All LLM tiers = Foundry deployment `grok-4.3` (script forces `gpt-chat-latest`); OAuth-free. `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` assume **dead/empty**.
- **Only proven signal:** authors' verl Search-QA-3B = `val/success_rate` **0.456** @ 150 steps.

---

## 5. Run plan (money-gated)

**⛔ Do NOT restart the VM / spend GPU $ without an explicit operator go.** The VM was found idle burning ~$20/hr and deliberately stopped; the grid caps at ~$400.

1. Land Wave 2 (#2) + #7; reconcile the `PYTORCH_CUDA_ALLOC_CONF` + eval-provenance-schema edges (§2.1/§2.3); wire the driver (§3.D). Commit the seam milestone; `ruff` + off-state suites green.
2. Restart `sdar-2model-a`; STEP-2 verl config edits on the staged disk (read-then-edit). Export `HF_HOME=/mnt/sdar-cache/hf`.
3. **Phase 1 (gate):** run ONLY `configs/sdar_execute_cells_phase1.json` (Search-3B). **PASS = harness-driven `val/success_rate` ≥ 0.40 (target 0.456) AND evidence guards clean (env_liveness, eval_provenance, zero_metrics) AND external validator no-veto.** A miss ⇒ seams/adapter wrong; debug before grid spend (~$30 slice).
4. **Phase 2 (grid):** `configs/sdar_execute_cells.json` (6 cells, serialize on 4×A100), `OPENRESEARCH_MAX_RUN_GPU_USD=400`, autostop ON. Pull `gs://deepinvent-ext-ut-sdar-runs/<PROJECT_ID>/`.

Artifacts ready: `configs/sdar_execute_run_spec.json` (add `HF_HOME` if driving the driver), `configs/sdar_execute_cells_phase1.json`, `configs/sdar_execute_cells.json`.

---

## 6. Working discipline / prompting / style (for the new session)

- **Roles:** Opus authors/owns the plan + **reviews EVERY diff** (verify the diff, not the summary); **Sonnet executes** (impl code included) against a tight spec. Do not substitute a Sonnet "reviewer."
- **Delegation:** fan out Sonnet implementers by **file-disjoint** clusters to run in parallel; **prefer synchronous** (`run_in_background:false`) for critical work (see §3.E). Use `general-purpose` subagent, `model: sonnet`.
- **TDD:** write the per-seam guard test first. **Every change default-OFF / byte-identical** when its new flag/field is absent, with a hermetic ON+OFF test. Mirror existing test patterns.
- **Recon before code:** read the seam sites; the code is named by function. Use read-only Explore agents for multi-file mapping to preserve the driver's context.
- **Surgical:** every changed line traces to the request; match local style; don't refactor adjacent code; flag unrelated dead code, don't delete it.
- **Verify:** `.venv/bin/python -m pytest …`; `uvx ruff@0.15.16 check <files>` (Python 3.12 via `uv sync --frozen`). Keep the cells-route / execute-mode / role-model off-state suites green.
- **Git:** commit **infrequently at milestones** (few substantial commits); no Conventional-Commit prefixes — descriptive present-tense headlines carrying what+symptom+resolution; **push `deepinvent` only**; identity `lolout1`; **no AI/Co-Authored-By trailer**.
- **Money/irreversible:** surface a checkpoint before VM restart or GPU spend; autostop ON; never route a tier to a dead key.
- Use `/implement` for implementation work (not Codex).

---

## 7. Production-grade & paper-agnostic framing

The seams are **generic, not SDAR-special** — they upgrade the harness for *any* execute-mode paper and stay byte-identical off:
- **Cell-job generalization** (`command`/`cell_env`/`metrics_source`): any paper whose repo ships a runnable pipeline can be driven verbatim behind a launcher, with a value-preserving metrics adapter — no from-scratch reimplementation.
- **Local-repo reuse + commit pin:** deterministic, offline-capable provisioning from a pre-staged/pinned tree (reproducibility: capture the SHA).
- **Execute-owns-deps:** the authors' own env (conda/venv/Docker) owns numerics; the harness stops fighting it.
- **Fail-loud (#6):** execute never silently degrades to a fabricated from-scratch result — honest failure over false success (the project's red line: evidence, not grade).
- **Pre-seed (#7):** operator declares the grid once; the harness guarantees it regardless of executor quality — the robustness lever that makes "any paper, any executor" work.

Keep the honesty guards (`ZERO_METRICS_GUARD`, `EVAL_PROVENANCE_GUARD`, `ENV_LIVENESS_GATE`, `EXTERNAL_VALIDATOR`) ON for the run — they are the deterministic fitness signal.

---

## 8. Paste-ready kickoff prompt

> Read `docs/runbooks/2026-07-05-sdar-execute-mode-session-handoff.md` in full (its §0–§3 are the state; §6 the discipline), then skim its three companions. Wave 1 (seams #1,#3,#4,#5,#6 + tests + `configs/sdar_execute_*.json`) is DONE, green (`177 passed`), and UNCOMMITTED. **First:** Opus deep-hunk review of the 6 Wave-1 source diffs (esp. the eval-provenance-schema flag and the `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` re-injection edge in §2.3). **Then** implement, TDD, file-disjoint, **synchronous Sonnet** agents: Change #2 (env passthrough + HF_HOME guard) and Change #7 (`OPENRESEARCH_CELLS_SEED_PATH` pre-seed) — reconcile the two edges above. Commit the seam milestone (isolate from the unrelated external-monitor uncommitted work; push `deepinvent`; identity `lolout1`; no Co-Authored-By). Then wire the driver for execute mode (§3.D) and bring me a money checkpoint before restarting `sdar-2model-a` — Phase 1 = Search-3B (`configs/sdar_execute_cells_phase1.json`), gate ≥0.40 (target 0.456) + guards clean, before the $400 grid. Opus plans + reviews every diff; Sonnet executes; default-OFF/byte-identical; autostop ON.
