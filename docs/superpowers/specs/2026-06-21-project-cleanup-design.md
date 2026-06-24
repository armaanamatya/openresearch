# Design — OpenResearch project cleanup (research-grounded, phased)

> **Status:** Design (brainstorming output) · 2026-06-21 · grounded in a 5-domain
> parallel research pass (dead-code, docs, naming, git-hygiene, test/flake).
> **Sequencing:** safe work NOW (off the trunk); tree-touching work AFTER the 6 open
> PRs (#107/#109/#111/#112/#114/#115/#116) merge, to avoid conflicts. Non-breaking
> throughout (keep the `REPROLAB_` bridge). Archive-don't-delete for files not authored
> here. YAGNI — no unrelated refactoring.
>
> **Research INVERTED three earlier assumptions — record them:**
> 1. The **env/db rename is already ~90% done.** DB default is already
>    `sqlite:///openresearch.db` with the exact non-breaking legacy fallback already
>    implemented (`config.py:50` + `:714-728`); the cross-process cell-seam is already
>    canonical `OPENRESEARCH_CELL_*` on BOTH ends (zero `REPROLAB_CELL_*` in code). The
>    real remaining work is tiny (4 code reads + paired tests).
> 2. The **big git bloat is `best_runs/`** (12 MB / 432 tracked files), not branches —
>    and its right home is the **Run/Test Bank** (`docs/superpowers/prompts/2026-06-21-run-test-bank-megaprompt.md`), not deletion.
> 3. Two **rename-safety mechanisms are UNTESTED** (`_apply_legacy_env_aliases`,
>    `_fall_back_to_legacy_sqlite_db`) — add those tests FIRST.

---

## ⚠️ Ownership gate (added 2026-06-21 after advisor review) — Phase A is NOT solo-executable
This is a **shared repo with an active second human** (`lolout1 <appradhann@gmail.com>`).
Branch merge-status and file-collision were not the binding constraint — **ownership is.**
Verified authorship:
- `feat/azure-aks-gpu`, `feat/gcp-gke-backend` → **lolout1** (do NOT delete unilaterally).
- `bes` → "Aayush C Baniya" (a different GitHub handle than the working account
  `thisisaayushbaniya`) — ambiguous; do NOT delete.
- **#110** (the PR A2 would close) → **lolout1's** PR; its content currently lives ONLY on
  the unmerged `feat/grounded-harness-integration` (#116). Closing it now could destroy the
  canonical PR if #116 reworks or never lands.
- The 3 root `.md` (`bes_integration.md`, `claude_gcp_sdar_handoff.md`, `issues.md`) →
  **lolout1's**, authored 3–5 days ago.
- `feat/gepa-integration` → the operator's own (`thisisaayushbaniya`).
**Rule:** every Phase A item is **"propose to the owner," not "do solo."** The only
solo-safe, in-bounds work in this whole design is B3's two additive safety tests
(`_apply_legacy_env_aliases`, the db fallback) — and even those stack on the unmerged tower,
so they are a choice, not a default. Net: **no cleanup is executable solo right now** — it is
all owner-coordination-gated (Phase A) or merge-gated (Phase B).

## Phase A — Safe now · one tiny PR off the trunk · zero conflict

**A1. Branch prune** (merged into trunk AND no open PR):
- Prune: `bes` (merged to trunk + main), `feat/azure-aks-gpu`, `feat/gcp-gke-backend`
  (local + remote). **Protected (open PR — do NOT prune):** `feat/grader-fidelity`
  (#104), `feat/grounded-self-improvement-harness-reliability` (#110), + all #107/#109/
  #111/#112/#114/#115/#116 branches. **Leave + flag:** `feat/gepa-integration`
  (3-week unmerged unique work; an `archive/gepa-*` tag set already exists) and the two
  `integrate/*` branches (active/landing). Tag clutter: 16 `archive/*` snapshot tags —
  consider a `refs/archive/` namespace later (not now).
**A2. PR hygiene:** close **#110** as superseded by #116 (content integrated there);
  flag **#104** (grader-fidelity already on trunk) for the owner to close.
**A3. Root tidy (collision-free files only):** `git mv bes_integration.md`,
  `claude_gcp_sdar_handoff.md`, `issues.md` → `docs/archive/`. **Defer `gcp_info.md`**
  (collides with #116; already gitignored anyway).

## Phase B — After the stack merges · dedicated cleanup PR(s) off the new `main`

**B1. CLAUDE.md de-drift (highest doc value).** De-duplicate the repeated whole blocks
(this alone fixes most contradictions, because the stale copies ARE the duplicates):
"Feature flags" ×2 (lines ~97-108 / 109-117), "One-GPU-per-cell" ×2 (~188-191 / 208-211),
the Sandboxes intro ×2, the primitive list + 17-vs-12 reconcile ×2. Then fix the 7
contradictions to the **code-verified** value:

| # | Claim | Correct value (per code) |
|---|---|---|
| 1 | SQLite DB default | `openresearch.db` (`config.py:50`; `reprolab.db` is fallback-only) |
| 2 | RunPod default image | `cuda-devel ~18 GB` (`config.py:272`; runtime-image revert `88c45b0`) |
| 3 | `RUNPOD_CLOUD_TYPE` default | `SECURE` (`config.py:275`) |
| 4 | Docker-daemon prereq | line ~194 correct (`runpod` short-circuits by default; only `docker`/`auto` need it) |
| 5 | RunPod API-key var | `OPENRESEARCH_RUNPOD_API_KEY` canonical (`REPROLAB_` bridged) |
| 6 | BES/flag prefix | `OPENRESEARCH_*` canonical (the `REPROLAB_*` block is the stale duplicate) |
| 7 | Primitive count | **17** bound (`tests/rlm/test_registry.py`); "12 core domain" is the subset framing |

Also: frontend `OPENRESEARCH_BACKEND_URL` required-vs-optional dup (lines 43/44); refresh
both `last-verified` doc-meta stamps. Use `system_overview.md` (cleaner, says "17") as the model.

**B2. Docs consolidation.** Archive **34 of 43 runbooks** (point-in-time handoffs/run-logs)
to `docs/archive/runbooks/` with a one-line index; keep the 9 non-dated operator docs live.
**5 of the 34 are cited as canonical** (CLAUDE.md:213/216/248/254/261/266, system_overview:248)
— update the citing line FIRST, then archive, or you create broken xrefs. Second pass: the
dated-handoff specs/plans (`2026-05-31-backend-core-remediation-handoff.md`, the `2026-05-23-*`
plan set, the triplicate `2026-06-14-sdar-on-azure-run`).

**B3. Env/name non-breaking canonicalize (TDD).**
- **FIRST add the two missing safety tests:** (a) `_apply_legacy_env_aliases` bridge
  (`REPROLAB_FOO`↔`OPENRESEARCH_FOO`); (b) `_fall_back_to_legacy_sqlite_db` (both-absent→new,
  only-legacy→legacy, explicit URL→untouched).
- **Then flip the 4 code reads + their paired post-import test setters in LOCKSTEP** (the
  import-timing trap — a var set after import isn't bridged): `REPROLAB_FLOOR_HARD`
  (`forced_iteration.py:627,699`), `REPROLAB_RLM_ROOT_MODEL_NAME` (`models.py:713`),
  `REPROLAB_FINALIZE_REGRADE` (`finalize_regrade.py:36`), `REPROLAB_LEAF_TRIAGE`
  (`leaf_triage.py:39`) — each with its `tests/...` setter.
- **Fix `.env.example:270-276`** (still legacy `REPROLAB_*`) + the false `config.py:24`
  claim that it's migrated. Tidy stray doc spellings (`CLAUDE.md:209` `REPROLAB_CELLS_ROUTE_RETENTION`).
- **KEEP (these ARE the non-breaking compat — do NOT touch):** the bridge (`config.py:13-33`),
  the pydantic `AliasChoices` legacy keys (`config.py:107/115/123/261/705`), the CLI
  legacy-prefix accept (`cli.py:871`), the two compat-path tests
  (`test_cli_run_spec.py:60-69`, `test_hermes_provider_settings.py:206-207`).
- **DANGER — do NOT rename (external Terraform/infra contracts):** the `reprolab` strings in
  `config.py` cloud defaults (Azure blob `reprolab-artifacts`/`reprolab-cache`, K8s namespace
  `reprolab`, SA `reprolab-sa`, ACR/AR `.../reprolab`, node labels `reprolab/sku`,
  `/mnt/reprolab-cache`) — they must match deployed infra. Leave for a coordinated infra change.

**B4. Dead-code removal (each verify-first — may be staged features on a `feat/` branch).**
- `backend/services/orchestration/` cluster (4 files, 387 lines: blackboard, task_lifecycle,
  spawn_policy, delegation — test-only, prod-orphaned, one cohesive removal). Highest value.
- `backend/services/runs/purge.py` (superseded by `retention.py` — clearest safe-delete).
- `backend/agents/resilience/engine.py` (541 lines, not re-exported, zero importers — verify
  not an intended entrypoint). `backend/evals/elo.py`, `backend/services/pricing/backfill_timing.py`
  (dormant features — confirm with owner). Scripts: `recompute_adam_rubric.py`, `_monitor_stages.py`.
- **Do NOT touch (false positives — live via dynamic load):** `gpu_holder.py` (subprocess `-m`),
  the 5 sandbox `*_env.py`/`agentic_rollout.py`/`sdar_env_base.py` (copied into the cell sandbox),
  rlm/ private helpers, rdr/hybrid packages.

**B5. Flake + test debt.**
- De-flake `tests/evals/test_leaf_scorer_parallel.py:176-185` (wall-clock parallel<serial under
  a thread pool — same class as the already-fixed `test_gpu_cell_runner_heterogeneous`).
- `tests/rlm/test_build_environment_timeout.py:11` xfail tracks a REAL bug (`build_environment`
  `pool.shutdown(wait=True)` hangs on a wedged worker; needs `cancel_futures=True`) — schedule the
  code fix, then flip the marker. (Other MED flake-risk timing tests are listed in the research; harden opportunistically.)

**B6. Git de-bloat — via the Bank.** Migrate `best_runs/` (12 MB / 432 files) into the Run/Test
Bank, remove it from git, add the `.gitignore` line (the lone gap). See the Bank mega-prompt.

**B7. Doc-flag backfill (optional).** 230 live `OPENRESEARCH_*` flags are undocumented and 2
documented flags (`OPENRESEARCH_BACKEND_URL`, `OPENRESEARCH_BES_ADAPTIVE_SKIP_SCORE`) are never
read. Backfill the high-value clusters (secrets, cloud-backend, watchdog) into CLAUDE.md; align
`.env.example` to the `OPENRESEARCH_` prefix. Large; lowest priority.

## Out of scope (YAGNI)
Infra `reprolab` Terraform-contract renames · rubric/harness logic changes · touching the 6
open PRs' content · the Run/Test Bank itself (its own mega-prompt).

## Success criteria
Remote branches ≈ open-PR set + main/trunk · root holds only README/CHANGELOG/CLAUDE/system_overview ·
CLAUDE.md self-consistent (no contradictory duplicates), code-verified values · `best_runs/` out of
git (−12 MB) and in the Bank · `REPROLAB_` still works (bridge + AliasChoices + CLI accept intact,
now test-covered) · dead-code cluster removed after verify · `ruff` + full hermetic suite green on every PR.
