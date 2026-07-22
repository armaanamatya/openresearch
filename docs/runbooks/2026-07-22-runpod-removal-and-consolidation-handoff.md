<!-- doc-meta: status=historical; created=2026-07-22 -->
# Handoff — RunPod removal + branch consolidation (2026-07-22)

Session handoff for the repo-cleanup / cloud-posture campaign. Everything below is
committed and recoverable; the working tree is clean (only protected untracked
`runs_logs/` + `.demo_backups/` remain, deliberately). Resume from **"Remaining work."**

## TL;DR
- **Authoritative trunk = `chore/repo-consolidation`** (= `origin/main` + 124 UNPUSHED local commits: durable controller, CPU lane, Track E scorecard, AWS/EKS, scheduler/evidence/lifecycle merges). Lives in worktree `/private/tmp/scientific-article-generator-consolidation`.
- **Integration branch = `integrate/degke-runpod-on-trunk`** (checked out in the MAIN worktree `/Volumes/CS_Stuff/scientific_article_generator`, which has the `.venv`). This is where all cleanup lands. End goal: fast-forward `origin/main` → this branch (clean FF, no force-push). **The operator runs the `origin/main` push** (stop-before-push).
- RunPod/Brev/Railway **removed**; GKE **parked** (in-background, fail-loud until perms); Azure/AKS + AWS/EKS + GCP-single-VM **live**; dashboard default sandbox **local**.

## Cloud posture (DECIDED — do not re-litigate)
| Backend | State |
|---|---|
| RunPod / Brev / Railway | **Removed** |
| GKE (`--sandbox gcp`/`gke` → `GkeJobBackend`) | **Parked** — fail-loud `RuntimeError` unless `OPENRESEARCH_ALLOW_GKE=1`. Operator wants it "in the background, not doing actual work yet — we don't have perms." KEEP parked; revive when IAM lands. |
| Azure / AKS | **Live** (fully isolated — shared `k8s_job_backend.py` preserved) |
| AWS / EKS | **Live** (kept from trunk's +124) |
| GCP single-VM (`VmComputeProvider`, `campaign --billing-sandbox gcp`) | **Live** — the supported GCP GPU path |
| Default sandbox | **local** (CLI/RLM already `auto`→docker/local; HTTP/UI default now `local`) |

## HARD RULE (operator directive)
**NEVER delete past runs or logs** — `runs/`, `runs_logs/`, `gcp_logs/`, `_archive/`, any `prj_*`/`pb_*` artifacts. `runs_logs/` + `.demo_backups/` are untracked and must stay untouched. Cleanup = branches / dead code / stale docs / caches ONLY.

## What was done
### Branch prune (remote 20 → 7; local trimmed)
Deleted (all recoverable via `backup/pruned/*` tags): absorbed branches (chore/cleanup-sweep, worktree-consolidated-cloud-lifecycle-tier1), old May–Jun experiments (gepa×3, rlm-wedge-hardening, harness-lifecycle-driver, integrate-perf-accelerator, pipeline-validation), superseded teammate branches (localqwen, integrate/azure-gcp-to-main, pre-import-snapshot), chore/onboarding-cleanup.
**KEPT:** `origin/main`, `origin/remove-runpod-railway-cleanup` (source of the removal), the 3 recent-July branches, and **2 substantive lolout1 branches** (`feat/azure-bicep-canonical-aoai-hardening` 17c, `feat/grounded-self-improvement-harness-reliability` 48c — core evidence/anti-fabrication work; NOT bullshit, held pending confirmation).

### RunPod/Brev/Railway removal (6 commits on the integration branch)
```
c64c8c8b Close terminal run process status          (consolidate authoritative-scheduler-runtime, 1 commit)
b067e564 Remove obsolete RunPod tests; regenerate flag registry
6da38286 Finish RunPod cleanup: start.sh default, orphaned preflight, frontend refs
66e0639b Complete RunPod/Brev removal in backend + frontend status surface
0b7c19ab Remove RunPod/Brev/Railway on the consolidation trunk; keep AWS/EKS + Azure/AKS; park GKE
6f7b11bb Restore COEFFICIENTS_KEY re-export from deterministic_leaf_checker  (fixes a TRUNK regression)
```
Notable: `_COST_BEARING_SANDBOXES → ("gcp","azure","aws")`; deleted the 6 backend files + railway.json + runpod test files; dead frontend runpod-status surface removed; `start.sh` default `runpod→local`; `flags.md` regenerated. **Legit keeps** (do NOT strip): `runpod_id` SSE schema fields, `OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH` (local data-root), `runpod/pytorch` public base image, `runpod_cloud_type` config field.

### ⚠️ Concurrent-session incident (important)
A SECOND agent session was working the SAME branch mid-session and pushed `0b7c19ab` (built on my `6f7b11bb`) to `origin/integrate/degke-runpod-on-trunk`. Operator chose "I take over" → **that other session must stay stopped.** My local `integrate/degke-runpod-on-trunk` has since diverged ahead of `origin/integrate/degke-runpod-on-trunk` with commits 66e0639b, 6da38286, b067e564, c64c8c8b. **Before resuming, confirm no other session is live** (`git status`, check file mtimes) and that HEAD hasn't moved unexpectedly.

## Test state (baseline to beat)
Full suite after removal: **10165 passed, 27 failed**. Removal adds **ZERO** new failures. The 27:
- **5 were obsolete runpod tests** → already removed in `b067e564`.
- **22 PRE-EXISTING trunk debt (do NOT chase as if you caused them):** 18 evidence-layer (`deterministic_leaf_checker`, `coefficient_provenance`, `binding`, `rubric_gen`, `coverage_pct`, `credential_vault`, `live_runs_proc_environ`); `test_flag_decision_manifest` (`OPENRESEARCH_K8S_COLLISION_GUARD` needs a `configs/flag_decisions.json` entry); `test_all_doc_citations_resolve` (34 dead citations in root CLAUDE.md); `test_no_tracked_but_gitignored_files` (4 tracked run dirs matching .gitignore — LEAVE per never-delete rule).
- Env for tests: `OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python -m pytest tests/ -q -p no:cacheprovider` (~7.5 min). Do NOT pipe through `tail` for a background run — it eats the summary; redirect full output to a file.

## Remaining work (resume here)
1. **Consolidate 2 heavy branches** onto `integrate/degke-runpod-on-trunk`. Resolve conflicts keeping: the removal (runpod gone), the GKE-park, the trunk's +124, AND the branch's new code. Full suite gates. Read-only preview: `git merge-tree --write-tree --name-only HEAD <branch>`.
   - `origin/scheduler-authority-runtime` (2 commits: "Complete scheduler evidence gate and EKS runtime", "Harden cloud cell scheduling"). ~20 conflicts: `primitives.py`, `k8s_job_backend.py`, `k8s_job_cell_runner.py`, `config.py`, `gpu_resolver.py`, `gpu_capacity.py`, `scheduler_evidence.py` (add/add), `branch_lineage.py`, `reproduction_campaign.py`, `live_runs.py`, `runtime/__init__.py`, cli.py, README, CLAUDE.md, flags.md, frontend upload-view, several tests + stale-doc modify/deletes (resolve doc modify/deletes toward DELETE unless the doc is current).
   - `gke-local-transport` (10 commits: GKE **local** transport fallback + honest Lab-UI rendering — compatible with parking since it runs cells locally). Conflicts: `k8s_job_cell_runner.py`, CHANGELOG, README, flags.md, `docker/gke-cell-base/*`, frontend hooks, `best_runs/*`, several 2026-07-07 stale-runbook modify/deletes.
   - `authoritative-scheduler-runtime` is ALREADY consolidated (c64c8c8b).
2. **Fix 34 dead doc citations** in root `CLAUDE.md` (`tests/test_claude_md_fidelity.py::test_all_doc_citations_resolve`) — repoint or remove references to docs that were deleted/reorganized. CLAUDE.md is guarded by fidelity tests; edit carefully.
3. **Prune consolidated local branches** (backup-tag first): once `scheduler-authority-runtime` + `gke-local-transport` are merged, their local/remote branches + `authoritative-scheduler-runtime` become redundant. Also stale worktrees at `.claude/worktrees/{authoritative-scheduler-runtime,gke-local-transport,integration-validation}` can be `git worktree remove`d (check each is clean first).
4. **Land trunk (operator runs):** fast-forward `chore/repo-consolidation` → `integrate/degke-runpod-on-trunk`, then FF `origin/main` → that, then push. Also decide whether to delete `origin/integrate/degke-runpod-on-trunk` + `origin/remove-runpod-railway-cleanup` afterward.

## Recovery / safety nets
- Backup tags: `backup/repo-consolidation-20260722`, `backup/runpod-removal-20260722`, `backup/local-main-20260722`, and 18× `backup/pruned/*` (every deleted branch tip; re-push any with `git push origin <sha>:refs/heads/<name>`).
- Insurance patch of the originally-adopted worktree removal: `/tmp/mystery-uncommitted-worktree.patch`.
- The 124 unpushed trunk commits exist ONLY on this disk (tag `backup/repo-consolidation-20260722`) + in `/private/tmp` worktree — consider pushing a backup ref before any risky op.

## Key files / entry points
- Sandbox enum: `backend/agents/execution.py` (`SandboxMode`). Backend factory + GKE-park guard: `backend/agents/rlm/primitives.py` (`_backend_for_sandbox_mode`). Config Literals + fields: `backend/config.py`. Cost-bearing: `backend/agents/baseline_implementation.py` (`_COST_BEARING_SANDBOXES`). Shared K8s base (do NOT break — AKS/GKE/EKS share it): `backend/services/runtime/k8s_job_backend.py`, `backend/agents/rlm/k8s_job_cell_runner.py`. Cross-cloud scheduler (gcp/azure only): `backend/services/runtime/cloud_failover.py`.
