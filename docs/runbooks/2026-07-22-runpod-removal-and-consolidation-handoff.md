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

## Remaining work (resume here — fresh `/clear`ed session recommended; agent infra dropped 2 subagents mid-merge, and hand-resolving the one core file wants clean context)

### A. `authoritative-scheduler-runtime` — DONE (commit c64c8c8b).

### B. `origin/scheduler-authority-runtime` — RECOMMEND SKIP (mostly redundant). Evidence: `git cherry chore/repo-consolidation origin/scheduler-authority-runtime` = **10 of 12 commits already absorbed**; `eks_job_backend.py`, `asha_authority_gate.py`, `scheduler_evidence.py` are ALL already on HEAD. Its 2 "unique" commits (fee8a5c6, 66786bf8) are refinements built atop the 10 absorbed ones → cherry-picking fee8a5c6 produced **17 conflicts** across core files (primitives.py, k8s_job_backend.py, gpu_resolver.py, etc.) for marginal net-new value, with real risk of regressing the trunk's existing EKS/scheduler. If desired later, review the 2 commits' diffs by hand and port only genuinely-missing hunks — do NOT bulk cherry-pick.

### C. `origin/gke-local-transport` — WORTH DOING, 90% mechanical (10 genuinely-unique commits, 0 absorbed). Confirmed merge shape:
```
git merge --no-commit --no-ff origin/gke-local-transport   # 36 files auto-merge clean; 4 conflicts + 5 stale-doc modify/deletes
```
- **5 stale docs → keep DELETED:** `git rm -f docs/runbooks/2026-07-07-{all-runs-triage-and-hardening-handoff,lab-provider-sandbox-foundry-wiring-handoff,sdar-gke-foundry-run-handoff,tool-rl-gke-reproduction-handoff}.md docs/runbooks/known-issues-and-monitoring.md`
- **CHANGELOG.md, README.md → union** (both sides add different entries; keep both).
- **docs/reference/flags.md →** take either side then regenerate: `OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python scripts/gen_flag_registry.py && git add docs/reference/flags.md`.
- **backend/agents/rlm/k8s_job_cell_runner.py → the ONE hard file (8 hunks).** The branch's additions are entirely behind default-OFF flag `OPENRESEARCH_GKE_LOCAL_TRANSPORT` and BYTE-IDENTICAL when off — that's the invariant. Resolve each hunk **keep-both**: preserve HEAD's durable-controller code AND add the branch's local-transport code (`_gke_local_transport_enabled`, `_local_transport_active`, `_LOCAL_WORKSPACE` staging/collection). Hunk 1 (~652–755) is a clean co-located keep-both (HEAD `_credential_env_vars()` + branch flag block). Confirm `grep -c '^<<<<<<<\|^>>>>>>>' backend/agents/rlm/k8s_job_cell_runner.py` == 0. Then: import smoke + `pytest tests/agents/rlm/test_k8s_job_cell_runner.py tests/agents/rlm/test_gke_local_transport.py tests/services/runtime/test_gke_*.py -q`, then the **full suite** (expect the same ~21 pre-existing failures, no new ones), then commit.
2. **Fix 34 dead doc citations** in root `CLAUDE.md` (`tests/test_claude_md_fidelity.py::test_all_doc_citations_resolve`) — repoint or remove references to docs that were deleted/reorganized. CLAUDE.md is guarded by fidelity tests; edit carefully.
3. **Prune consolidated local branches** (backup-tag first): once `scheduler-authority-runtime` + `gke-local-transport` are merged, their local/remote branches + `authoritative-scheduler-runtime` become redundant. Also stale worktrees at `.claude/worktrees/{authoritative-scheduler-runtime,gke-local-transport,integration-validation}` can be `git worktree remove`d (check each is clean first).
4. **Land trunk (operator runs):** fast-forward `chore/repo-consolidation` → `integrate/degke-runpod-on-trunk`, then FF `origin/main` → that, then push. Also decide whether to delete `origin/integrate/degke-runpod-on-trunk` + `origin/remove-runpod-railway-cleanup` afterward.

## Recovery / safety nets
- Backup tags: `backup/repo-consolidation-20260722`, `backup/runpod-removal-20260722`, `backup/local-main-20260722`, and 18× `backup/pruned/*` (every deleted branch tip; re-push any with `git push origin <sha>:refs/heads/<name>`).
- Insurance patch of the originally-adopted worktree removal: `/tmp/mystery-uncommitted-worktree.patch`.
- The 124 unpushed trunk commits exist ONLY on this disk (tag `backup/repo-consolidation-20260722`) + in `/private/tmp` worktree — consider pushing a backup ref before any risky op.

## Key files / entry points
- Sandbox enum: `backend/agents/execution.py` (`SandboxMode`). Backend factory + GKE-park guard: `backend/agents/rlm/primitives.py` (`_backend_for_sandbox_mode`). Config Literals + fields: `backend/config.py`. Cost-bearing: `backend/agents/baseline_implementation.py` (`_COST_BEARING_SANDBOXES`). Shared K8s base (do NOT break — AKS/GKE/EKS share it): `backend/services/runtime/k8s_job_backend.py`, `backend/agents/rlm/k8s_job_cell_runner.py`. Cross-cloud scheduler (gcp/azure only): `backend/services/runtime/cloud_failover.py`.
