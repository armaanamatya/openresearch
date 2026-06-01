# Backend-Core Remediation — Merge & Continuation Handoff (2026-06-01)

> **You are picking up two jobs, in order:** (1) **elegantly merge** the `harden/root-harness`
> backend-core remediation work into the active mainline (`5.30.26_sdar`, the branch checked
> out in the real repo `/home/sww35/openresearch`), reconciling the 15 commits that landed on
> `5.30.26_sdar` *after* the fork; then (2) **finish the rest of the backend-core backlog**
> (Phase B remainder + Phases C–F). **Strictly improve existing logic only — break nothing.**
>
> This doc is self-contained. Read it top to bottom. The detailed evidence for each finding
> lives in `docs/audits/2026-05-31-backend-core-opportunity-backlog.{md,json}` and the runnable
> per-finding prompts in `docs/superpowers/specs/2026-06-01-backend-core-task-prompts.md`
> (both already committed on `harden/root-harness`). Memory: `[[backend-core-audit]]`,
> `[[root-harness-hardening]]`, `[[commit-attribution-preference]]`.

---

## 0. TL;DR — do this first

1. **Decide nothing about scope** — the goal is a single integrated mainline that has BOTH the
   sdar/OOM/cell-runner work AND the harden remediation, with zero regressions.
2. **Merge direction:** integrate `harden/root-harness` (34 commits) into `5.30.26_sdar`. Only
   **4 backend files** conflict; everything else is clean. **F-30 is already superseded by sdar
   — drop it.** (§3.)
3. **Then continue the backlog**: Phase B remainder (**F-04, F-05, F-11, F-33, F-35**) → Phase C
   (cost) → D (budget/runtime = design P4) → E (security; F-37 SSRF is the one real risk) → F
   (cleanup). (§4.)
4. **Guardrails are unchanged** (§5): one tested commit per finding, full-suite gate, observe-first
   for score-movers, commit as `lolout1` **no co-author trailer**.

---

## 1. Where everything is

| Thing | Value |
|---|---|
| Real repo (primary checkout) | `/home/sww35/openresearch` — **on branch `5.30.26_sdar`** (HEAD `6ac50d5`) |
| Remediation worktree | `/home/sww35/openresearch-harden` — **on branch `harden/root-harness`** (HEAD `2daa70c`) |
| RL worktree (unrelated) | `/home/sww35/openresearch-rl-trl` — `feat/rl-trl-grpo-engine` (ignore) |
| Test interpreter (worktree has **no venv**) | `/home/sww35/openresearch/.venv/bin/python` (cwd must be the worktree so `backend` imports from it) |

### Branch topology (the whole point)
```
            2acbc8d  main  ("chore: docs cleanup + fix all frontend lint errors")
               │     └── main has NOT moved; it is 0-behind BOTH branches below.
               │
            …   (shared history)   …
               │
            9ba5dec   ← FORK POINT (merge-base of harden vs 5.30.26_sdar)
              ╱ ╲
     +34 commits   +15 commits
   harden/root-harness        5.30.26_sdar
   (2daa70c)                  (6ac50d5)
   = P0–P3 hardening          = SDAR one-GPU-per-cell runner, OOM remediation,
     + 47-finding audit docs    GPU capacity, lab-ui fixes, paper-title parse
     + THIS SESSION's 11         (the "new changes made while you worked in the
       finding fixes (§2)         worktree")
```
- `harden vs main`: **108 ahead / 0 behind**.  `5.30.26_sdar vs main`: **89 ahead / 0 behind**.
- `harden vs 5.30.26_sdar`: **34 ahead / 15 behind** (merge-base `9ba5dec`). The **15 behind** are
  the sdar-only commits you must reconcile (listed in §3.2).

---

## 2. What was completed THIS session (all on `harden/root-harness`, all clean)

**Phase A (Fidelity) — COMPLETE (7 findings) + Phase B started (4 findings). 11 findings, 10 commits.**
Suite went **3371 → 3391 passed** (+20 tests), **6 skipped, 1 xfailed, zero regressions** throughout.
Every finding: re-read cited code → red test → smallest fix → focused green → **full `pytest tests/ -q`
gate** → commit as `lolout1` (no trailer).

| Commit | Finding | One-liner |
|---|---|---|
| `7c369d8` | **F-27** | ingestion: rescue `<math>` alttext/text before `_DROP_TAGS` (restores SDAR invariants `g_t=σ(β·Δ_t)`, λ=0.1 to the HTML corpus) |
| `2a7afd9` | **F-32+F-34** | scoring: narrow rubric placeholder regex to empty/comma-only parens (`(?<!\w)\(\s*(?:,\s*)*\)`); keep `(%)`/`detach()`/`(θ)` leaves + pin the 3 shapes in tests |
| `28b310d` | **F-28** | ingestion: within-source retry (2 attempts) for transient arXiv HTML fetch (timeout/URLError/429/5xx); `_html_sleep_fn` test hook |
| `151be07` | **F-29** | rlm: surface degraded paper-text as `demo_status.json` `warnings` (observe-first — did **NOT** flip `allow_lossy` default; PR-ρ deferral stands). `_assert_paper_text_precondition` now returns the reason; `_write_demo_status` gained a `warnings` kwarg |
| `839df3d` | **F-30** | cli: reject `paper_text`/`document` title via extracted `_is_noise_title`. **⚠ SUPERSEDED by sdar `e791c60` — DROP on merge (§3.3).** |
| `e45c993` | **F-31** | ingestion: validated-cache reuse for `raw_paper.{pdf,html}` (+ `force_refetch` hatch); safe — attempt-isolation archives raw files before a different paper ingests |
| `88a9380` | **F-06** | rlm: reset two-experiment `FINAL_VAR` guard at the real turn boundary in `ReproLabRLMLogger.log()` (was only reset on a refusal → false refusals; W-9 sibling) |
| `d5b24cf` | **F-07** | rlm: inline `disk_exhausted` haystack detector (ENOSPC/errno 28/disk quota) — was "unknown" |
| `f0da033` | **F-08** | rlm: classify HF gated-repo 401/403 → `missing_dataset`(+auth extra); add `nccl_timeout` class for NCCL collective hangs |
| `2daa70c` | **F-03** | rlm: gate `no such file`/`errno 2` on a config/source co-signal in `_data_load_failure_is_code_bug`; drop standalone `has no attribute` (AttributeError already in `_CODE_BUG_RE`) |

Worktree is **clean** (the F-04 work was interrupted before any edit — nothing pending).

---

## 3. THE MERGE — design, conflict surface, exact reconciliation

### 3.1 What merges cleanly (no sdar overlap — verified)
7 findings live entirely in files **sdar never touched since the fork**, so they merge with **zero
conflicts**:
- `backend/services/ingestion/parser/html_parser.py` (F-27)
- `backend/agents/rlm/rubric_gen.py` (F-32/F-34)
- `backend/services/ingestion/intake/fetchers/arxiv.py` (F-28) + `remote_pdf.py` (F-31)
- `backend/agents/rlm/sse_bridge.py` (F-06) — **and** sdar's `forced_iteration.py` still exposes
  `on_iteration_advance` / `should_refuse_final_var` / `_experiments_in_iteration`, so the F-06 seam
  stays valid after merge.
- `backend/agents/rlm/failure_classifier.py` (F-07/F-08)
- All the new test files for the above (sdar didn't add them) — clean adds.

### 3.2 The 15 sdar-only commits to reconcile against (`9ba5dec..5.30.26_sdar`)
```
6ac50d5 fix(env): per-run venv inherits the repo base stack + fail-soft optional imports
f67373d docs: semantic invariant verification design (fidelity scoring redesign)
dea7900 fix(lab-ui): truly-unique run key via backend runDir
cb02ea2 fix(lab-ui): unique run keys + real paper title + run_N numbering + graph always visible
e791c60 feat(cli): parse the real paper title from the paper text; reject paper_text placeholder  ← supersedes F-30
12b8590 fix(rlm): don't false-drop HuggingFace/Kaggle dataset URLs in cell preflight              ← adjacent to F-03
e7e4e4c feat(rlm): GPU capacity descriptor + cell runner + forced-iteration terminal-OOM bypass
8e867ca feat(scripts): support uncapped SDAR runs via none/0 cap sentinels
8a933f6 docs: one-GPU-per-cell route + OOM remediation
62074b8 test(rlm): run_experiment cell-route branch wiring
db34bb3 feat(scripts): reconcile SDAR launcher flags for the cell runner
20fdea7 feat(rlm): terminal-stop wiring — stop + report, never re-OOM
43189d8 feat(rlm): route run_experiment through the GPU cell runner
970c8ec feat(rlm): single-cell trainer contract + GPU-budget guidance
dfa109d feat(rlm): SDAR env interface ABC + preflight backstop
```

### 3.3 The 4 conflict files — exact reconciliation
Backend `.py` touched by BOTH sides since `9ba5dec`:

| File | harden side | sdar side | Reconciliation |
|---|---|---|---|
| **`backend/cli.py`** | **F-30** (`839df3d`): extracted `_is_noise_title` + `_NOISE_TITLES={…,paper_text,document}` | **`e791c60`**: same `_is_noise_title` + a **superset** `_NOISE_TITLES` (`abstract,introduction,1 introduction,1. introduction,summary,overview,paper_text,paper text,untitled,untitled paper,title`) + first-line title extraction + **17 tests** | **Take sdar's version wholesale; DROP F-30.** sdar's set is a strict superset of mine **except `"document"`** — graft `"document"` into sdar's `_NOISE_TITLES` (one word). **Both branches add `tests/test_cli_paper_title.py` (different blobs) — keep sdar's 17-test file, discard mine.** |
| **`backend/agents/rlm/run.py`** | **F-29** (`151be07`): `_assert_paper_text_precondition` returns reason; `_write_demo_status(warnings=…)`; run-start passes the warning. Plus P0–P3 edits. | cell-route + terminal-stop wiring (`43189d8`/`20fdea7`/`e7e4e4c`) in `run_experiment`/finalize | **Hand-merge — hunks are in different functions** (F-29 = precondition gate + status writer near run start; sdar = run_experiment routing). Keep both. Verify `_write_demo_status` keeps the new `warnings` kwarg after merge. |
| **`backend/agents/rlm/primitives.py`** | **F-03** (`2daa70c`): `_CODE_BUG_PHRASES` − 3 entries, `+_CONFIG_CODE_COSIGNALS`, co-signal clause in `_data_load_failure_is_code_bug` (~L3280-3342). Plus P0–P3. | cell-runner routing in `run_experiment`/`_execute_in_sandbox` (`43189d8`), HF/Kaggle URL preflight (`12b8590`) | **Hand-merge — different regions.** F-03 is in the classifier-phrase block; sdar is in the experiment-execution path. Check `12b8590`'s HF/Kaggle-URL logic doesn't re-introduce the bare-data false-positive F-03 fixed (they're complementary: F-03 = `data_load_failure` classification; `12b8590` = cell **preflight** URL drop). |
| **`backend/agents/rlm/report.py`** | **P0–P3 only** (experiment manifest back-link `b733b30`, metric-projection §5b `7152a78`) — **none of this session's 11 findings touched report.py** | sdar report changes | Hand-merge P0–P3-vs-sdar. **(Note: pending finding F-11 will touch report.py — do it AFTER the merge, against merged code.)** |

### 3.4 Recommended merge procedure (elegant + safe)
> The real checkout `/home/sww35/openresearch` has **~20 dirty `runs/` artifacts** (run outputs for
> `prj_09047604e591d969`, deletions + 3 untracked `runs/…__timestamp/` dirs). **These are not code.**
> Stash or leave them; never commit them; they must not enter the merge. `git merge` may refuse with a
> dirty tree — `git stash push -- runs/` (or `git -c …` on a clean integration branch) first.

1. **Integration branch off sdar** (don't merge directly into the live branch):
   ```bash
   cd /home/sww35/openresearch
   git stash push -m "runs artifacts" -- runs/        # park the dirty run outputs
   git checkout -b integrate/harden-into-sdar 5.30.26_sdar
   git merge --no-ff harden/root-harness               # expect conflicts in the 4 files of §3.3
   ```
2. **Resolve the 4 files** per §3.3. For `cli.py`: `git checkout --theirs`? No — **take sdar's
   `_is_noise_title`/`_NOISE_TITLES` (ours, since we branched from sdar), add `"document"`, and
   `git checkout 5.30.26_sdar -- tests/test_cli_paper_title.py`** to keep sdar's 17-test file. The
   net effect: **F-30's diff is dropped**; F-30 is satisfied by sdar's superior implementation.
3. **Full-suite gate**: `/home/sww35/openresearch/.venv/bin/python -m pytest tests/ -q` from the repo
   root. Baseline to beat: **0 failures** (sdar's own count + harden's adds; expect ~6 env-skips).
   Fix any merge-induced breakage before finalizing.
4. **Land it**: fast-forward `5.30.26_sdar` to the integration branch (or open a PR). Optionally also
   merge the result into `main` (main is 0-behind, so it's a clean fast-forward target later).
5. `git stash pop` to restore the `runs/` artifacts if needed (or drop the stash — they're disposable).

**Alternative (if the `--no-ff` merge is too tangled):** cherry-pick the 7 clean findings' commits
onto the integration branch first (zero conflicts), then hand-apply F-03/F-29 to the 3 code files,
and skip F-30. Same end state, smaller conflict blasts.

### 3.5 Merge acceptance checklist
- [ ] `pytest tests/ -q` green (0 failures) on the integrated branch.
- [ ] F-30 NOT double-applied (only sdar's `_is_noise_title`; `"document"` grafted; one test file).
- [ ] `_write_demo_status` still accepts `warnings=`; `_assert_paper_text_precondition` still returns the reason.
- [ ] F-06: `ReproLabRLMLogger.log()` still calls `pol.on_iteration_advance()`; sdar's terminal-OOM bypass and this reset don't fight (different triggers).
- [ ] F-03 co-signal clause intact in `_data_load_failure_is_code_bug`; not undone by `12b8590`.
- [ ] Run-output `runs/` artifacts NOT committed.

---

## 4. Remaining backlog (do AFTER the merge, against merged code)

Re-read each finding's evidence in `docs/audits/2026-05-31-backend-core-opportunity-backlog.md` and
its copy-paste prompt in `docs/superpowers/specs/2026-06-01-backend-core-task-prompts.md` **before
touching it** (line anchors will have drifted post-merge — match on unique content, not line numbers).

### Phase B remainder (5 findings) — finish this first
- **F-33** `m/h/medium` — grader evidence reads only the **6 KB head** of each code file alphabetically
  (`backend/evals/paperbench/leaf_scorer.py` `_gather_evidence`, ~L190/270-312). Prioritize `train.py`/
  method files to the front + **head+tail** slice so end-of-file gate logic reaches the grader. Plan
  (designed, not yet built): add `_evidence_priority(rel)` + `_truncate_head_tail(data,cap)`, sort
  candidates, give priority files a larger cap. **Score-moving + LLM-mediated → gate behind a
  `REPROLAB_GRADER_*` env (default-on) or make it strictly-additive (always keep the head, *add* the
  tail of method files).** Test `_gather_evidence` deterministically (no LLM).
- **F-04** `l/l/small` — escalation loop misses watchdog-killed OOM. **Designed approach (use it):**
  the `_WatchdogKilled` return dict (`primitives.py` ~L2963) has no `exit_code`/infra flag, so the
  escalation gate (~L3906-3917) defaults `exit_code=1`, `is_oom=False`, and `break`s without advancing
  the ladder. Watchdog kills on **"no signal" (staleness), NOT memory** — so DON'T widen to all kills.
  **Extract `_is_oom_escalation_trigger(result, *, exit_code, stderr_tail)`**: returns
  `_detect_cuda_oom(...)` OR (`result.get("watchdog_killed")` AND a `_CUDA_OOM_MARKERS` hit in the
  **full** logs, not just the 4096-byte tail). Wire it at the gate. Test in
  `tests/rlm/test_cuda_oom_detection.py` (markers are **case-sensitive**: `"CUDA out of memory"`,
  `"torch.cuda.OutOfMemoryError"`, …). This is the exact spot the session was interrupted — start here.
- **F-05** `l/l/small` — `_reclassify_masked_code_bugs` (and other postflight guards) are success-gated
  (`primitives.py` ~L3968 `if result.get("success")`); a **failed** run with a masked code bug keeps its
  vaguer error. When `success is False` and no specific `failure_class` is set, still run the
  reclassifier and promote its precise message into `error`/`suggested_fix` (never flip `success`).
- **F-11** `l/l/small` (status partial) — `build_final_report` reconciles the verdict against the root's
  self-reported score **before** applying the best-of-run floor (`report.py` ~L755 vs ~L827-838). Move
  the floor block above the reconcile call (or pass the floored score in). The authoritative
  `leaf_scorer.amend_final_report` path already corrects normally-finalized runs — gap is only the
  no-amend path. **Do this against MERGED report.py.**
- **F-35** `l/m/medium` (observe-first, KNOWN-pending) — §5c programmatic citation-clamp + B2 spec-gate
  short-circuit in `leaf_scorer.py`. Wire `_citation_validates(justification, evidence)` + clamp→0 under
  `REPROLAB_RUBRIC_REQUIRE_CITATION` (emit would-clamp delta when enforce-off); early-return the
  hard-capped result before building grader batches when an invariant tripped. **Observe-first: enforce
  only after a real SDAR run confirms no spurious loss. Do LAST in Phase B.**

### Phase C — Cost/perf (7): **F-01+F-02** (verify cache key SHA-256s the whole `code/` incl. multi-GB
checkpoints — cap it like the leaf scorer; do together), F-09 (per-paper metadata busts prompt cache),
F-18 (OpenAI `reasoning_tokens` dropped in usage accounting), F-39 (leaderboard cache unbounded +
re-stats every dir), **F-12** (RDR cluster SDK cost written to `code/cost_ledger.jsonl` not run root →
under-reports), F-17 (RDR double-keys `total_usd`).

### Phase D — Budget/runtime = design **P4** (8): build the `RuntimeBackend.exec` template-method
(**F-26**) first — the seam the rest hang off — then F-22 (Brev no budget), F-23 (upfront/mid-exec cap),
F-15/F-42 (`--max-run-gpu-usd` not enforced on RDR/batch), F-24 (uncapped `LocalProcessBackend` output —
the M4 log-trunc family), F-25 (dead `artifacts.py` commands.log revival), **F-14** (RDR has no
wall-clock watchdog).

### Phase E — Security (5): **F-37** (`POST /runs/arxiv` SSRF when demo gate unset — the one real
security risk; allowlist/validate the arxiv-id→URL even with gate off), F-40 (`post_message` PII/secret
scrubber = design borrow #6), F-36 (SSE byte-offset reader drops events torn across poll reads), F-20
(`_resolve_mcp_servers` zero tests), F-41 (`resume_run` doesn't validate a checkpoint exists).

### Phase F — Cleanup (11): F-13 (dead `_ClusterWatchdog`), F-44, **F-46** (`cmd_reproduce` god-fn —
note this session already extracted `_is_noise_title`/`_NOISE_TITLES`; sdar extracted more title
helpers — coordinate), F-43, F-45, F-47, F-16, F-19, F-10, F-38, F-21 (per-role provider picker).
Plus separate, higher-risk, **not in this backlog**: **P1.5** (preventive `can_use_tool` + parry-guard
exfil-taint).

---

## 5. Guardrails & gotchas (carry-over — non-negotiable)

- **Bash cwd resets to `/home/sww35/openresearch` between calls.** Prefix worktree commands with
  `cd /home/sww35/openresearch-harden && …`.
- **Worktree has no venv** → test with `/home/sww35/openresearch/.venv/bin/python -m pytest …` from the
  worktree cwd (cwd wins, so `backend` imports from the worktree). Full suite ≈ **70 s**, baseline
  **6 env-skips + 1 xfail**.
- **Commit as `lolout1`, NO co-author trailer:**
  `git -c user.name=lolout1 -c user.email=lolout1@users.noreply.github.com commit …`. (You ARE lolout1.)
- **One tested commit per finding**; red-before/green-after; **full-suite zero-regression gate** is the
  phase exit gate (run it before advancing).
- **Only TIGHTEN existing logic** — no new deps/rewrites/frameworks. If a fix balloons, STOP and flag.
- **Score/outcome-movers ship observe-first behind a `REPROLAB_*` hatch.** Pure fidelity fixes whose
  change is a strict improvement (F-27/F-32 style) ship default-on; genuine score-direction-unbounded
  changes (F-33, F-35) are gated/observe-first.
- **Don't hallucinate** — re-Read the cited region right before editing; match `Edit` on unique content,
  not line numbers (anchors WILL have drifted post-merge).
- **`runs/` artifacts** in the real checkout are disposable run outputs — never commit them; keep them
  out of the merge.
- **`git checkout -- <file>` reverts the WHOLE file to HEAD** — during a "prove-red" step, prefer a
  scripted string-replace + revert (don't blow away uncommitted work; learned the hard way this session).
- The `.git/worktrees/openresearch-harden/gc.log` warning on commits is harmless background gc noise.

---

## 6. Memory pointers
Updated this session: `[[backend-core-audit]]` (progress + this merge topology + this doc's path).
Also relevant: `[[root-harness-hardening]]`, `[[harden-borrow-provenance]]`, `[[sdar-local-baseline-status]]`,
`[[commit-attribution-preference]]`.

**This doc** is committed on `harden/root-harness` at
`docs/runbooks/2026-06-01-backend-core-merge-and-continuation-handoff.md`. If you're working from the
`5.30.26_sdar` checkout before merging, read it via
`git show harden/root-harness:docs/runbooks/2026-06-01-backend-core-merge-and-continuation-handoff.md`.
