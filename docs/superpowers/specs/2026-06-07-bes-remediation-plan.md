# `bes` remediation plan — fix everything from the 2026-06-07 doc-alignment audit

> **Status:** Plan (design + adversarial critique complete) · 2026-06-07
> **Source of record:** [`docs/audits/2026-06-07-bes-doc-alignment-audit.md`](../../audits/2026-06-07-bes-doc-alignment-audit.md)
> **Authoring method:** 9 code-grounded design agents + 2 adversarial critics (coverage, sequencing/risk); all four blocking defects the critics found are folded in below.

## Decisions (locked with the operator)

1. **Merge `main`→`bes` FIRST**, then fix on top. The merge alone resolves the rename (D1) and brings SDK-isolation, docs-freshness, and `documentation.md`.
2. **Scope = everything** (Buckets A+B+C+D from the audit).
3. **Reconciliation default = fix the CODE to match documented intent** — with **three fix-the-DOC exceptions** where the code is the newer deliberate decision: **SBX-2** (RunPod `SECURE` default), **BEH-1** (always-on FSDP sharding), **SBX-1** (`cuda-devel` image; already corrected on `main`).
4. **Build** the C2 documented-but-absent features (PEEK-lite context-map, MUSE-lite negative-lessons, CLI signal handlers, Dockerfile shape guard).
5. **After the plan, auto-apply doc-only fixes** — *revised by the sequencing analysis*: see [§ Auto-apply reconciliation](#auto-apply-reconciliation). Doc fixes must follow Phase 1; a naive auto-apply is unsafe.

## The one insight that orders everything

The merge's **per-file conflict resolution IS the interface contract** every later component builds on. Two resolutions are load-bearing:
- `claude_runtime.py` **must resolve MAIN-wins** (always-explicit `mcp_servers` via `_agent_options_kwargs`). If the hand-merger applies "bes-logic-wins" here, it **silently reopens BUG-NEW-038** (the SDK-isolation security hole).
- `accelerator.py` and `executor.py` are **add/add** conflicts — they **must be UNIONed**, or the features Buckets A/B edit & document will vanish.

Therefore Phase 1 emits an explicit **per-file resolution table** (below), and every downstream component cites a *symbol* (not a pre-merge line number — those all shift in the merge).

---

## Cross-cutting rules (apply to every phase)

| Rule | Why |
|---|---|
| **R1 — Symbol anchors, not line numbers.** Every component references functions/constants/classes; each begins with "step 0: re-confirm anchors against the merged tree." | The merge shifts every line in the ~18 content-conflict files. |
| **R2 — New env vars use the `OPENRESEARCH_` prefix.** The bidirectional config shim (`config.py` `setdefault` both directions) aliases them to `REPROLAB_` too — so feature-flag tests must assert behavior under **both** spellings and note a stale `REPROLAB_*` export can activate a new flag. | Post-merge naming contract + shim reverse-direction. |
| **R3 — CLAUDE.md edits are serialized.** Feature builds (PEEK-lite, MUSE-lite) finalize the primitive registry/flag set **first**; Bucket B's count/flag prose is the **last** doc edit and reads the final state. Single owner per sentence. | Three-way write contention; 16-vs-17 primitive count. |
| **R4 — Docs-freshness is a merge gate.** Phase 1 acceptance runs `python3 scripts/docs_freshness_check.py` green on the merged tree; cite absent specs with **backtick refs, not `[](path)` links** (the checker flags broken relative links). | The merge PR triggers freshness CI on all `**/*.md`. |
| **R5 — Bucket B needs content-fidelity tests**, not just a freshness bump: assert each documented env var is read somewhere in `backend/`; documented primitive count == the number of **root-exposed tools** (`build_custom_tools` output, *not* `len(PRIMITIVE_REGISTRY)` — the registry includes non-root entries; this is the corrected PRIM-1 conclusion and it also dissolves the 16-vs-17 contention); documented SSE event names appear as literal emit strings; RunPod cloud-type doc string == `config.py` default. Confirm the root-exposed count empirically before any test asserts it. | A doc can pass freshness while describing wrong defaults. |
| **R6 — Merge-resolution provenance artifact.** Phase 1 commits a per-file "which side won each hunk" snapshot so a downstream test failure triages back to a merge decision without re-bisecting. | Botched resolutions surface far from the cause. |

---

## Phase 0 — Safety net  *(effort: S)*

- `git tag -a bes-preremediation -m 'pre-remediation rollback floor'` and push it. This is the hard reset floor.
- Confirm `pytest >= 9` (`.venv/bin/python -m pytest --version`; CLAUDE.md gotcha: `rlms==0.1.1` needs pytest 9, conflicts with `requirements-dev` 8 — install `backend/requirements.txt` alone).
- **Record the baseline** (acceptance is *delta-vs-baseline*, since bes green-ness is unconfirmed): `pytest tests/ -q` → save PASS/FAIL/ERROR counts + failing node-ids to `/tmp/bes_baseline_pytest.txt`; stub smoke `OPENRESEARCH_RLM_STUB_PRIMITIVES=1 python -c 'from backend.app import create_app; create_app()'` + `python -m backend.cli --help`.
- `git config rerere.enabled true; git config merge.conflictStyle zdiff3`.

## Phase 1 — Merge `main`→`bes`  *(effort: L · covers D1 · BLOCKS everything)*

**1a. Pre-rename normalization commit** (the trick that collapses naming-only conflicts). On a new `integrate/main-into-bes` branch off `bes`, mechanically replace the **uppercase token** `REPROLAB_`→`OPENRESEARCH_` across `backend/ scripts/ frontend/ tests/` only:
```
git switch -c integrate/main-into-bes bes
git grep -lI 'REPROLAB_' -- backend/ scripts/ frontend/ tests/ | xargs perl -pi -e 's/\bREPROLAB_/OPENRESEARCH_/g'
```
Exclude lowercase `reprolab` (84 package-path/identifier hits main owns), `backend/config.py` (main replaces it wholesale), and `.env.example` (main wins). Verify no lowercase id was hit; boot-smoke; commit. This makes ~24 add/add test conflicts + `sdk_isolation.py` + most `rlm_query.py` naming diffs auto-merge.

**1b. Merge** `git merge --no-ff --no-commit main` (NOT `-X ours/theirs`). Resolve per the table, validate (1d) **before** committing.

**1c. Per-file resolution table** (the interface contract — R6 provenance per file):

| File(s) | Resolution | Rationale |
|---|---|---|
| `claude_runtime.py` | **MAIN-wins** (`_agent_options_kwargs`, always-explicit `mcp_servers`); re-apply bes `__all__`/local adds on top | **Load-bearing for BUG-NEW-038** |
| `accelerator.py`, `executor.py` | **UNION** (add/add — diff both blobs, keep both features) | Buckets A/B depend on these |
| `config.py`, `.env.example` | **MAIN-wins wholesale**; Bucket A re-applies its Settings fields on top (Phase 3) | bes never modified config.py |
| `primitives.py`, `run.py`, `baseline_implementation.py`, `rlm_query.py`, `report.py`, `leaf_scorer.py`, `rubric_gen.py`, `failure_classifier.py`, `claude_oauth_client.py`, `context.py`, `invoke.py`, `openai_runtime.py`, `cli.py` | **BES-logic-wins** + `OPENRESEARCH_` naming | bes's SDAR/GPU work is the point |
| `live_runs.py` | **BES structure**; STAB-2 then extends the `RunStatus` Literal (Phase 4) | preserve the seam STAB-2 needs |
| `leaderboard.py`, `report_resolution.py`, `replay.py`, `recent-runs-panel*` | **Reconcile (both diverged)** → deferred to **Phase 2** | merge takes main's newer base, Phase 2 re-applies bes deltas |
| `.gitignore`, `system_prompt.py`, `factory.py`, `app.py`, `docs/policies/documentation.md`, `scripts/docs_freshness_check.py`, freshness workflow | **Clean / auto** (identical or main-only adds) | — |

**1d. Acceptance:** `git diff --name-only --diff-filter=U` empty; `grep -rn 'REPROLAB_' backend/ scripts/ frontend/` shows only the documented config shim; pytest shows **no NEW failures vs baseline**; stub smoke passes; **`python3 scripts/docs_freshness_check.py` green** (R4); provenance snapshot committed (R6). Then `git commit`.

> Open question to settle at merge time: are `accelerator.py`/`executor.py` add/add blobs the *same* feature (trivial union) or *divergent* (real reconciliation)? Diff before resolving.

## Phase 2 — Post-merge reconciliation + re-anchor  *(effort: M · the "phantom C1" the critic caught)*

The merge takes `main`'s newer `leaderboard.py` + `report_resolution.py` + replay route + recent-runs-panel as the base; **re-apply bes's deltas** on top (best-attempt scoring, recent-runs home panel, `?replay=` driver, `GET /runs/{id}/replay-events`). Re-confirm all downstream symbol anchors against the merged tree (R1). Acceptance: leaderboard + replay e2e/vitest pass; both surfaces render.

## Phase 3 — Bucket A code fixes  *(effort: M · covers ACC-1, ACC-2, AUTH-1, SSE-1)*

| Fix | Change | Test |
|---|---|---|
| **ACC-2** | `_resolve_endpoint()`: when host is `api.openai.com` and `OPENRESEARCH_ACCELERATOR_API_KEY` unset, default `api_key` to `OPENAI_API_KEY`; verify flow to `OpenAILlmClient` + `run.py` sub-backend | `test_accelerator_openai_route.py`: openai host + only `OPENAI_API_KEY` set → key used, no `Bearer local` |
| **ACC-1** | `build_accelerator_client()` reads `OPENRESEARCH_SUBRLM_OPENAI_TIMEOUT_S` (default **120**), passes `timeout=` to client | assert client timeout reflects the env var |
| **AUTH-1** | Add `OPENROUTER_API_KEY` (+`FEATHERLESS_API_KEY`) branch to `models._credential_value`; **re-apply** the Settings fields on top of main's `config.py` (R per table) | qwen3-coder/kimi-k2.5 resolve OpenRouter creds from `.env` |
| **SSE-1** | Route the bypassing emit sites through `sse_bridge.sanitize_iteration`, OR restate the invariant precisely ("every corpus-derived field passes through `redact_corpus`") and align Bucket B doc | assert flagged events pass the chokepoint |

## Phase 4 — Feature builds  *(effort: M+M+S+S+S)*

- **PEEK-lite context-map** (FLAG-1) — new `context_map.py` (`read_context_map`, write hook in `binding.py` unioning `understand_section`/`extract_hyperparameters`/`detect_environment` → `rlm_state/context_map.json`, caps ≤40 fields/≤8 values/≤8KB, thread-safe, fail-soft); flag `OPENRESEARCH_CONTEXT_MAP` (default off = no-op). **Must add `read_context_map` to all THREE collections** — `PRIMITIVE_REGISTRY`, `PRIMITIVE_DESCRIPTIONS`, `build_custom_tools` — **and** `EXPECTED` in `test_registry.py`, atomically (critic catch). Tests: union/caps/thread-safety/off-state/both-spellings (R2).
- **MUSE-lite negative-lessons** (FLAG-2) — new `lesson_distiller.py::mine_lessons` (hook in `run.py::_finalize`), `_negative_lessons_block` in `baseline_implementation.py`, `runs/_lessons/<arxiv_id>.json`; guardrails (occurrences≥2 except `dockerfile_invalid`@1, classifier-sourced `suggested_fix`, staleness≥3 retirement, ≤5/≤200 chars); flag `OPENRESEARCH_NEGATIVE_LESSONS` (default off); needs `ctx.arxiv_id` (graceful absence). Author the thin cited spec `2026-05-30-negative-lessons-design.md`.
- **CLI signal handlers + RunStatus** (STAB-3, STAB-2) — `cli.py::_install_termination_handlers` (SIGTERM/SIGHUP → `demo_status` `killed`+`killReason`, then `raise_signal(SIGINT)`; `_mark_demo_status_*` treat `killed` terminal; maintain `_ACTIVE_PROJECT_ID`); add `killed`+`interrupted` to `live_runs.py` `RunStatus` Literal. Tests: `test_termination_handler.py`, `test_live_runs_terminal_status.py`.
- **Dockerfile shape guard** (STAB-4) — `primitives.py::_validate_dockerfile_shape` (first non-blank/non-comment line ∈ `FROM`/`ARG`/`# syntax=`); `implement_baseline` snapshot-restore on prose (`code="dockerfile_shape_guard"`); `run_experiment` fail-fast `failure_class="dockerfile_invalid"` in `_RUN_EXPERIMENT_REPAIRABLE_FAILURES`; deterministic guard runs **before** the existing LLM-repair fallback. Test `test_dockerfile_shape_guard.py`.
- **SDK-isolation closure** (STAB-1) — verify the merge landed main's `claude_runtime.py` (hard gate, per 1c); then close the **two sites the merge does NOT touch** (`rlm_query.py`, `hermes_audit/providers.py`): `setting_sources=[]`, explicit `mcp_servers`, non-plan `permission_mode`. Test `test_sdk_options_isolation.py` asserts all three sites.

## Phase 5 — Bucket B docs + orphan triage + low-sev fixes  *(effort: M · covers the 22 doc findings + 11 orphans the critic found)*

Document, with **content-fidelity tests (R5)** and **serialized last (R3)**, into **both CLAUDE.md and `system_overview.md`**: GPU-reservation subsystem (NEW-1); replay route+UI (SCR-4, SSE-6); the 3 scripts (SCR-1); best-of-run floor + latest-metrics scoring (NEW-4/BEH-4); `brev` sandbox (SBX-3); `codex_repair` + 8 `*_CODEX_*` vars (PRIM-2); vLLM executor `OPENRESEARCH_EXECUTOR` (SCR-2/3); new SSE events `experiment_completed`/`run_fatal`/`run_interrupted`/`worker_report_completed`/`sandbox_*` (SSE-2/3/4/5 — confirm names against emit sites first); `OPENRESEARCH_ACCELERATOR_API_KEY` (ACC-3); rubric-fidelity spec status → "A+D shipped, B/C pending" + cite it (NEW-3/BEH-2); primitive count/list (PRIM-1, final number from Phase 4); model-exclusion (BEH-3). **Doc-fix exceptions:** document `SECURE` default (SBX-2) and always-on FSDP (BEH-1); SBX-1 resolves via merge.

**Orphan triage (critic gap, high):** assign and resolve NEW-2/5/6, PRIM-3/4, AUTH-2/3/4 (triage each against the audit, add doc step + content); **SCR-7** — create/restore `docs/runbooks/running-the-project.md` + `2026-05-29-monitoring-loops.md` *or* repoint the CLAUDE.md citations, with a test asserting **every CLAUDE.md doc citation resolves to an existing file**; **STAB-7** — delete orphaned `.pyc` in `tests/cli/` + `tests/services/events/`, guard against tracked `.pyc` without a sibling `.py`; **SBX-4** — fix the stale source-line citation.

## Phase 6 — Repo hygiene  *(effort: S · covers D2, D3)*

- **D2:** `git rm --cached paper-repro-bes-docs.zip`; add `*.zip` to `.gitignore`; extract the 4 markdown files to `docs/bes/*.md` with doc-status headers (drop the `out/` A/V scripts). *(History rewrite to purge the 6 MB blob is optional and risky on shared history — flagged, not default.)*
- **D3:** `git rm --cached` the run-dir files **outside** the whitelist (keep `final_report.{json,md}`, `demo_status.json`, `batch_child.log`, `experiment_runs.jsonl`); verify with `git check-ignore`. **Decision needed:** `cost_ledger.jsonl` is re-included by `.gitignore` line 73 (intentional diagnostic) — KEEP (follows gitignore) vs drop (per audit). Default: KEEP.
- `tests/test_repo_hygiene.py`: no tracked `.zip`; tracked run-dir files ⊆ whitelist; no orphan `.pyc` (folds STAB-7).

## Phase 7 — Verification  *(effort: S)*

Full `pytest tests/` (no new failures vs Phase 0 baseline) + `npm test`/`vitest` + `npx tsc --noEmit`; `scripts/docs_freshness_check.py` green; the R5 content-fidelity tests; `grep` confirms 0 stray `REPROLAB_` in code; the citation-resolves test; one real stub smoke (and optionally a COMMUNITY-GPU SDAR smoke). Update the audit doc's status to "remediated."

---

## Auto-apply reconciliation

You chose "plan, then auto-apply doc fixes." The sequencing analysis **overrides a naive reading** of that, for cause:

- Doc fixes that touch CLAUDE.md/`system_overview.md` **must run after Phase 1** (the merge renames prefixes and re-anchors lines) and **after Bucket B** (idempotency).
- A blind "fix code to match docs" pass would **corrupt the three intentional-code exceptions** (SBX-2/BEH-1/SBX-1) and double-apply SBX-1 (already fixed on main).

**Therefore:** auto-apply is gated as an **idempotent, verify-only doc sweep run after Phase 5**, carrying a hard **skip-list {SBX-2, BEH-1, SBX-1}**, emitting backtick refs for absent specs, and bumping `last-verified`. It is never the primary edit mechanism.

**Safe to do *right now*, pre-merge (no ordering constraint):** (a) Phase 0's `bes-preremediation` tag; (b) **D2 zip untrack**. Everything else waits for the merge.

---

## Effort & sequencing summary

```
Phase 0 (S) ─▶ Phase 1 MERGE (L, blocks all) ─▶ Phase 2 reconcile (M)
                                               ├▶ Phase 3 Bucket A code (M)
                                               ├▶ Phase 4 feature builds (M,M,S,S,S)
                                               └▶ Phase 5 docs+orphans (M)  [serialized after 3&4 per R3]
                                                  └▶ Phase 6 hygiene (S) ─▶ Phase 7 verify (S)
```
Phases 3 and 4 can proceed in parallel after Phase 2; Phase 5 (docs) is serialized last so it reads the final code/registry state.

## Risk register

| Risk | Mitigation |
|---|---|
| Merge silently drops a bes behavior | R6 provenance artifact + delta-vs-baseline pytest gate; `git reset --hard bes-preremediation` floor |
| `claude_runtime.py` resolved bes-wins → BUG-NEW-038 reopens | 1c table marks it MAIN-wins; STAB-1 test is a **hard merge-acceptance gate**, not post-hoc |
| add/add `accelerator.py`/`executor.py` taken one-sided → features vanish | 1c marks UNION; diff blobs before resolving |
| Stale CLAUDE.md fails freshness CI, blocks the whole merge PR | R4 freshness gate in Phase 1 acceptance |
| Pre-merge line anchors wrong post-merge | R1 symbol-anchoring + per-component step 0 |
| Bidirectional shim activates a new flag via stale `REPROLAB_*` export | R2 dual-spelling tests + risk note |
