# Autonomous Upload → Live Reproduction UI — Execution Handoff: T3b → T14 (2026-07-05)

**This is a clean context-clear point. T1–T3 are complete, committed, and Opus-reviewed clean.**
A fresh session should be able to `/clear`, read this doc + the plan + the ledger, and resume at T3b
with zero re-derivation. Everything hard-won this session is captured below.

## TL;DR / resume point

- **Worktree:** `/home/abheekp/openresearch-autonomous-ui`, branch `feat/autonomous-upload-ui`.
- **HEAD = `ee311db3`** (T3 done). Base of the feature = `06e6f3e1` (spec/plan commit) atop foundation `ed086edb`.
- **Done:** T1 `e53506bc`, T2 `df8497d6`, T3 `ee311db3` — all review-clean, no Critical/Important.
- **NEXT: T3b** (or jump to T4 — T3b is P1/deferrable). Then T4→T5→T6→T7→T8 (backend), then T9a→T14 (frontend).
- **Ledger (durable progress):** `.superpowers/sdd/progress.md` — trust it + `git log` after any resume.
- **Plan:** `docs/superpowers/plans/2026-07-05-autonomous-upload-ui-implementation-plan.md` (14 tasks, TDD, `file:line` anchors).
- **Spec:** `docs/superpowers/specs/2026-07-05-autonomous-upload-ui-and-live-reproduction-design.md`.
- **Original pre-T1 handoff:** `docs/runbooks/2026-07-05-autonomous-upload-ui-handoff.md` (still valid for env/invariants).

## 1. Environment (already provisioned — do not redo)

- **Python:** `.venv` (Python 3.12; full `backend/requirements.txt` + `requirements-dev.txt`, editable-installed to THIS worktree). Tests: `.venv/bin/python -m pytest …` (socket-hermetic; `pytest-socket` blocks non-loopback).
- **Node:** `node_modules` installed. `source ~/.nvm/nvm.sh && nvm use v22.14.0` (system Node is broken for vitest). `npm test` (vitest); `npx playwright test` (no npm alias); lab suite `--no-file-parallelism`.
- **`.env`:** copied from the primary worktree (funded `AZURE_FOUNDRY_API_KEY` = Opus 4.8 / Sonnet 5; gitignored). Present ⇒ the baseline env-dependent tests pass.
- **Lint:** `uvx ruff@0.15.16 check .` (E4/E7/E9/F). `npx tsc --noEmit` for TS.
- **Design reference screenshots** for the frontend tasks live in the PRIMARY worktree root: `/home/abheekp/openresearch/alphaxiv_{1..5,autoresearc}.png`.

## 2. Execution model (the SDD loop — `superpowers:subagent-driven-development`)

- **Implementers = Sonnet** (`Agent` tool, `subagent_type: general-purpose`, `model: sonnet`). **Reviewers = Opus** (fresh, mutation-armed, `model: opus`). **Final whole-branch review = Opus.** Never let a Sonnet agent do the review. The controller (you) stays on coordination + adjudication; do NOT write task code yourself.
- **Strictly sequential on this ONE branch.** Do not run two implementers in parallel: per-task review uses `git diff BASE..HEAD`, and an interleaved commit from a parallel lane pollutes the range. (The frontend kit T9a/T9b is file-disjoint and *could* be a parallel lane, but only at the cost of `.git/index.lock` races + dirty review ranges — not worth it. Keep it sequential.)
- **Per-task loop:**
  1. `BASE=$(git rev-parse HEAD)` — record BEFORE dispatching (never use `HEAD~1`; multi-commit tasks break it).
  2. Extract the task brief (see §3).
  3. Ground the task yourself first (read the actual anchors — line numbers in the plan have DRIFTED; locate by content). Resolve ambiguities in the dispatch, don't make the implementer guess.
  4. Dispatch the Sonnet implementer with: brief path, scene-setting, grounded interfaces/decisions from §5–§6, the report-file path, the report contract. TDD is mandatory.
  5. Handle status: DONE → review; DONE_WITH_CONCERNS → read concerns, address if correctness/scope; NEEDS_CONTEXT → provide + `SendMessage` the SAME agent (preserves recon context) or re-dispatch; BLOCKED → assess (context / bigger model / split / escalate).
  6. `review-package BASE HEAD` → dispatch the Opus reviewer with brief + report + diff paths + the binding global constraints as the attention lens. Do NOT tell it what to ignore or pre-rate severity.
  7. Fix loop: dispatch ONE fix subagent for ALL Critical/Important findings (name the covering tests; require test evidence in the fix report) → re-review. Record Minor findings in the ledger roll-up.
  8. Mark the ledger: `Task N: complete (commits <base7>..<head7>, review clean)`.
- **Commits:** exact message from the plan; present-tense; **NO AI-attribution trailers** (no `Co-Authored-By`, no "Generated with"); git identity is already `lolout1 <appradhann@gmail.com>` — never pass `-c user.email=…`. **Push to `deepinvent` ONLY on explicit request.**
- **Exit bar per task:** touched tests green; ruff clean (Py) / `tsc --noEmit` clean (TS); **NO NEW failures** beyond the pre-existing set (see §5).

## 3. Tooling + the task-brief gotcha

- **Skill dir:** `/home/abheekp/.claude/plugins/cache/claude-plugins-official/superpowers/6.1.1/skills/subagent-driven-development/`
  - `scripts/task-brief PLAN_FILE N` → writes `.superpowers/sdd/task-N-brief.md`, prints the path.
  - `scripts/review-package BASE HEAD` → writes `.superpowers/sdd/review-<base7>..<head7>.diff` (commit list + stat + full diff w/ context), prints the path. Reviewer reads it in one call; it never enters your context.
- **⚠️ AWK GOTCHA:** `task-brief`'s awk matches a NUMERIC prefix, so `Task 3`/`Task 3b` and `Task 9a`/`Task 9b` **collide** (the script would merge them). For sub-lettered tasks, extract manually:
  - T3b: `awk '/^## Task 3b/{f=1} /^## Task 4/{f=0} f' PLAN > .superpowers/sdd/task-3b-brief.md`
  - T9a: `awk '/^## Task 9a/{f=1} /^## Task 9b/{f=0} f' PLAN > .superpowers/sdd/task-9a-brief.md`
  - T9b: `awk '/^## Task 9b/{f=1} /^## Task 10/{f=0} f' PLAN > .superpowers/sdd/task-9b-brief.md`
  - Numeric-only tasks (**4, 5, 6, 7, 8, 10, 11, 12, 13, 14**) use `scripts/task-brief PLAN N` directly (verified clean: e.g. `Task 1` doesn't grab `Task 10` because the boundary regex is `Task N([^0-9]|$)`).
- **Prompt templates:** `implementer-prompt.md`, `task-reviewer-prompt.md` in the skill dir; final review = `../requesting-code-review/code-reviewer.md`.

## 4. What T1–T3 established (facts later tasks rely on)

- **T1 (`e53506bc`)** — advanced-field forwarding. `StartRunRequest` ALREADY declares all 8 advanced fields (`root_provider`, `subagent_auth`, `dynamic_gpu`, `force_single_gpu`, `gpu_parallelism`, `accelerator`, `max_gpu_usd_per_hour`, `vram_gb`) at `live_runs.py:180–218`. `_subprocess_env` now emits `OPENRESEARCH_DYNAMIC_GPU`/`FORCE_SINGLE_GPU`/`MAX_GPU_USD_PER_HOUR`/`VRAM_OVERRIDE_GB`. Helpers `_optional_form_float`/`_optional_form_int` exist in `app.py`. `root_provider`/`subagent_auth` are forwarded but INERT (no consumer).
- **T2 (`df8497d6`)** — `autonomous: bool = False` + `run_spec: str | None = None` on `StartRunRequest`; `autonomous` on `StartArxivRunRequest`. `_python_script(...)` (module-level, `live_runs.py:~1720`) builds a runtime `common` dict (`~:1761`) and embeds it into the generated subprocess script via `config = json.loads({json.dumps(json.dumps(common))})` (`~:1819`; the `{{ }}` elsewhere are f-string brace escapes). `run_spec` now rides `common` (`~:1801`) AND the uploaded-paper `cmd_reproduce` Namespace (`~:2005`). `cmd_reproduce` consumes it via `getattr(args,"run_spec",None)` → `_load_run_spec` (`cli.py:1621`). **The `/runs/upload` form read is wrapped `bool(_optional_form_bool(form,"autonomous"))`** — `autonomous` is a strict `bool` (no tri-state), and the bare `bool|None` return would 422 existing uploads; `bool(None)=False` fixes it.
- **T3 (`ee311db3`)** — `apply_autonomous_profile_override(request)` (module-level, `live_runs.py:~495`, after `apply_provider_override`), wired into `_start_python_run` **after** the sandbox/provider override chain (`~:845`, autonomous applied LAST so it wins). OFF (autonomous falsy) ⇒ returns the request object unchanged (true identity). ON ⇒ `model_copy(update={"sandbox":"gcp","model":"opus-foundry","run_spec": request.run_spec or "configs/autonomous_reproduction_run_spec.json"})`. Config `configs/autonomous_reproduction_run_spec.json` created (pure `OPENRESEARCH_*` + `models`, no secrets).

## 5. HARD-WON DECISIONS + LANDMINES (read before touching the backend)

1. **`sandbox="gcp"`, NOT `"gke"`.** `StartRunRequest.sandbox` is `Literal["auto","docker","local","runpod","azure","gcp"]` (`live_runs.py:43`) — **"gke" is not a member.** The `gke→gcp` alias lives only in the *execution.py* enum's `_missing_`. `"gcp"` is in the Literal, is what the UI emits, and directly selects `GkeJobBackend` (no `_missing_` needed). The plan's literal `"gke"` + `assert sandbox=="gke"` were **superseded** to `"gcp"` in T3. Any future "force the GKE backend" uses `"gcp"`.
2. **`run_spec_key_applies` is a pure PREFIX rule** (`run_spec_contract.py:40`): `OPENRESEARCH_`/`REPROLAB_` prefix, plus the two special keys `models`→`OPENRESEARCH_ROLE_MODELS` and `baseline_extra_guidance`→`OPENRESEARCH_BASELINE_EXTRA_GUIDANCE`. **Any** `OPENRESEARCH_*` key passes — including typos (which then silently no-op). The only typo guard is a manual `grep -rl KEY backend/`.
3. **Config forward-refs are intentional + inert:** `OPENRESEARCH_SPEC_VALIDATOR` and `spec_validator=grok` (inside `models`) are consumed only after T4/T6/T8. Verified inert today: `_parse_role_map` (`role_models.py:485`) keeps only `role in ROLES` (`:99`), which has no `spec_validator`, so the pair is dropped at parse and cannot raise. `sonnet-foundry`/`opus-foundry` are real `_ROLE_VOCAB` entries (`role_models.py:152–153`, provider `PROVIDER_ANTHROPIC_FOUNDRY`, in `_VALIDATED_SUBROLE_PROVIDERS` → no fidelity warning).
4. **⚠️ run_spec threads ONLY on the uploaded-paper branch.** In the generated script, `if config["uploaded_paper"]:` → `cmd_reproduce(Namespace(…, run_spec=config["run_spec"]))` (`live_runs.py:~2042`). The **non-upload `run_pipeline_hybrid` else-branch (`~:2069+`) does NOT thread `run_spec`.** If an autonomous run ever routes through the non-upload path, the ENTIRE autonomous profile (lifecycle-primary, spec-validator, all guards) goes **silently inert**. **VERIFY during T8/T14** that the autonomous arxiv/upload flow hits the uploaded-paper branch. (The arxiv path likely resolves to a downloaded PDF set as `uploaded_paper` — confirm.)
5. **⚠️ Rerun path drops the flags.** The rerun-path `StartRunRequest` (`app.py:~842–850`) forwards neither `autonomous` nor any advanced field (pre-existing narrowing, out of scope for T1–T3). A rerun of an autonomous run silently drops autonomy. Decide if "re-run stays autonomous" matters for the feature.
6. **`model_copy` skips validation** (pydantic v2) — that's *why* setting `"gcp"`/`"opus-foundry"` works cleanly, but don't rely on any later re-validation of an overridden request.
7. **Pre-existing full-suite failures (~16).** Exit bar = **no NEW** failures. If unsure whether a failure is yours, `git stash && pytest … && git stash pop` (T2/T3 implementers did this). Documented env-dependent baseline: `test_accelerator`, `test_external_validator`, `test_report_validation_stamp`, `test_gcp_orchestrator_settings::test_claude_code_oauth_token_prefixed_env_override` (these PASS with `.env` present).
8. **Corpus isolation (hard invariant):** the 4 new SSE spec-phase events + the session view carry ONLY ids/counts/verdict-enums/leaf-ids — NEVER paper text. Route through `sse_bridge`. `spec_validator` reads the full paper server-side; its persisted verdict holds leaf-ids only.
9. **Secrets server-side:** the Foundry key + cloud creds stay in `CredentialBroker`/`.env`; never in a run-spec on the wire or the browser. The config JSON is flags-only.

## 6. Per-task grounding for T3b → T14

Each task's implementer still reads its brief + does detailed recon. This front-loads the non-obvious (mirror-sources + anchors beyond the plan's own File/Interfaces sections). Line numbers drift — locate by content.

### T3b (P1) — `backend/routes/papers_resolve.py` (best-effort pre-run repo-resolve)
- `GET /papers/{arxiv_id}/repo` → `{repo_url, provider, confidence}` or `{repo_url: null}`. Reuse `backend/services/ingestion/repo/` `RepoResolver` — **VERIFY a resolve-WITHOUT-clone method exists** (`grep -n "class RepoResolver" backend/services/ingestion/repo/*.py`; find a resolve-only method, or add a thin `resolve_only`). 5 s bounded, fail-soft. Register the router in `app.py`. P1 = deferrable (T11 fail-soft-empties without it) — can be done later or skipped if time-boxed.

### T4 — `role_models.py` `spec_validator` sub-role (MIRROR the `validator` sub-role EXACTLY)
- `ROLES` (`:99`) + `_SUBROLES` (`:104`): add `"spec_validator"`.
- `RoleSelection` (`:276`): add `spec_validator: RoleSpec | None = None` (mirror `validator` `:291`); `explicit_subroles` loop (`:301`) + `RoleSelection.stamp` (`:305–318`) include it.
- **`RoleSpec.stamp` has a `validator` special-case** (`:250–257`: `OPENRESEARCH_VALIDATOR_MODEL` takes precedence over the role's own model for stamping). Decide whether `spec_validator` wants the parallel `OPENRESEARCH_SPEC_VALIDATOR_MODEL` precedence (likely yes for stamp fidelity, since T5 targets that env var).
- `resolve_role_models`: add `spec_validator_model_setting=None` param; resolve via `_resolve_subrole("spec_validator", …)` mirroring the validator block.
- The separation helper (`:343`, `executor × validator`) is reusable for `planner × spec_validator` (T8 needs it).
- Keep `tests/rlm/test_role_models.py` green (role-count / stamp assertions).

### T5 — `grader_transport.py` `build_spec_validator_client` (MIRROR `build_validator_client` `:358`)
- Shared dispatch is `build_transport_client(…, role_label=…)` (`:142`). `build_validator_client` reads `OPENRESEARCH_VALIDATOR_BACKEND` (`:390`).
- New: read `OPENRESEARCH_SPEC_VALIDATOR_BACKEND`/`_MODEL`, **fall back to** `OPENRESEARCH_VALIDATOR_BACKEND`/`_MODEL`; `role_label="spec_validator"`. **FAIL-CLOSED:** raise `ValueError` if the resolved client is identity-equal to the planner fallback (never judge the rubric with the lineage that wrote it). Add to `__all__` (`:952`? — it's near the grader_transport `__all__`, grep it).

### T6 — CREATE `backend/agents/rlm/spec_validator.py` (structural sibling of `external_validator.py`)
- Mirror the shape: `PredicateVerdict` (`:36`)/`ValidatorVerdict` (`:46`) → `SpecPredicateVerdict(predicate, leaf_id, violated, detail)` / `SpecValidatorVerdict(status, flagged_leaves, predicates, panel_models, separation, rubric_fingerprint)`; `external_validator_enabled`/`validator_panel_n` (`:64`/`:73`) → `spec_validator_enabled` (`OPENRESEARCH_SPEC_VALIDATOR`) / `spec_validator_panel_n` (`OPENRESEARCH_SPEC_VALIDATOR_PANEL_N`, default 2) / `spec_validator_block_enabled` (`OPENRESEARCH_SPEC_VALIDATOR_BLOCK`); `_parse_suspicions` (`:267`, fence-tolerant JSON-array parse); `run_validation_panel` (`:451`, calls `sample_completions` from `grader_transport` `:516`) → `run_spec_validation_panel(*, spec_validator_client, panel_models, rubric, paper_text, separation)`; `persist_verdict`/`load_verdict` (`:599`/`:630`, atomic + fingerprint staleness) → `persist_spec_verdict`/`load_spec_verdict(project_dir, *, expect_fingerprint=None)`; `rubric_fingerprint(rubric)->str` (sha256 canonical-JSON).
- **SWAP** the metrics-vs-disk machine-checks (`:91–266`) for **rubric-vs-paper** predicates (reuse `paper_grounding._normalize`/`_token_overlap`, `rubric_gen._is_placeholder_requirement`):
  - `hallucinated_leaf` — leaf cites a dataset/method/number with <0.5 max token-overlap vs the FULL paper text.
  - `wrong_target` — leaf's numeric target contradicts the paper's number for that metric (only when BOTH extract cleanly; else fail-soft `violated=False`).
  - `placeholder_leaf` — `_is_placeholder_requirement(leaf)` re-check.
  - `missing_key_claim` — **ADVISORY only**, never auto-remediated (open-world absence).
- min-aggregation: `violated` iff ANY panelist points at it AND the machine-check confirms (LLM opinion is never dispositive). `apply_block(rubric, verdict)` drops machine-CONFIRMED `hallucinated_leaf`/`wrong_target` leaves + renormalizes sibling weights via `rubric_gen._normalize_weights`; **NEVER** drops `missing_key_claim`; never hard-aborts.

### T7 — `sse_bridge.py` 4 spec-phase event builders (MIRROR `build_run_warning_event` `:913`)
- `_now_iso` is `:448`. Add after `build_repo_resolved_event` (`:539`): `build_spec_generation_started_event()`, `build_spec_generated_event(*, leaf_count:int)`, `build_spec_validation_started_event(*, validator_model:str)`, `build_spec_validated_event(*, verdict:str, flagged_leaves:list[str])`. Each `{"event": "<name>", "timestamp": _now_iso(), …}`. **CORPUS-FREE** (ids/counts/enums/leaf-ids only — the test must assert no builder accepts/emits paper text). Add all 4 to `__all__` (`:952`).
- **Event names are a cross-task contract** — T8 emits them, T12 reducers fold them: `spec_generation_started` / `spec_generated` / `spec_validation_started` / `spec_validated`. Spell identically everywhere.

### T8 — `run.py` integration (context field + client build + separation tier + hook + emits + report stamp)
- `context.py`: add `spec_validator_client: Any = None` (~`:58`).
- `run.py`: mirror the validator client build (`:3162–3194`, incl. the unified-surface bridge `:3166–3169`) using `build_spec_validator_client`; add `_spec_validator_separation_tier(role_selection)` mirroring `_validator_separation_tier` (`:2735`); thread `spec_validator_client` into `RunContext(...)`; the hook `_run_spec_validator(ctx, rubric, paper_text, project_dir, emit)` fires **strictly between the rubric cascade (`~:3564`) and the `RLM(...)` construction (`:3670`)** — before any of the 18 primitives can run (a true "before any GPU spend" gate). Emit `spec_generation_started` (before the cascade), `spec_generated{leaf_count}` (after), `spec_validation_started{validator_model}` / `spec_validated{verdict,flagged_leaves}` (bracketing the hook). All fail-soft.
- `report.py`: add `spec_validation: dict` field + a stamp parallel to the validator stamp (`load_spec_verdict(project_dir, expect_fingerprint=rubric_fingerprint(rubric))`).
- **OFF (`OPENRESEARCH_SPEC_VALIDATOR` unset) ⇒ no hook, no events, no `spec_validation` on the report — byte-identical.** BLOCK ON ⇒ confirmed-bad leaf dropped before `RLM(...)`.
- **⚠️ While here, resolve Landmine #4** — confirm the autonomous flow reaches this hook AND that `run_spec` actually applies on its subprocess branch.
- Tests: `tests/rlm/test_spec_validator_hook.py` (ON: hook+persist+stamp+4 events via a fake `emit`; OFF: none, byte-identical; BLOCK ON: leaf dropped). Also run `tests/rlm/test_report_validation_stamp.py` + a broad `tests/rlm/ -q`.

### Frontend T9a–T14
- Env: `nvm use v22.14.0`; `npm test` (vitest); `npx playwright test`; lab suite `--no-file-parallelism`; `npx tsc --noEmit`.
- **Do NOT repaint the dark lab** (`src/styles/tokens.css`). New surface is scoped `.autoresearch` tokens (`frontend/src/styles/autoresearch-tokens.css`) — light/maroon/neo-brutalist. Design tokens + component specs are in the plan's T9a/T9b. Match `alphaxiv_3.png`/`_4.png`/`_5.png`/`_autoresearc.png` (in `/home/abheekp/openresearch/`). Apply the `frontend-design` skill for polish.
- **T9a** tokens + base kit (Card/PrimaryButton/Eyebrow) → **T9b** flow kit (RepoField/Stepper/ReasoningChip/SessionRail/PaperTabs). TDD per component.
- **T10** thread `autonomous`+`repoUrl` through the launch chain (`use-run.ts` 3 launchers — camelCase FormData for upload/query, snake_case JSON for arxiv; `user-prefs.ts`; `lab-shell.tsx`; `upload-view.tsx` autonomous toggle + repoUrl field). Backend contract = T2/T3 (done): upload FormData key `autonomous`; arxiv JSON `autonomous` + `repo_url`.
- **T11** landing (`/abs/[arxivId]`) + repo-confirm + **cost-guard (D4)** (confirm dialog showing `OPENRESEARCH_MAX_RUN_GPU_USD`=$10 from the profile + stubbed auth gate → `startArxivRun`/`startUploadedRun` with `autonomous:true` → route `/sessions/<runId>`). Needs T9b, T10, T3b (repo pre-fill fail-soft).
- **T12** spec-validation stepper — consumes the 4 SSE events (names from T7), auto-redirects to `/sessions/<runId>` on `spec_validated`. `rlm-events.ts` union + `use-rlm-run.ts` `fold()` handlers + `session-events.ts` reducers. Needs T9b + T8 events.
- **T13** `SessionReasoningView` (`/sessions/[runId]`) — re-skin the EXISTING SSE stream (rides `useRlmRun` state; render only sanitized fields — corpus invariant). Build NEW; don't edit the dark `rlm-lab`. Needs T9b.
- **T14** Playwright e2e (mocked SSE emitting the 4 spec events then `repl_iteration`/`primitive_call`; assert corpus never in DOM) + a hermetic backend integration test (`autonomous=True` end-to-end forces `sandbox=gcp` + `model=opus-foundry` + `run_spec=<path>`; `autonomous=False` byte-identical). Needs T10–T13.

## 7. Accumulated Minor findings roll-up → feed to the FINAL whole-branch review

(Full list in `.superpowers/sdd/progress.md`.) Highlights: T1 coercion-helper + HTTP-path coverage gaps; T2 brittle substring assertion (drop it) + `_extract_common` regex fragility; T3 OFF-state `is r OR == r` (tighten to `is r`) + untested non-override-field preservation. Plus the two cross-cutting risks (Landmines #4, #5) — decide whether either needs a follow-up task before merge.

## 8. Start here (fresh session)

1. `cd /home/abheekp/openresearch-autonomous-ui` — `cat .superpowers/sdd/progress.md` and `git log --oneline -6` (confirm HEAD `ee311db3`, T1–T3 done). Trust the ledger over any recollection.
2. Read the **plan** + **this handoff** (+ the spec/original handoff if needed).
3. Invoke `superpowers:subagent-driven-development`. `BASE=$(git rev-parse HEAD)`. Extract the next brief (T3b: manual awk per §3; or start at T4 via the script). Ground the task (§5–§6), then dispatch a **Sonnet** implementer (TDD) with resolved ambiguities.
4. `review-package BASE HEAD` → dispatch an **Opus** reviewer with the binding constraints as the lens. Fix loop for Critical/Important. Mark the ledger.
5. Continue T4→T8 (backend), then T9a→T14 (frontend). After all tasks: dispatch the **Opus** whole-branch review (point it at the §7 roll-up) → `superpowers:finishing-a-development-branch`.
6. Push to `deepinvent` only when the user explicitly asks.
