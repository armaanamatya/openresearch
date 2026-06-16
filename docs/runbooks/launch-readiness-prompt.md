# Comprehensive ship-readiness prompt — paste verbatim into a fresh `/clear`ed session

_Authored 2026-05-23. Use when you need to drive the entire stack — UI, backend, APIs, middleware, paper reproduction, tests, docs, cleanup — from "current state" to "demo-grade, shippable" in one autonomous pass. Fires a fresh agent with no prior context, so it audits + fixes honestly. **The agent does not stop until every phase is green.**_

---

## PASTE THIS PROMPT

You are the lead engineer driving the **OpenResearch / ReproLab** codebase at `/Volumes/CS_Stuff/openresearch` to a fully shippable state in one autonomous session. A Microsoft VP review is imminent. Quality bar: **precise, accurate, 100% reliable, elegant**. Demo-grade work — every surface a viewer touches must be polished.

The system reproduces research papers end-to-end via three modes:
- `--mode rlm` (default since PR #80) — hybrid: RDR phase 1 + RLM phase 2.
- `--mode rdr` — pure rubric-driven controller.
- `--mode rlm-pure` — pre-hybrid escape hatch.

A persistent `/leaderboard` ranks runs. A `/lab` page visualizes a live run (exploration canvas, rubric strip, primitive history, score climb). A landing page sits at `/`.

### Your mission

Drive every surface listed below to **ship-ready** in one pass. **You do not stop, hand off, or "surface for direction" until every phase is green and every checkbox in the final summary is ticked.** Surface findings inside the run only when you are genuinely blocked (a destructive operation needs explicit authorization, or you hit a credential/network failure you can't resolve).

The 10 phases are sequential where data flows down (Phase 2 needs Phase 1, etc.) and you should batch parallel work within each phase. Open one PR at the end if there are changes; don't open intermediate PRs.

---

### Required reading (before any tool call beyond `git status` + `git log`)

In this exact order, fully:

1. `CLAUDE.md` — repo conventions, two-process Docker model, RLM auth surfaces, sandbox gotchas. The `.env` lines 14–18 pitfall is real.
2. `system_overview.md` — why the system fits together the way it does. SSE event types, file-backed run state, sandbox layers.
3. `docs/runbooks/e2e-testing.md` — canonical end-to-end debug + test reference.
4. `docs/runbooks/readiness.md` — describes the optional `scripts/readiness.sh` you should run for tiers 1–5. If the script exists, prefer it over re-implementing the checks; only manually verify tiers 6+7.
5. `docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md` — §3 locked decisions (do not relitigate), §5 phase specs, §8 acceptance.
6. `docs/superpowers/plans/2026-05-23-phase1-rdr-lab-ui.md` — what the lab-UI gap-close work looks like if it hasn't shipped yet.
7. `learn.md` top 10 entries — recent root causes, guardrails.
8. `progress.md` — running journal.

---

### Working mode

- **Parallel by default.** Within a phase, batch independent tool calls in one message. Spawn `feature-dev:code-explorer` sub-agents for read-only investigation across disjoint file sets.
- **TDD where new code is being written.** Failing test → minimal implementation → passing test → commit.
- **Evidence over assertion.** Run the verification, read the output, then claim done. A passing self-test ≠ evidence.
- **Commits.** Infrequent — one commit per major milestone, not per fix. **NEVER** a `Co-Authored-By` / AI-attribution trailer (project rule per CLAUDE.md). If the local hook adds one, amend before push.
- **Branch.** Work on a chore/ship-readiness or feat/ship-readiness branch off the latest `origin/main`, not directly on main. Open one PR at the end.
- **Remote.** Push to `origin` (`armaanamatya/openresearch`). NEVER `replix`.
- **Push policy.** Push at the very end of Phase 10 only. Do not push intermediate work.

---

## PHASE 1 — Comprehensive audit (parallel)

Spawn three `feature-dev:code-explorer` sub-agents **in parallel** in one message:

1. **Backend + API + middleware audit.** Read `backend/` exhaustively. Map: every FastAPI route (path + handler + auth + response model), every SSE event type (emitted by which file:line, sanitized by `sse_bridge.py`), every middleware (CORS, demo gate, error handlers), every dependency injection (`Depends`), every background task. Identify: dead routes, missing error handlers, routes lacking response models, SSE events not routed through the sanitizer, middleware that silently swallows exceptions. Output a per-route + per-event table.
2. **Frontend audit.** Read `frontend/src/` exhaustively. Map: every route under `app/`, every shared component, every hook in `hooks/`, every event consumer of SSE, every proxy route under `app/api/demo/`. Identify: console errors on any route, hardcoded colors (vs. tokens in `frontend/src/styles/tokens.css`), components missing loading/error/empty states, accessibility holes (`aria-*`, semantic HTML, color contrast), broken anchors (`href="#"`), broken images, hydration mismatches. Output a per-route + per-component disposition.
3. **Paper reproduction surface audit.** Read `backend/agents/{rlm,rdr,hybrid,runtime}/`, `backend/evals/paperbench/`, `backend/services/ingestion/`, `tests/{rlm,rdr,services/}`. Map: every primitive (`understand_section`, `extract_hyperparameters`, etc.), every rubric scorer, every honesty-cap site, every sandbox (`local`, `docker`, `runpod`). Identify: stale references to removed `--mode sdk` / `--mode offline`, leaks of paper corpus past `sanitize_iteration`, primitives without per-call deadlines, sandboxes without resource limits, the C2 honesty cap (`DEGRADED_LEAF_CEILING = 0.35` at `backend/evals/paperbench/leaf_scorer.py`) applied consistently. Output a per-primitive table + a "honesty surface" report.

Wait for all three reports. Then continue.

---

## PHASE 2 — Backend perfection

Acceptance: every check below passes; commit the fixes as one `fix(backend): close audit findings` commit.

- [ ] `from backend.app import create_app; create_app()` imports + boots without error.
- [ ] `python -m backend.cli reproduce --help` shows `--mode {rlm,rdr,rlm-pure}` (no stale `sdk` / `offline`).
- [ ] `python -m backend.cli ingest --help` works.
- [ ] All routes from Phase 1's audit have: a Pydantic response model OR a documented free-form shape, a typed handler, and at least one happy-path test. Add tests for any missing.
- [ ] Demo-gate enforcement: every mutating route (POST/PUT/PATCH/DELETE) either calls `_enforce_demo_gate(...)` or is explicitly marked as public in a comment. List the public routes; defend each.
- [ ] `backend/config.py::Settings` has zero `Optional[...] = None` fields that downstream code de-references without a None check. Grep for the pattern; fix any unguarded access.
- [ ] `backend/services/runtime/` sandboxes (`local`, `docker`, `runpod`) have a `max_pod_seconds` budget applied. Verify the `RunBudget` is threaded into each.
- [ ] Sanitizer: every SSE event emission goes through `backend/agents/rlm/sse_bridge.py::sanitize_iteration` (or the equivalent for rdr). Grep for `dashboard.emit(` and `ctx.emit(`; flag any that bypass the sanitizer with raw payload.
- [ ] Honesty cap: `DEGRADED_LEAF_CEILING = 0.35` at `backend/evals/paperbench/leaf_scorer.py`. The cap is applied (a) in `score_reproduction` and (b) in `amend_final_report`. Test that a stub run with `metrics={}` produces `degraded=True` and `overall_score ≤ 0.35`.

When all green: `git add -p` the fixes, commit as `fix(backend): close ship-readiness audit findings (Phase 2)`.

---

## PHASE 3 — Frontend perfection

Acceptance: every check below passes; commit as `fix(frontend): close audit findings` commit.

- [ ] `cd frontend && npm run lint` clean (zero errors; warnings noted).
- [ ] `cd frontend && npx tsc --noEmit` clean.
- [ ] `cd frontend && npm test` all pass.
- [ ] Visit each public route (`/`, `/lab`, `/leaderboard`, `/library`) in `npm run dev`: zero console errors, zero failed network requests, no hydration warnings.
- [ ] Every component reads colors / spacing / typography from `frontend/src/styles/tokens.css` — zero hardcoded hex / rgb / px values outside the tokens file itself. Use `tools/check-no-hardcoding.sh` if present.
- [ ] Every list / table component has a non-trivial empty state (`<EmptyState>` or equivalent — not just whitespace).
- [ ] Every async data-fetcher has a loading state, an error state, and a success state.
- [ ] Every `<a href>` resolves to a real route or external URL — no `href="#"` placeholders.
- [ ] Every `<button>` has an `aria-label` or accessible inner text.
- [ ] The lab page handles all 3 modes (`rlm`, `rdr`, `rlm-pure`) without crashing — the canvas, popup, and rubric strip all render. If the rdr cluster-event reducer isn't wired, follow `docs/superpowers/plans/2026-05-23-phase1-rdr-lab-ui.md`.
- [ ] The landing page renders all sections (hero + §1.0–§7.0 if ported) without overlapping cells. Check `1280×800`, `1440×900`, and `390×844` viewports via Playwright MCP.

When all green: commit as `fix(frontend): close ship-readiness audit findings (Phase 3)`.

---

## PHASE 4 — APIs + middleware perfection

Acceptance: full request lifecycle is correct + observable.

- [ ] Every Next.js `app/api/demo/*` proxy route forwards: the `X-Demo-Secret` header when present, the request body verbatim, and the response status + body. Verify with a `curl` round-trip on a running pair (backend on `:8001`, frontend on `:3001`).
- [ ] SSE bridge: a single `--mode rlm` run produces `repl_iteration`, `primitive_call`, `rubric_score`, `candidate_proposed`, `candidate_outcome`, `run_complete` event types. A `--mode rdr` run produces the rdr-flavored variants (`rdr_run_started`, `rdr_cluster_started`, etc.) plus the spec-named cluster events if Phase 1 plan landed (`cluster_started`, `cluster_artifact_emitted`, `cluster_scored`, `repair_dispatched`).
- [ ] CORS: there is NO CORS middleware (the frontend talks to backend server-side only). Verify; if a `CORSMiddleware` snuck in, remove it.
- [ ] Demo gate: with `OPENRESEARCH_DEMO_SECRET=foo` set, `curl http://127.0.0.1:8001/runs` (POST) returns 401 without the header, 200 with the header.
- [ ] `GET /leaderboard` is NOT gated (read-only, public by design per spec §3 #10).
- [ ] Error handlers: a 500 in a route returns a JSON `{detail: "..."}` body, not an HTML stack trace. Test by curl-ing a route with a deliberately malformed payload.
- [ ] Cost ledger: every primitive call appends to `runs/<id>/cost_ledger.jsonl` with `{primitive, cost_usd, tokens_in, tokens_out, timestamp}`. Verify after a tier-6 smoke run.
- [ ] File-backed state atomicity: `demo_status.json` writes are atomic (temp file + rename, not in-place). Grep `_write_demo_status` to verify.

When all green: commit as `fix(api+middleware): close ship-readiness audit findings (Phase 4)`.

---

## PHASE 5 — Paper reproduction surface perfection

Acceptance: every PaperBench bundle in `third_party/paperbench/` reproduces honestly.

- [ ] Inventory the bundles: `ls third_party/paperbench/`. For each, confirm `paper.md`, `rubric.json`, and any required code/data exist.
- [ ] Bundle identity test: each bundle's `paper.md` first 100 characters match the expected paper title (spec §5.3.1 mentions `tests/test_paperbench_bundle_identity.py` as the guard — confirm it exists or add it).
- [ ] Ingestion cascade: `backend/services/ingestion/parser/resolving_parser.py::ResolvingParser` tries HTML → PDF → OCR in order. Verify each path with a fixture in `tests/services/`.
- [ ] arXiv ingestion: `ArxivFetcher` writes the HTML sibling for the given arXiv ID. Test with `2512.24601` (the canonical sample).
- [ ] Per-primitive deadlines: every primitive in `backend/agents/rlm/primitives.py` is wrapped in a `RunContext` deadline. Grep `RunContext(`; verify no primitive bypasses.
- [ ] Verify-rubric primitive returns a structured `RubricVerification` with non-empty `areas` even if all scores are 0. If empty, the test `tests/rlm/test_verify_against_rubric.py` should fail — confirm it doesn't xfail.
- [ ] Real-run smoke: `python -m backend.cli reproduce ftrl --mode rlm --max-usd 0.50 --max-wall-clock 600 --sandbox local`. The run must complete in budget OR fail with `degraded=True` + `overall_score ≤ 0.35`. Anything else = silent failure mode; fix it.
- [ ] rdr-mode smoke: same as above with `--mode rdr`. Verify the rdr controller produces `cluster_*` events (per Phase 4 acceptance).

When all green: commit as `fix(paper): close ship-readiness audit findings (Phase 5)`. If acceptance fails on a bundle, file a `learn.md` entry with Symptom → Root cause → Fix → Lesson → Guardrail and continue.

---

## PHASE 6 — Testing perfection

Acceptance: zero failing tests, zero unexplained xfails, no flaky reruns needed.

- [ ] `pytest tests/ -n auto`: full pass. Baseline 2026-05-23 was 1083 passed / 0 failed / 1 xfailed (`tests/rlm/test_build_environment_timeout.py` — T24 ThreadPoolExecutor shutdown hang). If T24 is fixed, remove the xfail.
- [ ] `cd frontend && npm test`: full pass. Baseline ≥85 tests at the rubric-climb landing.
- [ ] `cd frontend && npx playwright test`: full pass. Skip if Playwright not configured.
- [ ] No test references a removed module (`backend.pipeline`, `backend.agents.sdk_agent`, `backend.agents.offline_pipeline`). Grep; fix any.
- [ ] No test uses `--mode sdk` or `--mode offline` literals. Grep `mode="sdk"`, `mode="offline"`; replace with `mode="rlm"` (impl ignores the arg anyway).
- [ ] Every test file in `tests/` imports at least one symbol from `backend.*` OR is a documented infra guard (e.g. `tests/services/test_ocr_eng_installed.py`).
- [ ] Coverage: `pytest --cov=backend tests/` reports ≥70% line coverage on `backend/agents/` and `backend/routes/`. If below, add tests for the gap.

When all green: commit as `test: close ship-readiness audit findings (Phase 6)`.

---

## PHASE 7 — Docs perfection

Acceptance: every doc still accurately describes reality.

- [ ] `find docs -name "*.md" | wc -l` ≤ 14 (allows for active 2026-05-23 work). Delete anything stale per cleanup-spec §5.4.1.
- [ ] `README.md` leads with one sentence describing what the system does, then 5–10 lines of setup, then links out. No marketing copy.
- [ ] `CLAUDE.md` describes the day-to-day correctly: every command runs as documented; every env var listed exists in `backend/config.py::Settings`; every architectural fact matches the current code.
- [ ] `system_overview.md` matches the actual architecture: bands of `/lab`, SSE event list, sandbox list, primitive list, mode list (`rlm` / `rdr` / `rlm-pure`).
- [ ] `CHANGELOG.md` `[Unreleased]` lists every change since the last tagged release. Add bullets for anything missing.
- [ ] `learn.md` has an entry for every non-obvious bug fixed in this audit (Symptom → Root cause → Fix → Lesson → Guardrail).
- [ ] `progress.md` updated with a one-paragraph "ship-readiness pass" entry.
- [ ] No doc references a deleted file/path. Grep doc files for paths; verify each exists.
- [ ] No spec mentions stale modes (`sdk`, `offline`). Grep; update.
- [ ] Cleanup-spec §3 reflects PR #80's hybrid landing (default `--mode rlm` = hybrid; `rlm-pure` is escape; `rdr` is peer). If §3 #1 still says "rlm = pure RLM," amend.

When all green: commit as `docs: ship-readiness sweep — accuracy + freshness (Phase 7)`.

---

## PHASE 8 — Cleanup perfection

Acceptance: zero stale anything left in the working tree, history, or remotes.

- [ ] `find . -maxdepth 1 -type f \( -name "*.png" -o -name "*.tmp" -o -name "*.bak" -o -name "*~" \) -not -path "./.git/*"` returns empty. `rm` any.
- [ ] `.gitignore` covers: `.venv/`, `frontend/node_modules/`, `frontend/.next/`, `runs/`, `logs/`, `*.db`, `.DS_Store`, `.playwright-mcp/`, `.deepeval/`, `frontend/tmp-screens/`, root `/*.png`. Add any missing.
- [ ] Tracked-but-junk: `git ls-files | grep -E "Screenshot|paperbench_componet|runs_e2e_(local|docker)/"` returns empty. `git rm` any matches.
- [ ] Hollow stub packages: `find backend -name "__init__.py" -size -100c -not -path "*/test*"` — open each; delete the dir if no other files + no external importer.
- [ ] Stale branches: `git branch -r --merged origin/main | grep -v "main\|HEAD"`. Delete each: `git push origin --delete <branch>`. Always preserve `origin/main`, `origin/HEAD`, and any branch with an open PR.
- [ ] Stale stashes: `git stash list`. If any reference a deleted branch or are older than today, `git stash drop` them.
- [ ] Untracked-but-stale: `git status --short | grep '^??' | head -50`. For each: KEEP (active work) / GITIGNORE / RM. Decide per-file; don't auto-rm.
- [ ] Open PRs: `gh pr list --state open`. For each: still relevant? If draft + conflicting + > 7 days old, surface to user before closing.

When all green: commit as `chore(cleanup): ship-readiness sweep — purge stale artifacts (Phase 8)`.

---

## PHASE 9 — Final verification

Run the readiness script (or its inline equivalent) end to end, including the opt-in tiers:

```bash
READINESS_FRONTEND_SMOKE=1 \
READINESS_RUN_SMOKE=1 \
scripts/readiness.sh --json > /tmp/readiness-final.json
```

Or, if the script doesn't exist, manually re-run every command from Phases 2–6 in sequence and confirm green.

- [ ] Every tier reports PASS or WARN. Zero FAIL.
- [ ] Working tree clean: `git status --porcelain` empty.
- [ ] `git rev-list --left-right --count origin/main...HEAD` shows your branch is ahead, behind 0.
- [ ] `git log -1 --format="%B"` on every new commit shows NO `Co-Authored-By` trailer. If present, amend each.

---

## PHASE 10 — Push + PR

- [ ] Push the branch: `git push -u origin HEAD:chore/ship-readiness-<YYYY-MM-DD>`.
- [ ] Open the PR with `gh pr create` using the body template below.
- [ ] DO NOT auto-merge. Stop after opening the PR.

PR body template:

```markdown
## Summary

Ship-readiness sweep — drove every surface (UI, backend, APIs, middleware, paper reproduction, tests, docs, cleanup) to the demo-grade bar in one autonomous pass.

## Phase results

| Phase | Result | Detail |
|---|---|---|
| 1 Audit | <complete / partial> | <one-line> |
| 2 Backend | <pass / amended N findings> | <one-line> |
| 3 Frontend | <pass / amended N findings> | <one-line> |
| 4 APIs + middleware | <pass / amended N findings> | <one-line> |
| 5 Paper reproduction | <pass / amended N findings> | <one-line> |
| 6 Testing | <pass count> | pytest=X passed / Y failed / Z xfailed; vitest=A passed |
| 7 Docs | <pass / amended N> | <one-line> |
| 8 Cleanup | <N files removed, M branches deleted, K stashes dropped> | <one-line> |

## Tier-6 smoke

<paste the ftrl --mode rlm + ftrl --mode rdr verdicts + scores>

## Tier-7 deploy surface

<paste docker / demo-gate / RunPod / Claude-auth states>

## Test plan

- [x] `scripts/readiness.sh --tier 7` — green
- [x] `pytest tests/ -n auto` — green
- [x] `cd frontend && npm test && npm run lint && npx tsc --noEmit` — green
- [x] Manual UI walkthrough: /, /lab, /leaderboard, /library

## Out of scope

- Anything that requires a `Co-Authored-By` hook change (out-of-band)
- Anything in PR #74 (separate cleanup-spec §5.4.3 / Phase 5)
- Long-term content additions (architecture.md, etc.) — flagged for next session
```

---

## Forbidden actions

- Do NOT push to `main` directly.
- Do NOT force-push to `main`.
- Do NOT skip pre-commit hooks unless they're actively broken (then fix the hook).
- Do NOT add `Co-Authored-By` trailers.
- Do NOT delete a branch that has an open PR.
- Do NOT modify `.env`, `.env.example`, or anything under `runs/`.
- Do NOT spend more than $5 in tier-6 smokes total.
- Do NOT use `--mode sdk` or `--mode offline` literals anywhere (they were removed in PR #72).
- Do NOT relitigate cleanup-spec §3 locked decisions.

## When to ask the user (only when truly blocked)

- A destructive operation outside the listed acceptance criteria (e.g. you'd `git rm` a file the user has open in their IDE).
- A credential is missing AND there's no graceful fallback (e.g. Claude auth fails AND no API key AND no Keychain OAuth).
- A test failure that requires API-level changes to `backend/agents/rlm/primitives.py` signatures — these are spec §3 territory.
- More than 3 phases produced FAIL outcomes that you couldn't fix in <5 minutes each — surface as "this is bigger than one session" and stop.

## Default when uncertain

Read more code, not more docs. The codebase is authoritative on what exists; the spec is authoritative on what should exist. If they conflict on a non-§3-locked decision, the codebase wins.

## Don't stop until done

This prompt is autonomous. **You do not stop, hand off, or "surface for direction" between phases unless one of the four "ask the user" conditions above triggers.** When all 10 phases are green and the PR is open, write a 3-sentence final report ("shipped X, Y, Z; flagged A, B for follow-up; ready for review at PR #N") and only then stop.

GO.
