# UI Progress Log

_Append-only log of UI-facing work. Newest first. Each entry: **what shipped → why → guardrail**. Distill failure modes here so the next session doesn't re-learn them._

---

## How to read / extend this doc

- **Format**: one entry per landed PR/commit cluster, dated, with a fixed shape (`Shipped` / `Why it broke` / `Root cause` / `Fix` / `Guardrail`).
- **Append-only**: never rewrite past entries; if a fix later regresses, add a NEW entry that references the old one.
- **Token-thrifty**: one-liners over paragraphs. Code snippets only when the rule is non-obvious. Tables for parallel facts.
- **No status theatre**: only landed work. WIP belongs in `docs/superpowers/plans/`.
- **Cross-link**: `(see learn.md §X)` for full RCAs, `CLAUDE.md` for live architecture, `docs/runbooks/e2e-testing.md` for ops.

---

## 2026-05-23 — Lab UI hardening sprint (commits `78f0870…1316b8a`)

### Shipped (7 commits, in dependency order)

| # | SHA | Subject | Lines |
|---|---|---|---|
| 1 | `78f0870` | RDR polling resilience — proxy timeouts + broader hook early-exit | +99/-31 |
| 2 | `a07c336` | Chat steering + collapsible right sidebar with kind-specific node detail | +1.4k/-83 |
| 3 | `f664659` | Runbook + CLAUDE.md + system_overview + start_backend.sh | +134/-11 |
| 4 | `290c4d0` | Preset arXiv chip row + API/OAuth chat-steering parity test | +302/0 |
| 5 | `e4c30a0` | Live status flip on primitive_call + real-time elapsed clock + RDR empty-200 stop | +90/-8 |
| 6 | `30d243f` (rebased to `1316b8a`) | Align rubric_area key contract + defensive empty-name handling | +95/-14 |

### New surfaces

| Feature | Component / endpoint | Source of truth |
|---|---|---|
| Right-docked node-detail sidebar (360px / 36px collapsed) | `components/lab/rlm/node-detail-sidebar.tsx` | Lifted `selectedNodeId` in `rlm-lab.tsx` |
| Steering chat panel (bottom of sidebar) | `components/lab/rlm/steering-chat.tsx` + `hooks/use-steering-chat.ts` | SSE events `user_message` / `user_message_response` |
| Backend chat endpoint | `POST /runs/<id>/messages` | `backend/routes/messages.py` |
| RLM primitives `check_user_messages()` / `respond_to_user()` | `backend/agents/rlm/primitives.py` | System prompt instructs polling at iteration start |
| Preset arXiv chip row | `components/lab/upload-view.tsx` | `lib/demo/preset-papers.ts` (4 papers) |
| Real-time elapsed clock | `components/lab/rlm/rlm-lab.tsx` | `runMeta.startedAt` + `setInterval(1000)` |
| RDR polling early-exit on empty | `hooks/use-rdr-artifacts.ts` | `isMissing(res, arrKey)` helper |
| New SSE event types | `lib/events/rlm-events.ts` | `user_message`, `user_message_response` |

### Failures we burned time on (and the rule they leave behind)

| # | Symptom seen | Root cause | Rule for next time |
|---|---|---|---|
| F1 | Lab UI stuck "Queued" + "0s elapsed" while backend run was clearly progressing | `useRlmRun.foldPrimitiveCall` only flipped status to "running" on `repl_iteration` events; SDK aclose deadlock blocked iteration completion → no event → no flip. Elapsed used event-span (also 0 with 0-1 events). | **Status/elapsed must derive from the earliest possible signal**: status = first event of any kind; elapsed = `startedAt` + ticking clock. Never gate UI liveness on a single optional event type. |
| F2 | Endless 404 spam on `/leaf-scores`, no convergence | F2-era fix stopped polling on 3× consecutive 404, but `/clusters` and `/repair-iterations` return 200 with empty arrays for non-RDR runs → `allMissing` stayed false. | **"Missing" means missing semantically, not just HTTP-404**: `status === 0 || status >= 400 || (status === 200 && empty array)`. Use a helper, not an inline expr. |
| F3 | 502 Bad Gateway on `/clusters` while backend hung | Proxy `fetch` had no timeout; Node's default ~5 min elapsed before the catch-all returned 502. | **Every proxy fetch needs an `AbortController` with a tight timeout (4s)**. On timeout/error, return a semantically-correct status (404 = "no data") not 502, so callers can converge. |
| F4 | React "duplicate key=`""`" warnings on rubric chip/breakdown rows (6 chips, all empty) | `_rubric_areas()` emitted `{"name": ...}` but `binding.py` read `a.get("area", "")` — silent key mismatch + auto-generated rubric had blank `requirements`. | **Pin wire-contract field names at the seam test**. Whenever a payload crosses a process boundary (primitive → SSE → React), add ONE contract test that asserts the exact field name. Defensive React keys: always `key={x.id \|\| `__idx_${i}`}` — never trust upstream to be non-empty. |
| F5 | `NodeDetailSidebar` was dead code (component existed but not wired into rlm-lab) | Sonnet sub-agent reported "all gaps closed" but had only created files, not integrated them. | **Trust the diff, not the summary**: every sub-agent claim is verified by direct file inspection. The probe agent that grep'd for the import caught it. Add `grep <import-name> <consumer-file>` to every sidebar/integration acceptance check. |
| F6 | Sonnet C marked work complete while tests for new components didn't exist | Async dispatch + "Already complete" status was a false signal. | **Acceptance criteria must include a verifiable command** (`ls test_file.tsx`). "Tests added" without the file present = task not done. |
| F7 | 4 tests pinned the primitive count = 10 and broke when we added `check_user_messages` + `respond_to_user` | Brittle pin tests freezing a constant that legitimately changes. | **Test contracts, not constants.** Prefer "every primitive resolves to a callable" over "exactly N primitives exist". Pinning the count is a regression magnet. |
| F8 | Backend hung in `do_wai` syscall, all endpoints timed out, run state file said "running" | SDK `aclose()` nested-generator deadlock under WSL2 futex. Workaround B exists in `claude_oauth_client.py` + `rlm_query.py` but doesn't catch every code path. | **UI must be resilient to backend hangs** (F1+F2+F3 jointly do this). Don't fix the backend by making the UI fragile. Document the kill+restart recipe in runbook §4f. |

### Architecture decisions worth preserving

- **Selection state lives in `rlm-lab.tsx`**, not in `ExplorationCanvas`. The canvas signals via `onSelectNode(id)`, the sidebar consumes the same `selectedNode` ref. One source of truth, no prop-drilling cascades.
- **Chat is pure file I/O** at the primitive layer (`user_messages.jsonl` + cursor). Works identically under API-key and OAuth auth. Auth-surface-agnostic; lock-in via `tests/rlm/test_chat_steering_auth_parity.py` (6/6 pass).
- **Sidebar collapse state is internal to the component** (`useState`), grid track stays a fixed `360px`. Don't lift collapse state up — the parent doesn't care, and lifting creates re-render cascades.
- **`runMeta.startedAt`** is the canonical clock anchor. Event-span elapsed is a legacy fallback for fixtures/replays only.
- **`isMissing(res, arrKey)`** is the canonical "no artifact" predicate for any polling hook. Don't reinvent the check inline — factor it.

### Files-of-record (don't break these without an entry here)

| File | Owns | Don't break |
|---|---|---|
| `lib/events/rlm-events.ts` | SSE event type union + `RLM_EVENT_TYPES` | Compile-time `_RlmEventTypesInSync` guard enforces the union ↔ array match |
| `hooks/use-rlm-run.ts:fold()` | Pure reducer; exhaustive switch over `RlmDashboardEvent["event"]` | Adding a new event type requires a switch case (default = no-op is fine; absence = TS error) |
| `hooks/use-rdr-artifacts.ts` | RDR artifact polling + early-exit | Counter MUST stay monotonic; "missing" predicate MUST include empty 200 |
| `components/lab/rlm/node-detail-sidebar.tsx` | Kind-dispatch for node detail (paper / work / candidate / subrlm / baseline / declined-group) | All 6 `TreeNode["kind"]` values must have a branch or a fallback |
| `app/api/demo/runs/[projectId]/{clusters,leaf-scores,repair-iterations}/route.ts` | Polling proxies | All three MUST have a 4s `AbortController` + normalize timeout/5xx → 404 |
| `backend/agents/rlm/binding.py:_emit_extra` | Event payload construction at the seam | Defensive key reads: `a.get("area") or a.get("name") or ""` survives upstream rename |

### Auth-mode parity invariants (locked in by `test_chat_steering_auth_parity.py`)

1. System prompt build with `["claude", "claude-oauth"]` must always include `check_user_messages` AND `respond_to_user`.
2. Primitive registry binds both new primitives under either alias.
3. `POST /runs/<id>/messages` MUST never gate on root-model config.
4. Both new primitives execute with zero LLM credentials present (file I/O only).

**Implication for prod cutover (API key)**: nothing in the chat surface needs changing. Just set `ANTHROPIC_API_KEY` with credits and pick `--model claude`.

### Open / deferred (don't claim "done")

- **SDK aclose deadlock**: documented (runbook §4f) but not fixed. UI is resilient; backend still needs upstream-SDK fix OR a primitive-side `repl_iteration` heartbeat track.
- **Cost ledger reports $0 for OAuth runs**: SDK doesn't surface token counts. Documented in CLAUDE.md. Use `max_wall_clock` not `max_usd` for OAuth.
- **Frontend vitest blocked on Node 21**: needs Node 22.12+. Document & enforce via `package.json` engines OR a preflight in `start_backend.sh`. (User has 21.7.3 in their shell; coworkers will hit this.)
- **Sub-RLM nodes**: the tree builds them from `sub_rlm_spawned` events, but the root model doesn't always spawn sub-RLMs. Not a bug; behavior depends on the run.
- **Node detail "now" block for `baseline`/`declined-group` kinds**: falls back to a "no iteration detail" string. Could be richer (last few primitive calls).
- **Backend watchdog timer**: kill the aclose-hung worker after N seconds with no event progress. Tracked, not built.

### Best practices distilled from this sprint

1. **Verify the diff, not the summary.** Every sub-agent claim → `grep`/`Read` confirm. The probe-agent pattern caught two cases of "completed but not really".
2. **Pin contracts at seams, not constants in tests.** Field names crossing process boundaries get a one-line contract test; primitive *counts* don't.
3. **Defensive React keys: always have an index fallback.** `key={x.id || `__i_${i}`}`. Costs one parameter, prevents a class of bug.
4. **Every fetch in a proxy needs an AbortController + tight timeout.** No exceptions; reuse the 4s pattern.
5. **UI liveness derives from the earliest available signal, not the most specific one.** Status flips on first event; elapsed ticks from `startedAt`.
6. **Commit infrequently at meaningful milestones, not per-fix.** This sprint = 6 commits for ~20 logical changes. Trade fewer commits for richer commit messages.
7. **Push `origin/main` directly; pull --rebase before push.** Coworker branches don't deadlock us; `--autostash` keeps the working tree intact.
8. **The runbook (`docs/runbooks/e2e-testing.md`) is the deploy bible.** Every new failure mode adds a `§4x` entry there.

### Regression checklist before any UI-affecting PR

```bash
# Backend
.venv/bin/python -m pytest tests/rlm/ tests/rdr/ tests/routes/ -q   # must pass clean
# Frontend
cd frontend && npx tsc --noEmit                                      # must be clean
node --version | grep -qE "v(22\\.|23\\.|24\\.)" && npm test         # if Node 22+

# Live smoke
./start_backend.sh &  # term 1
cd frontend && npm run dev   # term 2
# - open /lab → pick preset paper → Begin
# - confirm: status flips queued→running within 5s
# - confirm: elapsed ticks every 1s
# - confirm: no React duplicate-key console warnings
# - confirm: /clusters /leaf-scores /repair-iterations stop after ~15s for non-RDR run
# - confirm: clicking Paper / work / candidate nodes populates sidebar
# - confirm: typing in chat panel → POST 202 → user msg appears in panel
```

### Cross-references

- Architecture: `CLAUDE.md` §`RLM orchestrator`, `Chat steering surface (2026-05-23)`, `Collapsible right sidebar (2026-05-23)`
- Operations: `docs/runbooks/e2e-testing.md` §`4f Backend hangs`, §`6b Chat-steering walkthrough`
- Auth surfaces (API ↔ OAuth): `CLAUDE.md` §`RLM auth — two surfaces, billed separately`
- Past bugs + rules: `learn.md` (especially 2026-05-22 SDK aclose entry — Workaround B)

---

_End of 2026-05-23 entry. Append new entries above this line, leaving "How to read" at the top._
