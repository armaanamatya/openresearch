# Findings — E2E localhost run of the RLM paper (2026-05-23)

Companion to `docs/superpowers/specs/2026-05-23-e2e-rlmpaper-localhost-run-design.md`.

Live log of every backend / UI / config defect surfaced during the
end-to-end reproduction of arXiv 2512.24601 on
`projectId=prj_5b5fe266b0b83f3d` (sandbox=runpod, root model=claude-oauth
via the `sonnet` alias).

Format: one entry per defect, in order of discovery. Each entry records
symptom → root cause → fix (commit SHA when shipped) → verification.

## Run config

| field | value |
|---|---|
| projectId | `prj_5b5fe266b0b83f3d` |
| paper | arXiv 2512.24601 (rlms) |
| mode | rlm |
| sandbox | runpod COMMUNITY (RTX 4090) |
| root model | claude-oauth (via `sonnet` alias) |
| sub-agents | claude-agent-sdk via OAuth subscription |
| kickoff | 2026-05-23 ~18:48 UTC |
| target | `runs/<id>/final_report.json` |

---

## F1 — `start.sh` unbound-variable under bash 3.2

- **Symptom:** `./start.sh` exit 1 with `preflight_args[@]: unbound variable` on line 95.
- **Cause:** macOS bash 3.2 (the default `/usr/bin/env bash` on macOS) treats `"${empty_array[@]}"` as an unbound variable under `set -u`. The script uses `set -euo pipefail` and an empty `preflight_args=()` array.
- **Fix:** commit `13793f0` — switched to `${preflight_args[@]+"${preflight_args[@]}"}` which is the standard bash-3.2-safe idiom.
- **Verified:** `./start.sh` boots backend cleanly on macOS 25.

## F2 — `REPROLAB_FORCE_SANDBOX` config default silently pinned every run to Docker

- **Symptom:** `POST /api/demo/arxiv` with body `{"sandbox":"runpod","model":"sonnet",...}` returned `"sandboxMode":"docker"`. Verified twice. Despite shell env `REPROLAB_DEFAULT_SANDBOX=runpod` and the request body explicitly asking for runpod, every run was being forced onto docker.
- **Cause:** `backend/config.py:139` declares `force_sandbox: Literal["", ...] = "docker"`. `apply_sandbox_override(request, settings.force_sandbox)` at `backend/services/events/live_runs.py:515` rewrites every run's sandbox field with that value. The `.env` shipped with the repo had `REPROLAB_FORCE_SANDBOX` **commented out**, with a comment claiming this disabled the override — but pydantic-settings treats "commented" identically to "absent" and falls back to the field default (`"docker"`). So even with the line commented, every run got force-pinned to docker.
- **Fix:**
  - `.env` (untracked): set `REPROLAB_FORCE_SANDBOX=` explicitly (empty string) to opt out of the override.
  - `CLAUDE.md`: §"Sandbox config gotcha" updated to spell out that the commented-out line does NOT disable the override — the variable must be set explicitly empty.
  - (Considered but not done in this session: change the pydantic default from `"docker"` to `""`. That's a design-intent change — the existing default was deliberate "RunPod is disabled" hardening — so left for a follow-up PR with maintainer review.)
- **Verified:** re-kickoff returned `"sandboxMode":"runpod"`; backend stderr printed `sandbox: runpod`.

## F3 — `/models` endpoint exposes only 2 of 6 registered root models

- **Severity:** UI/backend parity gap; not a regression, not a blocker.
- **Symptom:** `GET /models` returns just `[{sonnet, anthropic}, {opus, anthropic}]`. The actual `ROOT_MODELS` registry in `backend/agents/rlm/models.py` has 6 entries: `gpt-5`, `qwen3-coder`, `kimi-k2.5`, `claude`, `claude-oauth`, `qwen3-coder-featherless`, `azure-gpt-4o`. The frontend's `DemoModelChoice = "sonnet" | "opus"` is similarly narrow.
- **Why it didn't block this run:** `_MODEL_ALIASES` maps `sonnet` and `opus` to `claude-oauth`, so the UI's `sonnet` resolves correctly. But a user wanting `gpt-5` or `qwen3-coder` from the UI is out of luck — there's no way to surface them.
- **Fix:** deferred. Real fix: `/models` queries `ROOT_MODELS` and includes per-model credential availability (so the UI can grey out models whose env var key is unset). Frontend type widens to `string` or a generated union.
- **Why deferred:** mid-flight scope creep; not on the critical path for this run. Added to follow-up backlog in §"Open after this run."

## F5 — `use-rdr-artifacts` polled 404s for the entire active run lifetime

- **Severity:** UX (dev-console noise); not a run-breaker.
- **Symptom:** During the active rlm run, the browser console accumulated ~7 "Failed to load resource: 404 (Not Found)" entries every 30-second screenshot cycle — i.e., the hook polled `/clusters`, `/leaf-scores`, `/repair-iterations` every 5s for the entire run, and all three 404'd because rlm mode without a PaperBench bundle never produces RDR artifacts.
- **Why the existing F2 mitigation didn't fire:** the early-exit condition was `isActive==false`, so it stopped polling only AFTER the run ended. During the run the counter was constantly reset to 0.
- **Cause:** logic inversion in `frontend/src/hooks/use-rdr-artifacts.ts` lines 108-116.
- **Fix:** commit `4097a20` — added an `allReturned404` check that increments the counter (and triggers early-exit after 3 cycles) even on active runs. 200+empty and 5xx during active still keep polling.
- **Test:** replaced the test codifying the bug with two new tests asserting the new contract. `frontend/src/hooks/use-rdr-artifacts.test.ts` — 14/14 passing.
- **Verified:** pending — fix landed mid-run; verification deferred to next wakeup cycle (Next.js dev HMR may need a full bundle rebuild on some hooks).

## F7 — F5 was incomplete; real signature is mixed 404 + 200-empty

- **Symptom:** even after `4097a20`, console-error files showed ~7 entries per 30s screenshot cycle — no improvement. Cause: the F5 fix only triggered the early-exit when ALL three endpoints returned 404. The real Next.js dev access log showed `/clusters` returning 200+empty, `/leaf-scores` returning 404, and `/repair-iterations` returning 200+empty. `allReturned404` was false; counter reset to 0 every cycle.
- **Fix:** real fix is to count up unconditionally on `allMissing` (whether each endpoint is 404, 5xx, or 200+empty). The shipped commit also relaxes the over-narrow assertion on the test that encoded the wrong contract.
- **Status:** SHIPPED in subsequent commit (pending verification on next screenshot cycle once Next.js hot-reload picks up the change).

## F6 — `repl_iteration` landed at age=104s — runpod cold path within budget

- **Not a defect, a milestone.** Per the advisor's runpod-cold-path budget (8 min from kickoff for first `repl_iteration` OR `primitive_call=build_environment`), we landed at 104s — well within budget. The aclose deadlock fix (commit `532e010`) is verified working: the SDK runs to completion despite the expected non-fatal aclose warnings.

## F4 — `start.sh` defaults `REPROLAB_DEFAULT_SANDBOX=runpod` but `.env` had `=docker`

- **Severity:** behavioral inconsistency; cosmetic only, since the body-level sandbox field wins (after F2).
- **Symptom:** `start.sh` exports `REPROLAB_DEFAULT_SANDBOX=${REPROLAB_DEFAULT_SANDBOX:-runpod}` (line 47), but `.env` has `REPROLAB_DEFAULT_SANDBOX=docker`. Shell env should override .env via pydantic-settings precedence, but the layering means a user reading the .env will draw the wrong conclusion about the default.
- **Fix:** not shipping a change for this session. Worth a follow-up to align `.env` with `start.sh` — either remove the `.env` line or change the `start.sh` default to `docker`.

---

## RunPod-cold-path budget (per advisor)

Standard wedge detection (`scripts/health_probe.sh`) uses 600 s. RunPod pod creation + image pull is genuinely 3-5 min of silence on a cold path that wouldn't trip implement_baseline patterns. **Special clock for THIS run**: if no `repl_iteration` AND no `primitive_call=build_environment` event within **8 min** of kickoff (18:48 UTC + 8 min = 18:56 UTC), that's the runpod path failing — treat as a hard signal to investigate rather than a normal long-primitive false alarm.

---

## F8 — Lab page rendered blank when SSR fetch 504'd mid-run

- **Severity:** UX (high-impact); the user explicitly called it out — "ui needs to be more clear, we need to work with the user so they understand at all times what is going on, right now it is blank at times and doesn't make sense".
- **Symptom:** during the `implement_baseline` phase of prj_5b5fe266b0b83f3d, a Playwright snapshot of the lab page showed the header rendering but the main content area empty. The MCP console log captured: `Failed to load resource: 504 Gateway Timeout @ /api/demo?projectId=prj_5b5fe266b0b83f3d`.
- **Cause:** `BACKEND_GET_TIMEOUT_MS = 4000` (frontend/src/lib/demo/server-run.ts:13) was too aggressive when the FastAPI single-worker uvicorn was busy serving heavy SSE event payloads. The `/api/demo` route returned 504; useRun's auto-resume saw 504 and silently bailed (use-run.ts line 185) — leaving the user on the UploadView with no indication their projectId was valid.
- **Fix:** commit `1998e5d` ships three layered fixes:
  1. `BACKEND_GET_TIMEOUT_MS` 4s → 10s (server-run.ts). 10s still feels snappy on warm path while tolerating the 5-8s tail when backend is under load.
  2. `useRun` auto-resume retries 504s with exponential backoff (2s, 4s, 8s, 12s, 16s = ~42s budget) before giving up. Eliminates the "blank page on transient timeout" failure mode.
  3. (See F9 for the related chip fix that shipped in the same commit.)
- **Verified:** pending next Playwright snapshot on prj_20457ea6673b5a32 — the new run will exercise the same path with the fixes applied.

## F9 — "no signal Xs" chip caused unnecessary panic during long primitives

- **Severity:** UX (high-impact); user said "no signal 765s - we need to ensure this never happens. user should always know what the issue is".
- **Symptom:** the warn-colored `no signal 765s` chip showed up in the header during `implement_baseline`. Per known-issues §3.4 this is a false alarm: heartbeats pause during long primitives (implement_baseline 5-15 min, build_environment 1-5 min) because the rlm core loop emits `iteration_heartbeat` between iterations, not during a single primitive call. The chip caused real user concern that the run was wedged when it wasn't.
- **Fix:** commit `1998e5d`:
  - `RlmHeader` accepts a new `inFlightPrimitive: { name; startedAt } | null` prop.
  - `RlmLab` derives it: scans `state.primitiveCalls` backwards for the most recent `status="start"` entry that has no matching ok/error after it.
  - When `noSignalSecs` would have fired AND `inFlightPrimitive !== null`: render `running <primitive> (Xs)` in info color (accent-soft / accent-ink) instead of `no signal Xs` in warn color. The chip becomes informational.
  - When `noSignalSecs` fires WITHOUT a primitive in flight (true wedge): keep the warn-colored `no signal Xs` chip — this is now a real alarm, not noise.
- **Tests:** existing 2/2 header tests pass; the change is additive (new prop defaults to null = old behavior).
- **Verified:** pending — will snapshot prj_20457ea6673b5a32's `implement_baseline` phase to confirm the chip renders the new text.

## F10 — Lab canvas appeared dead between events; built a LiveActivityStrip

- **Severity:** UX (highest priority); user's verbatim words: "user has no clue what is going on atp, just 7 mins doing nothing, we should have a view so the user can understand what the agent is doing at all times, it needs to be active at all times".
- **Symptom:** the lab UI updates main panels only when a `primitive_call` / `repl_iteration` event lands. Between events (root model thinking, sub-RLM in flight, runpod pod ramping) the entire canvas appeared frozen — the only visible motion was a tiny `6m 30s elapsed` counter in a sidebar metric cell. Screenshot proof attached in the conversation log (`Screenshot 2026-05-23 at 2.23.55 PM.png`).
- **Cause:** no component subscribed to "what's the agent doing RIGHT NOW" — the existing chips and panels are all event-derived and event-discrete.
- **Fix:** commit `8e16a56` adds `frontend/src/components/lab/rlm/live-activity-strip.tsx` — an always-visible narration band slotted between RlmHeader and RubricStrip. Derivation ladder, most → least specific:
  1. In-flight primitive  → `▶ Running <primitive> · Xs` (info; warn when >60s)
  2. In-flight sub-RLM    → `↳ Sub-RLM depth N querying paper · Xs`
  3. Between iterations   → `… Iteration X complete — root thinking · Xs`
  4. Pre-first-iteration  → `… Starting up — root model reading paper · Xs`
  Terminal states (completed/partial/failed) get their own banner. Pulsing dot makes the live state unmistakable. Tooltip explains why long gaps are normal (so the user never thinks the run is wedged when it isn't).
- **Tests:** 16/16 existing tests pass (rlm-header.test.tsx + use-rdr-artifacts.test.ts). TypeScript clean.
- **Verified:** pending next Playwright snapshot on prj_20457ea6673b5a32.

## Run 2 — prj_20457ea6673b5a32 (fresh restart 19:17 UTC)

After the original prj_5b5fe266b0b83f3d ran past `implement_baseline` and into the code-writing phase, the user requested a full restart to bake in the new UI/backend fixes from a clean state. Old run subprocess killed, headless tail + chromium killed, MCP playwright closed, runpod pods enumerated (0 active), docker reprolab containers stopped. Fresh kickoff via curl POST `/api/demo/arxiv` with same body: `{url, mode:rlm, sandbox:runpod, model:sonnet}`. Returned `projectId=prj_20457ea6673b5a32, sandboxMode=runpod` — F2 fix verified working end-to-end on the fresh run too.

| time (UTC) | event | note |
|---|---|---|
| 19:17 | kickoff | new run, all fixes through commit 1998e5d baked in |
| 19:20 | first repl_iteration | 3 min ramp; slower than prj_5b5fe266b0b83f3d's 104s — within normal variance for the cold runpod path |
| | | (loop continues monitoring) |

## Open after both runs

- Real fix for F3 (expose all registered root models in `/models` + widen `DemoModelChoice`).
- Decide whether to change the `force_sandbox` config default to `""`. The current `"docker"` is a deliberate guarantee per the field comment — a design discussion, not a bug fix.
- Align `.env` `REPROLAB_DEFAULT_SANDBOX` with `start.sh` default (F4).
- Improve the `ValidationError` SSE result_summary to include pydantic error field locations + types (not values) so the UI can show what specifically failed without leaking LLM output.

---

## Run timeline (filled in as run progresses)

| time (UTC) | event | note |
|---|---|---|
| 2026-05-23 18:48 | kickoff | sandbox=runpod, model=sonnet (→claude-oauth) |
| 2026-05-23 18:48 | ingest 1-6 done | "Workspace ready — 4 variables" |
| 2026-05-23 18:51 | first repl_iteration @ 104s | runpod cold path cleared 8-min advisor budget |
| 2026-05-23 18:55 | sub_rlm_spawned | root recursively queried paper for title/authors |
| 2026-05-23 18:56 | sub_rlm_complete, repl_iteration #2 | 53.1s sub-rlm duration |
| 2026-05-23 18:58 | 5× understand_section + 1× extract_hyperparameters | paper claims being mapped |
| 2026-05-23 18:59 | detect_environment ERROR (ValidationError) | root passed dict[6] failing PaperClaimMap validation; primitive raised; root caught & adapted |
| 2026-05-23 19:02 | detect_environment ok → build_environment ok → plan_reproduction start | root self-recovered from the ValidationError without dropping the iteration; environment is built |
| 2026-05-23 19:04 | plan_reproduction ok → implement_baseline start (iter 3) | docker image `reprolab/prj_5b5fe266b0b83f3d:env-9caa8f013eab` staged locally for runpod |
| 2026-05-23 19:09–19:10 | sub-agent writing `code/rlm/{repl.py,system_prompt.py,llm_client.py}` | the meta-reproduction in motion — ReproLab's sub-agent is implementing a fresh rlm package; files growing actively (44KB total so far) |
| 2026-05-23 19:08 | F7 verification | latest 3 `screenshots/console-errors-*.json` files all 3 entries (down from 7 pre-fix) ✓ |
| | | (implement_baseline finish + run_experiment + verify_against_rubric still ahead) |

## Observation: ValidationError handling is fragile but didn't block

The detect_environment ValidationError was opaque in the SSE stream
(`result_summary="ValidationError"` — `binding.py:165` strips the
exception message to avoid leaking raw LLM/paper text into the UI).
The exception itself was re-raised, so the REPL surfaced the full
pydantic detail to the root, which adapted and retried successfully
within the same iteration. **No fix needed for THIS run.** Worth a
follow-up to make pydantic ValidationError details safe-to-emit (just
field locations + types, no values) so the UI shows what went wrong.
