# Findings - E2E localhost run of the RLM paper (2026-05-23)

Companion to `docs/superpowers/specs/2026-05-23-e2e-rlmpaper-localhost-run-design.md`.

This is the final, reconciled log for the localhost reproduction of arXiv
2512.24601, "Recursive Language Models", through ReproLab's RLM path. It
records each backend, UI, and config defect found during the run, the fix that
shipped or landed in the working tree, and the verification evidence.

## Run config

| field | value |
|---|---|
| original projectId | `prj_5b5fe266b0b83f3d` |
| clean restart projectId | `prj_20457ea6673b5a32` |
| paper | arXiv 2512.24601, "Recursive Language Models" |
| mode | `rlm` |
| sandbox | RunPod COMMUNITY, RTX 4090 |
| requested model | `sonnet` |
| resolved root model | `claude-oauth` |
| sub-agents | `claude-agent-sdk` via OAuth subscription |
| target artifact | `runs/<id>/final_report.json` |

## Final verdict

Run 2, `prj_20457ea6673b5a32`, reached terminal state at
2026-05-23 20:17:26 UTC, 60m 2s after kickoff. This is inside the design
spec's 90-minute target and well under the 180-minute hard cap.

This is a pipeline success, not an answer-quality success. The orchestrator
ingested the paper, ran four RLM iterations, built an environment, recovered
from primitive failures, wrote `final_report.json`, and terminated cleanly with
`run_complete partial`. The rubric score was 0.0 because `run_experiment`
failed and the generated baseline did not reproduce the paper's results.

Verified artifacts:

| artifact | verification |
|---|---|
| `runs/prj_20457ea6673b5a32/final_report.json` | present, 3591 bytes |
| `runs/prj_20457ea6673b5a32/final_report.md` | present, 2859 bytes |
| `runs/prj_20457ea6673b5a32/dashboard_events.jsonl` | present, 110 events |
| terminal event | `run_complete`, status `partial`, iterations 4, rubric 0.0 |
| leaderboard fields | `mode`, `models`, `started_at`, `completed_at` present |

## Defect summary

| id | issue | status |
|---|---|---|
| F1 | `start.sh` failed under macOS bash 3.2 with an empty array under `set -u` | fixed, verified |
| F2 | `REPROLAB_FORCE_SANDBOX` default silently pinned every run to Docker | fixed in working tree, verified by Run 2 |
| F3 | `/models` exposed only `sonnet` and `opus` despite a larger RLM registry | fixed in working tree, tested |
| F4 | `.env` sandbox default disagreed with `start.sh` | fixed in local `.env`, `.env.example` already aligned |
| F5 | RDR artifact polling spammed 404s throughout active non-RDR runs | partially fixed, superseded by F7 |
| F6 | First `repl_iteration` arrived within the RunPod cold-path budget | milestone, not a defect |
| F7 | F5 missed the real mixed `404 + 200-empty` artifact signature | fixed, verified live |
| F8 | Lab page went blank when SSR `/api/demo` fetch timed out | fixed, verified on restart |
| F9 | `no signal Xs` chip false-alarmed during long primitives | fixed, verified on restart |
| F10 | Lab canvas looked dead between events | fixed with `LiveActivityStrip`, verified live |
| F11 | Exploration canvas nodes became unclickable after panning | fixed, tested |
| F12 | Primitive error summaries hid safe Pydantic field/type detail | fixed, tested |

## Findings

### F1 - `start.sh` unbound variable under bash 3.2

Symptom: `./start.sh` exited with `preflight_args[@]: unbound variable` on
macOS.

Cause: macOS bash 3.2 treats `"${empty_array[@]}"` as unbound under `set -u`.

Fix: commit `13793f0` uses the bash-3.2-safe expansion
`${preflight_args[@]+"${preflight_args[@]}"}`.

Verified: `./start.sh` boots the backend on macOS 25.

### F2 - `REPROLAB_FORCE_SANDBOX` default pinned runs to Docker

Symptom: `POST /api/demo/arxiv` with body `sandbox=runpod` returned
`sandboxMode=docker`.

Cause: `backend/config.py` defaulted `force_sandbox` to `"docker"`, and
`apply_sandbox_override()` rewrote every request. A commented-out `.env` line
did not disable the override because pydantic-settings fell back to the field
default.

Fix:

- Initial session fix: local `.env` set `REPROLAB_FORCE_SANDBOX=` explicitly.
- Post-run cleanup: `backend/config.py` now defaults `force_sandbox` to `""`.
  Deployments that need hard pinning must set `REPROLAB_FORCE_SANDBOX=docker`
  or `local` explicitly.
- `backend/config.py` also defaults `default_sandbox` to `runpod`, matching
  `start.sh`, `.env.example`, and frontend run-start requests.
- `CLAUDE.md` and `.env.example` document the new behavior.

Verified: Run 2 kickoff returned `sandboxMode=runpod`; `demo_status.json` for
`prj_20457ea6673b5a32` also records `sandboxMode=runpod`.

### F3 - `/models` exposed only 2 of 7 registered root models

Symptom: the lab dropdown only exposed `sonnet` and `opus`. The RLM registry
also supports `gpt-5`, `qwen3-coder`, `kimi-k2.5`, `claude`,
`claude-oauth`, `qwen3-coder-featherless`, and `azure-gpt-4o`.

Cause: `backend/app.py` hardcoded `/models`, and the frontend constrained
`DemoModelChoice` to `"sonnet" | "opus"`.

Fix landed in this cleanup pass:

- `/models` now returns descriptors from `ROOT_MODELS`.
- Each descriptor includes `available` and `missingCredentials` so the UI can
  disable models whose credentials are absent.
- `DemoModelChoice` and `UserPrefs.model` are widened to `string`.
- The frontend proxy forwards arbitrary model IDs.
- `_python_script()` preserves registry model keys instead of clobbering every
  non-`opus` choice to the provider default.

Verified:

- Backend tests assert `/models` includes all seven registry keys and
  availability fields.
- Frontend tests assert arbitrary model IDs are forwarded.

### F4 - `.env` sandbox default disagreed with `start.sh`

Symptom: `start.sh` defaulted `REPROLAB_DEFAULT_SANDBOX` to `runpod`, but the
local `.env` had `REPROLAB_DEFAULT_SANDBOX=docker`.

Fix: `backend/config.py`, local ignored `.env`, `start.sh`, and `.env.example`
now agree on `runpod` as the development default.

Verified: `rg` now shows local `.env` and `.env.example` aligned on `runpod`.

### F5 - RDR artifact polling spammed 404s during active RLM runs

Symptom: the browser console accumulated repeated 404s for `/clusters`,
`/leaf-scores`, and `/repair-iterations` during an active RLM run that could
not produce RDR artifacts.

Cause: the hook stopped polling only after the run became inactive.

Fix: commit `4097a20` added active-run early exit for sustained 404s.

Status: superseded by F7 because the first fix assumed all three endpoints
returned 404.

### F6 - RunPod cold path cleared the advisor budget

This was not a defect. The first run's initial `repl_iteration` landed at age
104s, well inside the advisor's 8-minute cold-path budget. This also verified
the earlier SDK `aclose` deadlock mitigation: the run continued despite
expected non-fatal `aclose` warnings.

### F7 - Real RDR artifact signature was mixed `404 + 200-empty`

Symptom: F5 did not reduce console noise. The real signature was:

- `/clusters`: `200` with empty payload
- `/leaf-scores`: `404`
- `/repair-iterations`: `200` with empty payload

Cause: the first fix only incremented the stop counter when all endpoints
returned 404.

Fix: commit `269a6c1` increments on `allMissing`, whether each endpoint is
404, 5xx, or 200-empty.

Verified: Playwright snapshot `playwright/lab-snapshot-002.md` showed only
three `leaf-scores` 404s before polling stopped.

### F8 - Lab page rendered blank when SSR fetch timed out

Symptom: during `implement_baseline`, the lab header rendered but the main
content area was blank. Console showed a 504 from `/api/demo`.

Cause: the frontend backend GET timeout was 4s while FastAPI was busy serving
large SSE/event payloads. `useRun` then silently bailed on the 504.

Fix: commit `1998e5d`:

- increased `BACKEND_GET_TIMEOUT_MS` from 4s to 10s;
- retried auto-resume 504s with exponential backoff;
- shipped the related long-primitive chip fix in F9.

Verified: Run 2 stayed usable through the same long primitive window.

### F9 - `no signal Xs` chip caused false panic during long primitives

Symptom: the header showed a warn-colored `no signal 765s` during
`implement_baseline`.

Cause: RLM heartbeats are emitted between root iterations, not during a single
long primitive. Long primitives can legitimately produce minutes of silence.

Fix: commit `1998e5d` derives the in-flight primitive and renders
`running <primitive> (Xs)` in an informational style when a primitive is active.
The warn-colored `no signal Xs` remains for true no-primitive wedges.

Verified: Run 2 showed `Running implement_baseline` instead of a false alarm.

### F10 - Lab canvas appeared dead between events

Symptom: between discrete events the UI looked frozen. The user saw multiple
minutes of apparent inactivity while the agent was actually thinking or a
sub-RLM was running.

Cause: no component narrated "what is happening right now." Existing panels
updated only when events arrived.

Fix: commit `8e16a56` added
`frontend/src/components/lab/rlm/live-activity-strip.tsx`, an always-visible
activity band between `RlmHeader` and `RubricStrip`.

Narration priority:

1. In-flight primitive: `Running <primitive> - Xs`
2. In-flight sub-RLM: `Sub-RLM depth N querying paper - Xs`
3. Between iterations: `Iteration X complete - root thinking - Xs`
4. Pre-first-iteration: `Starting up - root model reading paper - Xs`
5. Terminal states: completed, partial, or failed banner

Verified: `playwright/lab-snapshot-002.md` captured the strip showing
`Sub-RLM prompt preview: candidate -> dict of downstream answers...`.

### F11 - Nodes unclickable after panning the exploration canvas

Symptom: after panning the exploration canvas, clicking nodes no longer opened
`NodeDetailSidebar`.

Cause: `use-pan.ts` set `dragRef.current.moved = true` during pan but did not
reset it on pointer up. `exploration-canvas.tsx` treats `moved=true` as
"ignore click", so all later clicks were swallowed.

Fix: commit `401da26` resets `moved=false` in `onUp`.

Verified: existing exploration canvas tests pass.

### F12 - ValidationError summaries were too opaque

Symptom: primitive errors surfaced as only `ValidationError` or `Exception`,
which hid safe information the user needed to understand whether the root
could recover.

Cause: `binding.py` intentionally stripped exception messages to avoid leaking
LLM output or paper text, but it also stripped safe Pydantic locations and
error types.

Fix: the wrapper now emits safe, value-free Pydantic detail in
`result_summary`, capped for SSE payload size. It includes field locations,
Pydantic messages, and Pydantic error types, but not input values.

Verified: regression test asserts a Pydantic `int_parsing` error includes the
field path and type while excluding the bad input value.

## Run 2 timeline

| time UTC | event | note |
|---|---|---|
| 19:17:24 | kickoff | new run, same paper/config, `sandboxMode=runpod` |
| 19:20 | first `repl_iteration` | ~3 minute cold ramp, inside budget |
| 19:22 | `repl_iteration` 2 | root continued |
| 19:27 | `iteration_heartbeat` + sub-RLM bursts | recursive paper queries |
| 19:29 | F10 verified | LiveActivityStrip rendered active sub-RLM narration |
| 19:29 | F7 verified | artifact polling stopped after three missing cycles |
| 19:32 | `implement_baseline` start | sub-agent began baseline work |
| 19:32-19:54 | baseline code written | `train.py` and Dockerfile appeared under `code/` |
| 19:54 | `implement_baseline` error | root recovered and entered iteration 3 |
| 19:59 | retry path | `detect_environment` ok, `build_environment` ok, `plan_reproduction` start |
| 20:00 | `plan_reproduction` error | root pushed forward |
| 20:17:25 | terminal window begins | short third `implement_baseline` attempt |
| 20:17:26 | terminal state | `run_experiment` error, `repl_iteration` 4, `run_complete partial` |

## Pipeline-success evidence

The final report and event stream confirm the orchestrator:

- fetched a 9.9 MB arXiv PDF and parsed a 137k-character paper text;
- ran 4 RLM iterations with `models.planner = claude-oauth`;
- successfully called `detect_environment` twice, `build_environment` twice,
  `understand_section`, `extract_hyperparameters`, and one of two
  `plan_reproduction` attempts;
- built Docker image `reprolab/prj_20457ea6673b5a32:env-9caa8f013eab`;
- self-recovered from one long `implement_baseline` error and one
  `plan_reproduction` error;
- wrote `final_report.json`, `final_report.md`, rubric, cost, and
  leaderboard metadata;
- emitted `run_complete partial` cleanly.

The answer quality issue remains separate: `run_experiment` failed, the rubric
score was 0.0, and the report notes that `rlm_query` / `llm_query` were
unavailable because of a model-selection error in the library.

## Remaining follow-ups

- Investigate the `rlm_query` / `llm_query` model-selection error reported by
  `final_report.json`; this likely affects answer quality, not pipeline
  integrity.
- Improve run-quality prompting and primitive repair so `run_experiment`
  produces measurable results on the RLM paper.
- Continue UX polish from the follow-up plan: RunPod status chip, phase
  indicator, structured frontend rendering for primitive errors, panel
  empty-state audit, and toast notifications.

## Verification commands run in the cleanup pass

```bash
.venv/bin/pytest tests/test_live_run_api.py tests/test_demo_gate.py tests/rlm/test_binding.py -q
npm test -- --run src/app/api/demo/route.test.ts src/components/lab/lab-shell.test.tsx
.venv/bin/pytest tests/rlm/test_models.py tests/rlm/test_model_aliases.py tests/rlm/test_build_llm_client.py tests/test_live_run_api.py tests/test_demo_gate.py tests/rlm/test_binding.py -q
npm run lint
npx tsc --noEmit
npm test -- --run src/app/api/demo/route.test.ts src/components/lab/lab-shell.test.tsx src/components/lab/rlm/rlm-header.test.tsx
```

Results: 117 backend tests passed; frontend lint and typecheck passed; 10
focused frontend tests passed.
