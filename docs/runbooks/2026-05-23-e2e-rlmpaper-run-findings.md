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

## Open after this run

- Real fix for F3 (expose all registered root models in `/models` + widen `DemoModelChoice`).
- Decide whether to change the `force_sandbox` config default to `""`. The current `"docker"` is a deliberate guarantee per the field comment — a design discussion, not a bug fix.
- Align `.env` `REPROLAB_DEFAULT_SANDBOX` with `start.sh` default (F4).

---

## Run timeline (filled in as run progresses)

| time (UTC) | event | note |
|---|---|---|
| 2026-05-23 18:48 | kickoff | sandbox=runpod, model=sonnet (→claude-oauth) |
| | ingest 1-6 done | "Workspace ready — 4 variables" |
| | | |
