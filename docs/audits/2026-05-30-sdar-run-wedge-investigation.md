# SDAR Run Wedge Investigation — 2026-05-30

**Author:** Claude Code (Opus 4.8) operator session
**Branch:** `feat/rlm-wedge-hardening`
**Goal of the session:** Launch an SDAR (arXiv 2605.15155) reproduction end-to-end on `--sandbox runpod --model claude-oauth` and confirm the *whole* workflow ran (real `run_experiment` with `success=true` + metrics, ≥1 `run_experiment` cost-ledger row, `final_report` verdict ≠ `failed`).

**Outcome:** The pipeline runs **structurally** end-to-end (ingest → rubric → RLM loop → report) but **never produces a real experiment**. Root cause is a **bundled-CLI (`claude-agent-sdk`) transport degradation that is now reproduced in isolation**: the transport starts returning `is_error` (surfaced as `ConnectionRefused` / "Claude Code returned an error result: success") after ~33 sequential `query()` calls in one long-lived process. With `--model claude-oauth`, the RLM **root, navigation, grader, and implementer all share that one transport**, so the process blows past the degradation threshold long before `implement_baseline` runs.

None of the three success criteria were met on any attempt this session.

---

## 1. TL;DR — the causal chain

```
--model claude-oauth
   → RLM root + llm_query/rlm_query navigation + verify_against_rubric grader
     + implement_baseline  ALL use the bundled-CLI transport (claude-agent-sdk + OAuth)
   → root/navigation emit dozens–hundreds of bundled-CLI query() calls per run
   → the bundled-CLI transport leaks/degrades after ~33 calls in one process
     (observed: aclose() "asynchronous generator is already running",
      "Loop <...> that handles pid N is closed")
   → by the time the heavy implement_baseline call runs, new bundled-CLI
     connections fail: is_error=True, subtype="success"
     (ResultMessage.api_error_status set; CLI logs "ConnectionRefused")
   → claude_runtime raises "Claude Code returned an error result: success"
   → implement_baseline writes NO code, NO text  (code/ is empty)
   → no commands.json
   → run_experiment fails / never runs ("no commands.json")
   → no experiment_runs row, no run_experiment cost row
   → evidence gate (correctly) downgrades verdict to `failed`
```

This is the **FM-001 bundled-CLI / orphaned-child transport wedge** named in `CLAUDE.md` and `docs/superpowers/plans/2026-05-30-rlm-wedge-hardening-and-evolution.md`. This session **reproduced its tipping point** for the first time (Section 4).

---

## 2. Run attempts this session (timeline)

| # | Project id | Config | Result | Proximate failure |
|---|------------|--------|--------|-------------------|
| 0 | `sdar_<ts>` | runbook cmd w/ `--project-id sdar_$(date +%s)` | **instant death** | `UnknownProject` — ISSUE-1 |
| 1 | `prj_09047604e591d969` | `claude-oauth`, default `SUBRLM_MAX_CONCURRENCY=4` | full pipeline ran, `verdict=failed`, 0 experiments | `implement_baseline`: `current working directory was deleted` ×18 + `aclose()` races |
| 2 | `prj_09047604e591d969` | `claude-oauth`, `REPROLAB_SUBRLM_MAX_CONCURRENCY=1` | killed mid-run for investigation | `implement_baseline`: `[critical] SDK success-with-no-text` (5 failed workers); `build_environment` failed; 0 commands |
| — | `prj_09047604e591d969` (archived `attempts/20260529T224409`) | prior May-29 run | `verdict=partial`, score 0.0 | `implement_baseline`: **`API Error: Unable to connect to API (ConnectionRefused)`**; `run_experiment`: `no commands.json` |

Across all attempts the failure is the **same class** (bundled-CLI sub-agent transport failing on the heavy `implement_baseline` call), with **different surfaces** (`ConnectionRefused`, `cwd deleted`, `success-with-no-text`) depending on where in the connection lifecycle the degraded transport breaks.

`REPROLAB_SUBRLM_MAX_CONCURRENCY=1` **changed the surface** (eliminated `cwd deleted`) but did **not** fix the underlying degradation — because that cap only governs the *navigation* fan-out (`rlm_query.py:592`), not the root/grader/implementer share of bundled-CLI load.

---

## 3. The actual error (primary-source evidence)

From the bundled-CLI session transcripts the SDK writes to
`~/.claude/projects/-Volumes-CS-Stuff-openresearch-runs-prj-09047604e591d969-code/*.jsonl`,
**all three** failed `implement_baseline` sessions terminate with:

```
API Error: Unable to connect to API (ConnectionRefused)   (stop_reason: stop_sequence)
```

The SDK then raises (`_internal/query.py:304-343`): when a `ResultMessage` has
`is_error=True`, it raises `ProcessError("Claude Code returned an error result: <errors or subtype>")`.
The `errors` list is empty, so the message degrades to the bare subtype **`success`**.
`ResultMessage.api_error_status` (`types.py:1163-1166`: *"HTTP status … when is_error is True and subtype is 'success'"*) carries the real code — **but `claude_runtime.py` never reads it** (it only handles the non-error `ResultMessage` for usage), so the HTTP status is discarded. That is ISSUE-5.

`worker_reports.py:243-267` then pattern-matches `"Claude Code returned an error result:\s*success"` → the `[critical] SDK success-with-no-text` blocker seen in `/lab`.

`code/` is **empty** after every failed `implement_baseline` — no files, no text. The sub-agent produced nothing because the transport never connected.

---

## 4. Reproduction (Iron-Law: confirmed, not guessed)

All smoke tests run with `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY .venv/bin/python`
(same credential posture as the run; OAuth subscription, no API key), SDK `claude-agent-sdk==0.2.82`, Python `3.14.5`.

| Test | Shape | Result |
|------|-------|--------|
| Simple | 1 call, `Bash` only, tiny prompt | ✅ text returned |
| Heavy | 1 call, `claude-sonnet-4-6` + `max_thinking_tokens=4000` + Write/Bash/Read, real coding task | ✅ `is_error=False`, 4 turns, 18s, `$0.08` |
| Sequential ×10 | 10 calls, each its own `asyncio.run()` in a worker thread (mimics per-primitive bridge) | ✅ all 10 OK |
| Concurrent ×5 | 5 simultaneous `query()` children | ✅ all 5 OK |
| **Sequential ×60** | same as ×10 but driven to volume | ❌ **calls 1–33 OK, calls 34/35/36 fail** with `Claude Code returned an error result: success` |

**Conclusion:** auth, rate-limit/quota (a fresh heavy call costs $0.08 and succeeds; `api_error_status=None` on healthy calls), prompt size, sequential loop-churn, concurrency, and nested `agents=` are all **ruled out by reproduction**. The trigger is **cumulative bundled-CLI call volume** in one process: the transport leaks resources (child processes / fds / the async-generator cleanup races) and breaks at ~33 calls.

Note: `implement_baseline`'s registry spec is invoked with **no `sub_agents`** (`invoke.py` passes none) — the nested-`agents=` hypothesis is eliminated.

---

## 5. Why `claude-oauth` makes it worse

With `--model claude-oauth`, `_select_root_client` (`run.py:211-213`) returns the OAuth `ClaudeLlmClient`/`ClaudeOauthClient`, i.e. **the root model itself rides the bundled-CLI transport** — one query per REPL iteration, dozens per run — *plus* navigation (`llm_query`/`rlm_query`) *plus* the grader. So the shared transport easily crosses the ~33-call degradation threshold before `implement_baseline` is ever reached.

The `apply_oauth_backend_patch()` monkeypatch (`_oauth_backend_patch.py`) only swaps `rlm.clients.get_client` dispatch and adds a system-prompt cache wrapper that **explicitly skips the OAuth path** — it does **not** touch sockets/transport and is **not** implicated.

`docs/audits/2026-05-30-infra-optimization-audit.md` + the wedge plan already anticipated FM-001; the **gpt-5-mini navigation accelerator** (default off) was designed to move *navigation* off the bundled CLI — but it does **not** help the root (claude-oauth) or `implement_baseline`, which still use the bundled CLI.

---

## 6. All issues found (enumerated)

### ISSUE-1 — `--project-id <arbitrary>` breaks the CLI ingest path (defect)
`cli.py:1334-1346`: `register_project()` keys the event-store aggregate by the **source-derived** id (`project_id_for(source)` = `prj_<sha256>`), but the `--project-id` override (lines 1340-1341) repoints `project_id` to the arbitrary value **after** registration, so `fetch_paper(FetchPaper(project_id=<arbitrary>))` → `UnknownProject`. The override is only safe when the passed id **equals** the source-derived id (the REST-API spawn path). **The documented runbook command `--project-id sdar_$(date +%s)` is broken for direct CLI use.** Workaround used this session: omit `--project-id`, let the CLI use `prj_09047604e591d969`.
- **Fix:** either register under the override id, or reject/normalize a `--project-id` that ≠ `project_id_for(source)` with a clear error.

### ISSUE-2 — Bundled-CLI transport degradation (P0 root cause)
Section 1–5. Reproduced at ~33 calls. Blocks every real experiment.

### ISSUE-3 — `parsed_full_text.txt` missing → lossy paper fallback
The run logs `paper text degraded — proceeding with lossy workspace fallback (parsed_full_text.txt missing)`. `parsed_full_text.txt` never exists as a file for this project; the paper survives only as the offloaded workspace variable. Pre-existing (prior runs had it too). Degrades rubric/grader quality; not the blocker for this session but worth fixing for fidelity.

### ISSUE-4 — Killed run leaves `demo_status=running`
After `SIGTERM` then `SIGKILL`, `runs/prj_09047604e591d969/demo_status.json` is still `status:"running"`, `killReason:null`. The `cli.py::_install_termination_handlers` path (BUG-NEW-041) did not flip the status to `killed` on this kill (likely the SIGKILL beat the handler, or the handler did not run from the background launch context). This makes `/runs/latest` show a zombie "running" run. Should be reconciled by the orphan-liveness sweep, but verify.

### ISSUE-5 — `api_error_status` discarded (observability gap)
`claude_runtime.py:125` handles `ResultMessage` only for usage and never inspects `is_error`/`api_error_status`; the SDK's `ProcessError` keeps only the subtype (`"success"`). So the real HTTP status / `ConnectionRefused` is invisible in our logs — diagnosis required reading the SDK's own `~/.claude/projects` transcripts. **Fix:** in `claude_runtime.run_agent`, on `ResultMessage.is_error` (or on `ProcessError`), capture `api_error_status` + `ProcessError.stderr` and surface them in the worker report / `run_warning`.

### ISSUE-6 — No transport-failure retry/recycle for `implement_baseline`
`primitives.py:1803-1839`: a single SDK exception (`except Exception` at 1829) harvests whatever is on disk (nothing) and returns the error. There is **no** retry, no transport recycle, no orphan reap. One transient/accumulated transport failure kills the most expensive primitive and thus the whole run.

---

## 7. Recommended fixes (ranked)

**FIX-A (config, immediate, lowest risk) — move the root off the bundled CLI.**
Run with `--model gpt-5` (OpenAI, httpx, ~$1/run) or `--model qwen3-coder-featherless` (cheapest) for the root, keeping the OAuth subscription only for the Sonnet sub-agents. This removes the root's continuous bundled-CLI traffic, keeping cumulative call volume under the ~33 degradation threshold by the time `implement_baseline` runs. This is exactly `CLAUDE.md`'s documented "cheapest local-dev cost model." **Trades the zero-API-cost constraint for reliability** — needs operator sign-off. Testable in one run.

**FIX-B (config, complementary) — enable the navigation accelerator.**
`REPROLAB_ACCELERATOR=endpoint`, `REPROLAB_ACCELERATOR_BASE_URL=https://api.openai.com/v1`, `REPROLAB_ACCELERATOR_MODEL=gpt-5-mini`, `REPROLAB_ACCELERATOR_SCOPE=navigation`. Moves hot-volume nav calls off the bundled CLI. Stack with FIX-A for maximum load reduction. Keep grader on Sonnet-OAuth.

**FIX-C (code, durable root fix) — process-isolate the `implement_baseline` SDK call.**
Spawn each `implement_baseline` (and ideally each heavy sub-agent) SDK `query()` in a **fresh subprocess**, so it always gets a clean transport regardless of how many bundled-CLI calls the parent has made. `invoke.py` already thread-isolates (`run_isolated`); promote to **process** isolation for the code-writing surface. This fixes the root cause without changing the model/cost posture.

**FIX-D (code) — recycle transport / reap orphans on a call-count threshold.**
Track bundled-CLI call count; before the heavy implementer (or every N≈25 calls), reap `_bundled_claude_child_pids()` and recycle. Plus add a bounded retry on `is_error` for `implement_baseline`.

**FIX-E (code, observability) — surface `api_error_status` + `ProcessError.stderr`** (ISSUE-5) so future wedges are diagnosable without spelunking SDK transcripts.

**ISSUE-1 fix** — separately, repair or guard the `--project-id` override so the documented runbook command works.

**Validation:** any fix must be confirmed by a real run meeting the three criteria (`experiment_runs` success+metrics, ≥1 `run_experiment` cost row, `final_report` verdict ≠ failed) — see `docs/runbooks/2026-05-30-validation-first-sdar-ab.md`.

---

## 8. Environment (for reproducibility)

- OrbStack 2.1.1 (build 2010100); Docker `ServerVersion` 29.4.0 — `docker info` OK.
- `claude` CLI 2.1.158, OAuth subscription active (`claude --print` → `pong`); `ANTHROPIC_API_KEY` empty.
- RunPod preflight all-green (RTX 4090, COMMUNITY, image `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04`). **No pod was ever booted** this session (no `commands.json` ⇒ `run_experiment` never reached the pod ⇒ no GPU billing).
- `claude-agent-sdk==0.2.82`, Python 3.14.5.
- Repo on external volume `/Volumes/CS_Stuff` (not implicated; isolated SDK calls from the same cwd succeed).

## 9. Smoke-test artifacts (reusable, committed)

Run any with `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY .venv/bin/python scripts/diagnostics/<file>`:

- `scripts/diagnostics/sdk_smoke.py` — single simple Bash call (baseline: passes).
- `scripts/diagnostics/sdk_heavy_smoke.py` — single heavy sonnet+thinking coding call, captures full `ResultMessage` (passes; `$0.08`).
- `scripts/diagnostics/sdk_multicall_smoke.py` — **the reproduction**: N sequential `asyncio.run()` calls; degrades at call ~34. Adjust the loop bound to re-confirm.
- `scripts/diagnostics/sdk_concurrent_smoke.py` — 5 concurrent children (passes).
