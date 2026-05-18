# Unified Logging Launcher — Design

**Status:** Tier 1 implemented + verified · 2026-05-17
**Launchers:** `scripts/dev.ps1` (Windows-native, primary) · `scripts/dev.sh` (macOS / Linux / git-bash sibling). Both write the same `logs/<TS>/` layout.
**Goal:** One reviewable, verifiable folder per launch that captures frontend, backend, pipeline, agent, and sub-agent activity — without breaking existing `dev.sh` callers.
**Backend wire-up:** Tier 1 also required wiring `REPROLAB_RUNS_ROOT` through `backend/config.py` Settings → `FileLiveRunService` constructor → CLI argparse defaults. Without this, the launcher's env-var export was cosmetic (see §11).

### Why two launchers

`scripts/dev.sh` invoked from a Windows shell will silently dispatch to **WSL bash** if WSL is on PATH ahead of git-bash (Windows 11 ships `C:\WINDOWS\system32\bash.exe` by default). Under WSL, `$PWD` becomes `/mnt/c/...`, which the Windows Python inside `.venv\Scripts\python.exe` cannot resolve — so `REPROLAB_RUNS_ROOT` is silently ignored and the pipeline falls back to its default `runs/` directory. Verified on 2026-05-17: the run completed, server logs landed correctly under `server/`, but **zero `prj_*` workspaces appeared inside `logs/<TS>/`**, breaking the layout contract. `scripts/dev.ps1` avoids the issue by exporting a native Windows path and never crossing the WSL boundary. The `.sh` script remains the cross-platform sibling.

---

## 1. Constraints (from the ask + repo state)

- Windows + bash (git-bash). PowerShell helpers OK for port-free.
- Python venv at `.venv` (Windows `.venv/Scripts/python.exe`).
- `REPROLAB_RUNS_ROOT` already drops per-pipeline workspaces at `<root>/prj_<project_id>/`. Many agents read/write under it (`baseline_implementation.py:377`, `dashboard_emitter.py:62`, `improvement.py:107`, …). Do **not** reinterpret it.
- Backend agents already use `logging.getLogger(__name__)` (e.g. `orchestrator.py:88`, `baseline_implementation.py:23`) but there is **no central logging config** — handlers default to stderr → currently piped into `backend.log`.
- Sub-agent activity already streams as `StreamText` / `StreamToolCall` / `StreamUsage` events inside `backend/agents/resilience/engine.py:229–264`, but is collected in-memory and not persisted as a per-agent transcript.
- A `dashboard_events.jsonl` is already written per project (`dashboard_emitter.py:5`). Keep it; do not duplicate it.

---

## 2. Non-goals

- Building a log viewer / UI. (The output is plain files + JSONL — `rg`, `jq`, and `code logs/<TS>` are enough.)
- Centralized log aggregation (Loki/ELK). Local filesystem only.
- Replacing `dashboard_events.jsonl`.

---

## 3. Output layout

```
logs/<TS>/                              # one directory per launch
├── meta.json                           # launch metadata (existing) + ended_at, exit codes
├── manifest.json                       # NEW: index of every file written, sizes, sha256, line counts
├── server/
│   ├── backend.log                     # uvicorn combined stdout+stderr (text)
│   ├── backend.jsonl                   # NEW (Tier 2): structured records, one JSON / line
│   ├── frontend.log                    # Next.js dev combined stdout+stderr
│   └── access.log                      # OPTIONAL: uvicorn --access-log if enabled
└── prj_<project_id>/                   # per-pipeline-run workspace (existing path; unchanged contract)
    ├── pipeline.log                    # NEW (Tier 2): orchestrator stage-level text log
    ├── pipeline.jsonl                  # NEW (Tier 2): structured stage events {stage, status, ts, …}
    ├── dashboard_events.jsonl          # EXISTING
    ├── raw_paper.pdf / baseline_result.json / code/ / baseline/ / improvements/   # EXISTING artifacts
    └── agents/                         # NEW (Tier 2): per-agent-invocation transcripts
        └── <seq>-<agent_id>/           # seq=01,02,… so re-invocations of same agent are distinct
            ├── prompt.md               # exact user_input passed to runtime.run_agent
            ├── trace.log               # concatenated StreamText events (the model's narration)
            ├── tool_calls.jsonl        # one StreamToolCall per line: {ts, tool_name, input, output_preview}
            ├── usage.json              # final StreamUsage snapshot
            ├── result.txt              # final consolidated output passed back to orchestrator
            └── meta.json               # {agent_id, parent_stage, started_at, ended_at, status, msg_count, retries}
```

### Why this shape

- **Top-level by run, not by stream**: a reviewer opens one `logs/<TS>/` folder and has everything correlated by time.
- **`prj_*` stays where the backend expects it.** No code path reading `REPROLAB_RUNS_ROOT` needs to change.
- **One folder per agent invocation, not per agent name.** `supervisor-verifier` runs at three different gates; `01-supervisor-verifier`, `02-supervisor-verifier`, `03-supervisor-verifier` keeps them disjoint.
- **JSONL for machine-consumed, .log for human-consumed.** Don't try to make one file serve both.
- **`manifest.json` is the verification artifact**: if a reviewer needs to know "did agent X actually complete in run Y", they read manifest, not the tree.

---

## 4. Tiered rollout

The script lands Tier 1 today with zero backend changes. Tiers 2/3 are follow-ups that build on the same folder layout.

### Tier 1 — Launcher script only (no backend changes)

`scripts/dev.sh` is **extended in place** (additive). Concretely:

1. Create `logs/<TS>/server/` and write the uvicorn / Next.js streams there as `backend.log` / `frontend.log`. (Repo sweep confirmed no callers of the old top-level paths — see §5.)
2. Keep `REPROLAB_RUNS_ROOT=$PWD/logs/<TS>` (unchanged). Pipeline still writes `prj_*` siblings to `server/`.
3. On `cleanup`, walk the tree and write `manifest.json` with `{path, size, sha256, lines}` per file. ≤200 lines of shell or a tiny Python helper invoked via `$PY_BIN`.
4. `meta.json` gains: `exit_code_backend`, `exit_code_frontend`, `ended_reason` (`exit`/`int`/`term`).

Deliverable: one updated file (`scripts/dev.sh`) + one new helper (`scripts/_write_manifest.py`).

### Tier 2 — Backend observability hook (per-agent transcripts)

Add `backend/observability/run_logging.py`:

- `configure(run_dir: Path)` — installs a root `logging.FileHandler` writing to `<run_dir>/pipeline.log` and a JSON-formatted handler writing `pipeline.jsonl`. Called once from `backend/app.py` startup and from `backend/cli.py` `main()`.
- `AgentTranscriptRecorder` — context-managed in `_invoke_agent` (`orchestrator.py:506`). Wraps the existing `async for event in runtime.run_agent(...)` loop in `resilience/engine.py:229` to fan events out to `agents/<seq>-<agent_id>/`:
  - `StreamText` → append to `trace.log`
  - `StreamToolCall` → append one line to `tool_calls.jsonl`
  - `StreamUsage` → overwrite `usage.json`
- Recorder is **only active when `REPROLAB_LOG_DIR` is set** (the launcher sets it to `logs/<TS>`). Production / non-dev runs are unaffected.

Touches: ~3 backend files (one new module, two small edits). No behavior change when `REPROLAB_LOG_DIR` is unset.

### Tier 3 — `verify` subcommand

`scripts/verify_run.py <logs/TS>` walks `manifest.json` and prints a checklist:

- Every `pipeline.jsonl` `stage_start` has a matching `stage_end`.
- Every agent invocation in `pipeline.jsonl` has a populated `agents/<seq>-<id>/` folder with non-empty `result.txt`.
- No zero-byte files; no UTF-8 decode errors.
- Print a pass/fail summary and exit non-zero on gaps. Suitable for CI smoke.

---

## 5. Naming & callers — RESOLVED

**Repo sweep (2026-05-17):** grepped the entire tree for `backend.log` / `frontend.log` / `logs/<TS>` patterns across `*.py`, `*.sh`, `*.ts`, `*.tsx`, `*.js`, `*.json`, `*.md`, `*.yml`, `Dockerfile*`. The only matches were inside this design doc itself. The other `logs/run.log` / `logs/paperbench_eval.log` references (`backend/agents/pipeline.py:261`, `backend/agents/prompts/experiment_runner.py:22,32`, `backend/services/events/live_runs.py:649,961,1018`) live inside per-project workspaces under `prj_*/baseline/` — a different convention, not the launcher's top-level files.

**Decision:** move `backend.log` / `frontend.log` directly under `server/`. No parallel-write back-compat hack, no symlinks. One source of truth from day one.

---

## 6. Script invocation contract

`scripts/dev.sh` keeps its existing zero-arg invocation. New optional flags (all default off):

| Flag | Effect |
|---|---|
| `--no-frontend` | Skip Next.js dev (backend + pipeline only) |
| `--no-backend` | Skip uvicorn (pipeline CLI only) |
| `--label <name>` | Append `-<name>` to the timestamp folder so `logs/20260517-103211-rebuttal/` is easy to find |
| `--keep N` | After launch, prune `logs/` to the N most-recent dirs (default: keep all) |

These are **additive**. Existing `bash scripts/dev.sh` keeps working.

---

## 7. Failure modes & how the design handles them

| Failure | Handling |
|---|---|
| Backend crashes mid-stage | `cleanup` trap writes `meta.json` with `exit_code_backend != 0`; partial `pipeline.jsonl` is fine because it's append-only. |
| Frontend OOMs / hangs | Same; backend keeps running. |
| Disk fills up | `manifest.json` write is best-effort (`set +e` around it); meta.json still gets `ended_at`. |
| Re-running `dev.sh` while previous still running | Existing `free_port` logic handles it; new launch gets a fresh `<TS>` dir, no collision. |
| `prj_*` written but pipeline.jsonl never closes (orphan run) | `verify` subcommand (Tier 3) flags it; for now, look at the last `dashboard_events.jsonl` line. |

---

## 8. What this design **does not** do

- Doesn't change how `REPROLAB_RUNS_ROOT` is interpreted by agents.
- Doesn't add a new "log everything" handler that could leak prompts/secrets to disk in production — recorder is opt-in via `REPROLAB_LOG_DIR`.
- Doesn't tail/forward logs to any remote service.
- Doesn't gzip or rotate. Re-launches make new folders; old folders are the user's to delete.

---

## 9. Resolved decisions (2026-05-17)

1. **Top-level log callers** — none found in the repo (see §5). Files move straight under `server/`.
2. **Redaction** — store raw. This is dev-only, runs root is already in `.gitignore`.
3. **Tier scope** — Ship Tier 1 first, manual test, open PR, then start Tier 2 as a separate PR.

---

## 10. Validation checklist (run before merging Tier 1)

- [ ] `.\scripts\dev.ps1` (or `bash scripts/dev.sh` on Unix) produces a runnable backend + frontend.
- [ ] `logs/<TS>/meta.json` exists with `started_at` and `ended_at` after a clean Ctrl-C.
- [ ] `logs/<TS>/meta.json` `runs_root` is a Windows-native path (no `/mnt/c/...`).
- [ ] `logs/<TS>/manifest.json` lists every non-empty file with size + sha256.
- [ ] A full pipeline run from the lab UI produces a `prj_<id>/` sibling under `logs/<TS>/` (**not** under `./runs/`).
- [ ] After restart, a stale `localStorage["reprolab:lastRun"]` pointing at a project from the previous launch results in `GET /api/demo?projectId=...` returning 404 and the upload view rendering.

---

## 11. Backend wire-up (discovered during verification, 2026-05-17)

The launcher script alone was not sufficient to honor the `logs/<TS>/` layout. During end-to-end testing, repeated pipeline runs kept landing under `./runs/` despite `dev.ps1` correctly exporting `REPROLAB_RUNS_ROOT` to a Windows-native absolute path. Investigation traced the cause to three missing reads of the env var at the backend layer:

| File | Symptom |
|---|---|
| `backend/services/events/live_runs.py:169` | `self.runs_root = (runs_root or self.repo_root / "runs").resolve()` — constructor takes an arg or falls back to `repo_root/runs`. No `os.environ.get` anywhere. |
| `backend/app.py:42` | `FileLiveRunService()` constructed with no `runs_root` argument, so it always fell through to the `repo_root/runs` default above. |
| `backend/cli.py:374, 633` | `_REPRODUCE_DEFAULTS["runs_root"] = "runs"` and the argparse `--runs-root` default hard-coded to `"runs"`. CLI invocations ignored env entirely. |

**Fix (one feature, three files):**

1. `backend/config.py` — added a `runs_root: Path | None = None` field on `Settings`. Because the existing `model_config` sets `env_prefix="REPROLAB_"`, this field is automatically populated from `REPROLAB_RUNS_ROOT` whenever the env var is set. No new env-var parsing code; pydantic-settings already does the work.
2. `backend/app.py` — `FileLiveRunService(runs_root=settings.runs_root)` at startup. `None` (the default) preserves the previous behavior; a set value flows through.
3. `backend/cli.py` — both the argparse `--runs-root` default and the `_REPRODUCE_DEFAULTS` dict pull from `get_settings().runs_root`, identical pattern to `--database-url` next door.

**Why this matters for the design:** the launcher's "co-locate everything under `logs/<TS>/`" promise is the headline feature of Tier 1. Without these three wire-up lines, that promise was satisfied for the *server* streams (uvicorn + Next.js stdout) but silently failed for the *pipeline* workspaces. The repeat appearance of stale `prj_*` directories under `./runs/` and the corresponding localStorage hijack of `/lab` were the user-visible signal. Documenting it here so the PR description names both the launcher and the wire-up as part of one feature, not two unrelated changes.

**Knock-on benefit:** once `REPROLAB_RUNS_ROOT` is set, the backend's `runs_root.glob("*/demo_status.json")` no longer sees pre-existing projects under `./runs/`. That means `clearLastRun()` in `frontend/src/hooks/use-run.ts:163` fires automatically on the next `/lab` load — the localStorage self-cleans without DevTools intervention.
