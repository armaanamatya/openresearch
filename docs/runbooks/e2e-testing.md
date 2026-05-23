# End-to-end testing runbook — UI + backend on origin/main

_Last updated: 2026-05-23. Session that produced this doc: F1–F6 deployment-blocker fixes._

This doc walks through the full UI-enabled E2E test surface and tells you exactly where to look when something breaks. Use it as a checklist when validating a fresh deploy, after a non-trivial backend change, or when you're trying to triage a failing run.

---

## TL;DR — fastest happy path

```bash
# Terminal 1 — backend
cd /home/abheekp/openresearch
.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000

# Terminal 2 — frontend
cd /home/abheekp/openresearch/frontend
export REPROLAB_BACKEND_URL=http://127.0.0.1:8000
npm run dev
```

Then open http://localhost:3000/lab → pick a paper → pick mode → start. The most "should-just-work" path is **arXiv URL `https://arxiv.org/abs/2512.24601` + mode `RLM (Hybrid)` + sandbox `local`**.

If you hit "Queued indefinitely" or a 500 in the console, jump to **§4 — Diagnostic playbook**.

---

## 1. What's currently on `origin/main`

These are the commits relevant to UI-driven runs (newest first). Each is a clean Opus-reviewed, Sonnet-implemented fix.

| SHA | Subject | Why it matters |
|---|---|---|
| `65ee5a2` | `fix(rlm): resolve_root_model aliases (sonnet/opus/claude-sonnet-4-6 → registry keys)` | **F4.** Lab UI sends `model="sonnet"` → backend translates to `"claude-sonnet-4-6"` → RLM's `resolve_root_model` only knew registry keys (`claude`, `claude-oauth`, …). Now aliases bridge all three vocabularies. |
| `9da8646` | `feat(paths): cross-platform input path normalization (Windows ↔ WSL ↔ macOS)` | **F3.** Pasting `C:\Users\Foo\paper.pdf` on a WSL backend now resolves to `/mnt/c/Users/Foo/paper.pdf`. arXiv IDs/URLs/DOIs pass through unchanged. |
| `45e60df` | `fix(frontend): quiet useRdrArtifacts polling + null-render RubricBreakdown` | **F2.** The polling hook stops after 3 consecutive all-404 cycles (~15s) for non-RDR runs. RubricBreakdown null-renders when no data. Zero DOM noise for PDF/arXiv runs. |
| `8d534d8` | `fix(hybrid): no-bundle paper_id falls back to pure RLM (deployment unblock)` | **F1.** Hybrid controller used to call `run_pipeline_rdr` unconditionally; RDR needed a PaperBench bundle dir; PDF/arXiv uploads have `project_id=prj_*` and no bundle → `FileNotFoundError`, run dies in 5s. Now: no bundle → fall back to pure RLM (which generates its own rubric). |
| _in flight at write-time_ | F5 — sandbox auto-detect WSL → prefer local | When `sandbox="auto"` on a WSL backend without reachable Docker daemon, resolve to `local`. Explicit `--sandbox docker` still honored. |
| _in flight at write-time_ | F6 — cross-mount Claude OAuth detection on WSL | If `claude login` was run from Windows, the WSL backend now scans `/mnt/c/Users/*/.claude/.credentials.json` and surfaces a symlink hint. |

Run `git log --oneline origin/main -10` to see current state.

---

## 2. Stack startup — full checklist

### 2a. Prereqs

- Python venv exists: `ls .venv/bin/uvicorn`
- Node ≥ 20.19 or ≥ 22.12: `node --version`
- Frontend deps installed: `cd frontend && ls node_modules/next 2>/dev/null` (re-run `npm ci` if missing)
- `.env` at repo root has _at least_ one of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `FEATHERLESS_API_KEY` — OR `claude login` has been run (OAuth subscription)

### 2b. Auth sanity check

```bash
# Inside WSL — does the claude CLI see a logged-in session?
claude --version                           # CLI exists
ls ~/.claude/.credentials.json             # OAuth credentials in WSL home

# If the credentials file is missing AND you're on WSL, F6 will scan
# /mnt/c/Users/*/.claude/.credentials.json and log a symlink hint. To
# manually adopt Windows-side credentials:
ln -s /mnt/c/Users/<yourwindowsuser>/.claude/.credentials.json ~/.claude/.credentials.json
```

### 2c. Backend

```bash
cd /home/abheekp/openresearch
.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000
```

Expect to see `Uvicorn running on http://127.0.0.1:8000`. `--reload` auto-picks up Python file changes; on WSL, edits made from Windows may not trigger inotify — **restart the server manually if you edited via Windows tools**.

### 2d. Frontend

```bash
cd /home/abheekp/openresearch/frontend
export REPROLAB_BACKEND_URL=http://127.0.0.1:8000
npm run dev
```

Wait for `Ready in N ms` then `http://localhost:3000`. Browser opens at `/lab`.

### 2e. Sanity-curl the new RDR endpoints (optional)

Once backend is up, even before a run:
```bash
curl -s http://localhost:8000/health  # 200 OK
# These return 404 for projects that don't exist — that's correct, not a bug:
curl -s http://localhost:8000/runs/nonexistent/clusters         # 404
curl -s http://localhost:8000/runs/nonexistent/leaf-scores      # 404
curl -s http://localhost:8000/runs/nonexistent/repair-iterations # 404
```

---

## 3. Test scenarios — what to click, what to expect

For each scenario: action → expected backend behavior → expected UI behavior → expected artifacts.

### 3a. PDF upload (RLM / Hybrid, default mode)

**Action:**
1. Drop a PDF on the upload-view (e.g., `arxiv_2512.24601.pdf`).
2. Mode = `RLM (Hybrid)` (default).
3. Model = `Sonnet` (default).
4. Sandbox = `auto` (default).
5. Click start.

**Expected backend path (post-F1):**
- `POST /api/demo` → `POST /runs/upload` → ingest pipeline (`prj_*` project_id) → spawn run subprocess.
- Subprocess: `cmd_reproduce` → ingest 6 stages → `run_pipeline_hybrid` → **bundle-presence guard fires** (`prj_*` has no bundle) → falls back to `run_pipeline_rlm`.
- Pure RLM: `_build_llm_client` (post-005e3b6) routes per `root_model.rlm_backend`. With `--model sonnet` → F4 aliases → `claude-oauth` registry entry → `ClaudeOauthClient`. `rubric_gen.py` generates a rubric (no bundle to load one from).

**Expected UI behavior:**
- Status: `queued` → `running` within ~5s.
- "primitive call history" populates progressively.
- "rubric score" bar populates after first scoring iteration (~10–15 min).
- RubricBreakdown panel renders ONLY if RDR cluster artifacts arrive (won't, since this is pure RLM) — F2 makes the hook stop polling after ~15s of all-404, so console stays clean.
- Final state: `completed` or `partial` after ~20-30 min.

**Expected artifacts in `runs/prj_*/`:**
- `demo_status.json`, `parsed_full_text.txt`, `raw_paper.pdf`, `code/` directory with generated code
- `final_report.json` with `rubric.overall_score`, `leaf_scores`, baseline_metrics
- `dashboard_events.jsonl` with `primitive_call` events
- `runner.stderr.log` should be empty or only have informational lines

### 3b. arXiv URL (same flow)

**Action:** paste `https://arxiv.org/abs/2512.24601` (or `2512.24601`) into the URL field. Pick mode. Start.

**Expected:** same as 3a but ingestion fetches HTML + PDF from arxiv.org instead of reading a local upload. Slightly slower due to network. Should also succeed.

### 3c. Bundle paper (genuine hybrid path)

**Action (CLI only — UI doesn't yet expose bundle source):**
```bash
.venv/bin/python -m backend.cli reproduce sequential-neural-score-estimation \
    --mode rlm --model claude-oauth --sandbox local --max-wall-clock 1500 \
    --project-id pb_test_$(date +%s)
```

**Expected backend path:**
- CLI sees `sequential-neural-score-estimation` is a bundle dir → `_cmd_reproduce_rlm_paperbench` → `run_pipeline_hybrid` → **bundle dir EXISTS** → Phase 1 RDR runs → cluster checkpoints saved → if any leaf < 0.6, Phase 2 RLM repair runs.

**Expected artifacts:**
- `runs/<project>/iterations/cluster_*.json` (27 of them for seqnn)
- `runs/<project>/iterations/repair_*.json` (only if Phase 2 ran)
- Plus everything from 3a.

**To watch from UI:** browse to `http://localhost:3000/lab?projectId=<your_project_id>`. RubricBreakdown should populate with the 27-cell cluster grid + leaf scores when scoring completes.

### 3d. RDR-only (escape hatch)

**Action:** CLI `--mode rdr` on a bundle paper_id.

**Expected:** Phase 1 RDR only, no Phase 2 RLM repair. Predictable cost ceiling. Same artifacts shape as 3c minus Phase 2.

### 3e. RLM-pure (the second escape hatch, CLI-only)

**Action:** CLI `--mode rlm-pure` — pure recursive language model, no rubric decomposition. For research / cost-controlled use.

**Expected:** No Phase 1 RDR. Pure RLM loop. Same artifacts shape as 3a.

---

## 4. Diagnostic playbook — when something breaks

### 4a. Run shows "Queued, 0s elapsed" indefinitely

**First diagnostic:**
```bash
cat runs/<project_id>/demo_status.json | python3 -m json.tool
```
- `status: "failed"` + `error: <something>` → run died at startup. Read the error.
- `status: "running"` → run is going; UI just hasn't refreshed (force-refresh the page).
- `status: "queued"` AND `runner.stderr.log` is non-empty → run crashed mid-spawn. Read the stderr.

**Read the stderr:**
```bash
cat runs/<project_id>/runner.stderr.log
```

Common failure modes + remedies:

| Failure | Fix |
|---|---|
| `ValueError: Unknown root model 'X'` | F4 should have fixed this. If you still see it, you're on stale code — `git pull origin main` + restart uvicorn. |
| `FileNotFoundError: PaperBench bundle not found` | F1 should have fixed this. Same stale-code remedy. |
| `RuntimeError: Pipeline exited with status 3` | The pipeline returned a fail status. Look HIGHER in the stderr for the real exception that produced status=3. |
| `Error code: 401 - Incorrect API key provided` | The provider routing picked OpenAI but the key in `.env` is invalid. Either replace the key or remove it (the `_effective_provider` from Track D should prefer Anthropic OAuth when both are configured, but only on the rdr path). For pure RLM, the resolved root model is determined by `--model` or env. |
| `Sandbox preflight failed` | Docker not reachable. Use `--sandbox local` explicitly OR (on WSL) F5 should have auto-degraded sandbox=auto → local. If F5 hasn't landed yet, set `REPROLAB_DEFAULT_SANDBOX=local` in `.env`. |

### 4b. Frontend: 404 spam on `/api/demo/runs/<id>/leaf-scores`

**Post-F2 expectation:** ~9 fetches total over ~15s (3 endpoints × 3 cycles), then stops. If you're seeing more than that → either F2 isn't deployed (check the bundle ID — `npm run dev` should HMR the hook) or the run actually has rdr artifacts arriving intermittently.

To check the hook state from devtools:
```javascript
// In the lab page console:
// (won't work — useRdrArtifacts is React-internal — but you can check the network tab
// to confirm fetches stopped after 3 cycles.)
```

### 4c. Frontend: 400 on `/api/demo`

The initial `POST /api/demo` (run-start) — usually only fails if the demo gate is configured AND the secret doesn't match. Check `REPROLAB_DEMO_SECRET` env var on backend; if set, the frontend must send `X-Demo-Secret` header. For local dev, leave unset.

### 4d. Lab UI shows stale state after a run failed

The lab UI uses SSE for incremental updates + polling fallback. If the run crashed before the SSE stream was established, the UI shows the last good snapshot (often "queued"). Solution: refresh the page — the GET endpoint will fetch demo_status.json fresh.

### 4e. OAuth: "no anthropic credentials" but you ran `claude login`

You ran `claude login` from Windows, not WSL. Check:
```bash
ls ~/.claude/.credentials.json                       # WSL home — what the SDK reads
ls /mnt/c/Users/$USER/.claude/.credentials.json 2>/dev/null  # Windows home (may differ)
```

If credentials are in `/mnt/c/Users/...` only, F6 (when landed) will log this with a one-line symlink hint. Manual fix:
```bash
ln -s /mnt/c/Users/<your-windows-username>/.claude/.credentials.json ~/.claude/.credentials.json
```

### 4f. Orphan `claude` SDK subprocesses lingering

If you killed a backend mid-run, the SDK's bundled `claude` Node binary often becomes orphaned (PPID=1 after parent dies). Find + kill:
```bash
ps -eo pid,etime,cmd | grep "claude_agent_sdk/_bundled/claude" | grep -v grep
pkill -9 -f "claude_agent_sdk/_bundled/claude"
```

They consume OAuth quota silently — kill them between test runs if you've been iterating.

---

## 5. Logs and where they live

| What | Where |
|---|---|
| Backend uvicorn output | Foreground in your terminal (or `tee` to a file) |
| Frontend Next.js output | Foreground in your other terminal |
| Per-run subprocess stdout | `runs/<id>/runner.stdout.log` (small JSON summary at exit) |
| Per-run subprocess stderr | `runs/<id>/runner.stderr.log` (the failure trace if any) |
| Demo state snapshot | `runs/<id>/demo_status.json` (atomic write — never partial) |
| Pipeline state checkpoint | `runs/<id>/pipeline_state.json` (rdr/hybrid) |
| Live event stream (SSE source) | `runs/<id>/dashboard_events.jsonl` |
| RDR per-cluster checkpoints | `runs/<id>/iterations/cluster_*.json`, `repair_*.json` |
| Final report | `runs/<id>/final_report.{json,md}` |
| Generated code | `runs/<id>/code/` |
| Cost ledger (per-call) | `runs/<id>/cost_ledger.jsonl` |

For each cluster's failure (after F1 + Track D2), the cluster's `error` field is now persisted in the JSON checkpoint — no log archaeology needed.

---

## 6. Quick tests — covered by pytest

Run these to confirm the merged fixes work locally without touching the live stack:

```bash
cd /home/abheekp/openresearch

# F1 — no-bundle hybrid fallback
.venv/bin/python -m pytest tests/rlm/test_hybrid_controller.py -v -k "no_bundle"

# F3 — path normalization
.venv/bin/python -m pytest tests/services/test_paths.py -v
.venv/bin/python -m pytest tests/test_cli_normalizes_source_path.py -v

# F4 — model aliases
.venv/bin/python -m pytest tests/rlm/test_model_aliases.py -v

# F5 (when landed) — WSL sandbox detection
.venv/bin/python -m pytest tests/agents/test_sandbox_wsl_detect.py -v

# F6 (when landed) — WSL OAuth fallback
.venv/bin/python -m pytest tests/services/test_runtime_oauth_wsl.py -v

# Full RLM + RDR suite (current baseline: 579 passed, 1 xfailed)
.venv/bin/python -m pytest tests/rlm/ tests/rdr/ -q

# Frontend (current baseline: 101 passed)
cd frontend && npm test
```

---

## 7. Known limitations / not-fixed-in-this-pass

These are deferred follow-ups, not regressions:

- **ML paper variety:** the primitives (`extract_hyperparameters`, `implement_baseline`, etc.) are ML-training-shaped. Theory papers / RL papers with non-standard rubric structures may produce useless code. The `rubric_gen.py` generated rubric tries to be paper-agnostic but inherits PaperBench's shape.
- **Cost tracking on OAuth runs:** `ClaudeOauthClient.get_usage_summary` returns zero tokens (the SDK doesn't surface per-call usage). So `max_usd` budget enforcement is bypassed for pure-OAuth runs. Use `max_wall_clock` instead, OR set an Anthropic API key and `--model claude` for real cost tracking.
- **Frontend live cluster grid:** the RubricBreakdown shows cluster pills as `pending/running/success/failed` based on the polled JSON state, not on live SSE events. Phase 1 progress feels chunky on slow connections.
- **Multi-user WSL distros:** F6 scans `/mnt/c/Users/*` but picks the first match. If you have multiple Windows users with credentials, you'll get the first directory listed.

---

## 8. Where to file/triage issues

| Issue class | Where to look first |
|---|---|
| Backend Python error | `runs/<id>/runner.stderr.log` |
| UI shows wrong state | Browser devtools network tab → look at `/api/demo/runs/<id>` payload |
| Stuck "Queued" | `runs/<id>/demo_status.json` — if status="failed", read `error` field |
| 404 spam in console | Confirm F2 deployed (`git log --oneline | grep 633f2a5\|45e60df`); should stop after 15s |
| Auth errors mid-run | Check `cost_ledger.jsonl` for which provider was actually used per call |
| Hybrid not dispatching to RDR for a bundle | Verify `third_party/paperbench/<paper_id>/rubric.json` exists |
| SDK aclose timeouts | Already mitigated by Workaround B (`_run_sdk_in_thread` + `shutdown(wait=False)`) — see `learn.md` 2026-05-22 |

---

_End of runbook. Update this doc when adding new modes, new endpoints, or new failure-discovery patterns. Add new "common failure → fix" rows to §4 — those are the highest-leverage entries for future maintainers._
