# Attempt 3 handoff — 2026-05-28

**Branch:** `localqwen`  
**Paper:** SDAR arXiv 2605.15155  
**Project ID:** `prj_09047604e591d969` (same paper hash — expected, archive of attempt 2 will be created on next launch)

Pick this up and run attempt 3. Everything is pre-configured. Read the **Blockers resolved** section, verify the **Pre-flight checklist**, then run the launch command.

---

## What changed since the last working state

### Code changes (committed to branch, already in place)

| File | Change |
|------|--------|
| `backend/services/context/workspace/tools/openai_client.py` | Added `extra_body: dict \| None` param, forwarded to `chat.completions.create` — lets Ollama receive `num_ctx` |
| `backend/agents/rlm/run.py` `_build_llm_client` | Detects `:11434` in base_url → injects `extra_body={"options":{"num_ctx":32768}}` for all root-model Ollama calls |
| `backend/agents/rlm/models.py` local-qwen entry | `sub_backend_kwargs["base_url"]` now reads only from `REPROLAB_ANTHROPIC_PROXY_URL` env var (or `None`). Previously fell back to `ANTHROPIC_BASE_URL`/hardcoded default, which forced sub-agents through the broken LiteLLM→Qwen proxy |
| `.env` | `ANTHROPIC_API_KEY=` (empty), `ANTHROPIC_BASE_URL=` (empty), `REPROLAB_ANTHROPIC_PROXY_URL=` (empty) |

### Why these changes matter

**Rubric gen fix:** Ollama's default context is ~2048 tokens. Rubric gen sends 48k chars (~12k tokens) of paper. Without `num_ctx=32768`, Ollama truncates input silently → malformed JSON → "unparseable JSON" × 3 → run proceeds rubric-less.

**Sub-agent fix:** qwen3.5:9b via Ollama→LiteLLM returns prose instead of Anthropic `tool_use` blocks. The claude-agent-sdk terminates after 1 round-trip (sees "no tool calls = done"), writing zero files. Sub-agents now route to **claude-oauth** (your Claude subscription, $0 cost). Root model remains qwen3.5:9b on Ollama.

---

## Architecture for attempt 3

```
Root model:   Ollama qwen3.5:9b @ :11434   (local, $0)
              ↓ via OpenAI-compat API
Sub-agents:   claude-agent-sdk → claude-oauth  ($0, uses claude CLI subscription)
              ↓
GPU sandbox:  RunPod COMMUNITY RTX 4090  (~$0.34–$0.86/hr, starts at run_experiment)
```

---

## Pre-flight checklist

Run these before launching:

```powershell
# 1. Ollama loaded with qwen3.5:9b
ollama ps
# Expect: qwen3.5:9b in the list. If not: ollama run qwen3.5:9b --keepalive 2h

# 2. LiteLLM proxy alive (needed for root model sub-calls via rlm depth-1, even if sub-agents bypass it)
Invoke-RestMethod -Uri "http://localhost:4000/health/liveness" -TimeoutSec 5
# Expect: "I'm alive!"
# If dead: cd scripts/localqwen && docker compose up -d

# 3. claude-oauth active
claude --print "ping" 2>&1
# Expect: "ping" or similar. If "not logged in": claude login

# 4. Backend up (for UI)
Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction SilentlyContinue
# If dead: start it (see below)

# 5. Frontend up (for UI, optional)
# http://localhost:3002 — if dead, start it (see below)
```

---

## Launch commands

```powershell
# Set working dir
cd C:\Users\Armaan\Desktop\openresearch

# Start backend if not running (check first)
$backendLog = "scripts\localqwen\backend.log"
$env:OLLAMA_API_KEY = "ollama"
$be = Start-Process -FilePath ".venv\Scripts\python.exe" `
  -ArgumentList @("-m","uvicorn","backend.app:create_app","--factory","--host","127.0.0.1","--port","8000") `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput $backendLog -RedirectStandardError "$backendLog.err" `
  -PassThru -WindowStyle Hidden
"backend pid: $($be.Id)"

# Start frontend if not running (optional, for UI)
$feLog = "scripts\localqwen\frontend.log"
$fe = Start-Process -FilePath "cmd.exe" -ArgumentList "/c npm run dev" `
  -WorkingDirectory "frontend" `
  -RedirectStandardOutput $feLog -RedirectStandardError "$feLog.err" `
  -PassThru -WindowStyle Hidden
"frontend pid: $($fe.Id)"

# Launch attempt 3 — note: NO ANTHROPIC_BASE_URL, NO ANTHROPIC_API_KEY
# Sub-agents use claude-oauth automatically
$env:OLLAMA_API_KEY = "ollama"
$env:REPROLAB_BASELINE_EXTRA_GUIDANCE = "Scope: implement SDAR for the smallest two models only — Qwen3-1.7B and Qwen2.5-3B. Skip the 7B run. Use a single 24-48GB GPU. Train one short ALFWorld episode set; the rubric only requires that the SDAR algorithm invariants (g_t = sigmoid(beta * delta), stop-gradient on gate, lambda=0.1, beta=10) are visibly present and that real Qwen weights + real ALFWorld data are used."

$runLog = "scripts\localqwen\sdar_run3.log"
$run = Start-Process -FilePath ".venv\Scripts\python.exe" `
  -ArgumentList @("-m","backend.cli","reproduce","2605.15155",
    "--model","local-qwen",
    "--sandbox","runpod",
    "--max-run-gpu-usd","5.0",
    "--max-pod-seconds","7200",
    "--max-wall-clock","5400") `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput $runLog -RedirectStandardError "$runLog.err" `
  -PassThru -WindowStyle Hidden
"run pid: $($run.Id)"
"log: $runLog"
```

**Lab UI:** http://localhost:3002/lab?projectId=prj_09047604e591d969

---

## What to watch for

### Rubric gen (first ~2 min after start)
```
# Good (new with num_ctx fix):
run_pipeline_rlm: using a self-generated rubric (persisted to generated_rubric.json)

# Still failing (shouldn't happen but check):
generate_rubric_tree: unparseable JSON on attempt 1
```
If still failing: check `ollama ps` for CONTEXT value; it should show 32768 after first root call.

### implement_baseline (~10–15 min in)
```
# Good (claude-oauth working):
worker_report_started ... agent_id: implement_baseline
# Then: multiple POST /v1/messages calls in claude-agent-sdk
# Then: .py files appear in runs/prj_.../code/

# Bad (still broken):
"artifacts incomplete: missing runnable source file"  # Files: [] again
```
If still failing: check `claude --print "ping"` — oauth may have expired.

### run_experiment (after implement_baseline succeeds)
RunPod pod will spin up here. Costs begin (~$0.34–$0.86/hr). Monitor:
```powershell
Get-Content runs\prj_09047604e591d969\dashboard_events.jsonl -Tail 5
# Look for: primitive: run_experiment, status: start
```

---

## Monitor loop (2-min ticks)

Use ScheduleWakeup with this prompt:

```
Monitor SDAR attempt 3, project prj_09047604e591d969. Each tick:
1. Check run pid alive
2. Tail sdar_run3.log.err (20 lines)
3. Count+tail dashboard_events.jsonl (last 5)
4. Read demo_status.json (status, iter_count, cost_summary)
5. Check code dir for .py files: Get-ChildItem runs/prj_09047604e591d969/code -Recurse
6. Screenshot lab UI to scripts/localqwen/screens/sdar3-tick-N.png
7. Append tick entry to docs/runbooks/2026-05-28-sdar-localqwen-run-log.md
8. If implement_baseline still fails with "missing runnable source file": check oauth, document in runbook
9. Reschedule every 120s until status=completed/error
```

---

## Known remaining risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Ollama unloads qwen3.5:9b mid-run (5-min TTL) | Medium | Root model calls are continuous; TTL resets each call |
| RunPod no COMMUNITY capacity for RTX 4090 | Low | Run waits and retries; budget cap protects spend |
| claude-oauth token expired | Low | Run `claude login` to refresh before launch |
| SDAR needs CUDA compile (flash-attn, deepspeed) | Medium | devel image is set; EXTRA_GUIDANCE pins to 1.7B+3B to minimize deps |
| qwen3.5:9b root model hallucinates wrong primitive signature again | Medium | Self-corrects within 1-2 iters; acceptable |

---

## Key files

- `docs/runbooks/2026-05-28-session-summary.md` — full session history and root-cause analysis
- `docs/runbooks/2026-05-28-sdar-localqwen-run-log.md` — per-attempt per-tick log
- `scripts/localqwen/litellm_config.yaml` — LiteLLM proxy config (num_ctx=16384, think=false)
- `scripts/localqwen/screens/` — UI screenshots per tick
