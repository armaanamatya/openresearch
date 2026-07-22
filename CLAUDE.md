<!-- doc-meta: status=current; last-verified=2026-07-07 -->
# CLAUDE.md

> **Tier-2 day-to-day reference.** Read `docs/architecture.md` before
> non-trivial architecture changes.
>
> **This root is deliberately lean.** Subsystem detail lives in nested `CLAUDE.md` files that
> load automatically when you work in that subtree — `backend/agents/rlm/` (orchestrator,
> primitives, flags, auth), `backend/services/runtime/` (sandboxes, GPU, cloud), `frontend/`
> (UI/SSE), `tests/`. **Don't grow this file** — put detail in the nested one and link it.

## What OpenResearch is — and where it's going

An agent that **reproduces research papers end-to-end**: ingest a paper → offload it as a REPL
variable → an RLM root model writes Python to understand the claims, build an environment,
implement and run a baseline, score it against an auto-generated rubric, and explore improvements
→ emit `final_report.{json,md}`.

- **Today — autonomous reproduction.** A bare arXiv ID or PDF runs the full pipeline unattended;
  the `campaign` loop repeats until reproduced/exhausted. This is the reproduction engine behind
  **deepinvent.ai** (the product surface).
- **Next — experiment ideation.** The same evidence-grounded harness is the substrate for a
  research-ideation layer: propose and test *new* experiments, not only replicate existing ones.

**North-star invariant:** the fitness signal is the **deterministic evidence layer, never the LLM
grade** — every trust and self-improvement mechanism keys on measured on-disk artifacts. Keep it
that way; it is the red line the whole harness is built around.

## Quickstart

```bash
# Backend — Python ≥3.11 (3.12 in Docker/CI; dev venv may be newer), FastAPI factory pattern
pip install -r backend/requirements.txt            # (+ requirements-dev.txt for pytest/parallel)
.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000   # or ./start.sh
.venv/bin/python -m pytest tests/ -n auto           # suite is socket-hermetic (pytest-socket)
uvx ruff@0.15.16 check .                            # lint (config in pyproject.toml)

# Frontend — Next.js 16, Node ≥20.19<21 or ≥22.12
cd frontend && npm ci && npm run dev                # needs OPENRESEARCH_BACKEND_URL (default http://127.0.0.1:8000)
npm run build && npm run lint && npm test && npx tsc --noEmit

# CLI reproduction (no UI)
python -m backend.cli reproduce 2512.24601          # arXiv ID | paper.pdf
python -m backend.cli campaign <paper> --max-llm-usd X --max-gpu-usd Y --max-gpu-hours Z
```

Common flags: `--mode {rlm(default),rdr,rlm-pure}`, `--provider`, `--sandbox
{auto,docker,local,aws,azure,gcp}` (default `local`; `aws`=EKS, `azure`=AKS are the primary
clouds; `gcp`/`gke` is PARKED and raises unless `OPENRESEARCH_ALLOW_GKE=1`; auto = docker/local
only, never a paid remote), `--model`, `--models role=token,…`, `--vram-gb`, `--max-usd`.
Root-model vocabulary, the two auth surfaces, per-role selection, Foundry/Grok, and the full flag
catalog live in **`backend/agents/rlm/CLAUDE.md`**; sandbox/GPU knobs in
**`backend/services/runtime/CLAUDE.md`**.

## How it works (30-second map)

- **RLM orchestrator** (`backend/agents/rlm/run.py`) builds `rlm.RLM(...)` and calls `.completion()`
  on a worker thread. The paper is offloaded as the REPL `context` variable — the root sees only
  constant-size metadata, never the corpus. The root writes Python calling the **19 bound
  primitives** (`primitives.py`) and terminates via `FINAL_VAR(<var>)`.
- **File-backed run state** (`runs/<project_id>/`, not a service — each run is a long-lived
  subprocess): `demo_status.json`, `rlm_state/` (resume-safe checkpoints), `dashboard_events.jsonl`
  (SSE log), `final_report.{json,md}`, `cost_ledger.jsonl`, `experiment_runs.jsonl`, `code/` (the
  reproduced project). SQLite (`OPENRESEARCH_DATABASE_URL`) is the CQRS event store.
- **One image, two processes:** FastAPI on internal `:8000` + Next.js on public `:$PORT`; the
  browser reaches the backend only through server-side `/api/demo/*` proxy routes — no CORS.
- **SSE lifecycle:** UI → `POST /api/demo` → backend spawns the run → UI opens SSE via
  `/runs/<id>/events`. The egress sanitizer (`sse_bridge.sanitize_iteration`) strips REPL locals and
  bounds output; the paper corpus never reaches the stream.

### Where to look first
| Concern | Start here |
|---|---|
| HTTP layer · CLI | `backend/app.py` · `backend/cli.py` |
| RLM run · primitives · prompt | `backend/agents/rlm/{run,primitives,system_prompt}.py` → **`backend/agents/rlm/CLAUDE.md`** |
| Sandboxes · GPU · cloud | `backend/services/runtime/` → **`backend/services/runtime/CLAUDE.md`** |
| SSE egress chokepoint | `backend/agents/rlm/sse_bridge.py` |
| UI · lab · leaderboard | `frontend/src/` → **`frontend/CLAUDE.md`** |
| Paper ingestion | `backend/services/ingestion/parser/resolving_parser.py` |

`backend/` and `backend/services/` are named by function — read them directly.

## Rules that always apply

Load-bearing invariants; the owning nested file/spec carries the full rule + incident history.

- **Evidence, not grade.** Verdicts, trust gates, and self-improvement key on the deterministic
  evidence layer (measured artifacts on disk), never a scalar LLM grade. The evidence gate +
  fabrication guards are **fail-closed**. → `backend/agents/rlm/CLAUDE.md`
- **Keep the import-time patches.** `run.py` imports the forced-iteration policy (refuses a
  premature `FINAL_VAR`) and the REPL safe-builtins patch (restores `globals`/`locals` but **NEVER**
  `eval`/`exec`/`compile`/`input` — the real security boundary). → `backend/agents/rlm/CLAUDE.md`
- **claude-agent-sdk isolation.** Every `ClaudeAgentOptions(...)` MUST pass `setting_sources=[]`, an
  explicit `mcp_servers` dict, and a non-plan `permission_mode`, or the inner model inherits the
  developer's `~/.claude`. → `backend/agents/rlm/CLAUDE.md`
- **Two LLM auth surfaces, billed separately** (root model vs Sonnet sub-agents). A no-credit
  `ANTHROPIC_API_KEY` does **not** fall back to OAuth; a stale shell export shadows `.env`.
  → `backend/agents/rlm/CLAUDE.md`
- **Cost visibility.** `cost_ledger.jsonl`/`demo_status.json` are **blind** to Foundry-routed LLM
  spend and idle GPU-node time — a `$0` there is not proof of $0. Verify real cost via
  `tokens_total.json` + `kubectl get nodes` (stray A100s), never the ledger alone.
- **GKE runs go through the cell-matrix.** The monolithic `k8s_job_backend.exec` path never
  stages code into the pod; on gcp/gke, training routes via `cells.json`+`train_cell.py` (or the
  `OPENRESEARCH_GKE_SYNTH_CELL` synthesis). → `backend/services/runtime/CLAUDE.md`
- **Delegation.** The session's lead model owns design + reviews **every diff**; delegate
  mechanical impl + wide recon to Sonnet/`Explore` sub-agents against a tight spec. → memory.
- **Docker daemon** is a prerequisite only for the `docker`/`auto` sandboxes; `build_environment` is
  a no-op for `local`/`azure`/`aws`/`gcp`. → `backend/services/runtime/CLAUDE.md`
- **New feature flags** use `os.environ.get("FLAG","").strip().lower() in ("1","true","yes")`,
  default-OFF and byte-identical when off; a default-flip needs ≥3 paired A/B runs + the grader-σ
  gate. → `backend/agents/rlm/CLAUDE.md`
- **`OPENRESEARCH_SCHEDULER_AUTHORITATIVE`** is default-OFF and requires
  `OPENRESEARCH_SCHEDULER_TREE`; it may never adopt an LLM grade or bypass a
  deterministic terminal evidence decision. Its A/B gate is evidence for operator
  review, never an automatic default flip. → `backend/agents/rlm/CLAUDE.md`
- **Git.** Branch off `main`; no Conventional-Commit prefixes; descriptive present-tense headline;
  **no `Co-Authored-By`/AI-attribution trailer**; author = local config; commit at milestones;
  commit/push only when asked, to the operator-designated remote.
- **Docs.** Keep the *rule* here or in the nested file; keep the *incident narrative* in its
  spec/runbook/memory. When two docs disagree the higher tier wins — and **the code always wins**.

## Demo gate
When `OPENRESEARCH_DEMO_SECRET` is set, run-start endpoints require a matching `X-Demo-Secret`
header (`hmac.compare_digest`). Empty/unset disables the gate — local-dev behavior, not a bug.

## Doc map
- **Nested `CLAUDE.md` (load on-demand):** `backend/agents/rlm/` · `backend/services/runtime/` ·
  `frontend/` · `tests/`.
- **Current docs:** `README.md`, `docs/architecture.md`, and `docs/operations.md`.
- **Baseline test paper:** SDAR (arXiv 2605.15155) — the canonical stress test.

## Context-mode routing
Inherits context-mode MCP routing from the parent `CLAUDE.md`: use
`ctx_batch_execute`/`ctx_execute`/`ctx_execute_file` for any command or file read producing >20
lines, and `ctx_fetch_and_index` instead of `WebFetch`/`curl`/`wget`. The parent file has the full
blocked-vs-redirected table.

## Maintaining this doc
Root stays lean — orientation + always-on rules + pointers. When you add a primitive, SSE event,
sandbox, or flag, update the **nested** `CLAUDE.md` that owns it, not this file. Fidelity anchors
kept current here + guarded by `tests/test_claude_md_fidelity.py` (which reads the root **and**
nested set): the bound primitive count is **19**, and the default sandbox is `local`.
`docs/architecture.md` = the system map; this = the day-to-day.
