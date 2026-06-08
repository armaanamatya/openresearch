# SDAR reproduction campaign — run logs + token usage

Four end-to-end reproduction attempts of **SDAR: Self-Distilled Agentic Reinforcement Learning** (arXiv `2605.15155`) by the OpenResearch RLM agent, run back-to-back on a local 8×A5000 box over **2026-05-31 19:05 → 2026-06-01 03:15 UTC**. PDF in, scored reproduction out, no human in the coding loop.

SDAR is the project's hardest stress paper — 3 Qwen sizes (1.7B/3B/7B), 3 agentic environments (ALFWorld + WebShop + Search-QA), GRPO RL + sigmoid-gated OPSD self-distillation, 5 baselines, and a rubric whose leaves inspect for the algorithm's exact invariants (`g_t = σ(β·Δ_t)`, stop-gradient on the gate, λ=0.1, β=10, real Qwen weights, real episodes). A surrogate cannot pass. The four attempts are a real debugging arc, not four clean wins — three failed on infrastructure (GPU contention, a launch crash, a dead dataset endpoint) before the fourth produced a scored **partial**.

Each attempt's directory carries that run's **run logs** (`dashboard_events.jsonl`, `experiment_runs.jsonl`, `worker_reports.jsonl`, `batch_child.log`, and per-cell `train*.log` where present), its **token-usage logs** (`tokens_total.json`, `cost_ledger.jsonl`), the final report + rubric grading where the run got that far, the agent-written `code/`, and a computed `run_summary.json`. `campaign_summary.json` at this level aggregates all four.

> An earlier 2026-05-30 attempt (`runs/_archive/prj_09047604e591d969_prefsdp`, partial / 0.000) exists but predates the token-telemetry sidecars in its run dir, so it is not packaged here.

---

## The four attempts

| # | Started (UTC) | Verdict | Rubric | Iter | Wall | Est. $ | What happened |
|---|---|---|---:|---:|---:|---:|---|
| [1](attempt-1-oom-foreign-proc/) | 05-31 19:05 | failed | 0.000 | 3 | 80m | 10.93 | CUDA OOM — a foreign process held 22+ GiB of the 23.7 GiB A5000, leaving nothing for any Qwen. ALFWorld also hit an `AttributeError`; WebShop endpoint 404. |
| [2](attempt-2-launch-crash/) | 05-31 23:10 | crashed | — | — | <1m | 0.00 | asyncio event-loop closed during pipeline start. Died after ingest, before any LLM work. No final report. |
| [3](attempt-3-capacity-exhausted/) | 05-31 23:21 | failed | 0.089 | 1 | ~23m | 3.00 | `capacity_exhausted` — the Search-QA dataset (`nq_open` on HF) returned 404, so the capacity gate dropped every cell. Method/code leaves still graded to 0.089. |
| [4](attempt-4-partial/) | 05-31 23:58 | **partial** | **0.363** | 10 | 197m | 22.99 | Scored partial. Search-QA ran on Qwen3-1.7B + Qwen2.5-3B; ALFWorld/WebShop unavailable, 7B capacity-skipped. |

"Est. $" is the **estimated API-equivalent** cost from `cost_ledger.jsonl`. All four runs used the Claude Code **OAuth subscription** (`claude-oauth` root + `claude-sonnet-4-6` sub-agents), so **actual LLM cash spend is ~$0**; the dollar figure is a workload/billing signal — what an equivalent metered Anthropic-API deployment would have cost. GPU time was on local hardware.

The progression is the story: each failure is a different layer of the stack giving way — GPU contention (1), the orchestrator itself (2), an upstream data dependency (3) — before the capacity gate + one-GPU-per-cell scheduler let attempt 4 actually place work and score it.

---

## Attempt 4 — the scored partial

**0.363 / 0.600** overall · 26 rubric leaves (25 graded, 1 excluded) · PaperBench-bundle rubric.

### Rubric by area

| Area | Score |
|---|---:|
| Method + code fidelity to the paper | 0.627 |
| Data + preprocessing fidelity | 0.090 |
| Experiment execution + reproducibility | 0.370 |
| Evaluation protocol + metric correctness | 0.185 |
| Result match vs. the paper's targets | 0.015 |
| Artifact completeness + provenance | 0.330 |

Method/code fidelity lands highest — the agent implemented the SDAR objective (`L_total = L_GRPO + 0.1·L_OPSD`, `g_t = σ(10·Δ_t)`, `Δ_t = log π_teacher − log π_student`, stop-gradient on teacher + gate). Result-match lands near zero because the scope that actually ran is a small slice of the paper's grid (see below).

### What ran vs. what the paper asks for

**Ran** (4 cells, Search-QA, seed 42, 150 steps each):

| Cell | reward_mean | gate_mean | gate_active | wall |
|---|---:|---:|---:|---:|
| qwen3_1.7b / search_qa / **sdar** | 0.0543 | 0.418 | 0.317 | 878s |
| qwen3_1.7b / search_qa / grpo | 0.0567 | 0.000 | 0.000 | 825s |
| qwen3_1.7b / search_qa / grpo_opsd | 0.0521 | 1.000 | 1.000 | 539s |
| qwen2.5_3b / search_qa / grpo | 0.1450 | 0.000 | 0.000 | 836s |

The gate behaves as designed across the three 1.7B variants — `grpo` keeps the gate off (0.0), `grpo_opsd` saturates it (1.0), and `sdar`'s sigmoid gate sits in between (0.418, active on 32% of tokens). That is the SDAR mechanism doing something real.

**Honest caveat:** on the one cell where SDAR and GRPO are directly comparable (Qwen3-1.7B / Search-QA), **SDAR 0.0543 is marginally *below* GRPO 0.0567** at this 150-step budget — this run did **not** reproduce the paper's headline "+7.0% over GRPO" claim at this scale. The reproduction is partial in both senses: partial grid coverage *and* the headline effect not yet demonstrated.

**Did not run** (recorded as scope gaps, excluded from the score rather than penalized as zero):

- **ALFWorld** — simulator not installed on the host.
- **WebShop** — WebShop server not available (endpoint 404).
- **Qwen2.5-7B** — capacity-skipped: exceeds the 24 GB per-GPU VRAM budget (headroom 1.25×).
- **Skill-SD, RLSD, OPSD-standalone** baselines — not implemented.

Paper's own targets, for reference: SDAR over GRPO by **+9.4%** (ALFWorld), **+10.2%** (WebShop), **+7.0%** (Search-QA).

---

## Token usage

The user-facing deliverable. Each attempt's `tokens_total.json` carries counts `by_model` and `by_primitive`; `cost_ledger.jsonl` carries per-call USD with cache-token columns.

**Note vs. the adam/vae sidecars in the parent dir:** those older `tokens_total.json` files excluded the Sonnet sub-agent. **These SDAR files include it** (`claude-sonnet-4-6`), which is why output-token counts are large — the `implement_baseline` code-writing sub-agent dominates the workload.

### Per attempt (totals across all models)

| Attempt | Input | Output | Est. $ root | Est. $ sub-agents | Est. $ total |
|---|---:|---:|---:|---:|---:|
| 1 — oom | 183 | 156,462 | 0.78 | 10.15 | 10.93 |
| 2 — launch-crash | 0 | 0 | 0.00 | 0.00 | 0.00 |
| 3 — capacity | 89 | 51,781 | 0.15 | 2.85 | 3.00 |
| 4 — partial | 455 | 494,697 | 3.55 | 19.43 | 22.99 |
| **Combined** | **727** | **702,940** | **4.48** | **32.43** | **36.92** |

### Attempt 4 by model

| Model | Role | Input | Output |
|---|---|---:|---:|
| `claude-sonnet-4-6` | `implement_baseline` sub-agent (writes the code) | 372 | 464,581 |
| `claude-oauth` | RLM root (orchestration) | 73 | 28,721 |
| `claude-haiku-4-5` | helper | 10 | 1,395 |
| **Total** | | **455** | **494,697** |

### What this tells you

- **The code-writing sub-agent is the cost.** On attempt 4, `baseline-implementation` is **$19.43 of $22.99 (85%)** of the estimated spend and 464K of 495K output tokens. The RLM root's own orchestration is only $3.55. Across the campaign, sub-agents are 88% of estimated cost.
- **Failed runs still meter.** Attempt 1 burned ~156K output tokens and $10.93-equivalent generating code that then couldn't run because of GPU contention — the LLM did its job; the host didn't have a free GPU. Attempt 3 spent $3.00-equivalent before the dead dataset endpoint stopped it.
- **Cash cost was zero.** OAuth subscription on both surfaces. The $36.92 combined figure is the API-equivalent a metered deployment would have paid.

---

## What's in each attempt directory

```
attempt-N-*/
  run_summary.json          computed at package time: status, verdict, rubric score,
                            iterations, wall clock, token totals, est-cost split
  final_report.{json,md}    the scored report  (absent for attempt 2 — it crashed first)
  generated_rubric.json     rubric auto-derived from the paper text
  rubric_evaluation.json    per-leaf score + justification  (where the run scored)
  environment_spec.json     framework, packages, GPU/CPU plan
  tokens_total.json         TOKEN USAGE — counts by model and by primitive
  cost_ledger.jsonl         TOKEN/COST USAGE — per-call USD + cache columns (append-only)
  experiment_runs.jsonl     RUN LOG — every run_experiment: success, metrics, logs, failure-class
  dashboard_events.jsonl    RUN LOG — the full sanitized SSE event stream (no paper corpus)
  worker_reports.jsonl      RUN LOG — per-worker structured reports
  batch_child.log           RUN LOG — the batch launcher / child-process stdout (ingest → finish)
  train*.log                RUN LOG — per-cell training stdout  (attempt 1 only)
  timing.json               wall clock + per-primitive duration  (attempts 1 & 4)
  demo_status.json          run lifecycle snapshot
  code/                     the agent-written train.py + cells + outputs/ (figures, curves);
                            pycache / *.pyc / stray .heartbeat stripped
```

`dashboard_events.jsonl` is safe to share: every event routes through the orchestrator's SSE sanitizer, which strips REPL locals and bounds stdout/stderr — the paper corpus never reaches the stream.

### Where to look first as a coworker

- **"What did the agent do, in order?"** → `batch_child.log` (top-to-bottom narrative) then `dashboard_events.jsonl` (every primitive call + event).
- **"Why did it fail / what did each cell produce?"** → `experiment_runs.jsonl` and (attempt 1) `train_alfworld.log` / `train_run*.log`.
- **"What did it cost / how many tokens?"** → `tokens_total.json` + `cost_ledger.jsonl`, or the rolled-up `run_summary.json`.
- **"What code did it write?"** → `code/train.py`, `code/cells.json` (the matrix manifest), `code/train_cell.py` (single-cell trainer).

---

## How the reproduction runs

The orchestrator is a Recursive Language Model (RLM, arXiv 2512.24601): the paper is loaded into a Python REPL as a variable the root model never carries in its context window; when it needs paper content it writes Python that slices the variable. The root calls twelve domain primitives (`understand_section`, `extract_hyperparameters`, `detect_environment`, `build_environment`, `plan_reproduction`, `implement_baseline`, `run_experiment`, `verify_against_rubric`, `propose_improvements`, `record_candidate_outcome`, `check_user_messages`, `respond_to_user`); iteration count, primitive order, and termination are the root's decisions, guard-railed by rubric-guard schema assertions, a forced-iteration policy, a wall-clock watchdog, and the SSE sanitizer.

On a local GPU backend, `run_experiment` routes the training matrix through a one-GPU-per-cell scheduler (`gpu_cell_runner.run_matrix`) with per-cell OOM shrink-retry, and a capacity gate drops cells that exceed the per-GPU VRAM budget or target a dead dataset — the machinery that turned attempt 3's dataset-404 and attempt 4's 7B-too-big into clean recorded gaps instead of a re-OOM loop.
