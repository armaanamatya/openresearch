# GEPA Per-Run Sub-Agent Optimization — Design Spec

**Date:** 2026-05-31
**Branch:** feat/gepa-integration (based on feat/rlm-wedge-hardening)
**Status:** Implemented

## Problem

ReproLab's sub-agent prompts (`plan_reproduction`, `implement_baseline`, `propose_improvements`) are static strings that never adapt to the specific paper being reproduced. A bad `plan_reproduction` contract corrupts all downstream work.

## Solution

Wire GEPA (Genetic-Pareto Prompt Optimizer, arXiv 2507.19457) into `wrap_primitive()` so it runs a mini-optimization loop before every call to a targeted primitive, continuously improving prompts from each call's output within the same run.

## Architecture

GEPA fires in two places inside `wrap_primitive()` in `binding.py`:
- **`gepa_pre_call(ctx, name, args, kwargs)`** — before the primitive thread spawns; runs `gepa.optimize()` with a paper-specific trainset; stores winner in `ctx.gepa_prompt_overrides[name]`
- **`gepa_post_call(ctx, name, result)`** — after the primitive returns; appends the result to `ctx.gepa_example_buffer[name]` so the next call's trainset is warmer

Prompt injection uses two paths:
- **Path A** (plan_reproduction, propose_improvements): monkey-patch `ctx.llm_client.complete` to inject the optimized system prompt
- **Path B** (implement_baseline): thread `system_prompt_override` through `primitives.implement_baseline` → `baseline_implementation.run_with_sdk` → `collect_agent_text`

Re-entrancy is prevented by `ctx.gepa_optimization_active` — set True during the GEPA loop, so the evaluator's LLM calls don't trigger another GEPA loop.

## Target Primitives

| Primitive | Metric | Budget | Per-call cost |
|---|---|---|---|
| `plan_reproduction` | metrics_shape coverage + contract completeness | 10 calls, 60s | ~$0.03 |
| `implement_baseline` | structural validity (proxy, no sub-agent) | 5 calls, 30s | ~$0.015 |
| `propose_improvements` | weak-area coverage × category diversity | 10 calls, 60s | ~$0.03 |

## Feature Flags

All default OFF. Set `REPROLAB_GEPA_OPTIMIZATION=on` to enable for all three primitives, or `=plan-only`/`=baseline-only`/`=improve-only` for selective enabling.

See `backend/config.py` for the full flag list (`REPROLAB_GEPA_*`).

## New Module: backend/agents/gepa/

- `hooks.py` — entry points called by `binding.wrap_primitive`
- `optimizer.py` — thin `gepa.optimize()` wrapper with timeout + no-improvement stopping
- `callback.py` — `SSEGepaCallback` forwarding GEPA events to `ctx.emit()`
- `prompt_registry.py` — persist `gepa_example_buffer` to `gepa_examples.jsonl` at run end
- `adapters/` — evaluator callables for each primitive
- `metrics/` — deterministic scoring functions (no LLM judge)
- `trainset/` — `PaperExamplesBuilder` (cold start) + `historical_examples` (cross-run)

## SSE Events

Five new event types: `gepa_phase_start`, `gepa_candidate_proposed`, `gepa_candidate_accepted`, `gepa_candidate_rejected`, `gepa_phase_complete`. All flow through `ctx.emit()`.

## Lab UI

New `gepa_candidate` node kind in the constellation canvas (orange stroke `--gepa`). NodeDetailSidebar shows primitive name, score, score delta, and prompt preview. gepa-viz proxied at `/api/gepa-viz/*`.

## Invariants

- GEPA is never fatal: all exceptions → `run_warning` + seed prompt fallback
- `ctx.gepa_optimization_active` always restored to `False` in `finally` block
- `ctx.llm_client.complete` always restored after thread (success + timeout + exception paths)
- Flag off = zero behavioral change vs. today
