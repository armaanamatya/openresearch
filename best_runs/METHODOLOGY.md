# Methodology — how the agent reproduces a paper

A short, VP-readable description of what happens between "PDF in" and "scored reproduction out".

## Pipeline

```
PDF ─▶ ingest ─▶ RLM root model ─▶ 12 domain primitives ─▶ sandbox ─▶ rubric grader ─▶ final_report.{json,md}
```

The root model is a **Recursive Language Model** (RLM, arXiv 2512.24601). The paper is *offloaded* into the REPL as a variable; the root model only ever sees constant-size metadata about it. When it needs paper content, it writes Python that slices into the variable. This is what makes paper-length context tractable — the model never carries the corpus in its window.

## The 12 primitives the root model can call

| Primitive | What it does |
|---|---|
| `understand_section` | Pull datasets / metrics / training recipe / hardware clues from a slice |
| `extract_hyperparameters` | Optimizer, LR, batch size, epochs |
| `detect_environment` | Build an `EnvironmentSpec` (Dockerfile, framework, packages) |
| `build_environment` | Build the Docker image; self-repair on failure |
| `plan_reproduction` | Write a `ReproductionContract` (smoke-test plan, eval plan) |
| `implement_baseline` | Dispatch a code-writing sub-agent (Claude Sonnet) to produce `train.py` |
| `run_experiment` | Execute the baseline in a sandboxed container; return `{success, metrics, logs}` |
| `verify_against_rubric` | Score results against a PaperBench-style rubric (24 leaves / 6 areas) |
| `propose_improvements` | Generate paper-specific improvement hypotheses for the next iteration |
| `record_candidate_outcome` | Persist the root's decision about a candidate so the orchestrator can route |
| `check_user_messages` | Read steering input typed in the lab UI mid-run |
| `respond_to_user` | Reply on the chat panel without a separate LLM call |

The root model decides iteration count, *when* to call each primitive, and *when* to terminate. There are no fixed gates.

## What's outside the model

- **Forced-iteration policy (Lane H):** if the root tries to finalize with a sub-target rubric score before a configurable minimum iteration count, the orchestrator refuses the `FINAL_VAR` and emits a run-warning explaining the next concrete step.
- **Rubric guard (Lane G):** every agent-written `train.py` ends with `assert_metrics_schema(...)`; a missing metric key surfaces as a typed `RubricGuardFailure` whose JSON-shaped message becomes the next iteration's repair context.
- **Wall-clock watchdog:** hard-exits any wedged primitive.
- **Sandbox dispatch:** `local` / `docker` / `runpod` (with dynamic-GPU SKU selection per paper).
- **SSE sanitizer:** every UI event flows through one egress chokepoint that strips REPL locals and bounds stdout/stderr. The paper corpus never reaches the frontend.

## Observability surfaces (per run)

- `final_report.{json,md}` — the canonical artifact (in this directory's `adam/` and `vae/`).
- `tokens_total.json` — per-model + per-primitive token counts.
- `timing.json` — wall clock + per-primitive duration + GPU hours / GPU type.
- `cost_ledger.jsonl` — append-only per-primitive USD ledger.
- `experiment_runs.jsonl` — every `run_experiment` invocation: logs, success, metrics, failure-class, suggested-fix.
- `dashboard_events.jsonl` — append-only SSE event log (drives the live UI).

## Auth surfaces (no vendor lock-in)

Same orchestrator runs under any of:
- **Claude Code OAuth** — subscription billing, no API key (this is what the recorded runs used)
- **Anthropic API** — per-token billing
- **OpenAI** — per-token billing (`--model gpt-5`)
- **Azure OpenAI** — per-token billing (`--model azure`)
- **Featherless** — per-token, cheapest (`--model qwen3-coder-featherless`)

Swap is a CLI flag.
