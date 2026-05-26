# Best Runs

**Start here:** [`EXECUTIVE_SUMMARY.md`](./EXECUTIVE_SUMMARY.md) — 2-minute read with headline numbers and what this means for MSFT.

Then:
- [`PAPER_CLAIMS_VS_REPRODUCED.md`](./PAPER_CLAIMS_VS_REPRODUCED.md) — side-by-side of every paper claim vs. agent-derived metric.
- [`METHODOLOGY.md`](./METHODOLOGY.md) — how the agent does this (the 12 primitives + observability surfaces).

Best reproductions per paper. Compact artifacts (final report, rubric, generated code, environment spec) extracted from the full run directories under `runs/` (gitignored).

| Paper | Verdict | Overall Rubric Score | Iterations | Path |
|---|---|---:|---:|---|
| Adam: A Method for Stochastic Optimization (1412.6980) | reproduced | 0.7413 | — | [adam/](./adam/) |
| Auto-Encoding Variational Bayes (1312.6114) | partial | 0.6457 | 3 | [vae/](./vae/) |

Each directory carries:
- `final_report.{json,md}` — the orchestrator's final write-up + telemetry sidecars rendered
- `generated_rubric.json` — auto-derived PaperBench-style rubric
- `rubric_evaluation.json` — per-leaf scores + justifications
- `environment_spec.json` + `Dockerfile` — the built env
- `code/` — the agent-implemented baseline
- `tokens_total.json`, `timing.json`, `cost_ledger.jsonl`, `experiment_runs.jsonl` (Adam only — VAE pre-dates the sidecar-emit codepath)
