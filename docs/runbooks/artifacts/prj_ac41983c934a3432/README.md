# `prj_ac41983c934a3432` sanitized run artifact

Source run: `runs/prj_ac41983c934a3432`

Paper: `2512.24601`

Launch mode: `python -m backend.cli reproduce 2512.24601 --model claude-oauth --sandbox runpod --max-run-gpu-usd 5.0 --max-pod-seconds 7200 --max-wall-clock 5400`

Run summary:

| Field | Value |
|---|---|
| Started | `2026-05-27T21:23:08Z` |
| Completed | `2026-05-27T21:45:24Z` |
| Final event | `run_complete.status=failed` |
| Demo status | `status=completed`, `run_state.kind=failed` |
| Rubric | `0.0 / 0.6`, all leaves degraded/no metrics |
| RunPod pods | `0` |
| RunPod spend | `$0.00` |
| Main failure | `implement_baseline` failed three times; one invalid `run_experiment` call received an error dict instead of a code path. |

## Included

This directory contains a safe, reviewable subset of the run:

- `dashboard_events.jsonl`
- `demo_status.json`
- `cost_ledger.jsonl`
- `experiment_runs.jsonl`
- `final_report.json`
- `final_report.md`
- `generated_rubric.json`
- `rubric_evaluation.json`
- `tokens_total.json`
- `worker_reports.jsonl`
- `environment_spec.json`
- `Dockerfile`
- `code/config.json`
- `code/requirements.txt`
- `rlm_state/gpu_plan.json`
- `rlm_state/iterations.jsonl`
- `rlm_state/primitive_cache.jsonl`
- `reports/summary_report.json`
- `reports/worker_reports.jsonl`
- `iterations/iteration_0001.json`

## Excluded

The raw run directory contains files that are intentionally not committed:

- `raw_paper.pdf` and `code/paper.pdf` — large source PDFs duplicated from arXiv.
- `raw_paper.html` — large fetched HTML artifact.
- `parsed_full_text.txt` — large paper text; can be regenerated from source.
- `repl_state.pickle` — binary runtime state; not safe or useful for review.

The full local source remains under `runs/prj_ac41983c934a3432` on the machine
that produced the run.

## Related Docs

- `docs/runbooks/2026-05-27-rlm-reproduction-stability-plan.md`
- `docs/runbooks/2026-05-27-sdar-run-issues.md`

