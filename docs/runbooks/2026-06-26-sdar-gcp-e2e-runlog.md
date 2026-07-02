# SDAR-on-GCP e2e — coworker run ledger

> Append-only log of every SDAR / GCP reproduction run attempted from this repo.
> Auto-written by [`scripts/sdar_runlog.py`](../../scripts/sdar_runlog.py) — fail-soft;
> the outcome fields (status / score / verdict / cost / cells / stop) are derived from
> `runs/<project_id>/{final_report.json, demo_status.json, cost_ledger.jsonl,
> experiment_runs.jsonl}` on disk. Read top-to-bottom; **newest rows at the bottom.**

**Why this exists:** so coworkers can see, at a glance, every run we've tried — the
exact command, root model, scope, sandbox/instance, and the *honest* outcome (rubric
score, replication verdict, cost, cells) — without digging through `runs/`. The
`score` column is `overall_score/target ✓|✗` (✓ = `meets_target`).

## How to log a run

```bash
# at launch (static fields known up front):
python scripts/sdar_runlog.py --project-id <id> --event launched \
  --root claude-oauth --scope full-grid --sandbox "gcp/sdar-ultra/us-central1-c" \
  --command "PRIMARY=1 ROOT=claude-oauth scripts/sdar_gcp_e2e.sh run" \
  --note "first run on merged code (Ayush round-2)"

# any time after (outcome auto-derives from the run dir on disk):
python scripts/sdar_runlog.py --project-id <id> --event finished
```

The proven run recipe + all env knobs live in
[`2026-06-24-sdar-optimization-handoff.md`](2026-06-24-sdar-optimization-handoff.md)
and [`scripts/sdar_gcp_e2e.sh`](../../scripts/sdar_gcp_e2e.sh). Every run here also
appears on the `/leaderboard` (aggregated from the same `final_report.json`).

## Run log

| UTC | project_id | event | root/model | scope | sandbox/instance | status | score | verdict | cost USD | cells ok/total | stop | run dir | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-26 20:16Z | sdar_merged_full_2g | launched | claude-oauth | full-grid 30c/20-step | gcp/sdar-ultra/us-central1-c/a2-ultragpu-2g(2xA100-80GB) | - | - | - | - | - | - | runs/sdar_merged_full_2g/ | run 1/2 on merged code HEAD 260df7da (Ayush round-2 + grounded); LIFECYCLE_PRIMARY, GRADER_SAMPLES=3, repo-first, oauth keyless · `MIN_GPUS=2 GUIDANCE=full PRIMARY=1 scripts/sdar_gcp_optimal_run.sh` |
| 2026-06-26 20:22Z | sdar_merged_full_2g | progress | - | - | gcp/sdar-ultra/2xA100-80GB | - | - | - | - | - | - | runs/sdar_merged_full_2g/ | live: cells_with_metrics=0 |
| 2026-06-26 21:27Z | sdar_merged_full_2g | progress | - | - | gcp/sdar-ultra/2xA100-80GB | - | - | - | - | - | - | runs/sdar_merged_full_2g/ | live: cells_with_metrics=10 |
| 2026-06-26 22:32Z | sdar_merged_full_2g | finished | - | - | gcp/sdar-ultra/2xA100-80GB | - | - | - | - | - | - | runs/sdar_merged_full_2g/ | auto-finalized by ledger-monitor |
| 2026-06-27 03:01Z | sdar_merged_full_2g | finished-recovered | claude-oauth | full-grid 20-step | gcp/sdar-ultra/2xA100-80GB | failed | - | failed | 3.355 | 1/1 | - | runs/sdar_merged_full_2g/ | RECOVERED from disk: trained 22-23 real cells x 3 Qwen models end-to-end ($0 LLM, repo-first); verify_against_rubric TIMED OUT at 600s on the big grid -> finalized failed/not-scored. Training OK; grading-timeout bug, not a training failure. Search-QA rewards ~0.0016 non-zero, ALFWorld/WebShop ~0 (known 20-step ceiling; webshop env failed). |
| 2026-06-27 03:52Z | sdar_validate_full | launched | claude-oauth | full-grid (validation) | gcp/sdar-ultra/2xA100-80GB | - | - | - | - | - | - | runs/sdar_validate_full/ | VALIDATION of harness fixes (commit 12813d4a): verify-cap override + GRADER_SAMPLES=1 + finalize salvage + NO_AUTOSTOP. Confirmed live env: SAMPLES=1, NO_AUTOSTOP=1, VERIFY_TIMEOUT=1800. Tests train->grade->finalize-with-score->retrieve e2e. · `GRADER_SAMPLES=1 VERIFY_TIMEOUT_S=1800 NO_AUTOSTOP=1 KEEP_UP=1 scripts/sdar_gcp_optimal_run.sh` |
| 2026-06-27 03:52Z | sdar_validate_full | progress | - | - | gcp/sdar-ultra/2xA100-80GB | - | - | - | - | - | - | runs/sdar_validate_full/ | live: cells_with_metrics=0 |
| 2026-06-27 14:54Z | sdar_validate_full | finished | claude-oauth | - | - | completed | 0.216/0.6 ✗ | partial | 2.8583 | 1/1 | - | runs/sdar_validate_full/ | VALIDATION COMPLETE e2e: verify_against_rubric fired 5x WITHOUT 600s timeout (FIX 1+2 worked), finalized WITH real score 0.216/partial (not failed/not-scored). 20 cells trained. Monitor died after first poll so VM not auto-stopped — report recovered via SSH-tar+scp, VM stopped manually. |
