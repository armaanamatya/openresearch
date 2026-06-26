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
