# Launch / testing readiness runbook

_Last updated: 2026-05-23._

`scripts/readiness.sh` is the single canonical preflight. Run it before any deploy, before opening a "ready for review" PR, or when triaging an environment that "isn't working." It runs tiered checks and fails fast.

## Quick start

```bash
# Everything (deps + static + tests + boot + deploy surface). ~3-5 min.
scripts/readiness.sh

# Skip the heavy stuff for a quick lint/types/import sanity. ~30 sec.
scripts/readiness.sh --tier 3

# CI mode: machine-readable, only failures printed.
scripts/readiness.sh --json

# Auto-create venv + npm ci if missing.
scripts/readiness.sh --fix

# Pre-deploy: include the real-run smoke (≤$0.50, ≤10min).
READINESS_RUN_SMOKE=1 scripts/readiness.sh
```

## Tiers

| Tier | What | Skippable | Typical time |
|---|---|---|---|
| 1 | Environment (python/node/git/etc.) | no | <1s |
| 2 | Dependencies installed (venv + node_modules) | `--fix` rebuilds | 1–60s |
| 3 | Static analysis (lint, types, factory boot, CLI help) | no | 10–30s |
| 4 | Test suites (pytest + vitest) | `--skip-tests` | 30s–3min |
| 5 | Boot smoke (backend on :8001, hit /leaderboard, /runs) | `--skip-smoke` | 10–30s |
| 6 | Run smoke (real `--mode rlm` reproduction on ftrl) | opt-in via `READINESS_RUN_SMOKE=1` | 3–10min, costs ≤$0.50 |
| 7 | Deployment surface (Docker build, demo gate, RunPod creds, Claude OAuth) | `--skip-deploy` | 30s–5min |

Failing a lower tier blocks higher tiers from running. (e.g. `pytest` doesn't even start if `from backend.app import create_app` fails in Tier 3.)

## Flags

| Flag | Effect |
|---|---|
| `--tier N` | Run tiers 1..N only (1–7). |
| `--skip-tests` | Skip TIER 4. |
| `--skip-smoke` | Skip TIER 5 + 6. |
| `--skip-deploy` | Skip TIER 7. |
| `--json` | Emit a single JSON object for CI. Implies `--quiet`. |
| `--quiet` | Only print failures + summary. |
| `--fix` | If `.venv` missing, create it; if `frontend/node_modules` missing, run `npm ci`. |

## Environment overrides

| Var | Default | Meaning |
|---|---|---|
| `READINESS_REQUIRE_CLEAN` | unset | TIER 3 fails if `git status --porcelain` is non-empty. |
| `READINESS_FRONTEND_SMOKE` | unset | TIER 5 also boots Next.js on :3001 (slow — 60–90s extra). |
| `READINESS_RUN_SMOKE` | unset | TIER 6 actually runs (default: SKIP). |
| `OPENRESEARCH_READINESS_RUN_BUDGET_USD` | `0.50` | Hard cap for TIER 6 run. |
| `OPENRESEARCH_READINESS_RUN_WALL_S` | `600` | Wall-clock cap for TIER 6 run. |
| `NO_COLOR` | unset | Disable ANSI colors. |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | All checks passed (WARN ok). |
| 1 | TIER 1, 2, or 3 failure — blocking. |
| 2 | Test failure (TIER 4). |
| 3 | Boot smoke failure (TIER 5). |
| 4 | Run smoke failure (TIER 6). |
| 5 | Deploy-surface failure (TIER 7). |
| 10 | Script-internal error (bad flag, etc.). |

## When to use

- **Before opening a "ready for review" PR**: `scripts/readiness.sh --skip-deploy` (3 min) should be clean.
- **Pre-deploy**: `READINESS_RUN_SMOKE=1 scripts/readiness.sh` — full pass including a real reproduction.
- **CI gate**: `scripts/readiness.sh --json --tier 4` on every push; full pass on merge.
- **Triage "it's not working"**: `scripts/readiness.sh --tier 2` — most envs fail here (missing venv / node_modules / wrong python version).

## Common failures

| Failure | Fix |
|---|---|
| `.venv exists: missing` | Run `scripts/readiness.sh --fix` or `python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt`. |
| `node ≥ 20.19 (≠21) or ≥ 22.12: FAIL` | Install via `fnm` / `nvm`: `fnm install 22.12 && fnm use 22.12`. |
| `backend factory imports cleanly: FAIL` | Read `.readiness-backend.log` (if TIER 5 ran) or run `.venv/bin/python -c "from backend.app import create_app"` manually to see the traceback. |
| `Claude credentials: WARN` | Either `claude login` (subscription) or set `ANTHROPIC_API_KEY=<key>` in `.env`. Subscription is free, API key is per-token billed. |
| `RunPod creds present: WARN` | Set `OPENRESEARCH_RUNPOD_API_KEY` + `OPENRESEARCH_RUNPOD_SSH_KEY_PATH` in `.env`, or stick to `--sandbox local`/`docker`. |
| `ftrl --mode rlm smoke: FAIL` | Read the run dir under `${TMPDIR}` — `final_report.json` and `dashboard_events.jsonl` tell the story. Most failures are credential issues (sub-agent can't auth) — see CLAUDE.md §"RLM auth". |

## CI integration

GitHub Actions snippet (drop into `.github/workflows/ci.yml`):

```yaml
- name: Readiness — fast path (tier 1-3)
  run: scripts/readiness.sh --tier 3

- name: Readiness — tests + boot
  run: scripts/readiness.sh --tier 5 --json > readiness-tier5.json

- uses: actions/upload-artifact@v4
  with: { name: readiness, path: readiness-tier5.json }
```

## Maintenance

When you add a new prerequisite, add the check to the appropriate tier in `scripts/readiness.sh` and document it here. The script is intentionally readable bash — no clever frameworks, no parallelism — so every check is easy to add or audit.
