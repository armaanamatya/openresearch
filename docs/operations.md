<!-- doc-meta: status=current; last-verified=2026-07-20 -->
# Operations

## Start locally

```bash
make setup
cp .env.example .env
make dev
```

`make dev` starts the API on `:8000` and the frontend on `:3000`. Use
`make dev-backend` only when running the frontend separately.

For a no-cloud smoke run, keep `OPENRESEARCH_DEFAULT_SANDBOX=local` in `.env`.
Then run:

```bash
.venv/bin/python -m backend.cli reproduce demo_paper.pdf --sandbox local
```

## Verify before a change

```bash
make smoke
make docs-check
make check
```

`make check` runs the full backend and frontend checks. Use the smaller first
two commands while iterating.

## Cloud runs

Cloud sandboxes are operator-only. Configure the matching provider variables
in `.env`, validate credentials and capacity, then explicitly choose the
sandbox (`runpod`, `gcp`, `azure`, or `aws`) on the CLI. Do not assume a cloud
backend is available merely because code for it exists.

## Where output goes

Runs write to `runs/<project-id>/`; that directory is intentionally ignored.
Keep only reviewed, durable evidence in `best_runs/`. See
[policies/artifacts.md](policies/artifacts.md).
