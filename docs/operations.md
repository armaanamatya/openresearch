<!-- doc-meta: status=current; last-verified=2026-07-22 -->
# Operations

## Start locally

```bash
make setup
cp .env.example .env
make dev
```

`make dev` starts the API on `:8000` and the frontend on `:3000`. Use
`make dev-backend` only when running the frontend separately.

For a no-cloud smoke run, use `--sandbox local`; `local` is the configured
default and `auto` only resolves to `docker` or `local`. Then run:

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

Cloud sandboxes are operator-only. For GCP the supported GPU route is the
single-VM path (a fresh GPU VM running `reproduce --sandbox local`, then
auto-delete — see
[`2026-07-22-gcp-vm-e2e-run-procedure.md`](runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md));
GKE is not used — a fail-closed guard rejects it (`OPENRESEARCH_ALLOW_GKE` is an
inert operator-only escape hatch, not a supported path).
Azure runs on AKS and AWS on EKS. (RunPod/Brev/Railway were removed 2026-07-22.)
Configure the matching provider variables, validate identity, capacity, image,
storage, and budget preflight, then choose the sandbox explicitly. A submitted
Job is not a successful run: wait for the terminal receipt and authoritative
final report. Durable-controller mode additionally requires its lease/object
storage/PVC prerequisites.

## Where output goes

Runs write to `runs/<project-id>/`; that directory is intentionally ignored.
Keep only reviewed, durable evidence in `best_runs/`. See
[policies/artifacts.md](policies/artifacts.md).
