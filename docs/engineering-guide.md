<!-- doc-meta: status=current; last-verified=2026-07-20 -->
# Engineering guide

This is the compressed replacement for the old plans, handoffs, and incident
logs. Read it after [architecture.md](architecture.md). For dated context, use
the concise [development timeline](periods/) rather than a raw historical log.

## What we are building

OpenResearch is a single-user research-reproduction tool. It ingests a paper,
creates a reproduction, executes experiments, checks evidence against a rubric,
and produces a report. The goal is trustworthy evidence, not a fluent-looking
report or a maximized demo score.

## Non-negotiable rules

1. **Code and observed artifacts win.** A prose claim, an LLM grade, or a UI
   value never overrides an observed metric, terminal receipt, or deterministic
   verdict.
2. **A run is file-backed.** `runs/<project-id>/` is the source of operational
   state; API/UI processes may restart, but run files and final reports must
   remain interpretable.
3. **Default-off safety is not deployment.** A new guard needs a deliberate
   enabled-profile decision, tests for the off path, and evidence before it
   becomes a default.
4. **Keep one implementation of a trust predicate.** Higher-level callers
   delegate to it; duplicated graders or validators drift.
5. **Never convert missing or non-finite evidence into success.** Preserve
   failures, partial results, and provenance explicitly.

## Development model

- Start with `make setup`, `make dev`, `make smoke`, then `make check`.
- Make a narrow change, add or update its test, and keep unrelated cleanup out
  of the same change.
- Use the existing nested `CLAUDE.md` nearest to the code you touch for local
  conventions. Root `CLAUDE.md` is only the cross-project contract.
- New runtime behavior must have a clear off state, explicit configuration, and
  an observable result or receipt.

## Execution and cloud posture

| Surface | Use it for | Status |
|---|---|---|
| `local` | development and CPU/local-GPU experiments | supported |
| `docker` | isolated local execution | supported when Docker is available |
| `runpod` | remote GPU execution | supported with explicit credentials |
| `gcp` / `gke` | Kubernetes cell execution | operator-gated; preflight first |
| `azure` | AKS cell execution | operator-gated; preflight first |
| `aws` / `eks` | EKS cell execution | experimental |

Cloud availability is an operational fact, not a code fact. Never bill a run
until credentials, image, storage, quota, and the selected execution route have
been checked.

## Evidence model

- A reproduction claim needs scoped metrics, a concrete artifact path, and a
  final report that agrees with the authoritative verdict.
- Preserve partial, timed-out, capacity-limited, and ungraded states. They are
  information—not zeroes to overwrite or successes to infer.
- Cost ledgers are incomplete on some providers. Review token totals and cloud
  resource state before declaring a run cheap or free.
- Curate only reviewed evidence into `best_runs/`; normal run output stays in
  ignored `runs/`.

## Configuration model

- `.env.example` documents local defaults; secrets and operator settings stay
  in `.env` or the cloud secret mechanism.
- `configs/` contains versioned execution and paper-specific configuration.
  Paper overrides live in `configs/papers/`, not under documentation.
- `docs/reference/flags.md` is generated reference. Do not use it as a reason
  to enable a flag; inspect the owning code and run profile first.

## When something fails

Classify the failure before retrying: credentials, capacity, environment,
implementation, metric/evidence, or a transient service failure. Retry only
the transient class. For everything else, surface the failure in the run
artifact and fix the owning boundary.

If this guide becomes insufficient, improve this guide or the nearest current
document—do not create another dated handoff, scratch log, or plan tree.
