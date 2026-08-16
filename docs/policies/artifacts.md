<!-- doc-meta: status=current; last-verified=2026-07-20 -->
# Artifact retention policy

`runs/` is local operational state. It is ignored by default; a run may be
large, incomplete, credential-adjacent, and expensive to review. Never add a
whole run directory as a convenience.

`best_runs/` is the curated evidence set. A new entry must contain a concise
README, its final report, a reproducibility/provenance record, and only the
small inputs or code snapshots necessary to understand the result. Large model
weights, environments, raw event streams, and transient logs belong outside
Git (object storage or a release artifact).

The existing `best_runs/` history is frozen pending a separately reviewed
curation pass. It is intentionally not deleted or rewritten by this policy:
those files may be evidence for past results. New evidence must follow the
rules above.

To preserve an exceptional small file from `runs/`, use an explicit
`git add -f` and state why it is durable evidence in the commit. This makes
the exceptional retention decision visible in review.
