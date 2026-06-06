<!-- doc-meta: status=current; last-verified=2026-06-03 -->
# Documentation Policy — source of truth & freshness

> **Doc status:** Current · the policy that governs every other doc · last
> verified 2026-06-03.

This repository has no single `prd.md`. Documentation is layered, and each layer
has a defined authority and a defined freshness obligation. The goal is simple:
**nothing in the repo should silently present stale information as current.**

## 1. Source-of-truth hierarchy

When two documents disagree, the higher tier wins. When a document disagrees
with the code, **the code wins** (and the doc is a bug to be fixed).

| Tier | Source | Authority | Lives in |
|---|---|---|---|
| 0 | **The code** | Ground truth for *what the system does*. Named by function. | `backend/`, `frontend/`, `scripts/` |
| 1 | **`system_overview.md`** | The "why" and how the pieces fit. | repo root |
| 1 | **`docs/design/rlm-pivot-brief.md`** | Canonical RLM-as-orchestrator architecture. | `docs/design/` |
| 1 | **`docs/design/project-rebuild-spec.md`** | The closest thing to a PRD: the *what*/*why*, framework-agnostic. | `docs/design/` |
| 2 | **`CLAUDE.md`** | Day-to-day developer reference: commands, conventions, gotchas, invariants. | repo root |
| 3 | **`README.md`** | Public front door. Must not claim anything tiers 0–2 don't back. | repo root |
| 3 | **`docs/guides/`**, **current `docs/runbooks/`** | Setup, deployment, ops. | `docs/` |
| 4 | **Generated artifacts** | Point-in-time outputs of a run. Never edited by hand. | `best_runs/`, `runs/`, `docs/runbooks/artifacts/` |
| 5 | **Archived / historical** | Frozen. Provenance only. Never current. | `docs/archive/`, dated `docs/runbooks/2026-*`, `docs/superpowers/**` |

There are **no orphan docs**: every current-state doc either is a source of
truth itself or points up the hierarchy to one.

## 2. Document classes & their freshness obligation

Every Markdown file in the repo is exactly one of:

- **Current-state** — describes how the system works *right now*. **Must** carry
  a freshness marker (§3) and is listed in
  [`current-docs.txt`](current-docs.txt). These are the only docs the freshness
  checker holds to a "last verified" date.
- **Dated-historical** — a snapshot whose filename starts with a date
  (`docs/runbooks/2026-05-28-...md`) or which lives under `docs/superpowers/`
  (plans/specs are written-once design records). The date *is* the freshness
  marker. Never retro-edited; superseded work gets a new dated doc.
- **Archived** — moved to `docs/archive/` with an `ARCHIVED` banner. Was once
  current, now frozen. (See `docs/archive/` for engineering journals
  `learn.md`, `uiprogress.md`, etc.)
- **Generated** — emitted by a run (`final_report.md`, `*.jsonl`, rubric JSON).
  Carries the provenance of the run that produced it; never hand-edited.
- **Vendored** — third-party (`third_party/`, `.venv*/`). Out of scope.

## 3. Freshness marker

Current-state docs carry a machine-readable marker as the first line:

```
<!-- doc-meta: status=current; last-verified=YYYY-MM-DD -->
```

and, immediately below the H1, a human-readable status line. **Updating the date
is a claim that you re-verified the doc's factual statements against the code on
that date** — don't bump it without doing so.

Archived docs use `status=archived; archived=YYYY-MM-DD`. Generated docs need no
marker (their run directory is their provenance). A hand-written README that
*describes* generated artifacts (e.g. `best_runs/README.md`) may use
`status=generated-artifact; captured=YYYY-MM-DD` to signal that its contents
reflect a point-in-time run, not the current system.

## 4. What is NOT auto-generated (and why there is no `docs:generate`)

A doc-generation pipeline was deliberately **not** built, because nothing in
this repo is generated-from-source-markdown in a way that rots:

- **PDFs** (`paperbench1.pdf`, `demo_paper.pdf`, `best_runs/adam/code/paper.pdf`)
  are *input fixtures* — published research papers. A paper does not go stale.
- **`best_runs/*/final_report.md`** and `docs/runbooks/artifacts/**` are outputs
  of expensive GPU reproduction runs. "Regenerating" them means re-running a
  reproduction, not running a doc command. They are point-in-time artifacts and
  are labelled as such.
- The README, overview, and guides are **hand-maintained prose**, kept honest by
  the freshness checker (§5), not by codegen.

Building a `docs:generate` command here would be "empty machinery" — see the
audit in `docs/archive/` history. If a future feature *does* produce
generated-from-source docs, add the generator and list its outputs in §1 tier 4.

## 5. Enforcement — `make docs-check`

The freshness/consistency checker is `scripts/docs_freshness_check.py`. Run it
locally before pushing docs changes:

```bash
make docs-check                      # or:
python scripts/docs_freshness_check.py
```

It fails (exit 1) on:

- a **tracked PDF outside** the approved fixture locations;
- a **current-state doc** (in `current-docs.txt`) **missing a freshness marker**;
- a **broken relative link** in any current-state doc or the README;
- a **working-note file** (`progress.md`, `runlog.md`, `learn.md`, …)
  reappearing at the **repo root** (where it would masquerade as current);
- a **README inline-code reference to a missing file** — an extension-bearing
  repo path like `` `scripts/foo.py` `` (bare directory references are not
  checked, being too prose-ambiguous).

It warns (exit 0) on a current-state doc whose `last-verified` date is older than
90 days — a nudge to re-verify, not a hard gate.

CI: a GitHub Actions workflow (`.github/workflows/docs-freshness.yml`) runs the
same check on every PR. It is intentionally the *only* docs gate — cheap, and it
matches the local command exactly.

## 6. Contributor rules (how to not create stale docs)

1. **New current-state doc?** Add the freshness marker and add its path to
   `current-docs.txt`.
2. **Describing a session's progress / a run?** That is a *journal*, not a doc.
   It belongs in `git log`, a `docs/superpowers/specs/` design record, or
   (if it must be a file) a **dated** file under `docs/runbooks/` — never a bare
   `progress.md` at root.
3. **Superseding a doc?** Move the old one to `docs/archive/` with an `ARCHIVED`
   banner and fix inbound links — don't leave two "current" docs disagreeing.
4. **Touched code that a current-state doc describes?** Update the doc and bump
   its `last-verified` in the same change (the "Maintaining this doc" rule that
   `system_overview.md` and `CLAUDE.md` already state).
5. **Run `make docs-check`** before pushing.
