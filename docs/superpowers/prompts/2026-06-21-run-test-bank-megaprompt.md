# MEGA-PROMPT — A durable, shared BANK of every reproduction run, test, output, and log (for all contributors)

> **What this is.** A single self-contained brief for an autonomous engineering
> session to design and build a **Run/Test Bank** for OpenResearch: one durable,
> searchable, cross-contributor archive of **every reproduction run / experiment /
> test ever produced** — its outputs (reports, metrics, scored rubrics), its logs
> (SSE firehose, experiment runs, training logs), and its full provenance (who ran
> it, on what machine, which git SHA, which flags, what it cost, what it scored) —
> so the whole team can **learn from the past, compare arms, mine what worked,
> avoid repeating failures, and showcase results**, instead of run history living
> and dying on individual laptops or bloating git.
>
> **Why now (the two forcing functions):**
> 1. **Run history is being lost AND bloating git at the same time.** Live runs in
>    `runs/<project_id>/` are gitignored and per-machine — one person's SDAR
>    attempts vanish when their VM is reclaimed. Meanwhile `best_runs/` (the only
>    durable cross-person store) is **12 MB / 432 files committed to git** — full
>    run trees, raw training logs, and SSE firehoses that escaped the `runs/`
>    whitelist policy. We're simultaneously *losing* most history and *bloating*
>    the repo with a hand-picked slice of it. The Bank fixes both.
> 2. **The North Star needs it.** The SDAR score (the real goal) is driven up by
>    *mining what worked across everyone's attempts*. Today the positive-recipe and
>    negative-lesson memory only sees the local machine's re-runs of one paper. A
>    cross-contributor Bank is the substrate that lets recipe/lesson mining learn
>    from the whole team's history.
>
> **Non-negotiables (read first):**
> - **Zero unauthorized spend.** No paid object-store / cloud provisioning, no live
>   GPU, no network egress in tests. Hermetic tests only (the suite is
>   socket-hermetic via pytest-socket). A real remote store is operator-gated and
>   labelled as costing money; build against a local filesystem store by default.
> - **Default-OFF + fail-soft.** Every new ingestion hook / flag is opt-in; unset ⇒
>   byte-for-byte identical behaviour today. The bank never blocks or slows a run.
> - **Privacy + secrets.** "For all people" means PII and credentials WILL be in the
>   raw artifacts (`user_messages.jsonl`, env in logs). Reuse the existing egress
>   sanitizer (`sse_bridge.sanitize_iteration`, `redact_corpus`) and secret-redaction
>   on ingest. The paper corpus and secrets never enter the Bank.
> - **TDD, merges-only on shared branches, no force-push to others' branches.**
> - **Don't break the leaderboard.** It is the existing read surface; extend it,
>   don't fork it.

---

## 0. Orientation — what already exists (build ON this, do not reinvent)

Verify each with the code before designing; true at authoring (2026-06-21).

**The raw material — per-run artifacts in `runs/<project_id>/`** (file-backed run
state, CLAUDE.md "File-backed run state"):
- `final_report.{json,md}` — benchmark output; carries `mode`, `models`, `started_at`,
  `completed_at`, `experiment_arm` (A/B stamp), `stop_reason`, `reproducibility`.
- `demo_status.json` — UI status snapshot · `cost_ledger.jsonl` — per-primitive USD ·
  `experiment_runs.jsonl` — every `run_experiment` result (logs, success, metrics) ·
  `dashboard_events.jsonl` — the **SSE firehose** (large) · `rlm_state/` — per-iteration
  checkpoints incl. `bes_candidates.json`, `leaf_triage.json`, `gpu_plan.json`,
  `context_map.json`, `champions.json`, `repro_spec.json` · `code/` — the reproduced
  project + `code/metrics.json` (+ `code/outputs/<run_id>/metrics.json`) ·
  `generated_rubric.json`, `rubric_evaluation.json`, `rubric_tree.json` · Hermes audit chain.
- A/B + memory dirs: `runs/_ab/<key>/ab_report.{md,json}` (paired A/B) ·
  `runs/_lessons/<arxiv_id>.json` (per-paper negative lessons).

**The curated showcase — `best_runs/<paper>/`** (adam, allcnn, resnet, vae, sdar,
omnizip + `*_ab` arms; **12 MB / 432 files, tracked in git** — the bloat to migrate).

**Existing machinery to reuse, not rebuild:**
- `backend/services/runs/report_resolution.py` — `resolve_best_report(run_dir)`,
  `extract_scores(report)`, `normalized_score`, `two_axis_fidelity_key`,
  `ResolvedReport`. Canonical best-attempt + score-schema resolution (handles nested
  `rubric.overall_score` / `compute_adjusted_score` + legacy flat `rubric_score`).
- `backend/routes/leaderboard.py` — `aggregate_leaderboard`, `_read_run`,
  `GET /leaderboard?paper&mode&order_by&limit`. **Request-time** aggregation of
  `final_report.json` across the LOCAL `runs/` dir only — the read surface to extend
  to the (cross-contributor) Bank.
- `backend/services/runs/{archive,attempt_isolation,retention}.py` — per-attempt
  archival (`attempt_isolation._ARCHIVE_FILES`), retention daemon, `purge.py` (note:
  superseded by `retention.py` per the cleanup audit — fold in, don't depend on purge).
- `backend/agents/rlm/recipe_library.py` (positive recipes) + `lesson_distiller.py`
  (negative lessons) — the memory miners the Bank should feed cross-contributor.
- SQLite CQRS event store (`OPENRESEARCH_DATABASE_URL`, default `sqlite:///openresearch.db`).
- Egress sanitizers: `sse_bridge.sanitize_iteration`, `redact_corpus`.

**The gaps the Bank closes:** (a) no cross-contributor pooling (runs are local/
gitignored); (b) durable store is git-committed bloat (`best_runs/`); (c) no
systematic, compressed, deduped log capture; (d) no unified cross-time/cross-person
query ("all SDAR attempts by anyone, by score, with failure-class + flags"); (e) no
consistent provenance/attribution; (f) no Bank retention/dedup policy.

---

## 1. Architecture — the Bank in five units (each independently testable)

```
  run finalize ──(opt-in hook)──▶  [1] INGEST  ──▶  [2] STORE (content-addressed)
   runs/<id>/                         normalize         local FS default
   best_runs/<paper>/                 + redact          remote (operator-gated)
                                      + provenance              │
                                      + compress/sample logs    ▼
                                                         [3] INDEX (sqlite catalog)
                                                                │
        [5] MEMORY  ◀── feeds ──┐                               ▼
   recipe/lesson mining          └────────────────────  [4] QUERY / READ
   across ALL contributors            CLI: bank ls/show/compare/diff
                                       UI: leaderboard + replay read the Bank
```

### [1] Ingest — `bank ingest <run_dir>` (and an opt-in finalize hook)
Normalize a `runs/<id>` or `best_runs/<paper>/<attempt>` into a **Run Bundle**.
- **Provenance manifest** (`manifest.json`): `bundle_id` (content hash), `paper_id`,
  `mode`, `experiment_arm`, `models` (per-role), `flags` (the `OPENRESEARCH_*` env
  snapshot that shaped the run — minus secrets), `git_sha`, `contributor` (git
  user.email / `$USER` / explicit `--as`), `machine` (hostname, sandbox, GPU SKU),
  `started_at`/`completed_at`, `wall_clock_s`, `cost_usd`, `overall_score`,
  `target_score`, `meets_target`, `verdict`, `stop_reason`, `failure_class`,
  `rubric_tree_sha`. Reuse `report_resolution.extract_scores` + the `experiment_arm`
  stamp; do not re-derive score schemas.
- **Redaction on ingest** (non-negotiable): strip secrets (`*_API_KEY`, tokens) from
  the flag snapshot and logs; run `user_messages.jsonl` + any corpus-derived text
  through `redact_corpus`/`sanitize_iteration`. The Bank must be shareable.
- **Log handling**: keep `final_report.*`, `metrics.json`, `rubric_evaluation.json`,
  `cost_ledger.jsonl`, `experiment_runs.jsonl` verbatim (small/structured). For the
  large firehose (`dashboard_events.jsonl`) + raw training logs: **compress + sample**
  (head/tail + every error/warning line + a configurable max size); store the full
  compressed blob in the object store, a sampled excerpt in the bundle.
- Idempotent (content hash ⇒ re-ingest is a no-op), fail-soft (never crashes a run),
  default-OFF as an auto-hook (`OPENRESEARCH_BANK_AUTO_INGEST`, off).

### [2] Store — content-addressed bundle store (NOT git)
- Default backend: local filesystem at `OPENRESEARCH_BANK_DIR` (default `~/.openresearch/bank/`
  or a configured shared NFS path) — bundles keyed by content hash, logs gz-compressed,
  identical artifacts deduped (the audit found exact-duplicate `metrics.json` across one
  run tree — content-addressing dedupes these for free).
- Pluggable remote backend (S3 / GCS / Azure Blob) behind one interface
  (`BankStore.put/get/list`), **operator-gated** (real bucket = real money/credentials);
  tests run entirely against the local FS store. Mirror the existing
  `runtime/foundry_endpoint`-style single-resolver pattern for credentials.
- The **canonical INDEX is small and versioned/shared** (see [3]); the heavy bundles
  live in the store, never in git. This is the explicit anti-pattern fix for `best_runs/`.

### [3] Index — a queryable catalog (`bank.sqlite` + a flat `index.jsonl` mirror)
- One row per bundle with the manifest fields above, plus per-leaf scores if cheap.
- Rebuildable from the store (the store is the source of truth; the index is a
  projection — same CQRS spirit as the run event store).
- The flat `index.jsonl` mirror is committable (tiny) so the catalog itself is
  shareable/diffable in git while the bundles are not.

### [4] Query / Read — CLI + UI
- CLI: `bank ls [--paper --contributor --arm --since --failure-class --min-score]`,
  `bank show <bundle_id>`, `bank compare <a> <b>` (diff two runs' scores/flags/metrics),
  `bank diff-arms <pair_id>` (A/B), `bank logs <bundle_id> [--grep]` (fetch-on-demand
  from the store).
- UI: extend the **existing** leaderboard + recent-runs + replay surfaces
  (`frontend/src/app/leaderboard/`, `recent-runs-panel.tsx`) to read the Bank
  (cross-contributor) in addition to local `runs/`. `aggregate_leaderboard` grows a
  Bank source; per-run detail fetches logs from the store on demand (never bundling
  the firehose into the page). Add contributor / arm / failure-class facets.

### [5] Memory integration — the payoff for the North Star
- Point `recipe_library` (positive recipes) + `lesson_distiller` (negative lessons) at
  the **Bank** as their cross-contributor corpus, not just local same-paper re-runs.
  Now "what worked on SDAR" is mined from *everyone's* attempts. Keep the existing
  evidence-admission red line (recipes admitted only on Tier-1 deterministic + validator
  evidence, never an LLM grade).

---

## 2. Migration — resolve the `best_runs/` git bloat through the Bank
1. `bank ingest best_runs/**` → every showcase run becomes a bundle (provenance backfilled:
   contributor from git blame/commit author, dates from `final_report`).
2. Remove `best_runs/` from git tracking; add it to `.gitignore` (the audit confirmed the
   `.gitignore` is otherwise excellent and `best_runs/` is the lone gap). Replace it with a
   small `best_runs/README.md` pointer + the committable `index.jsonl` showcase subset.
3. Backfill historical `runs/` on contributors' machines via `bank ingest runs/*` (opt-in).
4. Net effect: 12 MB of run trees leave git, become a queryable cross-person Bank, and
   nothing is lost.

---

## 3. Data model (the Run Bundle — make it self-describing)

```
bundle/<bundle_id>/
  manifest.json        # provenance + headline scores (the indexed row; see [1])
  report/final_report.json, final_report.md
  scored/rubric_evaluation.json, rubric_tree.json   # + rubric_tree_sha in manifest
  metrics/metrics.json, per_cell/*.json
  ledger/cost_ledger.jsonl, experiment_runs.jsonl
  logs/dashboard_events.sample.jsonl        # sampled; full blob in store as .gz
  logs/training/*.log.gz                     # compressed
  memory/lessons.json, recipes.json          # deltas this run produced
  ATTRIBUTION                                # contributor, machine, git_sha, consent
```
Versioned schema (`bundle_schema_version`) so the format can evolve without breaking
old bundles.

---

## 4. Execution methodology
Use the superpowers workflow: **brainstorming** (converge any under-specified piece —
e.g. the exact remote-store choice, the contributor-identity source, the sampling
policy) → **writing-plans** (one bite-sized TDD plan per unit [1]–[5] + the migration)
→ **subagent-driven-development** (fresh subagent per task, two-stage review). Parallel
agents only for independent units (using-git-worktrees for isolation). Gate every step
on the **full hermetic suite**, and add the two currently-missing tests the audit
flagged in the same neighbourhood (env-bridge, db-default) if you touch config.

---

## 5. Phased plan
- **Phase 0 — Bundle schema + local FS store + ingest (one run).** `bank ingest <run_dir>`
  produces a redacted, content-addressed bundle on the local FS store. TDD with a
  synthetic `runs/<id>` fixture; zero network.
- **Phase 1 — Index + CLI query.** Catalog + `bank ls/show`; rebuild-from-store.
- **Phase 2 — Migration of `best_runs/`** into the Bank + git de-bloat (the cleanup win).
- **Phase 3 — Leaderboard/UI read the Bank** (cross-contributor), logs fetched on demand.
- **Phase 4 — Opt-in finalize auto-ingest hook** (default-OFF) so new runs self-bank.
- **Phase 5 — Memory cross-contributor mining** (recipes/lessons over the whole Bank).
- **Phase 6 — Remote store backend** (operator-gated; S3/GCS/Azure Blob) + access control.

## 6. Acceptance criteria
- One `bank ingest` turns any `runs/<id>` or `best_runs/<...>` into a redacted,
  content-addressed, deduped bundle with full provenance; idempotent; fail-soft; default-OFF.
- The Bank is **shareable**: no secrets, no PII, no paper corpus in any bundle (tested).
- `bank ls/show/compare` answers cross-contributor questions ("all SDAR attempts by
  anyone, by score, with failure-class + flags") from the index.
- `best_runs/` leaves git (−12 MB), lives in the Bank, nothing lost; `.gitignore` gap closed.
- Leaderboard + replay read the Bank without bundling firehoses into the page.
- Recipe/lesson mining draws on the whole Bank, not just local same-paper re-runs.
- Zero unauthorized spend; remote store strictly operator-gated; full suite green on every PR.

## 7. Guardrails recap
Default-OFF + fail-soft (the Bank never blocks/slows a run) · secrets+PII+corpus never
enter a bundle (reuse the existing sanitizers) · content-addressed store, NOT git ·
extend the leaderboard, don't fork it · reuse `report_resolution`/`experiment_arm`/
`attempt_isolation`, don't re-derive · TDD + full-suite gate · merges-only on shared
branches · remote store + real buckets are operator-gated and labelled as costing money ·
keep `CLAUDE.md` + `system_overview.md` updated (new CLI, new artifacts, new flags).

> **The single sentence:** turn the run history that today either vanishes on a laptop
> or bloats git into ONE durable, redacted, searchable, cross-contributor Bank of every
> reproduction's outputs, logs, and provenance — so the team learns from all of it and
> the North Star (a real, repeatable SDAR score) is mined from everyone's attempts, not
> rebuilt from scratch each time.
