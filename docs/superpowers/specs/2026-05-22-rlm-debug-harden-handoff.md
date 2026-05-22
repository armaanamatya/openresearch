# RLM Phase 5/6 — Debug & Harden Handoff

Continuation handoff for OpenResearch / ReproLab. The "do 3 and 1" deliverable
(5 papers reproduced end-to-end + leaf-scored, REST-retrievable) is **done**.
This handoff frames the next session as a focused **debug-and-harden pass**:
catalogue every known issue and fix each with a root-level, elegant solution.
Self-contained. Written 2026-05-22.

---

## Kickoff prompt — paste into the new session

> `/iterate` — Debug-and-harden pass for OpenResearch. Read this doc in full
> first; it carries the standing instructions, current state, and the issue
> catalogue. Then read `progress.md`, `learn.md`, `system_overview.md`, and
> `docs/design/rlm-pivot-brief.md`. Work the issue catalogue in §4 in priority
> order (P0 → P2). Every fix must be **root-level and elegant** — fix the
> invariant with one canonical abstraction plus a guard test, never a scattered
> patch. Delegate implementation to Sonnet sub-agents against tight specs; use
> Codex for review / adversarial review / giant-solution diagnosis only, never
> to write implementation code. Fan out parallel sub-agents for independent
> (disjoint-file) work. Commit sparingly at milestones; verify every diff; ask
> me when a fork is genuine.

---

## 1. Standing instructions (carry forward verbatim)

**Process.** `/iterate` is the working mode — PhD-researcher + principal-engineer
rigor, Karpathy's four anti-pattern guardrails. **Root-level elegant solutions**:
fix the invariant with one canonical abstraction + a guard test, never scattered
patches. Quality bar: *precise, accurate, 100% reliable, elegant* — this is a
demo for employers.

**Delegation.** *Opus* owns design, contracts, numerical correctness, and
**reviewing every diff** (the diff, never the agent's summary). Delegate
implementation to **Sonnet** sub-agents against a *tight* spec (exact files,
exact strings, acceptance command, what-not-to-touch). Use **Codex**
(`codex:codex-rescue`) for review / adversarial review / giant-solution
diagnosis / planning refactors — **never to write implementation code**. **Fan
out parallel sub-agents** for independent work (disjoint file sets).

**Commits & git.** Commit **infrequently** — one commit per *major milestone*,
not per fix. **Never** a `Co-Authored-By` / AI-attribution trailer. Work on
branch `merge`; fast-forward `main` to it (elegant linear merge, no merge
commit); push both to `origin` (armaanamatya/openresearch) — **never** the
`replix` remote. `gh` is not authenticated; `git push` over SSH works.

**Documentation.** Keep `progress.md` current, succinct, no bloat. `learn.md` —
a post-mortem (Symptom → Root cause → Fix → Lesson → Guardrail) per non-obvious
bug, append-only. `CHANGELOG.md` `[Unreleased]`. Keep `system_overview.md` /
`CLAUDE.md` in sync on architectural drift.

**When in doubt — ask.** Surface forks with concrete options. Evidence before
assertions: run the verification, read the output, *then* claim done.

**Operational facts (learned 2026-05-21/22).** RLM runs are **serial** —
Featherless `feather_pro_plus` caps at 4 concurrent units and one RLM run
saturates it; a concurrent run 429s. Re-running a paper needs **both** stores
purged — the run dir *and* the `event_store_events` aggregates in `reprolab.db`
(`rm -rf` of the dir alone is not enough; see I5).

---

## 2. State — what is done

`merge` == `main` == `091d2c6`, pushed to origin. Full suite **1179 passing**.
Five papers run end-to-end and leaf-scored (the authoritative PaperBench score):

| Paper | Rubric | Leaf score | Leaves |
|---|---|--:|--:|
| sequential-neural-score-estimation | PaperBench bundle | 0.404 | 92/92 |
| mechanistic-understanding | PaperBench bundle | 0.286 | 96/96 |
| ftrl | PaperBench bundle | 0.000 | 178/178 |
| Recursive Language Models (arXiv 2512.24601) | self-generated | 0.325 | 21/21 |
| Minimal Circuits for IOI (arXiv 2510.25013) | self-generated | 0.363 | 23/23 |

Shipped (commits `43af7db..091d2c6`): dynamic best-source ingestion
(`ResolvingParser` — HTML > PDF > OCR, quality-gated); self-generated arXiv
rubrics (`rubric_gen.generate_rubric_tree`); REST-retrievable RLM runs
(`run_pipeline_rlm` writes `demo_status.json`, sources `parsed_full_text.txt`);
honest `final_report.md` re-render on leaf-score.

---

## 3. The bar

Fix the issue catalogue below. Each fix is **root-level** (one canonical
abstraction, not a patch) and ships with a **guard test** whose docstring names
the symptom. The system is a demo for employers — every output must be precise,
accurate, and honest.

---

## 4. Issue catalogue — debug & fix everything

### P0 — correctness / honesty

**I1 — `run_experiment` never extracts metrics.** `_execute_in_sandbox`
(`backend/agents/rlm/primitives.py`) hardcodes `metrics: {}`. Every run reports
`metrics: {}` and can never earn a `reproduced` verdict backed by measured
numbers; every `reproduction_summary` says "metrics: {}". *Root fix:* after the
experiment, read the `metrics.json` the `SANDBOX_EXECUTION_CONTRACT` pins
(`runs/<id>/code/outputs/metrics.json`), thread it into the result, and let the
verdict logic use it.

**I2 — the in-loop `verify_against_rubric` is degenerate against a tree rubric.**
`verify_against_rubric` (`primitives.py:559`) reads `rubric.get("areas", [])`,
but `rubric_spec` is a PaperBench *tree* (`{id,requirements,weight,sub_tasks}`)
for *both* bundle and generated rubrics — so `areas` is always empty, the
in-loop score is meaningless, and the root gets no real rubric feedback during
the run. *Root fix:* `verify_against_rubric` should consume the tree (flatten
leaves → grade → weighted roll-up — reuse `backend/evals/paperbench/leaf_scorer`)
so the in-loop score and the post-run leaf score are the *same* computation.

**I3 — `ftrl` reproduced the wrong method.** The `ftrl` run wrote a 13 KB
`train.py` but its summary says "the RLM system was implemented" — leaf score
0.000. Either `third_party/paperbench/ftrl/paper.md` is wrong/confusing, or the
root mis-grounded. *Investigate:* confirm `paper.md` is the real FTRL paper; if
so, harden the root prompt's paper-grounding so the root reproduces the paper in
`context`, not the substrate it runs on.

**I4 — the index → workspace `paper_text` variable loses content.** For some
papers the workspace `paper_text` variable (reassembled from indexed chunks:
parser sections → indexer → chunker → workspace) is degraded/empty — the IOI
paper's was. RLM now bypasses it (reads `parsed_full_text.txt`); **SDK mode
still consumes it**. *Investigate* the indexer/chunker/workspace path — the
variable should equal the parser's `full_text`.

### P1 — robustness

**I5 — re-running a paper conflicts.** `rm -rf runs/<id>` does not clear the
`event_store_events` aggregates (`<id>`, `<id>:parsed`, `<id>:index`,
`<id>:discovery`) in `reprolab.db` → `ConcurrencyError` on re-run. *Root fix:* a
`reproduce --fresh` flag (or a purge helper) that resets both stores atomically.

**I6 — concurrent RLM runs 429 and fail hard.** The Featherless 4-unit cap plus
no 429-backoff in the LLM clients → a concurrent run's `generate_rubric_tree`
(and possibly `rlm.completion`) fails outright. *Root fix:* 429-aware
retry-with-backoff in the LLM clients so concurrent runs *wait* for a slot
instead of failing — then 2 runs can share the cap and the pipeline parallelises.

**I7 — `--sandbox` is a no-op for RLM `run_experiment`.** It hardcodes
`LocalDockerBackend`. *Root fix:* thread the resolved `sandbox_mode` through to
the primitive.

**I8 — REST `POST /runs/arxiv` two-dir split.** A REST-spawned RLM run writes
artifacts to `runs/prj_<hash>/` while the API watches `runs/ui_rlm_<…>/`;
`live_runs.finalize_benchmark` bridges only `final_report.{json,md}` and reads
SDK-schema fields absent on RLM reports — so events are not bridged and the
`benchmark` summary is empty for RLM. *Root fix:* one run dir, or a schema-aware
bridge that also copies the event log.

**I9 — verdict vs evidence.** Runs self-report `verdict: reproduced` even at
leaf score 0.0 (ftrl). The honesty guard drops fabricated *metrics* but not an
over-claimed *verdict*. *Root fix:* reconcile the verdict against the experiment
evidence / leaf score before the report presents it.

### P2 — smaller

**I10 — OCR is non-functional.** tesseract 5.4.1 is installed but lacks the
`eng` language data; `OcrPaperParser` fails soft and the OCR test skips. Install
the language data so the wired OCR fallback actually runs.

**I11 — `has_provider_credentials`** treats the `claude` CLI on `PATH` as proof
of a valid OAuth session — it is not.

**I12 — `build_environment`'s `ThreadPoolExecutor`** can block past its timeout.

**I13 — `HtmlPaperParser._extract_references` is dead code.** `full_text` is
space-joined (no newlines), so its line-based "References" scan always returns
`[]`. Either extract references from the HTML structure or drop the function.

This catalogue is the *known* set; the next session may surface more — add them
here as found.

---

## 5. Technical reference

**Key files.** `backend/agents/rlm/{run,primitives,binding,report,rubric_gen,
sse_bridge,models}.py`; `backend/services/ingestion/parser/{resolving_parser,
html_parser,ocr_parser,pymupdf_parser}.py`; `backend/cli.py`
(`_build_workspace_claim_map`, `cmd_reproduce`); `backend/services/events/
live_runs.py` (REST run spawn + `_python_script` + `finalize_benchmark`);
`backend/evals/paperbench/leaf_scorer.py`.

**Run a bundle paper:** `.venv/bin/python scripts/rlm_paperbench.py <paper_id>
--model qwen3-coder-featherless --sandbox docker --gpu-mode prefer
--max-wall-clock 7200` (fresh timestamped run id each time).
**Run an arXiv paper:** `.venv/bin/dotenv run -- .venv/bin/python -m backend.cli
reproduce <arxiv_id> --mode rlm --model qwen3-coder-featherless --sandbox docker
--gpu-mode prefer --max-wall-clock 7200` (deterministic run id — see I5 before a
re-run).
**Leaf-score:** `.venv/bin/python scripts/score_run.py runs/<run_dir> [<paper_id>]`
(omit `paper_id` for an arXiv run — it finds `generated_rubric.json`).
**Tests:** `.venv/bin/python -m pytest tests/ -q`.

**Models.** Root + sub-calls: Featherless `qwen3-coder-featherless`
(`FEATHERLESS_API_KEY`). `implement_baseline` sub-agent: Claude Sonnet
(`ANTHROPIC_API_KEY`, SDK-resolved). `.env` is loaded by `rlm_paperbench.py` and
`score_run.py` themselves; `backend.cli` is **not** dotenv-aware — wrap it in
`dotenv run` (or export the env).

## 6. Read before non-trivial work

`progress.md` · `learn.md` (post-mortems — read before touching a covered path)
· `system_overview.md` · `CLAUDE.md` · `docs/design/rlm-pivot-brief.md` · the
prior handoff `2026-05-21-rlm-phase5-session3-handoff.md` · the arXiv-e2e plan
`2026-05-21-rlm-phase5-arxiv-e2e-plan.md`.
