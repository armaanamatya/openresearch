# RLM Phase 5 — Progress

_Updated: 2026-05-22_

## Objective

Drive the RLM orchestrator end-to-end on real papers — deliver papers with
real rubric scores in `runs/<id>/final_report.json`, retrievable via the REST
API. ("do 3 and 1": 3 PaperBench bundle papers + 2 recent arXiv papers.)

## Status

`merge`: full test suite **1179 passing**. Deliverable met — **5 papers run
end-to-end and leaf-scored**, each retrievable through the REST API
(`GET /runs/{id}` + `/final-report`).

## Results

| Paper | Rubric | Leaf score | Leaves |
|-------|--------|-----------:|-------:|
| sequential-neural-score-estimation | PaperBench bundle | 0.404 | 92/92 |
| mechanistic-understanding (DPO / toxicity) | PaperBench bundle | 0.286 | 96/96 |
| ftrl | PaperBench bundle | 0.000 | 178/178 |
| Recursive Language Models — arXiv 2512.24601 | self-generated | 0.325 | 21/21 |
| Minimal Circuits for IOI — arXiv 2510.25013 | self-generated | 0.363 | 23/23 |

The leaf scorer is the authoritative PaperBench score (flatten → batched LLM
grading → weighted roll-up). Bundle scores grade against the vendored
`rubric.json`; arXiv scores grade against a self-generated rubric and are
labelled `rubric_source="generated"` — **not** PaperBench-official.
`ftrl` scored 0.0 honestly: the run self-reported `reproduced` but the leaf
scorer graded the produced code 0.0 — its summary says "the RLM system was
implemented", i.e. the root reproduced the wrong method.

## Shipped this session

**Dynamic best-source ingestion** (`ResolvingParser`, HTML > PDF > OCR,
quality-gated). `ArxivFetcher` fail-soft-fetches arXiv's clean LaTeXML HTML;
figure-heavy papers PDF-parse to figure-label noise, so HTML wins. OCR
(tesseract) is the scanned-PDF fallback.

**Self-generated rubric for arXiv runs.** `rubric_gen.generate_rubric_tree`
derives a PaperBench-shaped weighted rubric tree from a paper that has no
vendored bundle; persisted to `generated_rubric.json`; `score_run.py` finds it.

**arXiv RLM runs are REST-retrievable.** `run_pipeline_rlm` writes
`demo_status.json`; RLM context is sourced from `parsed_full_text.txt` (the
parser's clean, complete output) un-truncated, not the chunk-reassembled
workspace variable. `amend_final_report` re-renders `final_report.md` so the
REST-served report shows the authoritative leaf score with honest provenance.

## Known gaps (carry forward)

- **`run_experiment` does not extract metrics** (`metrics: {}`) — a `reproduced`
  verdict with measured numbers needs reading `metrics.json` from the run's
  outputs. The leaf score is unaffected (it grades artifacts, not metrics).
- **RLM runs are serial** — the Featherless plan caps at 4 concurrent units and
  one run saturates it; a concurrent run 429s.
- **Re-running a paper** needs its event-store aggregates purged — `rm -rf` of
  the run dir does not clear `reprolab.db` (see `learn.md`).
- The index→workspace `paper_text` variable loses content for some papers; RLM
  now bypasses it (reads `parsed_full_text.txt`), SDK mode still uses it.
- `--sandbox` is a no-op for RLM `run_experiment` (hardcodes `LocalDockerBackend`).
