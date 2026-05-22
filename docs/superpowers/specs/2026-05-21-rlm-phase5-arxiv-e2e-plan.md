# RLM Phase 5 — arXiv end-to-end run plan

Refines the session-3 handoff's "do 3 and 1" §5 with the user's 2026-05-21
direction: **do the two arXiv papers first**, end-to-end, with results
**retrievable via the REST API**. Self-generated rubrics, honestly labelled.

## Scope decision

The handoff's task 3 ("build LLM rubric generation") assumed the arXiv RLM path
already feeds the root a real paper. It does not — recon found two gaps:

1. **Truncated corpus.** `cli.py` builds one `workspace_claim_map` for all run
   modes and truncates every entry to 600 chars. Correct for SDK/offline (the
   paper goes into the agent prompt); **wrong for RLM** — the paper is offloaded
   whole into the REPL `context` variable, never a prompt. The workspace already
   preloads a full `paper_text` variable; RLM mode must carry it un-truncated.
   (`rlm/run.py::_build_context` even notes the full corpus is "#62 work".)
2. **No rubric.** arXiv runs set no `rubric_spec`. The existing generated-rubric
   path (`RUBRIC_VERIFIER_PROMPT` Phase 1) emits a flat `{area,weight}` list —
   not the `{id,requirements,weight,sub_tasks}` tree `leaf_scorer` scores.

## Phase A — code (no Docker)

- **A1** `cli.py`: RLM mode carries the full `paper_text` workspace variable
  into the claim map, un-truncated. SDK/offline truncation untouched.
- **A2** `backend/agents/rlm/rubric_gen.py`: `generate_rubric_tree(paper_text,
  llm_client, ...)` → PaperBench-shaped tree. The LLM proposes categories +
  leaves; code assigns ids, normalises weights, enforces leaf invariants;
  bounded retry; fail-soft to `None` with a loud WARNING.
- **A3** `rlm/run.py`: when `rubric_spec` is absent, generate from `paper_text`,
  inject into `context['rubric_spec']`, persist `runs/<id>/generated_rubric.json`.
- **A4** `score_run.py` + `leaf_scorer`: find the persisted generated rubric
  when no bundle; label `rubric_source="generated"`.
- **A5** tests for A1–A4; full suite green.

## Phase B — verify the REST path

`POST /runs/arxiv {mode:"rlm"}` → subprocess → `final_report.{json,md}` +
`demo_status.json` + `dashboard_events.jsonl` → `GET /runs/{id}` /
`/runs/{id}/final-report` / `/runs/{id}/events`. Fix any gap that stops an RLM
arXiv run from being retrievable through the API.

## Phase C — runs (Docker, ~2h each, SERIAL)

> Featherless `feather_pro_plus` caps at 4 concurrent units; one RLM run
> saturates it, so a second concurrent run 429s. Runs are serial, not the
> handoff's "2 concurrent" (observed + corrected 2026-05-21).


- **C1** RLM paper arXiv `2512.24601`.
- **C2** one more recent (~6-month) paper — candidates proposed, user picks.
- **C3** leaf-score both; confirm the REST API serves real scores; label the
  scores self-generated, **not** PaperBench-official.

## Phase D — follow-up

Bundle re-runs (`mechanistic-understanding`, `ftrl`) — handoff Part 3.

## Constraints

- Finish and commit Phase A **before** starting the backend for Phase C —
  `uvicorn --reload` kills in-flight runs (learn.md #8). Run the backend
  **without** `--reload` during Phase C.
- Self-generated-rubric scores are NOT PaperBench-official — label as such.
- Codex: review / adversarial diagnosis only, never implementation. Sonnet
  sub-agents implement against tight specs; Opus designs and reviews every diff.
