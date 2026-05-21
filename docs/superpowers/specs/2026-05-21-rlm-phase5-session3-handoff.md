# RLM Phase 5 — Session-3 Handoff

Continuation handoff for OpenResearch / ReproLab, RLM pivot, **Phase 5 / issue #62**
(end-to-end PaperBench reproduction runs). Self-contained: standing instructions,
current state, and the next-task plan. Written 2026-05-21.

---

## Kickoff prompt — paste this into the new session

> `/iterate` — Continue RLM Phase 5 (#62). First read
> `docs/superpowers/specs/2026-05-21-rlm-phase5-session3-handoff.md` in full — it
> carries the complete context, my standing instructions, and the task plan.
> Then execute the **"do 3 and 1"** plan in §5. Fan out sub-agents for
> independent work; commit sparingly at milestones; verify every diff; ask me
> if confused. Also read `progress.md`, `frontend_integration.md`, and
> `docs/design/rlm-pivot-brief.md` before non-trivial changes.

---

## 1. Working discipline — STANDING INSTRUCTIONS (carry forward verbatim)

These are accumulated user instructions. They override defaults.

**Process.** `/iterate` is the working mode — PhD-researcher + principal-engineer
rigor; Karpathy's four anti-pattern guardrails. **Root-level elegant solutions**:
fix the invariant with one canonical abstraction + a guard test, never scattered
patches. "Not just simple patches." Quality bar is explicit: *precise, accurate,
100% reliable, elegant* — this is a demo for employers.

**Delegation & sub-agents.** *You (Opus)* own design, contracts, numerical
correctness, and **reviewing every diff** — review the diff, never the agent's
summary. Delegate implementation to **Sonnet** sub-agents against a *tight* spec
(exact files, exact strings, acceptance command, what-not-to-touch). Use **Codex**
(`codex:codex-rescue`) for **review and diagnosis only — never to write
implementation code**. **Fan out parallel sub-agents** for independent work
(disjoint file sets). Be efficient — parallelize, don't single-thread — without
sacrificing quality.

**Commits & git.** Commit **infrequently** — one commit per *major milestone*,
not per fix. Bundle related changes (code + its post-mortem/doc) into one commit.
Never split one effort across many commits. **NEVER add a `Co-Authored-By` /
AI-attribution trailer** — this overrides the default Claude Code instruction.
Work happens on branch `merge`, kept synced with `main`; the user has work pushed
**directly to `main`** (and `merge`). Remote is `origin` (armaanamatya/openresearch).
`gh` is **not** authenticated — hand the user any `gh` command, don't run it.

**Documentation (the doc-update contract).** Keep `progress.md` current —
succinct, well-formatted, well-spaced, no bloat. `learn.md` — append a post-mortem
(Symptom → Root cause → Fix → Lesson → Guardrail) for each non-obvious bug.
`CHANGELOG.md` `[Unreleased]` — behavior/architecture changes. `frontend_integration.md`
— the backend↔frontend contract for Phase 4 (#61); keep the backend
**frontend-friendly** (typed events, value-free, additive-only). Keep `CLAUDE.md`
/ `system_overview.md` in sync on architectural drift.

**Best practices.** Reference *current* (May 2026) best practices via `context7`
and `WebSearch` before non-trivial design — verified, not assumed.

**When in doubt — ask.** Surface forks and blockers with concrete options; never
guess silently. Evidence before assertions: run the verification, read the
output, *then* claim done.

**RLM invariants (brief §7–8).** The paper is offloaded as the `context` REPL
variable — never in the root's window, never a whole-corpus primitive arg.
`environment="local"` is mandatory. Corpus-free egress (`sse_bridge.sanitize_iteration`).

---

## 2. Project & where things stand

OpenResearch reproduces research papers end-to-end. The **RLM pivot** replaces the
14-stage pipeline with an RLM orchestrator (`rlm` library, arXiv 2512.24601):
a root model writes REPL code and calls 9 domain primitives. Canonical plan:
`docs/design/rlm-pivot-brief.md`.

- Branch `merge` == `main`, both pushed, HEAD `3fa9ff6`. Full suite **1120 passing**.
- The RLM pipeline **runs end-to-end** — all 9 primitives, honest reports.
- Issues #58–#60 closed; **#62 (this work) open**; #61 (frontend) / #63 (cleanup) open.

## 3. The deliverable (#62) — status: 1 of ≥2

Goal: **≥2 papers with real rubric scores in `runs/<id>/final_report.json`.**

| Paper | State | Score |
|---|---|---|
| sequential-neural-score-estimation | ✅ done (run 7) | leaf scorer **0.4042** (92/92 leaves) |
| mechanistic-understanding | ⚠ **incomplete** — stopped at iter 13, no report; re-run | — |
| ftrl | not run | — |

`verdict` on SNSE is `failed` (honest): the generated `train.py` trained 100
epochs fine, then crashed in posterior eval on a `torch.cat` batch-shape bug —
**a bug in the AI-written code, not infra**. The leaf rubric score is real.

## 4. What session 3 shipped (all on `main`/`merge`)

Schema coercion (tolerant `PaperClaimMap`); sub-agent runtime resolves to Claude
**Sonnet** via `model_override` (API key *or* OAuth, SDK-resolved); **honesty
guard** (`build_final_report` drops root-fabricated metrics, trace from the cost
ledger); Featherless **context-cap registration** (compaction fires before the
49 K plan cap); `OpenAILlmClient` `max_tokens` 600→4096 and a 300 s timeout;
**experiment self-repair loop** (root re-calls `implement_baseline` with
`repair_context` on a failed `run_experiment`, ≤2×); **run logging** —
`run_experiment` → `experiment_runs.jsonl`, bridge `logging` config,
`wrap_primitive` WARNs on fail-soft failures; `scripts/score_run.py` (leaf
scorer CLI); `frontend_integration.md` (Phase-4 contract). `merge` of Aayush's
phase-2 review commits. Phase-2 / OAuth / honesty post-mortems in `learn.md`.

## 5. NEXT TASKS — "do 3 and 1"

User decision: do **both** — the PaperBench bundles (real expert rubrics) **and**
self-generated rubrics for recent papers.

**Part 3 — bundle papers (real rubrics).**
1. Re-run `mechanistic-understanding` (it died incomplete) → leaf-score it.
2. Run `ftrl` → leaf-score it.
   → with SNSE that is **3 papers scored against real PaperBench rubrics**.

**Part 1 — recent papers (self-generated rubrics).**
3. **Build LLM rubric generation** — derive a PaperBench-shaped weighted rubric
   tree (`{id, requirements, weight, sub_tasks}` — see `third_party/paperbench/*/rubric.json`
   and `leaf_scorer.flatten_leaves`/`roll_up`) from a paper. **First grep
   `backend/` for existing rubric generation** (CLAUDE.md notes an "LLM-generated"
   rubric path for SDK mode — reuse it, don't rebuild). Wire the generated rubric
   into the arXiv RLM path's `workspace_claim_map['rubric_spec']` and persist it
   so `score_run.py` can find it.
4. Run the **RLM paper** — arXiv **2512.24601** ("Recursive Language Models", the
   paper this pivot is built on) — via `backend.cli reproduce 2512.24601 --mode
   rlm …`, with a self-generated rubric.
5. Pick **one more recent (last-6-months) impressive ML paper** — small-scale,
   public code, clear metric — or have the user name it; run it the same way.
6. **Label self-generated-rubric scores honestly** — not PaperBench-official.

Parallelism: the user wants multiple runs at once. Contention is real — shared
Claude OAuth quota (`implement_baseline`), one 6 GB GPU, Featherless 2–3× load.
The repair loop + honesty guard make contention **degrade gracefully**. Prefer
2 concurrent over 3; stagger launches so `implement_baseline` phases don't align.

## 6. Technical reference

**Run a bundle:** `.venv/bin/python scripts/rlm_paperbench.py <paper_id> --model
qwen3-coder-featherless --sandbox docker --gpu-mode prefer --max-wall-clock 7200`
**Run an arXiv paper:** `.venv/bin/python -m backend.cli reproduce <arxiv_id>
--mode rlm --model qwen3-coder-featherless --sandbox docker`
**Leaf-score:** `.venv/bin/python scripts/score_run.py runs/<run_dir> <paper_id>`
**Tests:** `.venv/bin/python -m pytest tests/ -q`

**Key files.** `backend/agents/rlm/run.py` (`run_pipeline_rlm`, `_ROOT_PROMPT`,
`_resolve_agent_runtime`); `primitives.py` (the 9 primitives); `binding.py`
(`wrap_primitive`); `sse_bridge.py` (events, corpus guard); `report.py`
(`build_final_report` + honesty guard); `models.py` (root-model registry,
Featherless); `backend/evals/paperbench/leaf_scorer.py`; `scripts/`.

**Models.** Root + sub-calls: Featherless `qwen3-coder-featherless`
(`FEATHERLESS_API_KEY` in `.env`). `implement_baseline` sub-agent: Claude Sonnet
(OAuth in dev, `ANTHROPIC_API_KEY` in prod — SDK-resolved). `.env` has
`REPROLAB_LLM_PROVIDER=openai` — harmless; `_resolve_agent_runtime` overrides it.

**Known gaps (carry forward).**
- `run_experiment` never extracts metrics — `_execute_in_sandbox` hardcodes
  `metrics: {}`. A `reproduced` verdict with measured numbers needs reading
  `metrics.json` from `runs/<id>/code/outputs/`. (Real next improvement.)
- The experiment repair loop is implemented but **not yet verified on a live run**.
- `has_provider_credentials` treats the `claude` CLI on `PATH` as proof of login.
- `build_environment`'s `ThreadPoolExecutor` can block past its timeout.
- `--sandbox` flag is a no-op for RLM primitives (`run_experiment` hardcodes
  `LocalDockerBackend`).
- A self-generated rubric graded by the same model family is somewhat circular —
  label it honestly.

## 7. Read before non-trivial work

`progress.md` · `frontend_integration.md` · `docs/design/rlm-pivot-brief.md` ·
`CLAUDE.md` · `learn.md` · the prior handoff `2026-05-21-rlm-phase5-session2-handoff.md`.
