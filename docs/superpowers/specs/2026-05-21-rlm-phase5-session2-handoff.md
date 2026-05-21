# RLM Phase 5 — Session-2 Handoff (finish the end-to-end run + wrap)

> Self-contained context to continue **Phase 5 / issue #62** of the OpenResearch
> RLM pivot in a fresh Claude Code session. Written 2026-05-21 at the end of the
> session that merged #59+#60, vendored the PaperBench bundles, production-
> hardened the RLM surface, and launched the first real reproduction run.
>
> **To resume:** open a fresh session in `/home/abheekp/openresearch` and paste:
>
> > *Read `docs/superpowers/specs/2026-05-21-rlm-phase5-session2-handoff.md` in
> > full, then continue. Use /iterate. Opus designs + reviews every diff, Sonnet
> > sub-agents execute in parallel, no Codex on implementation. First action:
> > check the WS-E run (§5).*

Read top-to-bottom. §2 (working style), §4 (done), §5–§6 (state + plan) are load-bearing.

## 1. Who & where
- **User:** lolout1 / Abheek (`sww35@txstate.edu`).
- **Repo:** `/home/abheekp/openresearch`. Venv: `.venv/bin/python` (Py 3.12).
- **Branch: `feat/rlm-phase5-e2e`** — the Phase 5 integration branch; all work below lives here. Branched off `rlm-pivot`.
- **Remotes:** `origin` = `armaanamatya/openresearch` (**push here**). `replix` = `lolout1/Replix` (**NEVER push here**). `gh` CLI is **not authed**.
- **Stakes:** shown to a Microsoft VP + a Deepinvent funder — VP-grade, production bar.

## 2. Standing instructions & working style — ALL carry forward
- **Execution model:** Opus designs + **reviews every diff**; **Sonnet** sub-agents execute. **NO Codex on implementation** (Codex = adversarial spec/design-doc review only, and only when the user explicitly asks).
- **Process:** the **/iterate** skill — recon → plan → confirm with the user → implement → verify. **Grill the user on big design decisions**; surface contradictions; never guess silently. (/token: Opus plans + reviews, delegate execution.)
- **Quality:** root-level elegant solutions, not patches; production-grade. Opportunistically improve existing code — but **ask before significant changes**.
- **Delegation:** fan out **Sonnet** sub-agents in parallel — **disjoint file sets**, tight specs (exact paths, contract, what-not-to-touch, acceptance), Opus reviews every diff. Serialize commits.
- **Git:** push to `origin`, never `replix`. **Commit/push only when the user asks.** Commit messages end with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- **Doc-update contract:** `CHANGELOG.md` + `learn.md` (append-only); keep `system_overview.md` / `CLAUDE.md` in sync.
- **Verify with real test output before claiming done.** Investigate and fix every issue found.

## 3. The RLM pivot — context
ReproLab reproduces research papers end-to-end and scores them against a
PaperBench-style rubric. It is being re-architected from a 14-stage
`PipelineStage` machine to the **RLM** (Recursive Language Models, arXiv
2512.24601) paradigm on the **`rlms`** PyPI library. Canonical spec:
`docs/design/rlm-pivot-brief.md`. **Phase 5 / #62** = run the RLM system
end-to-end on real PaperBench papers and produce real reproduction reports
(≥2 papers with real rubric scores on disk).

## 4. What's DONE this session — all committed on `feat/rlm-phase5-e2e`
Commits (newest first): `ab78dd4` fix api_key plumbing · `9ab2c4a` WS-G ·
`240a7e9` WS-E code · `1651317` WS-F · `87e02d2` WS-H · `771547a` WS-B ·
`f23de61` merge #59.

- **WS-A** — #59 (the RLM primitive layer, all 14 tasks — complete) merged into the
  branch with #60 (the RLM orchestrator). One real conflict (`__init__.py __all__`)
  resolved as a union.
- **WS-B** — 3 real PaperBench bundles vendored from `openai/preparedness` into
  `third_party/paperbench/<id>/` (paper.md/addendum.md are Git LFS — fetched via
  the `openai/frontier-evals` LFS batch API): **ftrl** (178 rubric leaves),
  **sequential-neural-score-estimation** (92), **mechanistic-understanding** (96).
- **WS-H** — production-hardening: a 4-agent audit (~47 findings) → 4 fix batches.
  R2 per-primitive deadlines (`RunContext.deadline_utc`), `max_usd` cost cap
  (rlm's native `max_budget`), corpus-leak redaction at every egress
  (`redact_corpus`), run-status integrity (atomic writes, SIGKILL escalation),
  RunPod backend hardening. Full suite **1079 passed**.
- **WS-F** — Phase 5 docs (`CHANGELOG.md`, `learn.md`, `system_overview.md`, `CLAUDE.md`).
- **WS-G** — `backend/services/runtime/brev_backend.py` — Brev sandbox backend,
  **structural only, UNVERIFIED** (no `BREV_API_KEY`). brev.dev has no public REST
  API — it shells out to the `brev` CLI.
- **WS-E code** — `scripts/rlm_paperbench.py` (the bundle→RLM-run bridge) +
  `backend/evals/paperbench/leaf_scorer.py` (post-run 178-leaf rubric scorer).

## 5. Current state — the WS-E real run (CHECK THIS FIRST)
A real reproduction run is in progress (background): `sequential-neural-score-estimation`,
Claude root, local docker + RTX 2060 GPU, capped 90 min / $40.
**Log: `/tmp/ws-e-snse3.log`. Artifacts: `runs/pb_sequential-neural-score-estimation_<ts>/`.**

This is attempt 3 — two earlier bugs are now FIXED:
- Attempts 1–2 failed fast at the **Docker preflight** — Docker Desktop's WSL
  integration kept dropping. **Docker stability remains the #1 operational risk.**
- Attempt 2 surfaced an api_key bug: `models.py` did not plumb `api_key` into
  `backend_kwargs`, and rlm's `AnthropicClient` requires it. Fixed — commit `ab78dd4`.

**FIRST ACTION:** check the run — `tail -60 /tmp/ws-e-snse3.log`, `ls -dt runs/pb_*`,
`cat runs/<id>/final_report.json`, `runs/<id>/demo_status.json`. It may be still
running, completed, or failed on the next bug — if so, diagnose → fix → relaunch
the bridge (§7). Expect to "fix what breaks" iteratively, one failure class at a
time (the original handoff's §8 Step 5).

## 5b. Session-3 update — work now on branch `merge`
Work continued on a new branch `merge` (base `rlm-pivot`). `feat/rlm-phase2-foundation`
verified fully merged (`git rev-list --count merge..origin/feat/rlm-phase2-foundation` = 0).

DONE this session (each its own commit, pushed to `origin/merge`):
- ✅ **Featherless root-model backend** — `qwen3-coder-featherless` registry entry;
  `RootModel.api_key_env` decouples key-source from backend type (`bb2bba5`).
  The RLM root loop needs a real API key — Featherless (flat-rate, key owned).
- ✅ **runs_root absolute-path fix** — `Path(runs_root).resolve()`; primitives run
  in the RLM REPL's CWD, so a relative root broke every artifact write (`a750ea5`).
- ✅ **Bug A — tolerant `PaperClaimMap`** — `MetricSpec.definition` defaults to `""`;
  before-validators coerce plain-string claims/metrics; `PRIMITIVE_DESCRIPTIONS`
  publish the schema. Was causing fabricated mock 0.725 reports.
- ✅ **Bug B — sub-agent runtime via Claude OAuth** — `_resolve_agent_runtime` in
  `run.py`: an RLM run has two LLM layers — the root loop (real API key) and the
  sub-agent runtime (`implement_baseline`), which can run on the Claude Code
  subscription via OAuth (`claude-agent-sdk` spawns the `claude` CLI). Bypasses
  `.env`'s dead `REPROLAB_LLM_PROVIDER=openai`.

## 6. What's LEFT — the plan
1. ⏳ **WS-E — finish the run.** If it completed: produce the authoritative PaperBench
   score with the leaf scorer —
   `from backend.evals.paperbench.leaf_scorer import score_reproduction, amend_final_report`;
   `score_reproduction(bundle.rubric(), Path("runs/<id>"), llm_client)` then
   `amend_final_report(...)`. If it failed: diagnose, fix, relaunch the bridge.
2. ☐ **2nd paper** — repeat for `ftrl` or `mechanistic-understanding`. **Deliverable:
   ≥2 papers with real rubric scores in `runs/<id>/final_report.json`.**
3. ☐ **WS-F PR bodies** — #60 PR (`feat/rlm-phase3-orchestrator` → `rlm-pivot`) and
   the Phase 5 PR (`feat/rlm-phase5-e2e` → `rlm-pivot`). `gh` not authed — draft
   bodies; the user creates the PRs.
4. ☐ **Follow-up** — `run_experiment` (`backend/agents/rlm/primitives.py:387`)
   **hardcodes `LocalDockerBackend`**; `run_pipeline_rlm`'s `sandbox_mode` arg
   never reaches the primitives (`RunContext` has no sandbox field). So the RLM
   reproduction can only use **local Docker** — RunPod/Brev are unreachable for it.
   Plumb `sandbox_mode` → `RunContext` → `run_experiment` if that support is wanted.

## 7. Key technical facts
- **Run a paper:** `.venv/bin/python scripts/rlm_paperbench.py <paper_id> --model claude --provider anthropic --sandbox docker --gpu-mode prefer --max-usd 40 --max-wall-clock 5400`. The bridge calls `load_dotenv()` — nothing else loads `.env` for a local run.
- **Rubric:** PaperBench `rubric.json` is a recursive weighted tree
  (`{id, requirements, weight, sub_tasks}`). The in-loop `verify_against_rubric`
  primitive is fail-soft on it; the **authoritative** score is the post-run
  `leaf_scorer` (flatten leaves → batched LLM grading → weighted roll-up).
- **Root models:** `gpt-5` / `claude` are keyed in `.env`. `qwen3-coder` / `kimi-k2.5`
  need `OPENROUTER_API_KEY` (deferred — `resolve_root_model` fails fast without it).
  The RunPod key in `.env` returns HTTP 401 (invalid).
- **GPU:** RTX 2060, 6 GB. `LocalDockerBackend` + `--gpu-mode prefer` passes it
  into the container (small models only — SNSE's MLPs fit; GPT-2-medium DPO is tight).
- **Entry points:** `run_pipeline_rlm` in `backend/agents/rlm/run.py`. claim_map
  shape: `{project_id, entries:[{source_id,title,excerpt}], rubric_spec}`.

## 8. Open issues / risks
- Docker Desktop WSL integration flakiness — the run needs stable local Docker.
- Brev backend unverified (no key); RunPod key invalid.
- `sandbox_mode` not plumbed to the RLM primitives (§6.4).
- `run_experiment`'s per-primitive deadline returns on timeout but cannot kill
  the underlying worker thread — `run.py`'s process watchdog is the hard backstop.

## 9. Artifacts & references
- **This handoff.** Prior session docs: `2026-05-21-rlm-phase5-e2e-handoff.md`,
  `-execution-plan.md`, `-hardening-plan.md`, `-rlm-phase3-orchestrator-design.md`
  (all under `docs/superpowers/specs/`).
- Canonical brief: `docs/design/rlm-pivot-brief.md`. Lessons: `learn.md`.
  Changelog: `CHANGELOG.md`.
