# RLM Phase 5 — End-to-End Reproduction (#62) + the #59↔#60 Merge: Session Handoff

> **Purpose.** Self-contained context to drive **Phase 5 / issue #62 — end-to-end
> PaperBench reproduction runs** — and the **#59↔#60 branch merge** — in a fresh
> Claude Code session. Written 2026-05-21 at the end of the session that
> *completed #60* (the RLM orchestrator) and *verified the #59 integration*.
> Everything needed is here or in the docs this references — nothing is lost to a
> `/clear`.
>
> **To resume:** open a fresh session in `/home/abheekp/openresearch` and paste:
>
> > *Read `docs/superpowers/specs/2026-05-21-rlm-phase5-e2e-handoff.md` in full,
> > then execute it. Use /iterate. Don't use Codex for implementation — fan out
> > Sonnet sub-agents, work in parallel, Opus reviews every diff. Fetch origin
> > periodically to stay aligned. Be robust, thorough, methodical.*
>
> Read top-to-bottom first. §2 (standing instructions), §5–§6 (state + integration
> findings), §8 (the plan), §9 (blockers) are load-bearing.

---

## 1. Who & where

- **User:** lolout1 / Abheek (`sww35@txstate.edu`).
- **Repo:** `/home/abheekp/openresearch`. Python venv: `.venv/bin/python` (Py 3.12).
- **Branches:**
  - `feat/rlm-phase3-orchestrator` — **#60, the RLM orchestrator — COMPLETE** (8 commits, pushed to `origin`). This session's work.
  - `feat/rlm-phase2-foundation` — **#59, the primitive layer — Aayush, IN PROGRESS** (at "Task 7 of 14" as of 2026-05-21 05:09). Pull it from `origin`.
  - `rlm-pivot` — the pivot integration branch (PR #65 → `main`, open).
  - `main` — production.
- **Remotes:** `origin` = `armaanamatya/openresearch` (**canonical — push here**). `replix` = `lolout1/Replix` (**fork — NEVER push here**). Both public. `gh` CLI is **NOT logged in** and there is **no API token** — use `curl https://api.github.com/repos/armaanamatya/openresearch/...` for reads; PR creation needs the user (or have them run `! gh auth login`).
- **Stakes:** the RLM pivot is shown to a **Microsoft VP** and a **Deepinvent funder**. Bar = VP-grade, production rigor, genuinely elegant. The honesty driver: the pitch said "RLM-based" — the implementation must be.

## 2. Standing instructions & working preferences — ALL carry forward

These are the user's instructions accumulated across the whole effort. Honor every one.

- **Execution model:** **Opus designs + reviews every diff; Sonnet sub-agents execute** (implementation code included). **NO Codex on implementation** — not writing it, not reviewing it. Codex *adversarial review* is allowed for **specs / design docs only**, and only when the user **explicitly asks**.
- **Quality bar:** production-grade, modular, scalable, big-tech standards. **Root-level elegant solutions, not patches** — the smallest correct change plus a guard test, one canonical abstraction over scattered fixes.
- **Process — the `/iterate` discipline:** recon → plan → **confirm with the user** → implement → verify. Brainstorm intent before creative work; present a design and get approval *before* writing code. Transform imperatives into verifiable goals; evidence before assertions.
- **Delegation:** fan out **Sonnet** sub-agents for execution & recon — parallel, efficient, **without sacrificing quality** (held by a tight spec + Opus diff-review). Every delegated task gets: exact paths, the interface contract, what NOT to touch, acceptance commands. Disjoint file sets per parallel agent; serialize commits.
- **Grill the user** on design choices & tradeoffs before locking a design. Surface contradictions; never guess silently; present options when several interpretations exist.
- **Git:** push to **`origin`**, never `replix`. **Commit/push only when the user asks.** Commit messages end with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- **Stay aligned:** **fetch `origin` / other branches periodically** — #59 (Aayush) moves fast; re-verify against current state and realign elegantly when it does.
- **Doc-update contract:** architecture/behavior change → a `CHANGELOG.md` entry; a bug fixed or a non-obvious lesson → a `learn.md` entry (Symptom→Root cause→Fix→Lesson→Guardrail). `CHANGELOG.md` / `learn.md` are append-only. Keep `system_overview.md` / `CLAUDE.md` in sync on drift.
- **Verification before completion:** run the verification commands, read the output, *then* claim done. No success claim without evidence.
- **Investigate and fix every issue you find** — don't just report them.
- **Memory:** auto-loaded each session via `MEMORY.md`; `project_rlm_pivot` is the live project memory — keep it current.

## 3. The RLM pivot — context

ReproLab reproduces research papers end-to-end and scores them against a PaperBench-style rubric. It is being re-architected from a **14-stage `PipelineStage` state machine** to the **RLM (Recursive Language Models)** paradigm (arXiv 2512.24601), built on the **`rlms` PyPI library** (`pip install rlms`, import name `rlm`) — NOT hand-built.

- **Canonical spec:** `docs/design/rlm-pivot-brief.md` (the "RLM Pivot Plan", rewritten 2026-05-20 — single canonical document). Phase-1 mapping artifact: `docs/rlm-pivot-mapping.md`.
- **Phases #58–#63** (label `rlm-pivot`, umbrella #64): #58 Phase 1 (done) · #59 Phase 2 primitives · #60 Phase 3 orchestrator · #61 Phase 4 frontend · **#62 Phase 5 end-to-end** · #63 Phase 6 cleanup.
- The RLM root loop (Algorithm 1) is the orchestrator; the stage agents become 9 REPL-callable primitives the root invokes from Python code; Gates 1/2/3 and the hardcoded 5 improvement paths are deleted.

## 4. The task — Phase 5 / issue #62

**Merge #59 (primitives) + #60 (orchestrator), then run the RLM system end-to-end on real PaperBench papers and produce real reproduction reports.**

From brief §11 Phase 5 + issue #62:
1. Merge the #59 and #60 branches (pull #59 from `origin`).
2. Vendor the **real `ftrl` PaperBench bundle** (brief §13 — the current `third_party/paperbench/ftrl/` is a placeholder with a *synthetic* rubric; vendor the real upstream `paper.md` / `addendum.md` / `rubric.json` from the public PaperBench repo) plus 1–2 genuinely easy papers.
3. Run the RLM system end-to-end (`--mode rlm`) on those papers; fix everything that breaks.
4. **Done condition:** ≥2 PaperBench papers with completed runs and **real rubric scores** in `runs/<project_id>/final_report.{json,md}` on disk.

This is the deliverable bar of the whole pivot (brief §1): *no paper has ever been reproduced end-to-end by ReproLab, by any architecture.* Phase 5 is where that becomes true.

## 5. State of play (2026-05-21)

### #60 — the RLM orchestrator — COMPLETE
`feat/rlm-phase3-orchestrator`, 8 commits, pushed to `origin`. **1042-test full suite green** (163 RLM-specific). Codex-adversarial-reviewed spec (8 findings fixed). Modules in `backend/agents/rlm/`:
- `run.py` — `run_pipeline_rlm`: builds the `rlm.RLM`, runs `.completion()` on a worker thread, three-layer time bound, writes `final_report.{json,md}`, returns `RLMRunResult`.
- `system_prompt.py` · `models.py` (root-model registry: GPT-5 / Qwen3-Coder / Kimi K2.5 / Claude) · `sse_bridge.py` (the corpus sanitizer + `RLMLogger` subclass + event schema) · `checkpoint.py` (SQLite event store + sanitized snapshots) · `report.py` · `stub_primitives.py`.
- Wiring: `--mode {offline,sdk,rlm}` in `cli.py` / `pipeline.py` / `live_runs.py`; `offline`/`sdk` unchanged.
- Design spec: `docs/superpowers/specs/2026-05-21-rlm-phase3-orchestrator-design.md` (v3).
- A PR for #60 (`feat/rlm-phase3-orchestrator` → `rlm-pivot`) is **pending creation** — the branch is pushed; `gh`/token unavailable. PR title/body are in §10.

### #59 — the primitive layer — IN PROGRESS (Aayush)
`feat/rlm-phase2-foundation`, at **"Task 7 of 14"** (head `43d605b`, 2026-05-21 05:09). Shipped so far: `context.py` (`RunContext`), `binding.py` (`build_custom_tools` + `wrap_primitive`), `dashboard_emitter.py` (+`primitive_call`), `requirements.txt` (+`rlms==0.1.1`), and ~2–3 of the 9 primitives in `primitives.py` (`understand_section`, `extract_hyperparameters`, …). **Not yet done:** the remaining ~6 primitives, `primitives.PRIMITIVE_DESCRIPTIONS` (its Task 13), removing the vestigial `set_final` (its Task 12), and its `__init__.py` re-exports. #59's own plan: `docs/design/phase2-implementation-plan.md` (1684 lines, Tasks 0–14) on its branch.

## 6. The #59↔#60 integration — VERIFIED this session

A throwaway test-merge of `feat/rlm-phase2-foundation` into `feat/rlm-phase3-orchestrator` was run this session:
- **The merge is clean — zero conflicts.** `context.py` / `tests/rlm/conftest.py` were vendored byte-identical from #59, so they merge as no-ops; `primitives.py` / `dashboard_emitter.py` are #59-only; `__init__.py` did not conflict (#59 hasn't reached its `__init__.py` edits — when it does, both sides are append-only → trivial).
- **177 / 178 merged tests passed.** The one failure was the predicted seam gap: `build_custom_tools(ctx)` raises `AttributeError` because `primitives.PRIMITIVE_DESCRIPTIONS` does not exist yet (#59's Task 13).
- **Fixed this session** (commit `590ab26` on `feat/rlm-phase3-orchestrator`): `run.py::_resolve_custom_tools` now degrades to the stub provider — *loudly*, via a WARNING — when #59's `binding.py` is present-but-incomplete, not only when absent. So a merged tree runs cleanly today; the real primitives are picked up automatically once #59 completes `PRIMITIVE_DESCRIPTIONS`.

**Verdict:** the seam is correct and conflict-free. The only thing standing between "merges clean" and "reproduces a real paper" is **#59 finishing its 9 primitives** (§9).

## 7. Key technical facts — do not re-derive

- **`rlms` 0.1.1**, import `rlm`. `from rlm import RLM`. `RLM(...).completion(prompt, root_prompt) -> RLMChatCompletion` — synchronous, blocks; run it on a worker thread (`run.py` does `asyncio.to_thread`). `prompt` becomes the offloaded **`context`** REPL variable.
- **`environment="local"` is mandatory** — `rlm`'s `DockerREPL` does not inject `custom_tools`. Docker is used only *inside* the `build_environment` / `run_experiment` primitives.
- **Termination** = the root emits `FINAL_VAR(varname)`; `rlm` `str()`-ifies that variable. The report contract is therefore **JSON**: the root `json.dumps` its report into a variable and `FINAL_VAR`s that; `report.py` `json.loads` it (with an `ast.literal_eval` fallback).
- **`custom_system_prompt` is a `str.format()` template** — `rlm`'s `build_rlm_system_prompt` runs `prompt.format(custom_tools_section=…)` (`rlm/utils/prompts.py:156`). `build_system_prompt` escapes every literal brace and exposes exactly one `{custom_tools_section}` slot. (Post-mortem: `learn.md`, 2026-05-21.)
- **Algorithm-2 guard:** the paper is offloaded as `context` and must never reach the root's window or any log/event/snapshot. `sse_bridge.sanitize_iteration` is the single chokepoint — it value-strips `RLMIteration` (whose `locals` carry the corpus). Primitives take slices/specs, never the whole corpus.
- **The seam #60 consumes from #59:** `from backend.agents.rlm.context import RunContext`; `from backend.agents.rlm.binding import build_custom_tools`. `build_custom_tools(ctx) -> {name: {"tool": callable, "description": str}}` → `RLM(custom_tools=…)`. `run.py` lazily resolves it (real or stub).
- **Run entry:** `backend.agents.rlm.run.run_pipeline_rlm(project_id, runs_root, workspace_claim_map, *, model, provider, …)`. CLI: `python -m backend.cli reproduce <src> --mode rlm`.
- **Root model:** `REPROLAB_RLM_ROOT_MODEL` env var (`gpt-5` / `qwen3-coder` / `kimi-k2.5` / `claude`); default layered (GPT-5 if `OPENAI_API_KEY` else Qwen3-Coder). Paper-validated: GPT-5, Qwen3-Coder.
- **Cost:** `RLMChatCompletion.usage_summary` (root + sub-call LLM) + the `cost_ledger`. The `LlmClient` protocol returns no per-call usage — primitive-internal LLM cost is not itemized (spec §19; a #59/#60 seam evolution, deferred).
- **Time bound (Codex H2):** `rlm`'s `max_timeout` only checks between iterations; a hung primitive overruns it. `run.py` has a process-level wall-clock watchdog backstop. The real fix — per-primitive deadlines inside `build_environment`/`run_experiment` — is a **#59 requirement**; verify #59 added them.

## 8. The methodical plan for Phase 5

**Step 0 — Orient.** Read this handoff, `docs/design/rlm-pivot-brief.md` (§11 Phase 5, §13), `docs/superpowers/specs/2026-05-21-rlm-phase3-orchestrator-design.md`, and `learn.md`. `git fetch origin`. Assess #59's current state: `git log --oneline origin/feat/rlm-phase2-foundation`; check `primitives.py` (how many primitives real, is `PRIMITIVE_DESCRIPTIONS` defined, is `set_final` removed), `__init__.py`.

**Step 1 — Resolve #59 readiness (grill the user).** A true end-to-end run needs **all 9 #59 primitives complete + `PRIMITIVE_DESCRIPTIONS` + `set_final` removed**. If #59 is NOT complete, do not proceed silently — grill the user: (a) wait for Aayush to finish #59; (b) **help finish #59** — fan out Sonnet sub-agents against #59's own `docs/design/phase2-implementation-plan.md` (the remaining Tasks), Opus reviewing; (c) merge now and run a deliberately partial pass to shake out orchestrator-side bugs. Recommend (b) if the user wants Phase 5 to land soon and Aayush is not actively closing it out.

**Step 2 — Merge #59 + #60.** On an integration branch (or directly onto `rlm-pivot` if the user wants): `git fetch origin`, then merge `origin/feat/rlm-phase2-foundation` and `feat/rlm-phase3-orchestrator`. The test-merge was clean this session — re-verify (#59 will have moved). Resolve `__init__.py` if both branches edited it (append-only both sides — combine the import lists / `__all__`). Run `.venv/bin/python -m pytest tests/rlm/ -q` — must be green.

**Step 3 — Verify the real integration.** With `REPROLAB_RLM_STUB_PRIMITIVES` unset, run `run_pipeline_rlm` — `_resolve_custom_tools` must log `"real (#59 binding)"`, not a stub fallback. Drive the orchestrator with #59's real `build_custom_tools`: confirm the Algorithm-1 loop runs, primitives execute, `repl_iteration` / `primitive_call` SSE events stream, the SQLite checkpoint records each iteration, `final_report.{json,md}` is written, and the corpus never leaks (the `sanitize_iteration` invariant — assert no `paper_text` substring in events/store/snapshots).

**Step 4 — Vendor the real `ftrl` bundle.** Brief §13: replace the placeholder `third_party/paperbench/ftrl/` with the real upstream `ftrl` artifacts (real `paper.md`, `addendum.md`, `rubric.json` from the public PaperBench repo). Add 1–2 genuinely easy papers so scores are honestly comparable to PaperBench's published baselines.

**Step 5 — Run end-to-end & fix what breaks.** `python -m backend.cli reproduce <ftrl> --mode rlm --provider <…> --sandbox docker`. Prerequisites: a real root-model API key (`REPROLAB_RLM_ROOT_MODEL` + the provider key — confirm with the user which model/keys); Docker running. Expect to fix: primitive bugs surfaced by a real paper, sandbox/Docker issues, root-model behavior (prompt tuning), cost/timeout caps, the per-primitive deadline gap. Iterate methodically — one failure class at a time.

**Step 6 — Produce the deliverable.** Two PaperBench papers with completed runs and real rubric scores in `runs/<id>/final_report.{json,md}`. Pin one as the demo (brief §12).

**Step 7 — Wrap.** `CHANGELOG.md` + `learn.md` entries; update `system_overview.md` / `CLAUDE.md` (Phase 6 territory, but note drift); PRs for the merged work. Update the `project_rlm_pivot` memory.

## 9. Open issues, blockers, risks

1. **#59 incomplete is THE blocker** (§5). All 9 primitives + `PRIMITIVE_DESCRIPTIONS` must land for an end-to-end run. Step 1 resolves how to handle it.
2. **Per-primitive deadlines (Codex H2).** `rlm`'s `max_timeout` does not interrupt a primitive wedged in `execute_code`. Verify #59's `build_environment` / `run_experiment` bound their own execution and cancel the sandbox on deadline; until then `run.py`'s wall-clock watchdog is the only backstop.
3. **API keys + Docker.** A real run needs a real root-model key and a working Docker daemon. Confirm with the user (which model — they want GPT-5 / Qwen / Kimi / Claude all *supported*; pick the demo model + key).
4. **The `ftrl` bundle.** Vendoring needs access to the public PaperBench repo's real `ftrl` artifacts. The current bundle is a synthetic placeholder — a real rubric score requires the real one.
5. **`set_final`** is still in #59's `PRIMITIVE_REGISTRY` (10 entries); its Task 12 removes it. Harmless to #60 (it passes whatever `build_custom_tools` yields), but the registry should be 9 after #59 completes.
6. **PR #60** is uncreated (no `gh` auth) — see §10.

## 10. Artifacts & references

- **This handoff:** `docs/superpowers/specs/2026-05-21-rlm-phase5-e2e-handoff.md`.
- **#60 design spec:** `docs/superpowers/specs/2026-05-21-rlm-phase3-orchestrator-design.md` (v3).
- **#60 session handoff (predecessor):** `docs/superpowers/specs/2026-05-21-rlm-orchestrator-handoff.md`.
- **Canonical brief:** `docs/design/rlm-pivot-brief.md`. **Mapping:** `docs/rlm-pivot-mapping.md`.
- **#59's plan:** `docs/design/phase2-implementation-plan.md` + `phase2-execution-brief.md` (on `feat/rlm-phase2-foundation`).
- **Lessons:** `learn.md` (root). **Changelog:** `CHANGELOG.md` (root).
- **#60 PR (to create — push done, `gh` unavailable):** base `rlm-pivot` ← head `feat/rlm-phase3-orchestrator`. URL: `https://github.com/armaanamatya/openresearch/compare/rlm-pivot...feat/rlm-phase3-orchestrator?expand=1`. Title: `RLM Pivot Phase 3: RLM orchestrator (#60)`. Body: summary of the orchestrator + decoupled-from-#59 + the test evidence (1042 passed).

---

*Handoff written 2026-05-21. This session: continued #60 from its handoff → grilled the user (orchestrator-only, new `rlm` mode, event-log+snapshot checkpoint, multi-provider root) → wrote + Codex-reviewed the #60 spec (8 findings fixed) → realigned twice as #59 went unstarted → in-flight → fanned out 5 Sonnet agents across 3 waves, Opus reviewing every diff → caught + fixed the `rlm` `.format()`-template showstopper → completed #60 (8 commits, pushed, 1042 tests green) → verified the #59↔#60 merge is clean and fixed the present-but-incomplete-binding robustness gap. Next: Phase 5 / #62 — merge, vendor `ftrl`, run end-to-end, produce real reproduction reports. Execute from §8.*
