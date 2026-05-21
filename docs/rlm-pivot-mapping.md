# RLM Pivot — Stage-to-Primitive Mapping

**Phase 1 done-condition artifact (issue [#58](https://github.com/armaanamatya/openresearch/issues/58))**
**Canonical spec:** `docs/design/rlm-pivot-brief.md` + the brief's 2026-05-19 paper-accuracy corrections (which override the brief body where they conflict)
**Paper:** arXiv 2512.24601v3 (Zhang, Kraska, Khattab, MIT CSAIL, 11 May 2026)
**Reference impl:** https://github.com/alexzhang13/rlm

This document maps each existing stage agent to the primitive it becomes in the RLM REPL, records the function signatures Phase 2 must implement, identifies what survives from `rlm_query.py`, and notes the design items that need decisions before Phase 2 starts.

> **✅ Fork RESOLVED (2026-05-21) — partially superseded.** The architecture
> fork is closed: the `rlm` library wins (see `docs/design/rlms-spike-report.md`).
> What that means for this doc:
> - **§1 stage→primitive table and §2 primitive signatures SURVIVE** — domain
>   primitives are real work under the library architecture; they become the
>   `rlm` `custom_tools` dict.
> - **§5 (`rlm_query.py` reuse) and §6 ("frozen Phase 2 hand-build decisions")
>   are SUPERSEDED** — the `rlm` library provides the REPL host, root loop, and
>   `sub_LLM`/`sub_RLM`; we do not hand-build or reuse `rlm_query.py` for them.
> - Section refs to the pre-rewrite brief (§7.7, §13 FM#1–#10) are stale (drift
>   D4) — the current brief is `docs/design/rlm-pivot-brief.md`.

---

## 1. Stage agent → primitive

The 14 ordered stages in `topology.py:185–198` and `PipelineStage` (`orchestrator.py:162–178`) collapse into a flat library of REPL-callable functions. Order is no longer load-bearing — the root model decides the call sequence per paper.

| Current stage / module | Current entry point | Becomes primitive | Source of truth |
|---|---|---|---|
| `paper_understood` / `backend/agents/paper_understanding.py` | `run_offline(project_id, runs_root, workspace_claim_map) -> PaperClaimMap`<br>`run_with_sdk(...)` (async) | `understand_section(text_slice: str) -> dict`<br>`extract_hyperparameters(text_slice: str) -> dict` | `paper_understanding.py:31, 83` |
| `artifacts_discovered` / `backend/services/artifact_discovery/*` | discovery service | Folded into REPL init: `repo_files`, `prior_work_refs` populated at workspace bootstrap. No primitive — root reads the variable. | n/a |
| `environment_built` / `backend/agents/environment_detective.py` | `run_offline(project_id, runs_root, paper_claim_map, artifact_index=None) -> EnvironmentSpec`<br>`run_with_sdk(...)` (async) | `detect_environment(method_spec: dict) -> dict`<br>`build_environment(env_spec: dict) -> dict` (wraps the Docker build-and-repair loop — retry logic preserved inside the primitive) | `environment_detective.py:54, 105` |
| `plan_created` / reproduction planner | planner agent | `plan_reproduction(method_spec: dict, env_spec: dict) -> dict` | (planner module — see `agents/runtime`) |
| `gate_1_passed` / `gate_2_passed` / `gate_3_passed` | `verification.run_gate_offline(...)` / `run_improvement_gate_offline(...)` | Gates are deleted as control-flow checkpoints. `verify_against_rubric(results: dict, rubric: dict) -> dict` becomes a primitive the root calls when it judges appropriate. | `verification.py:245, 298` |
| `baseline_implemented` / `backend/agents/baseline_implementation.py` | `run_offline(project_id, runs_root, paper_claim_map, environment_spec, reproduction_contract=None, artifact_index=None) -> BaselineResult`<br>`run_with_sdk(...)` (async) | `implement_baseline(plan: dict) -> str` (returns path to generated code) | `baseline_implementation.py:365, 418` |
| `baseline_run` / `backend/agents/experiment_runner.py` | `run_offline(project_id, runs_root, baseline_result, reproduction_contract=None, *, simulate_metrics=None) -> ExperimentArtifacts`<br>`run_with_runtime/runpod/local_process/sdk(...)` (async sandbox variants) | `run_experiment(code_path: str, env_id: str) -> dict` (depends on sandbox state outside REPL — see §5) | `experiment_runner.py:41, 164, 345, 379, 431` |
| `improvements_selected` / `backend/agents/improvement.py` | `select_hypotheses_offline(paper_claim_map, baseline_metrics, *, user_hints=None, n_paths=3) -> list[ImprovementHypothesis]` | `propose_improvements(current_results: dict, rubric_scores: dict, k: int \| None = None) -> list[dict]` — **prompt rewritten** to return variable-length lists with proposer-assigned free-form tags. No 5-category taxonomy. | `improvement.py:68` |
| `improvements_run` | `run_path_offline(project_id, runs_root, hypothesis, baseline_metrics, *, simulate_success=True) -> PathResult` | (no new primitive — root re-uses `implement_baseline` + `run_experiment` on the chosen candidate) | `improvement.py:96` |
| Rubric verifier (currently a sub-agent invoked at gates) | `RubricVerification` output model | `verify_against_rubric(results: dict, rubric: dict) -> dict` | `schemas.py:336`, rubric-verifier agent prompt |
| `research_map_generated` + `complete` / `backend/agents/report_generator.py` | `generate_final_report(...)` | `set_final(report: dict) -> None` (convenience — sets the `FINAL_VAR`-pointed variable). Final-report rendering stays in the report module and is invoked from the primitive. | `report_generator.py:641, 865, 1023` |

**Primitives invariant (Algorithm-2 guard, brief §7.7):** No primitive signature accepts `paper_text` / `supplementary_text` / `repo_files` as a whole-corpus argument. Primitives take **slices and structured specs**; the root assembles them with REPL code and `llm_query` / `rlm_query` against constructed slices. A primitive that grows a `paper_text: str` parameter has slipped back into Algorithm 2 — reject at code review.

---

## 2. Primitive function signatures (Phase 2 contract)

These are the exact signatures Phase 2 will implement in `backend/agents/rlm/primitives.py`. They follow the brief §7 list plus the Algorithm-2 guard from §1 above.

```python
def understand_section(text_slice: str) -> dict: ...
def extract_hyperparameters(text_slice: str) -> dict: ...
def detect_environment(method_spec: dict) -> dict: ...
def build_environment(env_spec: dict) -> dict: ...
def plan_reproduction(method_spec: dict, env_spec: dict) -> dict: ...
def implement_baseline(plan: dict) -> str: ...                       # returns code_path
def run_experiment(code_path: str, env_id: str) -> dict: ...
def verify_against_rubric(results: dict, rubric: dict) -> dict: ...
def propose_improvements(current_results: dict,
                        rubric_scores: dict,
                        k: int | None = None) -> list[dict]: ...
def set_final(report: dict) -> None: ...
```

REPL variables initialized at bootstrap (brief §7):

```python
paper_text:        str                # full extracted text (PaperExtractor output)
paper_metadata:    dict                # title, authors, sections list, figure/table captions
supplementary_text: str | None         # appendix/supplementary if present
repo_files:        dict[str, str] | None  # filename → content if open-source repo found
prior_work_refs:   list[dict]          # cited prior-work entries
rubric_spec:       dict                # PaperBench-style rubric for this run
# Final: not pre-set. Root emits FINAL_VAR(name) tag → orchestrator reads state[name].
```

REPL functions exposed alongside primitives:

```python
def llm_query(prompt: str, model: str = "default") -> str: ...   # paper's sub_LLM
def rlm_query(context: str, query: str) -> str: ...              # paper's sub_RLM
def print(*args) -> None: ...                                    # stdlib print, captured
```

---

## 3. `rlm_query.py` survives vs. new code

`backend/services/context/workspace/tools/rlm_query.py` is 513 lines, currently the dormant tool wired into the orchestrator with no production caller. Classify each piece:

| Symbol / construct | Disposition | Notes |
|---|---|---|
| `LlmClient` Protocol (`complete(*, system, user) -> str`) | **Survives** — becomes the contract `sub_call.py` calls. | Sync. The async bridge from REPL code → sync `complete()` is a Phase 2 design item (§5). |
| `_RecursionState` dataclass | **Survives** — telemetry shape for `sub_RLM` calls. | Already tracks `max_depth_reached`, `calls_made`, `selection_path`, `hit_truncation_branch` — exactly the fields the `sub_rlm_spawned` event needs. |
| `_LEAF_SYSTEM`, `_SELECT_SYSTEM_TEMPLATE`, `_AGGREGATE_SYSTEM_TEMPLATE` | **Survive as defaults** but get superseded for the root by the new `system_prompt.py`. Leaf/select/aggregate prompts inside sub-calls stay. | The new root prompt follows paper Appendix C — long, includes in-context decomposition examples. |
| `RlmQueryTool.call(workspace_id, question, variable_name, context_key)` | **Survives** as the workspace-routed entry into `rlm_query` (the REPL-exposed function). | Continues to emit `ToolInvoked` / `rlm_query_executed` events. |
| `_recursive_query(content, question, state, *, depth)` | **Survives as the `sub_RLM` engine.** | This is the load-bearing recursion — depth-bounded, call-budgeted, selection/aggregate-shaped. |
| `_leaf_answer(...)` | **Survives** as the `sub_LLM` / `llm_query` leaf path. | When `len(content) ≤ leaf_budget`, this *is* `llm_query`. At depth cap, `rlm_query` falls back here (paper's documented fallback). |
| `ClaudeLlmClient` (in same file) | **Survives** — already wired by `orchestrator._build_rlm_llm_client()`. | Brief correction #5: Claude is not a paper-validated RLM root. Surface root-model choice as config; default to a validated root once we add a GPT-5/Qwen client adapter. |
| `OpenAILlmClient` (now in `tools/openai_client.py`, promoted by PR #56) | **Survives** — already wired. | This is the validated-root path until we add Qwen. |
| Cost gates `_DEFAULT_MAX_LLM_CALLS=24`, `_DEFAULT_MAX_DEPTH=3` | **Survives but defaults change.** | Phase 2: `_DEFAULT_MAX_DEPTH = 2` (correction #1 — depth=2 makes `sub_RLM` real). `_DEFAULT_MAX_LLM_CALLS` stays 24 inside one `rlm_query`; the *run-wide* sub-call cap (50, $10) and the *root-iteration* cap (20) are separate budgets in `sub_call.py` and `root_loop.py`. |
| **NEW** — `repl_host.py` | New | Hosts the persistent `globals` dict, runs `exec(code, namespace)`, captures stdout, serializes to `repl_state.pickle`. |
| **NEW** — `root_loop.py` | New | Algorithm 1. `MAX_ROOT_ITERATIONS=20`, FINAL_VAR-tag parser, `Metadata(stdout)` not `stdout` to history. |
| **NEW** — `sub_call.py` | New | Provides the Python callables `llm_query` and `rlm_query` that the REPL exposes. Bridges sync `LlmClient.complete()` from inside `exec`-ed code, handles the depth>1 → `sub_RLM` decision, falls back to `llm_query` at depth cap. |
| **NEW** — `system_prompt.py` | New | Root-model system prompt adapted from paper Appendix C. Long, includes ≥1 in-context decomposition example (Fig 4a), describes REPL variables and primitives by signature only (no contents). Per-model addenda where needed (Qwen anti-over-subcalling line). |
| **NEW** — `primitives.py` | New | Registry of the 10 callable primitives in §2. Each emits a `primitive_call` SSE event and updates `cost_ledger.jsonl`. |

---

## 4. Reference implementation — mirror vs. adapt

`alexzhang13/rlm` package layout (from `/repos/alexzhang13/rlm/contents/rlm`):

```
rlm/
├── __init__.py
├── clients/         # OpenAI, Anthropic, OpenRouter, Portkey adapters
├── core/            # the root loop + REPL host
├── environments/    # variants of the REPL environment (sandbox shapes)
└── ...
```

| Reference module | Mirror or adapt | Our equivalent |
|---|---|---|
| `rlm/core/` (root loop, REPL) | **Adapt.** Algorithm 1 mechanics mirror; integration points (event bus, run-state checkpointing, `RunBudget`) are ours. | `backend/agents/rlm/repl_host.py`, `root_loop.py` |
| `rlm/clients/` | **Mirror shape** (per-provider files); we already have `ClaudeLlmClient`, `OpenAILlmClient`. | `backend/services/context/workspace/tools/{rlm_query.py, openai_client.py}` |
| `rlm/environments/` | **Diverge.** Their environments are REPL variants; ours is a fixed REPL with `paper_text` + primitives that wrap our existing sandbox (Docker/RunPod). | `backend/agents/rlm/repl_host.py` |
| `REPL_SYSTEM_PROMPT_QWEN.txt` (paper Appendix C) | **Adapt verbatim, then specialize.** | `backend/agents/rlm/system_prompt.py` |
| `FINAL(...)` / `FINAL_VAR(...)` tag parsing (paper Appendix B says termination is "brittle") | **Mirror** the regex + safeguards. | `root_loop.py` — `parse_final_tag()` helper |

---

## 5. Paper-anchored design notes (paper-accuracy corrections applied)

**Algorithm 1 (paper §2):** Given prompt P, init persistent REPL with P as a variable + `sub_RLM` function. Root receives only constant-size metadata about the variable on each turn. Loop runs until termination is detected.

```
state = InitREPL(prompt=P)
state = AddFunction(state, sub_RLM_M)
hist = [Metadata(state)]
for iter in range(MAX_ROOT_ITERATIONS):       # paper Appendix A: 20 iterations
    code = LLM_M(hist)                         # ≤ 4096 output tokens per turn
    state, stdout = REPL(state, code)
    hist = hist || code || Metadata(stdout)    # Metadata(stdout), NOT stdout
    name = parse_final_var_tag(code or stdout) # paper Appendix B
    if name is not None and name in state:
        return state[name]
raise RootIterationCapExceeded
```

**Algorithm 2 (the wrong version, paper §2):** the paper annotates three flaws on Algorithm 2's pseudocode — (#1) P put into `hist` directly, (#2) `Finish` returns the model's autoregressive value, (#3) `RUN` returns raw stdout into `hist`. Brief §13 FM#1–#3 codify these as anti-patterns. The brief's correction #7 extends FM#1 down one level: **primitives must not receive whole-corpus args either** (§1 of this doc).

**Termination (paper Appendix B, "brittle"):** Reference impl parses `FINAL(text)` (autoregressive — failure-mode-prone) and `FINAL_VAR(name)` (reads from a REPL variable — the correct path). The brief's correction #2: terminate by parsing a `FINAL_VAR(name)` tag, then read the answer from the named variable. Allow `FINAL(text)` only with a safeguard (e.g. discard model trajectories that emit `FINAL(plan)` instead of an answer).

**System prompt (paper Appendix C, Fig 4a):** Includes in-context decomposition examples. Figure 4(a) shows these "greatly improve both overall performance and the initial decomposition attempt... even if the example is unrelated." Drop the 2000-token cap.

**Recursion depth (paper Table 1):** Default depth=2. At depth=1 the REPL exposes only `sub_LLM` (`llm_query`); `sub_RLM` exists only at depth>1. At the depth cap, `rlm_query` falls back to `llm_query` (the reference impl's documented behavior).

**Root model choice (paper §3.2, §4):** GPT-5 and Qwen3-Coder-480B-A35B are the validated RLM roots. Claude (Opus 4.1) appears only as a baseline coding agent. Pattern: strong root + cheaper sub-call model (paper uses GPT-5 root / GPT-5-mini sub). Per-model system-prompt addenda are required — paper's Qwen diff adds an anti-over-subcalling line:

> IMPORTANT: Be very careful about using `llm_query` as it incurs high runtime costs. Always batch as much information as reasonably possible into each call (aim for around ~200k characters per call)...

**Cost caps (brief §13 FM#10 + correction #8):**
- Per-`rlm_query` invocation: `max_depth=2`, `max_llm_calls=24` (existing `rlm_query.py` defaults except depth).
- Per-run sub-call cap: **50 sub-calls, $10 cost** (configurable). Sub-call guard only.
- Per-run root-iteration cap: **20 iterations** (paper Appendix A). Separate budget.
- Reproduction-run budget (Docker builds, experiment wall-clock): existing `RunBudget` fields, independent of the above.

---

## 6. Resolved design decisions for Phase 2 (2026-05-20)

Confirmed during Phase 1 review. These are the Phase 2 (#59) defaults — deviation requires re-opening the question.

1. **Sync/async bridge — worker-thread.** Orchestrator's async loop owns a worker thread that runs the synchronous `exec(code, namespace)`. `LlmClient.complete()` stays sync; `llm_query` / `rlm_query` block the worker thread, not the event loop. Matches the paper's reference implementation. Rejected: `nest_asyncio` (fragile on Windows), making `LlmClient` async (larger blast radius).
2. **REPL serialization — file refs for large strings.** `paper_text`, `supplementary_text`, and other large-string variables are stored on disk; the pickle holds the path under `_paper_text_path`, `_supplementary_text_path`, etc. Non-picklable handles (threads, open files) stripped on serialize, re-issued as pending on resume.
3. **`Metadata(stdout)` schema — frozen.** `{length: int, prefix: str (≤200 chars), has_traceback: bool, var_assignments: list[str]}`. `var_assignments` extracted by `ast.parse(code)` walking `ast.Assign` / `ast.AugAssign` targets, so the root sees *what* it bound without seeing the value.
4. **Root model selection — `REPROLAB_RLM_ROOT_MODEL` env var.** Layered on top of existing `REPROLAB_FORCE_LLM_PROVIDER`. Default: a paper-validated root (GPT-5 if OpenAI key present, else Qwen3-Coder-480B). Claude permitted but emits a `root_model_unvalidated` warning at run-start.
5. **`run_experiment` sandbox bridge — sync wrapper.** The primitive is a sync function that schedules `experiment_runner.run_with_runtime(...)` on the orchestrator's event loop via `asyncio.run_coroutine_threadsafe` (from inside the worker thread), then blocks on the resulting future. Same pattern any async-touching primitive uses.
6. **`propose_improvements` prompt — full rewrite, FM#4 validated.** Replace hardcoded `PPO_HYPOTHESES[:n_paths]` with an LLM prompt that returns a variable-length list of `{id, title, tag, description, reasoning, expected_delta}` dicts. Tags are proposer-assigned free-form strings — no taxonomy. FM#4 test: run on 3 distinct PaperBench papers, assert candidate lists differ in count or content.

---

## 7. Phase 1 done-condition checklist

- [x] Branch `rlm-pivot` cut off main, carrying current uncommitted state
- [x] RLM paper indexed (`rlm-paper-html`, 75 sections)
- [x] Reference implementation indexed (`rlm-reference-readme`, `rlm-ref-tree`)
- [x] Each stage agent's core function identified (§1, §2)
- [x] Function signatures for all brief §7 primitives (§2)
- [x] `rlm_query.py` survives-vs-new audit (§3)
- [x] Reference repo mirror/adapt table (§4)
- [x] Paper-anchored notes with corrections applied (§5)
- [x] Open design items surfaced (§6)
- [ ] `backend/agents/rlm/` skeleton with module stubs — produced alongside this doc; `import backend.agents.rlm` succeeds

When the last item is checked, Phase 1 is done; Phase 2 (#59) is unblocked.
