# Phase 2 Execution Brief — shared context for every implementer subagent

> This is the cross-task context for executing `docs/design/phase2-implementation-plan.md`.
> The controller pastes this brief into **every** Phase 2 subagent dispatch, ahead of the
> specific task's full text. A subagent has zero session history — this brief plus the
> task text is everything it gets. It is the distilled result of a long analysis +
> verification effort; **treat its facts as established — do not relitigate them.**

## 1. What you are building

The OpenResearch / ReproLab repo is mid-pivot from a 14-stage pipeline to an RLM
(Recursive Language Model) orchestrator built on the `rlms` PyPI library (`pip install
rlms`, import `rlm`). **Phase 2 (issue #59)** extracts each surviving stage agent's core
logic into a callable "primitive" in `backend/agents/rlm/primitives.py`, wraps each with
a `primitive_call` SSE event + a `cost_ledger.jsonl` row, and assembles them into the
`custom_tools` dict that `rlm.RLM(...)` consumes. The RLM root model writes Python in a
REPL and calls these primitives to reproduce a paper.

You implement one task at a time from the plan. **The plan is the spec.** Each task's
code blocks are complete and exact — implement them as written; they are not sketches.

## 2. Architecture

- **`RunContext`** (`backend/agents/rlm/context.py`) — a dataclass carrying everything a
  primitive needs that the root model does NOT pass: `project_id`, `project_dir`,
  `runs_root`, `dashboard`, `cost_ledger`, `llm_client`, `provider`, `model`, `runtime`,
  `workspace_service`.
- **Primitives** (`backend/agents/rlm/primitives.py`) — plain functions
  `name(<slices/specs>, *, ctx: RunContext) -> dict|str|list`. Heuristic primitives call
  existing stage-agent helpers; LLM-backed primitives call `ctx.llm_client.complete()`.
- **Binding** (`backend/agents/rlm/binding.py`) — `build_custom_tools(ctx)` closes each
  primitive over `ctx` and wraps it with event + ledger emission, producing the dict
  `rlm.RLM(custom_tools=...)` expects: `{name: {"tool": callable, "description": str}}`.
- **The 9 primitives** (brief §7): `understand_section`, `extract_hyperparameters`,
  `detect_environment`, `build_environment`, `plan_reproduction`, `implement_baseline`,
  `run_experiment`, `verify_against_rubric`, `propose_improvements`. (The Phase-1
  skeleton's 10th stub `set_final` is vestigial under `rlm`'s `FINAL_VAR`
  termination — plan Task 12 removes it.)

## 3. Verified `rlm` 0.1.1 facts — established, do not relitigate

- **The root REPL must use `environment="local"`.** `rlm`'s `DockerREPL` does NOT inject
  `custom_tools` into the in-container globals (verified, `rlm/environments/docker_repl.py`).
  Docker is used only *inside* the `build_environment` / `run_experiment` primitives, for
  the paper's sandbox — never by the root REPL.
- **Termination** is a `FINAL_VAR(name)` / `FINAL(text)` tag the model emits — there is
  **no reserved `answer` variable**. The reserved REPL variable is `context` (the
  offloaded prompt).
- **`on_iteration_start/complete` callbacks are declared but never fire** in `rlm` 0.1.1.
  `on_subcall_start/complete` DO fire. Per-iteration observability comes from a custom
  `RLMLogger` — never from `on_iteration_*`.
- **`custom_tools` entry format:** `{"name": {"tool": callable_or_value, "description": str}}`.
- `RLM(max_depth=2)` enables a real recursive child RLM via `rlm_query`; the constructor's
  `max_iterations` default is 30 (the paper's Appendix-A bound is 20).

## 4. Design decisions (full text: plan §"Design decisions & residual risks")

D0 root REPL `environment="local"`. D1 `env_id` is a Docker image tag; `build_environment`
returns one, `run_experiment` consumes it (no rebuild). D2 run commands travel via
`code/commands.json` (`implement_baseline` writes it, `run_experiment` reads it). D3
`build_environment` runs its own retry loop around `build_image`, repairs the Dockerfile
via `ctx.llm_client`, fail-soft. D4 primitives take `*, ctx: RunContext`; `build_custom_tools`
closes `ctx` in. D5 `understand_section` is the title-agnostic heuristic subset. D6
LLM-backed primitives use `ctx.llm_client.complete()` with existing prompt constants. D7
the wrapper records a zero-usage `CostLedgerEntry` per call (real token usage lands in
Phase 3).

## 5. Verified signatures of external code you will call

- `environment_detective.run_offline(project_id, runs_root, paper_claim_map: PaperClaimMap, artifact_index=None) -> EnvironmentSpec` — sync, no LLM.
- `baseline_implementation.run_with_sdk(project_id, runs_root, paper_claim_map, environment_spec, reproduction_contract=None, artifact_index=None, *, model=None, provider=None, runtime=None) -> BaselineResult` — async.
- `RuntimeAppService(backend)` (`backend/services/runtime/service.py`): `async create_sandbox(CreateSandbox(config=SandboxConfig)) -> Sandbox`, `async execute(ExecuteCommand(sandbox=, command=, timeout=int)) -> ExecResult` (`.succeeded` property), `async destroy(DestroySandbox(sandbox=))`. `SandboxConfig` requires `project_id`, `run_id`, `project_root`.
- `local_docker.build_image(dockerfile_path, context_dir, tag, *, timeout=..., client=None) -> tuple[bool, str, str]` — async; `BuildError` is repairable, `SandboxRuntimeError` is an infra failure that propagates.
- `RubricVerification.from_areas(areas: list[RubricAreaScore], *, rubric_source: Literal["paperbench_bundle","generated"], target_score: float, confidence=0.0, verified_at="") -> RubricVerification` — needs `RubricAreaScore` OBJECTS; recomputes `overall_score`/`meets_target` itself.
- `CostLedgerEntry(timestamp, agent_id, attempt_index, provider, model, input_tokens=0, …, estimated_usd=None)` — only the first 5 are required (`resilience/cost.py`).
- `DashboardEmitter(project_id, runs_root)` writes `dashboard_events.jsonl`; `_emit(dict)` and the module helper `_now()` already exist.
- `paper_understanding._extract_datasets/_extract_metrics/_extract_training_recipe/_extract_hardware/_extract_ambiguities` all take `sections: dict[str, str]` and are title-agnostic.

## 6. Codebase map

- Primitives + binding + context: `backend/agents/rlm/` (the Phase-1 skeleton lives here).
- Stage agents: `backend/agents/{paper_understanding,environment_detective,baseline_implementation,experiment_runner,verification,improvement}.py`.
- Schemas (Pydantic): `backend/agents/schemas.py`.
- Prompts: `backend/agents/prompts/`.
- Runtime/sandbox: `backend/services/runtime/`.
- Event emitter / cost ledger: `backend/agents/dashboard_emitter.py`, `backend/agents/resilience/cost.py`.
- Tests go in `tests/rlm/` (pytest `testpaths=["tests"]`).

## 7. Hard constraints

- **Python 3.14.2.** Use `.venv/bin/python` and `.venv/bin/pip` for everything. Run tests
  with `.venv/bin/python -m pytest tests/rlm/<file> -v`.
- Work on branch **`feat/rlm-phase2-foundation`** (already checked out).
- **TDD** — write the failing test first, watch it fail, implement, watch it pass.
- **Algorithm-2 guard** — a primitive's call signature takes slices / structured specs,
  never a whole-corpus argument and never `project_id`/`runs_root` (those come from `ctx`).
- Touch only the files the task names. Follow the plan's file structure.
- End the task with the plan's commit step; commit messages end with the trailer
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- If a verified signature in §5 contradicts what you find in source, trust source and
  report it — but §5 was checked against the installed code on 2026-05-21.
