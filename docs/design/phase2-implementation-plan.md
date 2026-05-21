# RLM Phase 2 — Domain Primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract each surviving stage agent's core logic into a callable "primitive" in `backend/agents/rlm/primitives.py`, wrap each with `primitive_call` SSE emission + cost-ledger recording, and assemble them into the `custom_tools` dict that `rlm.RLM(...)` consumes — closing out issue #59.

**Architecture:** Each primitive is a plain function `name(<slices/specs>, *, ctx: RunContext) -> <dict|str|list>`. The root RLM model passes only slices and structured specs (the "Algorithm-2 guard" — never the corpus, never `project_id`/`runs_root`). Everything else — paths, the event emitter, the cost ledger, the LLM client, the agent runtime — travels in a `RunContext` that `build_custom_tools(ctx)` closes over before exposing the callable to the REPL. Stage-agent core logic is **wrapped, not rewritten** (brief §4): heuristic primitives call existing inner helper functions; LLM-backed primitives call `ctx.llm_client.complete()` with the existing prompt constants and parse the result into the existing Pydantic schemas.

**Tech Stack:** Python 3.14.2; `rlms` 0.1.1 (`pip install rlms`, import `rlm`); pytest (`testpaths=["tests"]`); existing `backend/agents/` stage agents, `DashboardEmitter`, `RunCostLedger`, `RuntimeAppService`/`LocalDockerBackend`.

---

## Design decisions & residual risks

The recon (`docs/design/phase2-analysis.md` and a 2026-05-21 codebase sweep) surfaced design forks the stub `primitives.py` did not anticipate. This plan **resolves** them so every task is complete-code. Decisions:

- **D0 — the root RLM REPL runs `environment="local"`, not `"docker"`.** Verified in `rlm` 0.1.1 source: the module-level function `_build_exec_script` in `rlm/environments/docker_repl.py` (it is a module function, not a `DockerREPL` method) builds the in-container globals as `{__builtins__, __name__, llm_query, llm_query_batched, FINAL_VAR, SHOW_VARS}` — it **never injects `custom_tools`** (and never exposes `rlm_query`). Under `environment="docker"` the 10 primitives would not exist in the REPL namespace at all (`NameError`), and would otherwise execute *inside* the container with no host paths and no host Docker socket. `custom_tools` are plumbed only by `LocalREPL`. Phase 2's primitives are host-side by design; the root REPL **must** use `environment="local"`. Docker is used *by* the `build_environment` / `run_experiment` primitives for the paper's sandbox — never by the root REPL. ⚠ **Upstream conflict:** issue #60 deliverable 2 specifies `environment='docker'` and `rlms-spike-report.md` argues Docker was the deciding factor over `dspy.RLM` — both are inconsistent with this verified fact and need correcting (the spike report's point still holds *for the primitives' sandbox*, just not for the root REPL).
- **D1 — `env_id` is a Docker image tag.** No `env_id` concept exists today. `build_environment()` returns `{"image_tag": str, ...}`; `run_experiment(code_path, env_id)` treats `env_id` as a prebuilt image tag and runs a container from it with `SandboxConfig(image=env_id, dockerfile_path=None)` — no rebuild.
- **D2 — run commands travel via `commands.json`.** `run_with_runtime` reads commands from a `BaselineResult` today; the `(code_path, env_id)` signature has none. `implement_baseline()` writes `code/commands.json` (a JSON list of shell commands); `run_experiment()` reads it. The skeleton signatures are preserved.
- **D3 — `build_environment` does its own retry loop.** It wraps `build_image()` (the clean inner), reconstructs the retry loop (the orchestrator's loop is fused with `PipelineState`), regenerates the Dockerfile on failure via `ctx.llm_client`, caps at `environment_build_max_attempts`, and is **fail-soft** (returns `{"ok": False, ...}`, never raises) — except `SandboxRuntimeError` (Docker daemon down etc.), which is an infrastructure failure and propagates.
- **D4 — primitives take `*, ctx: RunContext`.** `build_custom_tools(ctx)` partials `ctx` in and wraps with event+ledger emission, so the REPL-exposed callable's signature is just the slice/spec args. The Algorithm-2 guard governs the root-passed args, not `ctx`.
- **D5 — `understand_section` is the heuristic subset.** `paper_understanding.py`'s inner helpers split into *title-agnostic* (work on a bare slice) and *title-aware* (silently degrade on an untitled slice). `understand_section` implements the five title-agnostic helpers only (verified against `paper_understanding.py` source — Task 3 lists the line numbers); `core_contribution`/`claims`/`model_architecture`/`evaluation_protocol` are left for the root to extract with `llm_query` over `context`. The returned dict is a documented **partial** `PaperClaimMap`.
- **D6 — LLM-backed primitives use `ctx.llm_client.complete()`** with the existing prompt constants from `backend/agents/prompts/`, returning JSON parsed into existing schemas. `implement_baseline` is the exception — it is a code-*writing* agent, so it wraps `run_with_sdk()` via `ctx.runtime`.
- **D7 — cost ledger in Phase 2 records a call entry per primitive with zero token usage.** The simple `LlmClient.complete()` protocol returns text only. Phase 3 (`run.py`, #60) supplies a usage-returning client; until then the wrapper appends a `CostLedgerEntry` with `input_tokens=output_tokens=0`, `estimated_usd=None`, so the ledger is a faithful *call* log and real cost attribution slots in later.

**Residual risks** (flagged at the relevant tasks): `build_environment`/`run_experiment` need a Docker daemon to run end-to-end — their unit tests mock the Docker layer; `implement_baseline`'s `run_with_sdk` is a heavier code-writing agent and is exercised against a fake runtime in tests; the sync↔async bridge (`asyncio.run` inside a worker thread) assumes the REPL host runs primitives off the event loop — re-verify when `run.py` lands.

## File structure

| File | Status | Responsibility |
|---|---|---|
| `backend/requirements.txt` | modify | add `rlms` |
| `backend/agents/rlm/context.py` | create | the `RunContext` dataclass |
| `backend/agents/dashboard_emitter.py` | modify | add a `primitive_call` event method |
| `backend/agents/rlm/binding.py` | create | `wrap_primitive`, `build_custom_tools`, `PRIMITIVE_DESCRIPTIONS` |
| `backend/agents/rlm/primitives.py` | modify | implement the 9 primitive bodies; drop the vestigial `set_final` stub |
| `backend/agents/rlm/__init__.py` | modify | export `RunContext`, `build_custom_tools` |
| `tests/rlm/conftest.py` | create | `FakeLlmClient`, the `make_context` fixture |
| `tests/rlm/test_*.py` | create | one test module per task |

## Build order

```
Task 1 (RunContext + scaffold) ─→ Task 2 (wrapper) ─→ Tasks 3-12 (primitives, SEQUENTIAL) ─→ Task 13 (registry) ─→ Task 14 (integration)
```
Tasks 3–12 are **strictly sequential — not parallel**: every one of them edits the same file `backend/agents/rlm/primitives.py`, and Task 7 defines `_extract_json` which Tasks 10–11 import. Do them in number order, one at a time. Subagent-driven execution (one fresh subagent per task, reviewed between) is sequential per-task and is fine; **never** dispatch primitive tasks concurrently — concurrent edits to `primitives.py` will clobber. Each task ends with a commit. **Task 0** (upstream reconciliation) is a preflight — a doc/issue correction, no code — and must be done before Task 1.

**Commit convention:** end every `git commit` message in this plan with a trailing line `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` (project convention). Use `git commit -F-` with a heredoc to include it.

---

### Task 0: Upstream reconciliation — pin `environment="local"` (preflight)

Design decision D0 established that `rlm`'s `DockerREPL` does not inject `custom_tools`, so the root REPL must be `environment="local"`. Two upstream artifacts still say `environment="docker"`; correct them **before Task 1** so Phase 3's `run.py` is not built against a broken config. This task is a doc/issue correction — no code, no test.

**Files / artifacts:**
- Edit on GitHub: issue #60 body
- Modify: `docs/design/rlms-spike-report.md`

- [ ] **Step 1: Correct issue #60.** Run `gh issue view 60 --json body -q .body`; in the `run.py` deliverable change `environment='docker'` to `environment='local'` and append: "the root REPL is `environment='local'` — `rlm`'s `DockerREPL` does not inject `custom_tools` (verified, `rlm/environments/docker_repl.py`); Docker is used only *inside* the `build_environment` / `run_experiment` primitives." Apply with `gh issue edit 60 --body-file <edited-file>`.

- [ ] **Step 2: Correct `docs/design/rlms-spike-report.md`.** Its Docker-as-deciding-factor argument conflated two things. Add a correction paragraph: the `rlm`-over-`dspy.RLM` verdict still holds, but the reason is that the *primitives* drive Docker host-side — **not** that the root REPL is `environment='docker'` (which would put the REPL in a container with no `custom_tools` and no host Docker socket). State that the root REPL is `environment='local'`.

- [ ] **Step 3: Commit the doc change.** `git add docs/design/rlms-spike-report.md` and commit (with the `Co-Authored-By` trailer). The issue #60 edit lives on GitHub, not in the commit.

**Done condition:** no ReproLab issue or design doc instructs Phase 3 to build `rlm.RLM(environment='docker')`.

---

### Task 1: `RunContext`, test scaffold, and the `rlms` dependency

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/agents/rlm/context.py`
- Create: `tests/rlm/conftest.py`
- Test: `tests/rlm/test_context.py`

- [ ] **Step 1: Add dependencies and install them**

`rlms` 0.1.1 requires `python-dotenv>=1.2.1`. The env currently has `python-dotenv` 1.0.1, and `backend/requirements.txt` does not pin `python-dotenv` at all — so installing `rlms` leaves it with an incompatible transitive dep (and `pip` will say so). Append **both** lines to `backend/requirements.txt` (the `<2` upper bound keeps `deepeval` — which needs `python-dotenv>=1.1.1,<2` — satisfied):

```
rlms==0.1.1
python-dotenv>=1.2.1,<2
```

Run: `.venv/bin/pip install -r backend/requirements.txt && .venv/bin/pip check`
Expected: installs/resolves and **upgrades `python-dotenv` from 1.0.1**; `pip check` reports "No broken requirements found." If `pip check` still flags `python-dotenv`, run `.venv/bin/pip install 'python-dotenv>=1.2.1,<2'` explicitly and re-check.

- [ ] **Step 2: Write the failing test for `RunContext`**

Create `tests/rlm/test_context.py`:

```python
from pathlib import Path

from backend.agents.rlm.context import RunContext


def test_run_context_holds_run_scoped_dependencies(tmp_path: Path):
    from backend.agents.dashboard_emitter import DashboardEmitter
    from backend.agents.resilience.cost import RunCostLedger

    project_dir = tmp_path / "prj"
    project_dir.mkdir()
    ctx = RunContext(
        project_id="prj",
        project_dir=project_dir,
        runs_root=tmp_path,
        dashboard=DashboardEmitter("prj", tmp_path),
        cost_ledger=RunCostLedger.load_jsonl(
            project_dir / "cost_ledger.jsonl", project_id="prj", attach_path=True
        ),
        llm_client=object(),
        provider="anthropic",
        model="test-model",
    )
    assert ctx.project_id == "prj"
    assert ctx.runtime is None
    assert ctx.workspace_service is None
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_context.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.agents.rlm.context'`.

- [ ] **Step 4: Create `backend/agents/rlm/context.py`**

```python
"""RunContext — run-scoped dependencies threaded into every RLM primitive.

Phase 2 (issue #59). The root RLM model passes only slices/specs as primitive
arguments (the Algorithm-2 guard). Everything else a primitive needs — paths,
the event emitter, the cost ledger, the LLM client, the agent runtime — lives
here and is closed over by `backend.agents.rlm.binding.build_custom_tools`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    """Everything a primitive needs that the root model does not pass.

    `llm_client` is the synchronous `LlmClient` protocol from
    `backend/services/context/workspace/tools/rlm_query.py` — `.complete(*,
    system, user) -> str`. `runtime` is an `AgentRuntime`; only
    `implement_baseline` needs it, so it defaults to None.
    """

    project_id: str
    project_dir: Path
    runs_root: Path
    dashboard: Any            # DashboardEmitter
    cost_ledger: Any          # RunCostLedger
    llm_client: Any           # LlmClient protocol: .complete(*, system, user) -> str
    provider: str             # "anthropic" | "openai"
    model: str
    runtime: Any = None       # AgentRuntime — only implement_baseline uses it
    workspace_service: Any = None
    workspace_id: str | None = None
```

- [ ] **Step 5: Create the test scaffold `tests/rlm/conftest.py`**

```python
"""Shared fixtures for RLM primitive tests (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.dashboard_emitter import DashboardEmitter
from backend.agents.resilience.cost import RunCostLedger
from backend.agents.rlm.context import RunContext


class FakeLlmClient:
    """Counting fake LlmClient. Returns scripted responses in order (last repeats)."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.calls: list[dict] = []
        self._responses = responses or ["{}"]

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


@pytest.fixture
def make_context():
    """Factory fixture: build a RunContext rooted at a tmp dir."""

    def _make(tmp_path: Path, llm_responses: list[str] | None = None,
              project_id: str = "test_proj") -> RunContext:
        project_dir = tmp_path / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        return RunContext(
            project_id=project_id,
            project_dir=project_dir,
            runs_root=tmp_path,
            dashboard=DashboardEmitter(project_id, tmp_path),
            cost_ledger=RunCostLedger.load_jsonl(
                project_dir / "cost_ledger.jsonl",
                project_id=project_id,
                attach_path=True,
            ),
            llm_client=FakeLlmClient(llm_responses),
            provider="anthropic",
            model="test-model",
        )

    return _make
```

`RunCostLedger.load_jsonl` is called here on a path that does not exist yet; it tolerates a missing file (the orchestrator calls it identically on fresh runs). If Task 1's test errors at this line, sanity-check with `.venv/bin/python -c "from pathlib import Path; from backend.agents.resilience.cost import RunCostLedger; print(RunCostLedger.load_jsonl(Path('/tmp/nope.jsonl'), project_id='x', attach_path=True))"`.

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_context.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/requirements.txt backend/agents/rlm/context.py tests/rlm/test_context.py tests/rlm/conftest.py
git commit -m "feat(rlm): RunContext + Phase 2 test scaffold"
```

---

### Task 2: Primitive wrapper — `primitive_call` event + cost ledger

**Files:**
- Modify: `backend/agents/dashboard_emitter.py`
- Create: `backend/agents/rlm/binding.py`
- Test: `tests/rlm/test_binding.py`

- [ ] **Step 1: Write the failing test**

Create `tests/rlm/test_binding.py`:

```python
import json

from backend.agents.rlm.binding import build_custom_tools


def test_wrapped_primitive_emits_event_and_ledger_row(make_context, tmp_path):
    ctx = make_context(tmp_path)
    registry = {"echo": lambda value, *, ctx: {"echoed": value}}
    tools = build_custom_tools(ctx, registry=registry, descriptions={"echo": "echo a value"})

    assert set(tools["echo"]) == {"tool", "description"}
    result = tools["echo"]["tool"]("hi")
    assert result == {"echoed": "hi"}

    events = [json.loads(ln) for ln in
              (ctx.project_dir / "dashboard_events.jsonl").read_text().splitlines() if ln]
    pe = [e for e in events if e.get("event") == "primitive_call"]
    assert [e["status"] for e in pe] == ["start", "ok"]
    assert pe[0]["primitive"] == "echo"

    ledger = [json.loads(ln) for ln in
              (ctx.project_dir / "cost_ledger.jsonl").read_text().splitlines() if ln]
    assert ledger[-1]["agent_id"] == "echo"


def test_wrapped_primitive_records_failure(make_context, tmp_path):
    ctx = make_context(tmp_path)

    def boom(*, ctx):
        raise ValueError("bad")

    tools = build_custom_tools(ctx, registry={"boom": boom}, descriptions={"boom": "fails"})
    try:
        tools["boom"]["tool"]()
    except ValueError:
        pass
    events = [json.loads(ln) for ln in
              (ctx.project_dir / "dashboard_events.jsonl").read_text().splitlines() if ln]
    statuses = [e["status"] for e in events if e.get("event") == "primitive_call"]
    assert statuses == ["start", "error"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_binding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.agents.rlm.binding'`.

- [ ] **Step 3: Add a `primitive_call` method to `DashboardEmitter`**

In `backend/agents/dashboard_emitter.py`, add this method to the `DashboardEmitter` class, alongside the existing `agent_started` / `verification_gate` methods. The field names follow **issue #61's Phase-4 event table** for `primitive_call` (`iteration, primitive, args_summary, status, result_summary, rubric_delta`) so Phase 4 can consume them. `_now()` is the module's existing timestamp helper.

```python
    def primitive_call(
        self,
        primitive: str,
        status: str,
        *,
        args_summary: dict | None = None,
        result_summary: str | None = None,
        iteration: int | None = None,
        rubric_delta: float | None = None,
    ) -> None:
        """Emit a `primitive_call` event (RLM Phase 2 — issue #61 schema).

        `status` is "start" | "ok" | "error". `iteration` is the root-loop
        index — a bare primitive wrapper cannot know it, so it is None here.
        Phase 3 (`run.py`, #60) supplies it via a custom `RLMLogger` subclass
        passed to `rlm.RLM`: `rlm` calls `logger.log(iteration)` once per loop
        (the verified per-iteration hook — `on_iteration_*` never fire), and
        that subclass stashes the index for the wrapper to read.
        `rubric_delta` is not applicable to a primitive call (always None).
        """
        self._emit({
            "event": "primitive_call",
            "timestamp": _now(),
            "primitive": primitive,
            "status": status,
            "args_summary": args_summary or {},
            "result_summary": result_summary,
            "iteration": iteration,
            "rubric_delta": rubric_delta,
        })
```

> **Casing note — flag for review.** Issue #61's new event table is snake_case (`args_summary`); the existing `DashboardEmitter` events are camelCase (`currentTask`, `argsSummary`-style). This plan follows #61's snake_case for the new `primitive_call` fields. The casing inconsistency between #61's new events and the existing emitter convention is an **upstream decision** — raise it on #61 (or in the #59 review), since #61 says the event schema is "Blocked by #59 — must be stable." Do not invent a third convention.

- [ ] **Step 4: Create `backend/agents/rlm/binding.py`**

```python
"""Bind primitives to a RunContext and assemble the rlm `custom_tools` dict.

Phase 2 (issue #59). `build_custom_tools(ctx)` produces the dict
`rlm.RLM(custom_tools=...)` consumes: `{name: {"tool": callable, "description": str}}`.
Each wrapped callable emits a `primitive_call` SSE event (start + complete) and
appends a row to `cost_ledger.jsonl`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from backend.agents.resilience.cost import CostLedgerEntry
from backend.agents.rlm.context import RunContext


def _summarize(args: tuple, kwargs: dict) -> dict:
    """A short, value-free summary of a primitive call's arguments."""
    out = {f"arg{i}": f"{type(a).__name__}[{len(a)}]" if hasattr(a, "__len__")
           else type(a).__name__ for i, a in enumerate(args)}
    out.update({k: type(v).__name__ for k, v in kwargs.items()})
    return out


def _result_summary(result: Any) -> str:
    """A short, value-free summary of a primitive's return value."""
    if isinstance(result, dict):
        return f"dict[{', '.join(sorted(map(str, result))[:6])}]"
    if isinstance(result, (list, str)):
        return f"{type(result).__name__}[{len(result)}]"
    return type(result).__name__


def wrap_primitive(name: str, fn: Callable[..., Any], ctx: RunContext) -> Callable[..., Any]:
    """Close `fn` over `ctx`, adding primitive_call emission and a cost-ledger row."""

    def _ledger() -> None:
        # Phase 2 (D7): a zero-usage call entry; real token usage lands with run.py (#60).
        ctx.cost_ledger.append(CostLedgerEntry(
            timestamp=datetime.now(timezone.utc),
            agent_id=name,
            attempt_index=0,
            provider=ctx.provider,
            model=ctx.model,
        ))

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        ctx.dashboard.primitive_call(name, "start", args_summary=_summarize(args, kwargs))
        try:
            result = fn(*args, ctx=ctx, **kwargs)
        except Exception as exc:
            ctx.dashboard.primitive_call(
                name, "error", result_summary=f"{type(exc).__name__}: {exc}"[:200])
            _ledger()
            raise
        ctx.dashboard.primitive_call(name, "ok", result_summary=_result_summary(result))
        _ledger()
        return result

    wrapped.__name__ = name
    return wrapped


def build_custom_tools(
    ctx: RunContext,
    *,
    registry: dict[str, Callable[..., Any]] | None = None,
    descriptions: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Return the rlm `custom_tools` dict, every primitive closed over `ctx`."""
    if registry is None or descriptions is None:
        # Lazy import: keeps this module usable before Task 13 adds
        # PRIMITIVE_DESCRIPTIONS, and lets callers pass both explicitly.
        from backend.agents.rlm import primitives as _p
        registry = registry if registry is not None else _p.PRIMITIVE_REGISTRY
        descriptions = descriptions if descriptions is not None else _p.PRIMITIVE_DESCRIPTIONS
    return {
        name: {"tool": wrap_primitive(name, fn, ctx),
               "description": descriptions.get(name, name)}
        for name, fn in registry.items()
    }
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_binding.py -v`
Expected: PASS (2 tests). The test passes `registry=` and `descriptions=` explicitly, so `build_custom_tools` skips the lazy import of `PRIMITIVE_REGISTRY` / `PRIMITIVE_DESCRIPTIONS` (those land in Task 13).

- [ ] **Step 6: Commit**

```bash
git add backend/agents/dashboard_emitter.py backend/agents/rlm/binding.py tests/rlm/test_binding.py
git commit -m "feat(rlm): primitive wrapper — primitive_call event + cost ledger"
```

---

### Task 3: `understand_section` primitive

**Files:**
- Modify: `backend/agents/rlm/primitives.py:23-29`
- Test: `tests/rlm/test_understand_section.py`

- [ ] **Step 1: Write the failing test**

Create `tests/rlm/test_understand_section.py`:

```python
from backend.agents.rlm.primitives import understand_section

SLICE = (
    "We train with the Adam optimizer at learning rate 3e-4, batch size 64, "
    "for 200 epochs. We evaluate on the CartPole-v1 dataset and report mean "
    "reward and success rate."
)


def test_understand_section_returns_partial_claim_map(make_context, tmp_path):
    ctx = make_context(tmp_path)
    result = understand_section(SLICE, ctx=ctx)
    assert set(result) == {"datasets", "metrics", "training_recipe",
                           "hardware_clues", "ambiguities"}
    assert isinstance(result["datasets"], list)
    assert result["training_recipe"]["optimizer"]  # Adam was detected
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_understand_section.py -v`
Expected: FAIL — `NotImplementedError: Phase 3 (#60) — wrap paper_understanding._extract_*`.

- [ ] **Step 3: Verify the helpers are title-agnostic, then implement `understand_section`**

Design decision D5 was verified against `backend/agents/paper_understanding.py` source: the five helpers `understand_section` uses are title-agnostic — `_extract_datasets` (`:162`), `_extract_metrics` (`:185`), `_extract_training_recipe` (`:264`), `_extract_hardware` (`:297`), `_extract_ambiguities` (`:314`) — each builds `" ".join(sections.values())` and **none branch on the section-title key**. (`_extract_metrics` does read the key, but only to label `MetricSpec.source_section`; under `{"_": slice}` that label is `"_"` — a cosmetic artifact, not under-extraction.) The title-*aware* helpers — `_extract_architecture` (`:256`), `_extract_eval_protocol` (`:289`), `_extract_method_name` (`:349`), which **do** branch on titles — are correctly excluded from the adapter. Replace the body of `understand_section` in `backend/agents/rlm/primitives.py`:

```python
def understand_section(text_slice: str, *, ctx: "RunContext") -> dict:
    """Extract datasets/metrics/training-recipe/hardware/ambiguities from a slice.

    Wraps the *title-agnostic* heuristic helpers in
    `backend/agents/paper_understanding.py`. Returns a PARTIAL PaperClaimMap
    dict — `core_contribution`, `claims`, `model_architecture` and
    `evaluation_protocol` need section titles and are left for the root model
    to extract with `llm_query` over `context` (design decision D5).
    """
    from backend.agents.paper_understanding import (
        _extract_datasets, _extract_metrics, _extract_training_recipe,
        _extract_hardware, _extract_ambiguities,
    )
    sections = {"_": text_slice}
    return {
        "datasets": [d.model_dump() for d in _extract_datasets(sections)],
        "metrics": [m.model_dump() for m in _extract_metrics(sections)],
        "training_recipe": _extract_training_recipe(sections).model_dump(),
        "hardware_clues": _extract_hardware(sections),
        "ambiguities": [a.model_dump() for a in _extract_ambiguities(sections)],
    }
```

Keep the existing `from __future__ import annotations` and the `RunContext` import at the top of `primitives.py` (add `from backend.agents.rlm.context import RunContext` under `TYPE_CHECKING`, or import directly — the file already imports `typing`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_understand_section.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/primitives.py tests/rlm/test_understand_section.py
git commit -m "feat(rlm): understand_section primitive (title-agnostic subset)"
```

---

### Task 4: `extract_hyperparameters` primitive

**Files:**
- Modify: `backend/agents/rlm/primitives.py:32-37`
- Test: `tests/rlm/test_extract_hyperparameters.py`

- [ ] **Step 1: Write the failing test**

Create `tests/rlm/test_extract_hyperparameters.py`:

```python
from backend.agents.rlm.primitives import extract_hyperparameters


def test_extract_hyperparameters_flat_dict(make_context, tmp_path):
    ctx = make_context(tmp_path)
    result = extract_hyperparameters(
        "Trained with Adam, learning rate 3e-4, batch size 64, for 200 epochs.",
        ctx=ctx,
    )
    assert set(result) == {"optimizer", "learning_rate", "batch_size",
                           "epochs_or_steps", "scheduler", "other_hparams"}
    # Assert the extracted CONTENT, not just non-emptiness. If the heuristic
    # captures the value with surrounding text, narrow these to the real
    # output — but they must check the extracted value, never `x or x`.
    assert "3e-4" in result["learning_rate"]
    assert "64" in result["batch_size"]
    assert "adam" in result["optimizer"].lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_extract_hyperparameters.py -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `extract_hyperparameters`**

Replace the body in `backend/agents/rlm/primitives.py`:

```python
def extract_hyperparameters(text_slice: str, *, ctx: "RunContext") -> dict:
    """Extract hyperparameters from a slice (typically the training-recipe section).

    Wraps `paper_understanding._extract_training_recipe`. Returns a flat dict:
    optimizer, learning_rate, batch_size, epochs_or_steps, scheduler,
    other_hparams. The heuristic populates the first four; the root model can
    fill scheduler/other_hparams via `llm_query` if needed.

    `ctx` is required by the primitive-wrapper protocol (design decision D4);
    this heuristic body does not use it.
    """
    from backend.agents.paper_understanding import _extract_training_recipe
    return _extract_training_recipe({"_": text_slice}).model_dump()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_extract_hyperparameters.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/primitives.py tests/rlm/test_extract_hyperparameters.py
git commit -m "feat(rlm): extract_hyperparameters primitive"
```

---

### Task 5: `detect_environment` primitive

**Files:**
- Modify: `backend/agents/rlm/primitives.py:40-46`
- Test: `tests/rlm/test_detect_environment.py`

- [ ] **Step 1: Write the failing test**

Create `tests/rlm/test_detect_environment.py`:

```python
from backend.agents.rlm.primitives import detect_environment


def test_detect_environment_produces_env_spec(make_context, tmp_path):
    ctx = make_context(tmp_path)
    method_spec = {"core_contribution": "A PyTorch RL agent.", "claims": [],
                   "datasets": [], "metrics": []}
    result = detect_environment(method_spec, ctx=ctx)
    assert result["python_version"]
    assert result["framework"]
    assert isinstance(result["pip_packages"], dict)
    assert result["dockerfile"].startswith("FROM")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_detect_environment.py -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `detect_environment`**

Replace the body in `backend/agents/rlm/primitives.py`:

```python
def detect_environment(method_spec: dict, *, ctx: "RunContext") -> dict:
    """Infer the runtime environment; return an EnvironmentSpec dict.

    Wraps `environment_detective.run_offline` — the deterministic, no-LLM entry
    point — directly (brief §4 "wrap, not rewrite"). Verified: `run_offline` is
    exactly the heuristic helper chain plus a Dockerfile write into the run
    dir; that file-write side effect is fine — a primitive may write run
    artifacts via `ctx`, and `build_environment` can reuse the written
    Dockerfile. `method_spec` is a (possibly partial) PaperClaimMap dict;
    `PaperClaimMap.core_contribution` is its one *required* field, so it is
    defaulted here — `understand_section`'s output omits it.
    """
    from backend.agents.environment_detective import run_offline
    from backend.agents.schemas import PaperClaimMap

    claim_map = PaperClaimMap(**{"core_contribution": "", **method_spec})
    spec = run_offline(
        ctx.project_id, ctx.runs_root, claim_map, method_spec.get("artifact_index"))
    return spec.model_dump()
```

Verified signature (`environment_detective.py:54`): `run_offline(project_id: str, runs_root: Path, paper_claim_map: PaperClaimMap, artifact_index: dict | None = None) -> EnvironmentSpec` — synchronous, no LLM. `EnvironmentSpec` has `model_config = {"extra": "ignore"}`, so `.model_dump()` is a stable dict.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_detect_environment.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/primitives.py tests/rlm/test_detect_environment.py
git commit -m "feat(rlm): detect_environment primitive"
```

---

### Task 6: `build_environment` primitive (the build-and-repair loop)

**Files:**
- Modify: `backend/agents/rlm/primitives.py:49-56`
- Test: `tests/rlm/test_build_environment.py`

Design (D3): the primitive reconstructs the retry loop around the async `build_image()`; on a Docker `BuildError` it asks `ctx.llm_client` to regenerate the Dockerfile and retries; `SandboxRuntimeError` (daemon down) propagates; fail-soft otherwise.

- [ ] **Step 1: Write the failing test**

Create `tests/rlm/test_build_environment.py`:

```python
import backend.agents.rlm.primitives as primitives
from backend.agents.rlm.primitives import build_environment


def test_build_environment_succeeds_first_try(make_context, tmp_path, monkeypatch):
    ctx = make_context(tmp_path)

    async def fake_build_image(dockerfile_path, context_dir, tag, **kw):
        return (True, tag, "")

    monkeypatch.setattr(primitives, "_build_image", fake_build_image)
    result = build_environment({"dockerfile": "FROM python:3.11-slim\n"}, ctx=ctx)
    assert result["ok"] is True
    assert result["image_tag"]
    assert result["attempts"] == 1


def test_build_environment_repairs_then_succeeds(make_context, tmp_path, monkeypatch):
    ctx = make_context(tmp_path, llm_responses=["FROM python:3.11-slim\nRUN pip install x\n"])
    calls = {"n": 0}

    async def fake_build_image(dockerfile_path, context_dir, tag, **kw):
        calls["n"] += 1
        return (calls["n"] > 1, tag, "" if calls["n"] > 1 else "pip failed")

    monkeypatch.setattr(primitives, "_build_image", fake_build_image)
    result = build_environment({"dockerfile": "FROM bad\n"}, ctx=ctx)
    assert result["ok"] is True
    assert result["attempts"] == 2
    assert len(ctx.llm_client.calls) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_build_environment.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_build_image'`.

- [ ] **Step 3: Implement `build_environment`**

Replace the body in `backend/agents/rlm/primitives.py`. Add the module-level indirection `_build_image` so tests can monkeypatch it:

```python
# Indirection so tests can monkeypatch the async Docker build.
def _build_image(dockerfile_path, context_dir, tag, **kw):
    from backend.services.runtime.local_docker import build_image
    return build_image(dockerfile_path, context_dir, tag, **kw)


_ENV_REPAIR_SYSTEM = (
    "You are a Docker environment repair assistant. Given a Dockerfile and the "
    "build error it produced, output a corrected Dockerfile and NOTHING else — "
    "no prose, no code fences."
)


def build_environment(env_spec: dict, *, ctx: "RunContext") -> dict:
    """Build the Docker image for `env_spec`, repairing the Dockerfile on failure.

    Genuinely fail-soft (design decision D3): any failure — a spent attempt
    cap, an `llm_client` error, a `write_text` error, a bad import — returns
    `{"ok": False, "error": ..., "attempts": ...}`; the primitive never raises.
    The ONE exception is `SandboxRuntimeError` (Docker daemon down / SDK
    missing): an infrastructure failure, not a Dockerfile problem, so it
    propagates.
    """
    import asyncio
    import concurrent.futures
    import tempfile
    from pathlib import Path

    dockerfile = (env_spec.get("dockerfile") or "").strip()
    if not dockerfile:
        return {"ok": False, "image_tag": "", "error": "env_spec.dockerfile is empty",
                "attempts": 0}

    attempts, ok, tag, error = 0, False, "", ""
    try:
        from backend.config import get_settings
        from backend.services.runtime.interface import SandboxRuntimeError

        max_attempts = max(1, get_settings().environment_build_max_attempts)
        tag = f"reprolab/{ctx.project_id}:env-check"
        with tempfile.TemporaryDirectory() as tmp:
            context_dir = Path(tmp)
            dockerfile_path = context_dir / "Dockerfile"
            while not ok and attempts < max_attempts:
                attempts += 1
                dockerfile_path.write_text(dockerfile, encoding="utf-8")
                # Async bridge: asyncio.run in a fresh worker thread, never
                # bare (a bare asyncio.run raises inside a running loop).
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    ok, tag, error = pool.submit(
                        asyncio.run, _build_image(dockerfile_path, context_dir, tag)
                    ).result()
                if not ok and attempts < max_attempts:
                    dockerfile = ctx.llm_client.complete(
                        system=_ENV_REPAIR_SYSTEM,
                        user=f"Dockerfile:\n{dockerfile}\n\nBuild error:\n{error}",
                    ).strip()
    except SandboxRuntimeError:
        raise  # infrastructure failure — not a Dockerfile problem; propagate
    except Exception as exc:  # noqa: BLE001 — fail-soft (D3): any other failure
        return {"ok": False, "image_tag": "",
                "error": f"{type(exc).__name__}: {exc}", "attempts": attempts}

    return {"ok": ok, "image_tag": tag if ok else "", "error": error,
            "attempts": attempts}
```

If `SandboxRuntimeError` is not importable from `backend.services.runtime.interface`, run `.venv/bin/python -c "import backend.services.runtime.interface as i; print([n for n in dir(i) if 'Error' in n])"` and use the real name.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_build_environment.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/primitives.py tests/rlm/test_build_environment.py
git commit -m "feat(rlm): build_environment primitive with repair loop"
```

> **Residual risk:** the real `build_image` needs a Docker daemon — only the monkeypatched logic is unit-tested here. An end-to-end Docker test belongs in Phase 5 (#62).

---

### Task 7: `plan_reproduction` primitive

**Files:**
- Modify: `backend/agents/rlm/primitives.py:59-61`
- Test: `tests/rlm/test_plan_reproduction.py`

- [ ] **Step 1: Write the failing test**

Create `tests/rlm/test_plan_reproduction.py`:

```python
import json

from backend.agents.rlm.primitives import plan_reproduction

CONTRACT_JSON = json.dumps({
    "reproduction_definition": "Same algorithm, same dataset.",
    "smoke_test_plan": "1000 timesteps.",
    "full_run_plan": "500k timesteps.",
    "expected_outputs": ["metrics.json"],
    "evaluation_plan": "Mean reward over 100 episodes.",
})


def test_plan_reproduction_parses_llm_contract(make_context, tmp_path):
    ctx = make_context(tmp_path, llm_responses=[CONTRACT_JSON])
    result = plan_reproduction({"core_contribution": "X"}, {"framework": "pytorch"}, ctx=ctx)
    assert result["reproduction_definition"] == "Same algorithm, same dataset."
    assert result["expected_outputs"] == ["metrics.json"]
    assert len(ctx.llm_client.calls) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_plan_reproduction.py -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `plan_reproduction`**

Replace the body in `backend/agents/rlm/primitives.py`. Add the JSON-extraction helper once near the top of the file (it is reused by Tasks 10 and 11):

```python
def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response.

    Robust to prose and ``` fences around the JSON: scans forward from each
    `{` and uses `json.JSONDecoder.raw_decode`, which correctly ignores braces
    inside strings and any trailing text — unlike a naive first-`{`/last-`}`
    span, which over-grabs when the response contains prose braces.
    """
    import json
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = text.find("{", idx + 1)
    raise ValueError(f"no JSON object in LLM response: {text[:200]!r}")


_PLAN_REPRODUCTION_SYSTEM = (
    "You are the Reproduction Planner for ReproLab. Given a paper's method "
    "spec and a target environment spec, produce a ReproductionContract: what "
    "counts as a faithful reproduction, a smoke-test plan, a full-run plan, "
    "the expected output artifacts, a dataset plan, an evaluation plan, and a "
    "verification checklist. Return exactly ONE JSON object with those fields "
    "and nothing else. Do NOT write files; do NOT reference any filesystem path."
)


def plan_reproduction(method_spec: dict, env_spec: dict, *, ctx: "RunContext") -> dict:
    """Generate a reproduction contract from structured specs via the LLM.

    Uses a primitive-specific system prompt (`_PLAN_REPRODUCTION_SYSTEM`). The
    orchestrator's `REPRODUCTION_PLANNER_PROMPT` is deliberately NOT reused: it
    instructs a file-writing agent ("write to `{runs_root}/{project_id}/...`"),
    which conflicts with a primitive that must return JSON inline. Returns a
    ReproductionContract dict.
    """
    import json

    from backend.agents.schemas import ReproductionContract

    user = (
        "method_spec:\n" + json.dumps(method_spec, indent=2, default=str)
        + "\n\nenvironment_spec:\n" + json.dumps(env_spec, indent=2, default=str)
    )
    raw = ctx.llm_client.complete(system=_PLAN_REPRODUCTION_SYSTEM, user=user)
    data = _extract_json(raw)
    return ReproductionContract(**data).model_dump()
```

Before implementing, confirm which `ReproductionContract` fields are required: `.venv/bin/python -c "from backend.agents.schemas import ReproductionContract; print({n: f.is_required() for n, f in ReproductionContract.model_fields.items()})"`. If any required field is absent from the test's `CONTRACT_JSON`, add it there with a sane value (a missing required field makes `ReproductionContract(**data)` raise `ValidationError`). If `ReproductionContract` instead rejects an *extra* key the LLM returned, confirm it has `model_config = {"extra": "ignore"}` or drop unknown keys before construction.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_plan_reproduction.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/primitives.py tests/rlm/test_plan_reproduction.py
git commit -m "feat(rlm): plan_reproduction primitive"
```

---

### Task 8: `implement_baseline` primitive

**Files:**
- Modify: `backend/agents/rlm/primitives.py:64-66`
- Test: `tests/rlm/test_implement_baseline.py`

Design: wraps the async `run_with_sdk` via `ctx.runtime`, bridged to sync with a worker thread; then writes `code/commands.json` (D2) so `run_experiment` can read the commands.

- [ ] **Step 1: Write the failing test**

Create `tests/rlm/test_implement_baseline.py`:

```python
import json

import backend.agents.rlm.primitives as primitives
from backend.agents.rlm.primitives import implement_baseline


class _FakeBaselineResult:
    commands_to_run = ["python train.py", "python eval.py"]


def test_implement_baseline_writes_commands_manifest(make_context, tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    ctx.runtime = object()

    async def fake_run_with_sdk(project_id, runs_root, pcm, env, contract,
                                artifact_index, **kw):
        return _FakeBaselineResult()

    monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)
    code_path = implement_baseline(
        {"paper_claim_map": {}, "environment_spec": {}, "reproduction_contract": None},
        ctx=ctx,
    )
    manifest = json.loads((tmp_path / "test_proj" / "code" / "commands.json").read_text())
    assert manifest == ["python train.py", "python eval.py"]
    assert code_path.endswith("code")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_implement_baseline.py -v`
Expected: FAIL — `AttributeError: ... '_run_baseline_with_sdk'`.

- [ ] **Step 3: Implement `implement_baseline`**

Replace the body in `backend/agents/rlm/primitives.py`:

```python
def _run_baseline_with_sdk(project_id, runs_root, pcm, env, contract, artifact_index, **kw):
    """Indirection over baseline_implementation.run_with_sdk so tests can patch it."""
    from backend.agents.baseline_implementation import run_with_sdk
    return run_with_sdk(project_id, runs_root, pcm, env, contract, artifact_index, **kw)


def implement_baseline(plan: dict, *, ctx: "RunContext") -> str:
    """Generate the baseline code from a reproduction plan; return the code path.

    `plan` is the aggregate dict the root assembles: `{"paper_claim_map":
    <understand_section output>, "environment_spec": <detect_environment
    output>, "reproduction_contract": <plan_reproduction output>}` (plus an
    optional `artifact_index`) — NOT a single producer's output. Wraps
    `baseline_implementation.run_with_sdk` (a code-writing agent) and writes
    `code/commands.json` so `run_experiment` can read the run commands without
    a BaselineResult (design decision D2).
    """
    import asyncio
    import concurrent.futures
    import json

    from backend.agents.schemas import PaperClaimMap, EnvironmentSpec, ReproductionContract

    # core_contribution is PaperClaimMap's one required field; default it so a
    # partial paper_claim_map (e.g. understand_section's output) validates.
    pcm = PaperClaimMap(**{"core_contribution": "", **plan.get("paper_claim_map", {})})
    env = EnvironmentSpec(**plan.get("environment_spec", {}))
    contract = (ReproductionContract(**plan["reproduction_contract"])
                if plan.get("reproduction_contract") else None)
    artifact_index = plan.get("artifact_index")

    async def _run():
        return await _run_baseline_with_sdk(
            ctx.project_id, ctx.runs_root, pcm, env, contract, artifact_index,
            runtime=ctx.runtime)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(asyncio.run, _run()).result()

    code_dir = ctx.project_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    commands = list(getattr(result, "commands_to_run", []) or [])
    (code_dir / "commands.json").write_text(json.dumps(commands), encoding="utf-8")
    return str(code_dir)
```

Verified signature (`baseline_implementation.py:418`): `async run_with_sdk(project_id: str, runs_root: Path, paper_claim_map: PaperClaimMap, environment_spec: EnvironmentSpec, reproduction_contract: ReproductionContract | None = None, artifact_index: dict | None = None, *, model=None, provider=None, runtime: AgentRuntime | None = None) -> BaselineResult`. The adapter's six positional args match this order; `runtime` is keyword-only. `BaselineResult.commands_to_run` (`schemas.py:153`) is the manifest source.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_implement_baseline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/primitives.py tests/rlm/test_implement_baseline.py
git commit -m "feat(rlm): implement_baseline primitive + commands.json manifest"
```

---

### Task 9: `run_experiment` primitive

**Files:**
- Modify: `backend/agents/rlm/primitives.py:69-77`
- Test: `tests/rlm/test_run_experiment.py`

Design (D1/D2): `env_id` is a prebuilt Docker image tag; commands come from `code_path/commands.json`; the primitive drives `RuntimeAppService` create/execute/destroy, async-bridged.

- [ ] **Step 1: Write the failing test**

Create `tests/rlm/test_run_experiment.py`:

```python
import json

import backend.agents.rlm.primitives as primitives
from backend.agents.rlm.primitives import run_experiment


def test_run_experiment_reads_commands_and_returns_metrics(make_context, tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))

    async def fake_exec(code_path, env_id, commands, *, project_id, run_id):
        assert env_id == "reprolab/test:env-check"
        assert commands == ["python train.py"]
        assert project_id  # run_experiment threads ctx.project_id through
        return {"metrics": {"mean_reward": 200.0}, "success": True, "logs": ""}

    monkeypatch.setattr(primitives, "_execute_in_sandbox", fake_exec)
    result = run_experiment(str(code_dir), "reprolab/test:env-check", ctx=ctx)
    assert result["success"] is True
    assert result["metrics"]["mean_reward"] == 200.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_run_experiment.py -v`
Expected: FAIL — `AttributeError: ... '_execute_in_sandbox'`.

- [ ] **Step 3: Implement `run_experiment`**

Replace the body in `backend/agents/rlm/primitives.py`:

```python
async def _execute_in_sandbox(
    code_path: str,
    env_id: str,
    commands: list[str],
    *,
    project_id: str,
    run_id: str,
) -> dict:
    """Run `commands` in a container started from the prebuilt image `env_id`.

    Drives the verified `RuntimeAppService` lifecycle (`service.py`): create a
    sandbox from the existing image (`dockerfile_path=None`, `build_context=None`
    → no rebuild, design decision D1), execute each command, destroy. The
    service methods take `Command` objects. Indirection so tests can patch it.
    """
    from pathlib import Path

    from backend.services.runtime.interface import SandboxConfig
    from backend.services.runtime.local_docker import LocalDockerBackend
    from backend.services.runtime.service import (
        CreateSandbox, DestroySandbox, ExecuteCommand, RuntimeAppService,
    )

    service = RuntimeAppService(LocalDockerBackend())
    config = SandboxConfig(
        project_id=project_id,
        run_id=run_id,
        image=env_id,
        project_root=Path(code_path),
        dockerfile_path=None,   # prebuilt image — no rebuild (design decision D1)
        build_context=None,
    )
    sandbox = await service.create_sandbox(CreateSandbox(config=config))
    results = []
    try:
        for command in commands:
            results.append(await service.execute(
                ExecuteCommand(sandbox=sandbox, command=command, timeout=3600)))
    finally:
        await service.destroy(DestroySandbox(sandbox=sandbox))
    return {
        "success": all(r.succeeded for r in results),
        "metrics": {},  # real metric extraction from artifacts is Phase 5 (#62)
        "logs": "\n".join(r.stdout for r in results),
    }


def run_experiment(code_path: str, env_id: str, *, ctx: "RunContext") -> dict:
    """Execute the baseline in a container from prebuilt image `env_id`; return metrics.

    Commands are read from `code_path/commands.json` (written by
    `implement_baseline`). `env_id` is a Docker image tag (design decisions
    D1/D2). Async sandbox work is bridged to sync via a worker thread.
    """
    import asyncio
    import concurrent.futures
    import json
    from pathlib import Path

    manifest = Path(code_path) / "commands.json"
    commands = json.loads(manifest.read_text()) if manifest.exists() else []
    if not commands:
        return {"success": False, "metrics": {},
                "error": f"no commands.json at {manifest}"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(
            asyncio.run,
            _execute_in_sandbox(code_path, env_id, commands,
                                project_id=ctx.project_id, run_id=ctx.project_id),
        ).result()
```

Verified API (`backend/services/runtime/service.py` + `interface.py`): `RuntimeAppService(backend)` exposes `async create_sandbox(CreateSandbox(config=SandboxConfig)) -> Sandbox`, `async execute(ExecuteCommand(sandbox=, command=, timeout=int)) -> ExecResult` (use its `.succeeded` property — `exit_code == 0 and not timed_out`), and `async destroy(DestroySandbox(sandbox=)) -> None`. The methods take `Command` objects, and the teardown method is `destroy`, **not** `destroy_sandbox`. `SandboxConfig` (`interface.py:31`) requires `project_id`, `run_id`, `project_root`. Confirm `LocalDockerBackend()` constructs without args (`local_docker.py:127`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_run_experiment.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/primitives.py tests/rlm/test_run_experiment.py
git commit -m "feat(rlm): run_experiment primitive (prebuilt-image sandbox run)"
```

> **Residual risk:** `_execute_in_sandbox` needs a Docker daemon; only the monkeypatched wiring is unit-tested. The `metrics` dict is left empty here — extracting real metrics from run artifacts is a follow-up (Phase 5 / #62).

---

### Task 10: `verify_against_rubric` primitive

**Files:**
- Modify: `backend/agents/rlm/primitives.py:80-86`
- Test: `tests/rlm/test_verify_against_rubric.py`

Design: wraps the rubric-verifier prompt; LLM scores only (weights are spec-fixed from `rubric`); applies the honesty backstop (run failed → cap every area at 0.35); computes the overall score with `RubricVerification.from_areas`.

- [ ] **Step 1: Write the failing test**

Create `tests/rlm/test_verify_against_rubric.py`:

```python
import json

from backend.agents.rlm.primitives import verify_against_rubric

RUBRIC = {
    "areas": [{"area": "code", "weight": 0.6}, {"area": "results", "weight": 0.4}],
    "source": "generated",
    "target_score": 0.7,
}
LLM_SCORES = json.dumps({"areas": [
    {"area": "code", "score": 0.9, "weak_points": []},
    {"area": "results", "score": 0.8, "weak_points": []},
], "confidence": 0.8})


def test_verify_caps_a_failed_run(make_context, tmp_path):
    ctx = make_context(tmp_path, llm_responses=[LLM_SCORES])
    result = verify_against_rubric({"success": False, "metrics": {"r": 1}}, RUBRIC, ctx=ctx)
    assert all(a["score"] <= 0.35 for a in result["areas"])


def test_verify_caps_a_metric_less_run(make_context, tmp_path):
    # success=True but no metrics — run_experiment returns metrics={} in Phase 2,
    # so a metric-less run must not score high (extends the honesty backstop).
    ctx = make_context(tmp_path, llm_responses=[LLM_SCORES])
    result = verify_against_rubric({"success": True, "metrics": {}}, RUBRIC, ctx=ctx)
    assert all(a["score"] <= 0.35 for a in result["areas"])


def test_verify_uses_llm_scores_for_a_real_run(make_context, tmp_path):
    ctx = make_context(tmp_path, llm_responses=[LLM_SCORES])
    result = verify_against_rubric(
        {"success": True, "metrics": {"mean_reward": 200.0}}, RUBRIC, ctx=ctx)
    assert any(a["score"] > 0.35 for a in result["areas"])
    assert 0.0 <= result["overall_score"] <= 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_verify_against_rubric.py -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `verify_against_rubric`**

Replace the body in `backend/agents/rlm/primitives.py`:

```python
def verify_against_rubric(results: dict, rubric: dict, *, ctx: "RunContext") -> dict:
    """Score `results` against `rubric` via the rubric-verifier prompt.

    The LLM scores areas only; weights come verbatim from `rubric`. The honesty
    backstop is enforced mechanically: every area score is capped at 0.35 when
    the run did not succeed OR produced no metrics — matching
    `orchestrator._run_rubric_verifier` (which caps on `success`) and extending
    it to the metric-less case (`run_experiment` returns `metrics={}` in
    Phase 2 — see Task 9). `overall_score` / `meets_target` are computed by
    `RubricVerification.from_areas`, never trusted from the model.
    """
    import json

    from backend.agents.prompts.rubric_verifier import RUBRIC_VERIFIER_PROMPT
    from backend.agents.schemas import RubricAreaScore, RubricVerification

    user = (
        "results:\n" + json.dumps(results, indent=2, default=str)
        + "\n\nrubric:\n" + json.dumps(rubric, indent=2, default=str)
        + "\n\nScore each rubric area in [0,1]. Return a JSON object: "
          '{"areas": [{"area": str, "score": float, "justification": str, '
          '"weak_points": [str]}], "confidence": float}.'
    )
    raw = ctx.llm_client.complete(system=RUBRIC_VERIFIER_PROMPT, user=user)
    parsed = _extract_json(raw)

    weights = {a["area"]: float(a.get("weight", 0.0)) for a in rubric.get("areas", [])}
    degraded = (not results.get("success")) or (not results.get("metrics"))
    areas: list[RubricAreaScore] = []
    for a in parsed.get("areas", []):
        name = str(a.get("area", ""))
        score = float(a.get("score", 0.0))
        if degraded:
            score = min(score, 0.35)  # honesty backstop
        areas.append(RubricAreaScore(
            area=name,
            weight=weights.get(name, 0.0),
            score=score,
            justification=str(a.get("justification", "")),
            weak_points=[str(w) for w in (a.get("weak_points") or [])],
        ))
    verification = RubricVerification.from_areas(
        areas,
        rubric_source=rubric.get("source", "generated"),
        target_score=float(rubric.get("target_score", 0.0)),
        confidence=float(parsed.get("confidence", 0.0)),
    )
    return verification.model_dump()
```

Verified signature (`schemas.py:352`): `RubricVerification.from_areas(areas: list[RubricAreaScore], *, rubric_source: Literal["paperbench_bundle","generated"], target_score: float, confidence: float = 0.0, verified_at: str = "")`. It requires **`RubricAreaScore` objects** (not plain dicts) and the keyword-only `rubric_source` + `target_score`; it recomputes `overall_score` / `meets_target` itself — the code above must not pre-set them. `RubricAreaScore` fields (`schemas.py:318`): `area, weight, score, justification, weak_points`.

**Honesty-backstop scope — do not over-build.** The mechanical `min(score, 0.35)` is the *same* backstop `orchestrator._run_rubric_verifier` enforces — one cap, on a non-successful run. `RUBRIC_VERIFIER_PROMPT` (`rubric_verifier.py:52-58`) *additionally* instructs the LLM with finer caps (≤0.20 for no executable code / missing target metric, ≤0.40 for missing provenance); those are enforced **LLM-side via the prompt**, exactly as the orchestrator does it. `min()` only lowers a score, so a prompt-following LLM's stricter score survives the backstop. Do **not** add mechanical 0.20/0.40 caps — that diverges from the canonical orchestrator and needs run-state (executable-code / provenance flags) that Phase 2's `run_experiment` does not produce.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_verify_against_rubric.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/primitives.py tests/rlm/test_verify_against_rubric.py
git commit -m "feat(rlm): verify_against_rubric primitive with honesty backstop"
```

---

### Task 11: `propose_improvements` primitive

**Files:**
- Modify: `backend/agents/rlm/primitives.py:89-101`
- Test: `tests/rlm/test_propose_improvements.py`

Design: reuses `IMPROVEMENT_ORCHESTRATOR_PROMPT` (already taxonomy-free per recon); returns a variable-length list of `ImprovementHypothesis` dicts with proposer-assigned free-form `category` tags; drops malformed items fail-soft.

- [ ] **Step 1: Write the failing test**

Create `tests/rlm/test_propose_improvements.py`:

```python
import json

from backend.agents.rlm.primitives import propose_improvements

HYP_JSON = json.dumps({"hypotheses": [
    {"path_id": "p1", "hypothesis": "Tune the learning rate.",
     "rationale": "Learning rate is the highest-leverage PPO hyperparameter.",
     "expected_outcome": "Higher mean reward at the same step budget.",
     "category": "optimizer"},
    {"path_id": "p2", "hypothesis": "Swap the backbone.",
     "rationale": "A wider value network may fit the return better.",
     "expected_outcome": "Lower value loss and faster convergence.",
     "category": "architecture"},
]})


def test_propose_improvements_returns_variable_length_tagged_list(make_context, tmp_path):
    ctx = make_context(tmp_path, llm_responses=[HYP_JSON])
    result = propose_improvements({"success": True}, {"areas": []}, ctx=ctx)
    assert isinstance(result, list)
    assert len(result) == 2
    assert {h["category"] for h in result} == {"optimizer", "architecture"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_propose_improvements.py -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `propose_improvements`**

Replace the body in `backend/agents/rlm/primitives.py`:

```python
def propose_improvements(current_results: dict, rubric_scores: dict,
                         k: int | None = None, *, ctx: "RunContext") -> list[dict]:
    """Propose paper-specific improvement hypotheses (variable-length, free-form tags).

    Reuses the `improvement-orchestrator` prompt — no fixed taxonomy. Each item
    is an `ImprovementHypothesis` dict; malformed items are dropped fail-soft.
    """
    import json

    from backend.agents.prompts.improvement import IMPROVEMENT_ORCHESTRATOR_PROMPT
    from backend.agents.schemas import ImprovementHypothesis

    target = k if k is not None else 3
    user = (
        "current_results:\n" + json.dumps(current_results, indent=2, default=str)
        + "\n\nrubric_scores (prioritise lifting the weakest areas):\n"
        + json.dumps(rubric_scores, indent=2, default=str)
        + f"\n\nPropose up to {target} improvement hypotheses. Return a JSON "
          'object {"hypotheses": [ImprovementHypothesis, ...]}. Each hypothesis '
          "carries a free-form `category` tag of your choosing."
    )
    raw = ctx.llm_client.complete(system=IMPROVEMENT_ORCHESTRATOR_PROMPT, user=user)
    items = _extract_json(raw).get("hypotheses", [])

    out: list[dict] = []
    for item in items:
        try:
            out.append(ImprovementHypothesis(**item).model_dump())
        except Exception:
            continue  # fail-soft: skip a malformed hypothesis
    return out
```

The test's `HYP_JSON` items above include all four **required** `ImprovementHypothesis` fields — `path_id`, `hypothesis`, `rationale`, `expected_outcome` (verified `schemas.py:221`) — plus the free-form `category`. This matters: the implementation drops malformed items fail-soft, so a `HYP_JSON` item missing a required field would be silently discarded and `len(result) == 2` would fail as `0 == 2`. If `ImprovementHypothesis` instead rejects an *extra* key, confirm it has `model_config = {"extra": "ignore"}` (it does — `schemas.py:223`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_propose_improvements.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/primitives.py tests/rlm/test_propose_improvements.py
git commit -m "feat(rlm): propose_improvements primitive (variable-length, free-form tags)"
```

---

### Task 12: Remove the vestigial `set_final` stub

**Files:**
- Modify: `backend/agents/rlm/primitives.py` (delete the `set_final` stub + its `PRIMITIVE_REGISTRY` entry)
- Modify: `backend/agents/rlm/__init__.py` (only if it imports/exports `set_final`)
- Test: `tests/rlm/test_registry_count.py`

The Phase-1 skeleton's `primitives.py` carries a 10th stub, `set_final`, in `PRIMITIVE_REGISTRY`. The canonical brief §7 lists **nine** primitives and omits `set_final`: under the `rlm` engine the root terminates with `FINAL_VAR(<var>)` — it assigns the report to a REPL variable and emits the tag directly, so no `set_final` helper is needed. `set_final` is vestigial; Phase 2 ships nine primitives. (If a future brief / issue-#59 amendment re-adds it, restore it then — do not keep dead scope now.)

- [ ] **Step 1: Write the failing test**

Create `tests/rlm/test_registry_count.py`:

```python
from backend.agents.rlm.primitives import PRIMITIVE_REGISTRY

NINE = {
    "understand_section", "extract_hyperparameters", "detect_environment",
    "build_environment", "plan_reproduction", "implement_baseline",
    "run_experiment", "verify_against_rubric", "propose_improvements",
}


def test_registry_has_exactly_the_nine_brief_primitives():
    assert set(PRIMITIVE_REGISTRY) == NINE
    assert "set_final" not in PRIMITIVE_REGISTRY
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_registry_count.py -v`
Expected: FAIL — the skeleton's `PRIMITIVE_REGISTRY` still contains `set_final` (10 keys).

- [ ] **Step 3: Remove `set_final`**

In `backend/agents/rlm/primitives.py`: delete the `set_final` function (the skeleton stub) entirely, and remove its `"set_final": set_final` entry from the `PRIMITIVE_REGISTRY` dict — leave the other nine entries unchanged. Then check `backend/agents/rlm/__init__.py`: if it imports or lists `set_final` in `__all__`, remove those lines too (`grep -n set_final backend/agents/rlm/__init__.py`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_registry_count.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/primitives.py tests/rlm/test_registry_count.py
git commit -F- <<'EOF'
refactor(rlm): drop the vestigial set_final primitive

The canonical brief §7 lists nine primitives. set_final was a Phase-1
skeleton stub; under the rlm engine the root terminates with FINAL_VAR(<var>),
so no set_final helper is needed. Phase 2 ships nine primitives.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
```

(If `__init__.py` changed, add it to the `git add` above.)

---

### Task 13: `PRIMITIVE_DESCRIPTIONS` + registry wiring + package exports

**Files:**
- Modify: `backend/agents/rlm/primitives.py` (the `PRIMITIVE_REGISTRY` block + add `PRIMITIVE_DESCRIPTIONS`)
- Modify: `backend/agents/rlm/__init__.py`
- Test: `tests/rlm/test_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/rlm/test_registry.py`:

```python
from backend.agents.rlm.primitives import PRIMITIVE_REGISTRY, PRIMITIVE_DESCRIPTIONS
from backend.agents.rlm.binding import build_custom_tools

EXPECTED = {
    "understand_section", "extract_hyperparameters", "detect_environment",
    "build_environment", "plan_reproduction", "implement_baseline",
    "run_experiment", "verify_against_rubric", "propose_improvements",
}


def test_registry_and_descriptions_cover_all_primitives():
    assert set(PRIMITIVE_REGISTRY) == EXPECTED
    assert set(PRIMITIVE_DESCRIPTIONS) == EXPECTED


def test_build_custom_tools_produces_rlm_tool_dict(make_context, tmp_path):
    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    assert set(tools) == EXPECTED
    for entry in tools.values():
        assert callable(entry["tool"])
        assert isinstance(entry["description"], str) and entry["description"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_registry.py -v`
Expected: FAIL — `ImportError: cannot import name 'PRIMITIVE_DESCRIPTIONS'`.

- [ ] **Step 3: Add `PRIMITIVE_DESCRIPTIONS` to `primitives.py`**

Below the existing `PRIMITIVE_REGISTRY` dict in `backend/agents/rlm/primitives.py`:

```python
PRIMITIVE_DESCRIPTIONS: dict[str, str] = {
    "understand_section": "understand_section(text_slice) -> dict — datasets, "
        "metrics, training recipe, hardware clues, ambiguities from a text slice. "
        "A PARTIAL PaperClaimMap (no core_contribution/claims/architecture).",
    "extract_hyperparameters": "extract_hyperparameters(text_slice) -> dict — "
        "optimizer, learning rate, batch size, epochs from a slice.",
    "detect_environment": "detect_environment(method_spec) -> dict — an "
        "EnvironmentSpec (dockerfile, python_version, framework, pip_packages). "
        "`method_spec` is a (partial) PaperClaimMap dict.",
    "build_environment": "build_environment(env_spec) -> dict — build the Docker "
        "image, repairing the Dockerfile on failure. Returns a BUILD RESULT "
        "{ok, image_tag, error, attempts} — NOT an EnvironmentSpec. Pass "
        "image_tag to run_experiment as env_id.",
    "plan_reproduction": "plan_reproduction(method_spec, env_spec) -> dict — a "
        "ReproductionContract (smoke test, full run, evaluation plan).",
    "implement_baseline": "implement_baseline(plan) -> str — generate the "
        "baseline code; returns the code dir path. `plan` is the aggregate "
        "{paper_claim_map (from understand_section), environment_spec (from "
        "detect_environment), reproduction_contract (from plan_reproduction)}.",
    "run_experiment": "run_experiment(code_path, env_id) -> dict — run the "
        "baseline in a container from image `env_id` (build_environment's "
        "image_tag); returns {success, metrics, logs}.",
    "verify_against_rubric": "verify_against_rubric(results, rubric) -> dict — "
        "score the results against a PaperBench-style rubric.",
    "propose_improvements": "propose_improvements(current_results, rubric_scores, "
        "k=None) -> list[dict] — paper-specific improvement hypotheses.",
}
```

- [ ] **Step 4: Update `backend/agents/rlm/__init__.py`**

Add to the imports and `__all__` in `backend/agents/rlm/__init__.py`:

```python
from backend.agents.rlm.context import RunContext
from backend.agents.rlm.binding import build_custom_tools
from backend.agents.rlm.primitives import PRIMITIVE_DESCRIPTIONS
```

And append `"RunContext"`, `"build_custom_tools"`, `"PRIMITIVE_DESCRIPTIONS"` to `__all__`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_registry.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add backend/agents/rlm/primitives.py backend/agents/rlm/__init__.py tests/rlm/test_registry.py
git commit -m "feat(rlm): primitive descriptions + build_custom_tools wiring"
```

---

### Task 14: Integration test — `custom_tools` against a mock `rlm.RLM`

**Files:**
- Test: `tests/rlm/test_integration_custom_tools.py`

This verifies the Phase 2 deliverable end to end: `build_custom_tools(ctx)` produces a dict the `rlm` library accepts, and a primitive is callable inside the REPL — reusing the mock-backend pattern from `tools/rlms_spike.py`.

- [ ] **Step 1: Write the integration test**

Create `tests/rlm/test_integration_custom_tools.py`:

```python
import json

# A mock paper, offloaded as the REPL `context` variable — the root model
# slices `context`; it never receives the paper in its own prompt (the RLM premise).
MOCK_PAPER = {
    "paper_text": ("Our method trains with the Adam optimizer at learning rate "
                   "3e-4, batch size 64, for 200 epochs on CartPole-v1. " * 12),
    "paper_metadata": {"title": "Mock RL Paper"},
}


def test_primitives_are_callable_inside_the_rlm_repl(make_context, tmp_path):
    import rlm.core.rlm as rlm_core
    from rlm import RLM
    from rlm.clients.base_lm import BaseLM
    from rlm.core.types import ModelUsageSummary, UsageSummary

    from backend.agents.rlm.binding import build_custom_tools

    ctx = make_context(tmp_path)
    custom_tools = build_custom_tools(ctx)

    class ScriptedLM(BaseLM):
        def __init__(self):
            super().__init__(model_name="scripted")
            self.turns = 0

        def completion(self, prompt):
            self.turns += 1
            if self.turns == 1:
                # Slice the offloaded `context` variable and pass the slice to
                # a primitive — exercises the paper-as-variable RLM flow.
                return ("```repl\n"
                        "slice_ = context['paper_text'][:600]\n"
                        "hp = extract_hyperparameters(slice_)\n"
                        "report = {'hyperparameters': hp}\n"
                        "print(report)\n```\n")
            return "Done.\nFINAL_VAR(report)"

        async def acompletion(self, prompt):
            return self.completion(prompt)

        def _u(self):
            return ModelUsageSummary(total_calls=self.turns, total_input_tokens=0,
                                     total_output_tokens=0, total_cost=0.0)

        def get_usage_summary(self):
            return UsageSummary(model_usage_summaries={self.model_name: self._u()})

        def get_last_usage(self):
            return self._u()

    original = rlm_core.get_client
    rlm_core.get_client = lambda backend, kw: ScriptedLM()
    try:
        rlm = RLM(backend="openai", backend_kwargs={"model_name": "scripted"},
                  environment="local", max_iterations=4, custom_tools=custom_tools,
                  custom_sub_tools={})
        result = rlm.completion(MOCK_PAPER)  # MOCK_PAPER becomes the REPL `context`
    finally:
        rlm_core.get_client = original
        rlm.close()

    assert result.response  # terminated via FINAL_VAR(report)
    events = [json.loads(ln) for ln in
              (ctx.project_dir / "dashboard_events.jsonl").read_text().splitlines() if ln]
    names = {e["primitive"] for e in events if e.get("event") == "primitive_call"}
    assert "extract_hyperparameters" in names  # a primitive ran on a slice of `context`


def test_every_primitive_binds_and_heuristic_ones_run(make_context, tmp_path):
    """Every primitive binds into custom_tools and is callable; the three
    no-dependency heuristic primitives actually run through the wrapper.

    Together with the REPL test above and the Task 6-11 unit tests, this
    covers issue #59's "every primitive callable from the REPL" done-condition.
    """
    from backend.agents.rlm.binding import build_custom_tools

    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    assert len(tools) == 9  # the nine brief-§7 primitives
    for entry in tools.values():
        assert callable(entry["tool"])

    # The heuristic primitives need no monkeypatching — invoke them for real
    # through the bound custom_tools wrapper.
    us = tools["understand_section"]["tool"]("Adam, lr 3e-4, batch 64, CartPole-v1.")
    assert {"datasets", "metrics", "training_recipe"} <= set(us)
    hp = tools["extract_hyperparameters"]["tool"]("Adam optimizer, batch size 64.")
    assert "64" in hp["batch_size"]
    env = tools["detect_environment"]["tool"]({"core_contribution": "A PyTorch agent."})
    assert env["dockerfile"].startswith("FROM")

    events = [json.loads(ln) for ln in
              (ctx.project_dir / "dashboard_events.jsonl").read_text().splitlines() if ln]
    ran = {e["primitive"] for e in events if e.get("event") == "primitive_call"}
    assert {"understand_section", "extract_hyperparameters", "detect_environment"} <= ran
```

- [ ] **Step 2: Run the integration tests**

Run: `.venv/bin/python -m pytest tests/rlm/test_integration_custom_tools.py -v`
Expected: PASS (2 tests) — a primitive runs end-to-end inside the rlm REPL on a slice of `context`, and all nine primitives bind into `custom_tools` with the three heuristic ones running through the wrapper.

- [ ] **Step 3: Run the whole RLM test suite**

Run: `.venv/bin/python -m pytest tests/rlm/ -v`
Expected: PASS — every test from Tasks 1–14.

- [ ] **Step 4: Commit**

```bash
git add tests/rlm/test_integration_custom_tools.py
git commit -m "test(rlm): integration — primitives callable in the rlm REPL"
```

---

## Self-review

**1. Spec coverage (issue #59).** #59 requires: extract each surviving stage agent's core logic into a function in `primitives.py` → Tasks 3–11 (the nine brief-§7 primitives; Task 12 removes the vestigial `set_final` skeleton stub); each primitive emits a `primitive_call` SSE event → Task 2 wrapper, verified Task 14; each updates `cost_ledger.jsonl` → Task 2 wrapper (D7 — zero-usage call entry in Phase 2); assembled into the `custom_tools` dict → Tasks 2 + 13; "wrap, not rewrite" → every primitive wraps an existing helper/agent/prompt. Done condition (primitives callable from the REPL with correct outputs and events) → Task 14. Covered.

**2. Placeholder scan.** No "TBD"/"implement later". The two `_build_image` / `_run_baseline_with_sdk` / `_execute_in_sandbox` indirections are real, named module functions (so tests can monkeypatch) — not placeholders. The `metrics: {}` in `run_experiment` is a deliberate, flagged Phase-5 follow-up, not a hidden gap.

**3. Type consistency.** `RunContext` (Task 1) is the keyword arg name `ctx` everywhere. `build_custom_tools(ctx)` (Task 2) is used in Tasks 13–14. `_extract_json` (defined Task 7) is reused in Tasks 10–11. Primitive signatures match `PRIMITIVE_REGISTRY` keys and `PRIMITIVE_DESCRIPTIONS` (Task 13 test asserts the sets are equal — nine primitives, after Task 12 drops `set_final`).

**Verification commands** are exact (`.venv/bin/python -m pytest tests/rlm/<file> -v`). Several tasks include a one-line introspection command to confirm a real schema/API name before relying on it — these are guards, not placeholders.

**Hyperanalysis corrections (2026-05-21 review pass).** A blocker review caught and fixed: **D0** — the root REPL must be `environment="local"`; `custom_tools` are not plumbed into `rlm`'s `DockerREPL` (verified in source), so issue #60 and `rlms-spike-report.md` (both say `environment='docker'`) need an upstream correction. **Task 1** now pins `python-dotenv>=1.2.1,<2` (`rlms` requires it; the env had 1.0.1). **Task 10** `verify_against_rubric` now builds `RubricAreaScore` objects and passes `from_areas`'s required keyword args (the prior call would have raised `TypeError`), and the honesty backstop also caps a metric-less run. **Task 2** `primitive_call` event field names follow issue #61's schema. **Task 6** `build_environment` uses the same worker-thread async bridge as Tasks 8–9. The build order is corrected to **strictly sequential**; `_extract_json` is now brace-robust. **Audit pass (2026-05-21, `phase2-plan-audit.md`).** An independent audit caught and fixed: Task 11's `HYP_JSON` was missing `ImprovementHypothesis`'s required `rationale`/`expected_outcome` (the fail-soft drop would have made the test `0 == 2`); `implement_baseline` now defaults `PaperClaimMap.core_contribution` so a partial `plan["paper_claim_map"]` validates instead of raising; **`set_final` is removed** — Task 12 now deletes the skeleton's vestigial 10th stub, aligning Phase 2 to brief §7's nine primitives (`FINAL_VAR` termination needs no `set_final` helper); `build_environment` is now genuinely fail-soft (D3); `plan_reproduction` uses a primitive-specific system prompt (the orchestrator's `REPRODUCTION_PLANNER_PROMPT` instructs a file-writing agent); Task 14 also verifies all nine primitives bind + the heuristic ones run; the D0 citation, the Task 10 test count, and the `PRIMITIVE_DESCRIPTIONS` precision (`build_environment` returns a build result, not an `EnvironmentSpec`) were corrected. Audit findings #5 (graduated rubric caps) and #9 (`PaperClaimMap.extra="ignore"` drop) were assessed as over-reach — the verify-rubric backstop matches the canonical orchestrator, and `extra="ignore"` is intentional schema behavior.
