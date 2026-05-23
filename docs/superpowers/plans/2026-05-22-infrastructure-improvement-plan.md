# Infrastructure Improvement Plan — OpenResearch / ReproLab

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-05-22
**Author:** infra planning session
**Goal:** Improve the execution + cost-safety substrate of the OpenResearch / ReproLab reproduction pipeline without conflicting with the in-flight RLM pivot.
**Architecture:** Two parts — (A) a synthesized audit + improvement catalog covering the sandbox layer, cost controls, observability, and test corpus; (B) one detailed TDD plan for Phase 1, a `max_pod_seconds` budget cap that closes the runaway-pod gap.
**Tech Stack:** Python 3.14, FastAPI, asyncssh, `claude-agent-sdk`, `rlms` 0.1.1, RunPod REST API, pytest.

---

## Part A — Current-state audit (operational lens)

This audit focuses on the *execution and cost-safety* substrate. For the wider RLM pivot, code-honesty, and frontend audit, see `docs/design/project-state-audit-2026-05-22.md` (read it first — this plan is sequenced *around* that one).

### A.1 Sandboxes — what we have

Three backends behind a `RuntimeBackend` interface:

| Backend | File | When used |
|---|---|---|
| Local | `backend/services/runtime/local_backend.py` | Dev, smoke tests, CPU-only papers |
| Docker | `backend/services/runtime/docker_backend.py` | Containerized local runs with limits |
| RunPod | `backend/services/runtime/runpod_backend.py` (857 lines) | Remote GPU, default for `runpod` sandbox mode |

The RunPod backend's lifecycle (`runpod_backend.py:117-371`):
1. `create_sandbox()` — POST `/pods`, poll until RUNNING (≤900s), SSH connect with TOFU host-key pin, SFTP project upload.
2. `exec()` — run command over SSH, tar artifacts back.
3. `destroy()` — final artifact sync, close SSH, DELETE `/pods/{id}` only if pod-id is in `_owned_pod_ids` (guards against deleting attached/persistent pods).

### A.2 Cost controls — what we have

`RunBudget` (`backend/agents/resilience/budget.py:12-51`, 54 lines total) provides:

- `max_usd` — LLM-spend cap, checked via `RunCostLedger`.
- `max_wall_clock_seconds` — total elapsed-time cap.
- `max_invocations_per_agent: dict[str, int]` — per-agent call cap.
- `rlm_calls_remaining: int = 120` — recursive-LM call cap.

Checked at `RunBudget.check()` before each agent invocation; raises `BudgetExhausted` on overage.

**Gap.** No pod-time cap. A hung pod with `delete_on_destroy=True` still bills until either (a) the wall-clock cap hits, or (b) the run is manually killed. Pod cost (cents/min on RTX 4090) is the largest *infra* cost vector and the *only* one not tracked or capped.

### A.3 Observability — what we have

- `runs/<project_id>/*.jsonl` — agent event logs, SSE source.
- `demo_status.json` — UI-facing snapshot.
- `pipeline_state.json` — checkpointed every stage transition, resume-safe.
- `final_report.{json,md}` — benchmark output.
- Hermes audit chain artifacts.

**Gap.** No unified per-run trace that joins (pod start, pod stop, exec wall-time, LLM tokens/cost, rubric scores) into one timeline. Reconstructing "what cost what" requires reading three files plus the SSH timestamps.

### A.4 Test corpus — what we have

Three vendored PaperBench bundles (`third_party/paperbench/`):
- `ftrl` — *Fine-tuning RL Models is Secretly a Forgetting Mitigation Problem* (ICML 2024)
- `mechanistic-understanding` — *DPO + Toxicity Mechanistic Analysis*
- `sequential-neural-score-estimation` — *Likelihood-Free Inference via Score-Based Diffusion*

Each bundle ships `paper.md`, `rubric.json`, `addendum.md`, `blacklist.txt`, `config.yaml`. Past runs live in `runs/pb_<paper-id>_<timestamp>/`.

**Gap.** Three papers is enough for a smoke benchmark but not enough to detect rubric-loop overfitting (the agent learning to game the verifier rather than reproduce the paper). PaperBench upstream ships ~20 papers; we vendor three.

### A.5 The pivot context (load-bearing for sequencing)

Per `docs/design/rlm-pivot-brief.md`, the 14-stage `PipelineStage` state machine is being replaced by an `rlms`-based recursive-LM orchestrator. Phases 1–3 merged; Phase 4/6 in open PRs. The `fix/leaf-scorer-honesty` branch we're on is fixing a regression introduced in the Phase 5 closeout.

**Implication for this plan.** Any improvement that touches orchestrator semantics (stage ordering, gate logic, rubric verification flow) lands in a moving target — it will need to be re-implemented twice or it will be ripped out by the pivot. Improvements at the **sandbox layer** (`runtime/`) and **resilience layer** (`resilience/`) are stable across the pivot because both layers are explicitly preserved in `rlm-pivot-brief.md §4` ("what survives").

This plan therefore biases hard toward sandbox + resilience work and defers orchestrator-touching improvements until after the pivot lands.

---

## Part B — Improvement catalog

Seven candidates, ranked by leverage (impact ÷ effort). Each item lists: **problem**, **change**, **effort**, **impact**, **blast radius**, **pivot-safe?**.

### B.1 ★★★ Pod-time budget cap (`max_pod_seconds`) — **Phase 1 of this plan**

- **Problem.** Nothing stops a hung pod from billing indefinitely. `--max-usd` only caps LLM spend.
- **Change.** Add `max_pod_seconds: float | None` to `RunBudget`. RunPod backend records pod-start timestamp on `create_sandbox()`, checks budget on each `exec()`, raises `BudgetExhausted` and forces `destroy()` on overage.
- **Effort.** ~1 day (one new dataclass field, one timestamp on `Sandbox`, two budget checks, ~5 tests).
- **Impact.** Closes the largest unbounded-cost vector. Cheap to add, expensive to be without.
- **Blast radius.** Sandbox + resilience layer only. Pure addition — opt-in via new flag, default `None` preserves current behavior.
- **Pivot-safe?** Yes — both layers survive the pivot.

### B.2 ★★★ Persistent pod reuse across the improvement loop

- **Problem.** Each rubric-improvement-loop iteration spins up a new pod when not using `REPROLAB_RUNPOD_POD_ID`. Boot wait is ~2-5 min and cold caches re-download datasets/weights every iteration.
- **Change.** Pipeline-level option `--reuse-pod` (and `REPROLAB_RUNPOD_REUSE_POD=true`). When set, `experiment-runner` reuses one pod across improvement iterations, with an explicit `clean_workspace()` step between iterations (rm `work/`, keep `artifacts/`).
- **Effort.** ~3 days. Touches orchestrator (knows when an iteration starts/ends), sandbox interface (add `reset_workspace()`), and the improvement loop.
- **Impact.** Saves 2-5 min × N iterations of wall-clock, eliminates dataset re-download cost. On a 5-iteration run that's 10-25 min saved + GB of network traffic.
- **Blast radius.** Orchestrator-touching. Higher.
- **Pivot-safe?** **Partial.** The sandbox-side reset is pivot-safe; the orchestrator-side iteration boundary is being rewritten in the RLM path. Defer until pivot lands OR implement against the RLM `experiment` primitive directly.

### B.3 ★★ Pod warm pool / template baking

- **Problem.** First-run boot tax is ~2-5 min even on a "fast" pod. The default image is a 4GB+ PyTorch container pulled from RunPod registry on every boot.
- **Change.** Build a custom RunPod template containing the typical reproduction prereqs (PyTorch, common ML libs, dataset cache mount-points) and ship it as `REPROLAB_RUNPOD_IMAGE` default. Optionally maintain a small warm pool (1-2 pre-booted pods) via a tiny manager script.
- **Effort.** ~2-3 days. Mostly Docker work + RunPod template upload, plus a manager script if warm pool included.
- **Impact.** Cuts boot wait from ~3 min to ~30s.
- **Blast radius.** External (RunPod templates) + minor backend.
- **Pivot-safe?** Yes — sandbox-only.

### B.4 ★★ Incremental artifact sync (rsync semantics)

- **Problem.** `runpod_backend.py:507-523` tars the entire `artifacts/` directory back after every `exec()`. Large checkpoints get re-shipped each time.
- **Change.** Replace tar-stream with `asyncssh` rsync-style transfer (file-modification-time + size check on each artifact, only ship changed bytes).
- **Effort.** ~2 days. One method rewrite, robust testing of edge cases (symlinks, hardlinks, the existing tar safety check at line 790-803 must be preserved).
- **Impact.** Meaningful on runs with large model checkpoints; negligible on small-artifact runs.
- **Blast radius.** Sandbox only.
- **Pivot-safe?** Yes.

### B.5 ★★ Rubric verdict caching by (criterion, output-hash)

- **Problem.** When the improvement loop re-runs an experiment that produced *identical* artifacts to a prior iteration (e.g. the agent only changed comments, or the change didn't affect the eval), the verifier re-pays its LLM cost to reach the same verdict.
- **Change.** Hash the input bundle the verifier sees (rubric criterion + artifact subset) and cache the verdict in `runs/<id>/.verifier-cache/`. Cache-hit short-circuits the LLM call.
- **Effort.** ~2 days. Cache layer + invalidation rules.
- **Impact.** Saves verifier LLM cost on no-op iterations. Variable — depends on how often the agent makes no-op changes.
- **Blast radius.** Verifier layer.
- **Pivot-safe?** **Partial.** The verifier is being rewritten under RLM. Defer or build against the new verifier interface.

### B.6 ★ Unified per-run cost+timing trace

- **Problem.** Diagnosing "this run cost $X over Y minutes" requires joining three files. Cost telemetry exists for LLM but not for pod-time.
- **Change.** New `runs/<id>/trace.jsonl` that emits canonical timed events: `pod.created`, `pod.exec.started/finished` (with command + duration), `pod.destroyed`, `llm.invocation` (with tokens + $), `verifier.scored`. One writer, one schema.
- **Effort.** ~3 days. Schema design + emitters at each callsite + a small reader/aggregator.
- **Impact.** Diagnostic + retrospective value. Foundational for cost dashboards.
- **Blast radius.** Cross-cutting (touches every callsite).
- **Pivot-safe?** Mostly yes — the emit points are stable. Schema should be agreed with the RLM event model in `rlm-pivot-brief.md §9`.

### B.7 ★ Expand PaperBench test corpus from 3 → 10

- **Problem.** Three papers can't detect rubric-loop overfitting. PaperBench upstream has more.
- **Change.** Vendor 7 more bundles from `openai/preparedness`, prioritizing topical breadth (NLP, vision, RL, tabular, theory).
- **Effort.** ~1 day (mostly LFS download + config wiring).
- **Impact.** Diagnostic, not operational. Important for trustworthy benchmarking, not for any single run.
- **Blast radius.** Data only.
- **Pivot-safe?** Yes.

### B.8 Summary table

| # | Item | Effort | Impact | Pivot-safe? | Phase |
|---|---|---|---|---|---|
| B.1 | `max_pod_seconds` budget | 1d | ★★★ | ✅ | **P1** |
| B.4 | Incremental artifact sync | 2d | ★★ | ✅ | P1 |
| B.3 | Pod warm pool / template | 2-3d | ★★ | ✅ | P2 |
| B.6 | Unified cost+timing trace | 3d | ★ | ✅ (schema align) | P2 |
| B.7 | Expand test corpus | 1d | ★ | ✅ | P2 |
| B.2 | Persistent pod reuse | 3d | ★★★ | ⚠ (orch) | P3 (post-pivot) |
| B.5 | Rubric verdict caching | 2d | ★★ | ⚠ (verifier) | P3 (post-pivot) |

### B.9 Phasing

- **P1 — Now, sandbox-only, pivot-safe.** B.1 (this plan, below), then B.4.
- **P2 — Soon, foundational.** B.6 first (because B.3 + B.7 benefit from having a trace to measure them).
- **P3 — After RLM pivot lands.** B.2 and B.5 — both depend on stable orchestrator/verifier interfaces.

---

## Part C — Detailed Phase 1 plan: `max_pod_seconds` budget

> Scope: add a pod-time budget that captures the largest unbounded-cost vector. Sandbox + resilience layer only. Default off; explicit opt-in via CLI flag and env var.

### File structure

**Create:**
- `tests/agents/resilience/test_run_budget_pod_seconds.py` — unit tests for the new budget field.
- `tests/services/runtime/test_runpod_pod_time_budget.py` — integration tests with mocked RunPod client.

**Modify:**
- `backend/agents/resilience/budget.py:13-51` — add field + check.
- `backend/services/runtime/runpod_backend.py:117-371` — record start time on `Sandbox`, check budget in `exec()`, force `destroy()` on overage.
- `backend/cli.py` — add `--max-pod-seconds` CLI flag wired into `RunBudget`.
- `.env.example` — document new variable `REPROLAB_MAX_POD_SECONDS`.

**Audit (read-only, no changes expected):**
- `backend/agents/resilience/failures.py` — `BudgetExhausted` shape; confirm new field carries through.
- `backend/agents/resilience/cost.py` — `RunCostLedger`; we do not extend it (pod-time is *separate* from $-spend by design — keep the orthogonality).

### Architecture decisions (locked in here, do not revisit during execution)

1. **Pod-time is its own budget axis.** Not lumped into `max_usd` because pod billing rate is not known to the system (varies by GPU type, RunPod price changes). We cap *seconds*, not dollars. Dollar conversion is a downstream concern (B.6 trace).
2. **Pod-time clock starts at `Sandbox.created_at`** (the moment the pod transitions to RUNNING + SSH connects), not at `POST /pods`. Boot time is RunPod's problem, not the user's budget.
3. **Enforcement happens at `exec()` boundary**, not by a background watchdog. Reason: deterministic, testable, no thread-management complexity. Trade-off: a single hung `exec()` can exceed budget by up to its own timeout; the existing `exec(timeout=...)` already caps that.
4. **On overage, force `destroy()` and raise `BudgetExhausted`.** The pod *must* be killed — leaving a runaway pod alive defeats the budget's purpose.
5. **`destroy()` is best-effort.** If the DELETE call fails, log loudly but still raise `BudgetExhausted` to the orchestrator. The user gets one alert; we don't loop trying to kill.
6. **Default `None` (off).** Backwards-compatible. Users opt in.

### Tasks

---

### Task 1: Add `max_pod_seconds` field to `RunBudget`

**Files:**
- Modify: `backend/agents/resilience/budget.py:13-16`
- Test: `tests/agents/resilience/test_run_budget_pod_seconds.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/agents/resilience/test_run_budget_pod_seconds.py`:

```python
"""Tests for the max_pod_seconds field on RunBudget."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.agents.resilience.budget import RunBudget
from backend.agents.resilience.cost import RunCostLedger
from backend.agents.resilience.failures import BudgetExhausted


def _frozen_now() -> datetime:
    return datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)


def test_run_budget_accepts_max_pod_seconds_field():
    budget = RunBudget(max_pod_seconds=600.0)
    assert budget.max_pod_seconds == 600.0


def test_run_budget_defaults_max_pod_seconds_to_none():
    budget = RunBudget()
    assert budget.max_pod_seconds is None


def test_check_pod_seconds_raises_when_pod_started_at_exceeds_cap():
    budget = RunBudget(max_pod_seconds=60.0)
    pod_started_at = _frozen_now() - timedelta(seconds=61)
    with pytest.raises(BudgetExhausted) as exc:
        budget.check_pod_seconds(
            pod_started_at=pod_started_at,
            agent_id="experiment-runner",
            now=_frozen_now(),
        )
    assert "61" in str(exc.value)
    assert "60" in str(exc.value)


def test_check_pod_seconds_noop_when_under_cap():
    budget = RunBudget(max_pod_seconds=600.0)
    pod_started_at = _frozen_now() - timedelta(seconds=10)
    budget.check_pod_seconds(
        pod_started_at=pod_started_at,
        agent_id="experiment-runner",
        now=_frozen_now(),
    )  # must not raise


def test_check_pod_seconds_noop_when_cap_is_none():
    budget = RunBudget(max_pod_seconds=None)
    pod_started_at = _frozen_now() - timedelta(seconds=99_999)
    budget.check_pod_seconds(
        pod_started_at=pod_started_at,
        agent_id="experiment-runner",
        now=_frozen_now(),
    )  # must not raise


def test_check_pod_seconds_noop_when_pod_started_at_is_none():
    budget = RunBudget(max_pod_seconds=60.0)
    budget.check_pod_seconds(
        pod_started_at=None,
        agent_id="experiment-runner",
        now=_frozen_now(),
    )  # must not raise — no pod, no enforcement
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/agents/resilience/test_run_budget_pod_seconds.py -v`
Expected: FAIL — `RunBudget` has no `max_pod_seconds` field or `check_pod_seconds` method.

- [ ] **Step 3: Implement minimal field + method**

Modify `backend/agents/resilience/budget.py`:

```python
"""Run-level budget checks for resilient provider attempts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from backend.agents.resilience.cost import RunCostLedger
from backend.agents.resilience.failures import BudgetExhausted


@dataclass(frozen=True)
class RunBudget:
    max_usd: float | None = None
    max_wall_clock_seconds: float | None = None
    max_pod_seconds: float | None = None
    max_invocations_per_agent: dict[str, int] = field(default_factory=dict)
    rlm_calls_remaining: int = 120

    def check(
        self,
        *,
        ledger: RunCostLedger,
        started_at: datetime,
        agent_id: str,
        attempt_count: int,
    ) -> None:
        if self.max_usd is not None and ledger.total_usd() >= self.max_usd:
            raise BudgetExhausted(
                f"Run cost budget exhausted before invoking {agent_id}: "
                f"${ledger.total_usd():.4f} >= ${self.max_usd:.4f}",
                provider=None,
                agent_id=agent_id,
            )
        if self.max_wall_clock_seconds is not None:
            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            if elapsed >= self.max_wall_clock_seconds:
                raise BudgetExhausted(
                    f"Run wall-clock budget exhausted before invoking {agent_id}: "
                    f"{elapsed:.1f}s >= {self.max_wall_clock_seconds:.1f}s",
                    provider=None,
                    agent_id=agent_id,
                    elapsed_seconds=elapsed,
                )
        max_attempts = self.max_invocations_per_agent.get(agent_id)
        if max_attempts is not None and attempt_count >= max_attempts:
            raise BudgetExhausted(
                f"Invocation budget exhausted for {agent_id}: "
                f"{attempt_count} >= {max_attempts}",
                provider=None,
                agent_id=agent_id,
            )

    def check_pod_seconds(
        self,
        *,
        pod_started_at: datetime | None,
        agent_id: str,
        now: datetime | None = None,
    ) -> None:
        if self.max_pod_seconds is None or pod_started_at is None:
            return
        current = now if now is not None else datetime.now(timezone.utc)
        elapsed = (current - pod_started_at).total_seconds()
        if elapsed >= self.max_pod_seconds:
            raise BudgetExhausted(
                f"Pod-time budget exhausted before invoking {agent_id}: "
                f"{elapsed:.1f}s >= {self.max_pod_seconds:.1f}s",
                provider=None,
                agent_id=agent_id,
                elapsed_seconds=elapsed,
            )


__all__ = ["RunBudget"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/agents/resilience/test_run_budget_pod_seconds.py -v`
Expected: 5 passed.

- [ ] **Step 5: Verify existing budget tests still pass**

Run: `.venv/bin/python -m pytest tests/agents/resilience/ -v`
Expected: all green — `frozen=True` dataclass change must be field-additive only.

- [ ] **Step 6: Commit**

```bash
git add backend/agents/resilience/budget.py tests/agents/resilience/test_run_budget_pod_seconds.py
git commit -m "feat(budget): add max_pod_seconds field with check_pod_seconds method"
```

---

### Task 2: Record pod-start timestamp on `Sandbox`

**Files:**
- Modify: the `Sandbox` dataclass (search location below in Step 1)
- Modify: `backend/services/runtime/runpod_backend.py:117-275` (`create_sandbox()` — set the timestamp after SSH connects)
- Test: `tests/services/runtime/test_runpod_pod_time_budget.py` (new)

- [ ] **Step 1: Locate the `Sandbox` dataclass**

Run: `grep -rn "^class Sandbox\|@dataclass.*\nclass Sandbox" backend/services/runtime/ --include="*.py"`

Expected: a hit for the canonical `Sandbox` definition (likely `backend/services/runtime/base.py` or `backend/services/runtime/types.py`). Note the file:line for Step 3.

- [ ] **Step 2: Write the failing test**

Create `tests/services/runtime/test_runpod_pod_time_budget.py`:

```python
"""Integration tests for RunPod pod-time budget enforcement.

These tests use a fully mocked RunPod client + asyncssh connection;
no real pod is created.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agents.resilience.budget import RunBudget
from backend.agents.resilience.failures import BudgetExhausted
from backend.services.runtime.runpod_backend import RunpodBackend


def _frozen_now() -> datetime:
    return datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_sandbox_records_pod_started_at_on_create(monkeypatch):
    """After create_sandbox(), the returned Sandbox has pod_started_at set."""
    # NOTE: This test will be filled in with the real mocking pattern in
    # Step 4 once we've inspected the existing test_runpod_*.py fixtures.
    # For the failing-test step we assert the attribute exists.
    from backend.services.runtime.runpod_backend import RunpodBackend
    backend = RunpodBackend.__new__(RunpodBackend)  # no __init__ side effects
    # Smoke: the Sandbox type produced by this backend must expose the field.
    from backend.services.runtime.base import Sandbox  # adjust import per Task 2 Step 1
    fields = {f.name for f in Sandbox.__dataclass_fields__.values()}
    assert "pod_started_at" in fields, (
        "Sandbox must expose pod_started_at for pod-time budget enforcement"
    )
```

(Tests for the *enforcement* path live in Task 3.)

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_runpod_pod_time_budget.py::test_sandbox_records_pod_started_at_on_create -v`
Expected: FAIL — `Sandbox` has no `pod_started_at` field.

- [ ] **Step 4: Add `pod_started_at: datetime | None = None` to `Sandbox`**

Edit the `Sandbox` dataclass (path identified in Step 1). Add the field with `None` default so non-RunPod backends (local, docker) don't have to populate it.

Example (adjust path to match Step 1's finding):

```python
@dataclass
class Sandbox:
    # ... existing fields ...
    pod_started_at: datetime | None = None
```

- [ ] **Step 5: Populate `pod_started_at` in `RunpodBackend.create_sandbox()`**

In `backend/services/runtime/runpod_backend.py`, locate the point where `create_sandbox` constructs the `Sandbox` to return (around line 250-275, after SSH connects). Set `pod_started_at=datetime.now(timezone.utc)`.

Concretely, find the `return Sandbox(...)` (or `sandbox = Sandbox(...)`) and add the kwarg. If the function returns a `Sandbox` that's mutated later, set the attribute at the point of successful SSH connect:

```python
from datetime import datetime, timezone
# ...
# Immediately after the SSH connection is established:
sandbox.pod_started_at = datetime.now(timezone.utc)
```

If `Sandbox` is `frozen=True`, use `dataclasses.replace(sandbox, pod_started_at=datetime.now(timezone.utc))` or set the field at construction.

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_runpod_pod_time_budget.py::test_sandbox_records_pod_started_at_on_create -v`
Expected: PASS.

- [ ] **Step 7: Run the full sandbox test suite to confirm no regressions**

Run: `.venv/bin/python -m pytest tests/services/runtime/ tests/test_runpod_delete_guardrails.py -v`
Expected: all green. The new field is `None`-defaulted, so local/docker backends are unaffected.

- [ ] **Step 8: Commit**

```bash
git add backend/services/runtime/base.py backend/services/runtime/runpod_backend.py tests/services/runtime/test_runpod_pod_time_budget.py
git commit -m "feat(sandbox): track pod_started_at on Sandbox for pod-time budgeting"
```

(Adjust the staged paths to match where `Sandbox` actually lives, per Step 1.)

---

### Task 3: Enforce `max_pod_seconds` at `exec()` boundary

**Files:**
- Modify: `backend/services/runtime/runpod_backend.py:278` (`exec()` method)
- Test: `tests/services/runtime/test_runpod_pod_time_budget.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/services/runtime/test_runpod_pod_time_budget.py`:

```python
@pytest.mark.asyncio
async def test_exec_raises_budget_exhausted_when_pod_time_exceeded():
    """exec() must raise BudgetExhausted when budget.max_pod_seconds is exceeded."""
    from backend.services.runtime.base import Sandbox

    backend = RunpodBackend.__new__(RunpodBackend)
    backend._owned_pod_ids = {"test-pod"}
    backend._ssh_connections = {}

    pod_started_at = _frozen_now() - timedelta(seconds=120)
    sandbox = Sandbox(
        sandbox_id="test-pod",
        # ... other required fields per the dataclass ...
        pod_started_at=pod_started_at,
    )
    budget = RunBudget(max_pod_seconds=60.0)
    backend._run_budget = budget  # set via the wiring added in Task 4

    with patch(
        "backend.services.runtime.runpod_backend.datetime"
    ) as mock_dt:
        mock_dt.now.return_value = _frozen_now()
        mock_dt.timezone = timezone

        with pytest.raises(BudgetExhausted) as exc:
            await backend.exec(sandbox, "echo hello", timeout=30)

    assert "120" in str(exc.value) or "Pod-time" in str(exc.value)


@pytest.mark.asyncio
async def test_exec_forces_destroy_on_budget_exhaustion():
    """When budget is exhausted in exec(), the pod must be destroyed."""
    from backend.services.runtime.base import Sandbox

    backend = RunpodBackend.__new__(RunpodBackend)
    backend._owned_pod_ids = {"test-pod"}
    backend._ssh_connections = {"test-pod": MagicMock()}
    backend.destroy = AsyncMock()

    pod_started_at = _frozen_now() - timedelta(seconds=120)
    sandbox = Sandbox(
        sandbox_id="test-pod",
        pod_started_at=pod_started_at,
    )
    backend._run_budget = RunBudget(max_pod_seconds=60.0)

    with patch(
        "backend.services.runtime.runpod_backend.datetime"
    ) as mock_dt:
        mock_dt.now.return_value = _frozen_now()
        mock_dt.timezone = timezone

        with pytest.raises(BudgetExhausted):
            await backend.exec(sandbox, "echo hello", timeout=30)

    backend.destroy.assert_awaited_once_with(sandbox)


@pytest.mark.asyncio
async def test_exec_does_not_check_when_no_budget_set():
    """exec() must not raise when backend has no run_budget configured."""
    from backend.services.runtime.base import Sandbox

    backend = RunpodBackend.__new__(RunpodBackend)
    backend._owned_pod_ids = {"test-pod"}
    backend._ssh_connections = {"test-pod": MagicMock(
        run=AsyncMock(return_value=MagicMock(stdout="ok", stderr="", exit_status=0))
    )}
    backend._run_budget = None

    pod_started_at = _frozen_now() - timedelta(seconds=99_999)
    sandbox = Sandbox(
        sandbox_id="test-pod",
        pod_started_at=pod_started_at,
    )

    # Should not raise — no budget configured even though elapsed is enormous.
    # (The actual exec() may still fail on SFTP/artifact-sync mocks; we only
    # assert the budget check itself doesn't trip.)
    try:
        await backend.exec(sandbox, "echo hello", timeout=30)
    except BudgetExhausted:
        pytest.fail("BudgetExhausted raised despite no budget configured")
    except Exception:
        pass  # other failures (SFTP, etc.) are acceptable in this skeletal mock
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_runpod_pod_time_budget.py -v`
Expected: the three new tests FAIL — `_run_budget` attribute not honored in `exec()`.

- [ ] **Step 3: Implement the budget check in `exec()`**

In `backend/services/runtime/runpod_backend.py`, at the top of `exec()` (right after the method signature, before any other work):

```python
async def exec(self, sandbox: Sandbox, command: str, timeout: int) -> ExecResult:
    budget = getattr(self, "_run_budget", None)
    if budget is not None and sandbox.pod_started_at is not None:
        try:
            budget.check_pod_seconds(
                pod_started_at=sandbox.pod_started_at,
                agent_id="experiment-runner",
            )
        except BudgetExhausted:
            # Force pod teardown before propagating — the entire point of
            # the budget is to stop the meter, and a runaway pod with the
            # destroy step skipped defeats that.
            try:
                await self.destroy(sandbox)
            except Exception:
                pass  # best-effort; surface the BudgetExhausted regardless
            raise
    # ... existing exec() body unchanged ...
```

Add the `BudgetExhausted` import at the top of the file if not already present:

```python
from backend.agents.resilience.failures import BudgetExhausted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_runpod_pod_time_budget.py -v`
Expected: 4 passed (the field test from Task 2 + the 3 new ones).

- [ ] **Step 5: Run the guardrail tests to confirm destroy() interaction is clean**

Run: `.venv/bin/python -m pytest tests/test_runpod_delete_guardrails.py -v`
Expected: all green. The forced `destroy()` on budget-exhaustion must respect the existing `_owned_pod_ids` allowlist — pods we don't own are still not deleted.

- [ ] **Step 6: Commit**

```bash
git add backend/services/runtime/runpod_backend.py tests/services/runtime/test_runpod_pod_time_budget.py
git commit -m "feat(runpod): enforce max_pod_seconds at exec() boundary with forced destroy"
```

---

### Task 4: Wire `RunBudget` into `RunpodBackend` at construction

**Files:**
- Modify: `backend/services/runtime/runpod_backend.py` (constructor)
- Modify: the backend factory that instantiates `RunpodBackend` (search below)

- [ ] **Step 1: Locate the RunpodBackend factory / instantiation site**

Run: `grep -rn "RunpodBackend(" backend/ --include="*.py"`

Expected: one or two callsites — likely `backend/services/runtime/factory.py` or similar. Note the path.

- [ ] **Step 2: Inspect `RunpodBackend.__init__`**

Read `backend/services/runtime/runpod_backend.py` near the top (lines ~80-116). Note the existing constructor signature.

- [ ] **Step 3: Add `run_budget: RunBudget | None = None` parameter to `__init__`**

```python
from backend.agents.resilience.budget import RunBudget

class RunpodBackend(RuntimeBackend):
    def __init__(
        self,
        # ... existing args ...
        run_budget: RunBudget | None = None,
    ):
        # ... existing init ...
        self._run_budget = run_budget
```

- [ ] **Step 4: Pass the budget from the factory**

At the factory callsite (from Step 1), thread the `run_budget` arg through from whatever already constructs `RunBudget`. Search for `RunBudget(` to find construction sites.

Run: `grep -rn "RunBudget(" backend/ --include="*.py"`

Add the `run_budget=...` kwarg to the `RunpodBackend(...)` call.

- [ ] **Step 5: Verify with a targeted integration test**

Run: `.venv/bin/python -m pytest tests/services/runtime/ -v`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add backend/services/runtime/runpod_backend.py backend/services/runtime/factory.py
git commit -m "feat(runpod): thread RunBudget through to backend for pod-time enforcement"
```

(Adjust factory path to match Step 1.)

---

### Task 5: CLI flag + env var

**Files:**
- Modify: `backend/cli.py` — add `--max-pod-seconds` flag
- Modify: `.env.example` — document `REPROLAB_MAX_POD_SECONDS`
- Modify: `.env` — add the new variable (empty, with a comment)

- [ ] **Step 1: Locate the existing `--max-usd` flag for pattern reference**

Run: `grep -n "max-usd\|max_usd" backend/cli.py`

Note the file:line — copy the surrounding flag-definition pattern.

- [ ] **Step 2: Write the failing test**

Create or append to `tests/cli/test_cli_budget_flags.py`:

```python
def test_cli_accepts_max_pod_seconds_flag():
    """The --max-pod-seconds flag must propagate into the constructed RunBudget."""
    from unittest.mock import patch
    from backend.cli import main  # adjust to actual entrypoint
    captured = {}

    def fake_run(*, run_budget, **kw):
        captured["budget"] = run_budget

    with patch("backend.cli._dispatch_reproduce", side_effect=fake_run):
        main([
            "reproduce", "dummy.pdf",
            "--mode", "offline",
            "--max-pod-seconds", "1800",
        ])

    assert captured["budget"].max_pod_seconds == 1800.0
```

(The exact import path for `main` and the dispatch hook depend on the CLI structure — read `backend/cli.py` first to wire this test against real symbols, not these placeholders.)

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cli/test_cli_budget_flags.py::test_cli_accepts_max_pod_seconds_flag -v`
Expected: FAIL — unknown flag.

- [ ] **Step 4: Add the flag**

In `backend/cli.py`, alongside the existing `--max-usd` definition:

```python
parser.add_argument(
    "--max-pod-seconds",
    type=float,
    default=None,
    help=(
        "Maximum elapsed pod time (seconds) before the run is killed and "
        "BudgetExhausted is raised. Counts from successful SSH connect, "
        "not from POST /pods. Default: no cap. Also settable via "
        "REPROLAB_MAX_POD_SECONDS."
    ),
)
```

And in the `RunBudget` construction site in cli.py:

```python
max_pod_seconds = args.max_pod_seconds
if max_pod_seconds is None:
    env_val = os.environ.get("REPROLAB_MAX_POD_SECONDS")
    if env_val:
        max_pod_seconds = float(env_val)

run_budget = RunBudget(
    max_usd=args.max_usd,
    max_wall_clock_seconds=args.max_wall_clock,
    max_pod_seconds=max_pod_seconds,
    # ... other fields ...
)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cli/test_cli_budget_flags.py -v`
Expected: PASS.

- [ ] **Step 6: Document in `.env.example`**

Append to the RunPod section of `.env.example`:

```bash
# Optional: maximum elapsed pod time (seconds) before the run is killed.
# Counts from successful SSH connect. Default: no cap.
# CLI override: --max-pod-seconds N
REPROLAB_MAX_POD_SECONDS=
```

Mirror the same block into `.env`.

- [ ] **Step 7: Commit**

```bash
git add backend/cli.py .env.example .env tests/cli/test_cli_budget_flags.py
git commit -m "feat(cli): --max-pod-seconds flag + REPROLAB_MAX_POD_SECONDS env var"
```

---

### Task 6: End-to-end smoke (manual verification, no real pod)

**Files:**
- None modified. Manual run.

- [ ] **Step 1: Confirm offline mode still works**

Run: `.venv/bin/python -m backend.cli reproduce paperbench1.pdf --mode offline --max-pod-seconds 60`
Expected: completes without the budget tripping (offline mode does not use RunPod).

- [ ] **Step 2: Confirm RunPod construction honors the budget (dry-run)**

Run: `.venv/bin/python -m pytest tests/ -k "runpod or budget" -v`
Expected: all green.

- [ ] **Step 3: Run the full test suite one more time as a regression sweep**

Run: `.venv/bin/python -m pytest tests/ -x`
Expected: green (or only pre-existing failures unrelated to this work — note them, don't fix them in this branch).

- [ ] **Step 4: Final commit if any docs need a touch-up**

If `CLAUDE.md` or `system_overview.md` reference budget controls, add `max_pod_seconds` alongside `max_usd` / `max_wall_clock`. Otherwise skip.

```bash
git add CLAUDE.md system_overview.md
git commit -m "docs: document max_pod_seconds budget alongside existing caps"
```

---

## Part D — Post-Phase-1: how to sequence the rest

Once B.1 lands, the next pick should be **B.4 (incremental artifact sync)** because:

1. It's sandbox-only — same surface area, same test patterns, same blast radius.
2. It composes with B.1 — a faster sync means fewer pod-seconds consumed, so the new budget bites less often on healthy runs.
3. The team is already in `runpod_backend.py` after Phase 1, context is hot.

B.6 (unified trace) should come *third* rather than second because its schema needs alignment with the RLM event model. Don't design the schema until the Phase 6 RLM PR is reviewable.

B.7 (expand test corpus) is a parallel-track effort — can run in another branch, not blocked on anything in this plan.

B.2 (persistent pod reuse) and B.5 (rubric caching) are **deliberately deferred** until after RLM pivot completion. Picking these up before then means writing the code twice. Re-evaluate when `feat/rlm-phase6-cleanup` (or its successor) merges.

---

## Self-review

**Spec coverage.** This plan covers:
- Audit of current infra ✅ (Part A.1-A.5)
- Improvement catalog ✅ (Part B.1-B.7, 7 items)
- Phasing with pivot-awareness ✅ (Part B.9)
- Detailed Phase 1 TDD plan ✅ (Part C, 6 tasks, real file:line refs)
- Post-Phase-1 sequencing ✅ (Part D)

**Placeholder scan.** Two known soft spots:
1. Task 2 Step 1 and Task 4 Step 1 ask the engineer to *locate* the canonical `Sandbox` dataclass and the factory site rather than hard-coding paths. This is deliberate — the repo's `runtime/` layout has shifted twice in the audit history and the file may be `base.py` or `types.py`. The grep is explicit and one-shot.
2. Task 5 Step 2 has placeholder symbol names (`backend.cli.main`, `_dispatch_reproduce`) because `backend/cli.py` wasn't fully read during plan-write. The engineer must read it before writing the test. The skill says no placeholders — this is a real soft spot. Mitigation: the test's *intent* (assert `max_pod_seconds=1800.0` propagates) is concrete; the engineer needs to wire it against real symbols, which is a 2-minute task they can't do wrong if they read the file first.

**Type consistency.** `max_pod_seconds: float | None` used consistently across Task 1 (dataclass), Task 3 (check), Task 5 (CLI parse). `pod_started_at: datetime | None` used consistently across Task 2 (Sandbox field), Task 3 (check argument). `BudgetExhausted` used consistently.

**Done.**

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-22-infrastructure-improvement-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach? Or do you want to revise the plan first — e.g. swap Phase 1 from `max_pod_seconds` to a different candidate, expand the catalog, or split Part C into its own plan file?
