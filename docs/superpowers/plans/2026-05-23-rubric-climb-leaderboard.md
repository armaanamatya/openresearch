# Rubric Climb Panel + Cross-Model Leaderboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land two coupled product surfaces — a live rubric climb panel inside the existing lab UI, and a read-only `/leaderboard` page ranking RLM root models on the same paper — without introducing new SSE events or new persistence layers.

**Architecture:** Enrich the existing `RubricStrip` band 2 in place with a count-up tween, SVG line-chart sparkline, per-area flip highlighting, and candidate attribution. Extend `RLMFinalReport` with `models`, `mode`, `started_at`, `completed_at` (forward-compatible with the cleanup-spec Phase 4 shape). Aggregate leaderboard rows at request time from `runs/*/final_report.json` + `demo_status.json`; no SQLite projection. Reducer (`useRlmRun`) gains `previousAreas` + `attributableCandidate` so the strip can render flips and attribution purely from state.

**Tech Stack:** Python 3.14 / FastAPI / Pydantic v2 (backend), Next.js 16 / React 19 / TypeScript / Vitest / Playwright (frontend), custom SVG + CSS animations (no chart lib).

**Reference spec:** `docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md` — read §3 (locked decisions) before each task; §4 has the architectural contract every task must match.

**Conventions:**
- Pytest commands run from repo root with `.venv/bin/python -m pytest`.
- Frontend commands run from `frontend/`.
- Commits do NOT include `Co-Authored-By` per project convention.
- Each task ends with a commit; if a hook fails, create a NEW commit, do not amend.
- "Confirm fail / Confirm pass" steps are non-skippable — they are the only honest signal a step is done.

---

## Phase 0 — Pre-flight

### Task 0.1: Confirm baseline test suite is green

**Files:** none — verification only.

- [ ] **Step 1: Run pytest baseline**

Run: `.venv/bin/python -m pytest tests/ -n auto --tb=no -q 2>&1 | tail -5`
Expected: `≥1083 passed, 0 failed, 1 xfailed, 5 skipped` (the T24 xfail is the only one).

- [ ] **Step 2: Run frontend baseline**

Run: `cd frontend && npm run lint 2>&1 | tail -3 && npx tsc --noEmit && npm test 2>&1 | tail -5`
Expected: lint clean, tsc clean, vitest passing.

- [ ] **Step 3: Capture the baseline numbers in a working note**

Run: `echo "baseline: pytest=PASS_COUNT, vitest=PASS_COUNT" > .plan-baseline.txt`
(This file is a working artifact; do not commit it.)

No commit for this task — verification only.

---

## Phase 1 — Backend: `final_report.json` extensions

### Task 1.1: Add `models`, `mode`, `started_at`, `completed_at` to `RLMFinalReport`

**Files:**
- Modify: `backend/agents/rlm/report.py` (the `RLMFinalReport` class)
- Test: `tests/rlm/test_final_report_extended_fields.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/rlm/test_final_report_extended_fields.py`:

```python
"""Tests for the extended fields on RLMFinalReport (spec §4.5):
models, mode, started_at, completed_at."""

from backend.agents.rlm.report import RLMFinalReport


def test_default_values_are_filled():
    report = RLMFinalReport()
    assert report.mode == "rlm"
    assert report.models == {
        "planner": None, "executor": None, "verifier": None, "grader": None,
    }
    assert report.started_at is None
    assert report.completed_at is None


def test_populated_values_round_trip():
    report = RLMFinalReport(
        mode="rlm",
        models={
            "planner": "gpt-5",
            "executor": "claude-sonnet-4-6",
            "verifier": None,
            "grader": None,
        },
        started_at="2026-05-23T04:10:09+00:00",
        completed_at="2026-05-23T04:12:33+00:00",
    )
    dumped = report.model_dump()
    assert dumped["mode"] == "rlm"
    assert dumped["models"]["planner"] == "gpt-5"
    assert dumped["models"]["executor"] == "claude-sonnet-4-6"
    assert dumped["started_at"] == "2026-05-23T04:10:09+00:00"
    assert dumped["completed_at"] == "2026-05-23T04:12:33+00:00"


def test_legacy_json_back_compat():
    """A pre-extension final_report.json missing the new keys must still parse."""
    legacy = {
        "paper": {},
        "verdict": "failed",
        "reproduction_summary": "x",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": {"overall_score": 0.0, "meets_target": False, "areas": []},
        "improvements": [],
        "primitive_trace": {},
        "cost": {"llm_usd": 0.0, "primitives": 0.0},
        "iterations": 0,
    }
    report = RLMFinalReport.model_validate(legacy)
    assert report.mode == "rlm"
    assert report.models["planner"] is None
    assert report.started_at is None
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_final_report_extended_fields.py -v`
Expected: 3 failures — fields don't exist on `RLMFinalReport`.

- [ ] **Step 3: Read `RLMFinalReport` to find the insertion point**

Run: `.venv/bin/python -c "import inspect, backend.agents.rlm.report as m; print(inspect.getsourcelines(m.RLMFinalReport)[1])"`
Use Read on `backend/agents/rlm/report.py` around that line.

- [ ] **Step 4: Add the new fields to `RLMFinalReport`**

In `backend/agents/rlm/report.py`, in the `class RLMFinalReport(BaseModel):` body, add (after existing fields, before the `model_config = ConfigDict(...)` line if any):

```python
    # --- Phase-4-forward-compat fields (spec 2026-05-23-rubric-climb-leaderboard §4.5)
    mode: Literal["rlm", "rdr"] = Field(
        default="rlm",
        description="Reproduction mode (rlm root-loop or rdr controller).",
    )
    models: dict[str, str | None] = Field(
        default_factory=lambda: {
            "planner": None,
            "executor": None,
            "verifier": None,
            "grader": None,
        },
        description=(
            "Per-role model identifiers — forward-compatible with the "
            "future per-role picker. Unknown roles stay None until populated."
        ),
    )
    started_at: str | None = Field(
        default=None,
        description="ISO-8601 UTC timestamp when the run started.",
    )
    completed_at: str | None = Field(
        default=None,
        description="ISO-8601 UTC timestamp when the run finished writing the report.",
    )
```

If `Literal` and `Field` are not already imported, add them: `from typing import Literal` and import `Field` from `pydantic`.

- [ ] **Step 5: Run the test to confirm it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_final_report_extended_fields.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run the full report-module tests to confirm no regression**

Run: `.venv/bin/python -m pytest tests/rlm/ -k "report or final_report" -v --tb=short`
Expected: no new failures vs baseline.

- [ ] **Step 7: Commit**

```bash
git add backend/agents/rlm/report.py tests/rlm/test_final_report_extended_fields.py
git commit -m "feat(report): add models/mode/started_at/completed_at to RLMFinalReport

Forward-compatible extension that lets every new run record its root and
sub-agent models plus its wall-clock window. Fields default to None /
'rlm' so legacy on-disk JSON still parses without migration.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.5"
```

### Task 1.2: Populate the new fields from `_finalize`

**Files:**
- Modify: `backend/agents/rlm/run.py` (the `_finalize` function)
- Test: `tests/rlm/test_final_report_populated_from_run.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/rlm/test_final_report_populated_from_run.py`:

```python
"""Verify _finalize populates the new RLMFinalReport fields from the run.

Spec: 2026-05-23-rubric-climb-leaderboard §4.5.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.agents.rlm.report import RLMFinalReport
from backend.agents.rlm.run import _finalize


class _FakeCtx:
    """Minimal RunContext stub — _finalize uses .project_id and .agent_model."""

    def __init__(self, project_id, agent_model):
        self.project_id = project_id
        self.agent_model = agent_model


def test_finalize_writes_planner_executor_and_timestamps(tmp_path: Path):
    # _finalize reads started_at from demo_status.json (already on disk at finalize time).
    started_iso = "2026-05-23T04:10:09+00:00"
    (tmp_path / "demo_status.json").write_text(json.dumps({
        "projectId": "prj_test_finalize",
        "status": "running",
        "startedAt": started_iso,
    }))

    ctx = _FakeCtx("prj_test_finalize", "claude-sonnet-4-6")
    emitted: list[dict] = []
    result = _finalize(
        project_dir=tmp_path,
        ctx=ctx,
        emit=emitted.append,
        result_obj=None,
        iterations=2,
        run_failed=False,
        llm_model="gpt-5",
        corpus_sentinels=None,
        tools_label="real",
    )

    fr_path = tmp_path / "final_report.json"
    assert fr_path.exists()
    payload = json.loads(fr_path.read_text())

    assert payload["mode"] == "rlm"
    assert payload["models"]["planner"] == "gpt-5"
    assert payload["models"]["executor"] == "claude-sonnet-4-6"
    assert payload["models"]["verifier"] is None
    assert payload["models"]["grader"] is None
    assert payload["started_at"] == started_iso
    # completed_at is timestamp at write time; ISO-8601 >= started_at.
    started = datetime.fromisoformat(started_iso)
    completed = datetime.fromisoformat(payload["completed_at"])
    assert completed >= started
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_final_report_populated_from_run.py -v`
Expected: fails — `_finalize` does not accept `llm_model` or write the new fields.

- [ ] **Step 3: Inspect the current `_finalize` signature**

Run: `grep -n "def _finalize" /Volumes/CS_Stuff/openresearch/backend/agents/rlm/run.py`
Read the function (it's around line 660–755 per earlier exploration).

- [ ] **Step 4: Extend `_finalize` to accept `llm_model` and populate the fields**

**Verified 2026-05-23:** `RunContext` (at `backend/agents/rlm/context.py:18`) has NO `started_at_utc` field. The honest source of `started_at` is `runs/<project_id>/demo_status.json::startedAt` — already written at run start. Read it back in `_finalize`.

In `backend/agents/rlm/run.py`:

1. Locate the `_finalize` signature. Add `llm_model: str | None = None,` as a keyword-only argument (after the existing keyword args).
2. After `report.iterations = iterations` and the `if run_failed:` line, but BEFORE `write_final_report_rlm(report, project_dir)`, add:

```python
    # --- Phase-4-forward-compat metadata (spec 2026-05-23-rubric-climb-leaderboard §4.5)
    import json as _json
    from datetime import datetime, timezone as _tz
    report.mode = "rlm"
    report.models = {
        "planner": llm_model,
        "executor": getattr(ctx, "agent_model", None),
        "verifier": None,
        "grader": None,
    }
    # started_at: lifted from demo_status.json (written at run start by _write_demo_status).
    started_at: str | None = None
    demo_status_path = project_dir / "demo_status.json"
    if demo_status_path.exists():
        try:
            started_at = _json.loads(demo_status_path.read_text()).get("startedAt")
        except (OSError, _json.JSONDecodeError):
            started_at = None
    report.started_at = started_at
    report.completed_at = datetime.now(_tz.utc).isoformat()
```

3. Find every call site of `_finalize` (`grep -n "_finalize(" backend/agents/rlm/run.py`) and thread `llm_model=llm_model` into each call. There should be at least one site in `run_pipeline_rlm` after `llm_client, llm_model = _build_llm_client(...)`.

- [ ] **Step 5: Run the test to confirm it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_final_report_populated_from_run.py -v`
Expected: 1 passed.

- [ ] **Step 6: Run the full rlm-test directory to catch regressions**

Run: `.venv/bin/python -m pytest tests/rlm/ -q --tb=short 2>&1 | tail -15`
Expected: no new failures.

- [ ] **Step 7: Commit**

```bash
git add backend/agents/rlm/run.py tests/rlm/test_final_report_populated_from_run.py
git commit -m "feat(run): populate models/timestamps in final_report at finalize

Threads the resolved root model (llm_model) and sub-agent model
(ctx.agent_model) into the new RLMFinalReport.models dict; records
started_at from RunContext and completed_at at write time.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.5"
```

---

## Phase 2 — Backend: Leaderboard endpoint

### Task 2.1: Aggregator + schema in a new module

**Files:**
- Create: `backend/routes/leaderboard.py`
- Test: `tests/routes/test_leaderboard_aggregator.py` (new)
- Test: `tests/routes/__init__.py` (new, empty — only if `tests/routes/` does not exist)

- [ ] **Step 1: Confirm whether `tests/routes/` exists**

Run: `ls /Volumes/CS_Stuff/openresearch/tests/routes/ 2>/dev/null || echo MISSING`
If MISSING, create `tests/routes/__init__.py` as an empty file.

- [ ] **Step 2: Write the failing test for the aggregator**

Create `tests/routes/test_leaderboard_aggregator.py`:

```python
"""Tests for the leaderboard aggregator.

Spec: 2026-05-23-rubric-climb-leaderboard §4.4–§4.5.
"""

import json
from pathlib import Path

import pytest

from backend.routes.leaderboard import (
    LeaderboardRow,
    RoleModels,
    aggregate_leaderboard,
)


def _write_run(
    runs_root: Path,
    project_id: str,
    *,
    paper_id: str,
    paper_title: str,
    overall_score: float,
    cost_usd: float,
    iterations: int,
    planner: str | None,
    executor: str | None,
    mode: str = "rlm",
    started_at: str = "2026-05-23T04:10:00+00:00",
    completed_at: str = "2026-05-23T04:15:00+00:00",
    meets_target: bool = False,
    degraded: bool = False,
    verdict: str = "partial",
) -> None:
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True, exist_ok=True)
    final = {
        "paper": {"id": paper_id, "title": paper_title},
        "verdict": verdict,
        "reproduction_summary": "x",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": {
            "overall_score": overall_score,
            "meets_target": meets_target,
            "areas": [],
        },
        "improvements": [],
        "primitive_trace": {},
        "cost": {"llm_usd": cost_usd, "primitives": 0.0},
        "iterations": iterations,
        "primitive_provider": "real",
        "degraded": degraded,
        "mode": mode,
        "models": {
            "planner": planner,
            "executor": executor,
            "verifier": None,
            "grader": None,
        },
        "started_at": started_at,
        "completed_at": completed_at,
    }
    (run_dir / "final_report.json").write_text(json.dumps(final))
    (run_dir / "demo_status.json").write_text(json.dumps({
        "projectId": project_id, "status": "completed",
    }))


def test_aggregate_returns_empty_when_runs_dir_is_empty(tmp_path: Path):
    rows = aggregate_leaderboard(tmp_path)
    assert rows == []


def test_aggregate_returns_one_row_per_completed_run(tmp_path: Path):
    _write_run(
        tmp_path, "prj_a",
        paper_id="p1", paper_title="Paper One",
        overall_score=0.42, cost_usd=1.23, iterations=8,
        planner="gpt-5", executor="claude-sonnet-4-6",
    )
    _write_run(
        tmp_path, "prj_b",
        paper_id="p1", paper_title="Paper One",
        overall_score=0.71, cost_usd=2.50, iterations=12,
        planner="claude-opus-4-7", executor="claude-sonnet-4-6",
    )

    rows = aggregate_leaderboard(tmp_path)
    assert len(rows) == 2
    # Default sort: score desc.
    assert rows[0].project_id == "prj_b"
    assert rows[0].overall_score == pytest.approx(0.71)
    assert rows[1].project_id == "prj_a"


def test_aggregate_skips_runs_with_no_final_report(tmp_path: Path):
    _write_run(
        tmp_path, "prj_complete",
        paper_id="p1", paper_title="P", overall_score=0.5,
        cost_usd=1.0, iterations=5, planner=None, executor=None,
    )
    # In-flight run: only demo_status, no final_report.
    in_flight = tmp_path / "prj_in_flight"
    in_flight.mkdir()
    (in_flight / "demo_status.json").write_text("{}")

    rows = aggregate_leaderboard(tmp_path)
    assert [r.project_id for r in rows] == ["prj_complete"]


def test_aggregate_back_compat_legacy_run_missing_new_fields(tmp_path: Path):
    legacy = tmp_path / "prj_legacy"
    legacy.mkdir()
    (legacy / "final_report.json").write_text(json.dumps({
        "paper": {"id": "p1", "title": "Legacy"},
        "verdict": "failed",
        "reproduction_summary": "x",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": {"overall_score": 0.0, "meets_target": False, "areas": []},
        "improvements": [],
        "primitive_trace": {},
        "cost": {"llm_usd": 0.0, "primitives": 0.0},
        "iterations": 0,
    }))
    (legacy / "demo_status.json").write_text("{}")

    rows = aggregate_leaderboard(tmp_path)
    assert len(rows) == 1
    assert rows[0].mode == "rlm"
    assert rows[0].models.planner is None
    assert rows[0].started_at is None


def test_aggregate_filter_by_paper(tmp_path: Path):
    _write_run(tmp_path, "a", paper_id="p1", paper_title="P1",
               overall_score=0.5, cost_usd=1, iterations=1,
               planner=None, executor=None)
    _write_run(tmp_path, "b", paper_id="p2", paper_title="P2",
               overall_score=0.6, cost_usd=1, iterations=1,
               planner=None, executor=None)

    rows = aggregate_leaderboard(tmp_path, paper="p1")
    assert [r.paper_id for r in rows] == ["p1"]


def test_aggregate_filter_by_mode(tmp_path: Path):
    _write_run(tmp_path, "a", paper_id="p1", paper_title="P",
               overall_score=0.5, cost_usd=1, iterations=1,
               planner=None, executor=None, mode="rlm")
    _write_run(tmp_path, "b", paper_id="p1", paper_title="P",
               overall_score=0.6, cost_usd=1, iterations=1,
               planner=None, executor=None, mode="rdr")

    rows = aggregate_leaderboard(tmp_path, mode="rdr")
    assert [r.project_id for r in rows] == ["b"]


def test_aggregate_order_by_cost(tmp_path: Path):
    _write_run(tmp_path, "a", paper_id="p1", paper_title="P",
               overall_score=0.5, cost_usd=3.0, iterations=1,
               planner=None, executor=None)
    _write_run(tmp_path, "b", paper_id="p1", paper_title="P",
               overall_score=0.6, cost_usd=1.0, iterations=1,
               planner=None, executor=None)

    rows = aggregate_leaderboard(tmp_path, order_by="cost")
    # Ascending cost
    assert [r.project_id for r in rows] == ["b", "a"]


def test_aggregate_computes_wall_clock_from_timestamps(tmp_path: Path):
    _write_run(tmp_path, "a", paper_id="p1", paper_title="P",
               overall_score=0.5, cost_usd=1, iterations=1,
               planner=None, executor=None,
               started_at="2026-05-23T04:10:00+00:00",
               completed_at="2026-05-23T04:15:00+00:00")
    rows = aggregate_leaderboard(tmp_path)
    assert rows[0].wall_clock_s == 300.0
```

- [ ] **Step 3: Run the test to confirm it fails**

Run: `.venv/bin/python -m pytest tests/routes/test_leaderboard_aggregator.py -v`
Expected: ImportError — `backend.routes.leaderboard` does not exist.

- [ ] **Step 4: Confirm `backend/routes/` exists**

Run: `ls /Volumes/CS_Stuff/openresearch/backend/routes/ 2>/dev/null || echo MISSING`
If MISSING, this project mounts routes directly from `backend/app.py`. Create `backend/routes/__init__.py` (empty file) — and adapt the test import path if a different package layout is conventional in this repo.

Re-check the convention: `grep -rn "from backend\." /Volumes/CS_Stuff/openresearch/backend/app.py | head -5`. If routes live under a different module (e.g. `backend/api/routes/` or directly in `backend/app.py`), adopt that location instead and update both the source file path and the test import.

- [ ] **Step 5: Implement `backend/routes/leaderboard.py`**

Create `backend/routes/leaderboard.py`:

```python
"""Leaderboard aggregator + HTTP router.

Scans `runs/<id>/final_report.json` (+ `demo_status.json` for sandbox /
status fallback) and returns ranked rows. Read-only; not gated by
REPROLAB_DEMO_SECRET per spec §3 #10.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.4–§4.5.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.config import get_settings

logger = logging.getLogger(__name__)

OrderBy = Literal["score", "cost", "time", "finished_at"]


class RoleModels(BaseModel):
    planner: str | None = None
    executor: str | None = None
    verifier: str | None = None
    grader: str | None = None


class LeaderboardRow(BaseModel):
    project_id: str
    paper_id: str
    paper_title: str | None
    mode: Literal["rlm", "rdr"] = "rlm"
    models: RoleModels = Field(default_factory=RoleModels)
    overall_score: float
    meets_target: bool
    degraded: bool
    cost_usd: float | None
    iterations: int
    wall_clock_s: float | None
    sandbox: str | None
    started_at: str | None
    completed_at: str | None
    verdict: str


def _read_run(run_dir: Path) -> LeaderboardRow | None:
    fr_path = run_dir / "final_report.json"
    if not fr_path.is_file():
        return None
    try:
        data = json.loads(fr_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("leaderboard: skipping %s — unreadable final_report (%s)", run_dir.name, e)
        return None

    paper = data.get("paper") or {}
    rubric = data.get("rubric") or {}
    cost = data.get("cost") or {}
    models = data.get("models") or {}
    started_at = data.get("started_at")
    completed_at = data.get("completed_at")

    wall_clock_s: float | None = None
    if started_at and completed_at:
        try:
            wall_clock_s = (
                datetime.fromisoformat(completed_at)
                - datetime.fromisoformat(started_at)
            ).total_seconds()
        except ValueError:
            wall_clock_s = None

    return LeaderboardRow(
        project_id=run_dir.name,
        paper_id=str(paper.get("id") or run_dir.name),
        paper_title=paper.get("title"),
        mode=data.get("mode", "rlm"),
        models=RoleModels(**{
            "planner": models.get("planner"),
            "executor": models.get("executor"),
            "verifier": models.get("verifier"),
            "grader": models.get("grader"),
        }),
        overall_score=float(rubric.get("overall_score", 0.0)),
        meets_target=bool(rubric.get("meets_target", False)),
        degraded=bool(data.get("degraded", False)),
        cost_usd=float(cost["llm_usd"]) if cost.get("llm_usd") is not None else None,
        iterations=int(data.get("iterations", 0)),
        wall_clock_s=wall_clock_s,
        sandbox=data.get("sandbox"),
        started_at=started_at,
        completed_at=completed_at,
        verdict=str(data.get("verdict", "failed")),
    )


def aggregate_leaderboard(
    runs_root: Path,
    *,
    paper: str | None = None,
    mode: Literal["rlm", "rdr"] | None = None,
    order_by: OrderBy = "score",
    limit: int = 50,
) -> list[LeaderboardRow]:
    """Scan ``runs_root`` for completed runs and return a ranked list."""

    rows: list[LeaderboardRow] = []
    if not runs_root.is_dir():
        return rows

    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir():
            continue
        row = _read_run(entry)
        if row is None:
            continue
        if paper is not None and row.paper_id != paper:
            continue
        if mode is not None and row.mode != mode:
            continue
        rows.append(row)

    def _sort_key(r: LeaderboardRow):
        if order_by == "score":
            return -r.overall_score
        if order_by == "cost":
            # Asc; missing cost last.
            return (r.cost_usd is None, r.cost_usd if r.cost_usd is not None else 0.0)
        if order_by == "time":
            return (r.wall_clock_s is None, r.wall_clock_s if r.wall_clock_s is not None else 0.0)
        if order_by == "finished_at":
            return (r.completed_at is None, r.completed_at or "")
        return 0

    rows.sort(key=_sort_key)
    return rows[:limit]


router = APIRouter()


@router.get("/leaderboard", response_model=list[LeaderboardRow])
def list_leaderboard_runs(
    paper: str | None = Query(default=None),
    mode: Literal["rlm", "rdr"] | None = Query(default=None),
    order_by: OrderBy = Query(default="score"),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Return ranked leaderboard rows aggregated from on-disk runs.

    Read-only; not gated by REPROLAB_DEMO_SECRET.
    """
    settings = get_settings()
    runs_root = Path(settings.runs_dir) if getattr(settings, "runs_dir", None) else Path("runs")
    try:
        return aggregate_leaderboard(
            runs_root, paper=paper, mode=mode, order_by=order_by, limit=limit
        )
    except Exception as e:
        logger.exception("leaderboard: aggregation failed")
        raise HTTPException(status_code=500, detail=str(e))


__all__ = [
    "LeaderboardRow",
    "RoleModels",
    "aggregate_leaderboard",
    "router",
]
```

If `backend.config.get_settings()` does not expose `runs_dir` (verify with `grep -n "runs_dir" /Volumes/CS_Stuff/openresearch/backend/config.py`), fall back to the literal `Path("runs")` and add a TODO comment requesting the settings field for a future cleanup task.

- [ ] **Step 6: Run the test to confirm it passes**

Run: `.venv/bin/python -m pytest tests/routes/test_leaderboard_aggregator.py -v`
Expected: 8 passed.

- [ ] **Step 7: Commit**

```bash
git add backend/routes/__init__.py backend/routes/leaderboard.py tests/routes/__init__.py tests/routes/test_leaderboard_aggregator.py
git commit -m "feat(leaderboard): add aggregator that ranks runs from disk

Scans runs/<id>/final_report.json + demo_status.json and returns a
ranked list of LeaderboardRow objects. Read-only; no SQLite projection
in this delivery (forward-compatible subset of cleanup-spec Phase 4).

Filters: paper, mode. Sort orders: score (default), cost, time, finished_at.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.4"
```

### Task 2.2: Mount the router on the FastAPI app + verify demo gate is bypassed

**Files:**
- Modify: `backend/app.py` (the `create_app` factory)
- Test: `tests/routes/test_leaderboard_http.py` (new)

- [ ] **Step 1: Write the failing HTTP test**

Create `tests/routes/test_leaderboard_http.py`:

```python
"""HTTP-level tests for the leaderboard endpoint.

Spec: 2026-05-23-rubric-climb-leaderboard §4.4, §3 #10.
"""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _seed_run(runs_root: Path, project_id: str, score: float, paper_id="p1") -> None:
    d = runs_root / project_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "final_report.json").write_text(json.dumps({
        "paper": {"id": paper_id, "title": paper_id.upper()},
        "verdict": "partial",
        "reproduction_summary": "x",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": {"overall_score": score, "meets_target": False, "areas": []},
        "improvements": [],
        "primitive_trace": {},
        "cost": {"llm_usd": 1.0, "primitives": 0.0},
        "iterations": 3,
        "mode": "rlm",
        "models": {"planner": "gpt-5", "executor": "claude-sonnet-4-6",
                   "verifier": None, "grader": None},
        "started_at": "2026-05-23T04:10:00+00:00",
        "completed_at": "2026-05-23T04:15:00+00:00",
    }))
    (d / "demo_status.json").write_text("{}")


@pytest.fixture
def app_with_runs(tmp_path, monkeypatch):
    """Boot the FastAPI app with REPROLAB_RUNS_DIR pointed at a tmp dir."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _seed_run(runs_root, "prj_a", 0.42)
    _seed_run(runs_root, "prj_b", 0.71)

    monkeypatch.setenv("REPROLAB_RUNS_DIR", str(runs_root))
    monkeypatch.delenv("REPROLAB_DEMO_SECRET", raising=False)

    # Reset cached settings.
    from backend.config import get_settings
    get_settings.cache_clear()  # type: ignore[attr-defined]

    from backend.app import create_app
    return create_app()


def test_get_leaderboard_returns_ranked_rows(app_with_runs):
    client = TestClient(app_with_runs)
    r = client.get("/leaderboard")
    assert r.status_code == 200
    rows = r.json()
    assert [row["project_id"] for row in rows] == ["prj_b", "prj_a"]


def test_get_leaderboard_filters_by_paper(app_with_runs, tmp_path):
    client = TestClient(app_with_runs)
    r = client.get("/leaderboard", params={"paper": "p1"})
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_get_leaderboard_ignores_demo_secret_gate(app_with_runs, monkeypatch):
    monkeypatch.setenv("REPROLAB_DEMO_SECRET", "shh")
    from backend.config import get_settings
    get_settings.cache_clear()  # type: ignore[attr-defined]

    from backend.app import create_app
    app = create_app()
    client = TestClient(app)
    # No X-Demo-Secret header — must still return 200.
    r = client.get("/leaderboard")
    assert r.status_code == 200
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `.venv/bin/python -m pytest tests/routes/test_leaderboard_http.py -v`
Expected: 404 or import errors — the router is not mounted yet, and `REPROLAB_RUNS_DIR` may not be a known setting.

- [ ] **Step 3: Add `runs_dir` field to `Settings` (verified missing)**

The 2026-05-23 advisor-checkpoint verification confirmed `Settings` (at `backend/config.py:12`) has no `runs_dir` field. Add one. Open `backend/config.py`, find `class Settings(BaseSettings):`, and add a field matching the neighbouring style:

```python
    runs_dir: str = Field(
        default="runs",
        description="Filesystem root for run directories.",
    )
```

Pydantic BaseSettings reads `REPROLAB_RUNS_DIR` from env automatically when the project's env_prefix is `REPROLAB_` (confirm by inspecting neighbouring fields like `demo_secret` → `REPROLAB_DEMO_SECRET`).

- [ ] **Step 4: Mount the router in `backend/app.py`**

Open `backend/app.py`. Inside `create_app()`, after the existing router registrations, add:

```python
    from backend.routes.leaderboard import router as leaderboard_router
    app.include_router(leaderboard_router)
```

**Verified 2026-05-23:** the demo gate is per-route (each `start_*` endpoint calls `_enforce_demo_gate(x_demo_secret, settings.demo_secret)` explicitly — see `backend/app.py:108`, `:125`, `:143`, `:205`). There is no middleware to allow-list — `GET /leaderboard` is unprotected by default because the gate function is never invoked for it.

- [ ] **Step 5: Run the test to confirm it passes**

Run: `.venv/bin/python -m pytest tests/routes/test_leaderboard_http.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run the full backend suite to confirm no regression**

Run: `.venv/bin/python -m pytest tests/ -n auto --tb=no -q 2>&1 | tail -5`
Expected: pass count ≥ baseline + 11 (8 from Task 2.1 + 3 from Task 2.2).

- [ ] **Step 7: Commit**

```bash
git add backend/app.py backend/config.py tests/routes/test_leaderboard_http.py
git commit -m "feat(api): mount GET /leaderboard, bypass demo-secret gate

Wires the aggregator into the FastAPI app and confirms the route is
unprotected (it is a read-only listing; the gate exists for mutating
endpoints only).

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §3 #10, §4.4"
```

---

## Phase 3 — Frontend: Reducer extensions

### Task 3.1: Add `previousAreas` to rubric state

**Files:**
- Modify: `frontend/src/hooks/use-rlm-run.ts`
- Test: `frontend/src/hooks/use-rlm-run.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `frontend/src/hooks/use-rlm-run.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { fold, INITIAL_RLM_STATE } from "./use-rlm-run";
import type { RubricScoreEvent } from "../lib/events/rlm-events";

function rubricEvent(iteration: number, score: number, areas: Array<{area: string; status: "pass" | "partial" | "fail"; score: number; weight: number}>): RubricScoreEvent {
  return {
    event: "rubric_score",
    timestamp: new Date(2026, 4, 23, 4, 10 + iteration).toISOString(),
    iteration,
    score,
    target: 0.8,
    areas,
  };
}

describe("rubric.previousAreas (spec §4.2)", () => {
  it("is empty on initial state", () => {
    expect(INITIAL_RLM_STATE.rubric.previousAreas).toEqual([]);
  });

  it("captures the prior areas snapshot when a new rubric_score arrives", () => {
    const e1 = rubricEvent(1, 0.30, [
      { area: "Setup", status: "fail", score: 0.1, weight: 1 },
      { area: "Train", status: "fail", score: 0.05, weight: 1 },
    ]);
    const e2 = rubricEvent(3, 0.55, [
      { area: "Setup", status: "pass", score: 0.85, weight: 1 },
      { area: "Train", status: "partial", score: 0.45, weight: 1 },
    ]);

    let s = fold(INITIAL_RLM_STATE, e1);
    // After e1, previousAreas is still empty (no prior snapshot).
    expect(s.rubric.previousAreas).toEqual([]);
    expect(s.rubric.areas.length).toBe(2);

    s = fold(s, e2);
    // After e2, previousAreas is the e1 areas snapshot.
    expect(s.rubric.previousAreas.map(a => a.area)).toEqual(["Setup", "Train"]);
    expect(s.rubric.previousAreas[0].status).toBe("fail");
    expect(s.rubric.areas[0].status).toBe("pass");
  });
});
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `cd frontend && npm test -- use-rlm-run`
Expected: type error — `previousAreas` not in `RubricState`.

- [ ] **Step 3: Update `RlmRunState.rubric` type + INITIAL_RLM_STATE + foldRubricScore**

In `frontend/src/hooks/use-rlm-run.ts`:

1. In the `RlmRunState.rubric` field (around line 106), add after `areas: RubricArea[];`:
```ts
    previousAreas: RubricArea[]; // §4.2 — snapshot of areas BEFORE the latest rubric_score
    attributableCandidate: {     // §4.4 — live candidate pointer
      id: string;
      title: string;
      outcome: TreeNode["outcome"] | null;
    } | null;
```

2. In `INITIAL_RLM_STATE.rubric` (around line 154), add:
```ts
    previousAreas: [],
    attributableCandidate: null,
```

3. In `foldRubricScore` (around line 359), update the return to snapshot `state.rubric.areas` into `previousAreas` BEFORE overwriting. The new return value's `rubric` object becomes:

```ts
    rubric: {
      baseline: isFirst ? ev.score : state.rubric.baseline,
      current: ev.score,
      target: ev.target,
      series: [...state.rubric.series, { iteration: ev.iteration, score: ev.score }],
      previousAreas: state.rubric.areas.map((a) => ({ ...a })),
      areas: ev.areas.map((a) => ({ ...a })),
      attributableCandidate: state.rubric.attributableCandidate,
    },
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `cd frontend && npm test -- use-rlm-run`
Expected: pass.

- [ ] **Step 5: Run the full vitest suite to verify no other tests broke**

Run: `cd frontend && npm test 2>&1 | tail -10`
Expected: no new failures (the existing rubric tests need to handle the new field — most likely they don't access it, so they pass).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/hooks/use-rlm-run.ts frontend/src/hooks/use-rlm-run.test.ts
git commit -m "feat(reducer): snapshot rubric areas before overwrite

Adds rubric.previousAreas — the areas array as it existed before the
most recent rubric_score event. Lets the strip render per-area status
flips (fail/partial -> pass) without component-level diffing.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §3 #3, §4.2"
```

### Task 3.2: Wire `attributableCandidate` into candidate folds

**Files:**
- Modify: `frontend/src/hooks/use-rlm-run.ts` (foldCandidateProposed, foldCandidateOutcome)
- Test: `frontend/src/hooks/use-rlm-run.test.ts` (append)

- [ ] **Step 1: Write the failing test**

Append to `frontend/src/hooks/use-rlm-run.test.ts`:

```ts
describe("rubric.attributableCandidate (spec §4.2)", () => {
  it("starts as null", () => {
    expect(INITIAL_RLM_STATE.rubric.attributableCandidate).toBeNull();
  });

  it("is set when candidate_proposed lands", () => {
    const ev = {
      event: "candidate_proposed" as const,
      timestamp: "2026-05-23T04:10:00.000Z",
      iteration: 5,
      round: 1,
      candidate: { id: "c1", title: "Bigger batch size", category: "training", description: "...", reasoning: "..." },
    };
    const s = fold(INITIAL_RLM_STATE, ev);
    expect(s.rubric.attributableCandidate).toEqual({
      id: "c1",
      title: "Bigger batch size",
      outcome: null,
    });
  });

  it("updates outcome when matching candidate_outcome lands", () => {
    let s = fold(INITIAL_RLM_STATE, {
      event: "candidate_proposed",
      timestamp: "2026-05-23T04:10:00.000Z",
      iteration: 5,
      round: 1,
      candidate: { id: "c1", title: "Bigger batch", category: "training", description: "", reasoning: "" },
    });
    s = fold(s, {
      event: "candidate_outcome",
      timestamp: "2026-05-23T04:11:00.000Z",
      iteration: 6,
      candidate_id: "c1",
      outcome: "promoted",
      rubric_delta: 0.12,
    });
    expect(s.rubric.attributableCandidate).toEqual({
      id: "c1",
      title: "Bigger batch",
      outcome: "promoted",
    });
  });

  it("overwrites the pointer when a newer candidate is proposed", () => {
    let s = fold(INITIAL_RLM_STATE, {
      event: "candidate_proposed",
      timestamp: "2026-05-23T04:10:00.000Z",
      iteration: 5,
      round: 1,
      candidate: { id: "c1", title: "First", category: "x", description: "", reasoning: "" },
    });
    s = fold(s, {
      event: "candidate_proposed",
      timestamp: "2026-05-23T04:11:00.000Z",
      iteration: 6,
      round: 1,
      candidate: { id: "c2", title: "Second", category: "x", description: "", reasoning: "" },
    });
    expect(s.rubric.attributableCandidate?.id).toBe("c2");
    expect(s.rubric.attributableCandidate?.title).toBe("Second");
  });

  it("reflects buffered outcome when candidate_outcome arrives before candidate_proposed", () => {
    // §4.2 + §5.3 (existing _pendingOutcomes path): outcome arrives first.
    let s = fold(INITIAL_RLM_STATE, {
      event: "candidate_outcome",
      timestamp: "2026-05-23T04:10:00.000Z",
      iteration: 5,
      candidate_id: "c1",
      outcome: "promoted",
      rubric_delta: 0.15,
    });
    // Before the proposal arrives, attributableCandidate is null (the buffered
    // outcome lives only in _pendingOutcomes).
    expect(s.rubric.attributableCandidate).toBeNull();

    s = fold(s, {
      event: "candidate_proposed",
      timestamp: "2026-05-23T04:11:00.000Z",
      iteration: 6,
      round: 1,
      candidate: { id: "c1", title: "Bigger batch", category: "x", description: "", reasoning: "" },
    });
    // Once the proposal lands, the pointer is set with the buffered outcome.
    expect(s.rubric.attributableCandidate).toEqual({
      id: "c1",
      title: "Bigger batch",
      outcome: "promoted",
    });
  });
});
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `cd frontend && npm test -- use-rlm-run`
Expected: 3 failures — `attributableCandidate` is `null` after candidate_proposed.

- [ ] **Step 3: Update `foldCandidateProposed` and `foldCandidateOutcome`**

In `frontend/src/hooks/use-rlm-run.ts`:

1. In `foldCandidateProposed` (around line 453), at the end of the function, modify the return so `state.rubric.attributableCandidate` is updated:

```ts
  return {
    ...state,
    tree,
    _pendingOutcomes,
    rubric: {
      ...state.rubric,
      attributableCandidate: {
        id: ev.candidate.id,
        title: ev.candidate.title,
        outcome: pending?.outcome ?? null,
      },
    },
  };
```

2. In `foldCandidateOutcome` (around line 490), if the candidate matches the current `attributableCandidate.id`, update its outcome. Modify the return:

```ts
  // Update the live candidate pointer's outcome if it matches.
  const rubric = state.rubric.attributableCandidate?.id === ev.candidate_id
    ? {
        ...state.rubric,
        attributableCandidate: {
          ...state.rubric.attributableCandidate,
          outcome: ev.outcome,
        },
      }
    : state.rubric;

  return { ...state, tree, rubric };
```

(The out-of-order buffering branch — the `if (nodeIdx === -1)` early-return — also needs the rubric update so the pointer's outcome is set even when the proposal hasn't landed. For simplicity, when the proposal lands later it will overwrite the pointer with `outcome: pending?.outcome ?? null` — which reads the buffered outcome. Good enough; the pointer is approximate and downstream consumers should treat `outcome: null` as "running.")

- [ ] **Step 4: Run the test to confirm it passes**

Run: `cd frontend && npm test -- use-rlm-run`
Expected: pass.

- [ ] **Step 5: Run the full vitest suite**

Run: `cd frontend && npm test 2>&1 | tail -10`
Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/hooks/use-rlm-run.ts frontend/src/hooks/use-rlm-run.test.ts
git commit -m "feat(reducer): track attributableCandidate live pointer

Maintains rubric.attributableCandidate as a 'most recent candidate the
root was working on' pointer. Updated by candidate_proposed (with
buffered outcome merged) and candidate_outcome (when ids match). Lets
the rubric strip surface 'jump from candidate X' attribution.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.2"
```

---

## Phase 4 — Frontend: Utilities + sub-components

### Task 4.1: `useCountUp` tween hook

**Files:**
- Create: `frontend/src/components/lab/rlm/use-count-up.ts`
- Test: `frontend/src/components/lab/rlm/use-count-up.test.ts` (new)

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/lab/rlm/use-count-up.test.ts`:

```ts
import { act, renderHook } from "@testing-library/react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { useCountUp } from "./use-count-up";

describe("useCountUp", () => {
  let rafCallbacks: Map<number, FrameRequestCallback>;
  let rafId: number;
  let nowMs: number;

  beforeEach(() => {
    rafCallbacks = new Map();
    rafId = 0;
    nowMs = 0;
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
      rafId += 1;
      rafCallbacks.set(rafId, cb);
      return rafId;
    });
    vi.stubGlobal("cancelAnimationFrame", (id: number) => {
      rafCallbacks.delete(id);
    });
    vi.stubGlobal("performance", { now: () => nowMs });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function tick(deltaMs: number) {
    nowMs += deltaMs;
    const cbs = Array.from(rafCallbacks.values());
    rafCallbacks.clear();
    cbs.forEach((cb) => cb(nowMs));
  }

  it("returns target immediately when target is null-equivalent (target=0 from 0)", () => {
    const { result } = renderHook(() => useCountUp(0, 400));
    expect(result.current).toBe(0);
  });

  it("eases from start to target over the duration", () => {
    const { result, rerender } = renderHook(({ target }) => useCountUp(target, 400), {
      initialProps: { target: 0 },
    });
    rerender({ target: 1.0 });

    // After ~half the duration, we should be > 0 but < 1.
    act(() => tick(200));
    expect(result.current).toBeGreaterThan(0);
    expect(result.current).toBeLessThan(1);

    // After full duration, we should land at 1.
    act(() => tick(400));
    expect(result.current).toBeCloseTo(1.0, 5);
  });

  it("cancels and restarts on target change mid-tween", () => {
    const { result, rerender } = renderHook(({ target }) => useCountUp(target, 400), {
      initialProps: { target: 0 },
    });

    rerender({ target: 1.0 });
    act(() => tick(100));
    const midValue = result.current;
    expect(midValue).toBeGreaterThan(0);

    // Retarget to 0.5 — restart from midValue.
    rerender({ target: 0.5 });
    act(() => tick(400));
    expect(result.current).toBeCloseTo(0.5, 5);
  });
});
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `cd frontend && npm test -- use-count-up`
Expected: cannot find module.

- [ ] **Step 3: Implement the hook**

Create `frontend/src/components/lab/rlm/use-count-up.ts`:

```ts
"use client";

import { useEffect, useRef, useState } from "react";

/**
 * useCountUp — tween a numeric value to a target with cubic-out easing.
 *
 * Returns the currently-displayed value. The tween restarts whenever
 * `target` changes; cancelled if the component unmounts.
 *
 * Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.3.
 */
export function useCountUp(target: number, durationMs = 400): number {
  const [value, setValue] = useState(target);
  const fromRef = useRef(target);
  const startRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    fromRef.current = value;
    // Initialize the start clock in the effect so the FIRST RAF tick sees a
    // non-zero elapsed window. (If we left startRef = null and assigned inside
    // step, the first frame would compute elapsed=0 and never advance.)
    startRef.current = performance.now();

    const step = (now: number) => {
      const elapsed = now - (startRef.current ?? now);
      const t = Math.min(1, elapsed / durationMs);
      const eased = 1 - Math.pow(1 - t, 3); // cubic-out
      const next = fromRef.current + (target - fromRef.current) * eased;
      setValue(next);

      if (t < 1) {
        rafRef.current = requestAnimationFrame(step);
      } else {
        rafRef.current = null;
      }
    };

    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
    }
    rafRef.current = requestAnimationFrame(step);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
    // We intentionally exclude `value` — we use it as the starting point but
    // do not re-trigger on every value tick (that would be an infinite loop).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, durationMs]);

  return value;
}
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `cd frontend && npm test -- use-count-up`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/lab/rlm/use-count-up.ts frontend/src/components/lab/rlm/use-count-up.test.ts
git commit -m "feat(lab): add useCountUp tween hook

Cubic-out easing over 400 ms by default. Cancels + restarts on target
change. Pure RAF; no animation library. Used by the rubric strip to
animate the big score between rubric_score events.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.3"
```

### Task 4.2: `Sparkline` SVG line chart

**Files:**
- Create: `frontend/src/components/lab/rlm/sparkline.tsx`
- Create: `frontend/src/components/lab/rlm/sparkline.module.css`
- Test: `frontend/src/components/lab/rlm/sparkline.test.tsx` (new)

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/lab/rlm/sparkline.test.tsx`:

```tsx
import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { Sparkline } from "./sparkline";

describe("Sparkline", () => {
  it("renders nothing when there are no points", () => {
    const { container } = render(<Sparkline series={[]} />);
    expect(container.querySelector("svg")).toBeNull();
  });

  it("renders a single circle when there is one point", () => {
    const { container } = render(<Sparkline series={[{ iteration: 1, score: 0.5 }]} />);
    expect(container.querySelector("svg")).not.toBeNull();
    expect(container.querySelectorAll("circle").length).toBe(1);
  });

  it("renders a polyline with N points for N>=2 points", () => {
    const series = [
      { iteration: 1, score: 0.1 },
      { iteration: 2, score: 0.4 },
      { iteration: 3, score: 0.7 },
    ];
    const { container } = render(<Sparkline series={series} />);
    const polyline = container.querySelector("polyline");
    expect(polyline).not.toBeNull();
    const points = polyline!.getAttribute("points")!.trim().split(/\s+/);
    expect(points.length).toBe(3);
  });

  it("clamps y-axis to [0,1]", () => {
    const series = [
      { iteration: 1, score: -0.2 },
      { iteration: 2, score: 1.5 },
    ];
    const { container } = render(<Sparkline series={series} width={100} height={20} />);
    const polyline = container.querySelector("polyline")!;
    const points = polyline.getAttribute("points")!.trim().split(/\s+/);
    // Each point is "x,y" — y is the height value.
    // Clamped: -0.2 -> 0 (top? or bottom?). Score 1 maps to top (y=0). Score 0 maps to bottom (y=height).
    // So -0.2 clamped to 0 -> y=20; 1.5 clamped to 1 -> y=0.
    const [, y1] = points[0].split(",").map(Number);
    const [, y2] = points[1].split(",").map(Number);
    expect(y1).toBeCloseTo(20, 1);
    expect(y2).toBeCloseTo(0, 1);
  });
});
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `cd frontend && npm test -- sparkline`
Expected: cannot find module.

- [ ] **Step 3: Implement the component**

Create `frontend/src/components/lab/rlm/sparkline.module.css`:

```css
.svg {
  display: block;
  overflow: visible;
}

.line {
  fill: none;
  stroke: var(--accent, currentColor);
  stroke-width: 1.5;
  stroke-linejoin: round;
  stroke-linecap: round;
}

.dot {
  fill: var(--accent, currentColor);
}
```

Create `frontend/src/components/lab/rlm/sparkline.tsx`:

```tsx
"use client";

import styles from "./sparkline.module.css";

interface SparklineProps {
  series: Array<{ iteration: number; score: number }>;
  width?: number;
  height?: number;
}

/**
 * Sparkline — SVG line chart for the rubric climb.
 *
 * Y axis is [0, 1] (rubric scores); X axis is series index (0..N-1).
 * Empty series renders nothing. One point renders a small circle.
 *
 * Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.1.
 */
export function Sparkline({ series, width = 120, height = 28 }: SparklineProps) {
  if (series.length === 0) return null;

  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  const yFor = (score: number) => height - clamp(score) * height;

  if (series.length === 1) {
    const cy = yFor(series[0].score);
    return (
      <svg
        className={styles.svg}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        aria-hidden="true"
      >
        <circle className={styles.dot} cx={width / 2} cy={cy} r={2} />
      </svg>
    );
  }

  const xStep = width / (series.length - 1);
  const points = series
    .map((p, i) => `${(i * xStep).toFixed(2)},${yFor(p.score).toFixed(2)}`)
    .join(" ");

  return (
    <svg
      className={styles.svg}
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      aria-hidden="true"
    >
      <polyline className={styles.line} points={points} />
    </svg>
  );
}
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `cd frontend && npm test -- sparkline`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/lab/rlm/sparkline.tsx frontend/src/components/lab/rlm/sparkline.module.css frontend/src/components/lab/rlm/sparkline.test.tsx
git commit -m "feat(lab): add SVG line-chart Sparkline component

Replaces the previous bar sparkline. Y-axis clamped to [0,1]; empty
series renders nothing; one point renders a dot. No chart library.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.1"
```

---

## Phase 5 — Frontend: RubricStrip enrichment

### Task 5.1: Wire `useCountUp` + Sparkline into the strip

**Files:**
- Modify: `frontend/src/components/lab/rlm/rubric-strip.tsx`
- Modify: `frontend/src/components/lab/rlm/rubric-strip.module.css`
- Modify: `frontend/src/components/lab/rlm/rubric-strip.test.tsx`

- [ ] **Step 1: Read the existing tests so we know what to preserve**

Run: `cat /Volumes/CS_Stuff/openresearch/frontend/src/components/lab/rlm/rubric-strip.test.tsx | head -120`
Note the existing assertions — they continue to apply.

- [ ] **Step 2: Write the failing tests for new behavior**

Append to `frontend/src/components/lab/rlm/rubric-strip.test.tsx`:

```tsx
describe("RubricStrip — count-up tween (spec §4.3)", () => {
  it("renders the current score formatted to 2dp when isScored", () => {
    const { container } = render(
      <RubricStrip rubric={{
        current: 0.71, baseline: 0.22, target: 0.8,
        series: [{ iteration: 1, score: 0.22 }, { iteration: 5, score: 0.71 }],
        areas: [], previousAreas: [], attributableCandidate: null,
      }} />
    );
    // The big score uses count-up which starts at the target value (no prior).
    expect(container.querySelector(`.${styles.scoreValue}`)?.textContent).toMatch(/0\.[67]\d/);
  });
});

describe("RubricStrip — sparkline (spec §4.1)", () => {
  it("renders an SVG polyline when there are >= 2 points", () => {
    const { container } = render(
      <RubricStrip rubric={{
        current: 0.55, baseline: 0.22, target: 0.8,
        series: [
          { iteration: 1, score: 0.22 },
          { iteration: 3, score: 0.35 },
          { iteration: 5, score: 0.55 },
        ],
        areas: [], previousAreas: [], attributableCandidate: null,
      }} />
    );
    expect(container.querySelector("polyline")).not.toBeNull();
  });
});
```

You will need to import `styles` from the module CSS — `import styles from "./rubric-strip.module.css";`.

- [ ] **Step 3: Run the test to confirm it fails**

Run: `cd frontend && npm test -- rubric-strip`
Expected: 2 failures (no polyline; score format wrong).

- [ ] **Step 4: Update `rubric-strip.tsx`**

In `frontend/src/components/lab/rlm/rubric-strip.tsx`:

1. Replace the bar-sparkline block with the `<Sparkline />` component.
2. Wrap the displayed `current` score with `useCountUp`.

Updated component body:

```tsx
"use client";

import type { RlmRunState } from "../../../hooks/use-rlm-run";
import { useCountUp } from "./use-count-up";
import { Sparkline } from "./sparkline";
import styles from "./rubric-strip.module.css";

interface RubricStripProps {
  rubric: RlmRunState["rubric"];
}

export function RubricStrip({ rubric }: RubricStripProps) {
  const { current, baseline, target, series } = rubric;

  const isScored = current !== null;
  const tweenedCurrent = useCountUp(isScored ? current! : 0, 400);

  const fmtScore = (n: number) => n.toFixed(2);
  const fillPct = isScored ? `${Math.min(1, Math.max(0, tweenedCurrent)) * 100}%` : "0%";

  const hasClimb = isScored && baseline !== null;
  const delta = hasClimb ? current! - baseline! : null;

  return (
    <div className={styles.strip} role="region" aria-label="Rubric score">
      <div className={styles.scoreBlock}>
        <span className={isScored ? styles.scoreValue : styles.scorePlaceholder}>
          {isScored ? fmtScore(tweenedCurrent) : "—"}
        </span>
        <span className={styles.scoreLabel}>rubric score</span>
      </div>

      <div className={styles.barBlock}>
        <div className={styles.barTrack} aria-hidden="true">
          <div className={styles.barFill} style={{ width: fillPct }} />
          {baseline !== null && (
            <div
              className={styles.baselineMarker}
              style={{ left: `${baseline * 100}%` }}
              title={`baseline ${fmtScore(baseline)}`}
            />
          )}
          {target !== null && (
            <div
              className={styles.targetMarker}
              style={{ left: `${target * 100}%` }}
              title={`target ${fmtScore(target)}`}
            />
          )}
        </div>
      </div>

      {series.length > 0 && (
        <div className={styles.sparkBlock} aria-hidden="true">
          <Sparkline series={series} width={120} height={28} />
        </div>
      )}

      {hasClimb && (
        <div className={styles.climbBlock}>
          <span className={styles.climbAnnotation}>
            {`baseline ${fmtScore(baseline!)} → ${fmtScore(current!)}`}
          </span>
          {delta !== null && target !== null && (
            <span className={styles.deltaAnnotation}>
              {`${delta >= 0 ? "+" : ""}${fmtScore(delta)} vs target ${fmtScore(target)}`}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Run the test to confirm it passes**

Run: `cd frontend && npm test -- rubric-strip`
Expected: pass (both new + existing tests).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/lab/rlm/rubric-strip.tsx frontend/src/components/lab/rlm/rubric-strip.test.tsx
git commit -m "feat(lab): wire useCountUp + Sparkline into RubricStrip

The big rubric number now animates between rubric_score events via the
cubic-out count-up tween. The previous bar sparkline is replaced by the
new SVG line-chart Sparkline component.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.1, §4.3"
```

### Task 5.2: Per-area chip row with flip detection + tint

**Files:**
- Modify: `frontend/src/components/lab/rlm/rubric-strip.tsx`
- Modify: `frontend/src/components/lab/rlm/rubric-strip.module.css`
- Modify: `frontend/src/components/lab/rlm/rubric-strip.test.tsx`

- [ ] **Step 1: Write the failing tests**

Append to `frontend/src/components/lab/rlm/rubric-strip.test.tsx`:

```tsx
describe("RubricStrip — per-area flip tint (spec §4.1, §4.2)", () => {
  it("renders a chip per area with status glyph", () => {
    const { container } = render(
      <RubricStrip rubric={{
        current: 0.55, baseline: 0.22, target: 0.8,
        series: [{ iteration: 1, score: 0.22 }, { iteration: 5, score: 0.55 }],
        areas: [
          { area: "Setup", score: 0.9, weight: 1, status: "pass" },
          { area: "Train", score: 0.45, weight: 1, status: "partial" },
        ],
        previousAreas: [
          { area: "Setup", score: 0.1, weight: 1, status: "fail" },
          { area: "Train", score: 0.05, weight: 1, status: "fail" },
        ],
        attributableCandidate: null,
      }} />
    );
    const chips = container.querySelectorAll('[data-area-chip]');
    expect(chips.length).toBe(2);
  });

  it("applies justFlipped to areas that transitioned to a higher status", () => {
    const { container } = render(
      <RubricStrip rubric={{
        current: 0.55, baseline: 0.22, target: 0.8,
        series: [{ iteration: 1, score: 0.22 }, { iteration: 5, score: 0.55 }],
        areas: [
          { area: "Setup", score: 0.9, weight: 1, status: "pass" },
          { area: "Train", score: 0.1, weight: 1, status: "fail" },
        ],
        previousAreas: [
          { area: "Setup", score: 0.1, weight: 1, status: "fail" },
          { area: "Train", score: 0.1, weight: 1, status: "fail" },
        ],
        attributableCandidate: null,
      }} />
    );
    const flipped = container.querySelectorAll('[data-area-chip][data-just-flipped="true"]');
    expect(flipped.length).toBe(1);
    expect(flipped[0].getAttribute('data-area')).toBe("Setup");
  });

  it("does not apply justFlipped on the first rubric_score (no previousAreas)", () => {
    const { container } = render(
      <RubricStrip rubric={{
        current: 0.22, baseline: 0.22, target: 0.8,
        series: [{ iteration: 1, score: 0.22 }],
        areas: [
          { area: "Setup", score: 0.9, weight: 1, status: "pass" },
        ],
        previousAreas: [],
        attributableCandidate: null,
      }} />
    );
    const flipped = container.querySelectorAll('[data-just-flipped="true"]');
    expect(flipped.length).toBe(0);
  });
});
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `cd frontend && npm test -- rubric-strip`
Expected: failures — no chips, no `data-just-flipped` attr.

- [ ] **Step 3: Add the chip row + flip detection to `rubric-strip.tsx`**

Add this helper at the top of the file (after imports):

```tsx
const STATUS_RANK: Record<"pass" | "partial" | "fail", number> = {
  fail: 0,
  partial: 1,
  pass: 2,
};

function justFlipped(
  area: { area: string; status: "pass" | "partial" | "fail" },
  previous: Array<{ area: string; status: "pass" | "partial" | "fail" }>,
): boolean {
  const prior = previous.find((p) => p.area === area.area);
  if (!prior) return false;
  return STATUS_RANK[area.status] > STATUS_RANK[prior.status];
}

function statusGlyph(status: "pass" | "partial" | "fail"): string {
  return status === "pass" ? "✓" : status === "partial" ? "◐" : "✗";
}
```

Inside the component body, after the sparkline `<div>` and before the climb annotation, insert the chip row:

```tsx
      {rubric.areas.length > 0 && (
        <ul className={styles.areaChipRow} aria-live="polite" aria-label="Rubric areas">
          {rubric.areas.map((a) => {
            const flipped = justFlipped(a, rubric.previousAreas);
            const cls = [
              styles.areaChip,
              a.status === "pass" ? styles.chipPass : a.status === "partial" ? styles.chipPartial : styles.chipFail,
              flipped ? styles.justFlipped : "",
            ].join(" ").trim();
            return (
              <li
                key={a.area}
                className={cls}
                data-area-chip
                data-area={a.area}
                data-just-flipped={flipped ? "true" : "false"}
                title={`${a.area}: ${a.status} (${a.score.toFixed(2)})`}
              >
                <span className={styles.chipGlyph} aria-hidden="true">{statusGlyph(a.status)}</span>
                <span className={styles.chipName}>{a.area}</span>
              </li>
            );
          })}
        </ul>
      )}
```

- [ ] **Step 4: Add CSS for the chip row + flip tint**

Append to `frontend/src/components/lab/rlm/rubric-strip.module.css`:

```css
.areaChipRow {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  list-style: none;
  padding: 0;
  margin: 6px 0 0;
}

.areaChip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  font-size: 11px;
  line-height: 1.6;
  border: 1px solid var(--border, rgba(0, 0, 0, 0.08));
  border-radius: 999px;
  background: transparent;
  transition: background 200ms ease-out;
}

.chipPass { color: var(--success-fg, #1f7a3a); }
.chipPartial { color: var(--warning-fg, #8a6d00); }
.chipFail { color: var(--muted-fg, #6e6e6e); }

.chipGlyph {
  font-size: 11px;
  width: 10px;
  display: inline-block;
  text-align: center;
}

.chipName {
  white-space: nowrap;
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
}

@keyframes flipTint {
  0%   { background: var(--success-bg, rgba(31, 122, 58, 0.18)); }
  100% { background: transparent; }
}

.justFlipped {
  animation: flipTint 1200ms ease-out 1 forwards;
}
```

- [ ] **Step 5: Run the test to confirm it passes**

Run: `cd frontend && npm test -- rubric-strip`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/lab/rlm/rubric-strip.tsx frontend/src/components/lab/rlm/rubric-strip.module.css frontend/src/components/lab/rlm/rubric-strip.test.tsx
git commit -m "feat(lab): add per-area chip row with flip tint to RubricStrip

Per-area chips show pass/partial/fail status with a glyph + name. Areas
whose status rank strictly increased since the last rubric_score event
get a 1.2 s background tint animation that fades to transparent.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.1"
```

### Task 5.3: Candidate attribution tail

**Files:**
- Modify: `frontend/src/components/lab/rlm/rubric-strip.tsx`
- Modify: `frontend/src/components/lab/rlm/rubric-strip.test.tsx`

- [ ] **Step 1: Write the failing test**

Append to `frontend/src/components/lab/rlm/rubric-strip.test.tsx`:

```tsx
describe("RubricStrip — candidate attribution (spec §4.1)", () => {
  it("renders attribution when last delta >= 0.05 and a candidate is live", () => {
    const { getByText } = render(
      <RubricStrip rubric={{
        current: 0.55, baseline: 0.22, target: 0.8,
        series: [
          { iteration: 1, score: 0.22 },
          { iteration: 5, score: 0.40 },
          { iteration: 7, score: 0.55 },  // delta = +0.15
        ],
        areas: [],
        previousAreas: [],
        attributableCandidate: { id: "c1", title: "Bigger batch", outcome: "promoted" },
      }} />
    );
    expect(getByText(/from candidate Bigger batch/i)).toBeTruthy();
  });

  it("omits attribution when last delta < 0.05", () => {
    const { queryByText } = render(
      <RubricStrip rubric={{
        current: 0.55, baseline: 0.22, target: 0.8,
        series: [
          { iteration: 5, score: 0.53 },
          { iteration: 7, score: 0.55 },  // delta = +0.02
        ],
        areas: [],
        previousAreas: [],
        attributableCandidate: { id: "c1", title: "Bigger batch", outcome: "promoted" },
      }} />
    );
    expect(queryByText(/from candidate/i)).toBeNull();
  });

  it("omits attribution when no candidate is live", () => {
    const { queryByText } = render(
      <RubricStrip rubric={{
        current: 0.55, baseline: 0.22, target: 0.8,
        series: [
          { iteration: 1, score: 0.22 },
          { iteration: 5, score: 0.55 },
        ],
        areas: [],
        previousAreas: [],
        attributableCandidate: null,
      }} />
    );
    expect(queryByText(/from candidate/i)).toBeNull();
  });
});
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `cd frontend && npm test -- rubric-strip`
Expected: 3 failures.

- [ ] **Step 3: Add the attribution tail to `rubric-strip.tsx`**

Inside the `{hasClimb && (...)}` block, after `delta vs target`, add a new conditional:

```tsx
      {hasClimb && (
        <div className={styles.climbBlock}>
          {/* existing baseline annotation and delta annotation stay as-is */}
          {(() => {
            // Attribution: derive last-step delta from series.
            if (series.length < 2 || !rubric.attributableCandidate) return null;
            const last = series[series.length - 1].score;
            const prev = series[series.length - 2].score;
            const stepDelta = last - prev;
            if (stepDelta < 0.05) return null;
            return (
              <span className={styles.attributionAnnotation}>
                {`from candidate ${rubric.attributableCandidate.title}`}
              </span>
            );
          })()}
        </div>
      )}
```

Add CSS for `.attributionAnnotation` in `rubric-strip.module.css`:

```css
.attributionAnnotation {
  font-size: 11px;
  color: var(--muted-fg, #6e6e6e);
  font-style: italic;
  margin-left: 8px;
}
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `cd frontend && npm test -- rubric-strip`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/lab/rlm/rubric-strip.tsx frontend/src/components/lab/rlm/rubric-strip.module.css frontend/src/components/lab/rlm/rubric-strip.test.tsx
git commit -m "feat(lab): add candidate attribution tail to RubricStrip

When the last step delta is >= +0.05 and a candidate is live (via
rubric.attributableCandidate), the climb annotation appends 'from
candidate <title>'. No attribution before the second rubric_score event.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.1"
```

### Task 5.4: Extend the lab fixture with a climb scenario

**Files:**
- Modify: `frontend/src/components/lab/rlm/__fixtures__/rlm-run.fixture.ts`
- Test: `frontend/src/components/lab/rlm/__fixtures__/rlm-run.fixture.test.ts`

- [ ] **Step 1: Inspect what scenario the existing fixture already covers**

Run: `wc -l /Volumes/CS_Stuff/openresearch/frontend/src/components/lab/rlm/__fixtures__/rlm-run.fixture.ts`
Read the existing fixture — confirm it covers the rubric climb already (baseline ~0.22 → ~0.53). It does, per the header comment.

- [ ] **Step 2: Append a verification test that the fixture exercises flips + attribution**

Append to `frontend/src/components/lab/rlm/__fixtures__/rlm-run.fixture.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { rlmRunFixture } from "./rlm-run.fixture";
import { fold, INITIAL_RLM_STATE } from "../../../../hooks/use-rlm-run";

describe("rlmRunFixture — climb scenario coverage (spec §6.3)", () => {
  it("produces at least one fail->pass area flip across the run", () => {
    let state = INITIAL_RLM_STATE;
    let sawFlip = false;
    for (const ev of rlmRunFixture) {
      const before = state.rubric.areas.map((a) => ({ area: a.area, status: a.status }));
      state = fold(state, ev);
      if (ev.event === "rubric_score") {
        const after = state.rubric.areas;
        for (const a of after) {
          const prior = before.find((p) => p.area === a.area);
          if (prior && prior.status === "fail" && a.status === "pass") {
            sawFlip = true;
            break;
          }
        }
      }
    }
    expect(sawFlip).toBe(true);
  });

  it("produces at least one attributable jump of >= 0.05 with a live candidate", () => {
    let state = INITIAL_RLM_STATE;
    let prevScore: number | null = null;
    let sawAttributable = false;
    for (const ev of rlmRunFixture) {
      state = fold(state, ev);
      if (ev.event === "rubric_score") {
        if (prevScore !== null && ev.score - prevScore >= 0.05 && state.rubric.attributableCandidate) {
          sawAttributable = true;
          break;
        }
        prevScore = ev.score;
      }
    }
    expect(sawAttributable).toBe(true);
  });
});
```

- [ ] **Step 3: Run the test**

Run: `cd frontend && npm test -- rlm-run.fixture`
Expected: pass if the existing fixture already covers flips + attribution; if it FAILS, edit the fixture to add a rubric_score event between candidate_proposed and candidate_outcome that shows a meaningful delta, AND to include at least one area that transitions fail→pass.

(Note: this is verification + an opportunistic fixture fix, not new behavior.)

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/lab/rlm/__fixtures__/rlm-run.fixture.test.ts frontend/src/components/lab/rlm/__fixtures__/rlm-run.fixture.ts
git commit -m "test(lab): assert fixture exercises rubric flips + attribution

Adds two assertions on the existing climb fixture so future fixture
edits cannot silently strip the scenarios the rubric strip relies on."
```

---

## Phase 6 — Frontend: Leaderboard page

### Task 6.1: Next.js proxy route

**Files:**
- Create: `frontend/src/app/api/demo/leaderboard/route.ts`
- Test: `frontend/src/app/api/demo/leaderboard/route.test.ts` (new)

- [ ] **Step 1: Read the existing route pattern**

Run: `cat /Volumes/CS_Stuff/openresearch/frontend/src/app/api/demo/events/route.ts 2>/dev/null || cat /Volumes/CS_Stuff/openresearch/frontend/src/app/api/demo/route.ts 2>/dev/null` — pattern adopted by neighboring proxies.

- [ ] **Step 2: Write the failing test**

Create `frontend/src/app/api/demo/leaderboard/route.test.ts` (mirrors the convention from `frontend/src/app/api/demo/route.test.ts` — `vi.stubEnv` + dynamic import):

```ts
// @vitest-environment node

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));

describe("/api/demo/leaderboard proxy", () => {
  beforeEach(() => {
    vi.stubEnv("REPROLAB_BACKEND_URL", "http://backend.test");
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("forwards query params to the backend and returns its JSON", async () => {
    const captured: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      captured.push(url);
      return Response.json([{ project_id: "x", overall_score: 0.5 }], { status: 200 });
    }));
    const { GET } = await import("./route");

    const req = new Request("http://localhost/api/demo/leaderboard?paper=p1&order_by=cost");
    const res = await GET(req);
    expect(res.status).toBe(200);
    expect(captured[0]).toBe("http://backend.test/leaderboard?paper=p1&order_by=cost");
    const body = await res.json();
    expect(body[0].project_id).toBe("x");
  });

  it("propagates 500 status on backend error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("boom", { status: 500 })));
    const { GET } = await import("./route");
    const req = new Request("http://localhost/api/demo/leaderboard");
    const res = await GET(req);
    expect(res.status).toBe(500);
  });

  it("returns 502 when fetch throws", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("ECONNREFUSED"); }));
    const { GET } = await import("./route");
    const req = new Request("http://localhost/api/demo/leaderboard");
    const res = await GET(req);
    expect(res.status).toBe(502);
  });
});
```

- [ ] **Step 3: Run the test to confirm it fails**

Run: `cd frontend && npm test -- api/demo/leaderboard`
Expected: cannot find module.

- [ ] **Step 4: Implement the proxy**

Create `frontend/src/app/api/demo/leaderboard/route.ts`:

```ts
import "server-only";

/**
 * GET /api/demo/leaderboard — proxy to the backend's GET /leaderboard.
 *
 * Read-only; not gated by the demo secret.
 *
 * Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.4.
 */
export async function GET(req: Request): Promise<Response> {
  // Sibling convention (see frontend/src/app/api/demo/events/route.ts:27):
  // default to localhost backend when env is unset.
  const backendUrl = (process.env.REPROLAB_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
  const incoming = new URL(req.url);
  const search = incoming.search; // includes leading ? when non-empty
  const target = `${backendUrl}/leaderboard${search}`;

  let resp: Response;
  try {
    resp = await fetch(target, {
      method: "GET",
      headers: { accept: "application/json" },
      cache: "no-store",
    });
  } catch (e) {
    return new Response(
      JSON.stringify({ detail: `Backend unreachable: ${String(e)}` }),
      { status: 502, headers: { "content-type": "application/json" } },
    );
  }

  const body = await resp.text();
  return new Response(body, {
    status: resp.status,
    headers: { "content-type": resp.headers.get("content-type") ?? "application/json" },
  });
}
```

- [ ] **Step 5: Run the test to confirm it passes**

Run: `cd frontend && npm test -- api/demo/leaderboard`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/app/api/demo/leaderboard/route.ts frontend/src/app/api/demo/leaderboard/route.test.ts
git commit -m "feat(api-proxy): add /api/demo/leaderboard server-side proxy

Forwards GET /api/demo/leaderboard (browser-facing) to GET /leaderboard
on the FastAPI backend, preserving query params. No CORS layer; the
browser only talks to the Next.js server. Returns 502 on backend
unreachable.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.4"
```

### Task 6.2: `LeaderboardTable` client component

**Files:**
- Create: `frontend/src/app/leaderboard/leaderboard-table.tsx`
- Create: `frontend/src/app/leaderboard/leaderboard-table.module.css`
- Test: `frontend/src/app/leaderboard/leaderboard-table.test.tsx` (new)

- [ ] **Step 1: Write the failing test**

Create `frontend/src/app/leaderboard/leaderboard-table.test.tsx`:

```tsx
import { render, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { LeaderboardTable, type LeaderboardRow } from "./leaderboard-table";

function row(overrides: Partial<LeaderboardRow> = {}): LeaderboardRow {
  return {
    project_id: "prj_x",
    paper_id: "p1",
    paper_title: "Paper One",
    mode: "rlm",
    models: { planner: "gpt-5", executor: "claude-sonnet-4-6", verifier: null, grader: null },
    overall_score: 0.5,
    meets_target: false,
    degraded: false,
    cost_usd: 1.23,
    iterations: 8,
    wall_clock_s: 300,
    sandbox: "docker",
    started_at: "2026-05-23T04:10:00+00:00",
    completed_at: "2026-05-23T04:15:00+00:00",
    verdict: "partial",
    ...overrides,
  };
}

describe("LeaderboardTable", () => {
  it("renders the empty state when rows is empty", () => {
    const { getByText } = render(<LeaderboardTable rows={[]} />);
    expect(getByText(/no completed runs yet/i)).toBeTruthy();
  });

  it("renders one row per leaderboard entry", () => {
    const { container } = render(
      <LeaderboardTable rows={[
        row({ project_id: "a", overall_score: 0.7 }),
        row({ project_id: "b", overall_score: 0.4 }),
      ]} />
    );
    expect(container.querySelectorAll("tbody tr").length).toBe(2);
  });

  it("sorts by overall_score desc by default", () => {
    const { container } = render(
      <LeaderboardTable rows={[
        row({ project_id: "a", overall_score: 0.4 }),
        row({ project_id: "b", overall_score: 0.7 }),
      ]} />
    );
    const firstRowId = container.querySelectorAll("tbody tr")[0].getAttribute("data-project-id");
    expect(firstRowId).toBe("b");
  });

  it("re-sorts when a sortable column header is clicked", () => {
    const { container, getByText } = render(
      <LeaderboardTable rows={[
        row({ project_id: "a", cost_usd: 3.0 }),
        row({ project_id: "b", cost_usd: 1.0 }),
      ]} />
    );
    fireEvent.click(getByText(/^Cost$/));
    const firstRowId = container.querySelectorAll("tbody tr")[0].getAttribute("data-project-id");
    expect(firstRowId).toBe("b");  // cost asc
  });

  it("renders an em dash for null cost / wall_clock / model", () => {
    const { container } = render(
      <LeaderboardTable rows={[row({
        cost_usd: null, wall_clock_s: null,
        models: { planner: null, executor: null, verifier: null, grader: null },
      })]} />
    );
    const dashes = container.querySelectorAll('[data-dash="true"]');
    expect(dashes.length).toBeGreaterThanOrEqual(3);
  });
});
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `cd frontend && npm test -- leaderboard-table`
Expected: cannot find module.

- [ ] **Step 3: Implement the component**

Create `frontend/src/app/leaderboard/leaderboard-table.module.css`:

```css
.table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.table th,
.table td {
  border-bottom: 1px solid var(--border, rgba(0, 0, 0, 0.08));
  padding: 8px 10px;
  text-align: left;
  vertical-align: middle;
}

.sortable {
  cursor: pointer;
  user-select: none;
}

.sortable:hover {
  background: var(--hover-bg, rgba(0, 0, 0, 0.03));
}

.numeric {
  font-variant-numeric: tabular-nums;
  text-align: right;
}

.scoreCellDegraded { color: var(--muted-fg, #6e6e6e); }
.scoreCellMet { color: var(--success-fg, #1f7a3a); font-weight: 600; }

.modeBadge {
  display: inline-block;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 4px;
  border: 1px solid var(--border, rgba(0, 0, 0, 0.12));
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.emptyState {
  padding: 32px;
  text-align: center;
  color: var(--muted-fg, #6e6e6e);
}

.dash {
  color: var(--muted-fg, #b0b0b0);
}

.paperLink {
  color: inherit;
  text-decoration: underline;
  text-decoration-thickness: 1px;
  text-underline-offset: 2px;
}

.paperLink:hover { color: var(--accent, #2563eb); }
```

Create `frontend/src/app/leaderboard/leaderboard-table.tsx`:

```tsx
"use client";

import { useMemo, useState } from "react";
import styles from "./leaderboard-table.module.css";

export interface LeaderboardRow {
  project_id: string;
  paper_id: string;
  paper_title: string | null;
  mode: "rlm" | "rdr";
  models: {
    planner: string | null;
    executor: string | null;
    verifier: string | null;
    grader: string | null;
  };
  overall_score: number;
  meets_target: boolean;
  degraded: boolean;
  cost_usd: number | null;
  iterations: number;
  wall_clock_s: number | null;
  sandbox: string | null;
  started_at: string | null;
  completed_at: string | null;
  verdict: string;
}

type SortKey =
  | "score"
  | "paper"
  | "mode"
  | "planner"
  | "executor"
  | "cost"
  | "iterations"
  | "time"
  | "verdict"
  | "completed";

type SortDir = "asc" | "desc";

const DEFAULT_DIR: Record<SortKey, SortDir> = {
  score: "desc",
  paper: "asc",
  mode: "asc",
  planner: "asc",
  executor: "asc",
  cost: "asc",
  iterations: "desc",
  time: "asc",
  verdict: "asc",
  completed: "desc",
};

function fmtElapsed(s: number | null): string | null {
  if (s === null) return null;
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}m ${sec}s`;
}

function Dash() {
  return <span className={styles.dash} data-dash="true">—</span>;
}

interface LeaderboardTableProps {
  rows: LeaderboardRow[];
}

export function LeaderboardTable({ rows }: LeaderboardTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const mul = sortDir === "asc" ? 1 : -1;
      const ak = keyFor(a, sortKey);
      const bk = keyFor(b, sortKey);
      if (ak === null && bk === null) return 0;
      if (ak === null) return 1;
      if (bk === null) return -1;
      if (ak < bk) return -1 * mul;
      if (ak > bk) return 1 * mul;
      return 0;
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  if (rows.length === 0) {
    return (
      <div className={styles.emptyState}>
        No completed runs yet — start one from the lab.
      </div>
    );
  }

  const handleHeaderClick = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(DEFAULT_DIR[key]);
    }
  };

  const Header = ({ label, k }: { label: string; k: SortKey }) => (
    <th className={styles.sortable} onClick={() => handleHeaderClick(k)}>
      {label}
      {sortKey === k && <span aria-hidden="true">{sortDir === "asc" ? " ▲" : " ▼"}</span>}
    </th>
  );

  return (
    <table className={styles.table}>
      <thead>
        <tr>
          <Header label="Paper" k="paper" />
          <Header label="Mode" k="mode" />
          <Header label="Planner" k="planner" />
          <Header label="Executor" k="executor" />
          <Header label="Score" k="score" />
          <Header label="Cost" k="cost" />
          <Header label="Iterations" k="iterations" />
          <Header label="Time" k="time" />
          <Header label="Verdict" k="verdict" />
          <Header label="Finished" k="completed" />
        </tr>
      </thead>
      <tbody>
        {sorted.map((r) => {
          const scoreCls = r.meets_target
            ? styles.scoreCellMet
            : r.degraded
            ? styles.scoreCellDegraded
            : "";
          const elapsed = fmtElapsed(r.wall_clock_s);
          return (
            <tr key={r.project_id} data-project-id={r.project_id}>
              <td>
                <a className={styles.paperLink} href={`/lab?projectId=${encodeURIComponent(r.project_id)}`}>
                  {r.paper_title ?? r.paper_id}
                </a>
              </td>
              <td><span className={styles.modeBadge}>{r.mode}</span></td>
              <td>{r.models.planner ?? <Dash />}</td>
              <td>{r.models.executor ?? <Dash />}</td>
              <td className={`${styles.numeric} ${scoreCls}`}>{r.overall_score.toFixed(2)}</td>
              <td className={styles.numeric}>{r.cost_usd !== null ? `$${r.cost_usd.toFixed(2)}` : <Dash />}</td>
              <td className={styles.numeric}>{r.iterations}</td>
              <td className={styles.numeric}>{elapsed ?? <Dash />}</td>
              <td>{r.verdict}</td>
              <td>{r.completed_at ?? <Dash />}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function keyFor(r: LeaderboardRow, k: SortKey): string | number | null {
  switch (k) {
    case "score": return r.overall_score;
    case "paper": return r.paper_title ?? r.paper_id;
    case "mode": return r.mode;
    case "planner": return r.models.planner;
    case "executor": return r.models.executor;
    case "cost": return r.cost_usd;
    case "iterations": return r.iterations;
    case "time": return r.wall_clock_s;
    case "verdict": return r.verdict;
    case "completed": return r.completed_at;
  }
}
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `cd frontend && npm test -- leaderboard-table`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/leaderboard/leaderboard-table.tsx frontend/src/app/leaderboard/leaderboard-table.module.css frontend/src/app/leaderboard/leaderboard-table.test.tsx
git commit -m "feat(leaderboard): add LeaderboardTable client component

Sortable table — click any header to sort that column. Default sort is
overall_score desc. Empty state renders a placeholder. Null fields
render an em dash. Row click navigates to /lab?projectId=<id>.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.6"
```

### Task 6.3: `/leaderboard` page shell

**Files:**
- Create: `frontend/src/app/leaderboard/page.tsx`

- [ ] **Step 1: Look at the existing `/lab` page for the pattern**

Run: `cat /Volumes/CS_Stuff/openresearch/frontend/src/app/lab/page.tsx 2>/dev/null | head -60`

- [ ] **Step 2: Implement the page**

Create `frontend/src/app/leaderboard/page.tsx`:

```tsx
import "server-only";
import { LeaderboardTable, type LeaderboardRow } from "./leaderboard-table";

export const dynamic = "force-dynamic";

async function fetchRows(): Promise<LeaderboardRow[]> {
  const backendUrl = process.env.REPROLAB_BACKEND_URL;
  if (!backendUrl) return [];
  try {
    const resp = await fetch(`${backendUrl.replace(/\/$/, "")}/leaderboard`, {
      cache: "no-store",
      headers: { accept: "application/json" },
    });
    if (!resp.ok) return [];
    return (await resp.json()) as LeaderboardRow[];
  } catch {
    return [];
  }
}

export default async function LeaderboardPage() {
  const rows = await fetchRows();
  return (
    <main style={{ padding: "24px 32px", maxWidth: 1280, margin: "0 auto" }}>
      <header style={{ marginBottom: 18 }}>
        <h1 style={{ fontSize: 20, fontWeight: 600, margin: 0 }}>Leaderboard</h1>
        <p style={{ color: "#6e6e6e", fontSize: 13, margin: "4px 0 0" }}>
          Ranked completed runs across models and papers.
        </p>
      </header>
      <LeaderboardTable rows={rows} />
    </main>
  );
}
```

- [ ] **Step 3: Verify it builds + tsc + lint**

Run: `cd frontend && npx tsc --noEmit && npm run lint 2>&1 | tail -5`
Expected: clean.

- [ ] **Step 4: Smoke check via dev server**

Skip if the dev server isn't reachable in this session. Otherwise:

Run (background): `cd frontend && npm run dev -- --port 3000`
Then navigate to `http://localhost:3000/leaderboard` and confirm the page renders.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/leaderboard/page.tsx
git commit -m "feat(leaderboard): add /leaderboard server-component page

Fetches rows from the backend at request time (no caching), renders the
LeaderboardTable. Server-side fetch ensures the browser never talks to
the backend directly.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.6"
```

---

## Phase 7 — E2E + verification + docs

### Task 7.1: Playwright in-session E2E

**Files:**
- Create: `frontend/e2e/rubric-climb.spec.ts`
- Create: `frontend/e2e/leaderboard.spec.ts`

- [ ] **Step 1: Confirm the existing e2e harness boots**

Run: `ls /Volumes/CS_Stuff/openresearch/frontend/e2e/`
If no `playwright.config.ts` is visible in `frontend/`, run `cat /Volumes/CS_Stuff/openresearch/frontend/playwright.config.ts 2>/dev/null` to verify it exists.

- [ ] **Step 2: Write the leaderboard E2E**

Create `frontend/e2e/leaderboard.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test.describe("/leaderboard", () => {
  test("renders empty state when no runs", async ({ page }) => {
    await page.goto("/leaderboard");
    await expect(page.getByText(/no completed runs yet/i)).toBeVisible();
    expect(await page.evaluate(() => (window as any).__consoleErrors ?? [])).toEqual([]);
  });

  // The populated state requires seeded fixtures via the backend. Documented
  // as docker-dependent in the plan §Phase 7 / spec §3 #12.
});
```

- [ ] **Step 3: Write the rubric-climb E2E (fixture-driven)**

The lab page supports `?fixture=1` URL — verify with `grep -n "fixture" /Volumes/CS_Stuff/openresearch/frontend/src/app/lab/page.tsx`. If it does, write the test against that path. If not, the fixture wiring is added in Task 7.2.

Create `frontend/e2e/rubric-climb.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test.describe("Rubric climb panel", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      (window as any).__consoleErrors = [];
      window.addEventListener("error", (e) => (window as any).__consoleErrors.push(String(e.error || e.message)));
    });
  });

  test("empty state renders no big score and no console errors", async ({ page }) => {
    await page.goto("/lab");
    // The placeholder dash is rendered when there is no rubric score.
    await expect(page.locator("text=/^—$/").first()).toBeVisible();
    expect(await page.evaluate(() => (window as any).__consoleErrors)).toEqual([]);
  });

  test("climb scenario renders sparkline + per-area chips + attribution", async ({ page }) => {
    await page.goto("/lab?fixture=1");
    // Sparkline polyline appears once enough rubric events have folded.
    await expect(page.locator("polyline").first()).toBeVisible({ timeout: 4000 });
    // At least one area chip is present.
    await expect(page.locator("[data-area-chip]").first()).toBeVisible();
    // The climb annotation contains the baseline→current pair.
    await expect(page.locator("text=/baseline 0\\.\\d+/")).toBeVisible();
  });
});
```

- [ ] **Step 4: Run the e2e (in-session, headless)**

Run: `cd frontend && npx playwright test e2e/leaderboard.spec.ts e2e/rubric-climb.spec.ts --reporter=line 2>&1 | tail -30`

If failing because the dev server isn't running, start it in the background first:

Run (background): `cd frontend && PORT=3000 REPROLAB_BACKEND_URL=http://127.0.0.1:8000 npm run dev`

Then re-run the playwright command.

Expected (in-session bucket — spec §3 #12):
- `empty state renders no big score`: PASS
- `climb scenario renders sparkline + per-area chips + attribution`: PASS
- `leaderboard renders empty state`: PASS

(Docker-dependent populated-state tests are documented as next-session work; do not author them in this task.)

- [ ] **Step 5: Commit**

```bash
git add frontend/e2e/rubric-climb.spec.ts frontend/e2e/leaderboard.spec.ts
git commit -m "test(e2e): playwright specs for rubric climb panel + leaderboard

In-session bucket per the plan: empty state + fixture-driven climb +
empty leaderboard. Docker-dependent populated-state tests are deferred.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §6.4"
```

### Task 7.2: Lab page fixture wiring (only if Task 7.1 needed it)

Skip if Task 7.1 found that `?fixture=1` already drives the lab. Otherwise:

**Files:**
- Modify: `frontend/src/app/lab/page.tsx`

- [ ] **Step 1: Read current lab page**

Run: `cat /Volumes/CS_Stuff/openresearch/frontend/src/app/lab/page.tsx | head -80`

- [ ] **Step 2: Add the fixture branch**

In the lab page, after the existing event-source setup, branch on `searchParams.fixture === "1"`:

```tsx
import { rlmRunFixture } from "../../components/lab/rlm/__fixtures__/rlm-run.fixture";

// ...

if (searchParams?.fixture === "1") {
  events = rlmRunFixture;
}
```

(Exact placement depends on the lab page's structure — match the pattern.)

- [ ] **Step 3: Re-run the e2e from Task 7.1**

Run: `cd frontend && npx playwright test e2e/rubric-climb.spec.ts --reporter=line 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 4: Commit (only if Step 2 was needed)**

```bash
git add frontend/src/app/lab/page.tsx
git commit -m "feat(lab): wire ?fixture=1 query to drive the lab from the climb fixture

Lets e2e tests exercise the rubric climb panel without a backend run."
```

### Task 7.3: Run full suites and confirm green

- [ ] **Step 1: Backend full pytest**

Run: `.venv/bin/python -m pytest tests/ -n auto --tb=no -q 2>&1 | tail -5`
Expected: pass count ≥ baseline + new tests; 0 failed; 1 xfailed (existing T24).

- [ ] **Step 2: Frontend lint + tsc + vitest**

Run: `cd frontend && npm run lint && npx tsc --noEmit && npm test 2>&1 | tail -10`
Expected: lint clean, tsc clean, all vitest tests pass.

- [ ] **Step 3: Capture the green snapshot**

Run: `echo "post-plan: pytest=PASS_COUNT, vitest=PASS_COUNT, lint+tsc=green" > .plan-post.txt`
(Working artifact; do not commit.)

No commit for this step.

### Task 7.4: Documentation updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `system_overview.md`

- [ ] **Step 1: Update `CLAUDE.md` SSE event types and new routes**

Find the "Run lifecycle (UI ↔ backend)" or "Where to look first" sections; add to the relevant list:
- "Leaderboard route: `frontend/src/app/leaderboard/`, backend `backend/routes/leaderboard.py`."
- Confirm under "SSE event types" that no new types were added; mention that the rubric climb panel derives all new state (per-area flips + candidate attribution) from existing `rubric_score`, `candidate_proposed`, `candidate_outcome` events.
- Add a one-line note in the `final_report.json` section that runs now record `mode`, `models`, `started_at`, `completed_at`.

- [ ] **Step 2: Update `system_overview.md`**

Add a new "Leaderboard endpoint" subsection in the lab/API overview:

```markdown
## Leaderboard endpoint

`GET /leaderboard` returns ranked rows aggregated at request time from
`runs/*/final_report.json` + `demo_status.json`. Read-only; not gated by
`REPROLAB_DEMO_SECRET`.

Query params:

| param | type | default | meaning |
|---|---|---|---|
| `paper` | string | — | filter by `paper.id` |
| `mode` | `rlm` \| `rdr` | — | filter by mode |
| `order_by` | `score` \| `cost` \| `time` \| `finished_at` | `score` | server-side sort |
| `limit` | int | 50 | row cap (1–500) |

Row shape (Pydantic):
- `project_id, paper_id, paper_title, mode`
- `models: {planner, executor, verifier, grader}` — per-role model ids
- `overall_score, meets_target, degraded`
- `cost_usd, iterations, wall_clock_s`
- `sandbox, started_at, completed_at, verdict`

`final_report.json` now carries `mode`, `models`, `started_at`,
`completed_at` for forward compatibility with the per-role picker
roadmap. Legacy reports without these fields parse fine — defaults are
filled.
```

- [ ] **Step 3: Confirm routing references**

Add to the "Where to look" list:
```
- **Leaderboard** — `frontend/src/app/leaderboard/` (page + table), `backend/routes/leaderboard.py` (read-only aggregation). Filesystem-backed; no SQLite projection at this scale.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md system_overview.md
git commit -m "docs: leaderboard endpoint + rubric-climb final_report fields

Updates CLAUDE.md and system_overview.md to
reflect the new /leaderboard surface and the four new optional fields
on final_report.json. Notes the rubric climb panel derives flip + 
attribution state purely from existing SSE events.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §5"
```

### Task 7.5: Diff review via subagent

- [ ] **Step 1: Run a code-reviewer subagent on the branch diff**

Dispatch the `feature-dev:code-reviewer` agent with this prompt:

```
Review the current branch diff against main for correctness, security,
and adherence to project conventions. Focus on:
- Backend: pydantic schemas + back-compat handling in
  backend/routes/leaderboard.py and the RLMFinalReport extension.
- Frontend: useRlmRun reducer changes — confirm previousAreas snapshot
  semantics and attributableCandidate pointer update rules match the
  spec at docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md
  §4.2.
- No SSE event widening; sanitize_iteration is unchanged.
- Demo-secret gate is not silently bypassed for other routes (only
  /leaderboard).
- RubricStrip backward-compat: existing tests still pass and the
  exported prop type is unchanged.

Report only high-confidence findings. Confirm or contradict the
'forward-compatible with cleanup spec Phase 4' claim.
```

- [ ] **Step 2: Address every high-confidence finding**

For each finding from the reviewer, either fix it inline (with a commit `fix(review): <subject>`) or document why the finding is not load-bearing in a one-line comment on the spec. No silent dismissals.

### Task 7.6: Final advisor checkpoint + summary

- [ ] **Step 1: Verify deliverables are durable**

Run: `git log main..HEAD --oneline | head -30`
Confirm each task's commit is present. Run: `git status` — must be clean.

- [ ] **Step 2: Call `advisor()`**

Pass the full transcript. The advisor sees the plan, the diff, the test outputs.

- [ ] **Step 3: Write the final report**

In a message to the user, list every Playwright criterion (1–8 from the original task spec) with PASS / FAIL / DEFERRED-TO-NEXT-SESSION evidence (test name or commit SHA). State honestly which docker-dependent criteria remain.

- [ ] **Step 4: Mark the task complete**

Per the user's spec, only when every criterion is PASS or every DEFERRED is documented with the next concrete step. Don't claim done if anything is silently SKIP.

---

## Spec coverage check

- Spec §3 #1 (band 2 enrichment in place) → Task 5.x (all RubricStrip work modifies the existing file).
- Spec §3 #2 (no new SSE event types) → Task 0.1 baseline, Task 7.4 doc note, Task 7.5 review.
- Spec §3 #3 (`previousAreas` in reducer) → Task 3.1.
- Spec §3 #4 (`attributableCandidate` pointer) → Task 3.2.
- Spec §3 #5 (SVG line chart) → Task 4.2.
- Spec §3 #6 (CSS + RAF, no animation lib) → Tasks 4.1 + 5.2 (CSS keyframes).
- Spec §3 #7 (`models` shape on final_report) → Tasks 1.1 + 1.2.
- Spec §3 #8 (filesystem aggregation, no SQLite) → Task 2.1.
- Spec §3 #9 (`/leaderboard` endpoint + proxy) → Tasks 2.2 + 6.1 + 6.3.
- Spec §3 #10 (demo gate bypass) → Task 2.2.
- Spec §3 #11 (replay works as-is) → Task 7.1 (fixture-driven E2E exercises the same render path as a replayed run).
- Spec §3 #12 (in-session vs deferred Playwright) → Task 7.1.
- Spec §6 test strategy → Tasks 1.1, 2.1, 3.1, 3.2, 4.1, 4.2, 5.1–5.4, 6.2.
- Spec §9 acceptance criteria → Task 7.3 + 7.4 + 7.5 + 7.6.

No placeholder steps. No "similar to Task N" references — code is repeated where needed.
