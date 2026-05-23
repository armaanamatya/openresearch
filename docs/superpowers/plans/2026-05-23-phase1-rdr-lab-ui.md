# Phase 1 — rdr Lab-UI Gap-Close — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the lab-UI gap so a `--mode rdr` run renders end-to-end in `/lab`. Backend gets 4 spec-shaped cluster SSE events (additive to the 12 existing `rdr_*` events). Frontend reducer captures cluster state, the exploration canvas grows a dual-layout (rlm dynamic / rdr rubric-cluster), `NodeDetailPopup` gains a cluster branch, and `?rdrFixture=1` joins the existing `?rlmFixture=1` mechanism. Closes cleanup-spec §5.2.3 acceptance modulo Region 5 cluster timeline (deferred to Phase 1b).

**Architecture:** Pure additive — no existing rdr or rlm code is renamed or moved. New event builders live alongside existing ones in `backend/agents/rlm/sse_bridge.py` (kept name despite serving both modes — see spec §5.2.3). The rdr controller emits *both* the existing `rdr_*` events (kept for backward compat with existing tests + the 3 REST endpoints from PR #79) and the 4 new spec-shaped events at the same lifecycle points. The frontend reducer (`useRlmRun`) adds a `mode` discriminator on state plus `clusterStatus` / `clusterScores` peer fields under `RlmRunState`; existing folds are unchanged. ExplorationCanvas reads `mode` from `runMeta` (now extended) and routes through a layout-function abstraction. NodeDetailPopup branches on a new `kind === "cluster"` TreeNode variant.

**Tech Stack:** Python 3.14 / FastAPI / Pydantic v2 (backend); Next.js 16 / React 19 / TypeScript / Vitest / Playwright (frontend).

**Reference spec:** `docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md` §5.2.3. Read §3 (locked decisions) and §5.2.3 verbatim before starting — do not relitigate decisions in those sections.

**Audit baseline (2026-05-23, see `docs/superpowers/specs/2026-05-23-cleanup-implementation-kickoff.md` §G):** PR #79 merged the rdr backend to `origin/main` 2026-05-23 07:31 UTC. Backend code under `backend/agents/rdr/`; 3 introspection REST endpoints at `backend/app.py:508-543` (`/runs/{id}/clusters`, `/runs/{id}/repair-iterations`, `/runs/{id}/leaf-scores`) — no frontend proxy yet. Controller emits 12 private `rdr_*` events via `_emit` (full inventory at the end of this doc). Frontend has zero rdr-aware code today.

**Conventions:**
- Pytest from repo root: `.venv/bin/python -m pytest`.
- Frontend commands from `frontend/`.
- Commits NEVER include `Co-Authored-By` per project convention (CLAUDE.md / spec §9).
- Each task ends with a commit. If pre-commit hook fails, create a NEW commit — never amend.
- TDD: failing test → minimal implementation → passing test → commit. "Confirm fail / Confirm pass" steps are non-skippable.
- Sub-agents should re-read this header before executing any task.

---

## Phase 0 — Pre-flight

### Task 0.1: Confirm baseline test suite is green

**Files:** none — verification only.

- [ ] **Step 1: Run backend baseline**

Run: `.venv/bin/python -m pytest tests/ -n auto --tb=no -q 2>&1 | tail -5`
Expected: ≥1083 passed / 0 failed / 1 xfailed (the T24 ThreadPoolExecutor xfail) / 5 skipped (optional-dep gated).

- [ ] **Step 2: Run frontend baseline**

Run from `frontend/`:
```bash
npm run lint 2>&1 | tail -3
npx tsc --noEmit 2>&1 | tail -3
npm test 2>&1 | tail -5
```
Expected: lint clean, tsc clean, vitest ≥85 passing.

- [ ] **Step 3: Confirm rdr mode runs end-to-end on a small bundle**

This is the most important pre-flight — if rdr is broken, Phase 1 cannot ship. Default model config is fine; bundle pick is the smallest PaperBench bundle to keep the smoke <5 min.

Run:
```bash
.venv/bin/python -m backend.cli reproduce ftrl --mode rdr --max-usd 0.5 --max-wall-clock 600 2>&1 | tail -30
```
Expected: process exits 0; `runs/pb_*/` contains `final_report.json` (with `mode: "rdr"`, `verdict` of `partial` or `failed`, non-empty `rubric.areas`), `demo_status.json` (status=`completed`), `dashboard_events.jsonl` (contains `rdr_run_started` + `rdr_run_completed` envelopes at minimum).

If this fails, **HALT.** Surface the failure mode to the user — Phase 1 work cannot proceed without a working rdr baseline. Do NOT attempt to "fix" rdr as part of Phase 1; that's spec §5.3.4 (Phase 2.4) territory.

- [ ] **Step 4: Capture the baseline numbers in a working note**

Run: `printf "baseline: pytest=PASS_COUNT, vitest=PASS_COUNT, rdr_smoke=OK\n" > .phase1-baseline.txt`
(Replace PASS_COUNT with actual numbers. File is a working artifact; do not commit.)

No commit for Task 0.1.

---

## Phase 1 — Backend cluster events

### Task 1.1: Define 4 cluster event builders in `sse_bridge.py`

**Files:**
- Modify: `backend/agents/rlm/sse_bridge.py` (append builder functions)
- Test: `tests/rlm/test_sse_bridge_cluster_events.py` (new)

- [ ] **Step 1: Read the existing `sse_bridge.py` to understand its builder pattern**

Run: `wc -l backend/agents/rlm/sse_bridge.py && head -40 backend/agents/rlm/sse_bridge.py`
Then read the file in full. Note the existing builder signatures (e.g. `build_repl_iteration`, `build_primitive_call`) — match their style: pure function, returns a plain dict, no side effects, no I/O.

- [ ] **Step 2: Write the failing test**

Create `tests/rlm/test_sse_bridge_cluster_events.py`:

```python
"""Tests for the 4 cluster-event builders.

Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3.
"""

import pytest

from backend.agents.rlm.sse_bridge import (
    build_cluster_started,
    build_cluster_artifact_emitted,
    build_cluster_scored,
    build_repair_dispatched,
)


def test_cluster_started_event_shape():
    event = build_cluster_started(
        cluster_id="c01",
        cluster_title="Method understanding",
        leaves=[{"id": "L1", "weight": 0.3, "requirements": "Define X"}],
        iteration=2,
    )
    assert event == {
        "event": "cluster_started",
        "cluster_id": "c01",
        "cluster_title": "Method understanding",
        "leaves": [{"id": "L1", "weight": 0.3, "requirements": "Define X"}],
        "iteration": 2,
    }


def test_cluster_started_rejects_malformed_leaf():
    with pytest.raises(KeyError):
        build_cluster_started(
            cluster_id="c01",
            cluster_title="T",
            leaves=[{"name": "L1"}],  # missing required keys
            iteration=1,
        )


def test_cluster_artifact_emitted_event_shape():
    event = build_cluster_artifact_emitted(
        cluster_id="c01",
        artifact_path="code/baseline/train.py",
        byte_size=4096,
        language="python",
    )
    assert event == {
        "event": "cluster_artifact_emitted",
        "cluster_id": "c01",
        "artifact_path": "code/baseline/train.py",
        "byte_size": 4096,
        "language": "python",
    }


def test_cluster_artifact_emitted_allows_null_language():
    event = build_cluster_artifact_emitted(
        cluster_id="c01",
        artifact_path="data/X.npy",
        byte_size=1024,
        language=None,
    )
    assert event["language"] is None


def test_cluster_scored_event_shape():
    event = build_cluster_scored(
        cluster_id="c01",
        score=0.72,
        leaf_scores={"L1": 0.8, "L2": 0.65},
        degraded=False,
    )
    assert event == {
        "event": "cluster_scored",
        "cluster_id": "c01",
        "score": 0.72,
        "leaf_scores": {"L1": 0.8, "L2": 0.65},
        "degraded": False,
    }


def test_cluster_scored_degraded_flag_passes_through():
    event = build_cluster_scored(
        cluster_id="c01",
        score=0.0,
        leaf_scores={},
        degraded=True,
    )
    assert event["degraded"] is True


def test_repair_dispatched_event_shape():
    event = build_repair_dispatched(
        cluster_id="c02",
        attempt=2,
        prior_score=0.42,
        failed_leaves=["L3", "L4"],
    )
    assert event == {
        "event": "repair_dispatched",
        "cluster_id": "c02",
        "attempt": 2,
        "prior_score": 0.42,
        "failed_leaves": ["L3", "L4"],
    }
```

- [ ] **Step 3: Run the test to confirm it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_sse_bridge_cluster_events.py -v`
Expected: ImportError — `cannot import name 'build_cluster_started' from 'backend.agents.rlm.sse_bridge'`.

- [ ] **Step 4: Implement the 4 builder functions**

Append to `backend/agents/rlm/sse_bridge.py` (after the existing builders, before any sanitizer / class definitions). Keep style consistent with neighbours.

```python
# --- Cluster event builders (spec 2026-05-23-cleanup-condensation-leaderboard §5.2.3)

_REQUIRED_LEAF_KEYS = ("id", "weight", "requirements")


def build_cluster_started(
    cluster_id: str,
    cluster_title: str,
    leaves: list[dict],
    iteration: int,
) -> dict:
    """Build a sanitized ``cluster_started`` SSE payload.

    Emitted when the rdr controller (or a future hybrid controller)
    dispatches work for a new rubric cluster. ``leaves`` carries the leaf
    descriptors that grade this cluster — id, weight, requirements text.
    """
    for leaf in leaves:
        if not isinstance(leaf, dict):
            raise TypeError(f"leaf must be a dict, got {type(leaf)!r}")
        for key in _REQUIRED_LEAF_KEYS:
            if key not in leaf:
                raise KeyError(f"leaf missing required key {key!r}")
    return {
        "event": "cluster_started",
        "cluster_id": cluster_id,
        "cluster_title": cluster_title,
        "leaves": leaves,
        "iteration": iteration,
    }


def build_cluster_artifact_emitted(
    cluster_id: str,
    artifact_path: str,
    byte_size: int,
    language: str | None,
) -> dict:
    """Build a sanitized ``cluster_artifact_emitted`` SSE payload.

    One event per file written by the cluster's reproduction agent.
    ``artifact_path`` is the relative path under the run dir;
    ``language`` is best-effort inference from file extension.
    """
    return {
        "event": "cluster_artifact_emitted",
        "cluster_id": cluster_id,
        "artifact_path": artifact_path,
        "byte_size": byte_size,
        "language": language,
    }


def build_cluster_scored(
    cluster_id: str,
    score: float,
    leaf_scores: dict[str, float],
    degraded: bool,
) -> dict:
    """Build a sanitized ``cluster_scored`` SSE payload.

    Emitted once per cluster after scoring. ``leaf_scores`` is a map of
    leaf-id → score in [0, 1]. ``degraded`` is True when the C2 honesty
    cap (leaf_scorer ``DEGRADED_LEAF_CEILING``) applied to this cluster.
    """
    return {
        "event": "cluster_scored",
        "cluster_id": cluster_id,
        "score": score,
        "leaf_scores": leaf_scores,
        "degraded": degraded,
    }


def build_repair_dispatched(
    cluster_id: str,
    attempt: int,
    prior_score: float,
    failed_leaves: list[str],
) -> dict:
    """Build a sanitized ``repair_dispatched`` SSE payload.

    Emitted at the start of a per-cluster repair attempt. ``attempt`` is
    1-indexed; ``failed_leaves`` lists the leaves that triggered the
    repair (score below the cluster's pass threshold).
    """
    return {
        "event": "repair_dispatched",
        "cluster_id": cluster_id,
        "attempt": attempt,
        "prior_score": prior_score,
        "failed_leaves": failed_leaves,
    }
```

- [ ] **Step 5: Run the test to confirm it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_sse_bridge_cluster_events.py -v`
Expected: 7 passed.

- [ ] **Step 6: Run the full sse_bridge / rlm test suite to verify no regression**

Run: `.venv/bin/python -m pytest tests/rlm/ -q --tb=short 2>&1 | tail -5`
Expected: pass count = baseline + 7.

- [ ] **Step 7: Commit**

```bash
git add backend/agents/rlm/sse_bridge.py tests/rlm/test_sse_bridge_cluster_events.py
git commit -m "feat(sse): add 4 cluster event builders for rdr → frontend bridge

Pure additive: cluster_started, cluster_artifact_emitted, cluster_scored,
repair_dispatched payload shapes per spec §5.2.3. Leaf-shape validation
on cluster_started catches malformed leaf inputs at the builder boundary.
Controller wiring lands in 1.2–1.5.

Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3"
```

### Task 1.2: Emit `cluster_started` from rdr controller

**Files:**
- Modify: `backend/agents/rdr/controller.py` (around the existing `rdr_cluster_started` emission at line 398)
- Test: `tests/rdr/test_controller_cluster_events.py` (new)

- [ ] **Step 1: Locate the controller's existing cluster-start emission**

Run: `grep -n "rdr_cluster_started" backend/agents/rdr/controller.py`
Read 30 lines of context around the match. Identify:
- The variable holding the current cluster
- The iteration index in the dispatch loop
- The leaves data structure attached to the cluster (probably from the decomposer)

- [ ] **Step 2: Write the failing test**

Create `tests/rdr/test_controller_cluster_events.py`:

```python
"""Tests that the rdr controller emits the 4 spec-shaped cluster events
alongside its existing rdr_* events.

Spec: 2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def captured_events() -> list[dict]:
    """A list the test will fill via the dashboard emit hook."""
    return []


def _make_emitter(captured: list[dict]):
    def emit(event_type: str, payload: dict) -> None:
        captured.append({"event_type": event_type, **payload})
    return emit


def test_cluster_started_is_emitted_with_spec_shape(captured_events, tmp_path: Path):
    """When the controller dispatches a cluster, a spec-shaped
    cluster_started event must land alongside the existing rdr_cluster_started.
    """
    from backend.agents.rdr.controller import RdrController
    # Build a minimal controller with a one-cluster decomposition. The
    # exact fixture depends on RdrController's __init__ — read its source
    # and provide the minimum kwargs (paper_id, bundle_dir, etc.). Stub the
    # agent dispatch to no-op so we exercise only the dispatch boundary.
    # See backend/agents/rdr/controller.py for the real signature.
    controller = _build_controller_stub(captured_events, tmp_path)
    controller.dispatch_first_cluster()

    spec_events = [e for e in captured_events if e["event_type"] == "cluster_started"]
    assert len(spec_events) == 1
    ev = spec_events[0]
    assert ev["cluster_id"]  # non-empty
    assert isinstance(ev["cluster_title"], str)
    assert isinstance(ev["leaves"], list)
    assert all({"id", "weight", "requirements"} <= set(leaf) for leaf in ev["leaves"])
    assert isinstance(ev["iteration"], int)
    # The old event is still there too — additive.
    legacy = [e for e in captured_events if e["event_type"] == "rdr_cluster_started"]
    assert len(legacy) == 1


def _build_controller_stub(captured: list[dict], tmp_path: Path):
    """Build a controller with the minimum scaffolding to test event emission.

    Read `backend/agents/rdr/controller.py::RdrController.__init__` to fill
    in the actual required arguments. The dashboard.emit hook should append
    to `captured` via _make_emitter.
    """
    raise NotImplementedError(
        "Read controller.py and fill in this stub. Use _make_emitter(captured) "
        "for the dashboard emit hook. See test_controller.py in tests/rdr/ for "
        "existing patterns to copy."
    )
```

- [ ] **Step 3: Implement `_build_controller_stub`**

Read `tests/rdr/test_controller.py` to find existing test scaffolding patterns. Copy the minimal stubbing used by the closest existing controller-emit test. The goal is a controller that can be told `dispatch_first_cluster()` (or whatever the equivalent method is — see step 1's grep) with a one-cluster decomposition pre-loaded.

If no `dispatch_first_cluster` method exists, call into the closest unit method that triggers the cluster_started emission directly. The audit found this at `backend/agents/rdr/controller.py:398` — name the method that contains that line.

- [ ] **Step 4: Add the new emission in the controller**

In `backend/agents/rdr/controller.py`, find the existing `rdr_cluster_started` emission. **Directly after** that line, add:

```python
from backend.agents.rlm.sse_bridge import build_cluster_started

# spec-shaped cluster_started — additive to rdr_cluster_started (spec §5.2.3)
self._emit(
    "cluster_started",
    build_cluster_started(
        cluster_id=cluster_id,
        cluster_title=cluster_title,
        leaves=[
            {"id": leaf.id, "weight": leaf.weight, "requirements": leaf.requirements}
            for leaf in cluster.leaves
        ],
        iteration=iteration_index,
    ),
)
```

Adjust attribute names (`leaf.id`, `leaf.weight`, `leaf.requirements`, `iteration_index`) to match whatever the controller's actual types use — these may need a `.id` → `.leaf_id` rename based on the decomposer's `Leaf` dataclass. Cross-reference `backend/agents/rdr/models.py` or `decomposer.py`.

Place the `from backend.agents.rlm.sse_bridge import build_cluster_started` at the top of the file if not already imported.

- [ ] **Step 5: Run the test to confirm it passes**

Run: `.venv/bin/python -m pytest tests/rdr/test_controller_cluster_events.py::test_cluster_started_is_emitted_with_spec_shape -v`
Expected: pass.

- [ ] **Step 6: Run the full rdr test suite to confirm no regression**

Run: `.venv/bin/python -m pytest tests/rdr/ -q --tb=short 2>&1 | tail -5`
Expected: pass count = baseline + 1 (the new test). Existing rdr tests still pass because emission is additive.

- [ ] **Step 7: Commit**

```bash
git add backend/agents/rdr/controller.py tests/rdr/test_controller_cluster_events.py
git commit -m "feat(rdr): emit spec-shaped cluster_started alongside rdr_cluster_started

Additive emission per spec §5.2.3 — existing rdr_cluster_started kept so
backend tests and the GET /runs/{id}/clusters REST endpoint continue
working. New event uses the spec's strict payload (leaves array with
id/weight/requirements, iteration index)."
```

### Task 1.3: Emit `cluster_artifact_emitted` per artifact file

**Files:**
- Modify: `backend/agents/rdr/controller.py` (the per-file artifact accumulation site)
- Modify: `backend/agents/rdr/agent.py` (if file writes happen there, the emission may belong there)
- Test: `tests/rdr/test_controller_cluster_events.py` (append)

- [ ] **Step 1: Locate where `art.files` (or equivalent) is populated**

Run: `grep -n "\.files" backend/agents/rdr/controller.py backend/agents/rdr/agent.py | head -30`
Read the surrounding lines. Identify the loop or comprehension that builds the list of artifact files for a cluster. Each iteration is the emission point.

If the file list is built atomically (e.g. as a single `.files = [...]` assignment without per-file iteration), the emission point becomes a new explicit loop that walks the populated list.

- [ ] **Step 2: Write the failing test**

Append to `tests/rdr/test_controller_cluster_events.py`:

```python
def test_cluster_artifact_emitted_one_per_file(captured_events, tmp_path: Path):
    """Each file produced by a cluster agent yields one cluster_artifact_emitted event.
    artifact_path is relative to the run dir; byte_size matches the file size on disk.
    """
    controller = _build_controller_stub(captured_events, tmp_path)
    # The stub builder seeds 3 fake artifacts; see _build_controller_stub.
    controller.dispatch_first_cluster()

    artifact_events = [e for e in captured_events if e["event_type"] == "cluster_artifact_emitted"]
    assert len(artifact_events) == 3
    for ev in artifact_events:
        assert ev["cluster_id"]
        assert ev["artifact_path"]  # non-empty relative path
        assert isinstance(ev["byte_size"], int) and ev["byte_size"] >= 0
        assert ev["language"] in {"python", "yaml", "markdown", "json", "shell", None}
```

You'll need to extend `_build_controller_stub` to seed 3 fake artifacts. Match the controller's artifact model — likely a dict with `path` + `bytes` keys, or a `RdrArtifact` dataclass.

- [ ] **Step 3: Run the test to confirm it fails**

Run: `.venv/bin/python -m pytest tests/rdr/test_controller_cluster_events.py::test_cluster_artifact_emitted_one_per_file -v`
Expected: fail — `cluster_artifact_emitted` events not present.

- [ ] **Step 4: Add language-inference helper**

In `backend/agents/rdr/controller.py` (or a new helper at the top of the file), add:

```python
_LANG_BY_SUFFIX = {
    ".py": "python",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".md": "markdown",
    ".json": "json",
    ".sh": "shell",
}


def _language_for(artifact_path: str) -> str | None:
    """Best-effort language inference from file extension.

    Returns None for unknown extensions; the frontend handles None
    by skipping syntax-aware rendering.
    """
    from pathlib import PurePosixPath
    return _LANG_BY_SUFFIX.get(PurePosixPath(artifact_path).suffix.lower())
```

- [ ] **Step 5: Emit one event per artifact**

After the artifact list is populated for a cluster, iterate and emit. Add:

```python
from backend.agents.rlm.sse_bridge import build_cluster_artifact_emitted

for artifact in art.files:  # adjust to actual artifact iterator
    self._emit(
        "cluster_artifact_emitted",
        build_cluster_artifact_emitted(
            cluster_id=cluster_id,
            artifact_path=artifact.path,         # relative to run dir
            byte_size=artifact.size_bytes,        # int >= 0
            language=_language_for(artifact.path),
        ),
    )
```

Adjust `artifact.path` / `artifact.size_bytes` to the actual attribute names. If `art.files` is a list of strings (paths only), `size_bytes` comes from `os.stat(run_dir / path).st_size`.

- [ ] **Step 6: Run the test to confirm it passes**

Run: `.venv/bin/python -m pytest tests/rdr/test_controller_cluster_events.py -v`
Expected: 2 passed (cluster_started + cluster_artifact_emitted).

- [ ] **Step 7: Commit**

```bash
git add backend/agents/rdr/controller.py tests/rdr/test_controller_cluster_events.py
git commit -m "feat(rdr): emit cluster_artifact_emitted per artifact file

One event per file produced by a cluster's reproduction agent. Language
inferred from file extension (python/yaml/markdown/json/shell or null).
artifact_path is relative to the run dir.

Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3"
```

### Task 1.4: Emit `cluster_scored` per cluster (not just run-level)

**Files:**
- Modify: `backend/agents/rdr/controller.py` (around the existing `rdr_scoring_completed` emission)
- Test: `tests/rdr/test_controller_cluster_events.py` (append)

- [ ] **Step 1: Locate the scoring loop**

Run: `grep -n "rdr_scoring_completed\|score_reproduction\|leaf_scor" backend/agents/rdr/controller.py | head -20`
Read context. Identify:
- Where per-cluster scores are computed (probably between `score_reproduction` returning the leaf_scores map and the aggregate `rdr_scoring_completed` emission)
- Where the per-cluster `degraded` flag is determined (likely tied to `leaf_scorer.DEGRADED_LEAF_CEILING`)

- [ ] **Step 2: Write the failing test**

Append:

```python
def test_cluster_scored_one_per_cluster(captured_events, tmp_path: Path):
    """One cluster_scored event per cluster after scoring completes.
    leaf_scores is a map of leaf_id → score in [0, 1]; degraded flag honest.
    """
    controller = _build_controller_stub(captured_events, tmp_path, cluster_count=2)
    controller.dispatch_all_clusters()
    controller.score_run()

    scored_events = [e for e in captured_events if e["event_type"] == "cluster_scored"]
    assert len(scored_events) == 2
    for ev in scored_events:
        assert ev["cluster_id"]
        assert 0.0 <= ev["score"] <= 1.0
        assert isinstance(ev["leaf_scores"], dict)
        assert all(0.0 <= s <= 1.0 for s in ev["leaf_scores"].values())
        assert isinstance(ev["degraded"], bool)


def test_cluster_scored_respects_degraded_cap(captured_events, tmp_path: Path):
    """If a cluster's reproduction returned no metrics, degraded=True and
    the score is capped at the C2 ceiling (≤ 0.35 per spec §5.3.1)."""
    controller = _build_controller_stub(
        captured_events, tmp_path,
        cluster_count=1, force_no_metrics=True,
    )
    controller.dispatch_all_clusters()
    controller.score_run()

    scored = [e for e in captured_events if e["event_type"] == "cluster_scored"][0]
    assert scored["degraded"] is True
    assert scored["score"] <= 0.35
```

The `force_no_metrics` kwarg makes the stub's reproduction agent return `metrics={}`, exercising the honesty cap path.

- [ ] **Step 3: Run the test to confirm it fails**

Run: `.venv/bin/python -m pytest tests/rdr/test_controller_cluster_events.py::test_cluster_scored_one_per_cluster -v`
Expected: fail.

- [ ] **Step 4: Emit per-cluster scored events**

In `backend/agents/rdr/controller.py`, in the scoring path, after each cluster's scores are computed (just before the aggregate `rdr_scoring_completed`):

```python
from backend.agents.rlm.sse_bridge import build_cluster_scored

for cluster_id, cluster_score in per_cluster_scores.items():
    leaf_scores_for_cluster = {
        leaf_id: score
        for leaf_id, score in all_leaf_scores.items()
        if leaf_id in cluster_to_leaves[cluster_id]
    }
    self._emit(
        "cluster_scored",
        build_cluster_scored(
            cluster_id=cluster_id,
            score=cluster_score,
            leaf_scores=leaf_scores_for_cluster,
            degraded=cluster_degraded.get(cluster_id, False),
        ),
    )
```

The exact variable names depend on the controller's internal types — read what `score_reproduction` returns and adapt. `cluster_to_leaves` is probably a mapping the decomposer already maintains.

If the C2 `degraded` flag isn't currently computed per-cluster, derive it: `cluster_degraded[cluster_id] = (cluster_score <= leaf_scorer.DEGRADED_LEAF_CEILING and all(metrics-empty conditions for the cluster))`. Reuse the predicate from `backend/evals/paperbench/leaf_scorer.py` rather than recomputing locally — import it.

- [ ] **Step 5: Run the tests to confirm pass**

Run: `.venv/bin/python -m pytest tests/rdr/test_controller_cluster_events.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/agents/rdr/controller.py tests/rdr/test_controller_cluster_events.py
git commit -m "feat(rdr): emit per-cluster cluster_scored events with honesty cap

One event per cluster after scoring. leaf_scores is the slice of all_leaf_scores
belonging to this cluster. degraded reuses leaf_scorer.DEGRADED_LEAF_CEILING
so the honesty fix from PR #75 applies per-cluster, not just run-level.

Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3 + §5.3.1"
```

### Task 1.5: Emit `repair_dispatched` per cluster repair attempt

**Files:**
- Modify: `backend/agents/rdr/controller.py` (the repair loop)
- Test: `tests/rdr/test_controller_cluster_events.py` (append)

- [ ] **Step 1: Locate the repair-dispatch loop**

Run: `grep -n "rdr_repair_pass_started\|rdr_repair_cluster_completed" backend/agents/rdr/controller.py`
Read context. Identify the inner loop that iterates over weak clusters within a repair pass.

- [ ] **Step 2: Write the failing test**

Append:

```python
def test_repair_dispatched_one_per_weak_cluster(captured_events, tmp_path: Path):
    """One repair_dispatched event per (pass, cluster) tuple. attempt counts
    repair passes for THIS cluster (1-indexed). prior_score is the cluster's
    score before this repair. failed_leaves are the leaves that scored below
    the cluster's pass threshold.
    """
    controller = _build_controller_stub(
        captured_events, tmp_path,
        cluster_count=3, weak_cluster_ids=["c01", "c03"],
    )
    controller.dispatch_all_clusters()
    controller.score_run()
    controller.run_repair_pass(pass_index=1)

    dispatched = [e for e in captured_events if e["event_type"] == "repair_dispatched"]
    assert len(dispatched) == 2
    assert {e["cluster_id"] for e in dispatched} == {"c01", "c03"}
    for ev in dispatched:
        assert ev["attempt"] == 1
        assert 0.0 <= ev["prior_score"] <= 1.0
        assert isinstance(ev["failed_leaves"], list)
        assert all(isinstance(lid, str) for lid in ev["failed_leaves"])
```

- [ ] **Step 3: Run the test to confirm it fails**

Run: `.venv/bin/python -m pytest tests/rdr/test_controller_cluster_events.py::test_repair_dispatched_one_per_weak_cluster -v`

- [ ] **Step 4: Add the emission**

In the repair-pass inner loop, just before dispatching the per-cluster repair agent:

```python
from backend.agents.rlm.sse_bridge import build_repair_dispatched

self._emit(
    "repair_dispatched",
    build_repair_dispatched(
        cluster_id=weak_cluster_id,
        attempt=repair_attempts_for_cluster[weak_cluster_id],  # 1-indexed, incremented before this point
        prior_score=per_cluster_scores[weak_cluster_id],
        failed_leaves=[
            leaf_id for leaf_id in cluster_to_leaves[weak_cluster_id]
            if all_leaf_scores.get(leaf_id, 0.0) < CLUSTER_PASS_THRESHOLD
        ],
    ),
)
```

`CLUSTER_PASS_THRESHOLD` — use the threshold the controller already uses to identify weak clusters; don't introduce a new constant. Likely already named (read the surrounding code).

`repair_attempts_for_cluster` — a dict the controller should already maintain (or add it: `defaultdict(int)` keyed by cluster_id, incremented when this cluster is repaired).

- [ ] **Step 5: Run the test to confirm pass**

Run: `.venv/bin/python -m pytest tests/rdr/test_controller_cluster_events.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/agents/rdr/controller.py tests/rdr/test_controller_cluster_events.py
git commit -m "feat(rdr): emit repair_dispatched per (pass, cluster) tuple

One event per weak-cluster repair attempt. attempt is 1-indexed per
cluster. failed_leaves names the leaves that fell below the pass
threshold and triggered the repair.

Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3"
```

### Task 1.6: Integration smoke — real rdr run emits all 4 spec events

**Files:**
- Test: `tests/rdr/test_rdr_emits_spec_cluster_events.py` (new)

- [ ] **Step 1: Write the integration test**

Create `tests/rdr/test_rdr_emits_spec_cluster_events.py`:

```python
"""End-to-end (offline) test that a real rdr run produces the spec's 4
cluster events in dashboard_events.jsonl.

Uses the same offline harness as tests/rdr/test_rdr_offline_e2e.py — copy
its setup (no LLM calls; stub reproduction agent returns canned artifacts).
"""

import json
from pathlib import Path

import pytest


def _run_rdr_offline(tmp_path: Path) -> Path:
    """Run rdr against the offline fixture bundle; return the run dir."""
    # Read tests/rdr/test_rdr_offline_e2e.py to copy the harness shape.
    raise NotImplementedError(
        "Copy the offline-run scaffolding from test_rdr_offline_e2e.py."
    )


def test_dashboard_events_contains_all_4_spec_cluster_events(tmp_path: Path):
    run_dir = _run_rdr_offline(tmp_path)
    events_path = run_dir / "dashboard_events.jsonl"
    assert events_path.exists()

    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    event_types = {e.get("event") for e in events}

    assert "cluster_started" in event_types, "missing cluster_started"
    assert "cluster_artifact_emitted" in event_types, "missing cluster_artifact_emitted"
    assert "cluster_scored" in event_types, "missing cluster_scored"
    # repair_dispatched only fires when a cluster fails — the offline fixture
    # may or may not exercise this. If the fixture's stub artifacts pass the
    # threshold, document the gap; otherwise assert presence.
    # See test_rdr_offline_e2e.py for what the offline fixture produces.
```

- [ ] **Step 2: Implement `_run_rdr_offline`**

Read `tests/rdr/test_rdr_offline_e2e.py` and copy the offline-run harness. The pattern likely involves a fixture bundle dir, stubbed LLM/agent calls, and `RdrController.run()`.

- [ ] **Step 3: Run the test, confirm pass**

Run: `.venv/bin/python -m pytest tests/rdr/test_rdr_emits_spec_cluster_events.py -v`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add tests/rdr/test_rdr_emits_spec_cluster_events.py
git commit -m "test(rdr): integration smoke for all 4 spec cluster events

Drives a full rdr run offline (no LLM calls) and asserts
cluster_started, cluster_artifact_emitted, cluster_scored land in
dashboard_events.jsonl. repair_dispatched gated by fixture shape;
documented inline."
```

---

## Phase 2 — Frontend reducer extensions

### Task 2.1: Add `mode` discriminator + cluster state to `RlmRunState`

**Files:**
- Modify: `frontend/src/hooks/use-rlm-run.ts`
- Modify: `frontend/src/components/lab/rlm/rlm-lab.tsx` (the `runMeta` interface — add `mode`)
- Test: `frontend/src/hooks/use-rlm-run.test.ts` (append)

- [ ] **Step 1: Read the existing `RlmRunState` shape**

Run: `wc -l frontend/src/hooks/use-rlm-run.ts && grep -n "RlmRunState\b\|INITIAL_RLM_STATE" frontend/src/hooks/use-rlm-run.ts | head -10`
Open the file at those lines; read 50 lines around `RlmRunState` and `INITIAL_RLM_STATE`.

- [ ] **Step 2: Write the failing test**

Append to `frontend/src/hooks/use-rlm-run.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { INITIAL_RLM_STATE } from "./use-rlm-run";

describe("RlmRunState cluster fields (spec §5.2.3)", () => {
  it("initial state carries a mode discriminator", () => {
    expect(INITIAL_RLM_STATE.mode).toBe("rlm");
  });

  it("initial state has empty cluster maps", () => {
    expect(INITIAL_RLM_STATE.clusterStatus).toEqual({});
    expect(INITIAL_RLM_STATE.clusterScores).toEqual({});
  });
});
```

- [ ] **Step 3: Run the test to confirm it fails**

Run: `cd frontend && npm test -- use-rlm-run`
Expected: type errors — `Property 'mode' does not exist on type 'RlmRunState'` etc.

- [ ] **Step 4: Extend `RlmRunState` and `INITIAL_RLM_STATE`**

In `frontend/src/hooks/use-rlm-run.ts`:

Add (near the existing `RlmRunState` interface):

```ts
export type RlmRunMode = "rlm" | "rdr";

export type ClusterStatus = "pending" | "running" | "scored" | "repaired" | "failed";
```

In `RlmRunState`:

```ts
export interface RlmRunState {
  // ...existing fields...
  mode: RlmRunMode;
  clusterStatus: Record<string, ClusterStatus>;
  clusterScores: Record<string, number>;
}
```

In `INITIAL_RLM_STATE`:

```ts
export const INITIAL_RLM_STATE: RlmRunState = {
  // ...existing fields...
  mode: "rlm",
  clusterStatus: {},
  clusterScores: {},
};
```

- [ ] **Step 5: Extend `runMeta` in `rlm-lab.tsx`**

In `frontend/src/components/lab/rlm/rlm-lab.tsx`, find the `runMeta` interface (around line 14–21 per the audit). Add `mode?: RlmRunMode` (optional for back-compat — pre-existing runs without metadata don't break).

```ts
import type { RlmRunMode } from "../../../hooks/use-rlm-run";

interface RunMeta {
  // ...existing fields...
  mode?: RlmRunMode;  // when set, the reducer is initialized with this mode.
}
```

If the `runMeta` is parameterized into `useRlmRun(initial)`, thread mode through: when `runMeta.mode` is `"rdr"`, the reducer's initial state should reflect that.

If there's no per-run initialization yet (only `INITIAL_RLM_STATE`), add a `useRlmRun(initialMode?: RlmRunMode)` overload that merges the mode into the initial state.

- [ ] **Step 6: Run the test, confirm pass**

Run: `cd frontend && npm test -- use-rlm-run`
Expected: pass.

- [ ] **Step 7: Run the full vitest suite + tsc**

Run from `frontend/`: `npm test 2>&1 | tail -10 && npx tsc --noEmit 2>&1 | tail -3`
Expected: vitest pass count = baseline + 2; tsc clean.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/hooks/use-rlm-run.ts frontend/src/hooks/use-rlm-run.test.ts frontend/src/components/lab/rlm/rlm-lab.tsx
git commit -m "feat(reducer): add mode discriminator + cluster state to RlmRunState

Extends RlmRunState with mode (rlm|rdr), clusterStatus, clusterScores.
runMeta gains an optional mode field; reducer initialized with it. Pure
additive — no existing folds touch the new fields.

Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3"
```

### Task 2.2: foldClusterStarted + foldClusterArtifactEmitted + foldClusterScored + foldRepairDispatched

**Files:**
- Modify: `frontend/src/hooks/use-rlm-run.ts`
- Modify: `frontend/src/lib/events/rlm-events.ts` (or wherever event types live — add 4 new event types)
- Test: `frontend/src/hooks/use-rlm-run.test.ts` (append)

- [ ] **Step 1: Locate the event-type union**

Run: `grep -rn "rubric_score\|repl_iteration\|RlmEvent" frontend/src/lib/events/ frontend/src/hooks/ | head -10`
Find the discriminated union (likely in `frontend/src/lib/events/rlm-events.ts` or similar). Add 4 new variants matching the backend builder shapes.

- [ ] **Step 2: Add 4 new event type interfaces**

In `frontend/src/lib/events/rlm-events.ts` (or equivalent), add:

```ts
export interface ClusterStartedEvent {
  event: "cluster_started";
  timestamp: string;
  cluster_id: string;
  cluster_title: string;
  leaves: Array<{ id: string; weight: number; requirements: string }>;
  iteration: number;
}

export interface ClusterArtifactEmittedEvent {
  event: "cluster_artifact_emitted";
  timestamp: string;
  cluster_id: string;
  artifact_path: string;
  byte_size: number;
  language: string | null;
}

export interface ClusterScoredEvent {
  event: "cluster_scored";
  timestamp: string;
  cluster_id: string;
  score: number;
  leaf_scores: Record<string, number>;
  degraded: boolean;
}

export interface RepairDispatchedEvent {
  event: "repair_dispatched";
  timestamp: string;
  cluster_id: string;
  attempt: number;
  prior_score: number;
  failed_leaves: string[];
}

// Update the discriminated union (find the existing union and add these):
export type RlmEvent =
  | /* existing variants */
  | ClusterStartedEvent
  | ClusterArtifactEmittedEvent
  | ClusterScoredEvent
  | RepairDispatchedEvent;
```

- [ ] **Step 3: Write the failing test**

Append to `frontend/src/hooks/use-rlm-run.test.ts`:

```ts
describe("cluster event folds (spec §5.2.3)", () => {
  it("foldClusterStarted marks the cluster running", () => {
    const ev = {
      event: "cluster_started" as const,
      timestamp: "2026-05-23T04:10:00Z",
      cluster_id: "c01",
      cluster_title: "Setup",
      leaves: [{ id: "L1", weight: 0.5, requirements: "Install X" }],
      iteration: 1,
    };
    const s = fold(INITIAL_RLM_STATE, ev);
    expect(s.clusterStatus["c01"]).toBe("running");
  });

  it("foldClusterScored captures score + flips status to scored", () => {
    let s = fold(INITIAL_RLM_STATE, {
      event: "cluster_started",
      timestamp: "2026-05-23T04:10:00Z",
      cluster_id: "c01",
      cluster_title: "Setup",
      leaves: [{ id: "L1", weight: 0.5, requirements: "X" }],
      iteration: 1,
    });
    s = fold(s, {
      event: "cluster_scored",
      timestamp: "2026-05-23T04:11:00Z",
      cluster_id: "c01",
      score: 0.78,
      leaf_scores: { L1: 0.78 },
      degraded: false,
    });
    expect(s.clusterStatus["c01"]).toBe("scored");
    expect(s.clusterScores["c01"]).toBeCloseTo(0.78);
  });

  it("foldClusterScored with degraded=true does not exceed 0.35 — honesty cap", () => {
    let s = fold(INITIAL_RLM_STATE, {
      event: "cluster_scored",
      timestamp: "2026-05-23T04:11:00Z",
      cluster_id: "c01",
      score: 0.30,
      leaf_scores: {},
      degraded: true,
    });
    // The reducer trusts the backend cap; it does NOT re-clamp here.
    // But it should preserve the degraded signal for the UI.
    expect(s.clusterScores["c01"]).toBeCloseTo(0.30);
  });

  it("foldRepairDispatched marks the cluster as running again", () => {
    let s = fold(INITIAL_RLM_STATE, {
      event: "cluster_scored",
      timestamp: "2026-05-23T04:11:00Z",
      cluster_id: "c01",
      score: 0.20,
      leaf_scores: {},
      degraded: false,
    });
    s = fold(s, {
      event: "repair_dispatched",
      timestamp: "2026-05-23T04:12:00Z",
      cluster_id: "c01",
      attempt: 1,
      prior_score: 0.20,
      failed_leaves: ["L1"],
    });
    // Status flips back to running until the next cluster_scored.
    expect(s.clusterStatus["c01"]).toBe("running");
  });

  it("foldClusterArtifactEmitted is a no-op on clusterStatus / clusterScores", () => {
    let s = fold(INITIAL_RLM_STATE, {
      event: "cluster_started",
      timestamp: "2026-05-23T04:10:00Z",
      cluster_id: "c01",
      cluster_title: "S",
      leaves: [{ id: "L1", weight: 1, requirements: "X" }],
      iteration: 1,
    });
    const before = s.clusterStatus["c01"];
    s = fold(s, {
      event: "cluster_artifact_emitted",
      timestamp: "2026-05-23T04:10:30Z",
      cluster_id: "c01",
      artifact_path: "code/train.py",
      byte_size: 1024,
      language: "python",
    });
    // Status unchanged (artifact events feed a separate UI surface).
    expect(s.clusterStatus["c01"]).toBe(before);
  });
});
```

- [ ] **Step 4: Run the test, confirm fail**

Run: `cd frontend && npm test -- use-rlm-run`

- [ ] **Step 5: Implement the 4 folds**

In `frontend/src/hooks/use-rlm-run.ts`, add 4 fold functions and wire them into the main `fold` dispatch:

```ts
function foldClusterStarted(state: RlmRunState, ev: ClusterStartedEvent): RlmRunState {
  return {
    ...state,
    clusterStatus: { ...state.clusterStatus, [ev.cluster_id]: "running" },
  };
}

function foldClusterArtifactEmitted(
  state: RlmRunState,
  _ev: ClusterArtifactEmittedEvent,
): RlmRunState {
  // Artifact-level events don't affect clusterStatus/Scores; consumers
  // wire to a separate state slice if they want to display them.
  // Spec §5.2.3 mentions artifacts in NodeDetailPopup — track separately
  // there when that branch lands.
  return state;
}

function foldClusterScored(state: RlmRunState, ev: ClusterScoredEvent): RlmRunState {
  return {
    ...state,
    clusterStatus: { ...state.clusterStatus, [ev.cluster_id]: "scored" },
    clusterScores: { ...state.clusterScores, [ev.cluster_id]: ev.score },
  };
}

function foldRepairDispatched(state: RlmRunState, ev: RepairDispatchedEvent): RlmRunState {
  return {
    ...state,
    clusterStatus: { ...state.clusterStatus, [ev.cluster_id]: "running" },
  };
}
```

Wire into the main `fold` switch:

```ts
export function fold(state: RlmRunState, ev: RlmEvent): RlmRunState {
  switch (ev.event) {
    // ...existing cases...
    case "cluster_started":          return foldClusterStarted(state, ev);
    case "cluster_artifact_emitted": return foldClusterArtifactEmitted(state, ev);
    case "cluster_scored":           return foldClusterScored(state, ev);
    case "repair_dispatched":        return foldRepairDispatched(state, ev);
    default: return state;
  }
}
```

- [ ] **Step 6: Run the test, confirm pass**

Run: `cd frontend && npm test -- use-rlm-run`

- [ ] **Step 7: Commit**

```bash
git add frontend/src/hooks/use-rlm-run.ts frontend/src/hooks/use-rlm-run.test.ts frontend/src/lib/events/rlm-events.ts
git commit -m "feat(reducer): handle 4 cluster events in fold

cluster_started → status=running; cluster_scored → status=scored + score
recorded; repair_dispatched → status=running again. cluster_artifact_emitted
is a no-op on cluster state today (consumed by a separate slice when
NodeDetailPopup gains the cluster branch).

Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3"
```

---

## Phase 3 — Frontend dual-layout canvas

### Task 3.1: Extract `dynamicCandidateLayout` from existing `layoutTree`

**Files:**
- Modify: `frontend/src/components/lab/rlm/layout-tree.ts`
- Test: `frontend/src/components/lab/rlm/layout-tree.test.ts` (new or append)

- [ ] **Step 1: Read existing `layoutTree`**

Run: `wc -l frontend/src/components/lab/rlm/layout-tree.ts`
Read the file in full.

- [ ] **Step 2: Write the failing test**

Create or append to `frontend/src/components/lab/rlm/layout-tree.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { dynamicCandidateLayout, layoutTree } from "./layout-tree";

describe("dynamicCandidateLayout (spec §5.2.3 — rlm path)", () => {
  it("is the existing layoutTree under a new name", () => {
    expect(dynamicCandidateLayout).toBe(layoutTree);
  });
});
```

- [ ] **Step 3: Run the test, confirm fail**

Run: `cd frontend && npm test -- layout-tree`

- [ ] **Step 4: Add the alias**

At the bottom of `frontend/src/components/lab/rlm/layout-tree.ts`:

```ts
/** Alias for the rlm-mode layout — name introduced for spec §5.2.3 readability. */
export const dynamicCandidateLayout = layoutTree;
```

Do NOT rename `layoutTree` — existing call sites rely on it. The alias is the new name; both refer to the same function.

- [ ] **Step 5: Confirm pass + commit**

```bash
cd frontend && npm test -- layout-tree
cd ..
git add frontend/src/components/lab/rlm/layout-tree.ts frontend/src/components/lab/rlm/layout-tree.test.ts
git commit -m "refactor(canvas): alias layoutTree as dynamicCandidateLayout

Introduces the spec-named alias without changing call sites. Sets up
the dual-layout abstraction in 3.2/3.3.

Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3"
```

### Task 3.2: Add `rubricClusterLayout` function

**Files:**
- Modify: `frontend/src/components/lab/rlm/layout-tree.ts`
- Test: `frontend/src/components/lab/rlm/layout-tree.test.ts` (append)

- [ ] **Step 1: Write the failing test**

Append:

```ts
import { rubricClusterLayout } from "./layout-tree";
import type { TreeNodeData } from "./tree-types";  // adjust import to actual location

describe("rubricClusterLayout (spec §5.2.3 — rdr path)", () => {
  it("places clusters as inner nodes and leaves as terminals", () => {
    const tree: TreeNodeData[] = [
      { id: "root", kind: "root", parentId: null, label: "Paper" },
      { id: "c01",  kind: "cluster", parentId: "root", label: "Setup" },
      { id: "L1",   kind: "leaf",    parentId: "c01",  label: "Install X", weight: 0.5 },
      { id: "L2",   kind: "leaf",    parentId: "c01",  label: "Conf Y",    weight: 0.5 },
      { id: "c02",  kind: "cluster", parentId: "root", label: "Train" },
      { id: "L3",   kind: "leaf",    parentId: "c02",  label: "Run epoch", weight: 1.0 },
    ];
    const placements = rubricClusterLayout(tree);
    // 6 nodes, 6 placements
    expect(placements).toHaveLength(6);
    // Leaves cluster under their parent.
    const c01 = placements.find(p => p.id === "c01")!;
    const L1  = placements.find(p => p.id === "L1")!;
    const L2  = placements.find(p => p.id === "L2")!;
    expect(L1.y).toBeGreaterThan(c01.y);   // leaves below their cluster
    expect(L2.y).toBeGreaterThan(c01.y);
    // L1 and L2 share a y-band (siblings) but differ in x.
    expect(L1.y).toBeCloseTo(L2.y, 1);
    expect(L1.x).not.toBe(L2.x);
  });

  it("returns an empty array on empty input", () => {
    expect(rubricClusterLayout([])).toEqual([]);
  });

  it("handles a root-only tree", () => {
    const placements = rubricClusterLayout([
      { id: "root", kind: "root", parentId: null, label: "Paper" },
    ]);
    expect(placements).toHaveLength(1);
  });
});
```

The `TreeNodeData` type may need extending — see Task 4.1.

- [ ] **Step 2: Run the test to confirm it fails**

- [ ] **Step 3: Implement `rubricClusterLayout`**

In `frontend/src/components/lab/rlm/layout-tree.ts`:

```ts
import type { TreeNodeData } from "./tree-types";  // adjust import

interface NodePlacement {
  id: string;
  x: number;
  y: number;
}

/**
 * Layout for the rdr mode — fixed rubric tree.
 *
 * Three levels:
 * - Root (the paper) at the top, centered.
 * - Clusters as inner nodes, evenly spaced in a horizontal band.
 * - Leaves below each cluster, evenly spaced.
 *
 * Y bands are fixed: root=0, clusters=120, leaves=240. X is evenly distributed
 * within each band (or within each cluster's leaves for the leaf band).
 *
 * Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3.
 */
export function rubricClusterLayout(tree: TreeNodeData[]): NodePlacement[] {
  if (tree.length === 0) return [];

  const Y_ROOT = 0;
  const Y_CLUSTERS = 120;
  const Y_LEAVES = 240;
  const X_SPAN = 800;

  const roots = tree.filter((n) => n.kind === "root");
  const clusters = tree.filter((n) => n.kind === "cluster");
  const leaves = tree.filter((n) => n.kind === "leaf");

  const placements: NodePlacement[] = [];

  roots.forEach((n) => {
    placements.push({ id: n.id, x: X_SPAN / 2, y: Y_ROOT });
  });

  clusters.forEach((c, i) => {
    const x = clusters.length === 1
      ? X_SPAN / 2
      : (i + 0.5) * (X_SPAN / clusters.length);
    placements.push({ id: c.id, x, y: Y_CLUSTERS });
  });

  // For each cluster, lay out its leaves underneath.
  clusters.forEach((c) => {
    const clusterPlacement = placements.find((p) => p.id === c.id)!;
    const clusterLeaves = leaves.filter((l) => l.parentId === c.id);
    const leafSpan = 100; // narrower than cluster span
    clusterLeaves.forEach((l, i) => {
      const offset = clusterLeaves.length === 1
        ? 0
        : (i - (clusterLeaves.length - 1) / 2) * (leafSpan / clusterLeaves.length);
      placements.push({
        id: l.id,
        x: clusterPlacement.x + offset,
        y: Y_LEAVES,
      });
    });
  });

  return placements;
}
```

- [ ] **Step 4: Run the test, confirm pass**

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/lab/rlm/layout-tree.ts frontend/src/components/lab/rlm/layout-tree.test.ts
git commit -m "feat(canvas): add rubricClusterLayout for rdr-mode tree

Fixed three-level layout (root / clusters / leaves). Clusters in a
horizontal band, leaves below their parent cluster. Independent from
the candidate-spawning layout used by rlm.

Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3"
```

### Task 3.3: Switch ExplorationCanvas via mode

**Files:**
- Modify: `frontend/src/components/lab/rlm/exploration-canvas.tsx`
- Test: `frontend/src/components/lab/rlm/exploration-canvas.test.tsx` (new or append)

- [ ] **Step 1: Locate the layout-call site in the canvas**

Run: `grep -n "layoutTree\|dynamicCandidateLayout" frontend/src/components/lab/rlm/exploration-canvas.tsx`

- [ ] **Step 2: Write the failing test**

Append:

```tsx
import { render } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { ExplorationCanvas } from "./exploration-canvas";

vi.mock("./layout-tree", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./layout-tree")>();
  return {
    ...actual,
    dynamicCandidateLayout: vi.fn(actual.dynamicCandidateLayout),
    rubricClusterLayout: vi.fn(actual.rubricClusterLayout),
  };
});

describe("ExplorationCanvas dual-layout (spec §5.2.3)", () => {
  it("calls dynamicCandidateLayout when mode=rlm", async () => {
    const { dynamicCandidateLayout, rubricClusterLayout } = await import("./layout-tree");
    render(<ExplorationCanvas tree={[]} iterations={[]} mode="rlm" />);
    expect(dynamicCandidateLayout).toHaveBeenCalled();
    expect(rubricClusterLayout).not.toHaveBeenCalled();
  });

  it("calls rubricClusterLayout when mode=rdr", async () => {
    const { dynamicCandidateLayout, rubricClusterLayout } = await import("./layout-tree");
    vi.clearAllMocks();
    render(<ExplorationCanvas tree={[]} iterations={[]} mode="rdr" />);
    expect(rubricClusterLayout).toHaveBeenCalled();
    expect(dynamicCandidateLayout).not.toHaveBeenCalled();
  });

  it("defaults to rlm when mode is undefined (back-compat)", async () => {
    const { dynamicCandidateLayout, rubricClusterLayout } = await import("./layout-tree");
    vi.clearAllMocks();
    render(<ExplorationCanvas tree={[]} iterations={[]} />);
    expect(dynamicCandidateLayout).toHaveBeenCalled();
    expect(rubricClusterLayout).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 3: Run the test to confirm it fails**

- [ ] **Step 4: Extend ExplorationCanvasProps with mode**

In `frontend/src/components/lab/rlm/exploration-canvas.tsx`:

```tsx
import type { RlmRunMode } from "../../../hooks/use-rlm-run";
import { dynamicCandidateLayout, rubricClusterLayout } from "./layout-tree";

export interface ExplorationCanvasProps {
  // ...existing fields...
  mode?: RlmRunMode;
}

export function ExplorationCanvas({ tree, iterations, mode = "rlm", ...rest }: ExplorationCanvasProps) {
  const layout = mode === "rdr" ? rubricClusterLayout : dynamicCandidateLayout;
  const placements = layout(tree);
  // ...rest of the component, using `placements`...
}
```

Find where the existing component called the layout and replace `layoutTree(tree)` (or `dynamicCandidateLayout(tree)`) with `layout(tree)`.

- [ ] **Step 5: Thread mode through the parent (rlm-lab.tsx)**

In `frontend/src/components/lab/rlm/rlm-lab.tsx`, where `<ExplorationCanvas ... />` is rendered, pass `mode={runMeta.mode ?? "rlm"}`.

- [ ] **Step 6: Confirm pass + commit**

```bash
cd frontend && npm test -- exploration-canvas
cd ..
git add frontend/src/components/lab/rlm/exploration-canvas.tsx frontend/src/components/lab/rlm/exploration-canvas.test.tsx frontend/src/components/lab/rlm/rlm-lab.tsx
git commit -m "feat(canvas): dual-layout — dynamicCandidateLayout for rlm, rubricClusterLayout for rdr

ExplorationCanvas now reads a mode prop (defaulting to rlm for back-compat)
and dispatches to the matching layout function. Threaded from runMeta.mode
via rlm-lab.tsx.

Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3"
```

---

## Phase 4 — NodeDetailPopup cluster branch

### Task 4.1: Add cluster TreeNode variant

**Files:**
- Modify: `frontend/src/components/lab/rlm/tree-types.ts` (or wherever the TreeNodeData type lives — find via grep)
- Test: not needed — type-only change.

- [ ] **Step 1: Locate the TreeNodeData type**

Run: `grep -rn "TreeNodeData\|kind: \"candidate\"" frontend/src/components/lab/rlm/ | head -5`

- [ ] **Step 2: Extend the union with `"cluster"` variant**

Add a new variant alongside `"candidate"`. The shape:

```ts
export interface ClusterTreeNodeData {
  id: string;
  kind: "cluster";
  parentId: string | null;
  label: string;
  clusterTitle: string;
  leaves: Array<{ id: string; weight: number; requirements: string }>;
  status: ClusterStatus;          // imported from use-rlm-run
  score: number | null;            // null until first cluster_scored
  degraded: boolean;
  artifacts: Array<{ path: string; bytes: number; language: string | null }>;
}

export type TreeNodeData =
  | /* existing variants */
  | ClusterTreeNodeData;
```

Also add a `LeafTreeNodeData` if not already present, for the cluster's leaf children:

```ts
export interface LeafTreeNodeData {
  id: string;
  kind: "leaf";
  parentId: string;  // the cluster id
  label: string;
  weight: number;
  requirements: string;
  score: number | null;
}
```

- [ ] **Step 3: Verify tsc clean**

Run: `cd frontend && npx tsc --noEmit 2>&1 | tail -10`
Expected: no new errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/lab/rlm/tree-types.ts
git commit -m "feat(tree): add cluster and leaf TreeNode variants

Strict types for the rdr mode's rubric-cluster tree: ClusterTreeNodeData
carries title, leaves, status, score, degraded flag, and artifacts.
LeafTreeNodeData carries weight, requirements, score. Pure type-only
extension; runtime call sites land in 4.2.

Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3"
```

### Task 4.2: Add cluster branch to NodeDetailPopup

**Files:**
- Modify: `frontend/src/components/lab/rlm/node-detail-popup.tsx`
- Test: `frontend/src/components/lab/rlm/node-detail-popup.test.tsx` (new or append)

- [ ] **Step 1: Read the existing popup**

Run: `wc -l frontend/src/components/lab/rlm/node-detail-popup.tsx`
Read in full.

- [ ] **Step 2: Write the failing test**

Append (or create):

```tsx
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { NodeDetailPopup } from "./node-detail-popup";
import type { ClusterTreeNodeData, LeafTreeNodeData } from "./tree-types";

describe("NodeDetailPopup cluster branch (spec §5.2.3)", () => {
  const cluster: ClusterTreeNodeData = {
    id: "c01",
    kind: "cluster",
    parentId: null,
    label: "Method understanding",
    clusterTitle: "Method understanding",
    leaves: [
      { id: "L1", weight: 0.5, requirements: "Define X" },
      { id: "L2", weight: 0.5, requirements: "Implement Y" },
    ],
    status: "scored",
    score: 0.72,
    degraded: false,
    artifacts: [
      { path: "code/train.py", bytes: 4096, language: "python" },
      { path: "config.yaml", bytes: 512, language: "yaml" },
    ],
  };

  it("renders the cluster title", () => {
    render(<NodeDetailPopup node={cluster} onClose={() => {}} />);
    expect(screen.getByText("Method understanding")).toBeInTheDocument();
  });

  it("renders all leaves", () => {
    render(<NodeDetailPopup node={cluster} onClose={() => {}} />);
    expect(screen.getByText("Define X")).toBeInTheDocument();
    expect(screen.getByText("Implement Y")).toBeInTheDocument();
  });

  it("renders the score", () => {
    render(<NodeDetailPopup node={cluster} onClose={() => {}} />);
    expect(screen.getByText(/0\.72/)).toBeInTheDocument();
  });

  it("renders the artifacts list", () => {
    render(<NodeDetailPopup node={cluster} onClose={() => {}} />);
    expect(screen.getByText("code/train.py")).toBeInTheDocument();
    expect(screen.getByText("config.yaml")).toBeInTheDocument();
  });

  it("shows degraded warning when degraded=true", () => {
    render(
      <NodeDetailPopup
        node={{ ...cluster, degraded: true, score: 0.30 }}
        onClose={() => {}}
      />
    );
    expect(screen.getByText(/degraded/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Run the test, confirm fail**

- [ ] **Step 4: Add the cluster branch**

In `frontend/src/components/lab/rlm/node-detail-popup.tsx`, find the existing `if (node.kind === "candidate")` branch. Add a new branch for `node.kind === "cluster"`:

```tsx
if (node.kind === "cluster") {
  return (
    <div className={styles.popup}>
      <header className={styles.header}>
        <h3>{node.clusterTitle}</h3>
        <button onClick={onClose} aria-label="Close popup">×</button>
      </header>

      <section>
        <p className={styles.score}>
          Score: {node.score !== null ? node.score.toFixed(2) : "—"}
          {node.degraded && (
            <span className={styles.degraded} role="status">
              (degraded — honesty cap applied)
            </span>
          )}
        </p>
      </section>

      <section>
        <h4>Leaves grading this cluster</h4>
        <ul className={styles.leafList}>
          {node.leaves.map((leaf) => (
            <li key={leaf.id}>
              <span className={styles.leafWeight}>w={leaf.weight.toFixed(2)}</span>
              <span className={styles.leafReq}>{leaf.requirements}</span>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h4>Artifacts</h4>
        {node.artifacts.length === 0 ? (
          <p className={styles.empty}>No artifacts yet.</p>
        ) : (
          <ul className={styles.artifactList}>
            {node.artifacts.map((a) => (
              <li key={a.path}>
                <code>{a.path}</code>
                <span className={styles.bytes}>{formatBytes(a.bytes)}</span>
                {a.language && <span className={styles.lang}>{a.language}</span>}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
```

Add corresponding CSS in `frontend/src/components/lab/rlm/node-detail-popup.module.css` for `.leafList`, `.artifactList`, `.degraded`, `.empty`, `.bytes`, `.lang`. Match the visual language of the candidate branch.

- [ ] **Step 5: Run the test, confirm pass**

- [ ] **Step 6: Run the full vitest + tsc**

Run from `frontend/`: `npm test 2>&1 | tail -5 && npx tsc --noEmit`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/lab/rlm/node-detail-popup.tsx frontend/src/components/lab/rlm/node-detail-popup.module.css frontend/src/components/lab/rlm/node-detail-popup.test.tsx
git commit -m "feat(popup): cluster branch — title, leaves, score, artifacts, degraded flag

Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3"
```

---

## Phase 5 — Fixture + `?rdrFixture=1` wiring

### Task 5.1: Create `rdr-run.fixture.ts`

**Files:**
- Create: `frontend/src/components/lab/rlm/__fixtures__/rdr-run.fixture.ts`
- Test: `frontend/src/components/lab/rlm/__fixtures__/rdr-run.fixture.test.ts` (new)

- [ ] **Step 1: Read the existing rlm fixture**

Run: `wc -l frontend/src/components/lab/rlm/__fixtures__/rlm-run.fixture.ts`
Read in full to understand the shape — what kind of event stream it produces.

- [ ] **Step 2: Create the rdr fixture**

`frontend/src/components/lab/rlm/__fixtures__/rdr-run.fixture.ts`:

```ts
import type { RlmEvent } from "../../../../lib/events/rlm-events";

/**
 * rdr-mode fixture run.
 *
 * Three clusters; one succeeds outright, one needs a repair, one stays
 * degraded. Used by tests and the ?rdrFixture=1 URL param in /lab.
 *
 * Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3.
 */
export const rdrRunFixture: RlmEvent[] = [
  // Cluster 1 — passes
  {
    event: "cluster_started",
    timestamp: "2026-05-23T04:10:00Z",
    cluster_id: "c01",
    cluster_title: "Environment setup",
    leaves: [
      { id: "L01", weight: 0.5, requirements: "Install required Python packages" },
      { id: "L02", weight: 0.5, requirements: "Set seed = 42" },
    ],
    iteration: 1,
  },
  {
    event: "cluster_artifact_emitted",
    timestamp: "2026-05-23T04:10:30Z",
    cluster_id: "c01",
    artifact_path: "code/setup.py",
    byte_size: 1024,
    language: "python",
  },
  {
    event: "cluster_artifact_emitted",
    timestamp: "2026-05-23T04:10:31Z",
    cluster_id: "c01",
    artifact_path: "requirements.txt",
    byte_size: 256,
    language: null,
  },
  {
    event: "cluster_scored",
    timestamp: "2026-05-23T04:11:00Z",
    cluster_id: "c01",
    score: 0.92,
    leaf_scores: { L01: 0.95, L02: 0.89 },
    degraded: false,
  },

  // Cluster 2 — fails, gets repaired, passes
  {
    event: "cluster_started",
    timestamp: "2026-05-23T04:11:00Z",
    cluster_id: "c02",
    cluster_title: "Training loop",
    leaves: [
      { id: "L03", weight: 0.4, requirements: "Implement forward pass" },
      { id: "L04", weight: 0.6, requirements: "Compute loss correctly" },
    ],
    iteration: 1,
  },
  {
    event: "cluster_artifact_emitted",
    timestamp: "2026-05-23T04:11:45Z",
    cluster_id: "c02",
    artifact_path: "code/train.py",
    byte_size: 8192,
    language: "python",
  },
  {
    event: "cluster_scored",
    timestamp: "2026-05-23T04:12:00Z",
    cluster_id: "c02",
    score: 0.38,
    leaf_scores: { L03: 0.50, L04: 0.30 },
    degraded: false,
  },
  {
    event: "repair_dispatched",
    timestamp: "2026-05-23T04:12:30Z",
    cluster_id: "c02",
    attempt: 1,
    prior_score: 0.38,
    failed_leaves: ["L04"],
  },
  {
    event: "cluster_artifact_emitted",
    timestamp: "2026-05-23T04:13:00Z",
    cluster_id: "c02",
    artifact_path: "code/train.py",  // overwritten
    byte_size: 9200,
    language: "python",
  },
  {
    event: "cluster_scored",
    timestamp: "2026-05-23T04:13:30Z",
    cluster_id: "c02",
    score: 0.78,
    leaf_scores: { L03: 0.80, L04: 0.76 },
    degraded: false,
  },

  // Cluster 3 — degraded (no metrics)
  {
    event: "cluster_started",
    timestamp: "2026-05-23T04:13:30Z",
    cluster_id: "c03",
    cluster_title: "Evaluation",
    leaves: [
      { id: "L05", weight: 1.0, requirements: "Compute Top-1 accuracy on test set" },
    ],
    iteration: 1,
  },
  {
    event: "cluster_scored",
    timestamp: "2026-05-23T04:14:30Z",
    cluster_id: "c03",
    score: 0.20,
    leaf_scores: { L05: 0.20 },
    degraded: true,
  },

  // Run-level rubric_score (mode-agnostic; spec §5.2.3 says Region 2 reuses)
  {
    event: "rubric_score",
    timestamp: "2026-05-23T04:14:30Z",
    iteration: 3,
    score: 0.61,
    target: 0.80,
    areas: [
      { area: "Environment setup", status: "pass",    score: 0.92, weight: 0.25 },
      { area: "Training loop",     status: "pass",    score: 0.78, weight: 0.40 },
      { area: "Evaluation",        status: "fail",    score: 0.20, weight: 0.35 },
    ],
  },

  {
    event: "run_complete",
    timestamp: "2026-05-23T04:15:00Z",
    verdict: "partial",
    final_score: 0.61,
  },
];
```

- [ ] **Step 3: Sanity test the fixture**

Create `frontend/src/components/lab/rlm/__fixtures__/rdr-run.fixture.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { rdrRunFixture } from "./rdr-run.fixture";
import { fold, INITIAL_RLM_STATE } from "../../../../hooks/use-rlm-run";

describe("rdr-run fixture", () => {
  it("folds end-to-end without errors", () => {
    const state = rdrRunFixture.reduce(fold, { ...INITIAL_RLM_STATE, mode: "rdr" });
    expect(Object.keys(state.clusterStatus).sort()).toEqual(["c01", "c02", "c03"]);
    expect(state.clusterStatus["c01"]).toBe("scored");
    expect(state.clusterStatus["c02"]).toBe("scored");
    expect(state.clusterStatus["c03"]).toBe("scored");
    expect(state.clusterScores["c01"]).toBeCloseTo(0.92);
    expect(state.clusterScores["c02"]).toBeCloseTo(0.78);
    expect(state.clusterScores["c03"]).toBeCloseTo(0.20);
  });

  it("contains at least one degraded cluster_scored", () => {
    expect(rdrRunFixture.some((e) => e.event === "cluster_scored" && e.degraded)).toBe(true);
  });

  it("contains at least one repair_dispatched", () => {
    expect(rdrRunFixture.some((e) => e.event === "repair_dispatched")).toBe(true);
  });
});
```

- [ ] **Step 4: Run + commit**

```bash
cd frontend && npm test -- rdr-run.fixture
cd ..
git add frontend/src/components/lab/rlm/__fixtures__/rdr-run.fixture.ts frontend/src/components/lab/rlm/__fixtures__/rdr-run.fixture.test.ts
git commit -m "feat(fixture): rdr-run.fixture.ts mirrors rlm-run.fixture for rdr mode

3 clusters: one passes, one needs repair (then passes), one stays
degraded. Covers all 4 spec cluster event types end-to-end.

Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3"
```

### Task 5.2: Wire `?rdrFixture=1` parsing in `lab-shell.tsx`

**Files:**
- Modify: `frontend/src/components/lab/lab-shell.tsx`
- Test: extend an existing lab-shell test or add a new one.

- [ ] **Step 1: Read the existing `?rlmFixture=1` branch**

Run: `grep -n "rlmFixture\|fixture" frontend/src/components/lab/lab-shell.tsx`
Read 30 lines around the match.

- [ ] **Step 2: Add the parallel `?rdrFixture=1` branch**

After the existing rlm-fixture branch:

```tsx
import { rdrRunFixture } from "./rlm/__fixtures__/rdr-run.fixture";

// ...inside the component, in the param-parsing logic:
const rdrFixtureFlag = searchParams.get("rdrFixture") === "1";

if (rdrFixtureFlag) {
  return (
    <RlmLab
      runMeta={{ projectId: "fixture-rdr", paperTitle: "Fixture rdr Paper", mode: "rdr" }}
      events={rdrRunFixture}
      // ...same other props as the rlm fixture branch
    />
  );
}
```

- [ ] **Step 3: Test it manually**

Run from `frontend/`: `npm run dev`
Open `http://localhost:3000/lab?rdrFixture=1`
Expected: page loads without console errors; ExplorationCanvas renders the cluster tree; clicking a cluster opens the popup with leaves + artifacts; clicking a leaf opens a leaf-specific popup (if you implemented the leaf branch); the rubric strip shows 0.61 with 3 areas.

If anything errors, fix it before committing. Capture screenshots if you want to keep a visual record.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/lab/lab-shell.tsx
git commit -m "feat(lab): wire ?rdrFixture=1 query param to render the rdr fixture

Spec: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3"
```

---

## Phase 6 — End-to-end verification

### Task 6.1: Vitest E2E — `?rdrFixture=1` renders all regions without console errors

**Files:**
- Test: `frontend/src/components/lab/lab-shell.test.tsx` (new or append) — Vitest with `@testing-library/react`
- OR: `frontend/e2e/rdr-fixture.spec.ts` (new) — Playwright

Pick Playwright if it's cheap to add another spec; vitest if Playwright setup is heavy. The audit will tell you whether `frontend/e2e/` is already configured.

- [ ] **Step 1: Write the test**

For Playwright (`frontend/e2e/rdr-fixture.spec.ts`):

```ts
import { test, expect } from "@playwright/test";

test("rdr fixture renders all regions without console errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });

  await page.goto("/lab?rdrFixture=1");

  // Region 1: header
  await expect(page.locator('[data-region="header"], header').first()).toBeVisible();
  // Region 2: rubric strip
  await expect(page.getByText(/0\.61/)).toBeVisible();
  // Region 4: exploration canvas — at least the SVG element
  await expect(page.locator("svg").first()).toBeVisible();
  // Region 6 (Primitive History Bar): present even if rdr doesn't emit primitive_call events
  // (the bar should render its empty state, not crash).
  await expect(page.locator('[data-band="primitive-history"], footer').first()).toBeVisible();

  expect(errors, `Console errors:\n${errors.join("\n")}`).toEqual([]);
});
```

For vitest equivalent: render `<LabShell />` with mocked `useSearchParams` returning `?rdrFixture=1`, then assert no thrown errors and key text is in the DOM.

- [ ] **Step 2: Run and confirm pass**

Run: `npx playwright test e2e/rdr-fixture.spec.ts` from `frontend/`.

- [ ] **Step 3: Commit**

```bash
git add frontend/e2e/rdr-fixture.spec.ts
git commit -m "test(e2e): rdr fixture renders all regions without console errors

Spec acceptance: docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md §5.2.3"
```

### Task 6.2: Real `--mode rdr` smoke (final acceptance gate)

**Files:** none — verification only.

- [ ] **Step 1: Start the backend + frontend dev servers**

Terminal A: `.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000`
Terminal B: from `frontend/`: `REPROLAB_BACKEND_URL=http://127.0.0.1:8000 npm run dev`

- [ ] **Step 2: Trigger a real `--mode rdr` run via the UI**

Open `http://localhost:3000/lab`. Upload a small PaperBench bundle (or hit `POST /runs` directly via the UI's upload flow). Select `--mode rdr`. Watch the canvas as it renders cluster nodes appearing live.

- [ ] **Step 3: Verify acceptance per spec §5.2.3:**

- Big rubric number animates from initial to final score smoothly (rubric-climb already validated; should "just work").
- Per-cluster status nodes appear, flip from running → scored as `cluster_scored` events arrive.
- One cluster shows a "degraded" indicator (if the run produces one).
- Click any cluster → popup opens with leaves + artifacts.
- No console errors throughout.

- [ ] **Step 4: Document the result in `learn.md` if a non-obvious issue surfaced**

Per project convention (CLAUDE.md / spec §9): if you hit a non-obvious bug during this smoke that you fixed, append a `learn.md` entry — Symptom → Root cause → Fix → Lesson → Guardrail. No commit yet — this comes with the PR commit in Phase 7.

---

## Phase 7 — PR

### Task 7.1: Open the Phase 1 PR

**Files:** none — git workflow.

- [ ] **Step 1: Verify branch state**

```bash
git status -b --short
git log origin/main..HEAD --oneline | wc -l   # expect ~14-18 commits
```

- [ ] **Step 2: Push the branch**

```bash
git push -u origin HEAD:feat/phase1-rdr-lab-ui
```

- [ ] **Step 3: Open the PR**

```bash
gh pr create \
  --title "Phase 1: rdr lab-UI gap-close — cluster SSE events + dual-layout canvas" \
  --body "$(cat <<'EOF'
## Summary

Closes cleanup-spec §5.2.3 acceptance modulo Region 5 cluster timeline (deferred to Phase 1b).

- **Backend cluster events.** 4 new spec-shaped SSE events (`cluster_started`, `cluster_artifact_emitted`, `cluster_scored`, `repair_dispatched`) built in `backend/agents/rlm/sse_bridge.py`, emitted from the rdr controller alongside (not replacing) the existing 12 `rdr_*` events. Honesty cap (PR #75 `DEGRADED_LEAF_CEILING = 0.35`) applied per-cluster on `cluster_scored.degraded`.
- **Frontend reducer.** `RlmRunState` gains `mode`, `clusterStatus`, `clusterScores`. 4 new folds (`foldClusterStarted/ArtifactEmitted/Scored/RepairDispatched`); existing rlm folds untouched.
- **Dual-layout canvas.** `dynamicCandidateLayout` (alias for existing rlm `layoutTree`) and new `rubricClusterLayout` (fixed root/cluster/leaf bands); `ExplorationCanvas` switches via `mode` prop threaded from `runMeta.mode`.
- **NodeDetailPopup.** New `kind: "cluster"` branch — title, leaves, score, degraded flag, artifacts list.
- **Fixture + URL param.** `rdr-run.fixture.ts` with 3 clusters (pass / repair-then-pass / degraded); `?rdrFixture=1` joins `?rlmFixture=1` in lab-shell.

## Spec links

- `docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md` §5.2.3
- `docs/superpowers/specs/2026-05-23-cleanup-implementation-kickoff.md` §G (post-rebase status)

## Test plan

- [x] `.venv/bin/python -m pytest tests/ -n auto` — `pass count = baseline + ~12` (5 cluster-event builders + 4 controller-emit tests + 1 integration smoke + 2 honesty-cap regression).
- [x] `npm test` in `frontend/` — `pass count = baseline + ~16` (reducer × 5, layout × 3, canvas × 3, popup × 5, fixture × 3, lab-shell × 1).
- [x] `npx tsc --noEmit` clean.
- [x] `npx playwright test e2e/rdr-fixture.spec.ts` — Region 1/2/4/6 present, zero console errors.
- [x] Manual real-run smoke (Task 6.2) — `--mode rdr` on a small PaperBench bundle drives the lab end-to-end.

## Deferred

- Region 5 cluster timeline (Rubric Timeline + Root Reasoning for rdr) — Phase 1b or rolls into Phase 4 leaderboard work. The current canvas + popup carries enough signal for the demo; the timeline is a richer rendering of the same data.
- Frontend proxy routes for `GET /runs/{id}/clusters` / `/repair-iterations` / `/leaf-scores` (the 3 REST endpoints from PR #79's `b3a2839`). Reducer state from SSE events covers the same surface for the lab use case; these REST endpoints become useful for the leaderboard's "rerun with config" detail view.
EOF
)"
```

- [ ] **Step 4: Update `progress.md` and `CHANGELOG.md`**

In the same PR (separate commit OK):

`progress.md`: append a one-paragraph "Phase 1 — rdr lab-UI gap-close" entry summarizing scope.

`CHANGELOG.md` `[Unreleased]`: add bullets under `Added` / `Changed`.

Commit:
```bash
git add progress.md CHANGELOG.md
git commit -m "docs: progress + changelog for Phase 1 rdr lab-UI gap-close"
git push
```

- [ ] **Step 5: Tick the spec §8 acceptance criteria for Phase 1**

When the PR merges, walk through spec §8 #1–#11 and confirm any Phase-1-relevant ones pass.

---

## Appendix A — rdr controller event inventory (current state)

From the audit at `docs/superpowers/specs/2026-05-23-cleanup-implementation-kickoff.md` §G:

| Existing event name | Where emitted | Payload keys | Spec-shaped replacement / additive |
|---|---|---|---|
| `rdr_run_started` | `controller.py:372` | `project_id`, `paper_id`, `cluster_count` | kept; no spec replacement |
| `rdr_cluster_started` | `controller.py:398` | `cluster_id`, `cluster_index`, `cluster_title`, `leaf_count`, `weight`, `dominant_category` | additive: emit `cluster_started` (Task 1.2) |
| `rdr_cluster_completed` | `controller.py:433` | `cluster_id`, `cluster_index`, `failed`, `file_count` | kept; no spec replacement |
| `rdr_environment_started` | `controller.py:496` | `project_id` | kept |
| `rdr_environment_completed` | `controller.py:535` | `project_id`, `env_id` | kept |
| `rdr_experiment_started` | `controller.py:541` | `project_id` | kept |
| `rdr_experiment_completed` | `controller.py:550` | `success`, `metrics_keys` | kept |
| `rdr_scoring_started` | `controller.py:574` | `project_id` | kept |
| `rdr_scoring_completed` | `controller.py:585` | `overall_score`, `leaf_count`, `graded` | kept; additive: emit per-cluster `cluster_scored` (Task 1.4) |
| `rdr_repair_pass_started` | `controller.py:624` | `pass`, `weak_count` | kept; additive: emit per-cluster `repair_dispatched` (Task 1.5) |
| `rdr_repair_cluster_completed` | `controller.py:661` | `pass`, `cluster_id`, `failed` | kept |
| `rdr_run_completed` | `controller.py:786` | `status`, `rubric_score`, `clusters_total`, `clusters_failed`, `repair_iterations` | kept |
| _(none today)_ | _(would land in agent.py file-write site)_ | _n/a_ | NEW: emit `cluster_artifact_emitted` per file (Task 1.3) |

Line numbers from the audit; recheck via grep at execution time (they may drift if the file is edited).

---

## Appendix B — Risk catalog

1. **rdr smoke fails in Task 0.1.** Halt the plan; surface to user. This is a Phase 2 problem (honesty queue + rdr cap), not Phase 1.
2. **Existing `rdr_*` event tests assume specific payload shapes.** Tests should pass as-is since Phase 1 is additive. If a test breaks, the change isn't additive — fix the cause, don't update the test.
3. **`art.files` (Task 1.3) doesn't exist or has a different name on the controller.** Step 1 of 1.3 is a grep + read; the agent needs to adapt code blocks to the real attribute names. This is expected — the audit's controller line numbers are pinned to one snapshot.
4. **`leaf_scorer.DEGRADED_LEAF_CEILING` (Task 1.4) not directly applicable per-cluster.** If the existing aggregate `degraded` predicate doesn't decompose cleanly per-cluster, the test `test_cluster_scored_respects_degraded_cap` will need adjusting. Surface to user before improvising — this touches §3 #5 / §5.3.1 honesty territory.
5. **`?rlmFixture=1` parse logic in lab-shell.tsx (Task 5.2) lives in a different file or hook than expected.** Grep first; the audit identified `lab-shell.tsx:63-68` as the param parser, but if lab-shell.tsx imports from a hook, the change might belong there instead.
6. **Plan size (14–18 commits) exceeds reviewer appetite.** If the PR is too big, split into 1a (Phases 1–2 backend) and 1b (Phases 3–6 frontend) — both pass independently. Spec §7 risk #4 anticipates this.

---

## Self-review checklist (for the plan author, before handoff)

1. **Spec coverage.** Walk spec §5.2.3 line by line and confirm each requirement maps to a task — done above (10 requirements ↔ Phases 1–6).
2. **Placeholder scan.** Plan has 4 spots where the agent must read code to fill in attribute names (Tasks 1.2, 1.3, 1.4, 1.5). These are bounded by grep + read steps with explicit instructions — not placeholders, but "look here for the real name." Acceptable.
3. **Type consistency.** `ClusterStatus`, `RlmRunMode`, `ClusterTreeNodeData`, `LeafTreeNodeData` are defined in Tasks 2.1 / 2.2 / 4.1 and referenced consistently downstream. `rubricClusterLayout` returns `NodePlacement[]` matching the existing `layoutTree` signature.
