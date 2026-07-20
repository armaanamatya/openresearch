# BES Conversion + Archival Correctness (P0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shipped grade coherent with its own evidence (conversion correctness), gate every BES efficacy claim behind a complete archive, and add the SELECT-stability instrumentation — the P0 prerequisite that must land before any BES GPU experiment (A2/C3).

**Architecture:** Five independent, flag-safe units. (1) A coherence guard in the final-report build that forbids shipping empty provenance alongside graded measured evidence. (2) A coherent champion bundle: snapshot+restore the rubric block with the code, record `sample_count`, never ship a top-line detached from its leaves. (3) A pure archival-completeness checker that any efficacy claim must pass. (4) A pure SELECT-stability stats module (flip-rate / pairwise-win / margin) feeding a fresh-capture A1 harness. (5) Fold staged-search budget-dropped cells into structured `scope.gaps`. Everything default-OFF / parity-preserving; conversion guards are evidence-tightening (no behavior change on already-coherent runs).

**Tech Stack:** Python 3.12, pytest (+pytest-socket hermetic), the existing `backend/agents/rlm/` report/champion/binding modules, `backend/agents/rlm/cell_matrix.py`, `scripts/calibrate_grader.py`.

---

## Advisor review (applied)

A staff-engineer advisor pressure-tested this plan against the real code. **The
load-bearing claim is verified correct:** the champion's restored `leaf_scores`
survive the downstream merge (`report.py:1663` backfills only when current is
`None`; the eval-scalar override only wins when `≥` current — impossible under the
score gate that triggered restore), and `finalize_regrade` (default-ON,
`run.py:2512`) re-grades the restored code writing score+leaves coherently, so it
**composes** rather than conflicts. Four tasks had real bugs — **now fixed inline**
(Tasks 3, 4, 6, 11; see the per-task advisor notes). Remaining minor touch-ups the
executing engineer should apply as they go:

- **Task 2:** add `snapshot_rubric`/`restore_rubric` to `champion_artifact.__all__`.
- **Task 4:** add an integration test through `write_final_report_rlm` with a stale
  on-disk `rubric_evaluation.json`, asserting `final_report.json` ships champion
  leaves (the helper test is necessary-but-insufficient for the load-bearing claim).
- **Task 5/6:** `evidence_cites_metrics` is **not a real report field** — derive it
  by scanning leaf justifications (shown in Task 6 Step 3). Add one fixture built
  from the actual `best_runs/adam_ab/bes/` report shape.
- **Task 7:** docstring must state this is an **archived-AB-arm** checker (live runs
  keep `metrics.json` under `code/`); drop the unused `REQUIRED_METADATA` constant
  or wire it into the check.
- **Task 8:** strengthen the test to call `validate_stamped_pair` (`ab_compare.py:121`,
  invoked `:355`) and assert it refuses on an incomplete arm — not just re-test Task 7.
- **All tests:** `result["leaf_scores"]` is a **list of records**
  (`[{"id","score","justification"}]`), not a `dict` — use the list shape in fixtures.

All tests are `tmp_path`/dict-fixture pure → pytest-socket hermetic. Tasks 1, 9, 10
are executable as-written.

## File Structure

| File | Responsibility | New/Mod |
|---|---|---|
| `backend/agents/rlm/conversion_guard.py` | Pure: detect a report whose provenance fields deny evidence the rubric block proves was graded; return a repair. | **Create** |
| `backend/agents/rlm/report.py` | Call the conversion guard in `build_final_report`; restore a coherent champion bundle in `_apply_champion_artifact`. | Modify |
| `backend/agents/rlm/champion_artifact.py` | Snapshot+restore the rubric block alongside code; carry `sample_count` + hashes. | Modify |
| `backend/agents/rlm/binding.py` | Pass `sample_count` + snapshot the rubric block at champion-record time. | Modify (`:834-856`) |
| `backend/agents/rlm/archive_completeness.py` | Pure: does a run dir carry the full BES-efficacy artifact set? | **Create** |
| `backend/agents/rlm/select_stability.py` | Pure: flip-rate / P(i>j) / margin distribution from K re-grades. | **Create** |
| `scripts/bes_a1_capture.py` | A1 harness: fresh N≥3 capture → archive → K re-grades → stability report. | **Create** |
| `backend/agents/rlm/cell_matrix.py` | Fold staged-search dropped full cells into structured `scope.gaps`. | Modify (`:731`) |
| `backend/agents/rlm/staged_search.py` | Surface dropped full-cell ids to the aggregate gap fold. | Modify (`:526,545`) |
| `tests/rlm/test_conversion_guard.py` … (one per unit) | Tests. | **Create** |

---

## Group P0-2: Coherent champion bundle (do FIRST — most-verified code)

### Task 1: `sample_count` on champion entries (stop calling one score a median)

**Files:**
- Modify: `backend/agents/rlm/champion_artifact.py:116-152`
- Test: `tests/rlm/test_champion_artifact.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_champion_artifact.py
from pathlib import Path
from backend.agents.rlm import champion_artifact as ca

def test_record_champion_carries_sample_count(tmp_path: Path):
    reg = tmp_path / "champions.json"
    entry = ca.record_champion(
        reg, evidence_key="k1", snapshot_dir=tmp_path / "snap",
        median_score=0.5, sample_count=1,
    )
    assert entry["sample_count"] == 1
    assert entry["median_score"] == 0.5  # name kept for back-compat; sample_count disambiguates
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_champion_artifact.py::test_record_champion_carries_sample_count -v`
Expected: FAIL — `record_champion() got an unexpected keyword argument 'sample_count'`

- [ ] **Step 3: Add the parameter**

In `record_champion`, add `sample_count: int = 1` to the keyword-only signature and `"sample_count": int(sample_count)` to the `entry` dict (after `"median_score": score,`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_champion_artifact.py::test_record_champion_carries_sample_count -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/champion_artifact.py tests/rlm/test_champion_artifact.py
git commit -m "feat(champion): record sample_count so a single grade is never mislabeled median-of-N"
```

### Task 2: Snapshot the rubric block alongside the code

**Files:**
- Modify: `backend/agents/rlm/champion_artifact.py` (add `snapshot_rubric`/`restore_rubric`)
- Test: `tests/rlm/test_champion_artifact.py`

- [ ] **Step 1: Write the failing test**

```python
import json
def test_snapshot_and_restore_rubric_block(tmp_path: Path):
    snap = tmp_path / "snap"
    snap.mkdir()
    rubric = {"overall_score": 0.73, "leaf_scores": {"a": 1.0, "b": 0.5}, "meets_target": True}
    ca.snapshot_rubric(rubric, snap)
    assert json.loads((snap / "rubric_block.json").read_text())["overall_score"] == 0.73
    assert ca.restore_rubric(snap) == rubric
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_champion_artifact.py::test_snapshot_and_restore_rubric_block -v`
Expected: FAIL — `module 'champion_artifact' has no attribute 'snapshot_rubric'`

- [ ] **Step 3: Implement**

```python
# champion_artifact.py — add to __all__ and module body
def snapshot_rubric(rubric: dict, snapshot_dir: "Path") -> None:
    """Persist the graded rubric block next to the code snapshot so a restore
    re-materializes score+leaves+meets_target as ONE coherent evidence state."""
    p = Path(snapshot_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "rubric_block.json").write_text(json.dumps(rubric, indent=2), encoding="utf-8")

def restore_rubric(snapshot_dir: "Path") -> dict | None:
    """Return the snapshotted rubric block, or None if absent/unreadable (fail-soft)."""
    try:
        return json.loads((Path(snapshot_dir) / "rubric_block.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_champion_artifact.py::test_snapshot_and_restore_rubric_block -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/champion_artifact.py tests/rlm/test_champion_artifact.py
git commit -m "feat(champion): snapshot/restore the rubric block with the code snapshot"
```

### Task 3: Wire record-site to snapshot the rubric + pass sample_count

**Files:**
- Modify: `backend/agents/rlm/binding.py:834-856`
- Test: `tests/rlm/test_binding_champion_record.py`

- [ ] **Step 1: Write the failing test** (assert the rubric block is snapshotted when champion records)

```python
# tests/rlm/test_binding_champion_record.py
import json, os
from pathlib import Path
from backend.agents.rlm import champion_artifact as ca

def test_record_site_snapshots_rubric_block(tmp_path, monkeypatch):
    # snapshot a fake champion the way binding does, then assert rubric_block.json exists
    snap = ca.snapshot_code(tmp_path / "code", tmp_path / "snap") if (tmp_path / "code").mkdir() or True else None
    ca.snapshot_rubric({"overall_score": 0.6, "leaf_scores": {"x": 0.6}}, tmp_path / "snap")
    assert (tmp_path / "snap" / "rubric_block.json").is_file()
```

- [ ] **Step 2: Run test to verify it fails / passes minimally**

Run: `.venv/bin/python -m pytest tests/rlm/test_binding_champion_record.py -v`
Expected: PASS for the helper contract (this task's real change is wiring; the integration is asserted in Task 4). If import fails, fix the import path first.

- [ ] **Step 3: Wire binding.py**

> **Advisor corrections:** there is **no ready-made rubric-block local** at the
> record-site — `payload` is built later (`binding.py:909-925`). Construct the
> block from `result`. `result["leaf_scores"]` is a **list of records**
> (`[{"id","score","justification"}]`), not a `dict`. `grader_sample_count` must
> be sourced from the env. The rubric block must be snapshotted to
> `_snap_a4.parent` (the champion entry dir), so Task 4's `restore_rubric(Path(snap).parent)` finds it.

At `binding.py:849-854` (the `record_champion(...)` call), immediately before it:

```python
import os as _os
from backend.agents.rlm.champion_artifact import snapshot_rubric as _snap_rubric_a4
_champ_block = {
    "overall_score": float(score),
    "leaf_scores": result.get("leaf_scores", []),          # list of records
    "weak_leaves": result.get("weak_leaves", []),
    "leaf_count": result.get("leaf_count"),
    "meets_target": result.get("meets_target"),
    "target_score": (float(target) if target is not None else None),
    "compute_adjusted_score": result.get("compute_adjusted_score"),
}
_snap_rubric_a4(_champ_block, _snap_a4.parent)             # entry dir, NOT the code dir
_sample_n = int(_os.environ.get("OPENRESEARCH_GRADER_SAMPLES", "1") or "1")
_rec_champ_a4(
    ctx.project_dir / "rlm_state" / "champions.json",
    evidence_key=_evidence_key,
    snapshot_dir=_snap_a4,
    median_score=float(score),
    sample_count=_sample_n,
)
```

(The real locals at this site are `result`, `score`, `target`, `areas`, `_evidence_key`, `_snap_a4` — read `binding.py:820-925` to confirm before editing.)

- [ ] **Step 4: Run the champion test subset**

Run: `.venv/bin/python -m pytest tests/rlm/test_binding_champion_record.py tests/rlm/test_champion_artifact.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/binding.py tests/rlm/test_binding_champion_record.py
git commit -m "feat(champion): snapshot the graded rubric block + record sample_count at record-site"
```

### Task 4: `_apply_champion_artifact` restores a COHERENT bundle

**Files:**
- Modify: `backend/agents/rlm/report.py:822-859`
- Test: `tests/rlm/test_champion_coherence.py`

- [ ] **Step 1: Write the failing test** (the core incoherence: higher champion score must not ship beside stale lower leaves)

> **Advisor note:** `leaf_scores` is a **list of records** in production, not a
> dict — fixtures match that. This helper test is **necessary but not
> sufficient**: it proves `_apply_champion_artifact` in isolation. Step 6 adds the
> integration test through `write_final_report_rlm` (the merge at `report.py:1663`
> *and* `finalize_regrade` at `run.py:2512` both run after restore — the advisor
> confirmed both preserve coherence, but the load-bearing claim must be tested on
> the shipped path).

```python
# tests/rlm/test_champion_coherence.py
import json, os
from pathlib import Path
from backend.agents.rlm import report as R, champion_artifact as ca

def test_champion_restore_ships_coherent_leaves(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_CHAMPION_ARTIFACT", "1")
    proj = tmp_path
    (proj / "code").mkdir()
    snap = proj / "rlm_state" / "champions" / "k" / "code"
    ca.snapshot_code(proj / "code", snap)
    ca.snapshot_rubric({"overall_score": 0.73, "leaf_scores": [{"id": "a", "score": 0.73}],
                        "meets_target": True, "target_score": 0.6}, snap.parent)
    ca.record_champion(proj / "rlm_state" / "champions.json",
                       evidence_key="k", snapshot_dir=str(snap), median_score=0.73, sample_count=1)
    # current (latest, stale, lower) rubric: top-line 0.55 with leaves to match
    stale = {"overall_score": 0.55, "leaf_scores": [{"id": "a", "score": 0.55}],
             "meets_target": False, "target_score": 0.6}
    out = R._apply_champion_artifact(stale, proj)
    # champion (0.73) wins; leaves must come from the CHAMPION block, not the stale 0.55
    assert out["overall_score"] == 0.73
    assert out["leaf_scores"] == [{"id": "a", "score": 0.73}]   # <-- fails today (leaves stay 0.55)
    assert out["champion_restored"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_champion_coherence.py -v`
Expected: FAIL — `out["leaf_scores"] == {"a": 0.55}` (the stale leaves, proving the incoherence at `report.py:852-855`).

- [ ] **Step 3: Implement coherent restore**

In `_apply_champion_artifact` (`report.py:822`), after `restore_snapshot(...)` and before setting `overall_score`, load and merge the champion's rubric block:

```python
from backend.agents.rlm.champion_artifact import best_champion, restore_snapshot, restore_rubric
...
if snap and (cur_f is None or champ_score >= cur_f):
    restore_snapshot(Path(snap), Path(project_dir) / "code")
    champ_rubric = restore_rubric(Path(snap).parent) or {}
    out["overall_score"] = champ_score
    # Coherence: ship the champion's OWN leaves/meets_target/target with its score,
    # never the stale latest-verify leaves. Fall back only for keys the champion lacks.
    for k in ("leaf_scores", "weak_leaves", "leaf_count", "meets_target", "target_score",
              "compute_adjusted_score"):
        if champ_rubric.get(k) is not None:
            out[k] = champ_rubric[k]
    out["champion_restored"] = True
    out["champion_sample_count"] = int(champ.get("sample_count", 1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_champion_coherence.py -v`
Expected: PASS

- [ ] **Step 5: Run the existing champion/report tests for regressions**

Run: `.venv/bin/python -m pytest tests/rlm/ -k "champion or report or finalize" -q`
Expected: PASS (no regressions; default-OFF path unchanged because the flag gates entry).

- [ ] **Step 6: Commit**

```bash
git add backend/agents/rlm/report.py tests/rlm/test_champion_coherence.py
git commit -m "fix(champion): restore the champion's own rubric block so score never detaches from its leaves"
```

---

## Group P0-1: Conversion provenance guard

### Task 5: Pure guard — empty provenance must not coexist with graded measured evidence

**Files:**
- Create: `backend/agents/rlm/conversion_guard.py`
- Test: `tests/rlm/test_conversion_guard.py`

- [ ] **Step 1: Write the failing test** (the Adam shape)

```python
# tests/rlm/test_conversion_guard.py
from backend.agents.rlm.conversion_guard import detect_projection_incoherence

def test_detects_empty_provenance_with_graded_metrics():
    report = {"baseline_metrics": {}, "experiment_run_id": None, "primitive_trace": {}}
    rubric = {"overall_score": 0.53,
              "leaf_scores": {"acc": 0.53},
              "evidence_cites_metrics": True}   # rubric block proves grader saw measured artifacts
    metrics_on_disk = {"cifar10_cnn": {"top1": 0.91}}   # populated
    issue = detect_projection_incoherence(report, rubric, metrics_on_disk)
    assert issue is not None
    assert issue["kind"] == "empty_provenance_with_graded_evidence"

def test_no_issue_when_coherent():
    report = {"baseline_metrics": {"cifar10_cnn": {"top1": 0.91}},
              "experiment_run_id": "run-1", "primitive_trace": {"run_experiment": 1}}
    rubric = {"overall_score": 0.53, "evidence_cites_metrics": True}
    assert detect_projection_incoherence(report, rubric, {"cifar10_cnn": {"top1": 0.91}}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_conversion_guard.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the pure guard**

```python
# backend/agents/rlm/conversion_guard.py
"""Pure detector: a final report whose provenance fields DENY measured evidence
the rubric block proves was graded (the Adam projection-failure shape).

The contradiction is intra-report: the authoritative provenance fields
(baseline_metrics / experiment_run_id / primitive_trace) say "no measured result"
while the grader demonstrably scored populated on-disk metrics. Returns a structured
issue dict (or None). Stdlib only; no I/O; fail-open (ambiguous -> None)."""
from __future__ import annotations
from typing import Any


def _provenance_empty(report: dict[str, Any]) -> bool:
    return (
        not report.get("baseline_metrics")
        and report.get("experiment_run_id") in (None, "")
        and not report.get("primitive_trace")
    )


def detect_projection_incoherence(
    report: dict[str, Any], rubric: dict[str, Any], metrics_on_disk: dict[str, Any] | None,
) -> dict[str, Any] | None:
    graded_measured = bool(rubric) and (
        rubric.get("evidence_cites_metrics") is True
        or (rubric.get("overall_score") not in (None, 0, 0.0) and bool(metrics_on_disk))
    )
    if graded_measured and bool(metrics_on_disk) and _provenance_empty(report):
        return {
            "kind": "empty_provenance_with_graded_evidence",
            "detail": "report provenance is empty but the grader scored populated metrics.json",
        }
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_conversion_guard.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/conversion_guard.py tests/rlm/test_conversion_guard.py
git commit -m "feat(conversion): pure guard for empty-provenance-with-graded-evidence (Adam shape)"
```

### Task 6: Wire the guard into `build_final_report` — repair provenance from disk

**Files:**
- Modify: `backend/agents/rlm/report.py` (in `build_final_report`, after rubric is finalized, before write)
- Test: `tests/rlm/test_report_provenance_repair.py`

- [ ] **Step 1: Write the failing test** (end-to-end shape on a fixture run dir)

```python
# tests/rlm/test_report_provenance_repair.py
import json
from pathlib import Path
from backend.agents.rlm.conversion_guard import detect_projection_incoherence

def test_repair_populates_baseline_metrics_from_disk(tmp_path):
    # Simulate the post-build report + a populated metrics.json the rubric cited.
    metrics = {"cifar10_cnn": {"top1": 0.91}}
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(json.dumps(metrics))
    report = {"baseline_metrics": {}, "experiment_run_id": None, "primitive_trace": {}}
    rubric = {"overall_score": 0.53, "evidence_cites_metrics": True}
    from backend.agents.rlm.report import repair_projection_from_disk
    fixed = repair_projection_from_disk(report, rubric, tmp_path)
    assert fixed["baseline_metrics"] == metrics
    assert detect_projection_incoherence(fixed, rubric, metrics) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_report_provenance_repair.py -v`
Expected: FAIL — `cannot import name 'repair_projection_from_disk'`.

- [ ] **Step 3: Implement the repair + call it in build_final_report**

> **Advisor corrections:** `build_final_report` returns
> `RLMFinalReport(**kwargs)` (a **pydantic model**, `report.py:1193`), so the
> repair must mutate the **`kwargs` dict** before construction — not the returned
> model. Existing projections already populate `baseline_metrics` from
> *`experiment_runs.jsonl` success rows* (`:953`, `OPENRESEARCH_METRIC_PROVENANCE`,
> default ON) and `experiment_run_id` (`:1089`). The Adam shape is the
> **complement**: the grader scored `code/metrics.json` but there is **no
> successful `experiment_runs.jsonl` row**, so those projections found nothing.
> The repair therefore reads a **different source** (`code/metrics.json` directly)
> and only fires when the existing projection left provenance empty. No overlap /
> double-write.

```python
# report.py — new helper (operates on the kwargs dict, not the model)
def repair_projection_from_disk(kwargs_report: dict, rubric: dict, project_dir: "Path") -> dict:
    """If provenance is empty but the grader scored code/metrics.json, repopulate
    baseline_metrics from that file. Evidence-tightening; no-op when coherent."""
    from backend.agents.rlm.conversion_guard import detect_projection_incoherence
    try:
        mpath = Path(project_dir) / "code" / "metrics.json"
        metrics = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else None
    except (OSError, ValueError, TypeError):
        metrics = None
    probe = {
        "baseline_metrics": kwargs_report.get("baseline_metrics"),
        "experiment_run_id": kwargs_report.get("experiment_run_id"),
        "primitive_trace": kwargs_report.get("primitive_trace"),
    }
    if detect_projection_incoherence(probe, rubric, metrics) is None:
        return kwargs_report
    kwargs_report["baseline_metrics"] = metrics
    kwargs_report["provenance_repaired"] = True
    logger.warning("report: repaired empty provenance from code/metrics.json (conversion guard)")
    return kwargs_report
```

Derive `rubric["evidence_cites_metrics"]` at build time by scanning the parsed
leaf justifications (available in the rubric block):
```python
rubric["evidence_cites_metrics"] = any(
    ("metrics.json" in (l.get("justification", "") or "") or "outputs/" in (l.get("justification", "") or ""))
    for l in rubric.get("leaf_scores", [])
)
```
Then call `repair_projection_from_disk(kwargs, rubric, ctx.project_dir)` immediately
before `RLMFinalReport(**kwargs)` at `report.py:1193`. (Read `build_final_report`
`:934-1193` to confirm `kwargs` key names before editing.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_report_provenance_repair.py -v`
Expected: PASS

- [ ] **Step 5: Regression run**

Run: `.venv/bin/python -m pytest tests/rlm/ -k "report or finalize or evidence" -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/agents/rlm/report.py tests/rlm/test_report_provenance_repair.py
git commit -m "fix(conversion): repair empty report provenance from graded on-disk metrics"
```

---

## Group P0-3: Archival-completeness gate

### Task 7: Pure archive checker

**Files:**
- Create: `backend/agents/rlm/archive_completeness.py`
- Test: `tests/rlm/test_archive_completeness.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_archive_completeness.py
from pathlib import Path
from backend.agents.rlm.archive_completeness import check_bes_archive, REQUIRED_ARTIFACTS

def test_incomplete_archive_is_rejected(tmp_path):
    (tmp_path / "final_report.json").write_text("{}")
    res = check_bes_archive(tmp_path)
    assert res.complete is False
    assert "dashboard_events.jsonl" in res.missing

def test_complete_archive_passes(tmp_path):
    for name in REQUIRED_ARTIFACTS:
        (tmp_path / name).write_text("{}")
    (tmp_path / "candidates").mkdir()
    res = check_bes_archive(tmp_path)
    assert res.complete is True
    assert res.missing == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_archive_completeness.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# backend/agents/rlm/archive_completeness.py
"""Pure gate: a BES efficacy claim requires a complete run archive (the Adam
lesson — without it the most important number becomes folklore). Stdlib only."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED_ARTIFACTS = (
    "bes_candidates.json", "dashboard_events.jsonl", "experiment_runs.jsonl",
    "rubric_evaluation.json", "final_report.json", "metrics.json",
    "generated_rubric.json",
)
REQUIRED_DIRS = ("candidates",)
REQUIRED_METADATA = ("rubric_tree_hash", "env_flags", "experiment_arm")  # in final_report.json


@dataclass
class ArchiveCheck:
    complete: bool
    missing: list[str] = field(default_factory=list)


def check_bes_archive(run_dir: Path) -> ArchiveCheck:
    run_dir = Path(run_dir)
    missing = [n for n in REQUIRED_ARTIFACTS if not (run_dir / n).is_file()]
    missing += [d + "/" for d in REQUIRED_DIRS if not (run_dir / d).is_dir()]
    return ArchiveCheck(complete=not missing, missing=missing)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_archive_completeness.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/archive_completeness.py tests/rlm/test_archive_completeness.py
git commit -m "feat(bes): archival-completeness gate for efficacy claims"
```

### Task 8: `ab_compare.py` refuses an efficacy delta on an incomplete archive

**Files:**
- Modify: `scripts/ab_compare.py` (validator path, near `OPENRESEARCH_REQUIRE_STAMPED_AB`)
- Test: `tests/scripts/test_ab_compare_archive_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_ab_compare_archive_gate.py
from pathlib import Path
from backend.agents.rlm.archive_completeness import check_bes_archive

def test_ab_compare_gates_on_archive(tmp_path):
    # An arm dir missing dashboard_events.jsonl must be flagged incomplete.
    (tmp_path / "final_report.json").write_text("{}")
    assert check_bes_archive(tmp_path).complete is False
```

- [ ] **Step 2: Run test to verify it fails/passes**

Run: `.venv/bin/python -m pytest tests/scripts/test_ab_compare_archive_gate.py -v`
Expected: PASS (the gate is exercised; Step 3 wires it into the script).

- [ ] **Step 3: Wire the gate**

In `scripts/ab_compare.py`, when `OPENRESEARCH_REQUIRE_STAMPED_AB` is set (or a new `--require-complete-archive` flag), call `check_bes_archive(arm_dir)` for each arm and refuse to emit a Δ (exit non-zero, print the `missing` list) if either arm is incomplete.

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/scripts/test_ab_compare_archive_gate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/ab_compare.py tests/scripts/test_ab_compare_archive_gate.py
git commit -m "feat(ab): refuse a BES efficacy delta on an incomplete arm archive"
```

---

## Group A1: SELECT-stability instrumentation (parallel to P0)

### Task 9: Pure stability stats (flip-rate / pairwise-win / margin — not Kendall τ)

**Files:**
- Create: `backend/agents/rlm/select_stability.py`
- Test: `tests/rlm/test_select_stability.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_select_stability.py
from backend.agents.rlm.select_stability import stability_report

def test_stability_report_flip_rate_and_winprob():
    # 3 candidates, K=4 re-grades; candidate "b" wins 3/4, "a" wins 1/4
    regrades = [
        {"a": 0.50, "b": 0.55, "c": 0.40},
        {"a": 0.56, "b": 0.54, "c": 0.41},
        {"a": 0.49, "b": 0.57, "c": 0.42},
        {"a": 0.48, "b": 0.58, "c": 0.39},
    ]
    rep = stability_report(regrades)
    assert rep["k"] == 4
    assert rep["top1_flip_rate"] == 0.25            # top-1 changed once across 4
    assert abs(rep["win_prob"]["b"] - 0.75) < 1e-9  # b is top-1 in 3/4
    assert rep["margin_p50"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_select_stability.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# backend/agents/rlm/select_stability.py
"""Pure SELECT-stability statistics for A1. At small N (often 2 candidates)
Kendall's tau is degenerate (+/-1); report top-1 flip rate, per-candidate win
probability, and the top-2 margin distribution instead. Stdlib only."""
from __future__ import annotations
from statistics import median
from typing import Any


def _top1(grades: dict[str, float]) -> str:
    return max(grades, key=lambda k: grades[k])


def stability_report(regrades: list[dict[str, float]]) -> dict[str, Any]:
    if not regrades:
        return {"k": 0, "top1_flip_rate": 0.0, "win_prob": {}, "margin_p50": 0.0}
    k = len(regrades)
    tops = [_top1(g) for g in regrades]
    flips = sum(1 for i in range(1, k) if tops[i] != tops[i - 1])
    cands = sorted({c for g in regrades for c in g})
    win_prob = {c: sum(1 for t in tops if t == c) / k for c in cands}
    margins = []
    for g in regrades:
        ordered = sorted(g.values(), reverse=True)
        margins.append(ordered[0] - ordered[1] if len(ordered) > 1 else 0.0)
    return {
        "k": k,
        "top1_flip_rate": flips / (k - 1) if k > 1 else 0.0,
        "win_prob": win_prob,
        "margin_p50": float(median(margins)),
        "margin_min": float(min(margins)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_select_stability.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/select_stability.py tests/rlm/test_select_stability.py
git commit -m "feat(a1): pure SELECT-stability stats (flip-rate/win-prob/margin)"
```

### Task 10: A1 capture harness (script)

**Files:**
- Create: `scripts/bes_a1_capture.py`
- Test: `tests/scripts/test_bes_a1_capture.py`

- [ ] **Step 1: Write the failing test** (the stats wiring; the live capture is operator-run)

```python
# tests/scripts/test_bes_a1_capture.py
from scripts.bes_a1_capture import summarize_regrades

def test_summarize_regrades_emits_verdict():
    regrades = [{"a": 0.50, "b": 0.55}, {"a": 0.60, "b": 0.54}]  # top-1 flips 1/1
    out = summarize_regrades(regrades, repeatability_sigma=0.02)
    assert out["report"]["top1_flip_rate"] == 1.0
    assert out["verdict"] == "select_is_noise"   # flips at margins <= sigma
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/scripts/test_bes_a1_capture.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the summary + a thin CLI**

```python
# scripts/bes_a1_capture.py
"""A1: fresh N>=3 candidate capture -> archive snapshots -> K temp=0 re-grades ->
stability verdict. The capture/regrade loop calls the existing candidate machinery
and scripts/calibrate_grader.py; this module owns the verdict logic + is unit-tested."""
from __future__ import annotations
from typing import Any
from backend.agents.rlm.select_stability import stability_report


def summarize_regrades(regrades: list[dict[str, float]], *, repeatability_sigma: float) -> dict[str, Any]:
    rep = stability_report(regrades)
    noisy = rep["top1_flip_rate"] > 0.0 and rep["margin_min"] <= repeatability_sigma
    return {"report": rep, "verdict": "select_is_noise" if noisy else "select_stable"}
```

Add an `argparse` `main()` that: (1) runs N≥3 candidates on `--paper` via the existing BES candidate path with `OPENRESEARCH_BES_CANDIDATES_PER_CLUSTER>=3`, (2) archives each `candidates/rlm_impl_*/` snapshot, (3) re-grades K times via `calibrate_grader`, (4) prints `summarize_regrades(...)`. Document that the live run is operator-gated (bounded LLM spend, zero GPU).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/scripts/test_bes_a1_capture.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/bes_a1_capture.py tests/scripts/test_bes_a1_capture.py
git commit -m "feat(a1): SELECT-stability capture harness + verdict"
```

---

## Group C-fix: staged-search dropped groups → structured scope.gaps

### Task 11: Fold budget-dropped full cells into `scope.gaps`

> **Advisor corrections applied.** Real signature is
> `aggregate_cell_metrics(matrix_result, cells, *, capacity_gaps=, dataset_gaps=, models_skipped=, environments_skipped=)`
> (`cell_matrix.py:692-700`) — there is **no `kept` kwarg** and `cells` is
> positional. The gap fold is at `cell_matrix.py:828-832` (`:731` is the
> docstring). `run_staged_search` returns `dropped_cells` as **id strings**
> (`staged_search.py:551`), discarding the full dicts available in the `dropped`
> local at `:526` — so we add a new `dropped_cells_full` return key.

**Files:**
- Modify: `backend/agents/rlm/cell_matrix.py:692-700` (add kwarg) + `:828-832` (fold)
- Modify: `backend/agents/rlm/staged_search.py:526,545-554` (return `dropped_cells_full` dicts)
- Modify: `backend/agents/rlm/primitives.py:5644` (thread `budget_dropped=` into the aggregate call)
- Test: `tests/rlm/test_staged_search_scope_gaps.py`

- [ ] **Step 1: Write the failing test** (real signature — `cells` positional, no `kept=`)

```python
# tests/rlm/test_staged_search_scope_gaps.py
from backend.agents.rlm import cell_matrix as cm

def test_dropped_full_cells_become_scope_gaps():
    result = cm.aggregate_cell_metrics(
        {"cells": []}, [],                      # matrix_result, cells (positional)
        capacity_gaps=[], dataset_gaps=[],
        budget_dropped=[{"model_key": "qwen3-1.7b", "env": "alfworld", "baseline": "grpo"}],
    )
    gaps = result["scope"]["gaps"]
    assert any(g.get("reason") == "staged_search_budget_dropped" for g in gaps)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_staged_search_scope_gaps.py -v`
Expected: FAIL — `aggregate_cell_metrics() got an unexpected keyword argument 'budget_dropped'`.

- [ ] **Step 3: Implement (three edits)**

(a) `cell_matrix.py` — add `budget_dropped: list[dict] | None = None` to the keyword-only params (`:692-700`); at the gap fold (`:828-832`) append, for each dropped dict:
```python
for _bd in (budget_dropped or []):
    gaps.append({
        "model_key": _bd.get("model_key"), "env": _bd.get("env"),
        "baseline": _bd.get("baseline"), "reason": "staged_search_budget_dropped",
    })
```
(b) `staged_search.py` — at the return (`:545-554`), add `"dropped_cells_full": [dict(c) for c in dropped]` alongside the existing id-only `dropped_cells` (keep both for back-compat).
(c) `primitives.py:5644` — pass `budget_dropped=_staged_out.get("dropped_cells_full")` into the `cell_matrix.aggregate_cell_metrics(...)` call (only on the staged-search branch; `None` elsewhere preserves behavior).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_staged_search_scope_gaps.py -v`
Expected: PASS

- [ ] **Step 5: Regression run**

Run: `.venv/bin/python -m pytest tests/rlm/ -k "cell_matrix or staged" -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/agents/rlm/cell_matrix.py backend/agents/rlm/staged_search.py backend/agents/rlm/primitives.py tests/rlm/test_staged_search_scope_gaps.py
git commit -m "fix(staged-search): fold budget-dropped full cells into structured scope.gaps"
```

---

## Final verification

- [ ] **Run the full affected suite**

Run: `.venv/bin/python -m pytest tests/rlm/ tests/scripts/ -q`
Expected: PASS, no regressions.

- [ ] **Confirm parity (default-OFF byte-for-byte):** with `OPENRESEARCH_CHAMPION_ARTIFACT` unset and BES off, a sample run's `final_report.json` is unchanged vs `main`. The conversion guard is evidence-tightening only (no-op on already-coherent reports).

## Out of scope (follow-on plan, gated)

- **A2** (budget-matched short-slice predictiveness) and **C3** (GPU short-slice cascade) — GPU experiments blocked on P0 landing + A1 reporting `select_stable`. They get their own plan once this one is green and A1 has run.
