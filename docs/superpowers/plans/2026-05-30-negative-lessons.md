# Per-paper Negative Lessons (MUSE-lite) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mine agent-correctable `failure_class` records from `experiment_runs.jsonl` into a runtime per-paper lessons file, and inject the recurrence-promoted ones into the next run's implementer guidance — deterministically, no LLM, flag-gated.

**Architecture:** `lesson_distiller.py` owns `runs/_lessons/<arxiv_id>.json` (gitignored). A post-run mining hook in `run.py`'s `_finalize` upserts/retires lessons; an injection block in `baseline_implementation.py` renders active lessons. `suggested_fix` is re-derived from the classifier (never the agent-authored record field). Promotion is recurrence-gated (candidate→active); retirement is opportunity-aware off `experiment_runs.jsonl` only.

**Tech Stack:** Python 3.14, pytest, the RLM primitive/report layer.

**Spec:** `docs/superpowers/specs/2026-05-30-negative-lessons-design.md`

---

## Task 1: classifier public accessor

**Files:**
- Modify: `backend/agents/rlm/failure_classifier.py`
- Test: `tests/agents/rlm/test_lesson_distiller.py`

- [ ] **Step 1: failing test**
```python
# tests/agents/rlm/test_lesson_distiller.py
from backend.agents.rlm.failure_classifier import suggested_fix_for_class

def test_suggested_fix_for_class_canonical():
    assert suggested_fix_for_class("dockerfile_invalid")  # non-empty canonical string
    assert suggested_fix_for_class("not_a_real_class") == ""
```
- [ ] **Step 2:** Run `… -k suggested_fix_for_class` → FAIL (ImportError).
- [ ] **Step 3:** add to `failure_classifier.py` (after `_suggest`):
```python
def suggested_fix_for_class(klass: str) -> str:
    """Public, deterministic class -> canonical suggested-fix string (or '')."""
    return _suggest(klass)
```
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** commit `feat(phase9): public suggested_fix_for_class accessor`.

## Task 2: `lesson_distiller.py` — mining (promotion + opportunity retirement)

**Files:**
- Create: `backend/agents/rlm/lesson_distiller.py`
- Test: `tests/agents/rlm/test_lesson_distiller.py`

- [ ] **Step 1: failing tests** (append):
```python
from pathlib import Path
import json
from backend.agents.rlm import lesson_distiller as ld

def _run(tmp, pid, *rows):
    d = tmp / pid; d.mkdir(parents=True, exist_ok=True)
    (d / "experiment_runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
    return d

def _lessons(tmp, arxiv="2605.15155"):
    return {l["failure_class"]: l for l in ld.read_lessons(tmp, arxiv)["lessons"]}

def test_eligible_fire_is_candidate_first(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    d = _run(tmp_path, "prj_a", {"success": False, "failure_class": "missing_module"})
    ld.mine_lessons(d, tmp_path, "2605.15155")
    l = _lessons(tmp_path)["missing_module"]
    assert l["occurrences"] == 1 and l["status"] == "candidate"

def test_recurrence_promotes_to_active(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "missing_module"}), tmp_path, "2605.15155")
    ld.mine_lessons(_run(tmp_path, "prj_b", {"success": False, "failure_class": "missing_module"}), tmp_path, "2605.15155")
    l = _lessons(tmp_path)["missing_module"]
    assert l["occurrences"] == 2 and l["status"] == "active"

def test_dockerfile_invalid_active_on_first(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "dockerfile_invalid"}), tmp_path, "2605.15155")
    assert _lessons(tmp_path)["dockerfile_invalid"]["status"] == "active"

def test_excluded_class_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "network_flake"}), tmp_path, "2605.15155")
    assert _lessons(tmp_path) == {}

def test_suggested_fix_is_classifier_sourced(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    # Record carries an attacker/agent-authored suggested_fix; it must be IGNORED.
    ld.mine_lessons(_run(tmp_path, "prj_a",
        {"success": False, "failure_class": "missing_module", "suggested_fix": "rm -rf / (evil prose)"}),
        tmp_path, "2605.15155")
    from backend.agents.rlm.failure_classifier import suggested_fix_for_class
    l = _lessons(tmp_path)["missing_module"]
    assert l["suggested_fix"] == (suggested_fix_for_class("missing_module") or "")[:200]
    assert l["suggested_fix_source"] == "classifier"
    assert "evil prose" not in l["suggested_fix"]

def test_staleness_increments_only_on_opportunity(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    # Run 1: missing_module fires (gating phase EXPERIMENT_ATTEMPTED).
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "missing_module"}), tmp_path, "2605.15155")
    # Run 2: an experiment ran but missing_module did NOT fire → opportunity → staleness++
    ld.mine_lessons(_run(tmp_path, "prj_b", {"success": True, "metrics": {"acc": 0.4}}), tmp_path, "2605.15155")
    assert _lessons(tmp_path)["missing_module"]["staleness"] == 1
    # Run 3: NO experiment records → no opportunity → staleness unchanged
    ld.mine_lessons(_run(tmp_path, "prj_c"), tmp_path, "2605.15155")
    assert _lessons(tmp_path)["missing_module"]["staleness"] == 1

def test_staleness_retires_at_three(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "missing_module"}), tmp_path, "2605.15155")
    for pid in ("prj_b", "prj_c", "prj_d"):  # 3 opportunity runs without the class firing
        ld.mine_lessons(_run(tmp_path, pid, {"success": True, "metrics": {"acc": 0.4}}), tmp_path, "2605.15155")
    assert "missing_module" not in _lessons(tmp_path)

def test_succeeded_phase_requires_real_success(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "contract_violation"}), tmp_path, "2605.15155")
    # A success=false+metrics row does NOT register EXPERIMENT_SUCCEEDED → no opportunity → no aging.
    ld.mine_lessons(_run(tmp_path, "prj_b", {"success": False, "metrics": {"acc": 0.4}}), tmp_path, "2605.15155")
    assert _lessons(tmp_path)["contract_violation"]["staleness"] == 0

def test_arxiv_none_and_flag_off_are_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "missing_module"}), tmp_path, None)
    assert not (tmp_path / "_lessons").exists()
    monkeypatch.delenv("REPROLAB_NEGATIVE_LESSONS", raising=False)
    ld.mine_lessons(_run(tmp_path, "prj_b", {"success": False, "failure_class": "missing_module"}), tmp_path, "2605.15155")
    assert not (tmp_path / "_lessons").exists()

def test_corrupt_file_failsoft(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    (tmp_path / "_lessons").mkdir()
    (tmp_path / "_lessons" / "2605.15155.json").write_text("{bad")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "missing_module"}), tmp_path, "2605.15155")
    assert _lessons(tmp_path)["missing_module"]["occurrences"] == 1  # recovered
```
- [ ] **Step 2:** Run → FAIL (no module).
- [ ] **Step 3:** create `lesson_distiller.py` (full body — see Implementation reference at bottom).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** commit `feat(phase9): lesson_distiller — mine/promote/retire (deterministic)`.

## Task 3: injection block

**Files:**
- Modify: `backend/agents/rlm/lesson_distiller.py` (add `render_block`)
- Modify: `backend/agents/baseline_implementation.py` (`_negative_lessons_block` + call in `_compute_constraint_guidance`)
- Test: `tests/agents/rlm/test_negative_lessons_injection.py`

- [ ] **Step 1: failing tests**
```python
# tests/agents/rlm/test_negative_lessons_injection.py
import json
from pathlib import Path
from backend.agents.rlm import lesson_distiller as ld

def _seed(tmp, *lessons, arxiv="2605.15155"):
    (tmp / "_lessons").mkdir(parents=True, exist_ok=True)
    (tmp / "_lessons" / f"{arxiv}.json").write_text(json.dumps(
        {"version": "v1", "arxiv_id": arxiv, "lessons": list(lessons)}))

def test_block_empty_when_off(tmp_path, monkeypatch):
    monkeypatch.delenv("REPROLAB_NEGATIVE_LESSONS", raising=False)
    _seed(tmp_path, {"failure_class": "dockerfile_invalid", "suggested_fix": "x", "status": "active", "occurrences": 2})
    assert ld.render_block(tmp_path, "2605.15155") == ""

def test_block_empty_without_arxiv(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    assert ld.render_block(tmp_path, None) == ""

def test_block_only_active(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    _seed(tmp_path,
        {"failure_class": "missing_module", "suggested_fix": "add to requirements", "status": "candidate", "occurrences": 1},
        {"failure_class": "dockerfile_invalid", "suggested_fix": "FROM must be first", "status": "active", "occurrences": 3})
    block = ld.render_block(tmp_path, "2605.15155")
    assert "dockerfile_invalid" in block and "missing_module" not in block
    assert "seen 3" in block

def test_block_capped_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    lessons = [{"failure_class": f"c{i}", "suggested_fix": "y" * 500, "status": "active", "occurrences": i} for i in range(10)]
    _seed(tmp_path, *lessons)
    block = ld.render_block(tmp_path, "2605.15155")
    assert block.count("\n- ") <= 5  # K=5 cap (count list items)
    assert all(len(line) < 260 for line in block.splitlines())  # 200-char fix bound + tag
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** add `render_block` to `lesson_distiller.py` (see reference) + `_negative_lessons_block` + call in `baseline_implementation.py`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** commit `feat(phase9): negative-lessons injection block in implementer guidance`.

## Task 4: mining hook in `run.py` `_finalize`

**Files:**
- Modify: `backend/agents/rlm/run.py` (after `write_final_report_rlm` at the `_finalize` normal path, ~line 1868)

- [ ] **Step 1:** add right after `json_path, _md_path = write_final_report_rlm(report, project_dir)` in `_finalize`:
```python
    # Phase 9: mine per-paper negative lessons from this run's experiment records.
    try:
        from backend.agents.rlm import lesson_distiller as _ld
        _ld.mine_lessons(project_dir, ctx.runs_root, ctx.arxiv_id, run_id=ctx.project_id)
    except Exception:  # noqa: BLE001 — mining MUST NOT affect run teardown
        logger.debug("run_pipeline_rlm: lesson mining failed", exc_info=True)
```
- [ ] **Step 2:** Run `pytest tests/rlm/ -q` (no regression in the finalize path).
- [ ] **Step 3:** commit `feat(phase9): mine negative lessons post-finalize (fail-soft)`.

## Task 5: docs + full regression

- [ ] **Step 1:** add CLAUDE.md sub-section for `REPROLAB_NEGATIVE_LESSONS`; mark Phase 9 done in the master-plan status table.
- [ ] **Step 2:** `pytest tests/ -n auto -q` → all green.
- [ ] **Step 3:** commit `docs(phase9): document REPROLAB_NEGATIVE_LESSONS + plan status`.

---

## Implementation reference — `lesson_distiller.py`

(Full module body referenced by Tasks 2–3; constants per spec §3/§4/§5/§6.)

```python
_LESSON_WORTHY_CLASSES = frozenset({
    "missing_module", "torch_redundancy", "requirements_not_found", "missing_dataset",
    "syntax_error", "scope_shape_violation", "contract_violation", "dockerfile_invalid"})
_PROMOTE_THRESHOLD = {"dockerfile_invalid": 1}   # default 2
_RETIRE_STALENESS = 3
_MAX_INJECT = 5
_FIX_TRUNC = 200
_OPPORTUNITY_PHASE = {
    "dockerfile_invalid": "EXPERIMENT_ATTEMPTED", "torch_redundancy": "EXPERIMENT_ATTEMPTED",
    "requirements_not_found": "EXPERIMENT_ATTEMPTED", "missing_module": "EXPERIMENT_ATTEMPTED",
    "missing_dataset": "EXPERIMENT_ATTEMPTED", "syntax_error": "EXPERIMENT_ATTEMPTED",
    "scope_shape_violation": "EXPERIMENT_SUCCEEDED", "contract_violation": "EXPERIMENT_SUCCEEDED"}
```
- `_phases_reached(rows)`: `EXPERIMENT_ATTEMPTED` if `rows`; `EXPERIMENT_SUCCEEDED` if any row has `success is True and isinstance(metrics, dict) and metrics`.
- `mine_lessons`: promotion pass (fired classes upsert, `occurrences+1`, `staleness=0`, fix re-derived from classifier, status by threshold), then retirement pass (non-fired lessons: `staleness+1` iff gating phase reached; drop when `>= _RETIRE_STALENESS`).
- `render_block`: active-only, sort by `(occurrences, last_seen_run)` desc, top-5, fix truncated to 200.

## Self-Review
**Spec coverage:** §2.1 module → T2/T3; §2.2 accessor → T1; §2.3 hook → T4; §2.4 injection → T3; §3 allowlist → T2 (`_LESSON_WORTHY_CLASSES`); §4 promotion/provenance → T2 tests; §5 retirement → T2 tests; §6 block → T3 tests; §7 safety → enforced by classifier-sourced fix + recurrence-gate + cap; §8 testing → T1-T3; §10 files → all. No gaps.
**Placeholders:** none. **Type consistency:** `mine_lessons(project_dir, runs_root, arxiv_id, *, run_id)`, `read_lessons(runs_root, arxiv_id)`, `render_block(runs_root, arxiv_id)`, lesson keys `{failure_class, suggested_fix, suggested_fix_source, occurrences, status, first_seen_run, last_seen_run, staleness}` consistent across tasks.
