# Issue 19 Provenance Manifests, Command Logging, And Artifact Writers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, tested Python artifact layer that creates PRD-shaped run directories, structured command logs, provenance manifests, and standard artifact writers even if the broader runtime integrations from #8 and #18 are not present yet.

**Architecture:** Add a small `src/reprolab/artifacts` package that owns run-directory initialization plus JSON/Markdown writers for the standard files defined in the PRD. Keep the implementation stdlib-only so it is easy to integrate later with a sandbox runner or orchestrator, and prove behavior with focused pytest coverage that treats the PRD as the source of truth.

**Tech Stack:** Python 3.11, pytest, pathlib, json, dataclasses

---

### Task 1: Bootstrap A Minimal Python Package And Lock The PRD Contract In Tests

**Files:**
- Create: `pyproject.toml`
- Create: `src/reprolab/__init__.py`
- Create: `src/reprolab/artifacts/__init__.py`
- Test: `tests/artifacts/test_writers.py`

- [ ] **Step 1: Write the failing tests for the artifact directory contract**

```python
def test_initialize_baseline_run_creates_prd_artifacts(tmp_path):
    run_dir = initialize_run_artifacts(
        root_dir=tmp_path,
        run_type="baseline",
        run_id="baseline_001",
    )
    assert run_dir.name == "baseline"
    assert (run_dir / "commands.log").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "assumptions.json").exists()
    assert (run_dir / "verification.json").exists()
    assert (run_dir / "provenance.json").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "logs").is_dir()
    assert (run_dir / "plots").is_dir()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/artifacts/test_writers.py -v`
Expected: FAIL because the package and writer functions do not exist yet.

- [ ] **Step 3: Add the minimal package scaffold**

```python
__all__ = []
```

- [ ] **Step 4: Run the tests again to confirm they still fail for the missing implementation, not test setup**

Run: `python -m pytest tests/artifacts/test_writers.py -v`
Expected: FAIL with import or missing symbol errors from `reprolab.artifacts`.

- [ ] **Step 5: Commit the scaffold**

```bash
git add pyproject.toml src/reprolab/__init__.py src/reprolab/artifacts/__init__.py tests/artifacts/test_writers.py
git commit -m "test: scaffold artifact writer package"
```

### Task 2: Implement Standard Artifact Writers, Structured Command Logging, And Provenance Manifest Support

**Files:**
- Create: `src/reprolab/artifacts/models.py`
- Create: `src/reprolab/artifacts/writers.py`
- Modify: `src/reprolab/artifacts/__init__.py`
- Test: `tests/artifacts/test_writers.py`

- [ ] **Step 1: Extend tests to cover the full issue scope**

```python
def test_append_command_log_writes_json_lines_with_exact_command(tmp_path):
    run_dir = initialize_run_artifacts(tmp_path, "baseline", "baseline_001")
    append_command_log(
        run_dir,
        CommandLogEntry(
            command="python train.py --config configs/repro.yaml",
            phase="experiment_runner",
            status="succeeded",
            exit_code=0,
        ),
    )
    line = (run_dir / "commands.log").read_text().strip().splitlines()[0]
    payload = json.loads(line)
    assert payload["command"] == "python train.py --config configs/repro.yaml"
    assert payload["status"] == "succeeded"
```

```python
def test_standard_writers_persist_prd_shaped_payloads(tmp_path):
    run_dir = initialize_run_artifacts(tmp_path, "improvement", "path_001")
    write_metrics(run_dir, {"run_id": "path_001", "metric_name": "accuracy"})
    write_assumptions(run_dir, {"paper_id": "paper-1", "assumptions": []})
    write_verification(run_dir, {"status": "blocked", "blocking_issues": ["missing runtime"]})
    write_provenance(run_dir, {"run_id": "path_001", "repo": {"commit": "abc123"}})
    write_report(run_dir, "# Report\n")
    assert json.loads((run_dir / "metrics.json").read_text())["run_id"] == "path_001"
```

- [ ] **Step 2: Run the focused test file and watch it fail**

Run: `python -m pytest tests/artifacts/test_writers.py -v`
Expected: FAIL because the public writer API is not implemented.

- [ ] **Step 3: Implement the minimal stdlib-only writer layer**

```python
@dataclass(slots=True)
class CommandLogEntry:
    command: str
    phase: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    cwd: str | None = None
    notes: str | None = None
```

```python
def append_command_log(run_dir: Path, entry: CommandLogEntry) -> Path:
    with (run_dir / "commands.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
    return run_dir / "commands.log"
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python -m pytest tests/artifacts/test_writers.py -v`
Expected: PASS

- [ ] **Step 5: Commit the writer implementation**

```bash
git add src/reprolab/artifacts/__init__.py src/reprolab/artifacts/models.py src/reprolab/artifacts/writers.py tests/artifacts/test_writers.py
git commit -m "feat: add auditable artifact writer utilities"
```

### Task 3: Prove The Safe Integration Boundary And Document The Current Blockers In Tests

**Files:**
- Modify: `tests/artifacts/test_writers.py`
- Modify: `src/reprolab/artifacts/writers.py`

- [ ] **Step 1: Add a test for explicit run layout selection and missing runtime integration safety**

```python
def test_initialize_improvement_run_uses_path_directory(tmp_path):
    run_dir = initialize_run_artifacts(tmp_path, "improvement", "path_007")
    assert run_dir == tmp_path / "runs" / "improvements" / "path_007"
```

```python
def test_report_writer_can_capture_known_blockers(tmp_path):
    run_dir = initialize_run_artifacts(tmp_path, "baseline", "baseline_001")
    write_report(run_dir, "## Blockers\n- Runtime backend from #8 is not in this repo.\n")
    assert "#8" in (run_dir / "report.md").read_text()
```

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 3: Review the repo status and confirm only issue-relevant files changed**

Run: `git status --short`
Expected: only the new Python package, tests, and plan doc changes for issue #19.

- [ ] **Step 4: Commit the final polish**

```bash
git add tests/artifacts/test_writers.py src/reprolab/artifacts/writers.py docs/superpowers/plans/2026-05-09-issue-19-provenance-manifests-command-logging.md
git commit -m "docs: capture issue 19 implementation plan"
```
