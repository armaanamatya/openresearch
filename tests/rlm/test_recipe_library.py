"""Tests for recipe_library.py — Tier-B cross-run positive recipe memory.

Acceptance criteria (from Task P4.1):
- A clean report (evidence_gate passed + success ledger row + validator clean +
  meets_target True) → admit_recipe writes a recipe.
- A high-GRADE but evidence_gate-FAILED report → NOT admitted (red line).
- A validator-'vetoed' report → NOT admitted.
- The static-import guard passes (no grade field read outside allowlisted line).
- recipe_guidance_block injects top-1 capped; returns "" when disabled.
- derive_paper_class buckets SDAR via PAPER_HINTS.
- poison-proof: agent prose never enters the recipe body.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from backend.agents.rlm.recipe_library import (
    GRADE_FIELDS,
    INJECT_TOP_K,
    MAX_RECIPES,
    RETIRE_STALENESS,
    Recipe,
    _evidence_gate_passed,
    _has_success_ledger_row,
    _problem_sig_hash,
    _store_path,
    _validator_ok,
    admit_recipe,
    derive_paper_class,
    positive_recipes_enabled,
    recipe_guidance_block,
    update_staleness,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ──────────────────────────────────────────────────────────────────────────────

_REAL_METRICS = {"reward": 0.42, "return": 31.1, "accuracy": 0.83}
_ZERO_METRICS = {"reward": 0.0, "return": 0.0, "accuracy": 0.0}


def _make_clean_report(*, evidence_gate_passed: bool = True) -> dict:
    """Minimal clean report dict that satisfies all four admission conditions.

    Gate 4 (deterministic_meets_target) is now satisfied by a real non-zero
    code/metrics.json on disk — NOT by rubric.meets_target or overall_score.
    The report carries ``deterministic_meets_target=True`` (the harness-set
    boolean path) as the fastest deterministic signal so tests that do NOT
    explicitly test gate-4 disk paths keep working without needing a real
    metrics.json.
    """
    return {
        "evidence_gate_passed": evidence_gate_passed,
        "deterministic_meets_target": True,  # harness-set boolean — Tier-1 gate
        "rubric": {
            "overall_score": 0.75,   # metadata only (copy-into-report stamp)
            "target_score": 0.60,
        },
        "scope": {
            "models": ["Qwen3-1.7B"],
            "datasets": ["ALFWorld"],
        },
        "arxiv_id": "2605.15155",
    }


def _make_project_dir(
    tmp_path: Path,
    *,
    with_success_row: bool = True,
    with_metrics: bool = False,
    metrics: dict | None = None,
) -> Path:
    """Create a minimal project dir with an experiment_runs.jsonl.

    Parameters
    ----------
    with_success_row:
        Write a ``success=True`` row into ``experiment_runs.jsonl``.
    with_metrics:
        Also write ``code/metrics.json`` so that gate 4 (measured-evidence
        path) can pass for tests that exercise the disk-based path.
    metrics:
        Custom metrics dict to write.  Defaults to _REAL_METRICS (non-zero).
    """
    project_dir = tmp_path / "test_proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    ledger = project_dir / "experiment_runs.jsonl"
    if with_success_row:
        ledger.write_text(
            json.dumps({"success": True, "metrics": {"reward": 0.42}, "logs": ""}) + "\n",
            encoding="utf-8",
        )
    else:
        ledger.write_text(
            json.dumps({"success": False, "failure_class": "cell_execution_error", "logs": ""}) + "\n",
            encoding="utf-8",
        )
    if with_metrics:
        code_dir = project_dir / "code"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "metrics.json").write_text(
            json.dumps(metrics if metrics is not None else _REAL_METRICS),
            encoding="utf-8",
        )
    return project_dir


def _clean_validator_verdict() -> dict:
    return {"status": "clean", "veto_set": [], "predicates": []}


def _vetoed_validator_verdict() -> dict:
    return {"status": "vetoed", "veto_set": ["metric_ref_1"], "predicates": []}


def _unavailable_validator_verdict() -> dict:
    return {"status": "unavailable", "veto_set": [], "predicates": []}


def _enable_recipes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_POSITIVE_RECIPES", "1")


def _read_recipes(runs_root: Path, paper_class: str) -> list[dict]:
    path = _store_path(runs_root, paper_class)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────────
# Feature gate
# ──────────────────────────────────────────────────────────────────────────────

def test_positive_recipes_disabled_by_default(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_POSITIVE_RECIPES", raising=False)
    assert positive_recipes_enabled() is False


@pytest.mark.parametrize("val", ["1", "on", "true", "yes", "TRUE", "YES"])
def test_positive_recipes_enabled_values(monkeypatch, val):
    monkeypatch.setenv("OPENRESEARCH_POSITIVE_RECIPES", val)
    assert positive_recipes_enabled() is True


# ──────────────────────────────────────────────────────────────────────────────
# derive_paper_class
# ──────────────────────────────────────────────────────────────────────────────

def test_derive_paper_class_sdar_arxiv_id():
    """SDAR (2605.15155) must be bucketed to 'sdar_agentic_rl'."""
    result = derive_paper_class(arxiv_id="2605.15155")
    assert result == "sdar_agentic_rl"


def test_derive_paper_class_sdar_with_version_suffix():
    result = derive_paper_class(arxiv_id="2605.15155v2")
    assert result == "sdar_agentic_rl"


def test_derive_paper_class_allcnn():
    result = derive_paper_class(arxiv_id="1412.6806")
    assert result == "image_classification"


def test_derive_paper_class_resnet():
    result = derive_paper_class(arxiv_id="1512.03385")
    assert result == "image_classification"


def test_derive_paper_class_adam():
    result = derive_paper_class(arxiv_id="1412.6980")
    assert result == "optimizer_comparison"


def test_derive_paper_class_omnizip():
    result = derive_paper_class(arxiv_id="2511.14582")
    assert result == "compression"


def test_derive_paper_class_rubric_shape_fallback():
    rubric = {
        "leaves": [
            {"task_category": "Result Analysis (reward)"},
            {"task_category": "Eval Protocol (return)"},
        ]
    }
    result = derive_paper_class(rubric=rubric)
    assert result == "rl_agent"


def test_derive_paper_class_rubric_image_classification_fallback():
    rubric = {"leaves": [{"task_category": "accuracy on cifar10"}]}
    result = derive_paper_class(rubric=rubric)
    assert result == "image_classification"


def test_derive_paper_class_generic_fallback():
    result = derive_paper_class()
    assert result == "generic"


def test_derive_paper_class_explicit_class_in_hints():
    result = derive_paper_class(paper_hints={"__class__": "custom_class"})
    assert result == "custom_class"


# ──────────────────────────────────────────────────────────────────────────────
# Admission gate — individual predicates
# ──────────────────────────────────────────────────────────────────────────────

def test_evidence_gate_passed_explicit_true():
    assert _evidence_gate_passed({"evidence_gate_passed": True}) is True


def test_evidence_gate_passed_string():
    assert _evidence_gate_passed({"evidence_gate": "passed"}) is True


def test_evidence_gate_passed_missing():
    assert _evidence_gate_passed({}) is False


def test_evidence_gate_passed_false():
    assert _evidence_gate_passed({"evidence_gate_passed": False}) is False


def test_evidence_gate_passed_nested_validation():
    report = {"validation": {"evidence_gate": "passed"}}
    assert _evidence_gate_passed(report) is True


def test_has_success_ledger_row_true(tmp_path):
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    assert _has_success_ledger_row(project_dir) is True


def test_has_success_ledger_row_false(tmp_path):
    project_dir = _make_project_dir(tmp_path, with_success_row=False)
    assert _has_success_ledger_row(project_dir) is False


def test_has_success_ledger_row_missing_file(tmp_path):
    project_dir = tmp_path / "empty_proj"
    project_dir.mkdir()
    assert _has_success_ledger_row(project_dir) is False


def test_validator_ok_clean():
    assert _validator_ok({"status": "clean"}, floor_passed=True) is True
    assert _validator_ok({"status": "clean"}, floor_passed=False) is True


def test_validator_ok_vetoed_never_admits():
    assert _validator_ok({"status": "vetoed"}, floor_passed=True) is False
    assert _validator_ok({"status": "vetoed"}, floor_passed=False) is False


def test_validator_ok_unavailable_with_floor_passed():
    assert _validator_ok({"status": "unavailable"}, floor_passed=True) is True


def test_validator_ok_unavailable_without_floor():
    assert _validator_ok({"status": "unavailable"}, floor_passed=False) is False


def test_validator_ok_none():
    assert _validator_ok(None, floor_passed=True) is True
    assert _validator_ok(None, floor_passed=False) is False


def test_validator_ok_object_with_status_attr():
    class _V:
        status = "clean"
    assert _validator_ok(_V(), floor_passed=False) is True


def test_validator_ok_vetoed_object():
    class _V:
        status = "vetoed"
    assert _validator_ok(_V(), floor_passed=True) is False


# ──────────────────────────────────────────────────────────────────────────────
# admit_recipe — happy path (clean report → recipe written)
# ──────────────────────────────────────────────────────────────────────────────

def test_admit_recipe_clean_report_writes_recipe(monkeypatch, tmp_path):
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    report = _make_clean_report()

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_clean_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert len(records) == 1
    r = records[0]
    assert r["paper_class"] == "sdar_agentic_rl"
    assert "problem_sig" in r
    assert "solution_sig" in r
    assert "evidence_key" in r


def test_admit_recipe_disabled_is_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENRESEARCH_POSITIVE_RECIPES", raising=False)
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    admit_recipe(
        project_dir, runs_root,
        report=_make_clean_report(),
        validator_verdict=_clean_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == []


# ──────────────────────────────────────────────────────────────────────────────
# admit_recipe — RED LINE: evidence_gate failed → NOT admitted (even if grade is high)
# ──────────────────────────────────────────────────────────────────────────────

def test_admit_recipe_evidence_gate_failed_not_admitted(monkeypatch, tmp_path):
    """A high-grade but evidence_gate-failed report must NOT be admitted."""
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    # Grade looks fine (meets_target=True, rubric.overall_score=0.9) but gate failed.
    report = {
        "evidence_gate_passed": False,   # ← the critical failure
        "rubric": {
            "overall_score": 0.9,         # "high grade" — must NOT matter
            "target_score": 0.60,
            "meets_target": True,
        },
        "scope": {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]},
    }

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_clean_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == [], "Must NOT admit when evidence_gate_passed=False"


# ──────────────────────────────────────────────────────────────────────────────
# admit_recipe — validator vetoed → NOT admitted
# ──────────────────────────────────────────────────────────────────────────────

def test_admit_recipe_validator_vetoed_not_admitted(monkeypatch, tmp_path):
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    report = _make_clean_report()

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_vetoed_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == [], "Must NOT admit when validator vetoed"


# ──────────────────────────────────────────────────────────────────────────────
# admit_recipe — no success ledger row → NOT admitted
# ──────────────────────────────────────────────────────────────────────────────

def test_admit_recipe_no_success_ledger_row_not_admitted(monkeypatch, tmp_path):
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=False)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    report = _make_clean_report()

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_clean_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == [], "Must NOT admit without success ledger row"


# ──────────────────────────────────────────────────────────────────────────────
# admit_recipe — gate 4 (deterministic_meets_target) — measured-evidence semantics
# ──────────────────────────────────────────────────────────────────────────────

def test_admit_recipe_no_deterministic_flag_no_metrics_not_admitted(monkeypatch, tmp_path):
    """Gate 4 fails when deterministic_meets_target is absent and no metrics.json exists."""
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True, with_metrics=False)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    # Report has high LLM grade — must NOT matter.
    report = {
        "evidence_gate_passed": True,
        "rubric": {"overall_score": 0.95, "target_score": 0.60, "meets_target": True},
        "scope": {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]},
    }

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_clean_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == [], "Must NOT admit: no deterministic_meets_target flag and no metrics.json"


def test_admit_recipe_real_metrics_without_target_not_admitted(monkeypatch, tmp_path):
    """CONSERVATIVE gate 4: a real non-zero metrics.json proves a run HAPPENED, not
    that it MET the target. Without a deterministic target proof (no
    deterministic_meets_target flag, no paper_claimed_target/measured_headline),
    admission must be refused — never fall back to 'real metrics exist' or the grade.
    """
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(
        tmp_path, with_success_row=True, with_metrics=True, metrics=_REAL_METRICS
    )
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    # Real metrics on disk + a high LLM grade, but NO deterministic target proof.
    report = {
        "evidence_gate_passed": True,
        "rubric": {"overall_score": 0.75, "target_score": 0.60},
        "scope": {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]},
        "arxiv_id": "2605.15155",
    }

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_clean_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == [], "Must NOT admit without a deterministic measured-vs-target proof"


def test_admit_recipe_zero_metrics_not_admitted(monkeypatch, tmp_path):
    """Gate 4 fails when code/metrics.json is all-zero (degenerate fabrication)."""
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(
        tmp_path, with_success_row=True, with_metrics=True, metrics=_ZERO_METRICS
    )
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    # LLM grade looks fine but real metrics are all-zero — must NOT admit.
    report = {
        "evidence_gate_passed": True,
        "rubric": {"overall_score": 0.95, "target_score": 0.60, "meets_target": True},
        "scope": {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]},
    }

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_clean_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == [], "Must NOT admit when metrics.json is all-zero (zero-metrics floor)"


def test_admit_recipe_high_grade_no_metrics_not_admitted(monkeypatch, tmp_path):
    """High overall_score with no metrics.json must NOT be admitted (red line)."""
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True, with_metrics=False)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    report = {
        "evidence_gate_passed": True,
        "rubric": {"overall_score": 0.99, "target_score": 0.60, "meets_target": True},
        "scope": {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]},
    }

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_clean_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == [], "High LLM grade must NOT substitute for missing measured evidence"


def test_admit_recipe_measured_headline_at_target_admits(monkeypatch, tmp_path):
    """paper_claimed_target + measured_headline comparison: at-or-above target → admitted."""
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(
        tmp_path, with_success_row=True, with_metrics=True, metrics=_REAL_METRICS
    )
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    report = {
        "evidence_gate_passed": True,
        "rubric": {"overall_score": 0.75},  # grade ignored for gate 4
        "paper_claimed_target": 0.80,
        "measured_headline": 0.83,  # above target
        "scope": {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]},
    }

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_clean_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert len(records) == 1, "Should admit: measured_headline >= paper_claimed_target"


def test_admit_recipe_measured_headline_below_target_not_admitted(monkeypatch, tmp_path):
    """paper_claimed_target + measured_headline: below target → NOT admitted."""
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(
        tmp_path, with_success_row=True, with_metrics=True, metrics=_REAL_METRICS
    )
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    report = {
        "evidence_gate_passed": True,
        "rubric": {"overall_score": 0.75},  # grade ignored for gate 4
        "paper_claimed_target": 0.90,
        "measured_headline": 0.83,  # below target
        "scope": {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]},
    }

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_clean_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == [], "Must NOT admit: measured_headline < paper_claimed_target"


def test_admit_recipe_deterministic_flag_bypasses_disk(monkeypatch, tmp_path):
    """deterministic_meets_target=True on the report bypasses the disk check (fast path)."""
    _enable_recipes(monkeypatch)
    # No metrics.json on disk — the harness-set boolean must be sufficient.
    project_dir = _make_project_dir(tmp_path, with_success_row=True, with_metrics=False)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    report = {
        "evidence_gate_passed": True,
        "deterministic_meets_target": True,  # harness-set Tier-1 signal
        "rubric": {"overall_score": 0.75},
        "scope": {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]},
    }

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_clean_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert len(records) == 1, "deterministic_meets_target=True must admit without disk evidence"


# ──────────────────────────────────────────────────────────────────────────────
# admit_recipe — validator unavailable + floor passed → admit
# ──────────────────────────────────────────────────────────────────────────────

def test_admit_recipe_validator_unavailable_with_floor_passes(monkeypatch, tmp_path):
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    report = _make_clean_report()

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_unavailable_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert len(records) == 1, "Should admit when validator unavailable but floor passes"


def test_admit_recipe_validator_unavailable_but_gate_failed(monkeypatch, tmp_path):
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    report = {
        "evidence_gate_passed": False,
        "rubric": {"overall_score": 0.8, "target_score": 0.6, "meets_target": True},
        "scope": {},
    }

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_unavailable_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == [], "Must NOT admit when gate failed even if validator unavailable"


# ──────────────────────────────────────────────────────────────────────────────
# admit_recipe — novelty dedup
# ──────────────────────────────────────────────────────────────────────────────

def test_admit_recipe_novelty_dedup(monkeypatch, tmp_path):
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    report = _make_clean_report()

    # Admit the same recipe twice.
    admit_recipe(project_dir, runs_root, report=report,
                 validator_verdict=_clean_validator_verdict(), paper_class="sdar_agentic_rl")
    admit_recipe(project_dir, runs_root, report=report,
                 validator_verdict=_clean_validator_verdict(), paper_class="sdar_agentic_rl")

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert len(records) == 1, "Must dedup identical problem_sig"


# ──────────────────────────────────────────────────────────────────────────────
# admit_recipe — cap enforcement
# ──────────────────────────────────────────────────────────────────────────────

def test_admit_recipe_cap_enforcement(monkeypatch, tmp_path):
    _enable_recipes(monkeypatch)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    for i in range(MAX_RECIPES + 5):
        project_dir = tmp_path / f"proj_{i}"
        project_dir.mkdir(parents=True)
        (project_dir / "experiment_runs.jsonl").write_text(
            json.dumps({"success": True, "metrics": {}, "logs": ""}) + "\n",
            encoding="utf-8",
        )
        report = {
            "evidence_gate_passed": True,
            "deterministic_meets_target": True,  # harness-set Tier-1 signal
            # Distinct scope per admission to pass novelty dedup.
            "scope": {"models": [f"ModelX-{i}"], "datasets": ["DS"]},
        }
        admit_recipe(
            project_dir, runs_root,
            report=report,
            validator_verdict=_clean_validator_verdict(),
            paper_class="sdar_agentic_rl",
        )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert len(records) <= MAX_RECIPES, "Store must be capped at MAX_RECIPES"


# ──────────────────────────────────────────────────────────────────────────────
# admit_recipe — atomic write + fail-soft
# ──────────────────────────────────────────────────────────────────────────────

def test_admit_recipe_atomic_write(monkeypatch, tmp_path):
    """The store file must be written atomically (no .tmp left behind)."""
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    report = _make_clean_report()

    admit_recipe(project_dir, runs_root, report=report,
                 validator_verdict=_clean_validator_verdict(), paper_class="sdar_agentic_rl")

    store_path = _store_path(runs_root, "sdar_agentic_rl")
    tmp_file = store_path.with_suffix(store_path.suffix + ".tmp")
    assert store_path.exists(), "Store file must exist after admit"
    assert not tmp_file.exists(), "Temp file must be cleaned up"


def test_admit_recipe_fail_soft_on_bad_input(monkeypatch, tmp_path):
    """admit_recipe must never raise — fail-soft contract."""
    _enable_recipes(monkeypatch)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    # project_dir does not exist — should not raise
    admit_recipe(
        tmp_path / "nonexistent", runs_root,
        report={"evidence_gate_passed": True},
        paper_class="test",
    )


# ──────────────────────────────────────────────────────────────────────────────
# recipe_guidance_block
# ──────────────────────────────────────────────────────────────────────────────

def test_recipe_guidance_block_disabled_returns_empty(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENRESEARCH_POSITIVE_RECIPES", raising=False)
    result = recipe_guidance_block(tmp_path, "sdar_agentic_rl")
    assert result == ""


def test_recipe_guidance_block_no_recipes_returns_empty(monkeypatch, tmp_path):
    _enable_recipes(monkeypatch)
    result = recipe_guidance_block(tmp_path, "sdar_agentic_rl")
    assert result == ""


def test_recipe_guidance_block_injects_top_recipe(monkeypatch, tmp_path):
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    report = _make_clean_report()

    admit_recipe(project_dir, runs_root, report=report,
                 validator_verdict=_clean_validator_verdict(), paper_class="sdar_agentic_rl")

    block = recipe_guidance_block(runs_root, "sdar_agentic_rl")
    assert block != ""
    assert "POSITIVE RECIPES" in block
    assert "sdar_agentic_rl" in block


def test_recipe_guidance_block_capped_at_inject_top_k(monkeypatch, tmp_path):
    _enable_recipes(monkeypatch)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    # Admit more than INJECT_TOP_K distinct recipes.
    for i in range(INJECT_TOP_K + 3):
        project_dir = tmp_path / f"proj_g_{i}"
        project_dir.mkdir(parents=True)
        (project_dir / "experiment_runs.jsonl").write_text(
            json.dumps({"success": True}) + "\n", encoding="utf-8"
        )
        report = {
            "evidence_gate_passed": True,
            "deterministic_meets_target": True,  # harness-set Tier-1 signal
            "scope": {"models": [f"M-{i}"], "datasets": ["DS"]},
        }
        admit_recipe(
            project_dir, runs_root,
            report=report,
            validator_verdict=_clean_validator_verdict(),
            paper_class="sdar_agentic_rl",
        )

    block = recipe_guidance_block(runs_root, "sdar_agentic_rl")
    # Count the bullet points injected.
    bullet_count = block.count("\n- ")
    assert bullet_count <= INJECT_TOP_K, (
        f"Must inject at most {INJECT_TOP_K} recipes, got {bullet_count}"
    )


def test_recipe_guidance_block_respects_max_chars(monkeypatch, tmp_path):
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    admit_recipe(project_dir, runs_root, report=_make_clean_report(),
                 validator_verdict=_clean_validator_verdict(), paper_class="sdar_agentic_rl")

    block = recipe_guidance_block(runs_root, "sdar_agentic_rl", max_chars=50)
    assert len(block) <= 50


# ──────────────────────────────────────────────────────────────────────────────
# Poison-proof: agent prose must never enter the recipe body
# ──────────────────────────────────────────────────────────────────────────────

def test_admit_recipe_poison_proof_no_agent_prose(monkeypatch, tmp_path):
    """Agent-generated free text must not appear in the recipe body."""
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    # Include a "paper_class_summary" that contains a benign canary — this is the
    # only prose-like field allowed (bounded to MAX_SUMMARY_CHARS).  We do NOT
    # include any fabricated metrics or arbitrary agent prose in problem_sig or
    # solution_sig's hyperparameters.
    AGENT_PROSE_CANARY = "DO_NOT_LEAK_THIS_AGENT_PROSE_XYZ"
    report = {
        "evidence_gate_passed": True,
        "deterministic_meets_target": True,  # harness-set Tier-1 signal
        "rubric": {"overall_score": 0.7},   # grade kept in report for copy-into-report stamp only
        "scope": {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]},
        # This field is explicitly NOT copied into problem_sig or solution_sig
        # hyperparameters — only technique_summary (bounded) may come from here.
        "agent_free_prose": AGENT_PROSE_CANARY,
    }

    admit_recipe(project_dir, runs_root, report=report,
                 validator_verdict=_clean_validator_verdict(), paper_class="sdar_agentic_rl")

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert len(records) == 1
    r = records[0]
    # The canary must NOT appear anywhere in problem_sig or solution_sig.
    assert AGENT_PROSE_CANARY not in json.dumps(r.get("problem_sig", {}))
    assert AGENT_PROSE_CANARY not in json.dumps(r.get("solution_sig", {}))


# ──────────────────────────────────────────────────────────────────────────────
# update_staleness
# ──────────────────────────────────────────────────────────────────────────────

def test_update_staleness_increments(monkeypatch, tmp_path):
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    admit_recipe(project_dir, runs_root, report=_make_clean_report(),
                 validator_verdict=_clean_validator_verdict(), paper_class="sdar_agentic_rl")

    # Staleness starts at 0.
    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records[0]["staleness"] == 0

    update_staleness(runs_root, "sdar_agentic_rl")
    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records[0]["staleness"] == 1


def test_update_staleness_retires_old_recipes(monkeypatch, tmp_path):
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    admit_recipe(project_dir, runs_root, report=_make_clean_report(),
                 validator_verdict=_clean_validator_verdict(), paper_class="sdar_agentic_rl")

    # Run staleness RETIRE_STALENESS times — recipe should be retired.
    for _ in range(RETIRE_STALENESS):
        update_staleness(runs_root, "sdar_agentic_rl")

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == [], "Recipe must be retired after RETIRE_STALENESS increments"


def test_update_staleness_disabled_is_noop(monkeypatch, tmp_path):
    """update_staleness must be a no-op when the feature is disabled."""
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(tmp_path, with_success_row=True)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    admit_recipe(project_dir, runs_root, report=_make_clean_report(),
                 validator_verdict=_clean_validator_verdict(), paper_class="sdar_agentic_rl")

    # Now disable the feature.
    monkeypatch.delenv("OPENRESEARCH_POSITIVE_RECIPES", raising=False)
    update_staleness(runs_root, "sdar_agentic_rl")

    # Records must be unchanged (no-op).
    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert len(records) == 1
    assert records[0]["staleness"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# Recipe dataclass
# ──────────────────────────────────────────────────────────────────────────────

def test_recipe_dataclass_frozen():
    r = Recipe(
        problem_sig={"models": ["Qwen3-1.7B"]},
        solution_sig={"hyperparameters": {"lr": 5e-5}},
        evidence_key="abc123",
        paper_class="sdar_agentic_rl",
    )
    with pytest.raises((AttributeError, TypeError)):
        r.paper_class = "other"  # type: ignore[misc]


def test_recipe_dataclass_defaults():
    r = Recipe()
    assert r.problem_sig == {}
    assert r.solution_sig == {}
    assert r.evidence_key == ""
    assert r.paper_class == ""


# ──────────────────────────────────────────────────────────────────────────────
# Problem-sig hash stability
# ──────────────────────────────────────────────────────────────────────────────

def test_problem_sig_hash_stable():
    sig = {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]}
    h1 = _problem_sig_hash(sig)
    h2 = _problem_sig_hash(sig)
    assert h1 == h2


def test_problem_sig_hash_order_independent():
    sig_a = {"datasets": ["ALFWorld"], "models": ["Qwen3-1.7B"]}
    sig_b = {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]}
    assert _problem_sig_hash(sig_a) == _problem_sig_hash(sig_b)


def test_problem_sig_hash_differs_on_different_sig():
    sig_a = {"models": ["Qwen3-1.7B"]}
    sig_b = {"models": ["Qwen2.5-7B-Instruct"]}
    assert _problem_sig_hash(sig_a) != _problem_sig_hash(sig_b)


# ──────────────────────────────────────────────────────────────────────────────
# RED LINE: static-import guard
# Asserts that no grade-derived field is READ in recipe_library.py except on an
# explicitly allowlisted line (marked with "# copy-into-report (red-line allowlisted)").
# ──────────────────────────────────────────────────────────────────────────────

def test_red_line_static_import_guard():
    """Grade fields must only appear on explicitly allowlisted lines, and those
    allowlisted lines must ALL be inside the ``report_stamp`` block.

    Two complementary checks:
    A. No dict-access read of a grade field appears WITHOUT the allowlist marker
       (i.e. the marker is required for every grade-field access in active code).
    B. Every line carrying the allowlist marker is inside the ``report_stamp``
       block (i.e. the marker is not misapplied to an admission-decision line).

    Check B is the strengthened invariant added after the discovery that the
    allowlist marker was MISapplied to the old ``_deterministic_meets_target``
    admission-decision line, defeating the guard's intent.

    The guard scans recipe_library.py source and exempts:
    - Pure comment lines (stripped line starts with ``#``).
    - Lines inside a docstring (the checker tracks triple-quote depth).
    - Lines that define the ``GRADE_FIELDS`` constant itself (frozenset literal).
    """
    import inspect
    import backend.agents.rlm.recipe_library as _mod

    source = inspect.getsource(_mod)
    lines = source.splitlines()

    ALLOWLIST_MARKER = "# copy-into-report (red-line allowlisted)"
    # The report_stamp block is delimited by these sentinel strings.
    REPORT_STAMP_OPEN = '"report_stamp"'
    REPORT_STAMP_CLOSE_TOKENS = ("},",  "}",)   # closing brace of the report_stamp dict

    # Patterns that constitute actual dict-access reads (not bare literals in sets
    # or docstrings).  We look for:
    #   .get("overall_score")   →  \.get\(\s*["']FIELD["']
    #   ["overall_score"]       →  \["FIELD"\]   or  \['FIELD'\]
    def _is_grade_access(line_text: str, field_name: str) -> bool:
        return bool(
            re.search(rf'\.get\(\s*["\']({re.escape(field_name)})["\']', line_text)
            or re.search(rf'\[["\']({re.escape(field_name)})["\']', line_text)
        )

    in_docstring = False
    violations_a: list[str] = []   # grade access without allowlist marker
    violations_b: list[str] = []   # allowlist marker outside report_stamp block

    # Track whether we are inside the report_stamp block.
    # Strategy: after seeing a line containing '"report_stamp"' and ':' (the key
    # assignment), we are "inside" until we see the closing brace at the same or
    # lower indentation as the opening line.
    in_report_stamp = False
    report_stamp_brace_depth = 0
    report_stamp_open_indent = 0

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Track docstring open/close (triple-quoted strings).
        triple_count = stripped.count('"""') + stripped.count("'''")
        if triple_count % 2 == 1:
            in_docstring = not in_docstring

        # Skip pure comment lines.
        if stripped.startswith("#"):
            continue

        # Skip lines inside a docstring.
        if in_docstring:
            continue

        # Skip the GRADE_FIELDS constant definition itself (frozenset literal —
        # these are just string-literal membership declarations, not reads).
        if "GRADE_FIELDS" in stripped and "frozenset" in stripped:
            continue
        # Also skip lines that are plainly part of the frozenset body (strings
        # inside the set literal that contain a grade field name).
        if stripped.startswith('"') and stripped.rstrip(",").rstrip() in (
            '"overall_score"', '"median_score"', '"compute_adjusted_score"', '"rubric_score"',
        ):
            continue

        # Detect entering the report_stamp block.
        if REPORT_STAMP_OPEN in line and ":" in line and not in_report_stamp:
            in_report_stamp = True
            report_stamp_open_indent = len(line) - len(line.lstrip())
            # Count open braces on this line to track depth correctly.
            report_stamp_brace_depth = line.count("{") - line.count("}")
        elif in_report_stamp:
            report_stamp_brace_depth += line.count("{") - line.count("}")
            # When all opened braces are closed (depth <= 0), we've left the block.
            if report_stamp_brace_depth <= 0:
                in_report_stamp = False

        # ── Check A: grade access must carry the allowlist marker ────────────
        for field_name in GRADE_FIELDS:
            if _is_grade_access(line, field_name):
                if ALLOWLIST_MARKER not in line:
                    violations_a.append(
                        f"Line {lineno}: grade field '{field_name}' read (dict access) "
                        f"without allowlist marker → '{stripped}'"
                    )

        # ── Check B: allowlist marker must only appear in report_stamp block ─
        if ALLOWLIST_MARKER in line and not in_report_stamp:
            violations_b.append(
                f"Line {lineno}: allowlist marker used OUTSIDE report_stamp block "
                f"(admission-path misuse!) → '{stripped}'"
            )

    all_violations = violations_a + violations_b
    assert not all_violations, (
        "RED LINE VIOLATION in recipe_library.py:\n"
        + (
            "  [A] Grade-field reads without allowlist marker:\n    "
            + "\n    ".join(violations_a) + "\n"
            if violations_a else ""
        )
        + (
            "  [B] Allowlist marker used outside report_stamp block (admission-path misuse):\n    "
            + "\n    ".join(violations_b)
            if violations_b else ""
        )
    )


# ──────────────────────────────────────────────────────────────────────────────
# REGRESSION GUARD TESTS (§7, Pillar 5)
#
# These tests prove that the cross-run recipe admission gate is:
#   (a) impervious to a fabricated/unbacked run (all-zero metrics or no success row)
#   (b) impervious to a run that looks good only in the LLM grade but has no
#       deterministic evidence backing
#
# "The fitness signal is the deterministic evidence layer, NEVER the LLM grade."
# (spec §2, §7 red line).
#
# Each test has an explicit comment naming WHICH gate it exercises and why.
# ──────────────────────────────────────────────────────────────────────────────

def test_regression_fabricated_all_zero_metrics_not_admitted(monkeypatch, tmp_path):
    """REGRESSION: A fabricated run with all-zero metrics.json must NOT be admitted.

    Simulates the SDAR-v6 hallucination: real GPU training, all-0.0 metrics with
    real metric keys (not stub keys). The zero_metrics floor in gate 4 must veto.
    This run looks plausible from the grade alone (0.9) but the actual measured
    evidence is degenerate.
    """
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(
        tmp_path,
        with_success_row=True,   # Gate 2 passes
        with_metrics=True,
        metrics=_ZERO_METRICS,   # all-zero — degenerate fabrication signal
    )
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    # Report: evidence gate passed, high LLM grade, measured_headline matches.
    # The ONLY thing that should block: code/metrics.json is all-zero.
    report = {
        "evidence_gate_passed": True,       # Gate 1 passes
        "rubric": {
            "overall_score": 0.90,           # high LLM grade — must NOT be the admission signal
            "target_score": 0.60,
            "meets_target": True,            # grade says yes — must NOT matter for gate 4
        },
        "paper_claimed_target": 0.60,
        "measured_headline": 0.90,           # headline claims to exceed target
        "scope": {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]},
        "arxiv_id": "2605.15155",
    }

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_clean_validator_verdict(),   # Gate 3 passes
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == [], (
        "FABRICATION REGRESSION: a run with all-zero code/metrics.json must NEVER be admitted "
        "regardless of the LLM grade or claimed measured_headline. "
        "Gate 4's zero-metrics disk check must veto this."
    )


def test_regression_no_experiment_evidence_not_admitted(monkeypatch, tmp_path):
    """REGRESSION: A run with no successful experiment ledger row must NOT be admitted.

    No success row in experiment_runs.jsonl = no evidence that an experiment actually
    ran. A fabricating agent could claim success in the report without running anything.
    Gate 2 must veto even if the report looks perfect.
    """
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(
        tmp_path,
        with_success_row=False,   # Gate 2 must fail this
        with_metrics=True,
        metrics=_REAL_METRICS,
    )
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    report = {
        "evidence_gate_passed": True,
        "deterministic_meets_target": True,   # harness-set boolean fast-path
        "rubric": {
            "overall_score": 0.88,
            "target_score": 0.60,
        },
        "scope": {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]},
        "arxiv_id": "2605.15155",
    }

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_clean_validator_verdict(),
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == [], (
        "LEDGER REGRESSION: no success row in experiment_runs.jsonl must block admission. "
        "Gate 2 must be the hard backstop against a run that never actually executed an experiment."
    )


def test_regression_llm_grade_high_evidence_poor_not_admitted(monkeypatch, tmp_path):
    """REGRESSION: High LLM grade with weak evidence must NOT be admitted.

    This is the canonical red-line test. The report carries:
    - rubric.overall_score = 0.95 (high LLM grade)
    - rubric.meets_target = True (LLM says yes)
    - evidence_gate_passed = True
    But there is NO deterministic_meets_target flag, NO paper_claimed_target,
    NO measured_headline, and NO code/metrics.json on disk.

    The recipe library must refuse because there is zero deterministic measured
    evidence that this run actually met its target. The LLM grade is NOT the signal.
    """
    _enable_recipes(monkeypatch)
    # Only a success ledger row — no metrics.json on disk.
    project_dir = _make_project_dir(
        tmp_path,
        with_success_row=True,
        with_metrics=False,   # no metrics.json at all
    )
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    report = {
        "evidence_gate_passed": True,        # Gate 1 passes
        "rubric": {
            "overall_score": 0.95,            # very high LLM grade — the red-line temptation
            "target_score": 0.60,
            "meets_target": True,             # LLM says the target was met
        },
        # NOTE: no "deterministic_meets_target", no "paper_claimed_target",
        # no "measured_headline" — purely an LLM-graded signal.
        "scope": {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]},
        "arxiv_id": "2605.15155",
    }

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_clean_validator_verdict(),   # Gate 3 passes
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == [], (
        "RED LINE REGRESSION: a high LLM grade (rubric.overall_score=0.95, "
        "rubric.meets_target=True) must NEVER be the admission signal. "
        "Without deterministic_meets_target or a measured_headline+paper_claimed_target "
        "comparison backed by real code/metrics.json, gate 4 must refuse."
    )


def test_regression_validator_vetoed_blocks_even_with_clean_evidence(monkeypatch, tmp_path):
    """REGRESSION: Validator veto must block admission even when all deterministic evidence
    is clean.

    The validator can discover adversarial patterns that pass the local predicates.
    A 'vetoed' verdict must be an unconditional hard block — NOT overridable by any
    other gate combination.
    """
    _enable_recipes(monkeypatch)
    project_dir = _make_project_dir(
        tmp_path,
        with_success_row=True,
        with_metrics=True,
        metrics=_REAL_METRICS,
    )
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    # Perfect deterministic evidence — but the validator found a problem.
    report = {
        "evidence_gate_passed": True,
        "deterministic_meets_target": True,
        "paper_claimed_target": 0.60,
        "measured_headline": 0.85,
        "rubric": {"overall_score": 0.85, "target_score": 0.60},
        "scope": {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]},
        "arxiv_id": "2605.15155",
    }

    admit_recipe(
        project_dir, runs_root,
        report=report,
        validator_verdict=_vetoed_validator_verdict(),   # Gate 3 must veto unconditionally
        paper_class="sdar_agentic_rl",
    )

    records = _read_recipes(runs_root, "sdar_agentic_rl")
    assert records == [], (
        "VALIDATOR REGRESSION: a 'vetoed' validator verdict must block admission unconditionally, "
        "even when all other gates pass and deterministic evidence looks clean. "
        "The validator's min-aggregation veto is a hard block — not advisory."
    )
