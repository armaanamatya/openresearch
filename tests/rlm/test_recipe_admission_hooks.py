"""Tests for recipe admission + injection wiring (Task P4.2).

Covers:
- admit_recipe writes a recipe when OPENRESEARCH_POSITIVE_RECIPES=1 and a clean
  report + success ledger row + real metrics.json are supplied.
- Flag OFF → admit_recipe writes nothing (no _recipes dir, byte-identical).
- recipe_guidance_block returns the injected block (flag on) / "" (flag off).
- Import check: run.py and baseline_implementation.py import cleanly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_REAL_METRICS = {"reward": 0.42, "return": 31.1}


def _make_project_dir(tmp_path: Path, *, with_success_row: bool = True, with_metrics: bool = True) -> Path:
    """Return a minimal project_dir with required on-disk artifacts."""
    proj = tmp_path / "proj"
    proj.mkdir()
    if with_success_row:
        row = json.dumps({"success": True, "metrics": _REAL_METRICS})
        (proj / "experiment_runs.jsonl").write_text(row + "\n", encoding="utf-8")
    if with_metrics:
        code = proj / "code"
        code.mkdir()
        (code / "metrics.json").write_text(json.dumps(_REAL_METRICS), encoding="utf-8")
    return proj


def _clean_report(*, evidence_gate_passed: bool = True) -> dict:
    """Minimal clean report dict that passes all four admission gates."""
    return {
        "evidence_gate_passed": evidence_gate_passed,
        # deterministic_meets_target=True is the fastest Tier-1 path; it lets
        # tests that do not need an on-disk metrics.json still exercise admission.
        "deterministic_meets_target": True,
        "rubric": {
            "overall_score": 0.75,
            "target_score": 0.60,
        },
        "scope": {"models": ["Qwen3-1.7B"], "datasets": ["ALFWorld"]},
        "arxiv_id": "2605.15155",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Import-clean check
# ──────────────────────────────────────────────────────────────────────────────

def test_run_module_imports_cleanly() -> None:
    """run.py must import without errors (no side-effects from recipe wiring)."""
    import backend.agents.rlm.run  # noqa: F401


def test_baseline_implementation_imports_cleanly() -> None:
    """baseline_implementation.py must import without errors."""
    import backend.agents.baseline_implementation  # noqa: F401


# ──────────────────────────────────────────────────────────────────────────────
# Default-OFF: with flag unset, no _recipes dir is ever created
# ──────────────────────────────────────────────────────────────────────────────

def test_admit_recipe_flag_off_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With OPENRESEARCH_POSITIVE_RECIPES unset, admit_recipe is a no-op."""
    monkeypatch.delenv("OPENRESEARCH_POSITIVE_RECIPES", raising=False)
    from backend.agents.rlm.recipe_library import admit_recipe

    proj = _make_project_dir(tmp_path)
    runs_root = tmp_path

    admit_recipe(proj, runs_root, report=_clean_report(), validator_verdict=None, paper_class="sdar_agentic_rl")

    assert not (runs_root / "_recipes").exists(), "_recipes dir must not be created when flag is OFF"


def test_recipe_guidance_block_flag_off_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """recipe_guidance_block returns '' when flag is unset."""
    monkeypatch.delenv("OPENRESEARCH_POSITIVE_RECIPES", raising=False)
    from backend.agents.rlm.recipe_library import recipe_guidance_block

    result = recipe_guidance_block(tmp_path, "sdar_agentic_rl")
    assert result == ""


# ──────────────────────────────────────────────────────────────────────────────
# Flag ON: admit_recipe writes a recipe; guidance block returns non-empty
# ──────────────────────────────────────────────────────────────────────────────

def test_admit_recipe_flag_on_writes_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With OPENRESEARCH_POSITIVE_RECIPES=1 and a clean report, admit_recipe
    should write a recipe file under runs_root/_recipes/."""
    monkeypatch.setenv("OPENRESEARCH_POSITIVE_RECIPES", "1")
    from backend.agents.rlm.recipe_library import admit_recipe, _store_path

    proj = _make_project_dir(tmp_path)
    runs_root = tmp_path
    paper_class = "sdar_agentic_rl"

    admit_recipe(proj, runs_root, report=_clean_report(), validator_verdict=None, paper_class=paper_class)

    store = _store_path(runs_root, paper_class)
    assert store.exists(), "Recipe file must be created when flag is ON and gates pass"
    records = json.loads(store.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["paper_class"] == paper_class


def test_recipe_guidance_block_flag_on_returns_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After admitting a recipe, recipe_guidance_block should return a non-empty string."""
    monkeypatch.setenv("OPENRESEARCH_POSITIVE_RECIPES", "1")
    from backend.agents.rlm.recipe_library import admit_recipe, recipe_guidance_block

    proj = _make_project_dir(tmp_path)
    runs_root = tmp_path
    paper_class = "sdar_agentic_rl"

    admit_recipe(proj, runs_root, report=_clean_report(), validator_verdict=None, paper_class=paper_class)

    block = recipe_guidance_block(runs_root, paper_class)
    assert block != "", "guidance block must be non-empty after a recipe is admitted"
    assert "POSITIVE RECIPES" in block


def test_admit_recipe_evidence_gate_failed_not_admitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A report that failed the evidence_gate must never be admitted (red line)."""
    monkeypatch.setenv("OPENRESEARCH_POSITIVE_RECIPES", "1")
    from backend.agents.rlm.recipe_library import admit_recipe, _store_path

    proj = _make_project_dir(tmp_path)
    runs_root = tmp_path
    paper_class = "sdar_agentic_rl"

    bad_report = _clean_report(evidence_gate_passed=False)
    admit_recipe(proj, runs_root, report=bad_report, validator_verdict=None, paper_class=paper_class)

    store = _store_path(runs_root, paper_class)
    assert not store.exists(), "No recipe should be written when evidence_gate failed"


def test_admit_recipe_vetoed_validator_not_admitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A vetoed validator verdict must block admission regardless of other gates."""
    monkeypatch.setenv("OPENRESEARCH_POSITIVE_RECIPES", "1")
    from backend.agents.rlm.recipe_library import admit_recipe, _store_path

    proj = _make_project_dir(tmp_path)
    runs_root = tmp_path
    paper_class = "sdar_agentic_rl"

    admit_recipe(proj, runs_root, report=_clean_report(), validator_verdict={"status": "vetoed"}, paper_class=paper_class)

    store = _store_path(runs_root, paper_class)
    assert not store.exists(), "No recipe should be written when validator vetoed"
