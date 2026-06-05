"""Scope-exclusion signals beyond models/datasets (2026-05-31 fix).

The agent declares out-of-scope work as structured scope in ``metrics.json``:
``scope.environments_skipped`` (e.g. ALFWorld/WebShop for a Search-QA-only run)
and ``scope.gaps`` entries that are ``{"item": ..., "reason": ...}`` dicts.
Before this fix ``_detect_data_unavailable_leaves`` honoured only
``scope.models_skipped`` and *string* gaps read from ``final_report.json`` — so
honestly de-scoped environments and every dict-form gap were ignored, scored
0.0, and dragged the overall rubric score down (the SDAR smallest-two run was
capped at ~0.33 despite a correct in-scope reproduction)."""
from __future__ import annotations

import json
from pathlib import Path

from backend.evals.paperbench.leaf_scorer import _detect_data_unavailable_leaves


def _write_metrics(run_dir: Path, metrics: dict) -> None:
    out = run_dir / "code" / "outputs" / "run1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def _write_report(run_dir: Path, report: dict) -> None:
    (run_dir / "final_report.json").write_text(json.dumps(report), encoding="utf-8")


# A de-scoped environment leaf, a de-scoped method leaf, and an in-scope SDAR-core
# leaf that must NEVER be excluded by these signals.
LEAVES = [
    {"id": "leaf_alfworld", "requirements": "ALFWorld success rate reaches 53.9% on Qwen3-1.7B"},
    {"id": "leaf_webshop", "requirements": "WebShop training uses 1000 tasks and a 128-task validation set"},
    {"id": "leaf_skill", "requirements": "Skill retrieval implements UCB, KM, Full and Random strategies"},
    {"id": "leaf_core", "requirements": "The sigmoid gate g_t uses beta=10 with stop-gradient on the gate"},
]


def test_environments_skipped_excludes_env_leaves(tmp_path: Path):
    _write_metrics(tmp_path, {"scope": {"environments_skipped": ["alfworld", "webshop"]}})
    skip = _detect_data_unavailable_leaves(LEAVES, tmp_path)
    assert "leaf_alfworld" in skip
    assert "leaf_webshop" in skip
    assert "leaf_core" not in skip          # in-scope SDAR leaf untouched
    assert "leaf_skill" not in skip         # not declared → still graded


def test_dict_form_gap_items_excluded_from_metrics(tmp_path: Path):
    _write_metrics(tmp_path, {"scope": {"gaps": [
        {"item": "alfworld", "reason": "out-of-scope per operator (Search-QA only)"},
        {"item": "skill retrieval", "reason": "SkillRL contribution out of scope"},
    ]}})
    skip = _detect_data_unavailable_leaves(LEAVES, tmp_path)
    assert "leaf_alfworld" in skip
    assert "leaf_skill" in skip
    assert "leaf_core" not in skip


def test_string_form_gap_in_final_report_still_works(tmp_path: Path):
    # Backward compatibility: prose strings in final_report.json::scope.gaps.
    _write_metrics(tmp_path, {"status": "completed"})
    _write_report(tmp_path, {"scope": {"gaps": ["WebShop — out of scope for this run"]}})
    skip = _detect_data_unavailable_leaves(LEAVES, tmp_path)
    assert "leaf_webshop" in skip
    assert "leaf_core" not in skip


def test_dict_gap_in_final_report_is_read(tmp_path: Path):
    _write_metrics(tmp_path, {"status": "completed"})
    _write_report(tmp_path, {"scope": {"gaps": [{"item": "webshop", "reason": "de-scoped"}]}})
    skip = _detect_data_unavailable_leaves(LEAVES, tmp_path)
    assert "leaf_webshop" in skip


def test_no_scope_signals_excludes_nothing(tmp_path: Path):
    _write_metrics(tmp_path, {"status": "completed", "scope": {"models_run": ["qwen3_1_7b"]}})
    skip = _detect_data_unavailable_leaves(LEAVES, tmp_path)
    assert skip == set()


def test_core_leaf_never_excluded_by_environment_signal(tmp_path: Path):
    # Even with several de-scoped components declared, the in-scope core leaf
    # (no env/method token in its text) is graded, not skipped.
    _write_metrics(tmp_path, {"scope": {
        "models_skipped": ["qwen2_5_7b"],
        "environments_skipped": ["alfworld", "webshop"],
        "gaps": [{"item": "skill retrieval", "reason": "x"}, {"item": "grpo_baseline_run", "reason": "y"}],
    }})
    skip = _detect_data_unavailable_leaves(LEAVES, tmp_path)
    assert "leaf_core" not in skip
    assert {"leaf_alfworld", "leaf_webshop", "leaf_skill"} <= skip
