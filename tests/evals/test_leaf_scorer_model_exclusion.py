"""A failed/skipped MODEL is excluded from the rubric (numerator + denominator)
exactly like an unobtainable dataset — graceful degradation so a model that
could not be loaded/run does not drag the score to zero (2026-05-30 mandate).

Updated 2026-05-31: tests now pass operator_skip_models so the intentional-skip
vs. code-bug distinction is exercised correctly.  An operator-intended skip is
excluded; a requested-but-failed model is NOT excluded (it's a repairable bug).
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.evals.paperbench.leaf_scorer import _detect_data_unavailable_leaves


def _write(run_dir: Path, metrics: dict) -> None:
    out = run_dir / "code" / "outputs" / "run1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


# Leaves: one bound to the 7B (should be excluded), one to the 3B (kept), one paper-wide (kept).
LEAVES = [
    {"id": "leaf_7b", "requirements": "Qwen2.5-7B reaches SDAR reward above GRPO baseline"},
    {"id": "leaf_3b", "requirements": "Qwen2.5-3B reaches SDAR reward above GRPO baseline"},
    {"id": "leaf_algo", "requirements": "The sigmoid gate g_t uses beta=10 with stop-gradient"},
]


def test_model_load_failed_excludes_that_models_leaf(tmp_path: Path):
    # Operator intentionally de-scoped the 7B → its leaf should be excluded.
    _write(tmp_path, {"per_model": {"qwen2_5_7b": {"status": "model_load_failed"}}})
    skip = _detect_data_unavailable_leaves(
        LEAVES, tmp_path, operator_skip_models=["qwen2_5_7b"]
    )
    assert "leaf_7b" in skip          # operator-skipped 7B → its leaf excluded
    assert "leaf_3b" not in skip      # 3B ran → kept
    assert "leaf_algo" not in skip    # paper-wide → kept


def test_models_skipped_scope_reduction_excludes(tmp_path: Path):
    # Operator explicitly de-scoped 7B → excluded.
    _write(tmp_path, {"scope": {"models_skipped": ["qwen2_5_7b"]}})
    skip = _detect_data_unavailable_leaves(
        LEAVES, tmp_path, operator_skip_models=["qwen2_5_7b"]
    )
    assert "leaf_7b" in skip
    assert "leaf_3b" not in skip


def test_no_failures_skips_nothing(tmp_path: Path):
    _write(tmp_path, {"per_model": {"qwen2_5_7b": {"status": "ok"}}})
    skip = _detect_data_unavailable_leaves(LEAVES, tmp_path)
    assert skip == set()


def test_mixed_dataset_and_model_both_excluded(tmp_path: Path):
    # Operator de-scoped 7B; dataset also failed independently.
    _write(tmp_path, {
        "data_load_failures": [{"dataset": "webshop"}],
        "per_model": {"qwen2_5_7b": {"status": "model_load_failed"}},
    })
    leaves = LEAVES + [{"id": "leaf_ws", "requirements": "WebShop task success rate matches paper"}]
    skip = _detect_data_unavailable_leaves(
        leaves, tmp_path, operator_skip_models=["qwen2_5_7b"]
    )
    assert "leaf_7b" in skip and "leaf_ws" in skip
    assert "leaf_3b" not in skip and "leaf_algo" not in skip
