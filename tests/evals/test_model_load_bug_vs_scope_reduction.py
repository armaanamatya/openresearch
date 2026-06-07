"""Distinguish intentional operator scope-reduction from agent code bugs masked
as scope reductions (2026-05-31 fix).

ROOT CAUSE: a SDAR run scored 0.188 because the agent's train.py wrapped model
loading in a broad ``except Exception`` and dumped BOTH Qwen models into
``scope.models_skipped`` (real errors: transformers architecture error +
``__init__() got an unexpected keyword argument 'dtype'``). The harness then
silently excluded every downstream leaf → 0.188.

THE DISTINCTION:
  - Operator-intended skip (in ``ScopeSpec.skip_models``) → legitimately
    excluded from rubric numerator AND denominator.
  - Requested model that appeared in models_skipped/model_load_failures/
    per_model[*].status==failed WITHOUT an operator-intended skip → code bug.
    NOT silently excluded; leaves stay in rubric so the score is honest.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.evals.paperbench.leaf_scorer import (
    _detect_data_unavailable_leaves,
    _normalise_model_name,
    _operator_skip_set,
)
from backend.agents.rlm.report import (
    _collect_data_unavailable_gaps,
    _operator_skip_set as report_operator_skip_set,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_metrics(run_dir: Path, metrics: dict) -> None:
    out = run_dir / "code" / "outputs" / "run1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


# SDAR-representative leaves: one per model + one algorithm-wide core leaf
SDAR_LEAVES = [
    {"id": "leaf_1_7b", "requirements": "Qwen3-1.7B SDAR reward exceeds GRPO baseline"},
    {"id": "leaf_3b",   "requirements": "Qwen2.5-3B SDAR reward exceeds GRPO baseline"},
    {"id": "leaf_7b",   "requirements": "Qwen2.5-7B SDAR reward exceeds GRPO baseline"},
    {"id": "leaf_core", "requirements": "The sigmoid gate g_t uses beta=10 with stop-gradient"},
]


# ---------------------------------------------------------------------------
# _operator_skip_set helpers
# ---------------------------------------------------------------------------


def test_operator_skip_set_empty():
    assert _operator_skip_set(None) == frozenset()
    assert _operator_skip_set([]) == frozenset()


def test_operator_skip_set_normalises_case():
    s = _operator_skip_set(["Qwen2.5-7B-Instruct", "  QWEN3-1.7B  "])
    assert "qwen2.5-7b-instruct" in s
    assert "qwen3-1.7b" in s


# ---------------------------------------------------------------------------
# _detect_data_unavailable_leaves: intentional vs code-bug
# ---------------------------------------------------------------------------


class TestOperatorIntendedSkip:
    """Operator-intended skips ARE excluded from the rubric (no change)."""

    def test_models_skipped_with_operator_skip_excludes_leaf(self, tmp_path: Path):
        # Operator says "skip 7B"; agent puts 7B in models_skipped → legitimate.
        _write_metrics(tmp_path, {"scope": {"models_skipped": ["qwen2_5_7b"]}})
        skip = _detect_data_unavailable_leaves(
            SDAR_LEAVES, tmp_path, operator_skip_models=["qwen2_5_7b"]
        )
        assert "leaf_7b" in skip, "operator-skipped model's leaf must be excluded"
        assert "leaf_1_7b" not in skip
        assert "leaf_3b" not in skip
        assert "leaf_core" not in skip

    def test_per_model_failed_with_operator_skip_excludes_leaf(self, tmp_path: Path):
        # Operator says skip 7B; per_model reports 7B as failed → legitimate.
        _write_metrics(tmp_path, {"per_model": {"qwen2_5_7b": {"status": "model_load_failed"}}})
        skip = _detect_data_unavailable_leaves(
            SDAR_LEAVES, tmp_path, operator_skip_models=["qwen2_5_7b"]
        )
        assert "leaf_7b" in skip

    def test_model_load_failures_with_operator_skip_excludes_leaf(self, tmp_path: Path):
        _write_metrics(tmp_path, {"model_load_failures": [{"model": "qwen2_5_7b", "error": "404"}]})
        skip = _detect_data_unavailable_leaves(
            SDAR_LEAVES, tmp_path, operator_skip_models=["Qwen2.5-7B-Instruct"]
            # Note: name normalisation means "qwen2_5_7b" and "qwen2.5-7b-instruct" are
            # both kept as-is after lower+strip; they won't match each other.
            # This test uses exact normalised match.
        )
        # "qwen2_5_7b" normalised == "qwen2_5_7b"; operator list "Qwen2.5-7B-Instruct"
        # normalises to "qwen2.5-7b-instruct" → they DON'T match → treated as bug.
        # This is intentional: operator skip lists should use the SAME identifier
        # the agent emits. When they don't match, be conservative (treat as bug).
        assert "leaf_7b" not in skip  # mismatch → NOT excluded (conservative)

    def test_exact_name_match_excludes(self, tmp_path: Path):
        # When operator name and agent name match exactly after normalisation.
        _write_metrics(tmp_path, {"model_load_failures": [{"model": "qwen2_5_7b"}]})
        skip = _detect_data_unavailable_leaves(
            SDAR_LEAVES, tmp_path, operator_skip_models=["qwen2_5_7b"]
        )
        assert "leaf_7b" in skip


class TestRequestedModelLoadBug:
    """Requested models that failed to load are NOT silently excluded — code bug."""

    def test_models_skipped_without_operator_skip_is_not_excluded(self, tmp_path: Path):
        # Agent dumps both 1.7B and 3B into models_skipped after a broad
        # except-Exception catch — but the operator never asked to skip them.
        # The harness MUST NOT exclude their leaves; the bug must stay visible.
        _write_metrics(tmp_path, {
            "scope": {"models_skipped": ["qwen3_1_7b", "qwen2_5_3b"]}
        })
        skip = _detect_data_unavailable_leaves(
            SDAR_LEAVES, tmp_path,
            operator_skip_models=[],  # operator skipped nothing
        )
        assert "leaf_1_7b" not in skip, "requested model must NOT be silently excluded"
        assert "leaf_3b" not in skip, "requested model must NOT be silently excluded"
        assert "leaf_core" not in skip

    def test_per_model_failed_without_operator_skip_not_excluded(self, tmp_path: Path):
        # per_model shows load failure but operator did not skip these models.
        _write_metrics(tmp_path, {
            "per_model": {
                "qwen3_1_7b": {"status": "model_load_failed", "error": "unexpected keyword arg"},
                "qwen2_5_3b": {"status": "model_load_failed", "error": "architecture mismatch"},
            }
        })
        skip = _detect_data_unavailable_leaves(
            SDAR_LEAVES, tmp_path, operator_skip_models=[]
        )
        assert "leaf_1_7b" not in skip
        assert "leaf_3b" not in skip
        assert "leaf_core" not in skip

    def test_model_load_failures_without_operator_skip_not_excluded(self, tmp_path: Path):
        _write_metrics(tmp_path, {
            "model_load_failures": [
                {"model": "qwen3_1_7b", "error": "dtype kwarg"},
                {"model": "qwen2_5_3b", "error": "architecture error"},
            ]
        })
        skip = _detect_data_unavailable_leaves(
            SDAR_LEAVES, tmp_path, operator_skip_models=[]
        )
        assert "leaf_1_7b" not in skip
        assert "leaf_3b" not in skip

    def test_no_operator_skip_list_means_no_exclusion(self, tmp_path: Path):
        # When operator_skip_models is None (unknown intent), the harness has no
        # operator-declared skip list to compare against.  The safe default is
        # to treat all failure signals as "not operator-intended" → no exclusion.
        # This is strictly more correct than the old behaviour of blindly excluding
        # everything: it avoids laundering code bugs into fake scope reductions
        # even for callers that haven't been updated to pass the skip list.
        _write_metrics(tmp_path, {"scope": {"models_skipped": ["qwen2_5_7b"]}})
        skip = _detect_data_unavailable_leaves(SDAR_LEAVES, tmp_path)
        # No operator list supplied → no exclusion (strict/safe default).
        assert "leaf_7b" not in skip

    def test_mixed_operator_skip_and_code_bug(self, tmp_path: Path):
        # Operator intentionally skips 7B. Agent bugs out on 1.7B and 3B.
        # Only 7B should be excluded; 1.7B and 3B stay in scoring.
        _write_metrics(tmp_path, {
            "scope": {
                "models_skipped": ["qwen2_5_7b", "qwen3_1_7b", "qwen2_5_3b"]
            }
        })
        skip = _detect_data_unavailable_leaves(
            SDAR_LEAVES, tmp_path,
            operator_skip_models=["qwen2_5_7b"],
        )
        assert "leaf_7b" in skip, "operator-skipped 7B must be excluded"
        assert "leaf_1_7b" not in skip, "requested 1.7B must NOT be excluded"
        assert "leaf_3b" not in skip, "requested 3B must NOT be excluded"
        assert "leaf_core" not in skip


# ---------------------------------------------------------------------------
# _collect_data_unavailable_gaps: phrasing distinction in report
# ---------------------------------------------------------------------------


class TestGapPhrasing:
    """Gap strings correctly distinguish intentional scope vs repairable bug."""

    def test_operator_skipped_model_uses_unavailable_phrasing(self, tmp_path: Path):
        _write_metrics(tmp_path, {"scope": {"models_skipped": ["qwen2_5_7b"]}})
        gaps = _collect_data_unavailable_gaps(tmp_path, operator_skip_models=["qwen2_5_7b"])
        assert any("model unavailable" in g and "excluded from rubric score" in g for g in gaps), (
            f"Operator-skipped model should say 'model unavailable ... excluded'. Got: {gaps}"
        )
        assert not any("repairable" in g for g in gaps), (
            "Operator-skipped model should NOT say 'repairable'. Got: {gaps}"
        )

    def test_code_bug_model_uses_repairable_phrasing(self, tmp_path: Path):
        _write_metrics(tmp_path, {"scope": {"models_skipped": ["qwen3_1_7b"]}})
        gaps = _collect_data_unavailable_gaps(tmp_path, operator_skip_models=[])
        assert any("repairable code bug" in g for g in gaps), (
            f"Code-bug model should say 'repairable code bug'. Got: {gaps}"
        )
        assert not any("excluded from rubric score, not penalised" in g for g in gaps
                       if "repairable" not in g), (
            f"Code-bug model must not claim 'excluded from rubric score, not penalised'. Got: {gaps}"
        )

    def test_code_bug_model_says_fix_and_rerun(self, tmp_path: Path):
        _write_metrics(tmp_path, {
            "per_model": {"qwen3_1_7b": {"status": "model_load_failed",
                                          "error": "unexpected keyword argument dtype"}}
        })
        gaps = _collect_data_unavailable_gaps(tmp_path, operator_skip_models=[])
        assert any("fix the loader" in g or "NOT excluded" in g for g in gaps), (
            f"Code-bug gap must instruct repair. Got: {gaps}"
        )

    def test_dataset_unobtainable_phrasing_unchanged(self, tmp_path: Path):
        # Dataset gaps are not affected by the model-skip fix.
        _write_metrics(tmp_path, {"data_load_failures": [{"dataset": "webshop", "error": "404"}]})
        gaps = _collect_data_unavailable_gaps(tmp_path)
        assert any("dataset unobtainable" in g and "webshop" in g for g in gaps)

    def test_no_operator_skip_list_preserves_old_phrasing(self, tmp_path: Path):
        # Backward compat: no operator_skip_models → old "model unavailable" phrasing
        # (all failures treated as scope reductions, matching pre-fix behaviour).
        _write_metrics(tmp_path, {"scope": {"models_skipped": ["qwen2_5_7b"]}})
        gaps = _collect_data_unavailable_gaps(tmp_path)
        # Old callers get old phrasing — nothing should explode.
        assert len(gaps) >= 1
        assert any("qwen2_5_7b" in g for g in gaps)

    def test_mixed_operator_skip_and_code_bug_phrasing(self, tmp_path: Path):
        # 7B intentionally skipped by operator; 3B is a code bug.
        _write_metrics(tmp_path, {
            "scope": {"models_skipped": ["qwen2_5_7b", "qwen2_5_3b"]}
        })
        gaps = _collect_data_unavailable_gaps(tmp_path, operator_skip_models=["qwen2_5_7b"])
        seven_b_gaps = [g for g in gaps if "qwen2_5_7b" in g]
        three_b_gaps = [g for g in gaps if "qwen2_5_3b" in g]
        assert any("model unavailable" in g and "excluded from rubric score" in g
                   for g in seven_b_gaps), "7B should be 'model unavailable'"
        assert any("repairable code bug" in g for g in three_b_gaps), "3B should be 'repairable code bug'"
