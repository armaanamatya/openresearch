"""Tests for Issue #28: Improvement Orchestrator + Path Agents."""

from pathlib import Path
import json
import pytest

from backend.agents.improvement import (
    select_hypotheses_offline,
    run_path_offline,
)
from backend.agents.schemas import (
    DatasetRequirement,
    ImprovementHypothesis,
    MetricSpec,
    PaperClaimMap,
    PathResult,
    RiskLevel,
    TrainingRecipe,
)


def _claim_map():
    return PaperClaimMap(
        core_contribution="PPO",
        datasets=[DatasetRequirement(name="CartPole-v1")],
        metrics=[MetricSpec(name="mean_reward", definition="Mean over 100 eps")],
        training_recipe=TrainingRecipe(optimizer="Adam", learning_rate="3e-4"),
    )


class TestSelectHypotheses:
    def test_returns_3_hypotheses_by_default(self):
        hypotheses = select_hypotheses_offline(_claim_map(), {"mean_reward": 487})
        assert len(hypotheses) == 3

    def test_hypotheses_have_ids(self):
        hypotheses = select_hypotheses_offline(_claim_map(), {"mean_reward": 487})
        ids = [h.path_id for h in hypotheses]
        assert ids == ["path_1", "path_2", "path_3"]

    def test_hypotheses_have_rationale(self):
        hypotheses = select_hypotheses_offline(_claim_map(), {"mean_reward": 487})
        for h in hypotheses:
            assert len(h.rationale) > 20
            assert len(h.hypothesis) > 10

    def test_user_hints_override(self):
        hints = ["Try learning rate 1e-3", "Double batch size"]
        hypotheses = select_hypotheses_offline(
            _claim_map(), {"mean_reward": 487}, user_hints=hints,
        )
        assert "learning rate" in hypotheses[0].hypothesis.lower()
        assert "batch size" in hypotheses[1].hypothesis.lower()

    def test_custom_n_paths(self):
        hypotheses = select_hypotheses_offline(
            _claim_map(), {"mean_reward": 487}, n_paths=2,
        )
        assert len(hypotheses) == 2


class TestRunPathOffline:
    def test_success_path(self, tmp_path: Path):
        h = ImprovementHypothesis(
            path_id="path_1",
            hypothesis="Test entropy",
            rationale="reason",
            expected_outcome="better",
        )
        result = run_path_offline("prj_ppo", tmp_path, h, {"mean_reward": 487})
        assert isinstance(result, PathResult)
        assert result.success is True
        assert result.metrics.get("mean_reward", 0) > 487

    def test_failure_path(self, tmp_path: Path):
        h = ImprovementHypothesis(
            path_id="path_1",
            hypothesis="Test",
            rationale="reason",
            expected_outcome="better",
        )
        result = run_path_offline(
            "prj_ppo", tmp_path, h, {"mean_reward": 487}, simulate_success=False,
        )
        assert result.success is False
        assert result.failure_notes != ""

    def test_creates_path_directory(self, tmp_path: Path):
        h = ImprovementHypothesis(
            path_id="path_1",
            hypothesis="Test",
            rationale="reason",
            expected_outcome="better",
        )
        run_path_offline("prj_ppo", tmp_path, h, {"mean_reward": 487})
        assert (tmp_path / "prj_ppo" / "improvements" / "path_1").exists()

    def test_path_2_shows_regression(self, tmp_path: Path):
        h = ImprovementHypothesis(
            path_id="path_2",
            hypothesis="Separate networks",
            rationale="test",
            expected_outcome="maybe worse",
        )
        result = run_path_offline("prj_ppo", tmp_path, h, {"mean_reward": 487})
        # Path 2 (separate networks) is simulated as a regression
        assert result.metrics.get("improvement", 0) < 0

    def test_has_recommendation(self, tmp_path: Path):
        h = ImprovementHypothesis(
            path_id="path_1",
            hypothesis="Test",
            rationale="reason",
            expected_outcome="better",
        )
        result = run_path_offline("prj_ppo", tmp_path, h, {"mean_reward": 487})
        assert result.recommendation != ""


class TestRiskLevelCoercion:
    """Regression: LLM sometimes emits trailing rationale in the enum field.

    A run failed at orchestrator.py with `risk="low — scripts already required
    for CartPole-v1"`. RiskLevel-typed fields must coerce to the bare token.
    """

    def _hypothesis_with_risk(self, risk_value):
        return ImprovementHypothesis(
            path_id="path_1",
            hypothesis="h",
            rationale="r",
            expected_outcome="o",
            risk=risk_value,
        )

    def test_clean_value_passes(self):
        assert self._hypothesis_with_risk("low").risk == RiskLevel.low

    def test_trailing_em_dash_rationale_coerces(self):
        h = self._hypothesis_with_risk("low — scripts already required")
        assert h.risk == RiskLevel.low

    def test_trailing_en_dash_rationale_coerces(self):
        h = self._hypothesis_with_risk("high – needs new dataset")
        assert h.risk == RiskLevel.high

    def test_colon_rationale_coerces(self):
        h = self._hypothesis_with_risk("medium: untested codepath")
        assert h.risk == RiskLevel.medium

    def test_whitespace_rationale_coerces(self):
        h = self._hypothesis_with_risk("critical because of GPU memory")
        assert h.risk == RiskLevel.critical

    def test_uppercase_coerces(self):
        h = self._hypothesis_with_risk("LOW")
        assert h.risk == RiskLevel.low

    def test_still_invalid_after_coercion_raises(self):
        with pytest.raises(Exception):
            self._hypothesis_with_risk("severe")
