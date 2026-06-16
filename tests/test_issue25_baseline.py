"""Tests for Issue #25: Baseline Implementation Agent."""

from pathlib import Path
import json

from backend.agents.baseline_implementation import run_offline, PPO_TRAIN_PY
from backend.agents.schemas import (
    BaselineResult,
    DatasetRequirement,
    EnvironmentSpec,
    MetricSpec,
    PaperClaimMap,
    TrainingRecipe,
)


def _ppo_claim_map():
    return PaperClaimMap(
        core_contribution="PPO for RL",
        datasets=[DatasetRequirement(name="CartPole-v1")],
        metrics=[MetricSpec(name="mean_reward", definition="Mean over 100 eps")],
        training_recipe=TrainingRecipe(optimizer="Adam", learning_rate="3e-4"),
    )


def _env_spec():
    return EnvironmentSpec(
        dockerfile="FROM python:3.11-slim\nRUN pip install torch==2.2.0",
        python_version="3.11",
        framework="pytorch",
        framework_version="2.2.0",
    )


class TestRunOffline:
    def test_produces_baseline_result(self, tmp_path: Path):
        result = run_offline("prj_ppo", tmp_path, _ppo_claim_map(), _env_spec())
        assert isinstance(result, BaselineResult)

    def test_mode_is_implement_from_paper(self, tmp_path: Path):
        result = run_offline("prj_ppo", tmp_path, _ppo_claim_map(), _env_spec())
        assert result.mode == "implement_from_paper"

    def test_writes_train_py(self, tmp_path: Path):
        run_offline("prj_ppo", tmp_path, _ppo_claim_map(), _env_spec())
        assert (tmp_path / "prj_ppo" / "code" / "train.py").exists()

    def test_writes_config_json(self, tmp_path: Path):
        run_offline("prj_ppo", tmp_path, _ppo_claim_map(), _env_spec())
        config_path = tmp_path / "prj_ppo" / "code" / "config.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["env_id"] == "CartPole-v1"
        assert config["adam_epsilon"] == 1e-5  # A001

    def test_applies_all_8_assumptions(self, tmp_path: Path):
        result = run_offline("prj_ppo", tmp_path, _ppo_claim_map(), _env_spec())
        assert len(result.assumptions_applied) == 8
        for i in range(1, 9):
            assert f"A{i:03d}" in result.assumptions_applied

    def test_writes_dockerfile(self, tmp_path: Path):
        run_offline("prj_ppo", tmp_path, _ppo_claim_map(), _env_spec())
        assert (tmp_path / "prj_ppo" / "code" / "Dockerfile").exists()

    def test_writes_commands_log(self, tmp_path: Path):
        run_offline("prj_ppo", tmp_path, _ppo_claim_map(), _env_spec())
        assert (tmp_path / "prj_ppo" / "code" / "commands.log").exists()

    def test_has_run_commands(self, tmp_path: Path):
        result = run_offline("prj_ppo", tmp_path, _ppo_claim_map(), _env_spec())
        assert len(result.commands_to_run) >= 1
        assert any("train.py" in cmd for cmd in result.commands_to_run)


class TestPPOImplementation:
    """Verify the generated PPO code has correct structure."""

    def test_train_py_has_ppo_components(self):
        assert "ActorCritic" in PPO_TRAIN_PY
        assert "def train" in PPO_TRAIN_PY
        assert "clip_range" in PPO_TRAIN_PY

    def test_train_py_applies_A001_adam_epsilon(self):
        assert "adam_epsilon" in PPO_TRAIN_PY
        assert "1e-5" in PPO_TRAIN_PY

    def test_train_py_applies_A002_orthogonal_init(self):
        assert "orthogonal_" in PPO_TRAIN_PY
        assert "layer_init" in PPO_TRAIN_PY

    def test_train_py_applies_A003_lr_schedule(self):
        assert "linear_schedule" in PPO_TRAIN_PY

    def test_train_py_applies_A004_per_minibatch_norm(self):
        assert "mb_advantages" in PPO_TRAIN_PY
        assert "mb_advantages.mean()" in PPO_TRAIN_PY

    def test_train_py_applies_A005_value_clipping(self):
        assert "v_loss_clipped" in PPO_TRAIN_PY

    def test_train_py_applies_A006_grad_clipping(self):
        assert "clip_grad_norm_" in PPO_TRAIN_PY

    def test_train_py_writes_metrics_json(self):
        assert "metrics.json" in PPO_TRAIN_PY

    def test_train_py_uses_gymnasium(self):
        assert "gymnasium" in PPO_TRAIN_PY or "gym" in PPO_TRAIN_PY
