"""Tests for Issue #23: Paper Understanding Agent.

Validates:
- Offline mode produces structured PaperClaimMap from PPO workspace data
- Ambiguity detection finds key missing details
- Output is written to disk correctly
- Heuristic extractors work on typical paper sections
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.paper_understanding import (
    run_offline,
    _extract_ambiguities,
    _extract_claims,
    _extract_contribution,
    _extract_datasets,
    _extract_metrics,
    _extract_training_recipe,
)
from backend.agents.schemas import PaperClaimMap, RiskLevel


# Simulate the workspace claim_map from the existing ingestion pipeline
PPO_WORKSPACE_CLAIM_MAP = {
    "project_id": "prj_test_ppo",
    "entries": [
        {
            "source_id": "src_001",
            "title": "Abstract",
            "excerpt": (
                "We propose a new family of policy gradient methods for reinforcement learning, which "
                "alternate between sampling data through interaction with the environment, and optimizing a "
                "surrogate objective function using stochastic gradient ascent. Whereas standard policy "
                "gradient methods perform one gradient update per data sample, we propose a new objective "
                "function that enables multiple epochs of minibatch updates."
            ),
        },
        {
            "source_id": "src_002",
            "title": "Introduction",
            "excerpt": (
                "In recent years, several different approaches have been proposed for reinforcement learning "
                "with neural network function approximators. The leading contenders are deep Q-learning, "
                "vanilla policy gradient methods, and trust region / natural policy gradient methods. "
                "PPO uses a clipped surrogate objective to constrain policy updates."
            ),
        },
        {
            "source_id": "src_003",
            "title": "Experiments",
            "excerpt": (
                "6.1 Comparison of Surrogate Objectives\n"
                "First, we compare several different surrogate objectives under different hyperparameters. "
                "We test on 7 simulated robotics environments using the OpenAI Gym. "
                "The environments include Hopper, Walker, HalfCheetah, Swimmer, and CartPole. "
                "We report the mean reward over 100 episodes after 500000 timesteps of training. "
                "We use Adam optimizer with learning rate 3e-4 and batch size 64."
            ),
        },
        {
            "source_id": "src_004",
            "title": "Conclusion",
            "excerpt": (
                "We have introduced proximal policy optimization, a family of policy optimization methods "
                "that use multiple epochs of stochastic gradient ascent to perform each policy update. "
                "These methods have the stability and reliability of trust-region methods."
            ),
        },
        {
            "source_id": "src_005",
            "title": "References",
            "excerpt": (
                "Schulman et al., 2015. Trust Region Policy Optimization. "
                "Mnih et al., 2016. Asynchronous Methods for Deep Reinforcement Learning."
            ),
        },
    ],
}


class TestRunOffline:
    """Test the deterministic offline extraction mode."""

    def test_produces_paper_claim_map(self, tmp_path: Path):
        result = run_offline("prj_test_ppo", tmp_path, PPO_WORKSPACE_CLAIM_MAP)
        assert isinstance(result, PaperClaimMap)
        assert result.core_contribution != ""
        assert len(result.core_contribution) > 20

    def test_extracts_ppo_contribution(self, tmp_path: Path):
        result = run_offline("prj_test_ppo", tmp_path, PPO_WORKSPACE_CLAIM_MAP)
        assert "policy gradient" in result.core_contribution.lower()

    def test_finds_datasets(self, tmp_path: Path):
        result = run_offline("prj_test_ppo", tmp_path, PPO_WORKSPACE_CLAIM_MAP)
        dataset_names = [d.name for d in result.datasets]
        assert "CartPole-v1" in dataset_names

    def test_finds_metrics(self, tmp_path: Path):
        result = run_offline("prj_test_ppo", tmp_path, PPO_WORKSPACE_CLAIM_MAP)
        metric_names = [m.name for m in result.metrics]
        assert "reward" in metric_names

    def test_extracts_training_recipe(self, tmp_path: Path):
        result = run_offline("prj_test_ppo", tmp_path, PPO_WORKSPACE_CLAIM_MAP)
        assert result.training_recipe.optimizer == "Adam"
        assert result.training_recipe.learning_rate == "3e-4"
        assert result.training_recipe.batch_size == "64"

    def test_detects_ambiguities(self, tmp_path: Path):
        result = run_offline("prj_test_ppo", tmp_path, PPO_WORKSPACE_CLAIM_MAP)
        # Should detect at least 5 ambiguities for PPO
        assert len(result.ambiguities) >= 5
        # All ambiguities have IDs
        ids = [a.assumption_id for a in result.ambiguities]
        assert all(id.startswith("A") for id in ids)

    def test_finds_ppo_specific_ambiguities(self, tmp_path: Path):
        """PPO is known to have 8 key ambiguities per the PRD."""
        result = run_offline("prj_test_ppo", tmp_path, PPO_WORKSPACE_CLAIM_MAP)
        details = [a.detail.lower() for a in result.ambiguities]
        # Should find at least some of the known PPO ambiguities
        found_adam_epsilon = any("adam" in d and "epsilon" in d for d in details)
        found_weight_init = any("weight" in d and "init" in d for d in details)
        found_grad_clip = any("gradient" in d and "clip" in d for d in details)
        found_count = sum([found_adam_epsilon, found_weight_init, found_grad_clip])
        assert found_count >= 2, f"Expected ≥2 PPO ambiguities, found {found_count}"

    def test_ambiguities_have_risk_levels(self, tmp_path: Path):
        result = run_offline("prj_test_ppo", tmp_path, PPO_WORKSPACE_CLAIM_MAP)
        for amb in result.ambiguities:
            assert amb.risk in (RiskLevel.low, RiskLevel.medium, RiskLevel.high, RiskLevel.critical)

    def test_writes_json_to_disk(self, tmp_path: Path):
        run_offline("prj_test_ppo", tmp_path, PPO_WORKSPACE_CLAIM_MAP)
        out_path = tmp_path / "prj_test_ppo" / "paper_claim_map.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "core_contribution" in data
        assert "ambiguities" in data

    def test_output_json_is_valid_claim_map(self, tmp_path: Path):
        """Round-trip: file on disk should deserialize back to PaperClaimMap."""
        run_offline("prj_test_ppo", tmp_path, PPO_WORKSPACE_CLAIM_MAP)
        out_path = tmp_path / "prj_test_ppo" / "paper_claim_map.json"
        data = json.loads(out_path.read_text())
        reconstructed = PaperClaimMap(**data)
        assert reconstructed.core_contribution != ""

    def test_extracts_claims(self, tmp_path: Path):
        result = run_offline("prj_test_ppo", tmp_path, PPO_WORKSPACE_CLAIM_MAP)
        assert len(result.claims) >= 1
        # Claims should reference the method (PPO or policy gradient)
        all_claims_str = str(result.claims).lower()
        assert "ppo" in all_claims_str or "policy" in all_claims_str


class TestHeuristicExtractors:
    """Unit tests for individual extraction functions."""

    def test_extract_contribution_from_abstract(self):
        sections = {
            "abstract": "We propose proximal policy optimization. It works well."
        }
        result = _extract_contribution(sections)
        assert "proximal policy optimization" in result.lower()

    def test_extract_datasets_finds_cartpole(self):
        sections = {"experiments": "We test on CartPole-v1 and Hopper-v4."}
        result = _extract_datasets(sections)
        names = [d.name for d in result]
        assert "CartPole-v1" in names

    def test_extract_metrics_finds_reward(self):
        sections = {"experiments": "We report the mean reward over 100 episodes."}
        result = _extract_metrics(sections)
        names = [m.name for m in result]
        assert "reward" in names

    def test_extract_training_recipe(self):
        sections = {
            "experiments": "We use Adam optimizer with learning rate 3e-4 and batch size 64."
        }
        recipe = _extract_training_recipe(sections)
        assert recipe.optimizer == "Adam"
        assert recipe.learning_rate == "3e-4"
        assert recipe.batch_size == "64"

    def test_extract_ambiguities_detects_missing_details(self):
        # Paper that mentions nothing about these details
        sections = {"abstract": "We propose a method.", "experiments": "It works."}
        ambiguities = _extract_ambiguities(sections)
        assert len(ambiguities) >= 5  # Should detect many missing details

    def test_empty_sections_still_works(self):
        sections = {}
        result = _extract_contribution(sections)
        assert "not found" in result.lower()


class TestIntegrationWithIngestionPipeline:
    """Test that the agent works with real ingestion pipeline output."""

    def test_handles_real_pipeline_output_format(self, tmp_path: Path):
        """The workspace claim_map from cli.py has this exact format."""
        real_format = {
            "project_id": "prj_abc123",
            "entries": [
                {"source_id": "src_1", "title": "Abstract", "excerpt": "A method..."},
                {"source_id": "src_2", "title": "Introduction", "excerpt": "RL is..."},
            ],
        }
        result = run_offline("prj_abc123", tmp_path, real_format)
        assert isinstance(result, PaperClaimMap)

    def test_handles_empty_entries(self, tmp_path: Path):
        """Should not crash on empty workspace."""
        empty = {"project_id": "prj_empty", "entries": []}
        result = run_offline("prj_empty", tmp_path, empty)
        assert isinstance(result, PaperClaimMap)
        assert result.core_contribution != ""


# ---------------------------------------------------------------------------
# Task 2 — _extract_metrics captures nearby numeric target values
# ---------------------------------------------------------------------------

class TestExtractMetricsTargetValue:
    """_extract_metrics must capture a numeric target_value near the keyword."""

    def test_extracts_target_value_for_reward(self):
        """With a numeric value near 'reward', target_value is populated."""
        sections = {"experiments": "we report a mean reward of 475.0 over 100 episodes"}
        result = _extract_metrics(sections)
        assert len(result) >= 1
        reward_spec = next((m for m in result if m.name == "reward"), None)
        assert reward_spec is not None, "Expected a MetricSpec with name='reward'"
        assert reward_spec.target_value is not None
        assert "475" in reward_spec.target_value

    def test_source_section_is_set_when_value_found(self):
        sections = {"experiments": "we report a mean reward of 475.0 over 100 episodes"}
        result = _extract_metrics(sections)
        reward_spec = next((m for m in result if m.name == "reward"), None)
        assert reward_spec is not None
        assert reward_spec.source_section == "experiments"

    def test_no_target_value_when_no_number_near_keyword(self):
        """If no number appears within ~120 chars, target_value stays None."""
        sections = {"abstract": "We discuss reward in qualitative terms only."}
        result = _extract_metrics(sections)
        reward_spec = next((m for m in result if m.name == "reward"), None)
        assert reward_spec is not None
        assert reward_spec.target_value is None

    def test_does_not_fabricate_value_from_distant_number(self):
        """A number far from the keyword (>120 chars away) should not be captured."""
        far_text = "reward" + " " * 150 + "42.0"
        sections = {"experiments": far_text}
        result = _extract_metrics(sections)
        reward_spec = next((m for m in result if m.name == "reward"), None)
        assert reward_spec is not None
        assert reward_spec.target_value is None
