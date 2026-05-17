"""Improvement Orchestrator + Path Agents — selects and executes improvement hypotheses.

Provides:
  - ``select_hypotheses_offline()`` — deterministic hypothesis selection for PPO
  - ``run_path_offline()`` — simulated path execution
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.agents.schemas import (
    ExperimentArtifacts,
    ImprovementHypothesis,
    PaperClaimMap,
    PathResult,
    RiskLevel,
)
from backend.utils.io import write_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PPO-specific improvement hypotheses (from PRD)
# ---------------------------------------------------------------------------

PPO_HYPOTHESES = [
    ImprovementHypothesis(
        path_id="path_1",
        hypothesis="Tune entropy coefficient: reduce from 0.01 to 0.005 to prevent premature convergence",
        rationale=(
            "The baseline entropy coefficient of 0.01 may cause the policy to explore too aggressively, "
            "leading to suboptimal convergence. Reducing to 0.005 allows tighter policy optimization."
        ),
        expected_outcome="Mean reward improves from ~487 to ~495+",
        compute_estimate="~5 minutes CPU",
        risk=RiskLevel.low,
    ),
    ImprovementHypothesis(
        path_id="path_2",
        hypothesis="Test separate vs. shared actor-critic network",
        rationale=(
            "The baseline uses a shared network for actor and critic. Separate networks may allow "
            "independent capacity for value estimation and policy learning."
        ),
        expected_outcome="May improve or hurt — architectural experiment",
        compute_estimate="~5 minutes CPU",
        risk=RiskLevel.medium,
    ),
    ImprovementHypothesis(
        path_id="path_3",
        hypothesis="Test GAE lambda sensitivity: 0.90 vs 0.95 (baseline) vs 0.99",
        rationale=(
            "GAE lambda controls the bias-variance tradeoff in advantage estimation. "
            "Lower lambda reduces variance but increases bias. Higher lambda does the opposite."
        ),
        expected_outcome="Optimal lambda may differ from baseline 0.95",
        compute_estimate="~15 minutes CPU (3 runs)",
        risk=RiskLevel.low,
    ),
]


def select_hypotheses_offline(
    paper_claim_map: PaperClaimMap,
    baseline_metrics: dict[str, Any],
    *,
    user_hints: list[str] | None = None,
    n_paths: int = 3,
) -> list[ImprovementHypothesis]:
    """Select improvement hypotheses without LLM.

    For the PPO demo, returns the 3 PRD-specified improvement paths.
    """
    hypotheses = list(PPO_HYPOTHESES[:n_paths])

    # If user provided hints, incorporate them
    if user_hints:
        for i, hint in enumerate(user_hints[:n_paths]):
            if i < len(hypotheses):
                hypotheses[i] = ImprovementHypothesis(
                    path_id=f"path_{i + 1}",
                    hypothesis=hint,
                    rationale=f"User-directed improvement: {hint}",
                    expected_outcome="TBD based on experiment results",
                    risk=RiskLevel.medium,
                )

    return hypotheses


def run_path_offline(
    project_id: str,
    runs_root: Path,
    hypothesis: ImprovementHypothesis,
    baseline_metrics: dict[str, Any],
    *,
    simulate_success: bool = True,
) -> PathResult:
    """Simulate running one improvement path.

    Generates realistic results based on the hypothesis.
    """
    path_dir = Path(runs_root) / project_id / "improvements" / hypothesis.path_id
    path_dir.mkdir(parents=True, exist_ok=True)
    (path_dir / "plots").mkdir(exist_ok=True)

    baseline_reward = baseline_metrics.get("mean_reward", 487.0)

    if simulate_success:
        # Simulate improvement results
        improvement_map = {
            "path_1": {  # Entropy tuning
                "mean_reward": baseline_reward + 8.5,
                "change": "entropy_coef: 0.01 → 0.005",
                "diff": "Changed entropy_coef from 0.01 to 0.005 in config",
                "recommendation": "Accept: +8.5 reward improvement with minimal risk",
            },
            "path_2": {  # Separate networks
                "mean_reward": baseline_reward - 12.0,
                "change": "Separate actor-critic networks",
                "diff": "Replaced shared ActorCritic with separate Actor and Critic networks",
                "recommendation": "Reject: -12 reward regression, shared network is better for CartPole",
            },
            "path_3": {  # GAE lambda
                "mean_reward": baseline_reward + 3.2,
                "change": "gae_lambda: 0.95 → 0.99",
                "diff": "Changed gae_lambda from 0.95 to 0.99 in config",
                "recommendation": "Marginal: +3.2 reward, within noise margin",
            },
        }
        result_data = improvement_map.get(hypothesis.path_id, {
            "mean_reward": baseline_reward + 1.0,
            "change": hypothesis.hypothesis,
            "diff": f"Applied: {hypothesis.hypothesis}",
            "recommendation": "Needs further analysis",
        })

        metrics = {
            "mean_reward": result_data["mean_reward"],
            "eval_episodes": 100,
            "baseline_reward": baseline_reward,
            "improvement": result_data["mean_reward"] - baseline_reward,
        }
        write_json(path_dir / "metrics.json", metrics)

        return PathResult(
            path_id=hypothesis.path_id,
            hypothesis=hypothesis.hypothesis,
            diff_summary=result_data["diff"],
            metrics=metrics,
            plots=[str(path_dir / "plots" / "reward_curve.png")],
            commands=["python train.py"],
            recommendation=result_data["recommendation"],
            success=True,
        )
    else:
        return PathResult(
            path_id=hypothesis.path_id,
            hypothesis=hypothesis.hypothesis,
            failure_notes="Simulated failure for testing",
            success=False,
        )
