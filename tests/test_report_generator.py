"""Invariant: the final report is *computed* from pipeline artifacts.

`backend/agents/report_generator.py` is the single source of truth for the
final benchmark report — the PaperBench-style rubric, the statistical rigor
fields, and the paper-vs-baseline-vs-improved deltas. Two regressions are
guarded here:

  1. The generator must stay *wired in*. It was previously dead code while
     `live_runs.py` fabricated placeholder zeros; these tests fail if the
     orchestrator or the offline pipeline stops calling it.
  2. Every number must be *derived*, never hardcoded or fabricated. The rubric
     responds to artifact completeness and gate decisions; the statistical
     fields stay ``None`` unless the runner actually emitted the variance.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from backend.agents.report_generator import (
    _find_std,
    generate_final_report,
    render_report_markdown,
    write_final_report,
)
from backend.agents.schemas import (
    BaselineResult,
    EnvironmentSpec,
    ExperimentArtifacts,
    GateDecision,
    GateStatus,
    ImprovementHypothesis,
    MetricSpec,
    PaperClaimMap,
    PathResult,
    ResearchMap,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_SRC = REPO_ROOT / "backend" / "agents" / "orchestrator.py"
PIPELINE_SRC = REPO_ROOT / "backend" / "agents" / "pipeline.py"


# ---------------------------------------------------------------------------
# Fixtures — a minimal but realistic "complete" pipeline state.
# ---------------------------------------------------------------------------

def _paper_claim_map() -> PaperClaimMap:
    return PaperClaimMap(
        core_contribution="Proximal Policy Optimization for CartPole-v1.",
        claims=[{"method": "PPO", "metric": "mean_reward", "expected_result": "475"}],
        datasets=[],
        metrics=[
            MetricSpec(
                name="mean_reward",
                definition="Mean reward over 100 evaluation episodes.",
                target_value="475.0",
                source_section="Experiments",
            )
        ],
        evaluation_protocol="100 deterministic evaluation episodes after training.",
    )


def _experiment_artifacts() -> ExperimentArtifacts:
    return ExperimentArtifacts(
        metrics={"mean_reward": 487.3, "eval_episodes": 100, "total_timesteps": 500000},
        log_path="baseline/logs/run.log",
        commands_log_path="baseline/commands.log",
        provenance_path="baseline/provenance.json",
        success=True,
    )


def _path_results() -> list[PathResult]:
    return [
        PathResult(
            path_id="path_1",
            hypothesis="Lower the entropy coefficient.",
            diff_summary="entropy_coef 0.01 -> 0.005",
            metrics={"mean_reward": 495.8, "eval_episodes": 100},
            recommendation="Accept: +8.5 reward improvement.",
            success=True,
        ),
        PathResult(
            path_id="path_2",
            hypothesis="Raise the learning rate.",
            metrics={"mean_reward": 470.1, "eval_episodes": 100},
            recommendation="Reject: regression.",
            success=True,
        ),
    ]


def _gate(name: str, status: GateStatus) -> GateDecision:
    return GateDecision(gate=name, passed=True, status=status)


def _full_report():
    return generate_final_report(
        "prj_test",
        _paper_claim_map(),
        _experiment_artifacts(),
        [ImprovementHypothesis(path_id="path_1", hypothesis="h", rationale="r", expected_outcome="o")],
        _path_results(),
        ResearchMap(next_experiments=["Run full 500k timesteps."]),
        environment_spec=EnvironmentSpec(
            dockerfile="FROM python:3.11",
            python_version="3.11",
            framework="stable-baselines3",
            pip_packages={"gymnasium": "0.29.1"},
        ),
        baseline_result=BaselineResult(code_path="code/train.py", commands_to_run=["python train.py"]),
        gate_1=_gate("gate_1", GateStatus.verified),
        gate_2=_gate("gate_2", GateStatus.verified),
        gate_3=_gate("gate_3", GateStatus.verified_with_caveats),
    )


# ---------------------------------------------------------------------------
# 1. The generator stays wired into the pipeline.
# ---------------------------------------------------------------------------

def test_report_generator_is_wired_into_orchestrator_and_pipeline():
    """Both finishing paths must call generate_final_report + write_final_report.

    Guards against regression to the old state where the generator was dead
    code and the UI bridge fabricated placeholder zeros.
    """
    for src in (ORCHESTRATOR_SRC, PIPELINE_SRC):
        tree = ast.parse(src.read_text())
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "generate_final_report" in called, f"{src.name} never calls generate_final_report"
        assert "write_final_report" in called, f"{src.name} never calls write_final_report"


# ---------------------------------------------------------------------------
# 2. The rubric is computed from artifacts, not hardcoded.
# ---------------------------------------------------------------------------

def test_rubric_is_computed_from_artifacts_not_hardcoded():
    """A complete run and a degenerate run must produce *different* rubric scores."""
    full = _full_report()
    areas = {a.area for a in full.rubric}
    assert areas == {
        "paper_understanding",
        "environment_reconstruction",
        "baseline_implementation",
        "execution_artifacts",
        "comparison_quality",
    }
    # Weighted overall is the weighted mean of the area scores.
    expected = sum(a.score * a.weight for a in full.rubric) / sum(a.weight for a in full.rubric)
    assert abs(full.rubric_overall_score - expected) < 1e-6
    assert 0.0 < full.rubric_overall_score <= 1.0

    # A degenerate run (no artifacts, no gates) must score strictly lower.
    degenerate = generate_final_report("prj_empty", None, None, [], [], None)
    assert degenerate.rubric_overall_score < full.rubric_overall_score

    # Gate status moves the score: a failed Gate 1 lowers env reconstruction.
    failed_gate = generate_final_report(
        "prj_test", _paper_claim_map(), _experiment_artifacts(), [], _path_results(), None,
        environment_spec=EnvironmentSpec(dockerfile="FROM python:3.11", python_version="3.11",
                                         framework="sb3", pip_packages={"gymnasium": "0.29.1"}),
        gate_1=_gate("gate_1", GateStatus.failed_reproduction),
    )
    env_full = next(a for a in full.rubric if a.area == "environment_reconstruction")
    env_failed = next(a for a in failed_gate.rubric if a.area == "environment_reconstruction")
    assert env_failed.score < env_full.score


# ---------------------------------------------------------------------------
# 3. Paper-vs-reproduction headline is computed with % and numerical deltas.
# ---------------------------------------------------------------------------

def test_paper_vs_reproduction_headline_has_numerical_and_pct_delta():
    report = _full_report()
    assert report.primary_metric == "mean_reward"
    assert report.paper_primary_target == 475.0
    # Best improved value across paths (495.8) is "our reproduction".
    assert report.reproduction_primary_value == 495.8
    assert report.reproduction_delta_vs_paper == 20.8
    assert report.reproduction_pct_vs_paper == round(20.8 / 475.0 * 100, 2)


# ---------------------------------------------------------------------------
# 4. Statistical fields are never fabricated.
# ---------------------------------------------------------------------------

def test_statistics_not_fabricated_when_runner_emits_no_variance():
    """No std in the metrics -> no effect size, no CI; relative error still computed."""
    report = _full_report()
    mean_reward = next(md for md in report.metric_deltas if md.metric_name == "mean_reward")
    assert mean_reward.n_eval_episodes == 100
    assert mean_reward.baseline_std is None
    assert mean_reward.improved_std is None
    assert mean_reward.effect_size is None
    assert mean_reward.ci95_half_width is None
    # Relative error vs the paper target is derivable from point estimates alone.
    assert mean_reward.relative_error_vs_paper == round(abs(495.8 - 475.0) / 475.0, 4)
    assert "point estimates only" in report.statistical_notes


def test_statistics_computed_when_runner_emits_variance():
    """When std + sample size are present, effect size and CI are computed."""
    artifacts = ExperimentArtifacts(
        metrics={"mean_reward": 487.3, "mean_reward_std": 12.0, "eval_episodes": 100},
        success=True,
    )
    paths = [PathResult(
        path_id="path_1", hypothesis="h",
        metrics={"mean_reward": 495.8, "mean_reward_std": 10.0, "eval_episodes": 100},
        success=True,
    )]
    report = generate_final_report(
        "prj_test", _paper_claim_map(), artifacts, [], paths, None,
    )
    mr = next(md for md in report.metric_deltas if md.metric_name == "mean_reward")
    assert mr.baseline_std == 12.0
    assert mr.improved_std == 10.0
    assert mr.effect_size is not None and mr.effect_size > 0
    assert mr.ci95_half_width is not None and mr.ci95_half_width > 0
    assert "effect size" in report.statistical_notes.lower()


def test_find_std_never_mistakes_a_metric_for_its_own_std():
    """`mean_reward` must NOT satisfy a lookup for `mean_reward_std`.

    Guards the substring-matching bug: a loose lookup returned the metric's own
    value as its standard deviation, producing fabricated effect sizes/CIs.
    """
    metrics = {"mean_reward": 487.3, "eval_episodes": 100, "total_timesteps": 500000}
    assert _find_std("mean_reward", metrics) is None
    assert _find_std("total_timesteps", metrics) is None
    assert _find_std("eval_episodes", metrics) is None
    # An explicit companion key *is* picked up.
    assert _find_std("mean_reward", {"mean_reward": 487.3, "mean_reward_std": 12.0}) == 12.0


# ---------------------------------------------------------------------------
# 5. Rendered markdown surfaces every section, and write_final_report persists.
# ---------------------------------------------------------------------------

def test_markdown_and_persistence(tmp_path):
    report = _full_report()
    md = render_report_markdown(report)
    for section in (
        "## Reproduction Fidelity",
        "## Paper vs. Reproduction",
        "## PaperBench-Style Rubric (Computed)",
        "## Metric Deltas",
        "## Statistical Rigor",
    ):
        assert section in md, f"missing section: {section}"
    assert "no rubric scores are hardcoded" in md

    json_path, md_path = write_final_report(report, tmp_path)
    assert json_path.exists() and md_path.exists()
    persisted = json.loads(json_path.read_text())
    assert persisted["primary_metric"] == "mean_reward"
    assert persisted["rubric"], "rubric must be persisted in final_report.json"
