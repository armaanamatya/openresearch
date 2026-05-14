"""Final Report Generator — computes deltas and synthesizes the pipeline outcome.

Deterministically assembles the final report from PipelineState data:
  - Compares baseline metrics vs paper targets
  - Compares each improvement path vs baseline
  - Identifies the best-performing path
  - Computes reproduction fidelity score
  - Generates structured JSON + markdown output
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from backend.agents.schemas import (
    BaselineResult,
    EnvironmentSpec,
    ExperimentArtifacts,
    FinalReport,
    GateDecision,
    GateStatus,
    ImprovementHypothesis,
    MetricDelta,
    MetricSpec,
    PaperClaimMap,
    PathResult,
    PathSummary,
    ResearchMap,
    RubricArea,
    RubricVerification,
)

logger = logging.getLogger(__name__)


# Verifier gate decisions map onto a 0-1 confidence multiplier. These are the
# only place gate semantics turn into numbers, so the rubric stays consistent.
_GATE_STATUS_SCORE: dict[GateStatus, float] = {
    GateStatus.verified: 1.0,
    GateStatus.verified_with_caveats: 0.85,
    GateStatus.partial_reproduction: 0.6,
    GateStatus.blocked_requires_human: 0.4,
    GateStatus.failed_reproduction: 0.25,
    GateStatus.invalid_claim: 0.1,
}


def _gate_status_score(gate: GateDecision | None) -> float:
    """Map a gate decision to a 0-1 confidence score (0.0 when the gate never ran)."""
    if gate is None:
        return 0.0
    return _GATE_STATUS_SCORE.get(gate.status, 0.0)


def _fraction(checks: list[bool]) -> float:
    """Fraction of boolean checks that passed, rounded for stable serialization."""
    if not checks:
        return 0.0
    return round(sum(1 for c in checks if c) / len(checks), 4)


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


# Metric names that read as cost/error quantities — for these, *lower* is the
# better outcome. Everything else defaults to higher-is-better. This keeps the
# report paper-agnostic: a loss/perplexity/FID paper is compared correctly
# without any per-paper configuration.
_LOWER_IS_BETTER_HINTS = (
    "loss", "error", "err_rate", "fid", "perplexity", "ppl", "wer", "cer",
    "mae", "mse", "rmse", "nll", "regret", "latency", "runtime", "elapsed", "cost",
)


def _infer_direction(metric_name: str) -> str:
    """Heuristically infer optimization direction from a metric name.

    Deterministic and paper-agnostic. A future ``MetricSpec.direction`` field
    set by the paper-understanding agent can override this fallback.
    """
    norm = metric_name.lower()
    if any(hint in norm for hint in _LOWER_IS_BETTER_HINTS):
        return "lower_is_better"
    return "higher_is_better"


def _is_improvement(raw_delta: float | None, direction: str) -> bool:
    """True when a raw (value - baseline) delta represents a real improvement."""
    if raw_delta is None:
        return False
    return raw_delta > 0 if direction == "higher_is_better" else raw_delta < 0


def _find_std(metric_name: str, metrics: dict[str, Any]) -> float | None:
    """Look for an explicit standard-deviation companion key for a metric.

    Uses exact / normalized key matching only — never substring matching — so a
    metric is never mistaken for its own standard deviation (e.g. ``mean_reward``
    must not satisfy a lookup for ``mean_reward_std``).
    """
    norm = metric_name.lower().replace(" ", "_").replace("-", "_")
    candidates = {
        f"{norm}_std", f"std_{norm}", f"{norm}_stddev", f"{norm}_sd",
    }
    if norm in ("mean_reward", "reward"):
        candidates |= {"reward_std", "std_reward", "std", "reward_stddev"}
    norm_metrics = {
        key.lower().replace(" ", "_").replace("-", "_"): value
        for key, value in metrics.items()
    }
    for cand in candidates:
        if cand in norm_metrics:
            try:
                return float(norm_metrics[cand])
            except (ValueError, TypeError):
                pass
    return None


def _parse_target_value(target: str | None) -> float | None:
    """Try to extract a numeric value from a target_value string."""
    if target is None:
        return None
    # Try direct float parse
    try:
        return float(target)
    except (ValueError, TypeError):
        pass
    # Try to find first number in the string (e.g. "0.82 for PPO...")
    import re
    match = re.search(r"[-+]?\d*\.?\d+", target)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return None


def _compute_reproduction_score(
    paper_metrics: list[MetricSpec],
    baseline_metrics: dict[str, Any],
) -> float:
    """Compute how closely baseline reproduced the paper's target values (0-1).

    Uses relative closeness: 1.0 if exact match, decreasing as distance grows.
    Returns average across all metrics that have both a target and a baseline value.
    """
    scores: list[float] = []
    for spec in paper_metrics:
        target = _parse_target_value(spec.target_value)
        if target is None:
            continue
        # Try to find matching baseline metric by name
        baseline_val = _find_metric_value(spec.name, baseline_metrics)
        if baseline_val is None:
            continue
        # Relative accuracy: 1 - |delta| / max(|target|, 1)
        denominator = max(abs(target), 1.0)
        accuracy = max(0.0, 1.0 - abs(baseline_val - target) / denominator)
        scores.append(accuracy)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _find_metric_value(metric_name: str, metrics: dict[str, Any]) -> float | None:
    """Fuzzy-match a metric name against a metrics dict."""
    # Direct match
    if metric_name in metrics:
        try:
            return float(metrics[metric_name])
        except (ValueError, TypeError):
            return None
    # Normalized key match
    normalized = metric_name.lower().replace(" ", "_").replace("-", "_")
    for key, val in metrics.items():
        if key.lower().replace(" ", "_").replace("-", "_") == normalized:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    # Substring match
    for key, val in metrics.items():
        if normalized in key.lower() or key.lower() in normalized:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def _build_metric_deltas(
    paper_metrics: list[MetricSpec],
    baseline_metrics: dict[str, Any],
    path_results: list[PathResult],
) -> list[MetricDelta]:
    """Build per-metric delta comparisons across paper, baseline, and improvement paths."""
    deltas: list[MetricDelta] = []

    # Collect all metric keys from baseline + paths
    all_metric_keys: set[str] = set(baseline_metrics.keys())
    for pr in path_results:
        if pr.success:
            all_metric_keys.update(pr.metrics.keys())

    for key in sorted(all_metric_keys):
        baseline_val = None
        try:
            baseline_val = float(baseline_metrics.get(key, 0))
        except (ValueError, TypeError):
            continue

        # Find paper target for this metric
        paper_target: float | None = None
        for spec in paper_metrics:
            spec_norm = spec.name.lower().replace(" ", "_").replace("-", "_")
            key_norm = key.lower().replace(" ", "_").replace("-", "_")
            if spec_norm == key_norm or key_norm in spec_norm or spec_norm in key_norm:
                paper_target = _parse_target_value(spec.target_value)
                break

        # Direction decides what "best" and "improvement" mean for this metric.
        direction = _infer_direction(key)

        # Find the best improvement value across paths (max for higher-is-better,
        # min for lower-is-better).
        best_val: float | None = None
        best_path: str | None = None
        for pr in path_results:
            if not pr.success:
                continue
            if key in pr.metrics:
                try:
                    val = float(pr.metrics[key])
                except (ValueError, TypeError):
                    continue
                is_better = (
                    best_val is None
                    or (val > best_val if direction == "higher_is_better" else val < best_val)
                )
                if is_better:
                    best_val = val
                    best_path = pr.path_id

        delta_vs_baseline: float | None = None
        pct_change: float | None = None
        if best_val is not None and baseline_val is not None:
            delta_vs_baseline = best_val - baseline_val
            if baseline_val != 0:
                # pct_change is signed so that positive always means "better",
                # regardless of metric direction — downstream improvement
                # counting and best-path selection can trust the sign.
                signed = delta_vs_baseline if direction == "higher_is_better" else -delta_vs_baseline
                pct_change = round((signed / abs(baseline_val)) * 100, 2)

        delta_vs_paper: float | None = None
        effective_best = best_val if best_val is not None else baseline_val
        if paper_target is not None and effective_best is not None:
            delta_vs_paper = effective_best - paper_target

        # --- Statistical rigor -------------------------------------------
        # All of this is populated only from data the runner actually emitted;
        # when variance is absent we leave the fields None rather than fabricate.
        n_episodes = _coerce_int(baseline_metrics.get("eval_episodes"))
        baseline_std = _find_std(key, baseline_metrics)
        improved_std: float | None = None
        if best_path is not None:
            for pr in path_results:
                if pr.path_id == best_path and pr.success:
                    improved_std = _find_std(key, pr.metrics)
                    if n_episodes is None:
                        n_episodes = _coerce_int(pr.metrics.get("eval_episodes"))
                    break

        relative_error: float | None = None
        if paper_target not in (None, 0) and effective_best is not None:
            relative_error = round(
                abs(effective_best - paper_target) / abs(paper_target), 4
            )

        effect_size: float | None = None
        if (
            baseline_std is not None
            and improved_std is not None
            and delta_vs_baseline is not None
        ):
            pooled = math.sqrt((baseline_std ** 2 + improved_std ** 2) / 2)
            if pooled > 0:
                effect_size = round(delta_vs_baseline / pooled, 4)

        ci95: float | None = None
        ci_std = improved_std if improved_std is not None else baseline_std
        if ci_std is not None and n_episodes and n_episodes > 0:
            ci95 = round(1.96 * ci_std / math.sqrt(n_episodes), 4)

        deltas.append(MetricDelta(
            metric_name=key,
            paper_target=paper_target,
            baseline_value=baseline_val,
            best_improved_value=best_val,
            best_path_id=best_path,
            delta_vs_paper=round(delta_vs_paper, 4) if delta_vs_paper is not None else None,
            delta_vs_baseline=round(delta_vs_baseline, 4) if delta_vs_baseline is not None else None,
            pct_change_vs_baseline=pct_change,
            direction=direction,
            n_eval_episodes=n_episodes,
            relative_error_vs_paper=relative_error,
            baseline_std=baseline_std,
            improved_std=improved_std,
            effect_size=effect_size,
            ci95_half_width=ci95,
        ))

    return deltas


def _build_path_summaries(
    path_results: list[PathResult],
    hypotheses: list[ImprovementHypothesis],
    baseline_metrics: dict[str, Any],
) -> list[PathSummary]:
    """Build a summary for each parallel improvement path."""
    hypothesis_map = {h.path_id: h for h in hypotheses}
    summaries: list[PathSummary] = []

    for pr in path_results:
        # Compute per-metric deltas vs baseline
        path_deltas: dict[str, float] = {}
        if pr.success:
            for key, val in pr.metrics.items():
                if key in baseline_metrics:
                    try:
                        path_deltas[key] = round(float(val) - float(baseline_metrics[key]), 4)
                    except (ValueError, TypeError):
                        pass

        # Determine verdict from recommendation
        verdict = "inconclusive"
        rec_lower = (pr.recommendation or "").lower()
        if "accept" in rec_lower:
            verdict = "accept"
        elif "reject" in rec_lower:
            verdict = "reject"
        elif "marginal" in rec_lower:
            verdict = "inconclusive"

        status = "success" if pr.success else "failed"
        if pr.success and not pr.metrics:
            status = "partial"

        summaries.append(PathSummary(
            path_id=pr.path_id,
            hypothesis=pr.hypothesis,
            status=status,
            diff_summary=pr.diff_summary,
            metrics=pr.metrics,
            delta_vs_baseline=path_deltas,
            recommendation=pr.recommendation,
            verdict=verdict,
        ))

    return summaries


def _compute_rubric(
    paper_claim_map: PaperClaimMap | None,
    environment_spec: EnvironmentSpec | None,
    baseline_result: BaselineResult | None,
    experiment_artifacts: ExperimentArtifacts | None,
    path_results: list[PathResult],
    metric_deltas: list[MetricDelta],
    research_map: ResearchMap | None,
    gate_1: GateDecision | None,
    gate_2: GateDecision | None,
    gate_3: GateDecision | None,
    project_dir: Path | None,
) -> list[RubricArea]:
    """Score a PaperBench-style rubric from concrete pipeline artifacts.

    Each area blends *content completeness* (did the agent produce the artifact
    and fill its key fields?) with the *verifier gate score* for that phase.
    Nothing is hardcoded — every number is derived from pipeline state.
    """
    areas: list[RubricArea] = []

    # 1. Paper understanding -------------------------------------------------
    pcm = paper_claim_map
    pu_checks = (
        [
            bool(pcm.core_contribution
                 and "not found" not in pcm.core_contribution.lower()),
            bool(pcm.claims),
            bool(pcm.metrics),
            bool(pcm.datasets),
            bool(pcm.evaluation_protocol),
        ]
        if pcm
        else [False] * 5
    )
    areas.append(RubricArea(
        area="paper_understanding",
        score=_fraction(pu_checks),
        weight=0.15,
        evidence=["paper_claim_map.json"],
        rationale=(
            f"{sum(pu_checks)}/5 claim-map fields populated "
            f"(contribution, claims, metrics, datasets, evaluation protocol)."
        ),
    ))

    # 2. Environment reconstruction -----------------------------------------
    env = environment_spec
    env_checks = (
        [
            bool(env.dockerfile),
            bool(env.python_version),
            bool(env.framework),
            bool(env.pip_packages),
        ]
        if env
        else [False] * 4
    )
    env_score = round(0.6 * _fraction(env_checks) + 0.4 * _gate_status_score(gate_1), 4)
    areas.append(RubricArea(
        area="environment_reconstruction",
        score=env_score,
        weight=0.20,
        evidence=["environment_spec.json", "Dockerfile"],
        rationale=(
            f"{sum(env_checks)}/4 environment fields populated; "
            f"Gate 1 status: {gate_1.status.value if gate_1 else 'n/a'}."
        ),
    ))

    # 3. Baseline implementation --------------------------------------------
    base_checks = [
        bool(baseline_result and baseline_result.code_path),
        bool(baseline_result and baseline_result.commands_to_run),
        bool(experiment_artifacts and experiment_artifacts.success),
        bool(experiment_artifacts and experiment_artifacts.metrics),
    ]
    base_score = round(0.5 * _fraction(base_checks) + 0.5 * _gate_status_score(gate_2), 4)
    areas.append(RubricArea(
        area="baseline_implementation",
        score=base_score,
        weight=0.25,
        evidence=["baseline_result.json", "baseline/metrics.json"],
        rationale=(
            f"{sum(base_checks)}/4 baseline checks passed; "
            f"Gate 2 status: {gate_2.status.value if gate_2 else 'n/a'}."
        ),
    ))

    # 4. Execution artifacts -------------------------------------------------
    def _artifact_present(rel: str | None) -> bool:
        if not rel:
            return False
        if project_dir is None:
            return True
        candidate = Path(rel)
        if not candidate.is_absolute():
            candidate = project_dir / candidate
        return candidate.exists() or (project_dir / Path(rel).name).exists()

    ea = experiment_artifacts
    exec_checks = [
        bool(ea and ea.metrics),
        bool(ea and _artifact_present(ea.log_path)),
        bool(ea and _artifact_present(ea.commands_log_path)),
        bool(ea and _artifact_present(ea.provenance_path)),
    ]
    areas.append(RubricArea(
        area="execution_artifacts",
        score=_fraction(exec_checks),
        weight=0.20,
        evidence=["metrics.json", "commands.log", "provenance.json"],
        rationale=(
            f"{sum(exec_checks)}/4 execution artifacts present "
            f"(metrics, run log, command log, provenance)."
        ),
    ))

    # 5. Comparison quality --------------------------------------------------
    paths_succeeded = sum(1 for p in path_results if p.success)
    comp_checks = [
        bool(metric_deltas),
        bool(path_results),
        paths_succeeded > 0,
        research_map is not None,
    ]
    comp_score = round(0.6 * _fraction(comp_checks) + 0.4 * _gate_status_score(gate_3), 4)
    areas.append(RubricArea(
        area="comparison_quality",
        score=comp_score,
        weight=0.20,
        evidence=["final_report.md", "research_map.json"],
        rationale=(
            f"{len(metric_deltas)} metric deltas computed; "
            f"{paths_succeeded}/{len(path_results)} improvement paths succeeded; "
            f"Gate 3 status: {gate_3.status.value if gate_3 else 'n/a'}."
        ),
    ))

    return areas


def _rubric_overall_score(rubric: list[RubricArea]) -> float:
    """Weighted aggregate of rubric area scores (weights need not sum to 1)."""
    total_weight = sum(a.weight for a in rubric)
    if total_weight <= 0:
        return 0.0
    return round(sum(a.score * a.weight for a in rubric) / total_weight, 4)


def _statistical_notes(metric_deltas: list[MetricDelta]) -> str:
    """Summarize the statistical basis of the comparison — honestly about limits."""
    has_std = any(
        md.baseline_std is not None or md.improved_std is not None
        for md in metric_deltas
    )
    n_values = sorted({md.n_eval_episodes for md in metric_deltas if md.n_eval_episodes})
    sample = (
        f"Evaluation sample size: {', '.join(str(n) for n in n_values)} episodes. "
        if n_values
        else "Evaluation sample size was not reported by the runner. "
    )
    if has_std:
        return (
            sample
            + "Per-metric standard deviations were emitted, so Cohen's d effect "
            "sizes and 95% confidence half-widths are reported below. Treat an "
            "improvement whose delta exceeds the CI half-width as statistically "
            "meaningful at the 95% level."
        )
    return (
        sample
        + "The runner emitted point estimates only (no per-episode variance), so "
        "confidence intervals and effect sizes cannot be computed. Deltas are "
        "reported as point differences and as relative error against the paper "
        "target; they should be read as directional, not significance-tested."
    )


def _resolve_paperbench_baseline(project_id: str) -> dict[str, Any] | None:
    """Look up the published PaperBench score for a vendored-bundle run.

    Bundle runs carry the paper id in the project id (``paperbench_<id>`` — set
    by ``bundle_to_workspace_claim_map``). Uploaded-paper runs (``prj_*``) have
    no published baseline, so this returns None. Fail-soft: any lookup error
    degrades to None rather than blocking the report.
    """
    prefix = "paperbench_"
    if not project_id.startswith(prefix):
        return None
    paper_id = project_id[len(prefix):]
    try:
        # Lazy import — `cli_paperbench` is a CLI module; keep it off the
        # report-generation import path unless a bundle run actually needs it.
        from backend.cli_paperbench import PUBLISHED_BASELINES

        entry = PUBLISHED_BASELINES.get(paper_id)
        if not entry:
            return None
        # Surface the strongest published agent — "match the best known result".
        model, scores = max(entry.items(), key=lambda kv: kv[1].get("mean", 0.0))
        return {
            "score": float(scores.get("mean", 0.0)),
            "source": "paperbench_published",
            "model": model,
        }
    except Exception:
        logger.warning(
            "PaperBench baseline lookup failed for %s", project_id, exc_info=True
        )
        return None


def _build_comparison_summary(
    improved: RubricVerification | None,
    baseline: RubricVerification | None,
    paperbench_baseline: dict[str, Any] | None,
    improvement_iterations: int,
) -> str:
    """A short, honest verdict comparing our rubric score to its references.

    Deterministic — no LLM call. States plainly when self-improvement did NOT
    help or the target was NOT met; never inflates the outcome.
    """
    if improved is None:
        return ""
    ours = improved.overall_score
    target = improved.target_score
    # Line 1 — our score against its most relevant reference.
    if paperbench_baseline:
        pb = float(paperbench_baseline.get("score", 0.0))
        rel = "above" if ours >= pb else "below"
        line1 = (
            f"Our rubric score {ours:.2f} is {rel} the published PaperBench "
            f"baseline {pb:.2f} ({paperbench_baseline.get('model', 'unknown')}) — "
            "different judges, read as directional."
        )
    elif baseline is not None:
        delta = ours - baseline.overall_score
        if delta > 0:
            line1 = (
                f"Self-improvement lifted our rubric score to {ours:.2f}, "
                f"+{delta:.2f} over the baseline {baseline.overall_score:.2f} "
                f"across {improvement_iterations} re-iteration round(s)."
            )
        elif delta < 0:
            line1 = (
                f"Self-improvement did not help: our rubric score {ours:.2f} fell "
                f"{delta:.2f} below the baseline {baseline.overall_score:.2f}."
            )
        else:
            line1 = (
                f"Our rubric score {ours:.2f} matched the baseline with no net "
                f"gain across {improvement_iterations} re-iteration round(s)."
            )
    else:
        line1 = f"Our rubric score is {ours:.2f}."
    # Line 2 — the honest target verdict.
    line2 = (
        f"Meets the {target:.2f} target."
        if improved.meets_target
        else f"Still below the {target:.2f} target."
    )
    return f"{line1} {line2}"


def generate_final_report(
    project_id: str,
    paper_claim_map: PaperClaimMap | None,
    experiment_artifacts: ExperimentArtifacts | None,
    improvement_hypotheses: list[ImprovementHypothesis],
    path_results: list[PathResult],
    research_map: ResearchMap | None,
    *,
    environment_spec: EnvironmentSpec | None = None,
    baseline_result: BaselineResult | None = None,
    gate_1: GateDecision | None = None,
    gate_2: GateDecision | None = None,
    gate_3: GateDecision | None = None,
    project_dir: Path | None = None,
    baseline_verification: RubricVerification | None = None,
    improved_verification: RubricVerification | None = None,
    improvement_iterations: int = 0,
) -> FinalReport:
    """Deterministically compute the final report from pipeline state.

    The optional keyword arguments feed the computed PaperBench-style rubric:
    when supplied, environment / baseline / gate state and on-disk artifacts
    are scored; when omitted, the rubric degrades gracefully to content-only
    scoring so older callers and unit tests keep working.
    """
    paper_metrics = paper_claim_map.metrics if paper_claim_map else []
    baseline_metrics = experiment_artifacts.metrics if experiment_artifacts else {}

    # Metric deltas
    metric_deltas = _build_metric_deltas(paper_metrics, baseline_metrics, path_results)

    # Path summaries
    paths = _build_path_summaries(path_results, improvement_hypotheses, baseline_metrics)

    # Identify primary metric: first paper metric that has both a target and a
    # baseline value. When the spec name matches several deltas (e.g. "reward"
    # substring-matches both `mean_reward` and the derived `baseline_reward`
    # key), prefer the metric the baseline actually measured, then an exact
    # name match, then the shortest name — so the headline names the real
    # claimed metric rather than an incidental derived key.
    def _norm(name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_")

    baseline_keys_norm = {_norm(k) for k in baseline_metrics}
    primary_metric: str | None = None
    for spec in paper_metrics:
        target = _parse_target_value(spec.target_value)
        if target is None:
            continue
        if _find_metric_value(spec.name, baseline_metrics) is None:
            continue
        spec_norm = _norm(spec.name)
        candidates = [
            md for md in metric_deltas
            if _norm(md.metric_name) == spec_norm
            or _norm(md.metric_name) in spec_norm
            or spec_norm in _norm(md.metric_name)
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda md: (
            _norm(md.metric_name) not in baseline_keys_norm,
            _norm(md.metric_name) != spec_norm,
            len(md.metric_name),
        ))
        primary_metric = candidates[0].metric_name
        break

    # Aggregate stats
    total = len(path_results)
    succeeded = sum(1 for p in path_results if p.success)
    failed = total - succeeded
    # Count paths that improved on the primary metric (or any metric if no
    # primary), respecting each metric's optimization direction.
    direction_map = {md.metric_name: md.direction for md in metric_deltas}
    if primary_metric:
        primary_dir = direction_map.get(primary_metric, "higher_is_better")
        improved = sum(
            1 for ps in paths
            if _is_improvement(ps.delta_vs_baseline.get(primary_metric), primary_dir)
        )
    else:
        improved = sum(
            1 for ps in paths
            if any(
                _is_improvement(d, direction_map.get(m, "higher_is_better"))
                for m, d in ps.delta_vs_baseline.items()
            )
        )

    # Find best path by the primary metric; fall back to first metric with a positive delta
    best_path_id: str | None = None
    best_improvement_pct: float | None = None
    for md in metric_deltas:
        if primary_metric and md.metric_name != primary_metric:
            continue
        if md.pct_change_vs_baseline is not None and md.best_path_id:
            if md.pct_change_vs_baseline > 0:
                best_improvement_pct = md.pct_change_vs_baseline
                best_path_id = md.best_path_id
            break
    # Fallback: if primary metric didn't yield an improvement, scan all
    if best_path_id is None:
        for md in metric_deltas:
            if md.pct_change_vs_baseline is not None and md.pct_change_vs_baseline > 0 and md.best_path_id:
                best_improvement_pct = md.pct_change_vs_baseline
                best_path_id = md.best_path_id
                break

    # Reproduction score
    repro_score = _compute_reproduction_score(paper_metrics, baseline_metrics)
    repro_status = "verified" if repro_score >= 0.8 else ("partial" if repro_score >= 0.5 else "failed")

    # Overall verdict
    if improved > 0 and best_improvement_pct and best_improvement_pct > 0:
        verdict = (
            f"Reproduction {'succeeded' if repro_score >= 0.5 else 'partial'}. "
            f"{improved}/{total} paths improved over baseline "
            f"(best: +{best_improvement_pct:.1f}% from {best_path_id})."
        )
    elif repro_score >= 0.8:
        verdict = f"Reproduction verified (score={repro_score:.2f}). No improvements exceeded baseline."
    else:
        verdict = f"Reproduction partial (score={repro_score:.2f}). {failed}/{total} paths failed."

    # Headline paper-vs-reproduction comparison: surface the primary claimed
    # metric so downstream consumers (UI bridge, PaperBench surface) don't have
    # to re-derive it from the deltas table.
    primary_md = next(
        (md for md in metric_deltas if md.metric_name == primary_metric),
        None,
    )
    paper_primary_target: float | None = None
    reproduction_primary_value: float | None = None
    reproduction_delta_vs_paper: float | None = None
    reproduction_pct_vs_paper: float | None = None
    if primary_md is not None:
        paper_primary_target = (
            float(primary_md.paper_target)
            if isinstance(primary_md.paper_target, (int, float))
            else _parse_target_value(primary_md.paper_target)
        )
        reproduction_primary_value = (
            primary_md.best_improved_value
            if primary_md.best_improved_value is not None
            else primary_md.baseline_value
        )
        if paper_primary_target is not None and reproduction_primary_value is not None:
            reproduction_delta_vs_paper = round(
                reproduction_primary_value - paper_primary_target, 4
            )
            if paper_primary_target != 0:
                reproduction_pct_vs_paper = round(
                    reproduction_delta_vs_paper / abs(paper_primary_target) * 100, 2
                )

    # Computed PaperBench-style rubric + statistical basis.
    rubric = _compute_rubric(
        paper_claim_map,
        environment_spec,
        baseline_result,
        experiment_artifacts,
        path_results,
        metric_deltas,
        research_map,
        gate_1,
        gate_2,
        gate_3,
        project_dir,
    )
    rubric_overall = _rubric_overall_score(rubric)
    statistical_notes = _statistical_notes(metric_deltas)

    # Track 3 — rubric-verifier assessment + honest self-improvement summary.
    paperbench_baseline = _resolve_paperbench_baseline(project_id)
    verification_delta = (
        round(
            improved_verification.overall_score
            - baseline_verification.overall_score,
            4,
        )
        if improved_verification is not None and baseline_verification is not None
        else None
    )
    comparison_summary = _build_comparison_summary(
        improved_verification,
        baseline_verification,
        paperbench_baseline,
        improvement_iterations,
    )

    return FinalReport(
        project_id=project_id,
        paper_title=paper_claim_map.core_contribution[:120] if paper_claim_map else "",
        core_contribution=paper_claim_map.core_contribution if paper_claim_map else "",
        reproduction_score=round(repro_score, 4),
        reproduction_status=repro_status,
        primary_metric=primary_metric,
        paper_primary_target=paper_primary_target,
        reproduction_primary_value=reproduction_primary_value,
        reproduction_delta_vs_paper=reproduction_delta_vs_paper,
        reproduction_pct_vs_paper=reproduction_pct_vs_paper,
        metric_deltas=metric_deltas,
        rubric=rubric,
        rubric_overall_score=rubric_overall,
        rubric_verification=improved_verification,
        baseline_rubric_verification=baseline_verification,
        paperbench_baseline=paperbench_baseline,
        verification_delta=verification_delta,
        improvement_iterations=improvement_iterations,
        comparison_summary=comparison_summary,
        statistical_notes=statistical_notes,
        paths=paths,
        best_path_id=best_path_id,
        best_overall_improvement_pct=best_improvement_pct,
        total_paths_run=total,
        paths_succeeded=succeeded,
        paths_failed=failed,
        paths_improved_over_baseline=improved,
        overall_verdict=verdict,
        next_steps=research_map.next_experiments if research_map else [],
    )


def render_report_markdown(report: FinalReport) -> str:
    """Render a FinalReport as a human-readable markdown document."""
    lines: list[str] = []

    lines.append(f"# Final Report: {report.project_id}")
    lines.append("")
    lines.append(f"**Core Contribution:** {report.core_contribution}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Reproduction fidelity
    lines.append("## Reproduction Fidelity")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Reproduction Score | **{report.reproduction_score:.2f}** / 1.00 |")
    lines.append(f"| Status | {report.reproduction_status} |")
    lines.append("")

    # Paper vs. reproduction headline
    if report.primary_metric:
        lines.append("## Paper vs. Reproduction")
        lines.append("")
        lines.append(f"Primary claimed metric: **{report.primary_metric}**")
        lines.append("")
        lines.append("| Source | Value |")
        lines.append("|--------|------:|")
        if report.paper_primary_target is not None:
            lines.append(f"| Paper-reported target | {report.paper_primary_target:.4g} |")
        if report.reproduction_primary_value is not None:
            lines.append(f"| Our reproduction (best) | {report.reproduction_primary_value:.4g} |")
        if report.reproduction_delta_vs_paper is not None:
            lines.append(f"| Numerical delta (ours − paper) | {report.reproduction_delta_vs_paper:+.4g} |")
        if report.reproduction_pct_vs_paper is not None:
            lines.append(f"| Percentage delta vs paper | {report.reproduction_pct_vs_paper:+.2f}% |")
        lines.append("")

    # PaperBench-style rubric (computed from artifacts)
    if report.rubric:
        lines.append("## PaperBench-Style Rubric (Computed)")
        lines.append("")
        lines.append(
            f"Weighted overall rubric score: **{report.rubric_overall_score:.2f}** / 1.00"
        )
        lines.append("")
        lines.append("| Area | Weight | Score | Evidence | Rationale |")
        lines.append("|------|-------:|------:|----------|-----------|")
        for area in report.rubric:
            ev = ", ".join(f"`{e}`" for e in area.evidence) or "-"
            lines.append(
                f"| {area.area} | {area.weight:.2f} | {area.score:.2f} | {ev} | {area.rationale} |"
            )
        lines.append("")
        lines.append(
            "> Every score is computed from pipeline artifacts and verifier gate "
            "decisions — no rubric scores are hardcoded."
        )
        lines.append("")

    # Metric deltas table
    if report.metric_deltas:
        lines.append("## Metric Deltas")
        lines.append("")
        lines.append("| Metric | Direction | Paper Target | Baseline | Best Improved | Delta vs Baseline | Improvement % | Best Path |")
        lines.append("|--------|-----------|-------------|----------|---------------|-------------------|---------------|-----------|")
        for md in report.metric_deltas:
            arrow = "↑ better" if md.direction == "higher_is_better" else "↓ better"
            paper = f"{md.paper_target}" if md.paper_target is not None else "-"
            baseline = f"{md.baseline_value:.4g}" if md.baseline_value is not None else "-"
            best = f"{md.best_improved_value:.4g}" if md.best_improved_value is not None else "-"
            delta = f"{md.delta_vs_baseline:+.4g}" if md.delta_vs_baseline is not None else "-"
            pct = f"{md.pct_change_vs_baseline:+.2f}%" if md.pct_change_vs_baseline is not None else "-"
            path = md.best_path_id or "-"
            lines.append(f"| {md.metric_name} | {arrow} | {paper} | {baseline} | {best} | {delta} | {pct} | {path} |")
        lines.append("")
        lines.append(
            "> *Improvement %* is signed so positive always means better, "
            "whatever the metric's direction."
        )
        lines.append("")

    # Statistical rigor
    lines.append("## Statistical Rigor")
    lines.append("")
    lines.append(report.statistical_notes or "No statistical notes available.")
    lines.append("")
    stat_rows = [
        md for md in report.metric_deltas
        if md.relative_error_vs_paper is not None
        or md.effect_size is not None
        or md.ci95_half_width is not None
        or md.n_eval_episodes is not None
    ]
    if stat_rows:
        lines.append("| Metric | n (episodes) | Rel. error vs paper | Effect size (d) | 95% CI half-width |")
        lines.append("|--------|-------------:|--------------------:|----------------:|------------------:|")
        for md in stat_rows:
            n = str(md.n_eval_episodes) if md.n_eval_episodes is not None else "-"
            rel = f"{md.relative_error_vs_paper * 100:.2f}%" if md.relative_error_vs_paper is not None else "-"
            es = f"{md.effect_size:+.3f}" if md.effect_size is not None else "n/a"
            ci = f"±{md.ci95_half_width:.4g}" if md.ci95_half_width is not None else "n/a"
            lines.append(f"| {md.metric_name} | {n} | {rel} | {es} | {ci} |")
        lines.append("")

    # Parallel improvement paths
    lines.append("## Improvement Paths")
    lines.append("")
    lines.append(f"**Total:** {report.total_paths_run} | "
                 f"**Succeeded:** {report.paths_succeeded} | "
                 f"**Failed:** {report.paths_failed} | "
                 f"**Improved over baseline:** {report.paths_improved_over_baseline}")
    lines.append("")

    for ps in report.paths:
        icon = {"accept": "+", "reject": "x", "inconclusive": "?"}
        marker = icon.get(ps.verdict, "?")
        lines.append(f"### [{marker}] {ps.path_id}: {ps.hypothesis}")
        lines.append("")
        lines.append(f"- **Status:** {ps.status}")
        lines.append(f"- **Verdict:** {ps.verdict}")
        if ps.diff_summary:
            lines.append(f"- **Change:** {ps.diff_summary}")
        if ps.recommendation:
            lines.append(f"- **Recommendation:** {ps.recommendation}")
        if ps.delta_vs_baseline:
            lines.append(f"- **Deltas vs baseline:**")
            for metric, delta in ps.delta_vs_baseline.items():
                direction = "+" if delta > 0 else ""
                lines.append(f"  - {metric}: {direction}{delta:.4g}")
        lines.append("")

    # Overall verdict
    lines.append("---")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"> {report.overall_verdict}")
    lines.append("")

    if report.best_path_id and report.best_overall_improvement_pct:
        lines.append(
            f"**Best path:** `{report.best_path_id}` "
            f"with **+{report.best_overall_improvement_pct:.2f}%** improvement over baseline."
        )
        lines.append("")

    # Next steps
    if report.next_steps:
        lines.append("## Next Steps")
        lines.append("")
        for step in report.next_steps:
            lines.append(f"- {step}")
        lines.append("")

    return "\n".join(lines)


def write_final_report(
    report: FinalReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write the final report as both JSON and Markdown to the project directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "final_report.json"
    json_path.write_text(report.model_dump_json(indent=2))

    md_path = output_dir / "final_report.md"
    md_path.write_text(render_report_markdown(report))

    logger.info("Final report written: %s, %s", json_path, md_path)
    return json_path, md_path
