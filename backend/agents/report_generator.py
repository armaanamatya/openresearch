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
from pathlib import Path
from typing import Any

from backend.agents.schemas import (
    ExperimentArtifacts,
    FinalReport,
    ImprovementHypothesis,
    MetricDelta,
    MetricSpec,
    PaperClaimMap,
    PathResult,
    PathSummary,
    ResearchMap,
)

logger = logging.getLogger(__name__)


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

        # Find best improvement value across paths
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
                if best_val is None or val > best_val:
                    best_val = val
                    best_path = pr.path_id

        delta_vs_baseline: float | None = None
        pct_change: float | None = None
        if best_val is not None and baseline_val is not None:
            delta_vs_baseline = best_val - baseline_val
            if baseline_val != 0:
                pct_change = round((delta_vs_baseline / abs(baseline_val)) * 100, 2)

        delta_vs_paper: float | None = None
        effective_best = best_val if best_val is not None else baseline_val
        if paper_target is not None and effective_best is not None:
            delta_vs_paper = effective_best - paper_target

        deltas.append(MetricDelta(
            metric_name=key,
            paper_target=paper_target,
            baseline_value=baseline_val,
            best_improved_value=best_val,
            best_path_id=best_path,
            delta_vs_paper=round(delta_vs_paper, 4) if delta_vs_paper is not None else None,
            delta_vs_baseline=round(delta_vs_baseline, 4) if delta_vs_baseline is not None else None,
            pct_change_vs_baseline=pct_change,
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


def generate_final_report(
    project_id: str,
    paper_claim_map: PaperClaimMap | None,
    experiment_artifacts: ExperimentArtifacts | None,
    improvement_hypotheses: list[ImprovementHypothesis],
    path_results: list[PathResult],
    research_map: ResearchMap | None,
) -> FinalReport:
    """Deterministically compute the final report from pipeline state."""
    paper_metrics = paper_claim_map.metrics if paper_claim_map else []
    baseline_metrics = experiment_artifacts.metrics if experiment_artifacts else {}

    # Metric deltas
    metric_deltas = _build_metric_deltas(paper_metrics, baseline_metrics, path_results)

    # Path summaries
    paths = _build_path_summaries(path_results, improvement_hypotheses, baseline_metrics)

    # Identify primary metric: first paper metric that has both a target and a baseline value
    primary_metric: str | None = None
    for spec in paper_metrics:
        target = _parse_target_value(spec.target_value)
        if target is not None:
            baseline_val = _find_metric_value(spec.name, baseline_metrics)
            if baseline_val is not None:
                for md in metric_deltas:
                    spec_norm = spec.name.lower().replace(" ", "_").replace("-", "_")
                    key_norm = md.metric_name.lower().replace(" ", "_").replace("-", "_")
                    if spec_norm == key_norm or key_norm in spec_norm or spec_norm in key_norm:
                        primary_metric = md.metric_name
                        break
            if primary_metric:
                break

    # Aggregate stats
    total = len(path_results)
    succeeded = sum(1 for p in path_results if p.success)
    failed = total - succeeded
    # Count paths that improved on the primary metric (or any metric if no primary)
    if primary_metric:
        improved = sum(
            1 for ps in paths
            if ps.delta_vs_baseline.get(primary_metric, 0) > 0
        )
    else:
        improved = sum(
            1 for ps in paths
            if any(d > 0 for d in ps.delta_vs_baseline.values())
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

    return FinalReport(
        project_id=project_id,
        paper_title=paper_claim_map.core_contribution[:120] if paper_claim_map else "",
        core_contribution=paper_claim_map.core_contribution if paper_claim_map else "",
        reproduction_score=round(repro_score, 4),
        reproduction_status=repro_status,
        metric_deltas=metric_deltas,
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

    # Metric deltas table
    if report.metric_deltas:
        lines.append("## Metric Deltas")
        lines.append("")
        lines.append("| Metric | Paper Target | Baseline | Best Improved | Delta vs Baseline | % Change | Best Path |")
        lines.append("|--------|-------------|----------|---------------|-------------------|----------|-----------|")
        for md in report.metric_deltas:
            paper = f"{md.paper_target}" if md.paper_target is not None else "-"
            baseline = f"{md.baseline_value:.4g}" if md.baseline_value is not None else "-"
            best = f"{md.best_improved_value:.4g}" if md.best_improved_value is not None else "-"
            delta = f"{md.delta_vs_baseline:+.4g}" if md.delta_vs_baseline is not None else "-"
            pct = f"{md.pct_change_vs_baseline:+.2f}%" if md.pct_change_vs_baseline is not None else "-"
            path = md.best_path_id or "-"
            lines.append(f"| {md.metric_name} | {paper} | {baseline} | {best} | {delta} | {pct} | {path} |")
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
