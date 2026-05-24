"""Rubric-contract validator — post-run check against paper_targets YAML.

The PaperBench-style rubric scores six areas. Five of them
(Data fidelity, Experiment execution, Evaluation protocol, Result match,
Artifact completeness) are **deterministically verifiable** against a
contract the paper declares in ``docs/papers/<arxiv_id>.yaml``:

    paper_targets:
      mnist_baseline_final_acc: 0.965        # numeric target
      mnist_bn_final_acc: 0.975
      required_metrics_keys: [...]
      required_artifacts: [...]
      variants_required: [...]

This module runs *after* ``_execute_in_sandbox`` produces a metrics.json and
artifact directory; it diffs them against the contract and returns a list
of typed violations. ``run_experiment`` includes the violations on the
result dict so the agent's next ``implement_baseline`` iteration sees them
as ``repair_context`` — the closed-loop fix that converts "trust the root
model to inspect weak leaves" into a deterministic feedback channel.

Design contract:

  * Pure function — no I/O side effects beyond reading the artifact dir.
  * Fail-soft — any unexpected shape returns an empty list rather than
    raising; observability MUST NOT block the run.
  * One violation per concrete issue, with a ``hint`` field that maps
    directly to a fix the agent can perform on its next iteration.

The validator deliberately does NOT call an LLM grader. The fidelity area
(method/code) still needs an LLM read of the code; that's the leaf
scorer's job. Areas 2-6 are pattern-matchable and live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Relative tolerance for "Result match versus the paper's reported targets".
# 10% relative error is the threshold below which we count as a hit; above,
# the validator surfaces the gap.
_RESULT_MATCH_TOLERANCE: float = 0.10

# Keys in paper_targets that are NOT numeric paper targets themselves.
_PAPER_TARGETS_META_KEYS: frozenset[str] = frozenset({
    "required_metrics_keys",
    "required_artifacts",
    "variants_required",
    "required_curves",
})


@dataclass(slots=True)
class RubricContractViolation:
    """A single deterministic violation of the paper's declared contract.

    Attributes
    ----------
    area : str
        The PaperBench rubric area name this violation maps to.  Used so the
        agent's repair_context surfaces violations grouped by area.
    detail : str
        Human-readable description of WHAT is wrong (the diff).
    hint : str
        Concrete actionable suggestion for the agent's next iteration.
    """

    area: str
    detail: str
    hint: str

    def to_dict(self) -> dict[str, str]:
        return {"area": self.area, "detail": self.detail, "hint": self.hint}


@dataclass(slots=True)
class RubricContractReport:
    """Aggregated validator output.

    The agent reads ``violations`` from repair_context and addresses each.
    ``summary`` is a one-line headline suitable for the UI / dashboard event.
    """

    violations: list[RubricContractViolation] = field(default_factory=list)
    summary: str = ""

    @property
    def compliant(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "compliant": self.compliant,
            "summary": self.summary,
            "violations": [v.to_dict() for v in self.violations],
        }


# ---------------------------------------------------------------------------
# Per-area validators — each adds 0+ violations to a list
# ---------------------------------------------------------------------------


def _check_required_metrics_keys(
    metrics: dict, paper_targets: dict, out: list[RubricContractViolation]
) -> None:
    required = paper_targets.get("required_metrics_keys") or []
    if not isinstance(required, (list, tuple)):
        return
    for key in required:
        if not isinstance(key, str):
            continue
        if key not in metrics:
            out.append(RubricContractViolation(
                area="Evaluation protocol and metric correctness",
                detail=f"metrics.json is missing required key {key!r}",
                hint=(
                    f"emit metrics.json[{key!r}] — the rubric grader looks for "
                    f"this exact key name when scoring the evaluation protocol area."
                ),
            ))


def _check_variants_required(
    metrics: dict, paper_targets: dict, out: list[RubricContractViolation]
) -> None:
    variants = paper_targets.get("variants_required") or []
    if not isinstance(variants, (list, tuple)) or not variants:
        return
    per_model = metrics.get("per_model") if isinstance(metrics.get("per_model"), dict) else {}
    omitted = metrics.get("omitted") if isinstance(metrics.get("omitted"), dict) else {}
    for v in variants:
        if not isinstance(v, str):
            continue
        if v in per_model:
            continue  # ran it
        if v in omitted:
            continue  # honestly declared as omitted with a reason
        out.append(RubricContractViolation(
            area="Experiment execution and reproducibility",
            detail=(
                f"variant {v!r} is in paper_targets.variants_required but missing "
                f"from both metrics.json.per_model and metrics.json.omitted"
            ),
            hint=(
                f"add per_model[{v!r}] = {{...}} with the variant's metrics if you ran it, "
                f"OR declare it omitted via metrics.json.omitted[{v!r}] = "
                f"'<one-line reason — license, compute, size>'"
            ),
        ))


def _check_required_artifacts(
    artifact_root: Path,
    paper_targets: dict,
    out: list[RubricContractViolation],
) -> None:
    required = paper_targets.get("required_artifacts") or []
    if not isinstance(required, (list, tuple)):
        return
    for rel in required:
        if not isinstance(rel, str):
            continue
        path = artifact_root / rel
        # Support glob patterns: required_artifacts: ["fig_*.png"] should match any.
        if any(c in rel for c in "*?["):
            matches = list(artifact_root.glob(rel))
            if not matches:
                out.append(RubricContractViolation(
                    area="Artifact completeness and provenance",
                    detail=f"no files match required artifact glob {rel!r}",
                    hint=(
                        f"emit at least one file matching {rel!r} into $OUTPUT_DIR — "
                        f"the rubric area 'Artifact completeness and provenance' "
                        f"scores 0 without it."
                    ),
                ))
            continue
        if not path.exists():
            out.append(RubricContractViolation(
                area="Artifact completeness and provenance",
                detail=f"required artifact {rel!r} not written",
                hint=(
                    f"write {rel!r} into $OUTPUT_DIR — e.g. README.md should explain "
                    f"what was reproduced, training_curves.json should carry per-step arrays."
                ),
            ))


def _check_result_match(
    metrics: dict,
    paper_targets: dict,
    out: list[RubricContractViolation],
    tolerance: float = _RESULT_MATCH_TOLERANCE,
) -> None:
    """For every numeric paper_target key, compare actual to expected."""
    for key, expected in paper_targets.items():
        if key in _PAPER_TARGETS_META_KEYS:
            continue
        if not isinstance(key, str):
            continue
        # Skip non-numeric targets (e.g. arch strings)
        try:
            expected_f = float(expected)
        except (TypeError, ValueError):
            continue
        if key not in metrics:
            # Missing-key violation already raised by _check_required_metrics_keys
            # if it was on that list. If not on that list, surface the missing
            # number as a result-match issue.
            if key not in (paper_targets.get("required_metrics_keys") or []):
                out.append(RubricContractViolation(
                    area="Result match versus the paper's reported targets",
                    detail=(
                        f"paper reports {key} = {expected_f}, but metrics.json "
                        f"does not contain this key"
                    ),
                    hint=f"compute and emit metrics.json[{key!r}] — paper target is {expected_f}",
                ))
            continue
        try:
            actual_f = float(metrics[key])
        except (TypeError, ValueError):
            out.append(RubricContractViolation(
                area="Result match versus the paper's reported targets",
                detail=f"metrics.json[{key!r}] is not numeric ({type(metrics[key]).__name__})",
                hint=f"emit {key!r} as a float number, not a string/None",
            ))
            continue
        if expected_f == 0.0:
            # Avoid divide-by-zero; require exact 0
            if actual_f != 0.0:
                out.append(RubricContractViolation(
                    area="Result match versus the paper's reported targets",
                    detail=f"{key} = {actual_f}, paper reports exactly 0",
                    hint="check the implementation — paper claims zero for this metric",
                ))
            continue
        rel_err = abs(actual_f - expected_f) / abs(expected_f)
        if rel_err > tolerance:
            out.append(RubricContractViolation(
                area="Result match versus the paper's reported targets",
                detail=(
                    f"{key} = {actual_f:.4f}, paper reports {expected_f:.4f} "
                    f"(relative error {rel_err:.1%}, tolerance {tolerance:.0%})"
                ),
                hint=(
                    f"the gap is large — train for more epochs, fix the algorithmic "
                    f"invariant, or use the full dataset.  Paper target {expected_f:.4f}, "
                    f"you produced {actual_f:.4f}."
                ),
            ))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate(
    metrics: dict | None,
    artifact_root: Path,
    paper_targets: dict | None,
) -> RubricContractReport:
    """Diff metrics + artifacts against the paper's declared contract.

    Returns ``RubricContractReport``.  Empty violations list = compliant.

    Fail-soft: any exception during validation logs and returns an empty
    report — the validator must NEVER block a run from being persisted.
    """
    report = RubricContractReport()

    if not isinstance(paper_targets, dict) or not paper_targets:
        return report  # no contract → nothing to check

    metrics = metrics if isinstance(metrics, dict) else {}

    try:
        _check_required_metrics_keys(metrics, paper_targets, report.violations)
        _check_variants_required(metrics, paper_targets, report.violations)
        _check_required_artifacts(artifact_root, paper_targets, report.violations)
        _check_result_match(metrics, paper_targets, report.violations)
    except Exception:  # noqa: BLE001 — observability must never block the run
        # Swallow and return whatever we collected; partial coverage beats none.
        return report

    if report.violations:
        # Headline by area count: "3 violations: 1 data, 1 eval, 1 result"
        by_area: dict[str, int] = {}
        for v in report.violations:
            by_area[v.area] = by_area.get(v.area, 0) + 1
        bits = [f"{n} {area}" for area, n in sorted(by_area.items(), key=lambda x: -x[1])]
        report.summary = f"{len(report.violations)} contract violation(s): " + "; ".join(bits)
    else:
        report.summary = "all paper_targets satisfied"

    return report


def load_paper_targets(arxiv_id: str | None, *, docs_root: Path | None = None) -> dict | None:
    """Load ``paper_targets`` from ``docs/papers/<arxiv_id>.yaml``.

    Returns ``None`` when no override exists OR when the override has no
    ``paper_targets`` section — both cases mean "no contract to validate
    against" and ``validate`` will return an empty report.
    """
    if not arxiv_id:
        return None
    if docs_root is None:
        docs_root = Path(__file__).resolve().parents[3] / "docs" / "papers"
    yaml_path = docs_root / f"{arxiv_id}.yaml"
    if not yaml_path.exists():
        return None
    try:
        import yaml as _yaml
    except ImportError:
        return None
    try:
        loaded = _yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — fail-soft on YAML errors
        return None
    if not isinstance(loaded, dict):
        return None
    targets = loaded.get("paper_targets")
    return targets if isinstance(targets, dict) else None


__all__ = [
    "RubricContractReport",
    "RubricContractViolation",
    "load_paper_targets",
    "validate",
]
