"""Self-validating rubric guard — imported by the agent-written ``train.py``.

The grader-as-self pattern (Lane G).  When the agent-written ``train.py``
finishes, the rubric grader is the only signal back to the root model — and
it runs AFTER the experiment has terminated.  By that point any cheap-to-fix
shape mistake (missing key, missing artifact, missing figure) has already
cost a full ``run_experiment`` budget.

This module gives the agent's own code a way to **self-validate before
returning**: at the end of ``train.py``, after writing ``metrics.json``,
the agent calls :func:`assert_metrics_schema` with the keys + artifacts the
paper's rubric requires.  Any missing key or missing artifact raises
:class:`RubricGuardFailure` whose message becomes the next iteration's
``repair_context`` — turning a silent-grader-miss into a loud, actionable
error the root model can repair on the next pass.

The agent-written ``train.py`` either ``import``s this module via a
``code/rubric_guard.py`` copy of the source (the implement_baseline prompt
tells the agent to paste the source verbatim) or via a known sys.path entry
when the sandbox can reach the backend.  The module has zero non-stdlib
dependencies so the copy-and-paste route always works.

Auth-agnostic by construction (no provider branching, no LLM calls).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from pathlib import Path
from typing import Any


def _path_resolves(metrics: Any, json_path: str) -> bool:
    """Walk ``metrics`` along ``json_path`` (dot-separated) defensively.

    Returns True iff every segment exists and every intermediate node is a dict.
    An empty ``json_path`` always returns False (a declared path must be non-empty).

    Example::

        _path_resolves({"a": {"b": 1}}, "a.b")  # True
        _path_resolves({"a": 1}, "a.b")          # False — 1 is not a dict
        _path_resolves({}, "a")                  # False — key missing
    """
    if not json_path or not isinstance(metrics, dict):
        return False
    node: Any = metrics
    for segment in json_path.split("."):
        if not isinstance(node, dict):
            return False
        if segment not in node:
            return False
        node = node[segment]
    return True


class RubricGuardFailure(AssertionError):
    """Raised when ``metrics.json`` is missing required keys / artifacts.

    Inherits from ``AssertionError`` so the agent's existing assertion-driven
    test harnesses pick it up.  The exception text is structured JSON-style
    plain text so the next iteration's ``repair_context`` carries an
    actionable, machine-greppable failure record rather than a free-form
    string.
    """


def _walk_keys(obj: Any, prefix: str = "") -> set[str]:
    """Return every dotted key path that exists in a nested dict.

    A nested key ``{"a": {"b": 1}}`` exposes both ``"a"`` and ``"a.b"``, so
    a ``required_keys=["a.b"]`` check resolves the same way the operator
    wrote it in the paper YAML.
    """
    keys: set[str] = set()
    if not isinstance(obj, dict):
        return keys
    for k, v in obj.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        keys.add(path)
        if isinstance(v, dict):
            keys |= _walk_keys(v, path)
    return keys


def _key_present(required_key: str, present_paths: set[str]) -> bool:
    """True iff ``required_key`` resolves under any path in ``present_paths``.

    Two-tier match:

    1.  Exact dotted-path lookup — legacy contract. ``mnist.acc`` matches
        ``mnist.acc`` exactly.

    2.  Fingerprint match — required key tokenised on ``_`` and ``.`` must
        appear as an ordered subsequence in some present path's tokens.
        Intermediate generic segments (``per_model``, ``per_dataset``) are
        tolerated; reordering is not.

        Example: required ``mnist_logistic_adam_final_nll`` matches present
        ``per_model.mnist_logistic.per_dataset.mnist.adam_final_nll`` because
        the five required tokens [mnist, logistic, adam, final, nll] appear
        in order inside the present path's token stream.

    Discrimination: ``adam_mnist_loss`` will NOT match ``mnist_adam_loss``
    (different token order), and ``mnist_adam_loss`` will NOT match
    ``mnist_loss`` (missing token). False positives are bounded by the
    in-order subsequence requirement; the only risk is a present path that
    legitimately contains every required token in order in a position that
    isn't the leaf — acceptable because every walked path IS a position the
    grader's metric resolver could already reach.
    """
    if required_key in present_paths:
        return True

    required_tokens = [t for t in re.split(r"[._]+", required_key) if t]
    if not required_tokens:
        return False

    for path in present_paths:
        path_tokens = re.split(r"[._]+", path)
        i = 0
        for tok in path_tokens:
            if i < len(required_tokens) and tok == required_tokens[i]:
                i += 1
        if i == len(required_tokens):
            return True
    return False


def _resolve_artifact_dir(artifact_dir: str | Path | None) -> Path:
    """Resolve the artifact directory, honoring ``OUTPUT_DIR`` env var.

    Mirrors the implement_baseline contract: artifacts are written under
    ``$OUTPUT_DIR`` (default ``/artifacts``) and the rubric grader reads
    from the same location.
    """
    if artifact_dir is None:
        artifact_dir = os.environ.get("OUTPUT_DIR", "/artifacts")
    return Path(artifact_dir)


def _artifact_matches(artifact_dir: Path, pattern: str) -> bool:
    """Return True iff at least one file under ``artifact_dir`` matches.

    ``pattern`` may be a literal filename (``"README.md"``) or a glob
    (``"fig_*.png"``).  Globs are resolved against the entries directly
    inside ``artifact_dir``; subdirectories are not searched recursively
    because the implement_baseline contract puts artifacts at the top level.
    """
    if not artifact_dir.is_dir():
        return False
    # Literal path first — cheapest check, no listing required.
    if "*" not in pattern and "?" not in pattern and "[" not in pattern:
        return (artifact_dir / pattern).exists()
    try:
        entries = [p.name for p in artifact_dir.iterdir()]
    except OSError:
        return False
    return any(fnmatch.fnmatch(name, pattern) for name in entries)


def assert_metrics_schema(
    metrics: dict[str, Any],
    *,
    required_keys: list[str],
    required_artifacts: list[str] | None = None,
    artifact_dir: str | Path | None = None,
    metrics_shape: list[dict] | None = None,
    structured_evidence: dict[str, Any] | None = None,
) -> None:
    """Raise :class:`RubricGuardFailure` if metrics / artifacts are incomplete.

    The raised message is a JSON-ish payload that becomes the agent's
    ``repair_context`` on the next iteration — precise so the root model
    can fix the gap rather than re-explore.

    Args:
        metrics:            The metrics dict the agent is about to write
                            (or has just written) to ``metrics.json``.
                            May be flat or nested; nested keys are checked
                            via dotted paths (e.g. ``"per_model.qwen3_1.7b.acc"``).
        required_keys:      Dotted-path keys the rubric grader will look for.
                            Each must exist in ``metrics`` or its nested children.
                            When ``metrics_shape`` is non-empty, this argument is
                            used only as the fingerprint fallback for any metric
                            whose json_path is absent from ``metrics_shape``.
        required_artifacts: Filename literals or globs that must exist under
                            ``artifact_dir``.  Globs match flat (no recursion);
                            literals are exact-path.
        artifact_dir:       Directory containing the run's artifacts.  Defaults
                            to ``$OUTPUT_DIR`` (or ``/artifacts`` when unset).
        metrics_shape:      Agent-declared metric paths from
                            ``ReproductionContract.metrics_shape`` (θ PR).
                            When non-empty, each entry's ``json_path`` is checked
                            directly via dotted-path lookup — no fingerprint
                            guesswork.  When empty or None, falls back to the
                            existing fingerprint matcher against ``required_keys``
                            (backward compat).
        structured_evidence: Optional per-paper declaration of convergence /
                            sweep / time-series evidence the rubric's
                            eval-protocol leaves require (e.g.
                            ``{"history_methods": [...], "sweeps": [...],
                            "series": ["regret"]}``).  Enforced via
                            :func:`convergence_evidence.missing_structured_evidence`
                            ONLY when ``OPENRESEARCH_FIDELITY_EVIDENCE`` is set — so an
                            unset flag or a None value is a no-op.  Turns
                            "curves/sweeps computed but left in logs" (the
                            2026-06-09 Adam 0.21 eval-protocol crash) into an
                            actionable repair signal.

    Raises:
        RubricGuardFailure: When any required key is absent OR any required
                            artifact has no match.  The message text contains
                            the full structured detail so the next iteration's
                            ``repair_context`` is actionable.

    Example::

        from rubric_guard import assert_metrics_schema

        metrics = {"mnist_baseline_final_acc": 0.81, "per_model": {...}}
        write_metrics(metrics)
        assert_metrics_schema(
            metrics,
            required_keys=["mnist_baseline_final_acc", "per_model"],
            required_artifacts=["README.md", "fig_*.png", "training_curves.json"],
            artifact_dir=os.environ.get("OUTPUT_DIR", "/artifacts"),
        )
    """
    if not isinstance(metrics, dict):
        raise RubricGuardFailure(
            json.dumps({
                "rubric_guard": "metrics_not_dict",
                "got_type": type(metrics).__name__,
                "hint": (
                    "assert_metrics_schema(metrics, ...) expects a dict — got "
                    f"{type(metrics).__name__}. Build a plain Python dict before "
                    "calling the guard."
                ),
            })
        )

    missing_keys: list[str] = []

    if metrics_shape:
        # θ: authoritative path — check each declared json_path via dotted-path
        # lookup.  No fingerprint guesswork: the agent declared exactly what it
        # would emit; deviations are unambiguous contract violations.
        for mp in metrics_shape:
            json_path = (
                mp.get("json_path") if isinstance(mp, dict)
                else getattr(mp, "json_path", None)
            ) or ""
            if not json_path:
                # A MetricPath with no json_path is malformed — skip silently.
                continue
            if not _path_resolves(metrics, json_path):
                metric_id = (
                    mp.get("metric_id") if isinstance(mp, dict)
                    else getattr(mp, "metric_id", None)
                ) or json_path
                missing_keys.append(f"declared path {json_path!r} (id={metric_id!r})")
    else:
        # Fingerprint fallback (backward compat — commit befb51c).
        present_keys = _walk_keys(metrics)
        missing_keys = [k for k in required_keys if not _key_present(k, present_keys)]

    missing_artifacts: list[str] = []
    if required_artifacts:
        resolved_dir = _resolve_artifact_dir(artifact_dir)
        for pattern in required_artifacts:
            if not _artifact_matches(resolved_dir, pattern):
                missing_artifacts.append(pattern)

    # Structured-evidence enforcement (Module A): when the paper makes convergence /
    # sweep / time-series claims, final scalars alone score 0 on the eval-protocol leaves.
    # The check is itself flag-gated (``OPENRESEARCH_FIDELITY_EVIDENCE``) inside
    # convergence_evidence, so an unset flag → empty list → no behaviour change. The import
    # is lazy + guarded so rubric_guard keeps working when the sibling helper was not copied
    # into the sandbox (degrades to "nothing missing", never a hard import error).
    missing_evidence: list[str] = []
    if structured_evidence:
        try:
            try:
                import convergence_evidence as _ce  # sandbox-copied sibling
            except ImportError:
                from backend.agents.rlm import convergence_evidence as _ce
            missing_evidence = _ce.missing_structured_evidence(metrics, structured_evidence)
        except Exception:  # noqa: BLE001 — never block a run on a guard-helper hiccup
            missing_evidence = []

    if not missing_keys and not missing_artifacts and not missing_evidence:
        return

    present_keys_sample: list[str] = sorted(_walk_keys(metrics))[:20]
    detail: dict[str, Any] = {
        "rubric_guard": "schema_violation",
        "missing_keys": missing_keys,
        "missing_artifacts": missing_artifacts,
        "missing_structured_evidence": missing_evidence,
        "artifact_dir": str(_resolve_artifact_dir(artifact_dir)) if required_artifacts else None,
        "present_keys_sample": present_keys_sample,
        "hint": (
            "The rubric grader will lose points (or score 0) on the affected "
            "areas. Fix train.py so every required key is written to "
            "metrics.json AND every required artifact exists under "
            "$OUTPUT_DIR before the script exits."
            + (
                " For missing_structured_evidence: the paper's claims are about "
                "convergence/sweeps/time-series — write the per-epoch trajectory "
                "('history'), the full sweep results, and any regret/time-series as "
                "ARRAYS in metrics.json (not only final scalars or stdout logs)."
                if missing_evidence else ""
            )
        ),
    }
    raise RubricGuardFailure(json.dumps(detail))


__all__ = ["RubricGuardFailure", "assert_metrics_schema", "_path_resolves"]
