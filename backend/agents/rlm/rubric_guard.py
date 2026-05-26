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
        required_artifacts: Filename literals or globs that must exist under
                            ``artifact_dir``.  Globs match flat (no recursion);
                            literals are exact-path.
        artifact_dir:       Directory containing the run's artifacts.  Defaults
                            to ``$OUTPUT_DIR`` (or ``/artifacts`` when unset).

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

    present_keys = _walk_keys(metrics)
    missing_keys = [k for k in required_keys if not _key_present(k, present_keys)]

    missing_artifacts: list[str] = []
    if required_artifacts:
        resolved_dir = _resolve_artifact_dir(artifact_dir)
        for pattern in required_artifacts:
            if not _artifact_matches(resolved_dir, pattern):
                missing_artifacts.append(pattern)

    if not missing_keys and not missing_artifacts:
        return

    detail: dict[str, Any] = {
        "rubric_guard": "schema_violation",
        "missing_keys": missing_keys,
        "missing_artifacts": missing_artifacts,
        "artifact_dir": str(_resolve_artifact_dir(artifact_dir)) if required_artifacts else None,
        "present_keys_sample": sorted(list(present_keys))[:20],
        "hint": (
            "The rubric grader will lose points (or score 0) on the affected "
            "areas. Fix train.py so every required key is written to "
            "metrics.json AND every required artifact exists under "
            "$OUTPUT_DIR before the script exits."
        ),
    }
    raise RubricGuardFailure(json.dumps(detail))


__all__ = ["RubricGuardFailure", "assert_metrics_schema"]
