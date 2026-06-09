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
    """True iff ``required_key`` resolves under any path in ``present_paths``."""
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
    """Resolve the artifact directory, honoring ``OUTPUT_DIR`` env var."""
    if artifact_dir is None:
        artifact_dir = os.environ.get("OUTPUT_DIR", "/artifacts")
    return Path(artifact_dir)


def _artifact_matches(artifact_dir: Path, pattern: str) -> bool:
    """Return True iff at least one file under ``artifact_dir`` matches."""
    if not artifact_dir.is_dir():
        return False
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
) -> None:
    """Raise :class:`RubricGuardFailure` if metrics / artifacts are incomplete."""
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
        for mp in metrics_shape:
            json_path = (
                mp.get("json_path") if isinstance(mp, dict)
                else getattr(mp, "json_path", None)
            ) or ""
            if not json_path:
                continue
            if not _path_resolves(metrics, json_path):
                metric_id = (
                    mp.get("metric_id") if isinstance(mp, dict)
                    else getattr(mp, "metric_id", None)
                ) or json_path
                missing_keys.append(f"declared path {json_path!r} (id={metric_id!r})")
    else:
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

    present_keys_sample: list[str] = sorted(_walk_keys(metrics))[:20]
    detail: dict[str, Any] = {
        "rubric_guard": "schema_violation",
        "missing_keys": missing_keys,
        "missing_artifacts": missing_artifacts,
        "artifact_dir": str(_resolve_artifact_dir(artifact_dir)) if required_artifacts else None,
        "present_keys_sample": present_keys_sample,
        "hint": (
            "The rubric grader will lose points (or score 0) on the affected "
            "areas. Fix train.py so every required key is written to "
            "metrics.json AND every required artifact exists under "
            "$OUTPUT_DIR before the script exits."
        ),
    }
    raise RubricGuardFailure(json.dumps(detail))


__all__ = ["RubricGuardFailure", "assert_metrics_schema", "_path_resolves"]
