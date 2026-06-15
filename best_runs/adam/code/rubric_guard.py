"""Self-validating rubric guard — verbatim copy from backend/agents/rlm/rubric_guard.py.
Zero non-stdlib dependencies; copy-paste route always works in any sandbox."""

from __future__ import annotations  # enables PEP 604 union syntax on Python <3.10

import fnmatch
import json
import os
import re
from pathlib import Path
from typing import Any


def _path_resolves(metrics: Any, json_path: str) -> bool:
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
    pass


def _walk_keys(obj: Any, prefix: str = "") -> set[str]:
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
    if artifact_dir is None:
        artifact_dir = os.environ.get("OUTPUT_DIR", "/artifacts")
    return Path(artifact_dir)


def _artifact_matches(artifact_dir: Path, pattern: str) -> bool:
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
    if not isinstance(metrics, dict):
        raise RubricGuardFailure(json.dumps({
            "rubric_guard": "metrics_not_dict",
            "got_type": type(metrics).__name__,
        }))

    missing_keys: list[str] = []

    if metrics_shape:
        for mp in metrics_shape:
            json_path = (mp.get("json_path") if isinstance(mp, dict) else getattr(mp, "json_path", None)) or ""
            if not json_path:
                continue
            if not _path_resolves(metrics, json_path):
                metric_id = (mp.get("metric_id") if isinstance(mp, dict) else getattr(mp, "metric_id", None)) or json_path
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
            "Fix train.py so every required key is written to metrics.json "
            "AND every required artifact exists under $OUTPUT_DIR before exit."
        ),
    }
    raise RubricGuardFailure(json.dumps(detail))


__all__ = ["RubricGuardFailure", "assert_metrics_schema", "_path_resolves"]
