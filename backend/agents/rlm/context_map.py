"""PEEK-lite intra-run context map (FLAG-1, ``OPENRESEARCH_CONTEXT_MAP``).

A free, deterministic orientation cache. The structured outputs of the three
orientation primitives — ``understand_section``, ``extract_hyperparameters``,
``detect_environment`` — are unioned per-field into
``runs/<id>/rlm_state/context_map.json`` so the root can consult already-known
facts before re-deriving them.

Navigation aid ONLY — never a report source (the evidence gate
``OPENRESEARCH_EVIDENCE_GATE`` remains the backstop).

Off-state contract: when ``OPENRESEARCH_CONTEXT_MAP`` is not enabled,
:func:`update_context_map` is a no-op and :func:`read_context_map` returns ``{}``.

Bounds (deterministic): ``MAX_FIELDS`` distinct fields, ``MAX_VALUES`` values
per field, ``MAX_BYTES`` serialized ceiling. Thread-safe (a module lock guards
the read-modify-write) and fail-soft (any error is swallowed — a broken
orientation cache must never break a primitive call).

The config shim aliases ``OPENRESEARCH_CONTEXT_MAP`` <-> ``REPROLAB_CONTEXT_MAP``
bidirectionally, so either spelling enables it.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

_FILENAME = "context_map.json"
MAX_FIELDS = 40
MAX_VALUES = 8
MAX_BYTES = 8192
MAX_VALUE_LEN = 200

# Only these primitives feed the map (orientation outputs, not work products).
SOURCE_PRIMITIVES = frozenset(
    {"understand_section", "extract_hyperparameters", "detect_environment"}
)

_LOCK = threading.Lock()


def _enabled() -> bool:
    return os.environ.get("OPENRESEARCH_CONTEXT_MAP", "").strip().lower() in (
        "on",
        "1",
        "true",
        "yes",
    )


def _path(project_dir: Path | str) -> Path:
    return Path(project_dir) / "rlm_state" / _FILENAME


def _scalars(value: Any) -> list[str]:
    """Flatten a payload value into a short list of scalar strings (or [])."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, str):
        v = value.strip()
        return [v[:MAX_VALUE_LEN]] if v else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, bool):
                out.append(str(item))
            elif isinstance(item, (str, int, float)):
                s = str(item).strip()
                if s:
                    out.append(s[:MAX_VALUE_LEN])
            if len(out) >= MAX_VALUES:
                break
        return out
    return []  # dicts / None / nested → skipped (navigation aid stays flat)


def read_context_map(project_dir: Path | str) -> dict[str, list[str]]:
    """Return the current context map, or ``{}`` when disabled / absent / unreadable."""
    if not _enabled():
        return {}
    try:
        path = _path(project_dir)
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def update_context_map(project_dir: Path | str, source: str, payload: Any) -> None:
    """Union ``payload``'s scalar/list fields into the on-disk map.

    No-op unless enabled and ``source`` is an orientation primitive. Fail-soft.
    """
    if not _enabled() or source not in SOURCE_PRIMITIVES:
        return
    if not isinstance(payload, dict):
        return
    try:
        with _LOCK:
            path = _path(project_dir)
            current: dict[str, list[str]] = {}
            if path.exists():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        current = {
                            k: list(v) for k, v in loaded.items() if isinstance(v, list)
                        }
                except Exception:
                    current = {}

            for key, raw in payload.items():
                if not isinstance(key, str):
                    continue
                values = _scalars(raw)
                if not values:
                    continue
                if key not in current and len(current) >= MAX_FIELDS:
                    continue  # field budget exhausted; don't grow unboundedly
                bucket = current.setdefault(key, [])
                for val in values:
                    if val not in bucket and len(bucket) < MAX_VALUES:
                        bucket.append(val)

            # Enforce the serialized byte ceiling deterministically: drop whole
            # fields (sorted by key, last first) until under MAX_BYTES.
            blob = json.dumps(current, ensure_ascii=False, indent=2)
            while len(blob.encode("utf-8")) > MAX_BYTES and current:
                drop = sorted(current)[-1]
                current.pop(drop, None)
                blob = json.dumps(current, ensure_ascii=False, indent=2)

            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(blob, encoding="utf-8")
            os.replace(tmp, path)
    except Exception:
        return  # fail-soft: never let the orientation cache break a primitive
