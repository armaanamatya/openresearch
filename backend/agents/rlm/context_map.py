"""Intra-run context map (PEEK-lite) — a bounded, deterministic orientation cache.

See docs/superpowers/specs/2026-05-30-intra-run-context-map-design.md.

The three orientation primitives (understand_section, extract_hyperparameters,
detect_environment) are deterministic heuristics the root calls *per paper
section*, so their per-slice outputs never form a unified view of a
multi-section paper. This module unions their structured fields into one
bounded JSON artifact the root reads in a single cheap primitive call, so it
stops re-deriving the same facts through paid rlm_query/llm_query sub-calls.

Storage: runs/<project_id>/rlm_state/context_map.json — one JSON object,
atomic-written. Union-per-field keying (NOT latest-wins) so a multi-section
paper (e.g. SDAR's 3 model sizes + 3 environments) accumulates instead of
clobbering.

Contract (mirrors primitive_cache.py):
  * Fail-soft on every path — persistence must never block/crash a run.
  * Opt-in: REPROLAB_CONTEXT_MAP must be truthy; default off.
  * Versioned ("v1") so a contract change invalidates without a manual purge.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

_VERSION: Final[str] = "v1"
_FILENAME: Final[str] = "context_map.json"
_ENABLE_ENV_VAR: Final[str] = "REPROLAB_CONTEXT_MAP"

# Soft caps (deterministic, refuse-new-keep-existing) + hard byte ceiling.
# _MAX_BYTES is 8 KB, NOT the ~1.5 KB a PEEK-style prompt-injected snapshot would
# use: this map is a ``read_context_map()`` RETURN VALUE, never injected into the
# prompt, so its size never touches the prompt cache. The only cost of a larger
# cap is REPL tokens when the root reads it — far cheaper than the rlm_query
# re-derivation it replaces. A measured full SDAR orientation pass (3 model sizes
# × 3 environments) is ~3.6 KB; 8 KB holds it with >2× headroom so the union
# accumulation isn't silently truncated at the byte layer.
_MAX_ENTRIES: Final[int] = 40
_MAX_VALUES_PER_ENTRY: Final[int] = 8
_MAX_BYTES: Final[int] = 8192

_lock = threading.Lock()

_LIST: Final[str] = "list"
_SCALAR: Final[str] = "scalar"

# {primitive: {source_field: (kind, out_field)}}. Only these fields are unioned;
# everything else in a primitive's result (notably _meta, the dockerfile blob,
# free-form other_hparams) is ignored.
_FIELD_SPEC: Final[dict[str, dict[str, tuple[str, str]]]] = {
    # NOTE: `ambiguities` is intentionally excluded — it is verbose (7+ dicts per
    # call) and the LEAST fact-like field (open questions, not facts the root
    # re-derives). Including it would consume the byte budget and crowd out the
    # valuable datasets/metrics/recipe facts. Ordered valuable-first so the
    # incremental byte ceiling keeps the high-value entries.
    "understand_section": {
        "datasets":        (_LIST, "datasets"),
        "metrics":         (_LIST, "metrics"),
        "training_recipe": (_SCALAR, "training_recipe"),
        "hardware_clues":  (_LIST, "hardware_clues"),
    },
    "extract_hyperparameters": {
        "optimizer":       (_SCALAR, "optimizer"),
        "learning_rate":   (_SCALAR, "learning_rate"),
        "batch_size":      (_SCALAR, "batch_size"),
        "epochs_or_steps": (_SCALAR, "epochs_or_steps"),
        "scheduler":       (_SCALAR, "scheduler"),
    },
    "detect_environment": {
        "framework":       (_SCALAR, "framework"),
        "python_version":  (_SCALAR, "python_version"),
    },
}

ORIENTATION_PRIMITIVES: Final[frozenset[str]] = frozenset(_FIELD_SPEC)


def is_enabled() -> bool:
    return os.environ.get(_ENABLE_ENV_VAR, "").strip().lower() in ("1", "on", "true", "yes")


def _empty() -> dict:
    return {"version": _VERSION, "bytes": 0, "entries": []}


def _path(project_dir: Path) -> Path:
    return project_dir / "rlm_state" / _FILENAME


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False


def _dedup_id(element: Any) -> str:
    blob = json.dumps(element, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _slice_hash_from(slice_hint: Any) -> str | None:
    if isinstance(slice_hint, str) and slice_hint.strip():
        return hashlib.sha256(slice_hint.encode("utf-8")).hexdigest()[:8]
    if isinstance(slice_hint, dict) and slice_hint:
        blob = json.dumps(slice_hint, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:8]
    return None


def _load(project_dir: Path) -> dict:
    p = _path(project_dir)
    if not p.exists():
        return _empty()
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty()
    if not isinstance(obj, dict) or not isinstance(obj.get("entries"), list):
        return _empty()
    obj.setdefault("version", _VERSION)
    return obj


def read(project_dir: Path) -> dict:
    """Return the map object {version, bytes, entries:[...]}; empty when off/error."""
    if not is_enabled():
        return _empty()
    if not isinstance(project_dir, Path):
        return _empty()
    try:
        return _load(project_dir)
    except Exception:  # noqa: BLE001 — fail-soft; a read must never raise
        return _empty()


def _union(obj: dict, primitive: str, key: str, field: str,
           element: Any, slice_hash: str | None, iteration: int | None, ts: str) -> bool:
    """Union one element into the keyed entry. Return True iff a value was added."""
    entries = obj.setdefault("entries", [])
    entry = next((e for e in entries if e.get("key") == key), None)
    if entry is None:
        if len(entries) >= _MAX_ENTRIES:
            logger.debug("context_map: entry cap (%d) reached, refusing key %s", _MAX_ENTRIES, key)
            return False
        entry = {"key": key, "primitive": primitive, "field": field,
                 "confidence": "heuristic", "values": []}
        entries.append(entry)
    values = entry.setdefault("values", [])
    dedup = _dedup_id(element)
    if any(v.get("dedup") == dedup for v in values):
        return False  # idempotent — already observed (e.g. cache-hit replay)
    if len(values) >= _MAX_VALUES_PER_ENTRY:
        logger.debug("context_map: value cap (%d) reached for %s, refusing value",
                     _MAX_VALUES_PER_ENTRY, key)
        return False
    values.append({"value": element, "dedup": dedup,
                   "slice_hash": slice_hash, "iteration": iteration, "ts": ts})
    return True


def _serialized_bytes(obj: dict) -> int:
    """Byte size of the map's persisted form, excluding the informational
    ``bytes`` field itself (which adds a fixed ~15 bytes)."""
    payload = {k: v for k, v in obj.items() if k != "bytes"}
    return len(json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8"))


def _undo_last_value(obj: dict, key: str) -> None:
    """Remove the value just appended to ``key`` (the last in its entry); drop
    the entry if it became empty."""
    entry = next((e for e in obj.get("entries", []) if e.get("key") == key), None)
    if entry is None:
        return
    if entry.get("values"):
        entry["values"].pop()
    if not entry.get("values"):
        obj["entries"].remove(entry)


def _persist(project_dir: Path, obj: dict) -> None:
    """Atomic write of the map (the byte ceiling is enforced incrementally in
    ``record`` before this is called)."""
    obj["version"] = _VERSION
    obj.pop("bytes", None)
    obj["bytes"] = _serialized_bytes(obj)
    final = json.dumps(obj, default=str, ensure_ascii=False)
    d = project_dir / "rlm_state"
    d.mkdir(parents=True, exist_ok=True)
    p = _path(project_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(final, encoding="utf-8")
    os.replace(tmp, p)


def record(project_dir: Path, primitive: str, result: Any, *,
           slice_hint: Any = None, iteration: int | None = None) -> None:
    """Union an orientation primitive's structured fields into the map. Fail-soft.

    The byte ceiling is enforced INCREMENTALLY: observations are added
    valuable-first (per ``_FIELD_SPEC`` order), and the first value that would
    push the map over ``_MAX_BYTES`` is undone and the rest skipped. This keeps
    the high-value early entries instead of an all-or-nothing rollback where one
    verbose field shuts out the whole map.
    """
    if not is_enabled():
        return
    spec = _FIELD_SPEC.get(primitive)
    if spec is None or not isinstance(result, dict) or not isinstance(project_dir, Path):
        return

    observations: list[tuple[str, str, Any]] = []
    for src_field, (kind, out_field) in spec.items():
        raw = result.get(src_field)
        key = f"{primitive}:{out_field}"
        if kind == _LIST:
            if not isinstance(raw, list):
                continue
            for elem in raw:
                if not _is_empty(elem):
                    observations.append((key, out_field, elem))
        elif not _is_empty(raw):
            observations.append((key, out_field, raw))
    if not observations:
        return

    sh = _slice_hash_from(slice_hint)
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with _lock:
            obj = _load(project_dir)
            changed = False
            for key, field, elem in observations:
                if not _union(obj, primitive, key, field, elem, sh, iteration, ts):
                    continue
                if _serialized_bytes(obj) > _MAX_BYTES:
                    _undo_last_value(obj, key)
                    logger.debug("context_map: byte ceiling (%d) reached, "
                                 "skipping overflow value for %s", _MAX_BYTES, key)
                    break
                changed = True
            if changed:
                _persist(project_dir, obj)
    except Exception:  # noqa: BLE001 — context map MUST NOT block the run
        logger.exception("context_map.record failed for %s", primitive)


__all__ = ["is_enabled", "read", "record", "ORIENTATION_PRIMITIVES"]
