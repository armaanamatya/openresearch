# Intra-run Context Map (PEEK-lite) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a free, deterministic, intra-run orientation cache that unions the structured outputs of the three orientation primitives into a bounded JSON artifact the RLM root reads in one cheap primitive call, so it stops re-deriving facts via paid `rlm_query`/`llm_query`.

**Architecture:** One new module `context_map.py` owns `runs/<id>/rlm_state/context_map.json` (union-per-field keying, dedup, soft entry/value caps + a hard byte ceiling, atomic write, thread-safe, fail-soft). A one-line write hook in `binding.py`'s primitive success path records orientation-primitive outputs. A new pure-I/O `read_context_map()` primitive exposes the map to the root. One prompt line points the root at it. Everything is gated by `REPROLAB_CONTEXT_MAP` (default off).

**Tech Stack:** Python 3.14, pytest, the existing RLM primitive layer (`backend/agents/rlm/`).

**Spec:** `docs/superpowers/specs/2026-05-30-intra-run-context-map-design.md`

---

## Task 1: `context_map.py` — read side + flag

**Files:**
- Create: `backend/agents/rlm/context_map.py`
- Test: `tests/agents/rlm/test_context_map.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/rlm/test_context_map.py
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from backend.agents.rlm import context_map as cm


def _on(monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")


def test_read_missing_file_returns_empty(tmp_path, monkeypatch):
    _on(monkeypatch)
    out = cm.read(tmp_path)
    assert out == {"version": "v1", "bytes": 0, "entries": []}


def test_read_disabled_returns_empty_even_if_file_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("REPROLAB_CONTEXT_MAP", raising=False)
    (tmp_path / "rlm_state").mkdir()
    (tmp_path / "rlm_state" / "context_map.json").write_text(
        json.dumps({"version": "v1", "bytes": 5, "entries": [{"key": "x"}]})
    )
    assert cm.read(tmp_path)["entries"] == []


def test_read_corrupt_file_returns_empty(tmp_path, monkeypatch):
    _on(monkeypatch)
    (tmp_path / "rlm_state").mkdir()
    (tmp_path / "rlm_state" / "context_map.json").write_text("{not json")
    assert cm.read(tmp_path)["entries"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_context_map.py -q`
Expected: FAIL — `ModuleNotFoundError: backend.agents.rlm.context_map`

- [ ] **Step 3: Write the module (read side + flag + helpers)**

```python
# backend/agents/rlm/context_map.py
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
_MAX_ENTRIES: Final[int] = 40
_MAX_VALUES_PER_ENTRY: Final[int] = 8
_MAX_BYTES: Final[int] = 2048

_lock = threading.Lock()

_LIST: Final[str] = "list"
_SCALAR: Final[str] = "scalar"

# {primitive: {source_field: (kind, out_field)}}. Only these fields are unioned;
# everything else in a primitive's result (notably _meta, the dockerfile blob,
# free-form other_hparams) is ignored.
_FIELD_SPEC: Final[dict[str, dict[str, tuple[str, str]]]] = {
    "understand_section": {
        "datasets":        (_LIST, "datasets"),
        "metrics":         (_LIST, "metrics"),
        "hardware_clues":  (_LIST, "hardware_clues"),
        "ambiguities":     (_LIST, "open_questions"),
        "training_recipe": (_SCALAR, "training_recipe"),
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


__all__ = ["is_enabled", "read", "record", "ORIENTATION_PRIMITIVES"]
```

(Note: `record` is referenced in `__all__` and added in Task 2.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_context_map.py -q`
Expected: 3 passed (Task-2 tests not written yet).

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/context_map.py tests/agents/rlm/test_context_map.py
git commit -m "feat(phase8): context_map read side + flag (REPROLAB_CONTEXT_MAP)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `record()` — union, dedup, caps, atomic persist

**Files:**
- Modify: `backend/agents/rlm/context_map.py`
- Test: `tests/agents/rlm/test_context_map.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/agents/rlm/test_context_map.py

def _key(out, entries):
    return next((e for e in entries if e["key"] == out), None)


def test_union_accumulates_distinct_scalars_no_clobber(tmp_path, monkeypatch):
    """The SDAR regression: batch_size from two sections must both survive."""
    _on(monkeypatch)
    cm.record(tmp_path, "extract_hyperparameters", {"batch_size": 8}, slice_hint="sec-1.7B")
    cm.record(tmp_path, "extract_hyperparameters", {"batch_size": 16}, slice_hint="sec-7B")
    entry = _key("extract_hyperparameters:batch_size", cm.read(tmp_path)["entries"])
    assert sorted(v["value"] for v in entry["values"]) == [8, 16]


def test_union_flattens_list_fields_per_element(tmp_path, monkeypatch):
    _on(monkeypatch)
    cm.record(tmp_path, "understand_section", {"datasets": [{"name": "ALFWorld"}]}, slice_hint="a")
    cm.record(tmp_path, "understand_section", {"datasets": [{"name": "WebShop"}]}, slice_hint="b")
    entry = _key("understand_section:datasets", cm.read(tmp_path)["entries"])
    names = sorted(v["value"]["name"] for v in entry["values"])
    assert names == ["ALFWorld", "WebShop"]


def test_dedup_identical_value_is_noop(tmp_path, monkeypatch):
    _on(monkeypatch)
    cm.record(tmp_path, "extract_hyperparameters", {"batch_size": 8}, slice_hint="x")
    cm.record(tmp_path, "extract_hyperparameters", {"batch_size": 8}, slice_hint="y")
    entry = _key("extract_hyperparameters:batch_size", cm.read(tmp_path)["entries"])
    assert len(entry["values"]) == 1


def test_empty_and_null_values_skipped(tmp_path, monkeypatch):
    _on(monkeypatch)
    cm.record(tmp_path, "extract_hyperparameters",
              {"batch_size": None, "optimizer": "", "learning_rate": 0.1}, slice_hint="x")
    entries = cm.read(tmp_path)["entries"]
    keys = {e["key"] for e in entries}
    assert keys == {"extract_hyperparameters:learning_rate"}


def test_non_orientation_primitive_writes_nothing(tmp_path, monkeypatch):
    _on(monkeypatch)
    cm.record(tmp_path, "run_experiment", {"success": True, "metrics": {"acc": 1.0}}, slice_hint="x")
    assert not (tmp_path / "rlm_state" / "context_map.json").exists()


def test_disabled_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("REPROLAB_CONTEXT_MAP", raising=False)
    cm.record(tmp_path, "extract_hyperparameters", {"batch_size": 8}, slice_hint="x")
    assert not (tmp_path / "rlm_state" / "context_map.json").exists()


def test_value_cap_refuses_new_keeps_existing(tmp_path, monkeypatch):
    _on(monkeypatch)
    for i in range(12):  # > _MAX_VALUES_PER_ENTRY (8)
        cm.record(tmp_path, "extract_hyperparameters", {"batch_size": i}, slice_hint=f"s{i}")
    entry = _key("extract_hyperparameters:batch_size", cm.read(tmp_path)["entries"])
    assert len(entry["values"]) == 8
    assert [v["value"] for v in entry["values"]] == list(range(8))  # earliest kept


def test_byte_ceiling_rolls_back_oversized_mutation(tmp_path, monkeypatch):
    _on(monkeypatch)
    cm.record(tmp_path, "extract_hyperparameters", {"batch_size": 8}, slice_hint="x")
    before = (tmp_path / "rlm_state" / "context_map.json").read_text()
    huge = {"name": "x" * 5000}
    cm.record(tmp_path, "understand_section", {"datasets": [huge]}, slice_hint="big")
    after = (tmp_path / "rlm_state" / "context_map.json").read_text()
    assert before == after  # oversized mutation rolled back; prior object stands


def test_concurrent_writes_do_not_lose_values(tmp_path, monkeypatch):
    _on(monkeypatch)

    def worker(n):
        cm.record(tmp_path, "extract_hyperparameters", {"learning_rate": n}, slice_hint=f"s{n}")

    threads = [threading.Thread(target=worker, args=(i / 100.0,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    entry = _key("extract_hyperparameters:learning_rate", cm.read(tmp_path)["entries"])
    assert len(entry["values"]) == 8  # no lost updates under the lock
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_context_map.py -q`
Expected: FAIL — `record` referenced in `__all__` but not defined → `AttributeError`.

- [ ] **Step 3: Add `record()` + private helpers to `context_map.py`**

Insert before `__all__`:

```python
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
        # drop the just-created empty entry so a full map never grows empties
        if not values and entry in entries:
            entries.remove(entry)
        return False
    values.append({"value": element, "dedup": dedup,
                   "slice_hash": slice_hash, "iteration": iteration, "ts": ts})
    return True


def _persist(project_dir: Path, obj: dict) -> bool:
    """Atomic write with a hard byte ceiling. Return False (rollback) if oversized."""
    obj["version"] = _VERSION
    obj.pop("bytes", None)
    obj["bytes"] = len(json.dumps(obj, default=str, ensure_ascii=False).encode("utf-8"))
    final = json.dumps(obj, default=str, ensure_ascii=False)
    if len(final.encode("utf-8")) > _MAX_BYTES:
        logger.debug("context_map: byte ceiling (%d) exceeded, rolling back", _MAX_BYTES)
        return False
    d = project_dir / "rlm_state"
    d.mkdir(parents=True, exist_ok=True)
    p = _path(project_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(final, encoding="utf-8")
    os.replace(tmp, p)
    return True


def record(project_dir: Path, primitive: str, result: Any, *,
           slice_hint: Any = None, iteration: int | None = None) -> None:
    """Union an orientation primitive's structured fields into the map. Fail-soft."""
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
                if _union(obj, primitive, key, field, elem, sh, iteration, ts):
                    changed = True
            if changed:
                _persist(project_dir, obj)
    except Exception:  # noqa: BLE001 — context map MUST NOT block the run
        logger.exception("context_map.record failed for %s", primitive)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_context_map.py -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/context_map.py tests/agents/rlm/test_context_map.py
git commit -m "feat(phase8): context_map record() — union/dedup/caps/atomic persist

Union-per-field (not latest-wins) defeats the SDAR multi-section clobber.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `read_context_map()` primitive + registry wiring

**Files:**
- Modify: `backend/agents/rlm/primitives.py` (add primitive after `respond_to_user`; register in `PRIMITIVE_REGISTRY` ~4151 and `PRIMITIVE_DESCRIPTIONS` ~4158)
- Modify: `backend/agents/rlm/binding.py` (`PRIMITIVE_TIMEOUT_S` ~line 51)
- Test: `tests/agents/rlm/test_read_context_map_primitive.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/rlm/test_read_context_map_primitive.py
from __future__ import annotations

import pytest

from backend.agents.rlm import context_map as cm
from backend.agents.rlm.primitives import read_context_map


class _Ctx:
    def __init__(self, project_dir):
        self.project_dir = project_dir


def test_read_context_map_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    out = read_context_map(ctx=_Ctx(tmp_path))
    assert out == {"version": "v1", "bytes": 0, "entries": []}


def test_read_context_map_returns_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    cm.record(tmp_path, "extract_hyperparameters", {"batch_size": 8}, slice_hint="x")
    out = read_context_map(ctx=_Ctx(tmp_path))
    assert out["entries"][0]["key"] == "extract_hyperparameters:batch_size"


def test_read_context_map_failsoft(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    (tmp_path / "rlm_state").mkdir()
    (tmp_path / "rlm_state" / "context_map.json").write_text("{bad")
    out = read_context_map(ctx=_Ctx(tmp_path))
    assert out["entries"] == []


def test_read_context_map_registered():
    from backend.agents.rlm.primitives import PRIMITIVE_REGISTRY, PRIMITIVE_DESCRIPTIONS
    assert "read_context_map" in PRIMITIVE_REGISTRY
    assert "read_context_map" in PRIMITIVE_DESCRIPTIONS
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_read_context_map_primitive.py -q`
Expected: FAIL — `ImportError: cannot import name 'read_context_map'`.

- [ ] **Step 3: Add the primitive (after `respond_to_user`, ~primitives.py:4040)**

```python
def read_context_map(*, ctx: "RunContext") -> dict:
    """Return the intra-run context map (PEEK-lite orientation cache).

    Pure file I/O — no LLM call. Mirrors check_user_messages. Returns the
    accumulated map {version, bytes, entries:[...]} where each entry unions a
    primitive field's observed values across paper sections (datasets, metrics,
    hyperparameters, environment facts), each with provenance. Returns the
    empty-map shape when REPROLAB_CONTEXT_MAP is off or on any error.

    Treat entries as heuristic hints, not ground truth: a field may list several
    observed values across sections. See
    docs/superpowers/specs/2026-05-30-intra-run-context-map-design.md.
    """
    from backend.agents.rlm import context_map as _cmap
    try:
        return _cmap.read(ctx.project_dir)
    except Exception:  # noqa: BLE001 — fail-soft; a navigation aid must never raise
        return {"version": "v1", "bytes": 0, "entries": []}
```

- [ ] **Step 4: Register in `PRIMITIVE_REGISTRY` (after `"respond_to_user": respond_to_user,`)**

```python
    "read_context_map": read_context_map,
```

- [ ] **Step 5: Register in `PRIMITIVE_DESCRIPTIONS` (after the `respond_to_user` entry)**

```python
    "read_context_map": "read_context_map() -> dict — the intra-run context "
        "map: accumulated datasets, metrics, hyperparameters, and environment "
        "facts already extracted this run, each with provenance. Call it before "
        "re-deriving a known fact via rlm_query/llm_query. Entries are heuristic "
        "hints; a field may list several observed values across paper sections.",
```

- [ ] **Step 6: Add the wall-clock entry in `binding.py` `PRIMITIVE_TIMEOUT_S` (after `"respond_to_user": 30,`)**

```python
    "read_context_map": 30,
```

- [ ] **Step 7: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_read_context_map_primitive.py -q`
Expected: all passed.

- [ ] **Step 8: Commit**

```bash
git add backend/agents/rlm/primitives.py backend/agents/rlm/binding.py tests/agents/rlm/test_read_context_map_primitive.py
git commit -m "feat(phase8): read_context_map primitive + registry/description/timeout

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: write hook in `binding.py` (the one DRY site)

**Files:**
- Modify: `backend/agents/rlm/binding.py` (success path, right after `_emit_supplemental(name, result, ctx, _emit_extra)` ~line 513)
- Test: `tests/agents/rlm/test_context_map_write_hook.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/rlm/test_context_map_write_hook.py
from __future__ import annotations

import pytest

from backend.agents.rlm.binding import build_custom_tools
from backend.agents.rlm import context_map as cm


def _tool(ctx, name, fn):
    tools = build_custom_tools(ctx, registry={name: fn}, descriptions={name: name})
    return tools[name]["tool"]


def test_orientation_success_writes_map(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    ctx = make_context(tmp_path)
    tool = _tool(ctx, "understand_section", lambda s, *, ctx: {"datasets": [{"name": "ALFWorld"}]})
    tool("a paper slice")
    entries = cm.read(ctx.project_dir)["entries"]
    assert any(e["key"] == "understand_section:datasets" for e in entries)


def test_non_orientation_success_writes_nothing(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    ctx = make_context(tmp_path)
    tool = _tool(ctx, "run_experiment", lambda *, ctx: {"success": True, "metrics": {}})
    tool()
    assert not (ctx.project_dir / "rlm_state" / "context_map.json").exists()


def test_failed_orientation_result_writes_nothing(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    ctx = make_context(tmp_path)
    # failure-shaped dict → wrap_primitive takes the `failed` branch → hook not reached
    tool = _tool(ctx, "understand_section", lambda s, *, ctx: {"error": "boom", "datasets": [{"name": "X"}]})
    tool("slice")
    assert not (ctx.project_dir / "rlm_state" / "context_map.json").exists()


def test_flag_off_writes_nothing(make_context, tmp_path, monkeypatch):
    monkeypatch.delenv("REPROLAB_CONTEXT_MAP", raising=False)
    ctx = make_context(tmp_path)
    tool = _tool(ctx, "understand_section", lambda s, *, ctx: {"datasets": [{"name": "ALFWorld"}]})
    tool("slice")
    assert not (ctx.project_dir / "rlm_state" / "context_map.json").exists()


def test_hook_exception_does_not_break_primitive(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    monkeypatch.setattr(cm, "record", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    ctx = make_context(tmp_path)
    tool = _tool(ctx, "understand_section", lambda s, *, ctx: {"datasets": [{"name": "ALFWorld"}]})
    result = tool("slice")  # must still return the primitive's result
    assert result["datasets"][0]["name"] == "ALFWorld"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_context_map_write_hook.py -q`
Expected: FAIL — `test_orientation_success_writes_map` (no hook yet, no file written).

- [ ] **Step 3: Add the hook in `binding.py`**

Find (in `wrap_primitive.wrapped`, the success-else branch ~line 511-513):

```python
            else:
                # --- Phase 6 (Task 13): post-success supplemental event emission ---
                _emit_supplemental(name, result, ctx, _emit_extra)
            return result
```

Replace the `else` body with (adds the context-map write hook):

```python
            else:
                # --- Phase 6 (Task 13): post-success supplemental event emission ---
                _emit_supplemental(name, result, ctx, _emit_extra)
                # --- Phase 8: union orientation outputs into the intra-run map ---
                try:
                    from backend.agents.rlm import context_map as _cmap
                    _slice_hint = args[0] if args else None
                    _cmap.record(ctx.project_dir, name, result,
                                 slice_hint=_slice_hint,
                                 iteration=getattr(ctx, "iteration", None))
                except Exception:  # noqa: BLE001 — context map MUST NOT break the run
                    pass
            return result
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_context_map_write_hook.py -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/binding.py tests/agents/rlm/test_context_map_write_hook.py
git commit -m "feat(phase8): write hook — union orientation outputs at the success chokepoint

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: system-prompt line

**Files:**
- Modify: `backend/agents/rlm/system_prompt.py` (near the `understand_section` guidance, ~line 145/303)
- Test: `tests/agents/rlm/test_context_map_prompt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/rlm/test_context_map_prompt.py
from backend.agents.rlm import system_prompt as sp


def test_prompt_mentions_read_context_map():
    text = sp.SYSTEM_PROMPT if hasattr(sp, "SYSTEM_PROMPT") else sp.build_system_prompt()
    assert "read_context_map" in text
```

(If neither symbol exists, Step 2's failure output names the right one; use the module-level prompt constant this repo exposes — grep `system_prompt.py` for the prompt string and assert against it.)

- [ ] **Step 2: Run to verify failure / discover the symbol**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_context_map_prompt.py -q`
Expected: FAIL — either AttributeError (fix the test to the real symbol) or AssertionError (string absent).

- [ ] **Step 3: Add the prompt line**

Insert near the `understand_section` re-derivation guidance:

```
Before re-deriving a known fact via `rlm_query`/`llm_query`, call
`read_context_map()` — it accumulates the datasets, metrics, hyperparameters,
and environment facts already extracted this run, each with provenance. Treat
its entries as heuristic hints, not ground truth; a field may list several
observed values across paper sections.
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_context_map_prompt.py -q`
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/system_prompt.py tests/agents/rlm/test_context_map_prompt.py
git commit -m "feat(phase8): point the root at read_context_map in the system prompt

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: docs — flag + plan status

**Files:**
- Modify: `CLAUDE.md` (new sub-section documenting `REPROLAB_CONTEXT_MAP`)
- Modify: `docs/superpowers/plans/2026-05-30-rlm-wedge-hardening-and-evolution.md` (mark Phase 8 implemented)

- [ ] **Step 1: Add a CLAUDE.md sub-section**

Under the RLM section, add:

```markdown
### Intra-run context map (PEEK-lite, spec 2026-05-30)
`REPROLAB_CONTEXT_MAP=on` (default off) turns on a free, deterministic intra-run
orientation cache. The write hook in `binding.py` (success chokepoint) unions
the structured outputs of `understand_section` / `extract_hyperparameters` /
`detect_environment` into `runs/<id>/rlm_state/context_map.json` — keyed
`primitive:field`, **union-per-field** (NOT latest-wins) so a multi-section
paper (SDAR's 3 model sizes + 3 environments) accumulates instead of clobbering.
The root reads it via the pure-I/O `read_context_map()` primitive and is told
(system prompt) to consult it before re-deriving a fact via `rlm_query`/
`llm_query`. Bounded: ≤40 fields, ≤8 values/field, ≤2 KB (hard ceiling rolls
back oversized mutations). The map is a **navigation aid, never a report
source** — the Phase 3 evidence gate remains the report backstop. Module:
`backend/agents/rlm/context_map.py`. A/B before making default: `=on` vs `=off`
across ≥3 paired SDAR runs, compare iteration count + `rlm_query`/`llm_query`
call counts; `rubric.overall_score` must not regress. Rollback: unset the flag.
```

- [ ] **Step 2: Mark Phase 8 in the master plan's status table**

Update the implementation-status row for Phase 8 to "implemented (flagged, default off)".

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/superpowers/plans/2026-05-30-rlm-wedge-hardening-and-evolution.md
git commit -m "docs(phase8): document REPROLAB_CONTEXT_MAP + mark plan status

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: full regression

- [ ] **Step 1: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q -x`
Expected: all green (prior 1397 + the new Phase-8 tests). If a pre-existing unrelated flaky network test fails, re-run with `--reruns 2`.

- [ ] **Step 2: Confirm flag-off is a true no-op**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_context_map.py tests/agents/rlm/test_context_map_write_hook.py -q`
Expected: the `*_disabled_*` / `*_flag_off_*` tests prove zero artifacts written when the flag is unset.

---

## Self-Review

**Spec coverage:** §2.1 module → T1+T2; §2.2 write hook → T4; §2.3 read primitive → T3; §2.4 prompt → T5; §3 field rules → T2 (`_FIELD_SPEC`); §4 union/dedup/caps/byte-ceiling → T2 tests; §5 consumption → T3; §6 safety (additive, navigation-only) → enforced by union (T2) + hook-not-on-failed-path (T4); §7 flag → T1/T2/T4 disabled tests; §8 measurement → manual A/B (documented T6, not code); §9 testing → T1-T5; §10 files → all tasks. No gaps.

**Placeholder scan:** none — every code/test step is complete. (T5 Step-1 notes a symbol-discovery fallback, with the exact command to resolve it — not a placeholder.)

**Type consistency:** `record(project_dir, primitive, result, *, slice_hint, iteration)`, `read(project_dir) -> dict`, entry shape `{key, primitive, field, confidence, values:[{value, dedup, slice_hash, iteration, ts}]}`, and the env var `REPROLAB_CONTEXT_MAP` are identical across T1-T6.
