"""Deterministic-by-construction leaf checker (grader-fidelity Workstream A2).

~Half of a PaperBench rubric's leaves are **mechanically checkable** —
hyperparameters (epochs, momentum, weight-decay, LR schedule), artifact /
script existence, and numeric target *trends* — yet today they all go to the
**noisy LLM grader** (no temperature/seed → ~2.5 % drift, 2026-06-16 design
§A1/§A2). Sending a leaf whose ground truth is a single field in
``provenance.json`` to an LLM is both wasteful (3× cost under median-of-N) and
*less* reliable than a string compare.

This module is the pure-Python checker for those leaves. At rubric-gen time an
**optional** structured annotation is attached to a leaf
(``leaf["check_kind"]`` + ``leaf["assertion"]``); the router in
``score_reproduction`` calls :func:`check_leaf` first and only falls through to
the LLM when this returns ``None``.

Backwards-compat guarantee (the load-bearing invariant)
--------------------------------------------------------
A leaf with **no** recognized ``check_kind`` (every leaf in an old rubric)
makes :func:`check_leaf` return ``None`` → the caller routes it to the LLM
exactly as before. Deterministic routing only *adds* coverage where the
annotation exists; it can **never break an un-annotated rubric**.

Leaf-annotation schema (the contract — rubric-gen + the integrator MUST match)
------------------------------------------------------------------------------
A deterministic leaf carries two extra keys on the leaf dict::

    {
      "id": "<leaf id, any JSON scalar>",          # existing rubric field
      ...,                                          # existing rubric fields
      "check_kind": "deterministic:hparam"          # one of the three below
                  | "deterministic:artifact"
                  | "deterministic:numeric",
      "assertion": { ... }                          # kind-specific, see below
    }

* ``deterministic:hparam`` — ``assertion`` is
  ``{"field": <str>, "op": <op>, "value": <scalar>, "tolerance": <float?>}``
  where ``op`` ∈ ``{"==", "!=", ">=", "<=", "~="}`` (``~=`` compares with an
  absolute ``tolerance``, default ``1e-9``). ``field`` is looked up in
  ``provenance.json`` — first at the manifest top level, then inside each
  ``experiments[*]`` record (where the agent's emitter actually writes
  ``epochs`` / ``batch_size`` / ``seed`` / ``per_optimizer.*``). A dotted
  ``field`` (``"per_optimizer.lr"``) traverses nested dicts.

* ``deterministic:artifact`` — ``assertion`` is ``{"glob": <str | [str]>}``
  (alias ``"globs"``). Existence is checked under ``run_dir`` **and**
  ``run_dir/code`` (recursively for a bare ``"name.py"`` pattern). Any one
  pattern matching → satisfied.

* ``deterministic:numeric`` — ``assertion`` is
  ``{"metric_key": <str>, "target": <float>, "tolerance": <float?>,
     "direction": <dir>}`` where ``dir`` ∈ ``{"higher_better",
  "lower_better", "trend_up", "trend_down", "within"}``. The value is read
  from the freshest results-bearing ``metrics.json`` (top level → dotted
  path → recursive key search → first numeric ``metric`` leaf). Graded on
  **trend / threshold satisfaction, not exact magnitude** (e.g.
  ``higher_better``: ``value >= target - tolerance`` → ``1.0``).

Return shape (uniform with the LLM grader's per-leaf record)
------------------------------------------------------------
On a graded leaf::

    {"id": str, "score": float in [0,1], "justification": str,
     "_graded": True, "check_kind": <kind>}

This mirrors the LLM grader's per-leaf record (``leaf_scorer.py`` emits
``{"id", "score", "justification", "_graded"}``) so the integrator can merge
deterministic + LLM leaves into one ``leaf_scores`` map uniformly. The extra
``check_kind`` key is additive provenance — the source of the grade.

Fail-soft contract
-------------------
Pure & deterministic (no clock, no network, no randomness). It **never
raises** on bad input:

* No recognized ``check_kind`` / no usable assertion → ``None``
  (route to LLM — the backwards-compat path).
* Recognized kind but the *evidence* is missing or malformed (file absent,
  bad JSON, field/metric not found) → a **graded ``0.0``** with a diagnostic
  ``justification`` (``provenance_missing:<field>`` / ``metric_missing:<key>``
  / ``artifact_missing``). A recognized-but-failing check is a real verdict
  (the run did not produce the evidence), not a routing fall-through — so it
  is ``_graded: True``, NOT ``None``.

The line between the two: a *malformed annotation* (the rubric asked for
something the checker can't interpret) falls through to the LLM (``None``);
*missing evidence* for a well-formed annotation is a failing grade (``0.0``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["check_leaf", "DETERMINISTIC_CHECK_KINDS"]

# The three recognized check kinds. A leaf whose ``check_kind`` is not in this
# set falls through to the LLM (returns None).
CHECK_HPARAM = "deterministic:hparam"
CHECK_ARTIFACT = "deterministic:artifact"
CHECK_NUMERIC = "deterministic:numeric"
DETERMINISTIC_CHECK_KINDS = frozenset({CHECK_HPARAM, CHECK_ARTIFACT, CHECK_NUMERIC})

# hparam comparison operators.
_HPARAM_OPS = frozenset({"==", "!=", ">=", "<=", "~="})
# numeric direction vocabulary.
_NUMERIC_DIRECTIONS = frozenset(
    {"higher_better", "lower_better", "trend_up", "trend_down", "within"}
)
# default absolute tolerance for ~= / numeric "within" when none supplied.
_DEFAULT_TOLERANCE = 1e-9


# --------------------------------------------------------------------------- #
# small, local helpers (deliberately NOT imported from leaf_scorer — those are
# private and may change; this module owns its own copies so it stays stable).
# --------------------------------------------------------------------------- #
def _is_number(x: Any) -> bool:
    """True for a real int/float (bools excluded — they're ints in Python)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _coerce_number(x: Any) -> float | None:
    """Best-effort numeric coercion: int/float pass; numeric strings parse."""
    if _is_number(x):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except (ValueError, AttributeError):
            return None
    return None


def _load_json(path: Path) -> Any | None:
    """Read + parse JSON; fail-soft to ``None`` (missing / unreadable / bad JSON)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — fail-soft: any read/parse error → None.
        return None


def _dotted_get(obj: Any, dotted: str) -> tuple[bool, Any]:
    """Traverse ``obj`` by a dotted key path.

    Returns ``(found, value)``. ``found`` is False the moment a segment is
    missing or a non-dict is hit. A single (un-dotted) key is the common case.
    """
    cur = obj
    for seg in dotted.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return (False, None)
    return (True, cur)


def _find_provenance_field(prov: Any, field: str) -> tuple[bool, Any]:
    """Locate ``field`` in a provenance manifest, fail-soft.

    Search order (first hit wins):
      1. manifest top level (dotted-aware) — e.g. ``run_id``.
      2. each ``experiments[*]`` record (dotted-aware) — where the agent's
         emitter writes ``epochs``/``batch_size``/``seed``/``per_optimizer.*``.

    Per-experiment search makes the common rubric assertion ("epochs == 45")
    resolve even though ``epochs`` lives one level down per experiment rather
    than at the manifest root.
    """
    if not isinstance(prov, dict):
        return (False, None)
    # 1. top level.
    found, val = _dotted_get(prov, field)
    if found:
        return (True, val)
    # 2. inside experiments.
    exps = prov.get("experiments")
    if isinstance(exps, dict):
        for exp in exps.values():
            found, val = _dotted_get(exp, field)
            if found:
                return (True, val)
    elif isinstance(exps, list):
        for exp in exps:
            found, val = _dotted_get(exp, field)
            if found:
                return (True, val)
    return (False, None)


def _provenance_paths(run_dir: Path) -> list[Path]:
    """``provenance.json`` candidates, newest-first.

    Mirrors ``leaf_scorer._provenance_paths`` (the producer contract): the
    agent writes ``code/provenance.json`` or
    ``code/outputs/<run_id>/provenance.json``. Re-implemented locally (not
    imported) so this module never depends on a private symbol that might move.
    """
    code_dir = run_dir / "code"
    if not code_dir.exists():
        return []
    cands = [
        p
        for p in (
            list(code_dir.glob("provenance.json"))
            + list(code_dir.glob("outputs/*/provenance.json"))
        )
        if p.is_file()
    ]
    cands.sort(key=lambda p: _safe_mtime(p), reverse=True)
    return cands


def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _metrics_has_measured_value(d: Any) -> bool:
    """True iff a metrics dict carries any *measured* value, not a placeholder.

    A run accumulates one ``metrics.json`` per ``run_experiment`` call,
    including just-created empty/in-progress ones. ``per_model: {m: {}}`` is a
    placeholder; a real result has a numeric ``metric`` somewhere (or a
    populated ``comparison``). Ranking on this (not on bare truthiness of
    ``per_model``) keeps a placeholder from outranking genuine older data —
    the same fix A6 makes to ``_latest_metrics_path``.
    """
    if not isinstance(d, dict):
        return False
    if d.get("comparison"):
        return True
    return _any_numeric_metric(d.get("per_model"))


def _any_numeric_metric(node: Any, _depth: int = 0) -> bool:
    """Recursively: does this per_model subtree hold a numeric ``metric``?"""
    if _depth > 8 or node is None:
        return False
    if isinstance(node, dict):
        m = node.get("metric")
        if _is_number(m):
            return True
        return any(_any_numeric_metric(v, _depth + 1) for v in node.values())
    if isinstance(node, list):
        return any(_any_numeric_metric(v, _depth + 1) for v in node)
    return False


def _latest_metrics(run_dir: Path) -> Any | None:
    """Freshest results-bearing ``metrics.json`` content (parsed), fail-soft.

    Ranks ``(has_measured_value, mtime)`` so the newest *measured* result wins
    over a newer-but-empty placeholder, falling back to newest-overall. Returns
    the parsed object, or ``None`` if no ``metrics.json`` exists / parses.
    A local re-implementation of ``_latest_metrics_path`` + read (A6-aligned).
    """
    cands: list[Path] = []
    outputs = run_dir / "code" / "outputs"
    if outputs.exists():
        cands.extend(p for p in outputs.rglob("metrics.json") if p.is_file())
    top = run_dir / "code" / "metrics.json"
    if top.is_file():
        cands.append(top)
    if not cands:
        return None

    def _rank(p: Path) -> tuple[int, float]:
        d = _load_json(p)
        return (1 if _metrics_has_measured_value(d) else 0, _safe_mtime(p))

    best = max(cands, key=_rank)
    return _load_json(best)


def _find_metric_value(metrics: Any, metric_key: str) -> tuple[bool, Any]:
    """Locate a metric value in a metrics dict, fail-soft.

    Search order (first hit wins):
      1. flat top-level key (dotted-aware) — e.g. ``status`` or a custom scalar.
      2. recursive search by the *last* key segment anywhere in the tree —
         catches ``per_model.<model>.<env>.<baseline>.<key>`` and the common
         convention where the headline lives under a ``metric`` field whose
         sibling names the metric.

    Returns ``(found, value)``. The value may be any JSON type; the numeric
    grader coerces it.
    """
    if not isinstance(metrics, (dict, list)):
        return (False, None)
    # 1. dotted top-level path (only meaningful for a dict root).
    if isinstance(metrics, dict):
        found, val = _dotted_get(metrics, metric_key)
        if found:
            return (True, val)
    # 2. recursive search by the final segment.
    leaf_key = metric_key.split(".")[-1]
    return _recursive_key_search(metrics, leaf_key)


def _recursive_key_search(node: Any, key: str, _depth: int = 0) -> tuple[bool, Any]:
    """First value found for ``key`` anywhere in a nested dict/list, fail-soft."""
    if _depth > 10:
        return (False, None)
    if isinstance(node, dict):
        if key in node:
            return (True, node[key])
        for v in node.values():
            found, val = _recursive_key_search(v, key, _depth + 1)
            if found:
                return (True, val)
    elif isinstance(node, list):
        for v in node:
            found, val = _recursive_key_search(v, key, _depth + 1)
            if found:
                return (True, val)
    return (False, None)


def _series_endpoints(value: Any) -> tuple[float, float] | None:
    """Extract ``(first, last)`` numeric endpoints for a trend check.

    Accepts a raw numeric list, or the provenance ``_summarize_series`` summary
    dict ``{"first":..,"last":..}``. Returns ``None`` if no usable endpoints.
    """
    if isinstance(value, dict):
        f = _coerce_number(value.get("first"))
        last = _coerce_number(value.get("last"))
        if f is not None and last is not None:
            return (f, last)
        return None
    if isinstance(value, list):
        nums = [n for n in (_coerce_number(v) for v in value) if n is not None]
        if len(nums) >= 2:
            return (nums[0], nums[-1])
        if len(nums) == 1:
            return (nums[0], nums[0])
    return None


def _result(leaf_id: str, kind: str, score: float, justification: str) -> dict[str, Any]:
    """Build the uniform per-leaf record (clamped score)."""
    return {
        "id": str(leaf_id),
        "score": max(0.0, min(1.0, float(score))),
        "justification": justification,
        "_graded": True,
        "check_kind": kind,
    }


# --------------------------------------------------------------------------- #
# the three kind-specific checkers.
# --------------------------------------------------------------------------- #
def _check_hparam(leaf_id: str, assertion: dict, run_dir: Path) -> dict[str, Any] | None:
    """``deterministic:hparam`` — compare a provenance field vs {field,op,value}."""
    field = assertion.get("field")
    op = assertion.get("op")
    if not isinstance(field, str) or not field or op not in _HPARAM_OPS:
        # malformed annotation → route to LLM (cannot interpret).
        return None
    expected = assertion.get("value")
    tol = _coerce_number(assertion.get("tolerance"))
    if tol is None:
        tol = _DEFAULT_TOLERANCE

    prov_paths = _provenance_paths(run_dir)
    if not prov_paths:
        return _result(leaf_id, CHECK_HPARAM, 0.0, f"provenance_missing:{field}")

    # Read newest-first; the first manifest that *contains* the field wins.
    found = False
    actual: Any = None
    for p in prov_paths:
        prov = _load_json(p)
        if prov is None:
            continue
        f, v = _find_provenance_field(prov, field)
        if f:
            found, actual = True, v
            break
    if not found:
        return _result(leaf_id, CHECK_HPARAM, 0.0, f"provenance_missing:{field}")

    ok = _compare(actual, op, expected, tol)
    if ok:
        return _result(
            leaf_id, CHECK_HPARAM, 1.0,
            f"provenance {field}={actual!r} satisfies {op} {expected!r}",
        )
    return _result(
        leaf_id, CHECK_HPARAM, 0.0,
        f"provenance {field}={actual!r} fails {op} {expected!r}",
    )


def _compare(actual: Any, op: str, expected: Any, tol: float) -> bool:
    """Apply one hparam operator, fail-soft (incomparable types → False)."""
    try:
        if op == "==":
            if _eq_scalar(actual, expected):
                return True
            # numeric-tolerant equality so 45 == 45.0 and "45" == 45 pass.
            an, en = _coerce_number(actual), _coerce_number(expected)
            return an is not None and en is not None and abs(an - en) <= tol
        if op == "!=":
            return not _compare(actual, "==", expected, tol)
        if op == "~=":
            an, en = _coerce_number(actual), _coerce_number(expected)
            return an is not None and en is not None and abs(an - en) <= tol
        # ordered comparisons require numbers.
        an, en = _coerce_number(actual), _coerce_number(expected)
        if an is None or en is None:
            return False
        if op == ">=":
            return an >= en - tol
        if op == "<=":
            return an <= en + tol
    except Exception:  # noqa: BLE001 — fail-soft: any comparison error → False.
        return False
    return False


def _eq_scalar(a: Any, b: Any) -> bool:
    """Exact equality with a string-insensitive fallback for scalars."""
    if a == b:
        return True
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    # one side a string representation of the other scalar.
    if isinstance(a, str) and not isinstance(b, (dict, list)):
        return a.strip().lower() == str(b).strip().lower()
    if isinstance(b, str) and not isinstance(a, (dict, list)):
        return str(a).strip().lower() == b.strip().lower()
    return False


def _check_artifact(leaf_id: str, assertion: dict, run_dir: Path) -> dict[str, Any] | None:
    """``deterministic:artifact`` — existence of a glob (or any of a list)."""
    raw = assertion.get("glob", assertion.get("globs"))
    if isinstance(raw, str):
        patterns = [raw]
    elif isinstance(raw, list):
        patterns = [p for p in raw if isinstance(p, str) and p]
    else:
        patterns = []
    if not patterns:
        return None  # malformed annotation → route to LLM.

    roots = [run_dir, run_dir / "code"]
    for pat in patterns:
        for root in roots:
            if not root.exists():
                continue
            try:
                # explicit glob first.
                if any(root.glob(pat)):
                    return _result(
                        leaf_id, CHECK_ARTIFACT, 1.0,
                        f"artifact found: {pat!r} under {root.name}/",
                    )
                # a bare filename (no separator/wildcard) → recursive search,
                # so "model.py" matches code/src/model.py without an explicit
                # rglob pattern in the rubric.
                if "/" not in pat and "*" not in pat and "?" not in pat:
                    if any(root.rglob(pat)):
                        return _result(
                            leaf_id, CHECK_ARTIFACT, 1.0,
                            f"artifact found (recursive): {pat!r} under {root.name}/",
                        )
            except Exception:  # noqa: BLE001 — a bad pattern just doesn't match.
                continue
    return _result(
        leaf_id, CHECK_ARTIFACT, 0.0,
        f"artifact_missing: none of {patterns!r} exist under run_dir or code/",
    )


def _check_numeric(leaf_id: str, assertion: dict, run_dir: Path) -> dict[str, Any] | None:
    """``deterministic:numeric`` — metric value vs {target,tolerance,direction}."""
    metric_key = assertion.get("metric_key")
    direction = assertion.get("direction")
    if not isinstance(metric_key, str) or not metric_key or direction not in _NUMERIC_DIRECTIONS:
        return None  # malformed annotation → route to LLM.
    target = _coerce_number(assertion.get("target"))
    tol = _coerce_number(assertion.get("tolerance"))
    if tol is None:
        tol = 0.0
    # trend_up/trend_down need no target; the others do.
    if direction in {"higher_better", "lower_better", "within"} and target is None:
        return None  # malformed (threshold direction with no numeric target).

    metrics = _latest_metrics(run_dir)
    if metrics is None:
        return _result(leaf_id, CHECK_NUMERIC, 0.0, f"metric_missing:{metric_key}")

    found, raw_val = _find_metric_value(metrics, metric_key)
    if not found:
        return _result(leaf_id, CHECK_NUMERIC, 0.0, f"metric_missing:{metric_key}")

    if direction in {"trend_up", "trend_down"}:
        return _grade_trend(leaf_id, metric_key, direction, raw_val)

    value = _coerce_number(raw_val)
    if value is None:
        return _result(
            leaf_id, CHECK_NUMERIC, 0.0,
            f"metric_missing:{metric_key} (non-numeric value {raw_val!r})",
        )
    return _grade_threshold(leaf_id, metric_key, direction, value, target, tol)


def _grade_threshold(
    leaf_id: str, metric_key: str, direction: str, value: float, target: float, tol: float
) -> dict[str, Any]:
    """Grade higher_better / lower_better / within against a target.

    Trend/threshold satisfaction, not magnitude — a value at-or-past the
    target (within tolerance) is 1.0; otherwise 0.0. Deterministic and simple.
    """
    if direction == "higher_better":
        ok = value >= target - tol
        rel = ">=" if ok else "<"
    elif direction == "lower_better":
        ok = value <= target + tol
        rel = "<=" if ok else ">"
    else:  # within
        ok = abs(value - target) <= tol
        rel = "≈" if ok else "≉"
    score = 1.0 if ok else 0.0
    return _result(
        leaf_id, CHECK_NUMERIC, score,
        f"metric {metric_key}={value:g} {rel} target {target:g} "
        f"(tol={tol:g}, {direction})",
    )


def _grade_trend(
    leaf_id: str, metric_key: str, direction: str, raw_val: Any
) -> dict[str, Any]:
    """Grade trend_up / trend_down on a series' first→last endpoints."""
    endpoints = _series_endpoints(raw_val)
    if endpoints is None:
        return _result(
            leaf_id, CHECK_NUMERIC, 0.0,
            f"metric_missing:{metric_key} (no usable series for {direction})",
        )
    first, last = endpoints
    if direction == "trend_up":
        ok = last >= first
        rel = "rose" if ok else "fell"
    else:  # trend_down
        ok = last <= first
        rel = "fell" if ok else "rose"
    score = 1.0 if ok else 0.0
    return _result(
        leaf_id, CHECK_NUMERIC, score,
        f"metric {metric_key} {rel} {first:g}->{last:g} ({direction})",
    )


# --------------------------------------------------------------------------- #
# public entrypoint.
# --------------------------------------------------------------------------- #
def check_leaf(leaf: dict, run_dir: Path) -> dict[str, Any] | None:
    """Deterministically grade one rubric leaf, or return ``None`` to route to the LLM.

    Returns ``None`` (→ LLM) when the leaf carries no recognized ``check_kind``
    or no usable ``assertion`` (the backwards-compat fall-through). Returns a
    uniform per-leaf record (``{"id","score","justification","_graded":True,
    "check_kind"}``) when the leaf is deterministically gradeable — including a
    graded ``0.0`` when the well-formed assertion's *evidence* is missing.

    Never raises: any unexpected error fails soft to ``None`` (route to LLM)
    so a checker bug can never break grading of an otherwise-fine rubric.
    """
    try:
        if not isinstance(leaf, dict):
            return None
        kind = leaf.get("check_kind")
        if kind not in DETERMINISTIC_CHECK_KINDS:
            return None  # no/unknown annotation → LLM (backwards-compat path).
        assertion = leaf.get("assertion")
        if not isinstance(assertion, dict) or not assertion:
            return None  # annotation present but no usable assertion → LLM.

        leaf_id = leaf.get("id", "")
        run_dir = Path(run_dir)

        if kind == CHECK_HPARAM:
            return _check_hparam(leaf_id, assertion, run_dir)
        if kind == CHECK_ARTIFACT:
            return _check_artifact(leaf_id, assertion, run_dir)
        if kind == CHECK_NUMERIC:
            return _check_numeric(leaf_id, assertion, run_dir)
        return None  # unreachable (kind ∈ set) — defensive.
    except Exception:  # noqa: BLE001 — a checker bug must never break grading.
        logger.exception(
            "deterministic_leaf_checker: unexpected error on leaf %r — routing to LLM",
            leaf.get("id") if isinstance(leaf, dict) else leaf,
        )
        return None
