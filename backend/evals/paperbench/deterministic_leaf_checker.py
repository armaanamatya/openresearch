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
  ``{"field": <str>, "op": <op>, "value": <scalar>, "tolerance": <float?>,
     "on_missing": <"fail"|"llm">?}``
  where ``op`` ∈ ``{"==", "!=", ">=", "<=", "~="}`` (``~=`` compares with an
  absolute ``tolerance``, default ``1e-9``). ``field`` is looked up in
  ``provenance.json`` — first at the manifest top level, then inside each
  ``experiments[*]`` record (where the agent's emitter actually writes
  ``epochs`` / ``batch_size`` / ``seed`` / ``per_optimizer.*``). A dotted
  ``field`` (``"per_optimizer.adam.lr"``) traverses nested dicts; a BARE field
  (``"lr"``) additionally falls back to a bounded recursive key search *inside
  each experiment record*, so it resolves against the agent emitter's real
  ``per_optimizer: {"adam": {"lr": …}}`` shape and not only the cell route's
  flat ``lr``.

  **Any-match semantics.** A run that searched a hyperparameter grid writes one
  experiment record per candidate. The check is satisfied when **any** record's
  value satisfies the assertion (for ``!=``, when **every** record does — a
  prohibition is universal, an equality is existential). Grading "the first
  record that happens to carry the field" would fail a faithful run whose
  paper-valued cell simply is not first (the lr-search false negative).

  **The ``coefficients.*`` namespace (paper-declared algorithmic constants).**
  A ``field`` of ``"coefficients.<name>"`` addresses the section
  ``provenance.emit_provenance(..., coefficients={"beta": 10, "lambda": 0.1})``
  writes — SDAR's ``g_t = σ(β·Δ_t)`` sharpening constant, its ``λ`` distillation
  weight, a temperature, a clip ε. These are the algorithmic invariants a
  surrogate gets wrong, which is exactly why they are worth checking
  mechanically. Nothing new is needed to resolve them: the dotted traversal
  above already searches the manifest top level (where a run-global coefficient
  lives) and then each ``experiments[*]`` record (where a cell that OVERRIDES one
  — an ablation sweeping λ — puts its own), and any-match then does the right
  thing: an ablation cell carrying ``alpha=0.0`` alongside the paper cell's
  ``alpha=1.0`` satisfies an ``alpha=1.0`` assertion rather than failing it.

  Because the field is DOTTED it never reaches the bare-field recursive search,
  so ``coefficients.beta`` can never accidentally bind to Adam's
  ``per_optimizer.adam.betas``. That separation is the point: the namespace
  encodes the coefficient's ROLE (a constant the *paper* declared), and a value
  is only ever compared against what the paper declared for that leaf. NOTHING
  here range-checks a coefficient — ``0.0`` (an ablation) and ``10`` are equally
  legitimate values, and a guard that keyed on the ambiguous NAME ``alpha``
  instead of its role once hard-blocked a faithful ``alpha=0.0`` ablation
  (learn.md 2026-07-07). Do not reintroduce that.

* ``deterministic:artifact`` — ``assertion`` is ``{"glob": <str | [str]>}``
  (alias ``"globs"``). Existence is checked under ``run_dir`` **and**
  ``run_dir/code`` (recursively for a bare ``"name.py"`` pattern). Any one
  pattern matching → satisfied.

* ``deterministic:numeric`` — ``assertion`` is
  ``{"metric_key": <str>, "target": <float>, "tolerance": <float?>,
     "direction": <dir>, "on_missing": <"fail"|"llm">?}`` where ``dir`` ∈
  ``{"higher_better", "lower_better", "trend_up", "trend_down", "within"}``.
  The value is read from the freshest results-bearing ``metrics.json`` (top
  level → dotted path → recursive key search → first numeric ``metric``
  leaf). Graded on **trend / threshold satisfaction, not exact magnitude**
  (e.g. ``higher_better``: ``value >= target - tolerance`` → ``1.0``).

``on_missing`` — the false-negative valve (the load-bearing knob for auto-annotation)
--------------------------------------------------------------------------------------
``"fail"`` (**the default** — unchanged, so every hand-authored annotation and
every existing test keeps today's semantics): well-formed assertion + missing
evidence → a graded ``0.0``.

``"llm"``: missing *evidence* → ``None`` (route to LLM) instead of ``0.0``.
This exists because an **auto-generated** annotation is written at rubric-gen
time — BEFORE the run — so it can only *predict* the artifact namespace. Two
cases make the strict ``0.0`` unsound for a predicted assertion:

* ``provenance.json`` is absent. The agent's ``emit_provenance`` call is
  explicitly **fail-soft and optional** (see ``baseline_implementation``'s
  provenance block: "Wrap both calls in try/except"). A faithful run that
  merely skipped the manifest would have EVERY hyperparameter leaf zeroed —
  strictly worse than the LLM, which can read ``lr=1e-4`` straight out of
  ``train.py``.
* ``metrics.json`` exists with real measured cells, but carries no key by the
  predicted NAME (the canonical shape is ``per_model[m][env][baseline] =
  {"metric": …}``, not ``{"top1_accuracy": …}``). That is a *naming* mismatch,
  not an absence of evidence — and "no evidence" is the only thing a ``0.0``
  is entitled to assert.

A wrong-value check still fails deterministically under ``"llm"``: the valve
fires only when the value cannot be *located*, never when it is located and
misses. Fabrication is not let through either — an LLM-credited result leaf
with no on-disk cell is independently vetoed by the A7 evidence gate
(``leaf_scorer._result_leaf_substantiated``), which is precisely the layer
that owns that direction.

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

__all__ = ["check_leaf", "DETERMINISTIC_CHECK_KINDS", "COEFFICIENTS_KEY"]

# The three recognized check kinds. A leaf whose ``check_kind`` is not in this
# set falls through to the LLM (returns None).
CHECK_HPARAM = "deterministic:hparam"
CHECK_ARTIFACT = "deterministic:artifact"
CHECK_NUMERIC = "deterministic:numeric"
DETERMINISTIC_CHECK_KINDS = frozenset({CHECK_HPARAM, CHECK_ARTIFACT, CHECK_NUMERIC})

# The paper-declared-coefficient namespace, re-exported from its owner so a
# consumer of the checker never hard-codes the string. ``provenance`` is the
# single source of truth for the address (``coefficients.<name>``); importing it
# here means a rename breaks loudly at import instead of silently emitting
# assertions that resolve to nothing.
from backend.agents.rlm.provenance import COEFFICIENTS_KEY  # noqa: E402

# hparam comparison operators.
_HPARAM_OPS = frozenset({"==", "!=", ">=", "<=", "~="})
# numeric direction vocabulary.
_NUMERIC_DIRECTIONS = frozenset(
    {"higher_better", "lower_better", "trend_up", "trend_down", "within"}
)
# default absolute tolerance for ~= / numeric "within" when none supplied.
_DEFAULT_TOLERANCE = 1e-9

# ``on_missing`` vocabulary. "fail" (default) = today's semantics: a well-formed
# assertion whose evidence is absent grades 0.0. "llm" = route to the LLM instead
# — the valve an auto-generated (pre-run, therefore *predicted*) annotation uses so
# it can never false-fail a faithful run. See the module docstring.
_ON_MISSING_LLM = "llm"


def _routes_to_llm_on_missing(assertion: dict) -> bool:
    """True iff this assertion asked to fall through to the LLM on missing evidence."""
    return str(assertion.get("on_missing", "fail")).strip().lower() == _ON_MISSING_LLM


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


def _collect_provenance_values(prov: Any, field: str) -> list[Any]:
    """Collect EVERY value of ``field`` in a provenance manifest, fail-soft.

    Search (all hits, not first):
      1. manifest top level (dotted-aware) — e.g. ``run_id``.
      2. each ``experiments[*]`` record (dotted-aware) — where the agent's
         emitter writes ``epochs``/``batch_size``/``seed``/``per_optimizer.*``.
      3. per record, for a BARE (un-dotted) field only: a bounded recursive key
         search *within that record*. This is what makes ``field="lr"`` resolve
         against the agent emitter's documented shape
         ``per_optimizer: {"adam": {"lr": 1e-4}}`` — the cell route writes a flat
         top-level ``lr`` (``provenance._PROVENANCE_PARAM_KEYS``) but the agent
         route nests it, and a bare-``lr`` assertion must not be a coin flip on
         which route produced the manifest. Scoped to inside a record (never the
         whole manifest) so it cannot reach unrelated structures like
         ``lr_search.grid``.

    Returning ALL values (rather than the first) is what lets the caller apply
    any-match semantics — a hyperparameter SEARCH writes one record per candidate,
    and grading whichever record happens to be first is a false-negative machine.
    """
    if not isinstance(prov, dict):
        return []
    out: list[Any] = []

    # 1. top level.
    found, val = _dotted_get(prov, field)
    if found:
        out.append(val)

    # 2/3. inside experiments.
    exps = prov.get("experiments")
    if isinstance(exps, dict):
        records = list(exps.values())
    elif isinstance(exps, list):
        records = list(exps)
    else:
        records = []

    bare = "." not in field
    for exp in records:
        found, val = _dotted_get(exp, field)
        if found:
            out.append(val)
        elif bare:
            found, val = _recursive_key_search(exp, field)
            if found:
                out.append(val)
    return out


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

    # Missing evidence: 0.0 by default, or route-to-LLM when the annotation asked
    # for the valve (a pre-run *predicted* assertion — see the module docstring).
    def _missing() -> dict[str, Any] | None:
        if _routes_to_llm_on_missing(assertion):
            logger.debug(
                "deterministic_leaf_checker: leaf %r — provenance field %r absent, "
                "on_missing=llm → routing to LLM (not a 0.0)", leaf_id, field,
            )
            return None
        return _result(leaf_id, CHECK_HPARAM, 0.0, f"provenance_missing:{field}")

    prov_paths = _provenance_paths(run_dir)
    if not prov_paths:
        return _missing()

    # Read newest-first; the first manifest that *contains* the field wins, and
    # within it EVERY record's value for that field is a candidate.
    values: list[Any] = []
    for p in prov_paths:
        prov = _load_json(p)
        if prov is None:
            continue
        values = _collect_provenance_values(prov, field)
        if values:
            break
    if not values:
        return _missing()

    # Any-match for an existential op; all-match for the universal "!=".
    if op == "!=":
        ok = all(_compare(v, op, expected, tol) for v in values)
    else:
        ok = any(_compare(v, op, expected, tol) for v in values)

    seen = values[0] if len(values) == 1 else values
    if ok:
        return _result(
            leaf_id, CHECK_HPARAM, 1.0,
            f"provenance {field}={seen!r} satisfies {op} {expected!r}",
        )
    return _result(
        leaf_id, CHECK_HPARAM, 0.0,
        f"provenance {field}={seen!r} fails {op} {expected!r}",
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

    # Missing/unlocatable metric: 0.0 by default, or route-to-LLM under the valve.
    # NB this fires only when the value cannot be LOCATED — a located value that
    # misses its target still fails deterministically.
    def _missing(detail: str = "") -> dict[str, Any] | None:
        if _routes_to_llm_on_missing(assertion):
            logger.debug(
                "deterministic_leaf_checker: leaf %r — metric %r unresolvable%s, "
                "on_missing=llm → routing to LLM (not a 0.0)",
                leaf_id, metric_key, f" ({detail})" if detail else "",
            )
            return None
        suffix = f" ({detail})" if detail else ""
        return _result(
            leaf_id, CHECK_NUMERIC, 0.0, f"metric_missing:{metric_key}{suffix}"
        )

    metrics = _latest_metrics(run_dir)
    if metrics is None:
        return _missing()

    found, raw_val = _find_metric_value(metrics, metric_key)
    if not found:
        return _missing()

    if direction in {"trend_up", "trend_down"}:
        endpoints = _series_endpoints(raw_val)
        if endpoints is None:
            return _missing(f"no usable series for {direction}")
        return _grade_trend(leaf_id, metric_key, direction, raw_val)

    value = _coerce_number(raw_val)
    if value is None:
        return _missing(f"non-numeric value {raw_val!r}")
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
