"""Content-addressed evidence fingerprint (Workstream A3).

The grader-fidelity remediation retires the upward-biased best-of-run ``max()``
floor and replaces it with *median-within-evidence-state*: grades are grouped
by the EVIDENCE they scored, and ``finalize_regrade`` fires only when a
GENUINELY NEW evidence state appears (rather than on a 120s mtime proxy).

``evidence_key`` is the canonical fingerprint of a run's measured evidence:

    evidence_key = sha256(canonical(metrics) + normalized(scope))

Canonicalization reuses ``leaf_scorer._compact_metrics_for_grader`` — a pure,
deterministic transform that collapses long numeric series to
``{"_series": {len, first, last, min, max}}``. That gives us two properties we
want for free:

* **Stable against churn that is not evidence growth** — re-serializing the same
  metrics in a different dict order yields the same key (sorted-keys dump).
* **Sensitive to real evidence growth** — appending one more epoch to a series
  changes its ``len`` (and usually ``last``/``min``/``max``), so the key
  changes. This is DESIRED: a longer-trained grid is genuinely new evidence the
  grader has not seen and ``finalize_regrade`` should re-score it.

The module is pure: no network, no global state, no disk I/O.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from backend.evals.paperbench.leaf_scorer import _compact_metrics_for_grader

__all__ = ["evidence_key", "normalize_scope"]


def normalize_scope(scope: Any) -> Any:
    """Return an order-independent, JSON-serializable scope descriptor.

    The scope half of the fingerprint must be stable against ordering so that
    the SAME models/environments/baselines (or the SAME rubric leaf-id set)
    declared in a different order produce the SAME key, while a genuinely
    different scope produces a different one.

    Accepts the shapes the harness actually carries:

    * ``None`` → ``None`` (scope-less grade; key depends on metrics alone).
    * ``dict`` (e.g. ``ScopeSpec``-like ``{"models": [...], "environments":
      [...], "baselines": [...], "seeds": [...]}`` or a rubric ``{"leaves":
      [...]}``) → every list/set/tuple value is sorted (as strings) and the
      dict keys are sorted by the final ``json.dumps(sort_keys=True)``.
    * ``list``/``set``/``tuple`` (e.g. a bare leaf-id set or model list) → a
      sorted list of stringified elements.

    Scalars pass through. Normalization is recursive so nested containers are
    also made order-independent.
    """
    if scope is None:
        return None
    if isinstance(scope, dict):
        # Recurse into values; keys are made order-independent by sort_keys at
        # the final dump, so we only need to normalize the values here.
        return {str(k): normalize_scope(v) for k, v in scope.items()}
    if isinstance(scope, (list, tuple, set, frozenset)):
        # Order-independent: normalize each element, then sort by a stable
        # string projection so e.g. ["b","a"] and ["a","b"] collapse together.
        normalized = [normalize_scope(v) for v in scope]
        return sorted(normalized, key=lambda x: json.dumps(x, sort_keys=True, default=str))
    # Scalar (str/int/float/bool) — pass through; json handles the rest via
    # default=str at the dump site.
    return scope


def evidence_key(metrics: dict, scope: Any | None = None) -> str:
    """Return the sha256 hex fingerprint of ``(canonical metrics, scope)``.

    ``metrics`` is a metrics.json-shaped dict (``per_model`` / ``comparison`` /
    series histories). ``scope`` is an optional scope descriptor (dict, list,
    set, or ``None``) — see :func:`normalize_scope`.

    Determinism: identical ``(metrics, scope)`` → identical key. Dict-key
    reordering → SAME key. One extra epoch appended to any series (its summary
    ``len`` changes) → DIFFERENT key. A scope change → DIFFERENT key.
    """
    # Defensive: a non-dict metrics object still fingerprints (fail-soft — the
    # caller may hand us a partial / unusual shape; we never raise).
    canonical_metrics = _compact_metrics_for_grader(metrics if isinstance(metrics, dict) else {})
    payload = {
        "metrics": canonical_metrics,
        "scope": normalize_scope(scope),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
