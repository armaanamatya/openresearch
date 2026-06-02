"""Structured exclusion records — the dynamic, *verified* rubric-exclusion contract.

The fairness principle (2026-06-01, the user's directive): never dock the rubric
for an experiment the harness could not run because of an out-of-our-control
limit (VRAM, OOM after shrink-retry, a dead dataset endpoint, an un-installable
environment) OR because the OPERATOR deliberately scoped it out (e.g. the
"smallest-two" cost-bounded scope). Such an ``(axis, item)`` is **excluded** from
the rubric — removed from BOTH numerator and denominator — instead of scored 0.

Anti-gaming, the load-bearing rule: an exclusion is honoured by the scorer ONLY
when ``verified is True`` — i.e. the harness itself produced the record, either
from a *measured* limit (``capacity_vram`` / ``dataset_dead`` /
``oom_shrink_exhausted`` / ``env_setup_failed``) or from the operator's
``ScopeSpec`` (``operator_scope``). An exclusion the AGENT merely *declared*
(``verified=False``) is NOT excluded; its leaves stay in scoring. This stops a
broad ``except Exception`` block from laundering a real code bug into a free
scope reduction — the root cause of the degenerate 0.188 SDAR score (see
``leaf_scorer._detect_data_unavailable_leaves``, which already gates *models*
this way via ``operator_skip_models``; this module generalises the pattern to
every axis and gives it a structured, serialisable shape).

Two distinct numbers fall out of one exclusion set:

* the **strict** ``overall_score`` excludes only ``verified`` exclusions — the
  official, anti-gaming-safe number; and
* an informational ``compute_adjusted_score`` may *additionally* fold in any
  exclusion the operator chose, answering "what would we have scored on the
  scope we actually attempted?".

This module is the single source of truth for the record shape and for the pure
classification helpers. It is **stdlib-only** (``dataclasses`` / ``enum`` /
``typing``), so — exactly like ``gpu_cell_runner`` and ``cell_matrix`` — it is
copy-pasteable into an agent sandbox and importable from every backend layer
without dragging a dependency in. No ``RunContext``, no network, no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

__all__ = [
    "AXIS_ENVIRONMENT",
    "AXIS_MODEL",
    "AXIS_DATASET",
    "AXIS_BASELINE",
    "VALID_AXES",
    "KIND_CAPACITY_VRAM",
    "KIND_DATASET_DEAD",
    "KIND_OOM_SHRINK_EXHAUSTED",
    "KIND_ENV_SETUP_FAILED",
    "KIND_OPERATOR_SCOPE",
    "HARD_LIMIT_KINDS",
    "VALID_KINDS",
    "Exclusion",
    "operator_scope_exclusions",
    "verified_only",
    "verified_items_by_axis",
    "build_scope_block",
    "exclusions_from_gaps",
]

# --- Axes an exclusion can target ------------------------------------------
AXIS_ENVIRONMENT = "environment"
AXIS_MODEL = "model"
AXIS_DATASET = "dataset"
AXIS_BASELINE = "baseline"
VALID_AXES: frozenset[str] = frozenset(
    {AXIS_ENVIRONMENT, AXIS_MODEL, AXIS_DATASET, AXIS_BASELINE}
)

# --- Exclusion kinds --------------------------------------------------------
# The first four are HARNESS-MEASURED hard limits. ``operator_scope`` is an
# operator decision the harness can confirm against the ``ScopeSpec`` (its
# ``--scope-spec`` file is the evidence). All five are ``verified=True`` when the
# harness mints them; ``verified=False`` is reserved for an agent-declared skip
# the harness cannot corroborate.
KIND_CAPACITY_VRAM = "capacity_vram"                 # est_vram × headroom > per-GPU budget
KIND_DATASET_DEAD = "dataset_dead"                   # dataset_url confirmed 404/410
KIND_OOM_SHRINK_EXHAUSTED = "oom_shrink_exhausted"   # cell OOM'd after every shrink retry
KIND_ENV_SETUP_FAILED = "env_setup_failed"           # environment could not be installed/started
KIND_OPERATOR_SCOPE = "operator_scope"               # operator's ScopeSpec excluded it (e.g. smallest-two)

# Kinds that represent a *measured* hard limit (vs an operator decision). Lets a
# caller separate "we physically could not" from "the operator chose not to"
# when reporting the two numbers above.
HARD_LIMIT_KINDS: frozenset[str] = frozenset(
    {
        KIND_CAPACITY_VRAM,
        KIND_DATASET_DEAD,
        KIND_OOM_SHRINK_EXHAUSTED,
        KIND_ENV_SETUP_FAILED,
    }
)
VALID_KINDS: frozenset[str] = HARD_LIMIT_KINDS | {KIND_OPERATOR_SCOPE}

# Truncation bound for the free-text evidence copied into a record — a raw OOM
# traceback is multi-KB and the scorer only needs the leading signature.
_MAX_EVIDENCE_CHARS: int = 500


def _clean_str(value: Any) -> str:
    """Coerce to a stripped ``str`` (never raise); non-str/empty → ``""``."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return str(value).strip()
    except Exception:  # noqa: BLE001 — a __str__ that throws must not crash scoring
        return ""


@dataclass(frozen=True)
class Exclusion:
    """One ``(axis, item)`` removed from rubric scoring, with provenance.

    Attributes:
        item:     The axis value being excluded — ``"ALFWorld"``, ``"WebShop"``,
                  ``"Qwen2.5-7B"``, ``"Search-QA"``, … . Matched against rubric
                  leaf *requirement text* (token-superset), never a hash id.
        axis:     One of :data:`VALID_AXES`.
        kind:     One of :data:`VALID_KINDS`.
        reason:   Human-readable one-liner (shown in the report / UI).
        verified: ``True`` iff the HARNESS produced this record (a measured limit
                  or the operator's ScopeSpec). ONLY verified exclusions are
                  removed from the strict ``overall_score`` — the anti-gaming
                  boundary. An agent-declared, un-corroborated skip is
                  ``verified=False`` and stays in scoring.
        evidence: Short proof — the capacity arithmetic, the dead url, the OOM
                  signature, or the scope-spec path. Truncated defensively.

    Frozen + validated at construction so a malformed record fails fast at the
    producer rather than silently mis-scoring downstream.
    """

    item: str
    axis: str
    kind: str
    reason: str
    verified: bool
    evidence: str = ""

    def __post_init__(self) -> None:
        # Normalise via object.__setattr__ (frozen dataclass) so callers may pass
        # loosely-typed values and still get a clean, validated record.
        object.__setattr__(self, "item", _clean_str(self.item))
        object.__setattr__(self, "axis", _clean_str(self.axis).lower())
        object.__setattr__(self, "kind", _clean_str(self.kind).lower())
        object.__setattr__(self, "reason", _clean_str(self.reason))
        object.__setattr__(self, "verified", bool(self.verified))
        object.__setattr__(self, "evidence", _clean_str(self.evidence)[:_MAX_EVIDENCE_CHARS])

        if not self.item:
            raise ValueError("Exclusion.item must be a non-empty string")
        if self.axis not in VALID_AXES:
            raise ValueError(
                f"Exclusion.axis {self.axis!r} not in {sorted(VALID_AXES)}"
            )
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"Exclusion.kind {self.kind!r} not in {sorted(VALID_KINDS)}"
            )

    @property
    def is_hard_limit(self) -> bool:
        """``True`` for a harness-measured limit (not an operator decision)."""
        return self.kind in HARD_LIMIT_KINDS

    def to_gap(self) -> dict[str, Any]:
        """Serialise to the gap dict shape the leaf scorer already consumes.

        The scorer reads ``item`` (and falls back to ``name`` / ``id``); we also
        carry ``axis`` / ``kind`` / ``verified`` / ``evidence`` so the scorer's
        anti-gaming gate and the report/UI can reason about provenance without a
        second lookup. Plain, JSON-serialisable.
        """
        return {
            "item": self.item,
            "axis": self.axis,
            "kind": self.kind,
            "reason": self.reason,
            "verified": self.verified,
            "evidence": self.evidence,
        }

    @classmethod
    def from_gap(cls, gap: dict[str, Any]) -> "Exclusion | None":
        """Best-effort inverse of :meth:`to_gap`; ``None`` if unrecoverable.

        Tolerates the legacy gap shapes already on disk (``{"item","reason",
        "kind"}`` from ``capacity_gate`` / ``dataset_url_preflight``) by mapping
        their ``kind`` (``"capacity"`` / ``"dataset_unavailable"``) onto this
        module's vocabulary and defaulting ``verified=True`` (those gaps are
        harness-produced) with a best-guess ``axis``.
        """
        if not isinstance(gap, dict):
            return None
        item = _clean_str(gap.get("item") or gap.get("name") or gap.get("id"))
        if not item:
            return None
        raw_kind = _clean_str(gap.get("kind")).lower()
        kind = _LEGACY_KIND_MAP.get(raw_kind, raw_kind)
        if kind not in VALID_KINDS:
            return None
        axis = _clean_str(gap.get("axis")).lower()
        if axis not in VALID_AXES:
            axis = _DEFAULT_AXIS_FOR_KIND.get(kind, AXIS_ENVIRONMENT)
        verified = bool(gap["verified"]) if "verified" in gap else True
        try:
            return cls(
                item=item,
                axis=axis,
                kind=kind,
                reason=_clean_str(gap.get("reason")),
                verified=verified,
                evidence=_clean_str(gap.get("evidence")),
            )
        except ValueError:
            return None


# Legacy ``kind`` strings emitted by cell_matrix's capacity_gate /
# dataset_url_preflight, mapped onto this module's vocabulary so already-on-disk
# metrics.json gaps round-trip through :meth:`Exclusion.from_gap`.
_LEGACY_KIND_MAP: dict[str, str] = {
    "capacity": KIND_CAPACITY_VRAM,
    "dataset_unavailable": KIND_DATASET_DEAD,
}
_DEFAULT_AXIS_FOR_KIND: dict[str, str] = {
    KIND_CAPACITY_VRAM: AXIS_MODEL,
    KIND_DATASET_DEAD: AXIS_ENVIRONMENT,
    KIND_OOM_SHRINK_EXHAUSTED: AXIS_ENVIRONMENT,
    KIND_ENV_SETUP_FAILED: AXIS_ENVIRONMENT,
    KIND_OPERATOR_SCOPE: AXIS_ENVIRONMENT,
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def operator_scope_exclusions(
    full_items: Iterable[str],
    active_items: Iterable[str],
    axis: str,
    *,
    evidence: str = "",
    reason_template: str = "{item} not in operator scope ({axis} de-scoped, e.g. smallest-two)",
) -> list[Exclusion]:
    """Build ``operator_scope`` exclusions for ``full_items - active_items``.

    These are ``verified=True``: the operator's ``ScopeSpec`` (``--scope-spec``)
    is the evidence that the de-scope was a deliberate human decision, not an
    agent laundering a failure. ``full_items`` is the paper's complete axis (all
    3 SDAR environments, say); ``active_items`` is what this run actually
    attempted (just ``Search-QA``). Case-insensitive set difference; order of the
    result is stable (first-seen in ``full_items``).

    Returns ``[]`` when ``axis`` is invalid or nothing is de-scoped, so a caller
    can always splice the result unconditionally.
    """
    axis = _clean_str(axis).lower()
    if axis not in VALID_AXES:
        return []
    active_norm = {_clean_str(a).lower() for a in active_items if _clean_str(a)}
    out: list[Exclusion] = []
    seen: set[str] = set()
    for raw in full_items:
        item = _clean_str(raw)
        if not item:
            continue
        key = item.lower()
        if key in active_norm or key in seen:
            continue
        seen.add(key)
        out.append(
            Exclusion(
                item=item,
                axis=axis,
                kind=KIND_OPERATOR_SCOPE,
                reason=reason_template.format(item=item, axis=axis),
                verified=True,
                evidence=evidence,
            )
        )
    return out


def verified_only(exclusions: Iterable[Exclusion]) -> list[Exclusion]:
    """Return only the ``verified`` exclusions — the strict-score input."""
    return [e for e in exclusions if isinstance(e, Exclusion) and e.verified]


def verified_items_by_axis(
    exclusions: Iterable[Exclusion],
) -> dict[str, set[str]]:
    """Map ``axis -> {item, ...}`` over the VERIFIED exclusions only.

    The leaf scorer uses this to decide which axis values are eligible for
    exclusion before it does requirement-text matching. Items are kept in their
    original case (the scorer tokenises case-insensitively).
    """
    out: dict[str, set[str]] = {}
    for e in verified_only(exclusions):
        out.setdefault(e.axis, set()).add(e.item)
    return out


def build_scope_block(
    exclusions: Iterable[Exclusion],
    *,
    models_run: Iterable[str] | None = None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fold exclusions into the canonical ``metrics.json::scope`` block.

    Produces the structured ``exclusions`` list (the new authoritative surface)
    AND derives the legacy ``environments_skipped`` / ``models_skipped`` /
    ``gaps`` lists from the VERIFIED exclusions so the existing leaf-scorer
    signals (which read those keys) keep working unchanged. Only verified
    exclusions populate the legacy skip lists — an unverified, agent-declared
    skip is recorded in ``exclusions`` (for transparency) but is NOT spliced into
    a skip list, so it stays in scoring.

    ``existing`` (an already-built scope dict, e.g. from
    ``cell_matrix.aggregate_cell_metrics``) is merged into, not replaced:
    ``models_run`` and any pre-existing skip entries / gaps are unioned with the
    derived ones. This lets the caller compose this with the capacity/dataset
    gates without losing their output.
    """
    excl = [e for e in exclusions if isinstance(e, Exclusion)]
    base: dict[str, Any] = dict(existing or {})

    def _merge_sorted(key: str, extra: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        for v in list(base.get(key) or []):
            if isinstance(v, str) and v:
                seen.add(v)
        for v in extra:
            if isinstance(v, str) and v:
                seen.add(v)
        return sorted(seen)

    ver = verified_only(excl)
    env_skips = [e.item for e in ver if e.axis == AXIS_ENVIRONMENT]
    model_skips = [e.item for e in ver if e.axis == AXIS_MODEL]

    # Gaps: keep any existing gap dicts, then append one per verified exclusion
    # (deduped by (item, kind)). The scorer reads gap["item"].
    gaps: list[dict[str, Any]] = [
        g for g in (base.get("gaps") or []) if isinstance(g, dict)
    ]
    have = {(str(g.get("item", "")), str(g.get("kind", ""))) for g in gaps}
    for e in ver:
        key = (e.item, e.kind)
        if key not in have:
            gaps.append(e.to_gap())
            have.add(key)

    out = dict(base)
    if models_run is not None:
        out["models_run"] = _merge_sorted("models_run", models_run)
    elif "models_run" in base:
        out["models_run"] = _merge_sorted("models_run", [])
    out["models_skipped"] = _merge_sorted("models_skipped", model_skips)
    out["environments_skipped"] = _merge_sorted("environments_skipped", env_skips)
    out["gaps"] = gaps
    # Full structured record (verified AND unverified) for the report/UI + the
    # scorer's verified-gate. Deduped by (axis, item, kind).
    seen_excl: set[tuple[str, str, str]] = set()
    excl_dicts: list[dict[str, Any]] = []
    for e in excl:
        k = (e.axis, e.item, e.kind)
        if k in seen_excl:
            continue
        seen_excl.add(k)
        excl_dicts.append(e.to_gap())
    out["exclusions"] = excl_dicts
    return out


def exclusions_from_gaps(gaps: Iterable[Any]) -> list[Exclusion]:
    """Recover :class:`Exclusion` records from a mixed gaps list (best-effort).

    Tolerates the legacy on-disk shapes; non-dict / unrecoverable entries are
    dropped. Lets a consumer that only has ``metrics.json`` on disk (e.g. the
    leaf scorer reading an older run) reconstruct the verified-exclusion set.
    """
    out: list[Exclusion] = []
    for g in gaps or []:
        e = Exclusion.from_gap(g) if isinstance(g, dict) else None
        if e is not None:
            out.append(e)
    return out
