"""Versioned schemas for the semantic ReproductionContract (round-2 foundation).

Additive + INERT: these models DEFINE the contract shape the foundation will populate
and consume behind ``OPENRESEARCH_REPRO_CONTRACT`` (default OFF). Nothing here mutates a
score, verdict, or run — they are pure data models. Per the design
(``docs/superpowers/specs/2026-06-21-semantic-evidence-foundation-design.md``):

* Every asserted fact carries ``Provenance`` (source spans + a 0..1 confidence).
* When extraction can't ground a field it stays empty/None and its name is added to
  ``unresolved`` — consumers retain current behaviour and NEVER invent a requirement.

All models ``extra="ignore"`` so an LLM-assisted generator that emits extra keys
validates cleanly (mirrors the existing ``ReproductionContract``).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

SEMANTIC_CONTRACT_VERSION = 1
_FILENAME = "semantic_contract.json"
_TRUE = frozenset({"1", "true", "yes", "on"})


class SourceSpan(BaseModel):
    """A reference into the paper text supporting an asserted fact."""

    model_config = {"extra": "ignore"}
    section: str | None = None  # e.g. "4.2", "Table 3", "Appendix B"
    quote: str | None = None  # bounded supporting snippet
    char_start: int | None = None
    char_end: int | None = None


class Provenance(BaseModel):
    """Source spans + confidence attached to every asserted fact."""

    model_config = {"extra": "ignore"}
    sources: list[SourceSpan] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Dimension(BaseModel):
    """A typed experimental axis — generalizes SDAR's model/env/baseline triple."""

    model_config = {"extra": "ignore"}
    name: str
    kind: str = "categorical"  # categorical | ordinal | numeric | boolean
    values: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)


class MetricContract(BaseModel):
    """Declared semantics of one metric (design §Typed dimensions and MetricContract)."""

    model_config = {"extra": "ignore"}
    identifier: str
    unit: str | None = None
    direction: str | None = None  # "higher_better" | "lower_better"
    value_min: float | None = None
    value_max: float | None = None
    aggregation: str | None = None  # "mean" | "max" | "final" | ...
    split: str | None = None  # "test" | "val" | "train"
    denominator: str | None = None  # what a rate is over
    uncertainty: str | None = None  # e.g. "stddev over seeds [42,43,44]"
    provenance: Provenance = Field(default_factory=Provenance)


class AlgorithmInvariant(BaseModel):
    """A must-hold algorithmic property (e.g. SDAR gate g_t = σ(β·Δ_t))."""

    model_config = {"extra": "ignore"}
    name: str
    statement: str = ""
    must_match: str | None = None  # optional regex over code/text
    provenance: Provenance = Field(default_factory=Provenance)


class ResourceIdentity(BaseModel):
    """A pinned external resource identity (weights, dataset id, container image)."""

    model_config = {"extra": "ignore"}
    kind: str  # "weights" | "dataset" | "image" | "service"
    identifier: str  # canonical name / URI
    provenance: Provenance = Field(default_factory=Provenance)


class CapabilityProfile(BaseModel):
    """Environment / capability requirements (design §capability/environment profiles)."""

    model_config = {"extra": "ignore"}
    name: str = ""
    requires_gpu: bool | None = None
    min_vram_gb: float | None = None
    frameworks: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    external_services: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)


class SemanticReproductionContract(BaseModel):
    """Versioned, source-linked intake contract — the round-2 foundation artifact.

    Populated/consumed ONLY behind ``OPENRESEARCH_REPRO_CONTRACT``. When extraction is
    incomplete a field stays empty/None and its name is added to ``unresolved``;
    consumers keep current behaviour and never invent a requirement. Default
    construction is a fully-empty, inert contract (version-stamped).
    """

    model_config = {"extra": "ignore"}
    schema_version: int = SEMANTIC_CONTRACT_VERSION
    requirements: list[str] = Field(default_factory=list)
    variants: list[str] = Field(default_factory=list)
    dimensions: list[Dimension] = Field(default_factory=list)
    metric_contracts: list[MetricContract] = Field(default_factory=list)
    algorithm_invariants: list[AlgorithmInvariant] = Field(default_factory=list)
    resource_identities: list[ResourceIdentity] = Field(default_factory=list)
    capability_profile: CapabilityProfile = Field(default_factory=CapabilityProfile)
    unresolved: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Fail-soft persistence (default OFF; distinct file from the legacy planner
# `reproduction_contract.json` so it can never disturb that artifact). A missing
# semantic contract resolves to None → consumers keep their CURRENT behaviour
# (the implicit legacy fallback; never invent a requirement).
# ---------------------------------------------------------------------------
def enabled() -> bool:
    """True only when the round-two contract migration is explicitly enabled."""
    return os.environ.get("OPENRESEARCH_REPRO_CONTRACT", "").strip().lower() in _TRUE


def _path(project_dir: Path | str) -> Path:
    return Path(project_dir) / "rlm_state" / _FILENAME


def persist(
    project_dir: Path | str, contract: SemanticReproductionContract
) -> Path | None:
    """Atomically write a versioned semantic contract, or return None on any error."""
    try:
        target = _path(project_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SEMANTIC_CONTRACT_VERSION,
            "contract": contract.model_dump(mode="json"),
        }
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, delete=False,
            prefix=".tmp_semantic_contract_", suffix=".json",
        ) as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            temp_name = fh.name
        os.replace(temp_name, target)
        return target
    except Exception:  # noqa: BLE001 -- persistence is an optional observability layer
        return None


def load(project_dir: Path | str) -> SemanticReproductionContract | None:
    """Load a stored semantic contract when valid; torn/legacy state is ignored
    (returns None → caller keeps current behaviour)."""
    try:
        raw = json.loads(_path(project_dir).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != SEMANTIC_CONTRACT_VERSION:
            return None
        contract = raw.get("contract")
        return (
            SemanticReproductionContract.model_validate(contract)
            if isinstance(contract, dict)
            else None
        )
    except Exception:  # noqa: BLE001 -- old/torn state must not affect a run
        return None


# ---------------------------------------------------------------------------
# Input adapter — build a contract from EXISTING sources (effective scope,
# PaperHint invariants, generated rubric) WITHOUT touching their consumers.
# Duck-typed (getattr) so this module stays decoupled + inert. Anything that
# can't be grounded is named in `unresolved`; the adapter never invents a fact.
# ---------------------------------------------------------------------------
def _dataset_label(d: object) -> str:
    for attr in ("name", "dataset", "id", "slug"):
        v = getattr(d, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return str(d)


def build_contract(
    *, paper_hint: object = None, scope: object = None, rubric: object = None
) -> SemanticReproductionContract:
    """Map existing sources into a SemanticReproductionContract (additive, fail-soft).

    Reads the effective ``scope`` (model/dataset/seed axes), ``paper_hint`` invariants +
    guidance, and (presence of) a generated ``rubric``. Each source it cannot ground is
    recorded in ``unresolved`` rather than invented. Pure — mutates nothing it reads.
    """
    c = SemanticReproductionContract()
    unresolved: list[str] = []

    # effective scope -> typed dimensions (generalizes SDAR model/env/baseline + seed)
    models = list(getattr(scope, "models", None) or [])
    if models:
        c.dimensions.append(Dimension(name="model", values=[str(m) for m in models]))
        c.variants.extend(str(m) for m in models)
    else:
        unresolved.append("models")
    datasets = [
        _dataset_label(d) for d in (getattr(scope, "datasets", None) or [])
    ]
    datasets = [d for d in datasets if d]
    if datasets:
        c.dimensions.append(Dimension(name="dataset", values=datasets))
    else:
        unresolved.append("datasets")
    seeds = list(getattr(scope, "seeds", None) or [])
    if seeds:
        c.dimensions.append(
            Dimension(name="seed", kind="numeric", values=[str(s) for s in seeds])
        )
    else:
        unresolved.append("seeds")

    # PaperHint invariants -> algorithm invariants; guidance -> a requirement
    for inv in (getattr(paper_hint, "invariants", None) or []):
        mm = getattr(inv, "must_match", None) or []
        c.algorithm_invariants.append(
            AlgorithmInvariant(
                name=str(getattr(inv, "name", "") or "invariant"),
                statement=str(getattr(inv, "rationale", "") or ""),
                must_match=(str(mm[0]) if mm else None),
            )
        )
    guidance = getattr(paper_hint, "guidance", "") or ""
    if guidance.strip():
        c.requirements.append(guidance.strip()[:500])
    elif paper_hint is None:
        unresolved.append("paper_hint")

    # generated rubric -> metric contracts are NOT yet extracted (later plan task);
    # be honest about the gap rather than invent metric semantics.
    if rubric:
        unresolved.append("rubric_metric_contracts")

    c.unresolved = sorted(set(unresolved))
    return c


# ---------------------------------------------------------------------------
# MetricContract diagnostics (Phase 2) — WARNING-ONLY. Compares observed metrics
# against their declared contracts and returns advisory strings. It NEVER raises,
# rejects a run, or mutates a score/verdict (design §Rollout: "warning-only first").
# ---------------------------------------------------------------------------
def _collect_numeric(obj: object, out: dict[str, list[float]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out.setdefault(str(k), []).append(float(v))
            else:
                _collect_numeric(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numeric(v, out)


def diagnose_metrics(
    contracts: list[MetricContract], metrics: object
) -> list[str]:
    """Warning-only check of observed ``metrics`` against their ``MetricContract``s.

    Returns advisory strings (declared-but-absent, below-min, above-max). Fail-soft:
    any internal error returns an empty list. Does not reject a run or change a score.
    """
    try:
        found: dict[str, list[float]] = {}
        _collect_numeric(metrics, found)
        warnings: list[str] = []
        for mc in contracts or []:
            ident = mc.identifier
            values = found.get(ident)
            if not values:
                warnings.append(f"metric '{ident}' declared in contract but absent from metrics")
                continue
            for v in values:
                if mc.value_min is not None and v < mc.value_min:
                    warnings.append(
                        f"metric '{ident}'={v} below declared min {mc.value_min}"
                    )
                if mc.value_max is not None and v > mc.value_max:
                    warnings.append(
                        f"metric '{ident}'={v} above declared max {mc.value_max}"
                    )
        return warnings
    except Exception:  # noqa: BLE001 -- diagnostics must never break a run
        return []


__all__ = [
    "SEMANTIC_CONTRACT_VERSION",
    "enabled",
    "persist",
    "load",
    "build_contract",
    "diagnose_metrics",
    "SourceSpan",
    "Provenance",
    "Dimension",
    "MetricContract",
    "AlgorithmInvariant",
    "ResourceIdentity",
    "CapabilityProfile",
    "SemanticReproductionContract",
]
