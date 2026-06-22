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


__all__ = [
    "SEMANTIC_CONTRACT_VERSION",
    "enabled",
    "persist",
    "load",
    "SourceSpan",
    "Provenance",
    "Dimension",
    "MetricContract",
    "AlgorithmInvariant",
    "ResourceIdentity",
    "CapabilityProfile",
    "SemanticReproductionContract",
]
