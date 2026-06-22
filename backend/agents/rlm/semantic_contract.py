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

from pydantic import BaseModel, Field

SEMANTIC_CONTRACT_VERSION = 1


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


__all__ = [
    "SEMANTIC_CONTRACT_VERSION",
    "SourceSpan",
    "Provenance",
    "Dimension",
    "MetricContract",
    "AlgorithmInvariant",
    "ResourceIdentity",
    "CapabilityProfile",
    "SemanticReproductionContract",
]
