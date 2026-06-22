"""Semantic-evidence foundation — versioned ReproductionContract schemas (Phase 1).

The schemas are additive + inert: default construction is a fully-empty version-stamped
contract; every asserted fact carries source spans + a bounded-0..1 confidence; an
`unresolved` list records ungrounded fields; extra LLM keys are ignored.
"""
import pytest
from pydantic import ValidationError

from backend.agents.rlm.semantic_contract import (
    SEMANTIC_CONTRACT_VERSION,
    CapabilityProfile,
    Dimension,
    MetricContract,
    Provenance,
    SemanticReproductionContract,
    SourceSpan,
)


def test_default_contract_is_empty_and_versioned():
    c = SemanticReproductionContract()
    assert c.schema_version == SEMANTIC_CONTRACT_VERSION == 1
    assert c.requirements == [] and c.variants == []
    assert c.dimensions == [] and c.metric_contracts == []
    assert c.algorithm_invariants == [] and c.resource_identities == []
    assert c.unresolved == []
    assert isinstance(c.capability_profile, CapabilityProfile)


def test_roundtrip_preserves_source_and_confidence():
    c = SemanticReproductionContract(
        metric_contracts=[
            MetricContract(
                identifier="success_rate", direction="higher_better",
                value_min=0.0, value_max=1.0, split="test",
                provenance=Provenance(
                    confidence=0.9,
                    sources=[SourceSpan(section="4.2", quote="success rate of 0.71")],
                ),
            )
        ],
        unresolved=["seed_count"],
    )
    raw = c.model_dump(mode="json")
    back = SemanticReproductionContract.model_validate(raw)
    mc = back.metric_contracts[0]
    assert mc.identifier == "success_rate"
    assert mc.provenance.confidence == 0.9
    assert mc.provenance.sources[0].section == "4.2"
    assert back.unresolved == ["seed_count"]


def test_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        Provenance(confidence=1.5)
    with pytest.raises(ValidationError):
        Provenance(confidence=-0.1)


def test_extra_keys_ignored_like_planner_contract():
    # An LLM-assisted generator may emit unknown keys; they must not break validation.
    c = SemanticReproductionContract.model_validate(
        {"schema_version": 1, "requirements": ["reproduce Table 3"], "llm_noise": "x"}
    )
    assert c.requirements == ["reproduce Table 3"]
    assert not hasattr(c, "llm_noise")


def test_dimension_generalizes_axes():
    d = Dimension(name="model", kind="categorical", values=["Qwen3-1.7B", "Qwen2.5-3B"])
    assert d.name == "model" and "Qwen3-1.7B" in d.values
    # an unresolved dimension is simply empty-valued + named in the contract's unresolved
    c = SemanticReproductionContract(dimensions=[d], unresolved=["baseline_set"])
    assert c.dimensions[0].kind == "categorical"
    assert "baseline_set" in c.unresolved
