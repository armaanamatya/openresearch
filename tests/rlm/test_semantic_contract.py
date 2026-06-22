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


# --- fail-soft persistence -------------------------------------------------
def test_persist_then_load_roundtrip(tmp_path):
    from backend.agents.rlm import semantic_contract as sc

    c = SemanticReproductionContract(requirements=["reproduce Table 3"], unresolved=["seeds"])
    p = sc.persist(tmp_path, c)
    assert p is not None and p.name == "semantic_contract.json"
    loaded = sc.load(tmp_path)
    assert loaded is not None
    assert loaded.requirements == ["reproduce Table 3"]
    assert loaded.unresolved == ["seeds"]


def test_load_missing_returns_none(tmp_path):
    from backend.agents.rlm import semantic_contract as sc
    assert sc.load(tmp_path) is None  # absent → caller keeps current behaviour


def test_load_torn_state_is_failsoft(tmp_path):
    from backend.agents.rlm import semantic_contract as sc
    target = tmp_path / "rlm_state"
    target.mkdir()
    (target / "semantic_contract.json").write_text("{ this is not valid json")
    assert sc.load(tmp_path) is None  # never raises


def test_wrong_schema_version_ignored(tmp_path):
    from backend.agents.rlm import semantic_contract as sc
    target = tmp_path / "rlm_state"
    target.mkdir()
    (target / "semantic_contract.json").write_text('{"schema_version": 999, "contract": {}}')
    assert sc.load(tmp_path) is None


def test_does_not_collide_with_legacy_planner_contract_file(tmp_path):
    # The semantic contract uses a DISTINCT filename so it can't disturb the legacy
    # reproduction_contract.json the planner store owns.
    from backend.agents.rlm import semantic_contract as sc

    sc.persist(tmp_path, SemanticReproductionContract())
    assert (tmp_path / "rlm_state" / "semantic_contract.json").exists()
    assert not (tmp_path / "rlm_state" / "reproduction_contract.json").exists()
