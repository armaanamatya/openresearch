"""Unit tests for RunPlan.required_assets extraction (Phase 1b, Task 9)."""

from backend.services.runtime.run_plan import RequiredAsset, extract_required_assets
from backend.agents.rlm.semantic_contract import (
    SemanticReproductionContract, ResourceIdentity, CapabilityProfile,
)
from backend.agents.schemas import PaperClaimMap, DatasetRequirement


def test_from_contract_resource_identities_and_capabilities():
    c = SemanticReproductionContract(
        resource_identities=[ResourceIdentity(kind="dataset", identifier="alfworld"),
                             ResourceIdentity(kind="weights", identifier="Qwen/Qwen3-1.7B")],
        capability_profile=CapabilityProfile(datasets=["search-qa"], frameworks=["pytorch"],
                                             external_services=["webshop-server"]))
    got = extract_required_assets(contract=c)
    kinds = {(a.kind, a.identifier) for a in got}
    assert ("dataset", "alfworld") in kinds
    assert ("weights", "Qwen/Qwen3-1.7B") in kinds
    assert ("dataset", "search-qa") in kinds
    assert ("framework", "pytorch") in kinds
    assert ("service", "webshop-server") in kinds


def test_fallback_to_claim_map_when_no_contract():
    cm = PaperClaimMap(core_contribution="x",
                       datasets=[DatasetRequirement(name="CIFAR-10")],
                       model_architecture="ResNet-18", hardware_clues=["1x A100"])
    got = extract_required_assets(claim_map=cm)
    ids = {(a.kind, a.identifier) for a in got}
    assert ("dataset", "CIFAR-10") in ids
    assert ("weights", "ResNet-18") in ids


def test_rubric_fallback_scans_leaf_text():
    rubric = {"children": [{"requirements": "Train on the IMDB dataset with PyTorch", "weight": 1.0}]}
    got = extract_required_assets(rubric=rubric)
    assert any(a.kind == "dataset" and a.identifier.lower() == "imdb" for a in got)


def test_dedupe_and_never_raises():
    assert extract_required_assets() == []
    c = SemanticReproductionContract(
        resource_identities=[ResourceIdentity(kind="dataset", identifier="alfworld"),
                             ResourceIdentity(kind="dataset", identifier="ALFWorld")])
    got = extract_required_assets(contract=c)
    assert len(got) == 1 and isinstance(got[0], RequiredAsset)     # case-insensitive dedupe
    assert got[0].kind == "dataset" and got[0].identifier == "alfworld"  # first-seen wins
