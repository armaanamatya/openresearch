"""Tests for backend.agents.rlm.skill_matcher (Release-1 paper -> skill matcher).

The primary cases run against the REAL vendored skill catalog (loaded via
skill_catalog.load_catalog(), no mocks). A handful of tie-break/pairing unit
tests build a tiny synthetic catalog directly out of real SkillMeta dataclass
instances (plain data, not mocks) so the scoring/ordering contract can be
pinned down independent of the shipped library's exact contents.
"""
from __future__ import annotations

from pathlib import Path

from backend.agents.rlm.skill_catalog import SkillMeta, load_catalog
from backend.agents.rlm.skill_matcher import SkillMatch, match_skills

_DUMMY_PATH = Path("/dev/null")


def _meta(name: str, tags: tuple[str, ...] = (), category: str = "test") -> SkillMeta:
    return SkillMeta(
        name=name, description="", category=category, tags=tags, path=_DUMMY_PATH
    )


# ---------------------------------------------------------------------------
# Real catalog
# ---------------------------------------------------------------------------

def test_match_skills_sdar_like_claim_map_is_ml_rl():
    catalog = load_catalog()
    claim_map = {
        "core_contribution": "GRPO reinforcement learning for LLM agents",
        "model_architecture": "Qwen",
        "datasets": ["ALFWorld"],
        "training_recipe": {"optimizer": "GRPO"},
    }
    environment_spec = {"framework": "pytorch"}

    result = match_skills(claim_map, environment_spec, catalog)

    assert result.domain == "ml-rl"
    assert any(
        name in result.skill_names for name in ("verl-rl-training", "grpo-rl-training")
    )


def test_match_skills_vision_flavored_claim_map_different_domain():
    catalog = load_catalog()
    claim_map = {
        "core_contribution": (
            "A vision transformer for image classification and object detection"
        ),
        "model_architecture": "ResNet",
        "datasets": ["ImageNet"],
    }
    environment_spec = {"framework": "pytorch"}

    result = match_skills(claim_map, environment_spec, catalog)

    assert result.domain != "ml-rl"
    assert result.domain == "ml-vision"


def test_match_skills_empty_dicts_returns_empty_match():
    catalog = load_catalog()
    result = match_skills({}, {}, catalog)
    assert result == SkillMatch(domain="ml-other", skill_names=(), reasons=())


def test_match_skills_real_catalog_reasons_paired_with_skill_names():
    catalog = load_catalog()
    claim_map = {
        "core_contribution": "GRPO reinforcement learning for LLM agents",
        "training_recipe": {"optimizer": "GRPO"},
    }
    result = match_skills(claim_map, {"framework": "pytorch"}, catalog)

    assert len(result.skill_names) == len(result.reasons)
    assert len(result.skill_names) > 0


def test_match_skills_real_catalog_top_k_limits_results():
    catalog = load_catalog()
    claim_map = {
        "core_contribution": "GRPO reinforcement learning for LLM agents",
        "training_recipe": {"optimizer": "GRPO"},
    }
    result = match_skills(claim_map, {"framework": "pytorch"}, catalog, top_k=2)

    assert len(result.skill_names) <= 2


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------

def test_match_skills_empty_catalog_returns_no_skills_but_domain_still_classified():
    claim_map = {"core_contribution": "GRPO reinforcement learning agent"}
    result = match_skills(claim_map, {}, {})
    assert result.domain == "ml-rl"
    assert result.skill_names == ()
    assert result.reasons == ()


def test_match_skills_none_like_missing_fields_never_raises():
    # Fields entirely absent (not just empty) from both mappings.
    result = match_skills({"unrelated_field": "x"}, {"other_field": "y"}, {})
    assert result == SkillMatch(domain="ml-other", skill_names=(), reasons=())


def test_match_skills_non_string_scalars_in_claim_map_do_not_raise():
    catalog = {"some-skill": _meta("some-skill", tags=("42",))}
    claim_map = {
        "core_contribution": 42,
        "claims": [{"metric": 3.14}, "text claim"],
        "hardware_clues": None,
    }
    result = match_skills(claim_map, {"framework": None}, catalog)
    # Must not raise; "42" tokenizes out of the int scalar and overlaps the
    # synthetic skill's tag.
    assert isinstance(result, SkillMatch)


# ---------------------------------------------------------------------------
# Domain dominance boundary (documents the "strictly greater" judgment call)
# ---------------------------------------------------------------------------

def test_match_skills_pure_physics_claim_map_is_physics_domain():
    """No ML keyword competes at all -> a lone non-ML signal wins outright."""
    claim_map = {"core_contribution": "A study of quantum particle physics"}
    result = match_skills(claim_map, {}, {})
    assert result.domain == "physics"


def test_match_skills_tied_ml_and_non_ml_signal_favors_ml():
    """Exactly one ml-rl keyword ("reward") and one chemistry keyword
    ("chemistry") tie 1-1 -- dominance requires a STRICT non-ML majority, so
    a tie stays with the ML classification (this system reproduces ML papers
    almost exclusively)."""
    claim_map = {
        "core_contribution": "reward function analysis in a chemistry lab setting"
    }
    result = match_skills(claim_map, {}, {})
    assert result.domain == "ml-rl"


def test_match_skills_no_keyword_signal_is_ml_other():
    claim_map = {"core_contribution": "a completely generic study of things"}
    result = match_skills(claim_map, {}, {})
    assert result.domain == "ml-other"


# ---------------------------------------------------------------------------
# Synthetic-catalog unit tests (real SkillMeta instances, not mocks) --
# pin down scoring/tie-break determinism independent of the shipped library.
# ---------------------------------------------------------------------------

def test_match_skills_tie_break_by_name_deterministic():
    catalog = {
        "zeta-skill": _meta("zeta-skill", tags=("grpo",)),
        "alpha-skill": _meta("alpha-skill", tags=("grpo",)),
    }
    claim_map = {"core_contribution": "grpo training"}
    result = match_skills(claim_map, {}, catalog)

    assert result.skill_names == ("alpha-skill", "zeta-skill")


def test_match_skills_higher_overlap_ranks_first():
    catalog = {
        "one-match": _meta("one-match", tags=("grpo",)),
        "three-match": _meta("three-match", tags=("grpo", "reward", "policy")),
    }
    claim_map = {"core_contribution": "grpo reward policy optimization"}
    result = match_skills(claim_map, {}, catalog)

    assert result.skill_names[0] == "three-match"


def test_match_skills_zero_overlap_skill_excluded():
    catalog = {
        "matching-skill": _meta("matching-skill", tags=("grpo",)),
        "unrelated-skill": _meta("unrelated-skill", tags=("gardening", "cooking")),
    }
    claim_map = {"core_contribution": "grpo training"}
    result = match_skills(claim_map, {}, catalog)

    assert result.skill_names == ("matching-skill",)


def test_match_skills_reasons_name_top_matched_tokens():
    catalog = {"rl-skill": _meta("rl-skill", tags=("grpo", "reward", "policy"))}
    claim_map = {"core_contribution": "grpo reward policy gradient method"}
    result = match_skills(claim_map, {}, catalog)

    assert result.skill_names == ("rl-skill",)
    assert "grpo" in result.reasons[0]
