"""Unit tests for backend.agents.rlm.stub_primitives.

Coverage:
- build_stub_custom_tools returns exactly 9 entries
- Each entry has {"tool": callable, "description": str} shape
- Each callable invoked with the §5 root-visible signature returns the §5-shaped type

Spec: §13 (2026-05-21-rlm-phase3-orchestrator-design.md).
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.stub_primitives import build_stub_custom_tools

# ---------------------------------------------------------------------------
# Expected primitive names (spec §5)
# ---------------------------------------------------------------------------

_EXPECTED_PRIMITIVES = {
    "understand_section",
    "extract_hyperparameters",
    "detect_environment",
    "build_environment",
    "plan_reproduction",
    "implement_baseline",
    "run_experiment",
    "verify_against_rubric",
    "propose_improvements",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_tools(make_context, tmp_path):
    """Build stub tools using the standard make_context fixture."""
    ctx = make_context(tmp_path)
    return build_stub_custom_tools(ctx)


# ---------------------------------------------------------------------------
# Tests: structure
# ---------------------------------------------------------------------------


class TestBuildStubCustomToolsStructure:
    """Tests for the structure of the returned tools dict."""

    def test_returns_exactly_nine_entries(self, stub_tools):
        """build_stub_custom_tools returns exactly 9 entries."""
        assert len(stub_tools) == 9

    def test_contains_all_expected_primitive_names(self, stub_tools):
        """All 9 spec §5 primitive names are present."""
        assert set(stub_tools.keys()) == _EXPECTED_PRIMITIVES

    def test_each_entry_has_tool_and_description_keys(self, stub_tools):
        """Each entry is {'tool': ..., 'description': ...}."""
        for name, entry in stub_tools.items():
            assert "tool" in entry, f"Missing 'tool' key for {name}"
            assert "description" in entry, f"Missing 'description' key for {name}"

    def test_each_tool_is_callable(self, stub_tools):
        """Each entry['tool'] is callable."""
        for name, entry in stub_tools.items():
            assert callable(entry["tool"]), f"entry['tool'] for {name} is not callable"

    def test_each_description_is_nonempty_string(self, stub_tools):
        """Each entry['description'] is a non-empty string."""
        for name, entry in stub_tools.items():
            assert isinstance(entry["description"], str), (
                f"description for {name} is not str"
            )
            assert entry["description"].strip(), f"description for {name} is empty"


# ---------------------------------------------------------------------------
# Tests: return shapes (spec §5)
# ---------------------------------------------------------------------------


class TestStubPrimitiveReturnShapes:
    """Each stub callable returns a §5-conformant type when called with its signature."""

    def test_understand_section_returns_dict(self, stub_tools):
        """understand_section(text_slice: str) -> dict."""
        result = stub_tools["understand_section"]["tool"]("This is a section about accuracy.")
        assert isinstance(result, dict)
        # Partial PaperClaimMap shape (T20/I7): datasets + metrics are the key fields
        assert "datasets" in result
        assert isinstance(result["datasets"], list)

    def test_extract_hyperparameters_returns_dict(self, stub_tools):
        """extract_hyperparameters(text_slice: str) -> dict."""
        result = stub_tools["extract_hyperparameters"]["tool"]("lr=0.001, batch_size=32")
        assert isinstance(result, dict)

    def test_detect_environment_returns_dict(self, stub_tools):
        """detect_environment(method_spec: dict) -> dict."""
        result = stub_tools["detect_environment"]["tool"]({"framework": "pytorch"})
        assert isinstance(result, dict)

    def test_build_environment_returns_ok_dict(self, stub_tools):
        """build_environment(env_spec: dict) -> {ok, image_tag, error, attempts}."""
        result = stub_tools["build_environment"]["tool"]({"base_image": "ubuntu:22.04"})
        assert isinstance(result, dict)
        assert "ok" in result
        assert "image_tag" in result
        assert "error" in result
        assert "attempts" in result
        assert result["ok"] is True
        assert result["image_tag"] == "stub:latest"
        assert result["error"] == ""
        assert result["attempts"] == 1

    def test_plan_reproduction_returns_dict(self, stub_tools):
        """plan_reproduction(method_spec: dict, env_spec: dict) -> dict."""
        result = stub_tools["plan_reproduction"]["tool"](
            {"method": "sgd"}, {"python_version": "3.10"}
        )
        assert isinstance(result, dict)
        # ReproductionContract shape (T20/I7): expected_outputs is the key list field
        assert "expected_outputs" in result
        assert isinstance(result["expected_outputs"], list)

    def test_implement_baseline_returns_str(self, stub_tools):
        """implement_baseline(plan: dict) -> str (code-dir path)."""
        result = stub_tools["implement_baseline"]["tool"]({"steps": []})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_run_experiment_returns_success_dict(self, stub_tools):
        """run_experiment(code_path: str, env_id: str) -> {success, metrics, logs}."""
        result = stub_tools["run_experiment"]["tool"]("/code/train.py", "stub:latest")
        assert isinstance(result, dict)
        assert "success" in result
        assert "metrics" in result
        assert "logs" in result
        assert result["success"] is True
        assert isinstance(result["metrics"], dict)

    def test_verify_against_rubric_returns_dict(self, stub_tools):
        """verify_against_rubric(results: dict, rubric: dict) -> dict (RubricVerification)."""
        result = stub_tools["verify_against_rubric"]["tool"](
            {"accuracy": 0.88}, {"areas": [{"name": "accuracy", "weight": 1.0}]}
        )
        assert isinstance(result, dict)
        assert "overall_score" in result
        assert "meets_target" in result
        # T20/I7: real primitive returns rubric_source, leaf_count, degraded (not "verdict")
        assert "rubric_source" in result
        assert "leaf_count" in result

    def test_propose_improvements_returns_list(self, stub_tools):
        """propose_improvements(current_results, rubric_scores, k=None) -> list[dict]."""
        result = stub_tools["propose_improvements"]["tool"](
            {"accuracy": 0.88},
            {"overall_score": 0.5},
        )
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)

    def test_propose_improvements_respects_k(self, stub_tools):
        """propose_improvements with k=1 returns at most 1 candidate."""
        result = stub_tools["propose_improvements"]["tool"](
            {"accuracy": 0.88},
            {"overall_score": 0.5},
            k=1,
        )
        assert isinstance(result, list)
        assert len(result) <= 1

    def test_propose_improvements_k_none(self, stub_tools):
        """propose_improvements with k=None returns the full stub list."""
        result = stub_tools["propose_improvements"]["tool"](
            {},
            {},
            k=None,
        )
        assert isinstance(result, list)
        # The stub provides 2 entries by default
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Tests: determinism
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# T20: guard tests — stub keys must subset real schema fields (review I7)
# ---------------------------------------------------------------------------


def test_stub_keys_are_a_subset_of_real_schema(make_context, tmp_path):
    """Symptom: a stub run trained the root's REPL code against keys the real chain never produces.

    Stubs returned ad-hoc dicts not derived from the schemas (review I7 / T20).
    Verify: every stub primitive's return-key set is a subset of (or equals)
    the real schema's fields.
    """
    from backend.agents.schemas import (
        EnvironmentSpec,
        ReproductionContract,
        TrainingRecipe,
    )
    from backend.agents.rlm.stub_primitives import build_stub_custom_tools

    ctx = make_context(tmp_path)
    stubs = build_stub_custom_tools(ctx)

    # Map stub-name → expected schema fields it must subset.
    # For dict-shaped returns (run_experiment, build_environment), enumerate
    # the documented keys explicitly.
    schema_fields_for = {
        "detect_environment": set(EnvironmentSpec.model_fields),
        "plan_reproduction": set(ReproductionContract.model_fields),
        "extract_hyperparameters": set(TrainingRecipe.model_fields),
        "understand_section": {"datasets", "metrics", "training_recipe",
                               "hardware_clues", "ambiguities"},  # partial PaperClaimMap
        "build_environment": {"ok", "image_tag", "error", "attempts"},
        "run_experiment": {"success", "metrics", "logs"},
    }

    sample_args = {
        "understand_section": ("Adam, lr 3e-4, batch 64, CartPole-v1.",),
        "extract_hyperparameters": ("Adam optimizer, batch size 64.",),
        "detect_environment": ({"core_contribution": "A PyTorch agent."},),
        "build_environment": ({"dockerfile": "FROM python:3.11-slim\n"},),
        "plan_reproduction": ({"core_contribution": "x"}, {"framework": "pytorch"}),
        "run_experiment": ("/tmp/code", "reprolab/test:env-x"),
    }

    for name, expected_fields in schema_fields_for.items():
        result = stubs[name]["tool"](*sample_args[name])
        assert isinstance(result, dict), f"{name} stub must return a dict, got {type(result).__name__}"
        drift = set(result.keys()) - expected_fields
        assert not drift, f"stub {name} drift keys: {drift}"


def test_propose_improvements_stub_items_match_ImprovementHypothesis(make_context, tmp_path):
    """Symptom: stub propose_improvements returned items with zero overlap with the real schema.

    Review I7 / T20 — each stub hypothesis must have ImprovementHypothesis's fields.
    """
    from backend.agents.schemas import ImprovementHypothesis
    from backend.agents.rlm.stub_primitives import build_stub_custom_tools

    ctx = make_context(tmp_path)
    stubs = build_stub_custom_tools(ctx)

    out = stubs["propose_improvements"]["tool"]({"success": True}, {})
    assert isinstance(out, list)
    assert out, "stub must return at least one hypothesis"
    for item in out:
        drift = set(item.keys()) - set(ImprovementHypothesis.model_fields)
        assert not drift, f"propose_improvements stub drift keys: {drift}"


class TestStubPrimitiveDeterminism:
    """Stubs return identical output across repeated calls (no randomness)."""

    def test_build_environment_is_deterministic(self, stub_tools):
        tool = stub_tools["build_environment"]["tool"]
        r1 = tool({"base_image": "ubuntu:22.04"})
        r2 = tool({"base_image": "ubuntu:22.04"})
        assert r1 == r2

    def test_run_experiment_is_deterministic(self, stub_tools):
        tool = stub_tools["run_experiment"]["tool"]
        r1 = tool("/code/train.py", "stub:latest")
        r2 = tool("/code/train.py", "stub:latest")
        assert r1 == r2
