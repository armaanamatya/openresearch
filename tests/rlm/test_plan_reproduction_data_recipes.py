"""Tests for plan_reproduction populating data_recipes (PR-λ)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.schemas import ReproductionContract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(project_dir, method_spec_text: str = "") -> MagicMock:
    """Minimal ctx mock sufficient for plan_reproduction."""
    ctx = MagicMock()
    ctx.project_dir = project_dir
    ctx.run_id = "test-run"
    # Store method_spec_text on ctx so we can use it in the LLM stub.
    ctx._test_method_spec_text = method_spec_text

    # LLM client returns a minimal valid ReproductionContract JSON.
    # Build it from field defaults so list/nested fields are correct types.
    contract_defaults = ReproductionContract().model_dump()
    contract_defaults["reproduction_definition"] = "reproduce the paper"
    contract_defaults["metrics_shape"] = []
    contract_defaults["data_recipes"] = []
    ctx.llm_client.complete.return_value = json.dumps(contract_defaults)

    ctx.remaining_s.return_value = 300.0
    ctx.gpu_plan = None
    ctx.sandbox_mode = "local"
    ctx.minimize_compute = False
    ctx.arxiv_id = None
    ctx.vram_override = None
    return ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_plan_reproduction_populates_data_recipes_for_mnist_and_cifar10(tmp_path):
    """Stub paper text containing 'MNIST' + 'CIFAR-10' → contract.data_recipes has both."""
    from backend.agents.rlm.primitives import plan_reproduction

    method_spec = {
        "description": "We train on MNIST and CIFAR-10 with a CNN.",
        "datasets": ["MNIST", "CIFAR-10"],
    }
    env_spec = {"framework": "pytorch"}

    ctx = _make_ctx(tmp_path, method_spec_text=json.dumps(method_spec))

    # Patch the primitive cache so it never returns a cached value.
    with patch("backend.agents.rlm.primitive_cache.maybe_get", return_value=None), \
         patch("backend.agents.rlm.primitive_cache.put"):
        result = plan_reproduction(method_spec, env_spec, ctx=ctx)

    # Should succeed.
    assert result.get("outcome") == "ok" or "error" not in result, result.get("error")

    recipes = result.get("data_recipes", [])
    canonical_names = {r["canonical_name"] for r in recipes}
    assert "MNIST" in canonical_names, f"Expected MNIST in data_recipes, got: {canonical_names}"
    assert "CIFAR-10" in canonical_names, f"Expected CIFAR-10 in data_recipes, got: {canonical_names}"


def test_plan_reproduction_empty_data_recipes_for_no_known_datasets(tmp_path):
    """Stub paper with no recognisable datasets → empty data_recipes (backward compat)."""
    from backend.agents.rlm.primitives import plan_reproduction

    method_spec = {
        "description": "This paper proposes a new optimizer. No well-known datasets.",
    }
    env_spec = {"framework": "jax"}

    ctx = _make_ctx(tmp_path)

    with patch("backend.agents.rlm.primitive_cache.maybe_get", return_value=None), \
         patch("backend.agents.rlm.primitive_cache.put"):
        result = plan_reproduction(method_spec, env_spec, ctx=ctx)

    recipes = result.get("data_recipes", [])
    assert isinstance(recipes, list)
    assert len(recipes) == 0, f"Expected empty data_recipes, got: {recipes}"


def test_plan_reproduction_data_recipes_contains_canonical_loader(tmp_path):
    """Each recipe in data_recipes must have canonical_loader populated."""
    from backend.agents.rlm.primitives import plan_reproduction

    method_spec = {
        "description": "The model is evaluated on the IMDB sentiment dataset.",
        "datasets": ["IMDB"],
    }
    env_spec = {"framework": "pytorch"}

    ctx = _make_ctx(tmp_path)

    with patch("backend.agents.rlm.primitive_cache.maybe_get", return_value=None), \
         patch("backend.agents.rlm.primitive_cache.put"):
        result = plan_reproduction(method_spec, env_spec, ctx=ctx)

    recipes = result.get("data_recipes", [])
    imdb = next((r for r in recipes if r.get("canonical_name") == "IMDB"), None)
    assert imdb is not None, "IMDB not found in data_recipes"
    assert imdb.get("canonical_loader"), "IMDB recipe missing canonical_loader"
    assert "stanfordnlp/imdb" in imdb["canonical_loader"]
