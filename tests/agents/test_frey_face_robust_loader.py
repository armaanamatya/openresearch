"""PR-ν.1 — Frey Face robust loader + fallback_mirrors visible to the agent.

Two regression guards:
1. The Frey Face primary URL must be the working github raw mirror, not the
   cs.nyu.edu mirror that 403s in many networks.
2. The agent-facing dataset table must surface `fallback_mirrors` so a 403/404
   on one mirror leaves the agent with a usable alternative URL inline in its
   guidance rather than buried in a separate registry.
"""
from __future__ import annotations

from backend.agents.dataset_recipes import find_recipe
from backend.agents.baseline_implementation import _data_recipes_binding_block


def test_frey_face_primary_loader_uses_github_raw_mirror() -> None:
    recipe = find_recipe("frey face")
    assert recipe is not None, "Frey Face must be in the registry"
    assert "raw.githubusercontent.com/y0ast" in recipe.canonical_loader, (
        "Frey Face canonical_loader must point at the working github raw mirror "
        f"(saw: {recipe.canonical_loader!r})"
    )
    # cs.nyu.edu is now a fallback, not the primary.
    assert "cs.nyu.edu" not in recipe.canonical_loader, (
        "cs.nyu.edu returns 403 in many networks — must not be the primary loader"
    )


def test_frey_face_fallback_mirrors_present_and_ordered() -> None:
    recipe = find_recipe("frey face")
    assert recipe is not None
    assert len(recipe.fallback_mirrors) >= 2, (
        "Frey Face needs both mirrors listed for resilience"
    )
    # The github mirror must be first (primary), cs.nyu.edu second (deprecated).
    assert "raw.githubusercontent.com" in recipe.fallback_mirrors[0]


def test_dataset_guidance_table_surfaces_fallback_mirrors() -> None:
    """When the agent prompt is built, fallback_mirrors must be visible — not
    just primary loader. Without this, the agent has no URL to try on failure."""
    recipes = [
        {
            "canonical_name": "Frey Face",
            "canonical_import": "import urllib.request, pickle",
            "canonical_loader": "pickle.loads(urllib.request.urlopen('https://example.com/x.pkl').read())",
            "fallback_mirrors": (
                "https://mirror-A.example.com/x.pkl",
                "https://mirror-B.example.com/x.mat",
            ),
            "notes": "primary URL flakes",
        }
    ]
    guidance = _data_recipes_binding_block(recipes)
    assert "mirror-A.example.com" in guidance, (
        "fallback_mirrors must be embedded in the agent-facing guidance"
    )
    assert "mirror-B.example.com" in guidance
    assert "fallback mirrors" in guidance.lower()


def test_dataset_guidance_loader_truncation_does_not_break_urls() -> None:
    """The agent inlines the loader verbatim — truncating mid-URL would make the
    recipe a footgun. Cap is now 200 chars (was 60), enough for long-URL loaders."""
    long_url = "https://raw.githubusercontent.com/some/long-path/to/some/dataset/file.pkl"
    loader = (
        f"pickle.loads(urllib.request.urlopen('{long_url}', timeout=60).read())"
    )
    assert len(loader) < 200, "test invariant — keep the test loader under the new cap"
    recipes = [{
        "canonical_name": "Test",
        "canonical_import": "import urllib.request, pickle",
        "canonical_loader": loader,
        "notes": "",
    }]
    guidance = _data_recipes_binding_block(recipes)
    # The full URL must survive the table render.
    assert long_url in guidance, (
        "loader URL must not be truncated mid-string in the agent table"
    )
