"""Tests for backend/agents/dataset_recipes.py (PR-λ)."""

from __future__ import annotations

import pytest

from backend.agents.dataset_recipes import (
    DATASET_RECIPES,
    DatasetRecipe,
    find_recipe,
    find_recipes_in_text,
)


# ---------------------------------------------------------------------------
# find_recipe
# ---------------------------------------------------------------------------

def test_find_recipe_canonical_name():
    r = find_recipe("IMDB")
    assert r is not None
    assert r.canonical_name == "IMDB"


def test_find_recipe_canonical_name_case_insensitive():
    r = find_recipe("imdb")
    assert r is not None
    assert r.canonical_name == "IMDB"


def test_find_recipe_alias():
    r = find_recipe("imdb sentiment")
    assert r is not None
    assert r.canonical_name == "IMDB"


def test_find_recipe_alias_case_insensitive():
    r = find_recipe("IMDB-REVIEWS")
    assert r is not None
    assert r.canonical_name == "IMDB"


def test_find_recipe_nonexistent():
    r = find_recipe("nonexistent dataset")
    assert r is None


def test_find_recipe_mnist():
    r = find_recipe("mnist")
    assert r is not None
    assert r.canonical_name == "MNIST"


def test_find_recipe_cifar10():
    r = find_recipe("CIFAR-10")
    assert r is not None
    assert r.canonical_name == "CIFAR-10"


def test_find_recipe_cifar10_alias_no_dash():
    r = find_recipe("cifar10")
    assert r is not None
    assert r.canonical_name == "CIFAR-10"


# ---------------------------------------------------------------------------
# find_recipes_in_text
# ---------------------------------------------------------------------------

def test_find_recipes_in_text_mnist_and_cifar10():
    text = "We train on MNIST and CIFAR-10 in our experiments."
    found = find_recipes_in_text(text)
    names = [r.canonical_name for r in found]
    assert "MNIST" in names
    assert "CIFAR-10" in names


def test_find_recipes_in_text_imagenet_ilsvrc():
    text = "We evaluate on the ILSVRC-2010 validation set."
    found = find_recipes_in_text(text)
    names = [r.canonical_name for r in found]
    assert "ImageNet" in names


def test_find_recipes_in_text_no_datasets():
    text = "This paper proposes a new optimizer for deep learning."
    found = find_recipes_in_text(text)
    assert found == []


def test_find_recipes_in_text_no_duplicates():
    # "imdb" appears twice but should only be returned once.
    text = "We use imdb for sentiment. The imdb dataset is standard."
    found = find_recipes_in_text(text)
    imdb_hits = [r for r in found if r.canonical_name == "IMDB"]
    assert len(imdb_hits) == 1


def test_find_recipes_in_text_order_preserved():
    # DATASET_RECIPES declares MNIST before CIFAR-10; result must follow that order.
    text = "Using MNIST and CIFAR-10."
    found = find_recipes_in_text(text)
    names = [r.canonical_name for r in found]
    assert names.index("MNIST") < names.index("CIFAR-10")


# ---------------------------------------------------------------------------
# Registry invariants
# ---------------------------------------------------------------------------

def test_registry_all_have_canonical_name():
    for r in DATASET_RECIPES:
        assert r.canonical_name, f"Recipe {r!r} missing canonical_name"


def test_registry_all_have_at_least_one_alias():
    for r in DATASET_RECIPES:
        assert len(r.aliases) >= 1, (
            f"Recipe '{r.canonical_name}' has no aliases — "
            "every recipe needs at least one alias so find_recipe('...') works"
        )


def test_registry_no_shared_aliases():
    seen: dict[str, str] = {}
    for r in DATASET_RECIPES:
        for alias in r.aliases:
            key = alias.lower()
            assert key not in seen, (
                f"Alias '{alias}' claimed by both '{seen[key]}' and '{r.canonical_name}'"
            )
            seen[key] = r.canonical_name


def test_registry_no_shared_canonical_names():
    names = [r.canonical_name.lower() for r in DATASET_RECIPES]
    assert len(names) == len(set(names)), "Duplicate canonical_name entries in DATASET_RECIPES"


def test_registry_canonical_name_not_also_another_alias():
    """The canonical_name of recipe A must not appear as an alias of recipe B."""
    alias_map: dict[str, str] = {}
    for r in DATASET_RECIPES:
        for alias in r.aliases:
            alias_map[alias.lower()] = r.canonical_name
    for r in DATASET_RECIPES:
        key = r.canonical_name.lower()
        if key in alias_map:
            assert alias_map[key] == r.canonical_name, (
                f"canonical_name '{r.canonical_name}' appears as alias of '{alias_map[key]}'"
            )
