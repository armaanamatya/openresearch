"""Tests for _data_recipes_binding_block (PR-λ)."""

from __future__ import annotations


from backend.agents.baseline_implementation import _data_recipes_binding_block


def _make_recipe(canonical_name: str, canonical_import: str, canonical_loader: str,
                 normalization_stats: dict | None = None) -> dict:
    return {
        "canonical_name": canonical_name,
        "aliases": (canonical_name.lower(),),
        "canonical_import": canonical_import,
        "canonical_loader": canonical_loader,
        "fallback_mirrors": (),
        "normalization_stats": normalization_stats,
        "license_note": "",
        "notes": "",
    }


def test_data_recipes_binding_block_contains_both_loaders():
    recipes = [
        _make_recipe(
            "IMDB",
            "from datasets import load_dataset",
            "load_dataset('stanfordnlp/imdb')",
        ),
        _make_recipe(
            "MNIST",
            "from torchvision import datasets, transforms",
            "datasets.MNIST(root, train=True, download=True, transform=transforms.ToTensor())",
        ),
    ]
    block = _data_recipes_binding_block(recipes)
    assert block, "Expected non-empty block for 2 recipes"
    assert "stanfordnlp/imdb" in block
    assert "MNIST" in block
    assert "IMDB" in block


def test_data_recipes_binding_block_empty_list():
    block = _data_recipes_binding_block([])
    assert block == "", f"Expected empty string for empty list, got: {block!r}"


def test_data_recipes_binding_block_none_recipes():
    # Defensive: None should also return empty string.
    block = _data_recipes_binding_block(None)
    assert block == ""


def test_data_recipes_binding_block_includes_normalization_stats():
    recipes = [
        _make_recipe(
            "CIFAR-10",
            "from torchvision import datasets, transforms",
            "datasets.CIFAR10(root, ...)",
            normalization_stats={"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]},
        ),
    ]
    block = _data_recipes_binding_block(recipes)
    # Normalization stats should be present in the block.
    assert "0.4914" in block or "normalization" in block.lower() or "mean" in block.lower(), (
        "Expected normalization stats or guidance in block"
    )


def test_data_recipes_binding_block_license_note_surfaced():
    recipes = [
        _make_recipe("ImageNet", "from torchvision import datasets",
                     "datasets.ImageNet(root, split='train', download=False)"),
    ]
    # Manually add license_note.
    recipes[0] = dict(recipes[0])
    recipes[0]["license_note"] = "ImageNet requires registration."

    block = _data_recipes_binding_block(recipes)
    assert "ImageNet" in block
    # The block should mention the license note.
    assert "registration" in block or "license" in block.lower() or "license_note" in block


def test_data_recipes_binding_block_single_recipe():
    recipes = [
        _make_recipe(
            "SQuAD",
            "from datasets import load_dataset",
            "load_dataset('rajpurkar/squad')",
        ),
    ]
    block = _data_recipes_binding_block(recipes)
    assert "rajpurkar/squad" in block
    assert "SQuAD" in block
