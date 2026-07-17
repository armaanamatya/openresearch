"""E1 — framework->validated-image floor for the GKE cell dispatch.

Covers the pure resolver `_resolve_framework_image` directly (no settings)
plus the `OPENRESEARCH_FRAMEWORK_IMAGES` flag gate `_maybe_framework_image`.
No network/subprocess — pure functions only.
"""
from __future__ import annotations

import pytest

from backend.agents.rlm import k8s_job_cell_runner as m


# ---------------------------------------------------------------------------
# _resolve_framework_image — pure
# ---------------------------------------------------------------------------

def test_empty_mapping_returns_base_image_unchanged():
    cells = [{"image_key": "verl", "framework": "verl"}]
    assert m._resolve_framework_image(cells, "BASE", {}) == "BASE"


def test_verl_cell_maps_to_image():
    cells = [{"image_key": "verl"}]
    assert m._resolve_framework_image(cells, "BASE", {"verl": "IMG"}) == "IMG"


def test_verl_cell_empty_base_image_still_resolves():
    # The "no manual base_image needed" case: an empty operator base_image is
    # still overridden by the mapped image.
    cells = [{"image_key": "verl"}]
    assert m._resolve_framework_image(cells, "", {"verl": "IMG"}) == "IMG"


def test_unmapped_framework_falls_back_to_base_image():
    cells = [{"image_key": "pytorch"}]
    assert m._resolve_framework_image(cells, "BASE", {"verl": "IMG"}) == "BASE"


def test_empty_mapped_image_is_ignored():
    cells = [{"image_key": "verl"}]
    assert m._resolve_framework_image(cells, "BASE", {"verl": ""}) == "BASE"


def test_ambiguous_distinct_images_falls_back_to_base_image():
    cells = [{"image_key": "verl"}, {"image_key": "hf"}]
    mapping = {"verl": "A", "hf": "B"}
    assert m._resolve_framework_image(cells, "BASE", mapping) == "BASE"


def test_two_cells_same_mapped_image_resolves():
    cells = [{"image_key": "verl"}, {"image_key": "verl"}]
    assert m._resolve_framework_image(cells, "BASE", {"verl": "IMG"}) == "IMG"


def test_image_key_takes_precedence_over_framework():
    cells = [{"image_key": "verl", "framework": "hf"}]
    mapping = {"verl": "V", "hf": "H"}
    assert m._resolve_framework_image(cells, "BASE", mapping) == "V"


def test_missing_keys_and_non_dict_entries_are_skipped():
    cells = [{}, "not-a-dict", {"unrelated": "value"}]
    assert m._resolve_framework_image(cells, "BASE", {"verl": "IMG"}) == "BASE"


# ---------------------------------------------------------------------------
# _maybe_framework_image — flag gate
# ---------------------------------------------------------------------------

def test_maybe_framework_image_off_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENRESEARCH_FRAMEWORK_IMAGES", raising=False)
    monkeypatch.setattr(
        m,
        "_cloud_setting",
        lambda logical, default=None: {"verl": "IMG"} if logical == "framework_images" else default,
    )
    cells = [{"image_key": "verl", "framework": "verl"}]
    assert m._maybe_framework_image(cells, "BASE") == "BASE"


def test_maybe_framework_image_on_resolves_mapped_image(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENRESEARCH_FRAMEWORK_IMAGES", "1")
    monkeypatch.setattr(
        m,
        "_cloud_setting",
        lambda logical, default=None: {"verl": "IMG"} if logical == "framework_images" else default,
    )
    cells = [{"image_key": "verl", "framework": "verl"}]
    assert m._maybe_framework_image(cells, "BASE") == "IMG"
