"""Bundled-paper registry (2026-06-15): the top reproduction targets ship in-repo
so a fresh clone can select + reproduce them offline. SDAR's future-dated arXiv id
does not fetch — bundling is what makes it work. These tests pin resolution by
id/alias/arXiv-id, the on-disk PDF guard, preset listing, and fail-soft behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services.ingestion import paper_registry as pr

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_bundled_pdfs_and_registry_exist():
    """The repo ships papers/registry.json + a PDF per entry (the 'pushed to git' contract)."""
    reg = REPO_ROOT / "papers" / "registry.json"
    assert reg.is_file(), "papers/registry.json must be committed"
    entries = pr.load_registry()
    assert {e.id for e in entries} >= {"sdar", "adam", "allcnn"}
    for e in entries:
        assert (REPO_ROOT / e.pdf).is_file(), f"bundled PDF missing for {e.id}: {e.pdf}"


@pytest.mark.parametrize("source,expected", [
    ("sdar", "sdar"),
    ("SDAR", "sdar"),
    ("2605.15155", "sdar"),
    ("adam", "adam"),
    ("1412.6980", "adam"),
    ("all-cnn", "allcnn"),
    ("allcnn", "allcnn"),
    ("1412.6806", "allcnn"),
])
def test_resolve_by_id_alias_arxiv(source, expected):
    e = pr.resolve(source)
    assert e is not None and e.id == expected


def test_resolve_unregistered_is_none():
    assert pr.resolve("9999.99999") is None
    assert pr.resolve("") is None
    assert pr.resolve("some-random-paper") is None


def test_resolve_pdf_path_points_at_existing_file():
    p = pr.resolve_pdf_path("sdar")
    assert p is not None and p.is_file() and p.name == "sdar.pdf"
    assert pr.resolve_pdf_path("9999.99999") is None


def test_sdar_carries_hint_and_datasets():
    e = pr.resolve("2605.15155")
    assert e.hint == "2605.15155"  # auto-applied by the CLI when --paper-hint is unset
    assert "ALFWorld" in e.datasets


def test_list_presets_shape():
    presets = pr.list_presets()
    ids = {p["id"] for p in presets}
    assert {"sdar", "adam", "allcnn"} <= ids
    for p in presets:
        assert {"id", "title", "arxiv_id", "datasets", "default"} <= set(p)


def test_failsoft_on_missing_registry(tmp_path):
    """A missing/malformed registry returns [] / None — never raises (normal fetch path)."""
    missing = tmp_path / "nope.json"
    assert pr.load_registry(missing) == []
    assert pr.resolve("sdar", missing) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert pr.load_registry(bad) == []


def test_registry_json_wellformed():
    data = json.loads((REPO_ROOT / "papers" / "registry.json").read_text(encoding="utf-8"))
    assert data.get("version") and isinstance(data.get("papers"), list)
    for p in data["papers"]:
        assert p["id"] and p["pdf"] and p.get("arxiv_id")
