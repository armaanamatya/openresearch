"""GET /papers — the bundled, selectable reproduction targets surfaced to the UI."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import create_app


def test_get_papers_lists_bundled_presets():
    client = TestClient(create_app())
    r = client.get("/papers")
    assert r.status_code == 200
    papers = r.json()["papers"]
    ids = {p["id"] for p in papers}
    assert {"sdar", "adam", "allcnn"} <= ids
    sdar = next(p for p in papers if p["id"] == "sdar")
    assert sdar["arxiv_id"] == "2605.15155"
    assert sdar["title"] and "ALFWorld" in sdar["datasets"]
    # every preset is well-shaped for the dropdown
    for p in papers:
        assert p["id"] and p["title"]
        assert isinstance(p["datasets"], list)
