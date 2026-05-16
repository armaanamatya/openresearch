"""Tests for the GET /pipeline/topology endpoint."""

from __future__ import annotations

from starlette.testclient import TestClient

from backend.app import create_app


def test_pipeline_topology_endpoint_returns_default_shape() -> None:
    client = TestClient(create_app())
    response = client.get("/pipeline/topology")

    assert response.status_code == 200
    body = response.json()

    assert "nodes" in body
    assert "edges" in body
    assert "gates" in body
    assert "stages" in body
    assert "improvement_path_ids" in body

    assert len(body["nodes"]) == 12
    assert len(body["edges"]) == 16
    assert len(body["gates"]) == 3
    assert len(body["stages"]) == 14
    assert body["improvement_path_ids"] == ["opt", "bb", "aug", "hor", "div"]


def test_pipeline_topology_node_has_required_fields() -> None:
    client = TestClient(create_app())
    body = client.get("/pipeline/topology").json()

    for node in body["nodes"]:
        assert "id" in node
        assert "kind" in node
        assert "internal_label" in node
        assert "demo_label" in node
        assert "step" in node
        assert "icon" in node
        assert "tone" in node
        assert "agent_ids" in node
        assert isinstance(node["agent_ids"], list)


def test_pipeline_topology_audit_node_has_audit_kind() -> None:
    """The 'audit' node has kind='audit' so the frontend can route the
    HermesAuditPanel render conditionally on kind, not on the literal
    id string."""
    client = TestClient(create_app())
    body = client.get("/pipeline/topology").json()
    audit_node = next(n for n in body["nodes"] if n["id"] == "audit")
    assert audit_node["kind"] == "audit"

    report_node = next(n for n in body["nodes"] if n["id"] == "report")
    assert report_node["kind"] == "report"


def test_get_models_returns_sonnet_and_opus() -> None:
    """GET /models exposes the dropdown the upload-view renders from.

    The frontend's UploadView used to ship its `<option>` list hardcoded;
    Phase D.10 routes it through this endpoint so adding (or retiring) a
    model is a backend-only change.
    """
    client = TestClient(create_app())
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    ids = {m["id"] for m in body}
    assert "sonnet" in ids and "opus" in ids
    for m in body:
        assert "id" in m and "label" in m and "provider" in m
