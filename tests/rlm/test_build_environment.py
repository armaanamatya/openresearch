import pytest

import backend.agents.rlm.primitives as primitives
from backend.agents.rlm.primitives import build_environment


def test_build_environment_succeeds_first_try(make_context, tmp_path, monkeypatch):
    ctx = make_context(tmp_path)

    async def fake_build_image(dockerfile_path, context_dir, tag, **kw):
        return (True, tag, "")

    monkeypatch.setattr(primitives, "_build_image", fake_build_image)
    result = build_environment({"dockerfile": "FROM python:3.11-slim\n"}, ctx=ctx)
    assert result["ok"] is True
    assert result["image_tag"]
    assert result["attempts"] == 1


def test_build_environment_repairs_then_succeeds(make_context, tmp_path, monkeypatch):
    ctx = make_context(tmp_path, llm_responses=["FROM python:3.11-slim\nRUN pip install x\n"])
    calls = {"n": 0}

    async def fake_build_image(dockerfile_path, context_dir, tag, **kw):
        calls["n"] += 1
        return (calls["n"] > 1, tag, "" if calls["n"] > 1 else "pip failed")

    monkeypatch.setattr(primitives, "_build_image", fake_build_image)
    result = build_environment({"dockerfile": "FROM bad\n"}, ctx=ctx)
    assert result["ok"] is True
    assert result["attempts"] == 2
    assert len(ctx.llm_client.calls) == 1


def test_build_environment_cap_exhausted_returns_fail_soft(make_context, tmp_path, monkeypatch):
    ctx = make_context(tmp_path, llm_responses=["FROM repaired\n"])

    async def always_fail(dockerfile_path, context_dir, tag, **kw):
        return (False, tag, "always fails")

    monkeypatch.setattr(primitives, "_build_image", always_fail)
    result = build_environment({"dockerfile": "FROM bad\n"}, ctx=ctx)
    assert result["ok"] is False
    assert result["image_tag"] == ""
    assert result["attempts"] >= 1


def test_build_environment_maps_sandbox_runtime_error(make_context, tmp_path, monkeypatch):
    from backend.services.runtime.interface import RuntimeCauseKind, SandboxRuntimeError

    ctx = make_context(tmp_path)

    async def daemon_down(dockerfile_path, context_dir, tag, **kw):
        raise SandboxRuntimeError(RuntimeCauseKind.backend_unavailable, "daemon unreachable")

    monkeypatch.setattr(primitives, "_build_image", daemon_down)
    result = build_environment({"dockerfile": "FROM python:3.11-slim\n"}, ctx=ctx)
    assert result["ok"] is False
    assert result["outcome"] == "fatal"
    assert "daemon unreachable" in result["error"]


def test_build_environment_distinct_dockerfiles_get_distinct_tags(
        make_context, tmp_path, monkeypatch):
    # A Docker tag is a mutable pointer — two builds in one run must not share
    # a tag, or run_experiment runs whichever image the tag last pointed at.
    ctx = make_context(tmp_path)

    async def fake_build_image(dockerfile_path, context_dir, tag, **kw):
        return (True, tag, "")

    monkeypatch.setattr(primitives, "_build_image", fake_build_image)
    a = build_environment({"dockerfile": "FROM python:3.11-slim\n"}, ctx=ctx)
    b = build_environment({"dockerfile": "FROM python:3.12-slim\n"}, ctx=ctx)
    again = build_environment({"dockerfile": "FROM python:3.11-slim\n"}, ctx=ctx)
    assert a["image_tag"] != b["image_tag"]      # distinct Dockerfiles -> distinct tags
    assert a["image_tag"] == again["image_tag"]  # same Dockerfile -> stable tag
