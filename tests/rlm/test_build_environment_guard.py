"""Tests for the image-exists guard in build_environment (D1-D5)."""

import pytest
import backend.agents.rlm.primitives as primitives
from backend.agents.rlm.primitives import build_environment


def test_guard_skips_build_when_image_exists(make_context, tmp_path, monkeypatch):
    """When _image_exists returns True, build is skipped and skipped=True is set."""
    ctx = make_context(tmp_path)
    build_called = {"n": 0}

    async def should_not_be_called(*args, **kwargs):
        build_called["n"] += 1
        return (True, args[2] if len(args) > 2 else "tag", "")

    monkeypatch.setattr(primitives, "_build_image", should_not_be_called)
    monkeypatch.setattr(primitives, "_image_exists", lambda tag: True)

    result = build_environment({"dockerfile": "FROM python:3.11-slim\n"}, ctx=ctx)

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["image_tag"]         # tag must be the content-addressed value
    assert result["attempts"] == 0     # no build attempt was made
    assert build_called["n"] == 0      # Docker build path was NOT entered


def test_guard_proceeds_when_image_absent(make_context, tmp_path, monkeypatch):
    """When _image_exists returns False, the normal build path runs."""
    ctx = make_context(tmp_path)
    build_called = {"n": 0}

    async def fake_build(dockerfile_path, context_dir, tag, **kw):
        build_called["n"] += 1
        return (True, tag, "")

    monkeypatch.setattr(primitives, "_build_image", fake_build)
    monkeypatch.setattr(primitives, "_image_exists", lambda tag: False)

    result = build_environment({"dockerfile": "FROM python:3.11-slim\n"}, ctx=ctx)

    assert result["ok"] is True
    assert result.get("skipped") is not True  # guard did NOT fire
    assert result["attempts"] == 1             # exactly one build attempt
    assert build_called["n"] == 1             # Docker build path WAS entered


def test_guard_tag_matches_content_addressed_value(make_context, tmp_path, monkeypatch):
    """The tag returned when guard fires must equal the content-addressed tag."""
    import hashlib

    ctx = make_context(tmp_path)
    dockerfile = "FROM python:3.11-slim\n"
    # build_environment strips the dockerfile before hashing — mirror that here.
    stripped = dockerfile.strip()
    expected_digest = hashlib.sha1(stripped.encode("utf-8")).hexdigest()[:12]
    expected_tag = f"reprolab/{ctx.project_id}:env-{expected_digest}"

    monkeypatch.setattr(primitives, "_image_exists", lambda tag: True)
    monkeypatch.setattr(primitives, "_build_image",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not call")))

    result = build_environment({"dockerfile": dockerfile}, ctx=ctx)
    assert result["image_tag"] == expected_tag


def test_guard_reruns_when_image_absent_after_rmi(make_context, tmp_path, monkeypatch):
    """D5: guard re-evaluates per call; False → build happens even after a prior True."""
    ctx = make_context(tmp_path)
    calls = {"exists": 0, "build": 0}

    def exists_toggling(tag):
        calls["exists"] += 1
        # First call: image present (skip). Second call: gone (build).
        return calls["exists"] == 1

    async def fake_build(dockerfile_path, context_dir, tag, **kw):
        calls["build"] += 1
        return (True, tag, "")

    monkeypatch.setattr(primitives, "_image_exists", exists_toggling)
    monkeypatch.setattr(primitives, "_build_image", fake_build)

    dockerfile = "FROM python:3.11-slim\n"
    r1 = build_environment({"dockerfile": dockerfile}, ctx=ctx)
    r2 = build_environment({"dockerfile": dockerfile}, ctx=ctx)

    assert r1["skipped"] is True
    assert r2.get("skipped") is not True
    assert calls["build"] == 1  # only the second call triggered a build
