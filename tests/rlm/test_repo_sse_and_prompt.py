import pytest

from backend.agents.rlm.sse_bridge import (
    build_repo_resolved_event,
    build_repo_cloned_event,
)
from backend.agents.rlm.system_prompt import build_system_prompt


def test_repo_resolved_event_shape():
    ev = build_repo_resolved_event(
        url="https://github.com/me/mine", source="user", mode="adapt", reason="r",
    )
    assert ev["event"] == "repo_resolved"
    assert ev["url"] == "https://github.com/me/mine"
    assert ev["source"] == "user"
    assert ev["mode"] == "adapt"
    assert "timestamp" in ev


def test_repo_cloned_event_shape():
    ev = build_repo_cloned_event(commit_sha="abc1234", size_mb=1.5, key_files=["README.md"])
    assert ev["event"] == "repo_cloned"
    assert ev["commit_sha"] == "abc1234"
    assert ev["size_mb"] == 1.5
    assert ev["key_files"] == ["README.md"]
    assert "timestamp" in ev


def _ctx_meta():
    return {"context": {"type": "str", "length": 10}}


def _root_model():
    from backend.agents.rlm.models import ROOT_MODELS
    return ROOT_MODELS["gpt-5"]


def test_prompt_omits_repo_section_when_flag_off(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    prompt = build_system_prompt(context_metadata=_ctx_meta(), root_model=_root_model())
    assert "AUTHOR REPOSITORY" not in prompt


def test_prompt_includes_repo_section_when_flag_on(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    prompt = build_system_prompt(context_metadata=_ctx_meta(), root_model=_root_model())
    assert "AUTHOR REPOSITORY" in prompt
    assert "repo_files" in prompt
