"""Executor-tier resolver — default→Sonnet, qwen+probe-ok→OpenAI/vLLM plan, fail→fallback."""
from __future__ import annotations

import backend.agents.rlm.executor as ex
from backend.agents.rlm.executor import ExecutorPlan, resolve_executor


def test_default_returns_none(monkeypatch):
    monkeypatch.delenv("REPROLAB_EXECUTOR", raising=False)
    assert resolve_executor() is None
    monkeypatch.setenv("REPROLAB_EXECUTOR", "sonnet")
    assert resolve_executor() is None


def test_unknown_mode_falls_back(monkeypatch):
    monkeypatch.setenv("REPROLAB_EXECUTOR", "gpt9")
    assert resolve_executor() is None


def test_qwen_probe_ok_returns_plan(monkeypatch):
    monkeypatch.setenv("REPROLAB_EXECUTOR", "qwen")
    monkeypatch.setenv("REPROLAB_EXECUTOR_MODEL", "Qwen/Qwen2.5-Coder-14B-Instruct")
    monkeypatch.setattr(ex, "_probe", lambda *a, **k: True)
    plan = resolve_executor()
    assert isinstance(plan, ExecutorPlan)
    assert plan.model == "Qwen/Qwen2.5-Coder-14B-Instruct"
    assert getattr(plan.runtime, "_base_url", None)
    assert plan.runtime.provider_name == "openai"


def test_qwen_probe_fail_falls_back_to_sonnet(monkeypatch):
    monkeypatch.setenv("REPROLAB_EXECUTOR", "qwen")
    monkeypatch.setattr(ex, "_probe", lambda *a, **k: False)
    assert resolve_executor() is None
