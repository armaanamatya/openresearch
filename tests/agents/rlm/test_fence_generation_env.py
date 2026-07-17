"""Env fallback for the WS3 durable-controller cell fence.

Inside the durable controller Pod nothing binds the ``fence_generation``
ContextVar, so ``_get_fence_generation`` reads the stable fence epoch the submit
stamped into the Pod env (``OPENRESEARCH_CELL_FENCE_EPOCH``). An explicit
ContextVar binding still wins over the env fallback.
"""

from __future__ import annotations

import backend.agents.rlm.k8s_job_cell_runner as kjcr
from backend.agents.rlm.k8s_job_cell_runner import bind_run_context


def test_env_fallback_when_contextvar_unbound(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_CELL_FENCE_EPOCH", "5")
    assert kjcr._get_fence_generation() == 5


def test_none_when_unbound_and_no_env(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_CELL_FENCE_EPOCH", raising=False)
    assert kjcr._get_fence_generation() is None


def test_invalid_env_yields_none(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_CELL_FENCE_EPOCH", "not-an-int")
    assert kjcr._get_fence_generation() is None


def test_contextvar_binding_wins_over_env(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_CELL_FENCE_EPOCH", "5")
    with bind_run_context(fence_generation=9):
        assert kjcr._get_fence_generation() == 9
