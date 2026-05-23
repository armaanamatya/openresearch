"""Guard test for handoff P2-I12 / T24 — build_environment timeouts actually enforce."""
import asyncio
import time
from types import SimpleNamespace

import pytest

import backend.agents.rlm.primitives as primitives_mod


def test_build_environment_attempt_timeout_actually_bounds(monkeypatch, make_context, tmp_path):
    """Symptom: a hung Docker build wedges build_environment past its declared cap.

    WS-H Batch P / A2-C3 redesigned build_environment to bound each attempt with
    .result(timeout=build_timeout). Verify the bound actually enforces — a fake
    _build_image that sleeps 1 hour must cause TimeoutError + a fail-soft result,
    not a wedge.
    """

    async def slow_build(*args, **kwargs):
        await asyncio.sleep(3600)  # 1 hour — far past any test cap

    monkeypatch.setattr(primitives_mod, "_build_image", slow_build)

    # Pin a short per-attempt / repair cap via settings.
    fake_settings = SimpleNamespace(
        environment_build_max_attempts=1,
        environment_build_attempt_s=2,         # 2 s cap on the build
        environment_build_llm_repair_s=1,
    )
    monkeypatch.setattr("backend.config.get_settings", lambda **kw: fake_settings)

    ctx = make_context(tmp_path)
    start = time.monotonic()
    result = primitives_mod.build_environment({"dockerfile": "FROM alpine\n"}, ctx=ctx)
    elapsed = time.monotonic() - start

    assert elapsed < 10, f"build_environment took {elapsed:.1f}s — bound did not enforce"
    assert result["ok"] is False
    assert "timed out" in result["error"].lower()
