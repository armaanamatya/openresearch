"""Guard test for handoff P2-I12 / T24 — build_environment timeouts actually enforce.

Historical note: this test shipped as a strict xfail (run=False) describing a
`with ThreadPoolExecutor(...) as pool:` block whose `__exit__` waited on the
wedged worker. The production code has since moved to a try/finally with
`pool.shutdown(wait=False, cancel_futures=True)` (primitives.py, A2-C3/I12), so
the bound now enforces — this is the live regression guard for that fix.
"""
import asyncio
import threading
import time
from types import SimpleNamespace

import backend.agents.rlm.primitives as primitives_mod


def test_build_environment_attempt_timeout_actually_bounds(monkeypatch, make_context, tmp_path):
    """A hung Docker build must NOT wedge build_environment past its cap.

    WS-H Batch P / A2-C3 bounds each attempt with .result(timeout=build_timeout)
    and releases the pool via shutdown(wait=False, cancel_futures=True). A fake
    _build_image that never finishes must produce a fail-soft timeout result
    promptly — not block on pool cleanup.
    """
    release = threading.Event()  # lets the wedged worker exit at teardown

    async def slow_build(*args, **kwargs):
        while not release.is_set():
            await asyncio.sleep(0.05)
        return False, "", "released"

    monkeypatch.setattr(primitives_mod, "_build_image", slow_build)
    # Hermetic: never touch a real Docker daemon for the exists-check.
    monkeypatch.setattr(primitives_mod, "_image_exists", lambda tag: False)

    # Pin a short per-attempt / repair cap via settings.
    fake_settings = SimpleNamespace(
        environment_build_max_attempts=1,
        environment_build_attempt_s=2,         # 2 s cap on the build
        environment_build_llm_repair_s=1,
        runpod_image="runpod/pytorch:test",  # read by _normalize_runpod_from_line
    )
    monkeypatch.setattr("backend.config.get_settings", lambda **kw: fake_settings)

    ctx = make_context(tmp_path)
    start = time.monotonic()
    try:
        result = primitives_mod.build_environment({"dockerfile": "FROM alpine\n"}, ctx=ctx)
        elapsed = time.monotonic() - start

        assert elapsed < 10, f"build_environment took {elapsed:.1f}s — bound did not enforce"
        assert result["ok"] is False
        assert "timed out" in result["error"].lower()
    finally:
        release.set()  # unwedge the worker so interpreter exit never waits on it
