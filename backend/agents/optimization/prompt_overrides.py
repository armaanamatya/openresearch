"""Thread-local prompt-override context for GEPA evaluations.

The GEPA adapter wraps each per-paper evaluation in
``with PROMPT_OVERRIDES.use({component_id: candidate_text}): ...`` so that
``AgentSpec.to_runtime_spec`` and ``build_system_prompt`` resolve the
candidate text at INVOCATION time. This is the fix for the import-time
``AgentSpec.prompt`` snapshot hazard documented in
``docs/superpowers/specs/2026-05-26-gepa-phase0-audit.md`` §0.3.

Outside an active override context, ``current()`` returns ``{}`` — production
code sees today's defaults byte-for-byte.

Thread-local (not contextvar) because the existing RLM dispatch uses raw
threads (``run_isolated`` and ``ProcessPoolExecutor``-spawned subprocesses);
the override propagates into the same thread that calls ``invoke_agent``.
Process boundaries do NOT inherit — the GEPA adapter materializes overrides
into a JSON file under the candidate run-dir and re-establishes the context
inside the child process.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager


class PromptOverrideContext:
    """Thread-local stack of component_id → text override dicts."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _stack(self) -> list[dict[str, str]]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    @contextmanager
    def use(self, overrides: Mapping[str, str]) -> Iterator[None]:
        """Push ``overrides`` onto the stack for the duration of the block."""
        frame = dict(overrides)
        stack = self._stack()
        stack.append(frame)
        try:
            yield
        finally:
            popped = stack.pop()
            assert popped is frame, "PromptOverrideContext stack corrupted"

    def current(self) -> dict[str, str]:
        """Return the *merged* view of all stacked frames (top wins)."""
        merged: dict[str, str] = {}
        for frame in self._stack():
            merged.update(frame)
        return merged

    def get(self, component_id: str) -> str | None:
        """Return the override for ``component_id`` if active, else ``None``."""
        for frame in reversed(self._stack()):
            if component_id in frame:
                return frame[component_id]
        return None

    def has_active_override(self) -> bool:
        return bool(self._stack())


PROMPT_OVERRIDES = PromptOverrideContext()


__all__ = ["PROMPT_OVERRIDES", "PromptOverrideContext"]
