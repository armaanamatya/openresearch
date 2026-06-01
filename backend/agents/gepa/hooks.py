"""GEPA pre/post call hooks — called by wrap_primitive()."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

GEPA_TARGETS = frozenset({"plan_reproduction", "implement_baseline", "propose_improvements"})


def _is_enabled(ctx: Any, primitive_name: str) -> bool:
    """Return True if GEPA should run for this primitive call."""
    if ctx.gepa_optimization_active:
        return False  # re-entrancy guard
    try:
        from backend.config import Settings
        cfg = Settings()
        mode = cfg.gepa_optimization
    except Exception:
        return False
    if mode == "off":
        return False
    if mode == "on":
        return primitive_name in GEPA_TARGETS
    mode_map = {
        "plan-only": "plan_reproduction",
        "baseline-only": "implement_baseline",
        "improve-only": "propose_improvements",
    }
    return mode_map.get(mode) == primitive_name


def gepa_pre_call(ctx: Any, primitive_name: str, args: tuple, kwargs: dict) -> None:
    """Run GEPA optimization before a targeted primitive call.

    On return, ctx.gepa_prompt_overrides[primitive_name] holds the winning
    optimized system prompt (or remains unset if GEPA is disabled/failed).
    """
    if not _is_enabled(ctx, primitive_name):
        return
    # Full implementation added in later task.


def gepa_post_call(ctx: Any, primitive_name: str, result: Any) -> None:
    """Record real primitive result as a training example for future calls."""
    if primitive_name not in GEPA_TARGETS:
        return
    if ctx.gepa_example_buffer.get(primitive_name) is None:
        ctx.gepa_example_buffer[primitive_name] = []
    # Full implementation added in later task.
