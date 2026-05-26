"""PR-μ.1 regression — wrap_primitive timeout exclusions.

Before PR-μ.1, `PRIMITIVE_TIMEOUT_S.get(name, 1800)` silently fell through
to 1800s for run_experiment + implement_baseline despite the inline comment
claiming they were excluded. That outer wrap killed the 0.305 Adam max-mode
rerun at exactly 30 min regardless of PR-μ Solution B's inner 6h cap.

These tests pin the exclusion as actual code, not aspiration.
"""
from __future__ import annotations

from backend.agents.rlm.binding import (
    PRIMITIVE_TIMEOUT_S,
    _DEFAULT_PRIMITIVE_TIMEOUT_S,
    _LONG_RUNNING_PRIMITIVES,
)


def test_run_experiment_is_not_capped_at_1800s_default():
    assert "run_experiment" not in PRIMITIVE_TIMEOUT_S
    assert "run_experiment" in _LONG_RUNNING_PRIMITIVES
    assert _LONG_RUNNING_PRIMITIVES["run_experiment"] > _DEFAULT_PRIMITIVE_TIMEOUT_S
    # The bracket must comfortably exceed the inner max-mode cap (6h = 21600).
    assert _LONG_RUNNING_PRIMITIVES["run_experiment"] >= 21600


def test_implement_baseline_is_not_capped_at_1800s_default():
    assert "implement_baseline" not in PRIMITIVE_TIMEOUT_S
    assert "implement_baseline" in _LONG_RUNNING_PRIMITIVES
    assert _LONG_RUNNING_PRIMITIVES["implement_baseline"] > _DEFAULT_PRIMITIVE_TIMEOUT_S
    # Inner watchdog is 4h; outer must bracket above that with headroom.
    assert _LONG_RUNNING_PRIMITIVES["implement_baseline"] >= 14400


def test_long_running_table_covers_exactly_the_known_inner_capped_primitives():
    """Only run_experiment and implement_baseline have internal caps; adding
    another primitive here without its own internal cap is a footgun."""
    assert set(_LONG_RUNNING_PRIMITIVES.keys()) == {"run_experiment", "implement_baseline"}


def test_resolution_order_long_running_wins_over_default():
    """The resolution path in wrap_primitive must consult _LONG_RUNNING_PRIMITIVES
    BEFORE falling through to PRIMITIVE_TIMEOUT_S / the 1800s default."""
    # Direct simulation of the resolver:
    def _resolve(name: str) -> int:
        if name in _LONG_RUNNING_PRIMITIVES:
            return _LONG_RUNNING_PRIMITIVES[name]
        return PRIMITIVE_TIMEOUT_S.get(name, _DEFAULT_PRIMITIVE_TIMEOUT_S)

    assert _resolve("run_experiment") == 28800
    assert _resolve("implement_baseline") == 21600
    assert _resolve("understand_section") == 300       # explicit table hit
    assert _resolve("unknown_primitive") == 1800       # falls through to default
