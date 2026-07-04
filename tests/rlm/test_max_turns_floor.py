"""Unit tests for the per-cell max_turns long-horizon floor in gpu_cell_runner.

Tests the pure helper ``_floor_cell_max_turns`` directly.  The helper is
intentionally a side-effect-free function (it returns a NEW dict on a floor
application, never mutates the caller's dict), making it easy to unit-test
in isolation without touching subprocesses or GPU state.
"""
from __future__ import annotations

import pytest

from backend.agents.rlm.gpu_cell_runner import (
    _LONG_HORIZON_TURN_FLOORS,
    _floor_cell_max_turns,
)


# ---------------------------------------------------------------------------
# Sanity check: the floor constants themselves
# ---------------------------------------------------------------------------


def test_floor_constants_present():
    """ALFWorld and WebShop floors must be registered and positive."""
    assert "alfworld" in _LONG_HORIZON_TURN_FLOORS
    assert "webshop" in _LONG_HORIZON_TURN_FLOORS
    assert _LONG_HORIZON_TURN_FLOORS["alfworld"] >= 1
    assert _LONG_HORIZON_TURN_FLOORS["webshop"] >= 1


# ---------------------------------------------------------------------------
# ALFWorld — env key present
# ---------------------------------------------------------------------------


def test_alfworld_below_floor_is_raised():
    """max_turns=6 is way below the 30-turn ALFWorld floor; must become 30."""
    cell = {"id": "qwen2_5_3b__sdar__alfworld__s0", "env": "alfworld", "max_turns": 6}
    result = _floor_cell_max_turns(cell)
    assert result["max_turns"] == _LONG_HORIZON_TURN_FLOORS["alfworld"]


def test_alfworld_at_floor_is_unchanged():
    """max_turns already equal to the floor is kept as-is."""
    floor = _LONG_HORIZON_TURN_FLOORS["alfworld"]
    cell = {"id": "qwen3_1b__alfworld__s0", "env": "alfworld", "max_turns": floor}
    result = _floor_cell_max_turns(cell)
    assert result["max_turns"] == floor


def test_alfworld_above_floor_is_unchanged():
    """max_turns=50 already exceeds the ALFWorld floor; must remain 50."""
    cell = {"id": "qwen3_1b__alfworld__s0", "env": "alfworld", "max_turns": 50}
    result = _floor_cell_max_turns(cell)
    assert result["max_turns"] == 50


def test_alfworld_no_max_turns_key_sets_floor():
    """Absent max_turns key is treated as missing; floor must be inserted."""
    cell = {"id": "qwen3_1b__alfworld__s0", "env": "alfworld"}
    result = _floor_cell_max_turns(cell)
    assert result["max_turns"] == _LONG_HORIZON_TURN_FLOORS["alfworld"]


def test_alfworld_non_int_max_turns_sets_floor():
    """A string or None max_turns that cannot be converted to int is treated as missing."""
    cell = {"id": "c__alfworld__s0", "env": "alfworld", "max_turns": "six"}
    result = _floor_cell_max_turns(cell)
    assert result["max_turns"] == _LONG_HORIZON_TURN_FLOORS["alfworld"]


# ---------------------------------------------------------------------------
# WebShop — env key present
# ---------------------------------------------------------------------------


def test_webshop_below_floor_is_raised():
    """max_turns=5 is below the WebShop floor of 15; must become 15."""
    cell = {"id": "qwen2_5_3b__sdar__webshop__s0", "env": "webshop", "max_turns": 5}
    result = _floor_cell_max_turns(cell)
    assert result["max_turns"] == _LONG_HORIZON_TURN_FLOORS["webshop"]


def test_webshop_above_floor_is_unchanged():
    """max_turns=20 already exceeds the WebShop floor."""
    cell = {"id": "q3b__webshop__s0", "env": "webshop", "max_turns": 20}
    result = _floor_cell_max_turns(cell)
    assert result["max_turns"] == 20


def test_webshop_no_max_turns_key_sets_floor():
    cell = {"id": "q3b__webshop__s0", "env": "webshop"}
    result = _floor_cell_max_turns(cell)
    assert result["max_turns"] == _LONG_HORIZON_TURN_FLOORS["webshop"]


# ---------------------------------------------------------------------------
# Search-QA — NOT a long-horizon env; must pass through unchanged
# ---------------------------------------------------------------------------


def test_search_qa_not_floored():
    """search_qa has no long-horizon floor; max_turns=4 must stay 4."""
    cell = {"id": "q1b__sdar__search_qa__s0", "env": "search_qa", "max_turns": 4}
    result = _floor_cell_max_turns(cell)
    assert result["max_turns"] == 4


def test_unknown_env_not_floored():
    """An env not in _LONG_HORIZON_TURN_FLOORS must not be touched."""
    cell = {"id": "c__cifar10__s0", "env": "cifar10", "max_turns": 2}
    result = _floor_cell_max_turns(cell)
    assert result["max_turns"] == 2


# ---------------------------------------------------------------------------
# Env detected from cell id when "env" key is absent
# ---------------------------------------------------------------------------


def test_env_from_id_alfworld_no_env_key():
    """When the 'env' key is absent, the id substring 'alfworld' triggers the floor."""
    cell = {"id": "q__sdar__alfworld__s0", "max_turns": 6}
    result = _floor_cell_max_turns(cell)
    assert result["max_turns"] == _LONG_HORIZON_TURN_FLOORS["alfworld"]


def test_env_from_id_webshop_no_env_key():
    cell = {"id": "q__sdar__webshop__s0", "max_turns": 3}
    result = _floor_cell_max_turns(cell)
    assert result["max_turns"] == _LONG_HORIZON_TURN_FLOORS["webshop"]


def test_env_from_id_search_qa_not_floored():
    """search_qa in the id does not trigger any floor."""
    cell = {"id": "q__sdar__search_qa__s0", "max_turns": 4}
    result = _floor_cell_max_turns(cell)
    assert result["max_turns"] == 4


# ---------------------------------------------------------------------------
# Immutability: the caller's original dict must NEVER be mutated
# ---------------------------------------------------------------------------


def test_original_dict_not_mutated_when_floor_applied():
    """When the floor is applied, the caller's cell dict must be untouched."""
    cell = {"id": "qwen3_1b__alfworld__s0", "env": "alfworld", "max_turns": 6}
    original_max_turns = cell["max_turns"]
    result = _floor_cell_max_turns(cell)
    # The returned dict carries the floor value…
    assert result["max_turns"] == _LONG_HORIZON_TURN_FLOORS["alfworld"]
    # …but the original is unchanged.
    assert cell["max_turns"] == original_max_turns


def test_original_dict_not_mutated_when_no_change():
    """When no floor is needed, the function returns the original dict (identity)."""
    cell = {"id": "q1b__alfworld__s0", "env": "alfworld", "max_turns": 50}
    result = _floor_cell_max_turns(cell)
    # Same object returned (no copy made on the no-change path).
    assert result is cell


def test_original_dict_not_mutated_non_long_horizon():
    """Non-long-horizon cells also get the original dict back (no copy)."""
    cell = {"id": "q1b__cifar10__s0", "env": "cifar10", "max_turns": 2}
    result = _floor_cell_max_turns(cell)
    assert result is cell


# ---------------------------------------------------------------------------
# Other cell keys are preserved when a floor copy is made
# ---------------------------------------------------------------------------


def test_other_keys_preserved():
    """Applying a floor must not drop any other cell keys."""
    cell = {
        "id": "qwen3_1b__alfworld__s0",
        "env": "alfworld",
        "max_turns": 6,
        "model": "Qwen3-1.7B",
        "seed": 0,
        "env_name": "alfworld_train",
    }
    result = _floor_cell_max_turns(cell)
    assert result["model"] == "Qwen3-1.7B"
    assert result["seed"] == 0
    assert result["env_name"] == "alfworld_train"
    assert result["id"] == cell["id"]
    assert result["max_turns"] == _LONG_HORIZON_TURN_FLOORS["alfworld"]
