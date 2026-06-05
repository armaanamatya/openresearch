"""F-05: an already-FAILED run with no specific failure_class should still surface a
masked code bug's precise message (without flipping success), so the next repair
targets the real loader/parse bug instead of a vague error.
"""
from __future__ import annotations

from backend.agents.rlm.primitives import _surface_masked_bug_on_failed_run


def _result(success, failure_class=None, data_load_failures=None, **extra):
    r = {"success": success, "metrics": {"data_load_failures": data_load_failures or []}}
    if failure_class is not None:
        r["failure_class"] = failure_class
    r.update(extra)
    return r


# An AttributeError caught and stored as data-unavailability — a CODE bug masked as a
# data_load_failure (the kind _data_load_failure_is_code_bug flags).
_CODE_BUG_FAILURE = [
    {"dataset": "alfworld", "error": "AttributeError: 'NoneType' object has no attribute 'reset'"}
]
# A genuine, provably-uncontrollable data absence — NOT a code bug.
_GENUINE_DATA_FAILURE = [{"dataset": "webshop", "error": "HTTP 404: dataset not found on the hub"}]


def test_failed_run_no_class_surfaces_masked_bug():
    out = _surface_masked_bug_on_failed_run(_result(False, data_load_failures=_CODE_BUG_FAILURE))
    assert out is not None
    assert out["failure_class"] == "code_bug"
    assert "code_bug:" in out["error"]
    assert "alfworld" in out["error"]
    assert out["suggested_fix"]  # precise message promoted into suggested_fix


def test_does_not_flip_success_true():
    # The success path is handled by the existing block; this helper is a no-op there.
    assert _surface_masked_bug_on_failed_run(_result(True, data_load_failures=_CODE_BUG_FAILURE)) is None


def test_does_not_override_specific_failure_class():
    out = _surface_masked_bug_on_failed_run(
        _result(False, failure_class="dockerfile_invalid", data_load_failures=_CODE_BUG_FAILURE)
    )
    assert out is None


def test_genuine_data_failure_is_not_surfaced():
    assert _surface_masked_bug_on_failed_run(_result(False, data_load_failures=_GENUINE_DATA_FAILURE)) is None


def test_no_failures_returns_none():
    assert _surface_masked_bug_on_failed_run(_result(False, data_load_failures=[])) is None


def test_preserves_existing_suggested_fix():
    out = _surface_masked_bug_on_failed_run(
        _result(False, data_load_failures=_CODE_BUG_FAILURE, suggested_fix="pre-existing fix")
    )
    assert out is not None
    assert out["suggested_fix"] == "pre-existing fix"
