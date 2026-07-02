"""
Unit tests for Phase 1e Unit A — root-cause FailureAttribution
(backend/agents/rlm/failure_attribution.py).
Pure unit tests — no network, no filesystem, no subprocess.
"""

from backend.agents.rlm.failure_attribution import FailureAttribution, attribute_failure  # noqa: F401


def test_infra_class_routes_infra():
    att = attribute_failure({"success": False, "error": "ImportError: No module named 'flash_attn'",
                             "failure_class": "missing_module"})
    assert att.scope == "infra" and att.root_cause == "missing_module" and att.signature


def test_method_class_routes_method():
    att = attribute_failure({"success": False, "failure_class": "scope_shape_violation"})
    assert att.scope == "method"


def test_unknown_class_defaults_method_not_infra():
    # conservative: never let an unclassified failure pollute cross-paper infra memory.
    att = attribute_failure({"success": False, "failure_class": "totally_unknown_class"})
    assert att.scope == "method" and att.confidence < 1.0


def test_same_root_cause_same_signature_across_runs():
    a = attribute_failure({"failure_class": "cuda_shlib_load", "error": "libcupti.so.12: cannot open (pid 4821)"})
    b = attribute_failure({"failure_class": "cuda_shlib_load", "error": "libcupti.so.12: cannot open (pid 9930)"})
    assert a.signature == b.signature       # pid/path differences normalized away
