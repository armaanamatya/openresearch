"""Spec §11.2 — scoped default-OFF contract.

Every NEW flag introduced by the 2026-06-20 grounded-self-improvement work is
default-OFF, so with them unset the harness is byte-identical to its prior
baseline. That baseline already includes default-ON rails (leaf-triage,
metric-provenance) — those are NOT new flags and stay ON. This test pins both
sides in one place.
"""

from __future__ import annotations


_NEW_FLAGS = [
    "OPENRESEARCH_ZERO_METRICS_GUARD",
    "OPENRESEARCH_LIFECYCLE_LEDGER",
    "OPENRESEARCH_EXTERNAL_VALIDATOR",
    "OPENRESEARCH_VALIDATOR_BACKEND",
    "OPENRESEARCH_POSITIVE_RECIPES",
]


def test_all_new_flags_default_off(monkeypatch):
    for var in _NEW_FLAGS:
        monkeypatch.delenv(var, raising=False)
    from backend.agents.rlm.external_validator import external_validator_enabled
    from backend.agents.rlm.lifecycle_ledger import lifecycle_ledger_enabled
    from backend.agents.rlm.recipe_library import positive_recipes_enabled
    from backend.agents.rlm.zero_metrics_detection import zero_metrics_guard_enabled

    assert zero_metrics_guard_enabled() is False
    assert lifecycle_ledger_enabled() is False
    assert external_validator_enabled() is False
    assert positive_recipes_enabled() is False


def test_existing_default_on_rails_remain_baseline(monkeypatch):
    # These are the §11.2 baseline rails — default-ON and NOT new flags.
    for var in ("REPROLAB_LEAF_TRIAGE", "OPENRESEARCH_METRIC_PROVENANCE"):
        monkeypatch.delenv(var, raising=False)
    from backend.agents.rlm.leaf_triage import is_enabled as leaf_triage_enabled
    from backend.agents.rlm.report import _metric_provenance_enabled

    assert leaf_triage_enabled() is True
    assert _metric_provenance_enabled() is True


def test_repair_max_default_is_generous(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_REPAIR_MAX_ITERATIONS", raising=False)
    from backend.agents.rlm.forced_iteration import _default_repair_max

    assert _default_repair_max() == 4
