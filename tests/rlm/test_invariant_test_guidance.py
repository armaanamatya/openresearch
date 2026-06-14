"""Fidelity-certificate invariant-test directive (the two-axis producer unlock).

Without code/test_reproduction.py the certificate can never go green, so the
two-axis verdict is stuck at 'inconclusive'. This directive (gated on
REPROLAB_TWO_AXIS_VERDICT) asks the agent to write it — paper-agnostic, naming
registered invariants when PAPER_HINTS has them, degrading gracefully otherwise.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.fidelity_certificate_builder import invariant_test_guidance_block


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch):
    monkeypatch.delenv("REPROLAB_TWO_AXIS_VERDICT", raising=False)


class TestInvariantTestGuidance:
    def test_off_is_empty(self):
        # Default behaviour byte-for-byte unchanged when the flag is off.
        assert invariant_test_guidance_block("2605.15155") == ""

    def test_on_emits_directive(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_TWO_AXIS_VERDICT", "1")
        g = invariant_test_guidance_block("1412.6980")  # Adam — 0 invariants
        assert "test_reproduction.py" in g
        assert "CORE algorithmic invariants" in g
        # No invariants registered → no concrete-naming line.
        assert "registered invariants to assert" not in g

    def test_on_names_registered_invariants(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_TWO_AXIS_VERDICT", "1")
        g = invariant_test_guidance_block("2605.15155")  # SDAR — 6 invariants
        assert "registered invariants to assert" in g
        assert "sigmoid_gate_on_advantage" in g

    def test_paper_agnostic_unknown_id(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_TWO_AXIS_VERDICT", "1")
        # Unknown / None arxiv → graceful general directive, never a crash.
        for pid in ("9999.99999", None):
            g = invariant_test_guidance_block(pid)
            assert "test_reproduction.py" in g
            assert "registered invariants to assert" not in g
