"""Unit tests for the PaperHint Pydantic schema (PR A foundation).

The PAPER_HINTS table itself is tested in tests/test_paper_hints.py (PR A
Wave 2). This file covers the schema's defaults, composition, and validation
in isolation.
"""

from __future__ import annotations

from backend.agents.schemas import (
    DatasetSlice,
    InvariantSpec,
    PaperHint,
    ScopeSpec,
)


class TestPaperHintDefaults:
    def test_empty(self):
        h = PaperHint()
        assert h.guidance == ""
        assert h.default_scope is None
        assert h.invariants == []
        assert h.primitive_share is None


class TestPaperHintWithEverything:
    def test_full_construction(self):
        h = PaperHint(
            guidance="Use real weights.",
            default_scope=ScopeSpec(
                models=["A", "B"],
                datasets=[DatasetSlice(name="X")],
                seeds=[42],
            ),
            invariants=[
                InvariantSpec(
                    name="stop_grad",
                    rationale="gate must be detached",
                    must_match=[r"\.detach\(\)"],
                )
            ],
            primitive_share={"implement_baseline": 0.2, "run_experiment": 0.7},
        )
        assert h.guidance.startswith("Use real")
        assert h.default_scope is not None
        assert h.default_scope.is_multi_model is True
        assert len(h.invariants) == 1
        assert h.primitive_share["run_experiment"] == 0.7

    def test_invariant_validation_propagates(self):
        # A PaperHint carrying a malformed regex should fail at construction.
        import pytest
        with pytest.raises(ValueError, match="malformed regex"):
            PaperHint(
                invariants=[
                    InvariantSpec(name="x", rationale="y", must_match=[r"[bad"])
                ]
            )
