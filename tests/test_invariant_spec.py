"""Unit tests for InvariantSpec and InvariantResult schemas (PR A foundation)."""

from __future__ import annotations

import pytest

from backend.agents.schemas import InvariantResult, InvariantSpec


class TestInvariantSpecConstruction:
    def test_minimal(self):
        inv = InvariantSpec(name="x", rationale="y")
        assert inv.name == "x"
        assert inv.file_glob == "**/*.py"
        assert inv.must_match == []
        assert inv.must_not_match == []

    def test_default_glob_when_falsy(self):
        # Empty string for file_glob should fall back to the default.
        inv = InvariantSpec(name="x", rationale="y", file_glob="")
        assert inv.file_glob == "**/*.py"

    def test_custom_glob(self):
        inv = InvariantSpec(name="x", rationale="y", file_glob="src/**/*.py")
        assert inv.file_glob == "src/**/*.py"


class TestInvariantSpecRegexValidation:
    def test_valid_regex_must_match(self):
        InvariantSpec(name="x", rationale="y", must_match=[r"\bsigmoid\(.+?\)", r"detach"])

    def test_valid_regex_must_not_match(self):
        InvariantSpec(name="x", rationale="y", must_not_match=[r"class\s+TinyLM\b"])

    def test_malformed_regex_raises(self):
        with pytest.raises(ValueError, match="malformed regex"):
            InvariantSpec(name="x", rationale="y", must_match=[r"[unclosed"])

    def test_malformed_in_must_not_match_raises(self):
        with pytest.raises(ValueError, match="malformed regex"):
            InvariantSpec(name="x", rationale="y", must_not_match=[r"(?P<unclosed"])


class TestInvariantResult:
    def test_minimal(self):
        r = InvariantResult(name="x", passed=True)
        assert r.passed is True
        assert r.must_match_evidence == {}
        assert r.must_not_match_violations == {}
        assert r.files_scanned == 0

    def test_with_evidence(self):
        r = InvariantResult(
            name="stop_grad",
            passed=True,
            must_match_evidence={r"\.detach\(\)": ["loss.py:42: gate = (...).detach()"]},
            files_scanned=3,
        )
        assert r.files_scanned == 3
        assert len(r.must_match_evidence) == 1

    def test_with_violations(self):
        r = InvariantResult(
            name="no_stub",
            passed=False,
            must_not_match_violations={"class TinyLM": ["model.py:10: class TinyLM(...):"]},
            files_scanned=1,
        )
        assert r.passed is False
