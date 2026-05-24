"""CLI parsing + scope-spec composition tests (PR A Wave 3)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from backend.agents.schemas import ScopeSpec
from backend.cli import _build_parser, _load_scope_spec_arg, _REPRODUCE_DEFAULTS


class TestReproduceDefaults:
    def test_paper_hint_default_present(self):
        assert "paper_hint" in _REPRODUCE_DEFAULTS
        assert _REPRODUCE_DEFAULTS["paper_hint"] is None

    def test_scope_spec_default_present(self):
        assert "scope_spec" in _REPRODUCE_DEFAULTS
        assert _REPRODUCE_DEFAULTS["scope_spec"] is None


class TestCliFlagParsing:
    def test_paper_hint_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["reproduce", "foo.pdf", "--paper-hint", "2605.15155"])
        assert args.paper_hint == "2605.15155"

    def test_scope_spec_inline_json(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["reproduce", "foo.pdf", "--scope-spec", '{"models":["X"],"seeds":[1]}']
        )
        assert args.scope_spec == '{"models":["X"],"seeds":[1]}'

    def test_scope_spec_path(self):
        parser = _build_parser()
        args = parser.parse_args(["reproduce", "foo.pdf", "--scope-spec", "/tmp/scope.json"])
        assert args.scope_spec == "/tmp/scope.json"

    def test_neither_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["reproduce", "foo.pdf"])
        assert args.paper_hint is None
        assert args.scope_spec is None


class TestLoadScopeSpecArg:
    def test_none_returns_empty_scope(self):
        s = _load_scope_spec_arg(None)
        assert isinstance(s, ScopeSpec)
        assert s.models == []

    def test_empty_string_returns_empty_scope(self):
        assert _load_scope_spec_arg("").models == []
        assert _load_scope_spec_arg("   ").models == []

    def test_inline_json(self):
        s = _load_scope_spec_arg('{"models":["A","B"],"seeds":[7]}')
        assert s.models == ["A", "B"]
        assert s.seeds == [7]

    def test_inline_json_with_leading_whitespace(self):
        s = _load_scope_spec_arg('   {"models":["A"]}   ')
        assert s.models == ["A"]

    def test_path(self, tmp_path: Path):
        p = tmp_path / "scope.json"
        p.write_text(json.dumps({"models": ["P"], "datasets": ["D1"]}))
        s = _load_scope_spec_arg(str(p))
        assert s.models == ["P"]
        assert s.dataset_ids() == ["D1"]

    def test_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="--scope-spec"):
            _load_scope_spec_arg(str(tmp_path / "missing.json"))


class TestEndToEndComposition:
    """The composition block in cmd_reproduce is exercised indirectly via env vars.

    Here we verify the merge logic produces what cli.cmd_reproduce will write.
    """

    def test_sdar_default_alone(self):
        from backend.agents.prompts.paper_hints import lookup_paper_hint
        hint = lookup_paper_hint("2605.15155")
        operator = ScopeSpec()  # empty operator
        effective = operator.merge_with_paper_default(hint.default_scope)
        assert len(effective.models) == 3
        assert "Qwen3-1.7B-Instruct" in effective.models

    def test_operator_narrows_sdar(self):
        from backend.agents.prompts.paper_hints import lookup_paper_hint
        hint = lookup_paper_hint("2605.15155")
        operator = ScopeSpec(models=["Qwen3-1.7B-Instruct"])  # operator picks one
        effective = operator.merge_with_paper_default(hint.default_scope)
        assert effective.models == ["Qwen3-1.7B-Instruct"]
        # datasets + seeds fall back to paper defaults
        assert len(effective.datasets) == 3
        assert effective.seeds == [42, 43, 44]

    def test_operator_skip_models_drops_from_paper_default(self):
        from backend.agents.prompts.paper_hints import lookup_paper_hint
        hint = lookup_paper_hint("2605.15155")
        operator = ScopeSpec(skip_models=["Qwen2.5-7B-Instruct"])
        effective = operator.merge_with_paper_default(hint.default_scope)
        assert "Qwen2.5-7B-Instruct" not in effective.models
        assert len(effective.models) == 2
