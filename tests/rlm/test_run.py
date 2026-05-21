"""Tests for backend.agents.rlm.run — the orchestrator entry helpers.

The full `run_pipeline_rlm` end-to-end path (a real `rlm.RLM` + stub primitives
+ a stub model backend) is covered by the Wave-C integration harness
(`test_run_integration.py`).  This file covers the pure helpers and contracts.
"""

from __future__ import annotations

from backend.agents.rlm.run import (
    RLMRunResult,
    _build_context,
    _build_llm_client,
    _context_metadata,
    _resolve_custom_tools,
    _verdict_to_status,
)


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------

class TestBuildContext:

    def test_assembles_paper_text_from_entries(self):
        claim_map = {
            "project_id": "prj",
            "entries": [
                {"source_id": "s1", "title": "Abstract", "excerpt": "We propose X."},
                {"source_id": "s2", "title": "Method", "excerpt": "We train Y."},
            ],
        }
        ctx = _build_context(claim_map)
        assert "We propose X." in ctx["paper_text"]
        assert "We train Y." in ctx["paper_text"]
        assert "## Abstract" in ctx["paper_text"]

    def test_metadata_lists_sections(self):
        claim_map = {
            "entries": [
                {"source_id": "s1", "title": "Abstract", "excerpt": "a"},
                {"source_id": "s2", "title": "Results", "excerpt": "b"},
            ]
        }
        ctx = _build_context(claim_map)
        assert ctx["paper_metadata"]["sections"] == ["Abstract", "Results"]
        assert ctx["paper_metadata"]["source_ids"] == ["s1", "s2"]
        assert ctx["paper_metadata"]["title"] == "Abstract"

    def test_required_context_keys_present(self):
        ctx = _build_context({"entries": []})
        for key in (
            "paper_text",
            "paper_metadata",
            "supplementary_text",
            "repo_files",
            "prior_work_refs",
            "rubric_spec",
        ):
            assert key in ctx

    def test_empty_claim_map_is_safe(self):
        ctx = _build_context({})
        assert ctx["paper_text"] == ""
        assert ctx["paper_metadata"]["sections"] == []
        assert ctx["prior_work_refs"] == []

    def test_rubric_spec_passed_through(self):
        ctx = _build_context({"entries": [], "rubric_spec": {"areas": ["a"]}})
        assert ctx["rubric_spec"] == {"areas": ["a"]}


# ---------------------------------------------------------------------------
# _context_metadata
# ---------------------------------------------------------------------------

class TestContextMetadata:

    def test_exposes_type_and_length_never_value(self):
        context = {"paper_text": "abcdef", "prior_work_refs": [1, 2, 3]}
        meta = _context_metadata(context)
        assert meta["paper_text"] == {"type": "str", "length": 6}
        assert meta["prior_work_refs"] == {"type": "list", "length": 3}
        # No value is ever surfaced.
        for entry in meta.values():
            assert set(entry.keys()) == {"type", "length"}

    def test_none_value_length_zero(self):
        meta = _context_metadata({"supplementary_text": None})
        assert meta["supplementary_text"]["type"] == "NoneType"
        assert meta["supplementary_text"]["length"] == 0


# ---------------------------------------------------------------------------
# _resolve_custom_tools
# ---------------------------------------------------------------------------

class TestResolveCustomTools:

    def test_forced_stub_returns_nine_tools(self, tmp_path, make_context, monkeypatch):
        monkeypatch.setenv("REPROLAB_RLM_STUB_PRIMITIVES", "1")
        ctx = make_context(tmp_path)
        tools, label = _resolve_custom_tools(ctx)
        assert len(tools) == 9
        assert "stub" in label
        for entry in tools.values():
            assert callable(entry["tool"])
            assert isinstance(entry["description"], str)

    def test_uses_real_binding_when_present(self, tmp_path, make_context, monkeypatch):
        """#59's binding.py is merged in — resolution binds the real primitive
        layer, not the stub. This is the post-#59-merge default path."""
        monkeypatch.delenv("REPROLAB_RLM_STUB_PRIMITIVES", raising=False)
        ctx = make_context(tmp_path)
        tools, label = _resolve_custom_tools(ctx)
        assert len(tools) == 9
        assert label == "real (#59 binding)"
        for entry in tools.values():
            assert callable(entry["tool"])
            assert isinstance(entry["description"], str)

    def test_falls_back_to_stub_when_binding_absent(self, tmp_path, make_context, monkeypatch):
        """binding.py not importable (e.g. #59 not yet merged) — resolution must
        degrade to the stub, not raise ImportError. #59's binding.py is present
        post-merge, so absence is simulated by blocking it in sys.modules."""
        import sys

        monkeypatch.delenv("REPROLAB_RLM_STUB_PRIMITIVES", raising=False)
        # None in sys.modules makes `import backend.agents.rlm.binding` raise ImportError.
        monkeypatch.setitem(sys.modules, "backend.agents.rlm.binding", None)

        ctx = make_context(tmp_path)
        tools, label = _resolve_custom_tools(ctx)
        assert len(tools) == 9
        assert "stub" in label
        assert "absent" in label

    def test_falls_back_to_stub_when_binding_incomplete(self, tmp_path, make_context, monkeypatch):
        """binding.py present but build_custom_tools raises (e.g. #59 mid-build —
        PRIMITIVE_DESCRIPTIONS not yet shipped). Resolution must degrade to the
        stub, loudly, not crash the run."""
        import sys
        import types

        monkeypatch.delenv("REPROLAB_RLM_STUB_PRIMITIVES", raising=False)

        fake_binding = types.ModuleType("backend.agents.rlm.binding")

        def _raising_build_custom_tools(ctx):
            raise AttributeError(
                "module 'backend.agents.rlm.primitives' has no attribute "
                "'PRIMITIVE_DESCRIPTIONS'"
            )

        fake_binding.build_custom_tools = _raising_build_custom_tools
        monkeypatch.setitem(sys.modules, "backend.agents.rlm.binding", fake_binding)

        ctx = make_context(tmp_path)
        tools, label = _resolve_custom_tools(ctx)
        assert len(tools) == 9
        assert "stub" in label
        assert "incomplete" in label


# ---------------------------------------------------------------------------
# _build_llm_client
# ---------------------------------------------------------------------------

class TestBuildLlmClient:

    def test_openai_provider_builds_openai_client(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client, model = _build_llm_client("openai")
        assert client.__class__.__name__ == "OpenAILlmClient"
        assert model == "gpt-4o-mini"

    def test_non_openai_provider_builds_claude_client(self):
        client, model = _build_llm_client("anthropic")
        assert client.__class__.__name__ == "ClaudeLlmClient"
        assert model == "claude"

    def test_client_exposes_complete(self):
        client, _ = _build_llm_client("anthropic")
        assert hasattr(client, "complete")


# ---------------------------------------------------------------------------
# _verdict_to_status + RLMRunResult
# ---------------------------------------------------------------------------

class TestVerdictAndResult:

    def test_reproduced_maps_to_completed(self):
        assert _verdict_to_status("reproduced") == "completed"

    def test_partial_and_failed_pass_through(self):
        assert _verdict_to_status("partial") == "partial"
        assert _verdict_to_status("failed") == "failed"

    def test_run_result_fields(self):
        result = RLMRunResult(
            project_id="prj",
            status="completed",
            iterations=7,
            rubric_score=0.8,
            cost_usd=0.05,
            final_report_path="/runs/prj/final_report.json",
        )
        assert result.project_id == "prj"
        assert result.iterations == 7
        assert result.status == "completed"
