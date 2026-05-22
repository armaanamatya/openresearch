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
    _write_demo_status,
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
        assert len(tools) == 10  # 9 original + record_candidate_outcome (Task 13)
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

    def _plain_root_model(self):
        """A plain openai root model with no custom base_url (standard path)."""
        from backend.agents.rlm.models import RootModel
        return RootModel(
            key="gpt-5",
            rlm_backend="openai",
            backend_kwargs={"model_name": "gpt-5"},
            sub_backend="openai",
            sub_backend_kwargs={"model_name": "gpt-5-mini"},
        )

    def _anthropic_root_model(self):
        from backend.agents.rlm.models import RootModel
        return RootModel(
            key="claude",
            rlm_backend="anthropic",
            backend_kwargs={"model_name": "claude-opus-4-7"},
            sub_backend="anthropic",
            sub_backend_kwargs={"model_name": "claude-haiku-4-5-20251001"},
        )

    def test_openai_provider_builds_openai_client(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client, model = _build_llm_client("openai", self._plain_root_model())
        assert client.__class__.__name__ == "OpenAILlmClient"
        assert model == "gpt-4o-mini"

    def test_non_openai_provider_builds_claude_client(self):
        client, model = _build_llm_client("anthropic", self._anthropic_root_model())
        assert client.__class__.__name__ == "ClaudeLlmClient"
        assert model == "claude"

    def test_client_exposes_complete(self):
        client, _ = _build_llm_client("anthropic", self._anthropic_root_model())
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


# ---------------------------------------------------------------------------
# _write_demo_status
# ---------------------------------------------------------------------------

class TestWriteDemoStatus:
    """_write_demo_status writes a demo_status.json that GET /runs/{id} can read.

    Regression guard: an RLM run that never wrote demo_status.json 404'd on
    GET /runs/{id}. The contract here is that the file round-trips through
    live_runs.LiveRunState (the model the HTTP layer constructs from it), and
    that a later terminal write merges onto the earlier one — preserving
    startedAt rather than discarding it.
    """

    @staticmethod
    def _load(project_dir):
        import json
        return json.loads((project_dir / "demo_status.json").read_text(encoding="utf-8"))

    def test_running_status_round_trips_through_live_run_state(self, tmp_path):
        from backend.services.events.live_runs import LiveRunState

        _write_demo_status(tmp_path, "running")
        status = self._load(tmp_path)
        # The real contract: GET /runs/{id} does LiveRunState(**status).
        state = LiveRunState(**status)
        assert state.status == "running"
        assert state.runMode == "rlm"
        assert state.projectId == tmp_path.name
        assert state.outputDir == str(tmp_path)
        assert state.startedAt is not None

    def test_terminal_write_merges_and_preserves_started_at(self, tmp_path):
        from backend.services.events.live_runs import LiveRunState

        _write_demo_status(tmp_path, "running")
        started = self._load(tmp_path)["startedAt"]
        _write_demo_status(tmp_path, "completed")
        status = self._load(tmp_path)
        assert status["status"] == "completed"
        assert status["startedAt"] == started   # merged, not lost
        assert status["completedAt"] is not None
        LiveRunState(**status)  # still valid

    def test_failed_status_records_error(self, tmp_path):
        _write_demo_status(tmp_path, "failed", error="watchdog timeout")
        status = self._load(tmp_path)
        assert status["status"] == "failed"
        assert status["error"] == "watchdog timeout"


# ---------------------------------------------------------------------------
# T9 — deadline_utc integration guard
# ---------------------------------------------------------------------------

def test_run_context_deadline_is_armed_from_wall_clock(tmp_path, monkeypatch):
    """Symptom: ctx.remaining_s() always returns None; per-primitive deadlines never tighten.

    run.py's docstring documents three time bounds (rlm max_timeout, per-primitive
    deadlines, process watchdog) but the constructor never armed deadline_utc on
    RunContext — so _timeout_for(ctx, cap_s) always returned the static cap, never
    the run-wide remaining (review I1 / T9). Verify: with a wall_clock budget,
    the constructed RunContext.deadline_utc is set and remaining_s() returns a
    positive value less than the configured budget.
    """
    import asyncio

    from backend.agents.resilience.budget import RunBudget
    import backend.agents.rlm.run as run_mod

    captured: dict = {}

    original_resolve = run_mod._resolve_custom_tools

    def spy_resolve(ctx):
        captured["ctx"] = ctx
        return original_resolve(ctx)

    monkeypatch.setattr(run_mod, "_resolve_custom_tools", spy_resolve)

    # Replace RLM so we don't dispatch a real engine.
    class _FakeRLM:
        def __init__(self, **kwargs):
            pass

        def completion(self, *args, **kwargs):
            return type("R", (), {"response": "{}", "usage_summary": None, "metadata": {}})()

    monkeypatch.setattr(run_mod, "RLM", _FakeRLM)

    asyncio.run(run_mod.run_pipeline_rlm(
        project_id="test_t9",
        runs_root=tmp_path,
        workspace_claim_map={"entries": []},
        run_budget=RunBudget(max_wall_clock_seconds=300),
    ))

    assert "ctx" in captured, "_resolve_custom_tools was never called — hook point changed"
    ctx = captured["ctx"]
    assert ctx.deadline_utc is not None, "RunContext.deadline_utc was never set"
    remaining = ctx.remaining_s()
    assert remaining is not None, "remaining_s() returned None — deadline not armed"
    assert 0 < remaining <= 300, f"remaining_s() = {remaining}, expected in (0, 300]"


# ---------------------------------------------------------------------------
# T21 — stub run is honestly observable in final_report.json + demo_status.json
# ---------------------------------------------------------------------------

def test_stub_run_is_honestly_observable_in_artifacts(monkeypatch, tmp_path):
    """Symptom: a stub run is structurally indistinguishable from a real reproduction.

    Only a logger.info line signaled degradation (review I8 / T21); final_report.json
    and demo_status.json carried no marker. Verify: a stub run yields
    primitive_provider='stub', degraded=True, and verdict != 'reproduced' on disk.
    """
    import asyncio
    import json
    import backend.agents.rlm.run as run_mod

    monkeypatch.setenv("REPROLAB_RLM_STUB_PRIMITIVES", "1")

    class _FakeRLM:
        def __init__(self, **kwargs): ...
        def completion(self, *args, **kwargs):
            raw = json.dumps({
                "verdict": "reproduced",
                "baseline_metrics": {"x": 1.0},
                "rubric": {"overall_score": 0.5, "meets_target": False},
            })
            return type("R", (), {"response": raw, "usage_summary": None, "metadata": {}})()

    monkeypatch.setattr(run_mod, "RLM", _FakeRLM)

    asyncio.run(run_mod.run_pipeline_rlm(
        project_id="t21_stub",
        runs_root=tmp_path,
        workspace_claim_map={"entries": [{"title": "T", "excerpt": "x" * 600}]},
    ))

    report = json.loads((tmp_path / "t21_stub" / "final_report.json").read_text(encoding="utf-8"))
    assert report["primitive_provider"] == "stub", (
        f"expected primitive_provider='stub', got {report.get('primitive_provider')!r}"
    )
    assert report["degraded"] is True, "expected degraded=True in final_report.json"
    assert report["verdict"] != "reproduced", (
        f"stub run must not claim 'reproduced', got {report['verdict']!r}"
    )

    status = json.loads((tmp_path / "t21_stub" / "demo_status.json").read_text(encoding="utf-8"))
    assert status["primitiveProvider"] == "stub", (
        f"expected primitiveProvider='stub' in demo_status.json, got {status.get('primitiveProvider')!r}"
    )
