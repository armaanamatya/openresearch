"""Tests for arXiv ID propagation through RunContext.

P0 fix: arXiv-sourced runs get hashed project IDs (`prj_<digest>`) so the
regex `_extract_arxiv_id` in baseline_implementation.py never fires for them.
The fix threads `arxiv_id` through `RunContext` (read from
`artifact_index.json` / `demo_status.json` / URL-in-metadata), and prefers
`ctx.arxiv_id` over the fallback regex when calling
`_compute_constraint_guidance`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# 1. RunContext carries arxiv_id field
# ---------------------------------------------------------------------------

class TestRunContextArxivIdField:
    """RunContext MUST have `arxiv_id: str | None` defaulting to None."""

    def test_runcontext_has_arxiv_id_field(self):
        import dataclasses
        from backend.agents.rlm.context import RunContext
        names = {f.name for f in dataclasses.fields(RunContext)}
        assert "arxiv_id" in names, (
            "RunContext.arxiv_id field is required for per-paper YAML override "
            "routing.  Without it, docs/papers/<id>.yaml is dead code for every "
            "hashed-project-id run."
        )

    def test_arxiv_id_default_is_none(self):
        from backend.agents.rlm.context import RunContext
        ctx = RunContext(
            project_id="prj_09047604e591d969",
            project_dir=Path("/tmp/test"),
            runs_root=Path("/tmp"),
            dashboard=None,
            cost_ledger=None,
            llm_client=None,
            provider="anthropic",
            model="claude-sonnet-4-6",
        )
        assert ctx.arxiv_id is None


# ---------------------------------------------------------------------------
# 2. arxiv_id extracted from artifact_index.json
# ---------------------------------------------------------------------------

class TestArxivIdExtractedFromArtifactIndex:
    """run_pipeline_rlm reads artifact_index.json → paper.arxiv_id."""

    def test_arxiv_id_from_artifact_index(self, tmp_path):
        """When artifact_index.json has paper.arxiv_id, ctx.arxiv_id is set."""
        from backend.agents.rlm.run import _extract_arxiv_id_from_project_dir

        project_dir = tmp_path / "prj_09047604e591d969"
        project_dir.mkdir(parents=True)
        artifact_index = {
            "paper": {
                "title": "SDAR",
                "arxiv_id": "2605.15155",
            }
        }
        (project_dir / "artifact_index.json").write_text(
            json.dumps(artifact_index), encoding="utf-8"
        )
        result = _extract_arxiv_id_from_project_dir(project_dir)
        assert result == "2605.15155"

    def test_arxiv_id_from_artifact_index_wins_over_demo_status(self, tmp_path):
        """artifact_index.json is preferred over demo_status.json."""
        from backend.agents.rlm.run import _extract_arxiv_id_from_project_dir

        project_dir = tmp_path / "prj_09047604e591d969"
        project_dir.mkdir(parents=True)
        # Both files present
        artifact_index = {"paper": {"arxiv_id": "2605.15155"}}
        (project_dir / "artifact_index.json").write_text(
            json.dumps(artifact_index), encoding="utf-8"
        )
        demo_status = {"sourceLabel": "arxiv_9999.11111.pdf"}
        (project_dir / "demo_status.json").write_text(
            json.dumps(demo_status), encoding="utf-8"
        )
        result = _extract_arxiv_id_from_project_dir(project_dir)
        assert result == "2605.15155"


# ---------------------------------------------------------------------------
# 3. arxiv_id extracted from demo_status.json sourceLabel
# ---------------------------------------------------------------------------

class TestArxivIdExtractedFromDemoStatus:
    """Fall back to demo_status.json → sourceLabel when no artifact_index."""

    def test_arxiv_id_from_source_label(self, tmp_path):
        from backend.agents.rlm.run import _extract_arxiv_id_from_project_dir

        project_dir = tmp_path / "prj_hash"
        project_dir.mkdir(parents=True)
        demo_status = {
            "sourceLabel": "arxiv_2604.01733.pdf",
            "projectId": "prj_hash",
        }
        (project_dir / "demo_status.json").write_text(
            json.dumps(demo_status), encoding="utf-8"
        )
        result = _extract_arxiv_id_from_project_dir(project_dir)
        assert result == "2604.01733"

    def test_arxiv_id_from_source_url_in_demo_status(self, tmp_path):
        """sourceUrl field with arxiv.org URL is also parsed."""
        from backend.agents.rlm.run import _extract_arxiv_id_from_project_dir

        project_dir = tmp_path / "prj_hash2"
        project_dir.mkdir(parents=True)
        demo_status = {
            "sourceUrl": "https://arxiv.org/abs/2605.15155",
            "projectId": "prj_hash2",
        }
        (project_dir / "demo_status.json").write_text(
            json.dumps(demo_status), encoding="utf-8"
        )
        result = _extract_arxiv_id_from_project_dir(project_dir)
        assert result == "2605.15155"

    def test_no_json_files_returns_none(self, tmp_path):
        """Empty project dir → None (no crash)."""
        from backend.agents.rlm.run import _extract_arxiv_id_from_project_dir

        project_dir = tmp_path / "prj_empty"
        project_dir.mkdir(parents=True)
        result = _extract_arxiv_id_from_project_dir(project_dir)
        assert result is None

    def test_non_arxiv_source_label_returns_none(self, tmp_path):
        """PDF uploads with no arxiv ID in the label → None, no crash."""
        from backend.agents.rlm.run import _extract_arxiv_id_from_project_dir

        project_dir = tmp_path / "prj_pdf"
        project_dir.mkdir(parents=True)
        demo_status = {
            "sourceLabel": "my_custom_paper.pdf",
            "sourceKind": "uploaded_pdf",
        }
        (project_dir / "demo_status.json").write_text(
            json.dumps(demo_status), encoding="utf-8"
        )
        result = _extract_arxiv_id_from_project_dir(project_dir)
        assert result is None


# ---------------------------------------------------------------------------
# 4. _load_paper_override uses ctx.arxiv_id via _compute_constraint_guidance
# ---------------------------------------------------------------------------

class TestLoadPaperOverrideViaCTXArxivId:
    """_compute_constraint_guidance uses the arxiv_id kwarg for the override."""

    def test_load_paper_override_uses_explicit_arxiv_id(self, tmp_path, monkeypatch):
        """Pass arxiv_id='9999.99999' directly; yaml is found and surfaced."""
        from backend.agents import baseline_implementation as bi

        monkeypatch.setattr(bi, "_REPO_ROOT", tmp_path)
        yaml_dir = tmp_path / "docs" / "papers"
        yaml_dir.mkdir(parents=True)
        (yaml_dir / "9999.99999.yaml").write_text(
            "algorithm: my_special_algo\nstep: test_step\n", encoding="utf-8"
        )

        result = bi._compute_constraint_guidance(
            "docker", None,
            project_dir=None,
            arxiv_id="9999.99999",
        )
        assert "PAPER-SPECIFIC GUIDANCE" in result
        assert "9999.99999" in result
        assert "my_special_algo" in result

    def test_load_paper_override_no_yaml_no_crash(self, tmp_path, monkeypatch):
        """arxiv_id for a paper with no yaml → empty override, no crash."""
        from backend.agents import baseline_implementation as bi

        monkeypatch.setattr(bi, "_REPO_ROOT", tmp_path)
        # No yaml in docs/papers/ — make sure the dir doesn't exist
        result = bi._compute_constraint_guidance(
            "docker", None,
            project_dir=None,
            arxiv_id="0000.00001",
        )
        assert "PAPER-SPECIFIC GUIDANCE" not in result

    def test_sdar_yaml_surfaces_via_explicit_arxiv_id(self, tmp_path, monkeypatch):
        """The real 2605.15155.yaml is surfaced when arxiv_id='2605.15155'."""
        from backend.agents.baseline_implementation import _load_paper_override
        # Uses the real repo root (not tmp_path), so the actual yaml is loaded
        result = _load_paper_override("2605.15155")
        assert result != ""
        assert "PAPER-SPECIFIC GUIDANCE" in result
        assert "2605.15155" in result


# ---------------------------------------------------------------------------
# 5. _extract_arxiv_id fallback still works for legacy non-hashed project IDs
# ---------------------------------------------------------------------------

class TestExtractArxivIdFallbackStillWired:
    """The regex fallback must still work for project IDs that embed the ID."""

    def test_bare_arxiv_id_in_project_id(self):
        from backend.agents.baseline_implementation import _extract_arxiv_id
        assert _extract_arxiv_id("2605.15155") == "2605.15155"

    def test_prefixed_arxiv_id_in_project_id(self):
        from backend.agents.baseline_implementation import _extract_arxiv_id
        assert _extract_arxiv_id("arXiv_2605.15155_abc123") == "2605.15155"

    def test_hashed_prj_id_returns_none(self):
        from backend.agents.baseline_implementation import _extract_arxiv_id
        # Hashed project IDs produce no match — this is the bug case
        assert _extract_arxiv_id("prj_09047604e591d969") is None

    def test_empty_project_id_returns_none(self):
        from backend.agents.baseline_implementation import _extract_arxiv_id
        assert _extract_arxiv_id("") is None

    def test_five_digit_suffix_arxiv_id(self):
        from backend.agents.baseline_implementation import _extract_arxiv_id
        assert _extract_arxiv_id("1706.03762") == "1706.03762"

    def test_run_with_sdk_prefers_ctx_arxiv_id_over_regex(
        self, tmp_path, monkeypatch
    ):
        """When run_with_sdk is called with a hashed project_id but an
        explicit arxiv_id kwarg is threaded through, the override yaml must
        appear in the prompt (proving ctx.arxiv_id won over the fallback
        regex which would have returned None for a hashed id)."""
        import asyncio
        from backend.agents import baseline_implementation as bi
        from backend.agents.schemas import EnvironmentSpec, PaperClaimMap

        monkeypatch.setattr(bi, "_REPO_ROOT", tmp_path)
        yaml_dir = tmp_path / "docs" / "papers"
        yaml_dir.mkdir(parents=True)
        (yaml_dir / "2605.15155.yaml").write_text(
            "algorithm: SDAR_sentinel_value\n", encoding="utf-8"
        )

        captured: list[dict] = []

        async def _fake_collect(agent_name, prompt, **kwargs):
            captured.append({"prompt": prompt})
            return ""

        monkeypatch.setattr(
            "backend.agents.runtime.invoke.collect_agent_text",
            _fake_collect,
        )

        runs_root = tmp_path / "runs"
        runs_root.mkdir(parents=True)
        project_dir = runs_root / "prj_09047604e591d969"
        project_dir.mkdir(parents=True)
        (project_dir / "code").mkdir(parents=True)

        pcm = PaperClaimMap(core_contribution="SDAR")
        env = EnvironmentSpec(dockerfile="FROM python:3.11", framework="pytorch")

        asyncio.run(bi.run_with_sdk(
            "prj_09047604e591d969",  # hashed — regex returns None
            runs_root,
            pcm,
            env,
            None,
            arxiv_id="2605.15155",   # explicit id
            sandbox_mode="docker",
            gpu_mode=None,
        ))

        assert captured, "collect_agent_text was not called"
        prompt = captured[0]["prompt"]
        assert "SDAR_sentinel_value" in prompt, (
            "run_with_sdk must prefer the explicit arxiv_id kwarg over "
            "_extract_arxiv_id(project_id) for the paper-override lookup"
        )
