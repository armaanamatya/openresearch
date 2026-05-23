"""Regression tests for `--mode rlm` with a PaperBench bundle ID as source.

Covers the fix that adds `_is_paperbench_bundle_id` + `_cmd_reproduce_rlm_paperbench`
to `backend/cli.py`, allowing:

    python -m backend.cli reproduce sequential-neural-score-estimation \\
        --mode rlm --model claude-oauth

without needing an arXiv ID or PDF path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch


class TestIsPaperbenchBundleId:
    """_is_paperbench_bundle_id returns True only when a bundle directory exists."""

    def test_real_bundle_id_returns_true(self):
        from backend.cli import _is_paperbench_bundle_id

        # The vendored bundle exists in the repo tree.
        assert _is_paperbench_bundle_id("sequential-neural-score-estimation", Path("runs")) is True

    def test_arxiv_id_returns_false(self):
        from backend.cli import _is_paperbench_bundle_id

        assert _is_paperbench_bundle_id("2209.04739", Path("runs")) is False

    def test_doi_returns_false(self):
        from backend.cli import _is_paperbench_bundle_id

        assert _is_paperbench_bundle_id("10.1145/1234567.1234568", Path("runs")) is False

    def test_nonexistent_bundle_returns_false(self):
        from backend.cli import _is_paperbench_bundle_id

        assert _is_paperbench_bundle_id("this-paper-does-not-exist-xyz", Path("runs")) is False

    def test_absolute_pdf_path_returns_false(self, tmp_path):
        """An absolute path that happens to exist but is not in third_party/paperbench/."""
        from backend.cli import _is_paperbench_bundle_id

        fake_pdf = str(tmp_path / "paper.pdf")
        assert _is_paperbench_bundle_id(fake_pdf, tmp_path) is False


class TestCmdReproduceRlmPaperbenchDispatch:
    """cmd_reproduce routes to _cmd_reproduce_rlm_paperbench for bundle IDs with --mode rlm."""

    def test_paperbench_source_with_mode_rlm_dispatches_to_rlm_paperbench(
        self, tmp_path, monkeypatch
    ):
        """A PaperBench bundle ID with --mode rlm should bypass ingest and call run_pipeline_rlm."""
        from argparse import Namespace
        from backend.cli import _is_paperbench_bundle_id

        # Patch _cmd_reproduce_rlm_paperbench to capture the call.
        dispatched: list[Any] = []

        def _fake_rlm_paperbench(args, runs_root):
            dispatched.append({"args": args, "runs_root": runs_root})
            return 0

        monkeypatch.setattr(
            "backend.cli._cmd_reproduce_rlm_paperbench",
            _fake_rlm_paperbench,
        )
        # Patch _is_paperbench_bundle_id so we don't depend on the real filesystem in this unit test.
        monkeypatch.setattr(
            "backend.cli._is_paperbench_bundle_id",
            lambda source, runs_root: source == "sequential-neural-score-estimation",
        )
        # Patch configure_root_logger to no-op.
        monkeypatch.setattr(
            "backend.observability.run_logging.configure_root_logger",
            lambda: None,
        )

        from backend.cli import cmd_reproduce

        args = Namespace(
            source="sequential-neural-score-estimation",
            mode="rlm",
            runs_root=str(tmp_path),
            model="claude-oauth",
            project_id=None,
            sandbox="local",
            provider=None,
            verification_provider=None,
            execution_mode="efficient",
            gpu_mode="auto",
            max_usd=None,
            max_wall_clock=None,
            allow_sandbox_network=False,
            sandbox_platform=None,
            sandbox_memory=None,
            sandbox_cpus=None,
            command_timeout=None,
            max_invocations=None,
            seed=None,
            attempt_id=None,
            run_group_id=None,
            blacklist=None,
            hints=None,
            n_paths=3,
            agent="default",
            fresh=False,
            resume=False,
            max_repair_iterations=2,
            repair_target=0.6,
            database_url="sqlite:///:memory:",
        )
        result = cmd_reproduce(args)
        assert result == 0
        assert len(dispatched) == 1
        assert dispatched[0]["args"].source == "sequential-neural-score-estimation"

    def test_non_bundle_arxiv_id_does_not_dispatch_to_rlm_paperbench(
        self, monkeypatch
    ):
        """An arXiv ID should NOT be routed to _cmd_reproduce_rlm_paperbench."""
        from backend.cli import _is_paperbench_bundle_id

        result = _is_paperbench_bundle_id("2209.04739", Path("runs"))
        assert result is False, (
            "arXiv ID '2209.04739' must NOT match as a PaperBench bundle ID"
        )


class TestWorkspaceClaimMapHasRubricSpec:
    """_cmd_reproduce_rlm_paperbench sets rubric_spec on the workspace claim map."""

    def test_workspace_claim_map_includes_rubric_spec(self, tmp_path, monkeypatch):
        """The workspace claim map passed to the runner must include rubric_spec
        from the bundle so run_pipeline_rlm skips the rubric-gen LLM call.

        Patches run_pipeline_hybrid (the default --mode rlm path since the hybrid
        controller was wired in) to capture the claim map without making any real
        API calls.  The test validates that _cmd_reproduce_rlm_paperbench populates
        rubric_spec on the claim map before handing off to the runner.
        """
        from argparse import Namespace

        captured_wcm: list[dict] = []

        async def _fake_run_pipeline_hybrid(project_id, runs_root, workspace_claim_map, **kwargs):
            captured_wcm.append(workspace_claim_map)
            # Return a minimal RLMRunResult-like object.
            from backend.agents.rlm.run import RLMRunResult
            return RLMRunResult(
                project_id=project_id,
                status="completed",
                iterations=1,
                rubric_score=0.5,
                cost_usd=0.0,
                final_report_path=str(tmp_path / project_id / "final_report.json"),
            )

        monkeypatch.setattr(
            "backend.agents.hybrid.controller.run_pipeline_hybrid",
            _fake_run_pipeline_hybrid,
        )
        # Patch ensure_sandbox_mode_available to no-op.
        monkeypatch.setattr(
            "backend.agents.execution.ensure_sandbox_mode_available",
            lambda mode: None,
        )

        from backend.cli import _cmd_reproduce_rlm_paperbench

        args = Namespace(
            source="sequential-neural-score-estimation",
            mode="rlm",
            runs_root=str(tmp_path),
            model="claude-oauth",
            project_id="test_wcm_smoke",
            sandbox="local",
            provider=None,
            execution_mode="efficient",
            gpu_mode="auto",
            max_usd=None,
            max_wall_clock=None,
            allow_sandbox_network=False,
            sandbox_platform=None,
            sandbox_memory=None,
            sandbox_cpus=None,
            command_timeout=None,
            seed=None,
            attempt_id=None,
            run_group_id=None,
        )
        # Create a minimal run directory so the result path exists.
        (tmp_path / "test_wcm_smoke").mkdir(parents=True, exist_ok=True)
        (tmp_path / "test_wcm_smoke" / "final_report.json").write_text("{}", encoding="utf-8")

        _cmd_reproduce_rlm_paperbench(args, tmp_path)

        assert len(captured_wcm) == 1, "run_pipeline_hybrid must have been called exactly once"
        wcm = captured_wcm[0]
        assert "rubric_spec" in wcm, (
            "workspace_claim_map must include 'rubric_spec' from the bundle "
            "so run_pipeline_rlm skips the rubric-gen LLM call"
        )
        assert isinstance(wcm["rubric_spec"], dict), "rubric_spec must be a dict (the bundle rubric)"
        assert wcm["rubric_spec"], "rubric_spec must be non-empty"
