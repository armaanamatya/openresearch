"""Tests for the CLI hybrid/rlm-pure/rdr mode routing introduced by the
hybrid RDR+RLM orchestration feature.

All external I/O (RDR, RLM, hybrid runners) is monkeypatched so no network
or LLM calls are made.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _namespace(**kwargs: Any) -> Namespace:
    """Build a minimal argparse Namespace for reproduce tests."""
    defaults = {
        "mode": "rlm",
        "source": "sequential-neural-score-estimation",
        "source_kind": "auto",
        "model": None,
        "provider": None,
        "sandbox": "local",
        "runs_root": "runs",
        "database_url": "sqlite:///test.db",
        "execution_mode": "efficient",
        "gpu_mode": "auto",
        "max_usd": None,
        "max_wall_clock": None,
        "seed": None,
        "attempt_id": None,
        "run_group_id": None,
        "project_id": "test_proj",
        "verification_provider": None,
        "hints": None,
        "agent": "default",
        "command_timeout": None,
        "allow_sandbox_network": False,
        "sandbox_platform": None,
        "sandbox_memory": None,
        "sandbox_cpus": None,
        "max_pod_seconds": None,
        "max_invocations": None,
        "blacklist": None,
        "n_paths": 3,
        "fresh": False,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


# ---------------------------------------------------------------------------
# Mode-routing tests: _cmd_reproduce_rlm_paperbench
# ---------------------------------------------------------------------------


class TestCmdReproduceRlmPaperbenchRouting:
    """_cmd_reproduce_rlm_paperbench routes by mode to hybrid or rlm-pure."""

    def _patch_bundle_load(self, monkeypatch):
        """Patch bundle loading so we don't need the filesystem."""
        fake_bundle = MagicMock()
        fake_bundle.rubric.return_value = {"id": "root", "sub_tasks": []}
        monkeypatch.setattr(
            "backend.cli.load_paperbench_bundle",
            lambda root, paper_id: fake_bundle,
            raising=False,
        )
        monkeypatch.setattr(
            "backend.evals.paperbench.bundle.load_paperbench_bundle",
            lambda *a, **kw: fake_bundle,
            raising=False,
        )
        return fake_bundle

    def test_default_mode_routes_to_hybrid(self, tmp_path: Path, monkeypatch) -> None:
        """--mode rlm (default) dispatches to run_pipeline_hybrid, NOT run_pipeline_rlm."""
        from backend.cli import _cmd_reproduce_rlm_paperbench

        dispatched_hybrid: list[Any] = []
        dispatched_rlm: list[Any] = []

        async def _fake_hybrid(*args, **kwargs):
            from backend.agents.rlm.run import RLMRunResult
            dispatched_hybrid.append(kwargs)
            return RLMRunResult(
                project_id="test", status="completed",
                iterations=2, rubric_score=0.8, cost_usd=0.01,
                final_report_path=None,
            )

        async def _fake_rlm(*args, **kwargs):
            dispatched_rlm.append(kwargs)
            from backend.agents.rlm.run import RLMRunResult
            return RLMRunResult(
                project_id="test", status="completed",
                iterations=1, rubric_score=0.7, cost_usd=0.01,
                final_report_path=None,
            )

        monkeypatch.setattr(
            "backend.agents.hybrid.controller.run_pipeline_hybrid",
            _fake_hybrid,
        )
        monkeypatch.setattr(
            "backend.agents.rlm.run.run_pipeline_rlm",
            _fake_rlm,
        )

        # Patch the heavy setup helpers
        monkeypatch.setattr(
            "backend.cli.load_paperbench_bundle",
            lambda *a, **kw: MagicMock(rubric=lambda: {}),
            raising=False,
        )
        monkeypatch.setattr(
            "backend.cli.bundle_to_workspace_claim_map",
            lambda b: {"project_id": "test", "entries": [], "paperbench": {}},
            raising=False,
        )
        monkeypatch.setattr(
            "backend.cli.ensure_sandbox_mode_available",
            lambda mode: None,
            raising=False,
        )

        args = _namespace(mode="rlm", project_id="test_hybrid")
        _cmd_reproduce_rlm_paperbench(args, tmp_path)

        assert len(dispatched_hybrid) == 1, "hybrid should have been called exactly once"
        assert len(dispatched_rlm) == 0, "pure rlm must NOT be called when mode=rlm"

    def test_rlm_pure_routes_to_run_pipeline_rlm(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """--mode rlm-pure dispatches to run_pipeline_rlm, NOT run_pipeline_hybrid."""
        from backend.cli import _cmd_reproduce_rlm_paperbench

        dispatched_hybrid: list[Any] = []
        dispatched_rlm: list[Any] = []

        async def _fake_hybrid(*args, **kwargs):
            dispatched_hybrid.append(kwargs)
            from backend.agents.rlm.run import RLMRunResult
            return RLMRunResult(
                project_id="test", status="completed",
                iterations=1, rubric_score=0.7, cost_usd=0.01,
                final_report_path=None,
            )

        async def _fake_rlm(*args, **kwargs):
            dispatched_rlm.append(kwargs)
            from backend.agents.rlm.run import RLMRunResult
            return RLMRunResult(
                project_id="test", status="completed",
                iterations=1, rubric_score=0.7, cost_usd=0.01,
                final_report_path=None,
            )

        monkeypatch.setattr(
            "backend.agents.hybrid.controller.run_pipeline_hybrid",
            _fake_hybrid,
        )
        monkeypatch.setattr(
            "backend.agents.rlm.run.run_pipeline_rlm",
            _fake_rlm,
        )
        monkeypatch.setattr(
            "backend.cli.load_paperbench_bundle",
            lambda *a, **kw: MagicMock(rubric=lambda: {}),
            raising=False,
        )
        monkeypatch.setattr(
            "backend.cli.bundle_to_workspace_claim_map",
            lambda b: {"project_id": "test", "entries": [], "paperbench": {}},
            raising=False,
        )
        monkeypatch.setattr(
            "backend.cli.ensure_sandbox_mode_available",
            lambda mode: None,
            raising=False,
        )

        args = _namespace(mode="rlm-pure", project_id="test_rlm_pure")
        _cmd_reproduce_rlm_paperbench(args, tmp_path)

        assert len(dispatched_rlm) == 1, "pure rlm should have been called exactly once"
        assert len(dispatched_hybrid) == 0, "hybrid must NOT be called when mode=rlm-pure"


class TestCmdReproduceRdrRouting:
    """cmd_reproduce routes --mode rdr to _cmd_reproduce_rdr."""

    def test_rdr_routes_to_rdr_controller(self, tmp_path: Path, monkeypatch) -> None:
        """--mode rdr dispatches to _cmd_reproduce_rdr, not hybrid."""
        from backend.cli import cmd_reproduce

        dispatched_rdr: list[Any] = []
        dispatched_hybrid: list[Any] = []

        def _fake_rdr(args, runs_root):
            dispatched_rdr.append((args, runs_root))
            return 0

        def _fake_hybrid(args, runs_root):
            dispatched_hybrid.append((args, runs_root))
            return 0

        monkeypatch.setattr("backend.cli._cmd_reproduce_rdr", _fake_rdr)
        monkeypatch.setattr(
            "backend.cli._cmd_reproduce_rlm_paperbench", _fake_hybrid
        )
        monkeypatch.setattr(
            "backend.cli.configure_root_logger", lambda: None, raising=False
        )

        args = _namespace(mode="rdr", runs_root=str(tmp_path))
        cmd_reproduce(args)

        assert len(dispatched_rdr) == 1
        assert len(dispatched_hybrid) == 0


class TestArgparseChoices:
    """Argparse rejects unknown modes and accepts all three valid modes."""

    def test_valid_modes_accepted(self) -> None:
        from backend.cli import _build_parser

        parser = _build_parser()
        for mode in ("rlm", "rdr", "rlm-pure"):
            ns = parser.parse_args(["reproduce", "dummy-source", "--mode", mode])
            assert ns.mode == mode

    def test_unknown_mode_rejected(self) -> None:
        import sys
        from io import StringIO
        from backend.cli import _build_parser

        parser = _build_parser()
        # argparse raises SystemExit on invalid choice
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            parser.parse_args(["reproduce", "dummy-source", "--mode", "invalid"])
        except SystemExit as exc:
            assert exc.code != 0
        finally:
            sys.stderr = old_stderr
