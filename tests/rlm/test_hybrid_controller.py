"""Tests for the hybrid RDR+RLM controller
(``backend/agents/hybrid/controller.py``).

All LLM / RDR / RLM I/O is monkeypatched — no network, API keys, or Docker.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import dataclasses

from backend.agents.hybrid.controller import (
    _extract_weak_clusters,
    run_pipeline_hybrid,
)
from backend.agents.rdr.models import RdrResult
from backend.agents.rlm.run import RLMRunResult


# ---------------------------------------------------------------------------
# Helpers — synthetic Phase 1 / Phase 2 results
# ---------------------------------------------------------------------------


def _make_rdr_result(
    project_id: str,
    rubric_score: float,
    final_report_path: str | None = None,
    clusters_total: int = 3,
    clusters_failed: int = 0,
    status: str = "completed",
) -> RdrResult:
    return RdrResult(
        project_id=project_id,
        status=status,
        rubric_score=rubric_score,
        clusters_total=clusters_total,
        clusters_failed=clusters_failed,
        repair_iterations=0,
        final_report_path=final_report_path,
        cost_usd=None,
    )


def _make_rlm_result(project_id: str, status: str = "completed") -> RLMRunResult:
    return RLMRunResult(
        project_id=project_id,
        status=status,
        iterations=3,
        rubric_score=0.9,
        cost_usd=0.05,
        final_report_path=f"runs/{project_id}/final_report.json",
    )


def _write_report(path: Path, leaf_scores: list[dict]) -> None:
    """Write a minimal final_report.json with the given leaf_scores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "verdict": "partial",
        "rubric": {
            "overall_score": sum(e["score"] for e in leaf_scores) / max(len(leaf_scores), 1),
            "leaf_scores": leaf_scores,
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# _extract_weak_clusters unit tests
# ---------------------------------------------------------------------------


class TestExtractWeakClusters:
    def test_returns_weak_leaves_below_target(self, tmp_path: Path) -> None:
        report = tmp_path / "final_report.json"
        _write_report(report, [
            {"id": "l1", "score": 0.3, "justification": "missing X"},
            {"id": "l2", "score": 0.9, "justification": "correct"},
        ])
        weak = _extract_weak_clusters(str(report), repair_target=0.6)
        assert len(weak) == 1
        assert weak[0]["id"] == "l1"
        assert weak[0]["score"] == 0.3

    def test_empty_when_all_pass(self, tmp_path: Path) -> None:
        report = tmp_path / "final_report.json"
        _write_report(report, [
            {"id": "l1", "score": 0.7},
            {"id": "l2", "score": 1.0},
        ])
        assert _extract_weak_clusters(str(report), repair_target=0.6) == []

    def test_empty_on_missing_file(self, tmp_path: Path) -> None:
        result = _extract_weak_clusters(str(tmp_path / "nonexistent.json"), 0.6)
        assert result == []

    def test_empty_on_corrupt_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        assert _extract_weak_clusters(str(bad), 0.6) == []


# ---------------------------------------------------------------------------
# run_pipeline_hybrid integration tests
# ---------------------------------------------------------------------------


class TestRunPipelineHybrid:

    @pytest.fixture
    def claim_map(self) -> dict:
        return {
            "project_id": "hybrid_test",
            "paperbench": {"paper_id": "sequential-neural-score-estimation"},
            "entries": [],
        }

    # ------------------------------------------------------------------
    # Test: Phase 1 only when all leaves meet target
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_phase1_only_when_target_met(self, tmp_path: Path, claim_map: dict) -> None:
        """When all leaf scores are >= repair_target, Phase 2 (RLM) must NOT be called."""
        project_id = "hybrid_target_met"
        report_path = tmp_path / project_id / "final_report.json"
        _write_report(report_path, [
            {"id": "l1", "score": 0.8},
            {"id": "l2", "score": 0.9},
        ])

        rdr_result = _make_rdr_result(
            project_id, rubric_score=0.85, final_report_path=str(report_path)
        )

        mock_rdr = AsyncMock(return_value=rdr_result)
        mock_rlm = AsyncMock()  # should never be called

        result = await run_pipeline_hybrid(
            project_id, tmp_path, claim_map,
            repair_target=0.6,
            _rdr_runner=mock_rdr,
            _rlm_runner=mock_rlm,
        )

        mock_rdr.assert_awaited_once()
        mock_rlm.assert_not_awaited()  # Phase 2 must be skipped

        assert result.project_id == project_id
        assert result.rubric_score == pytest.approx(0.85)

    # ------------------------------------------------------------------
    # Test: Phase 2 is called when weak clusters exist
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_phase2_runs_when_weak_clusters_exist(
        self, tmp_path: Path, claim_map: dict
    ) -> None:
        """When weak clusters exist, run_pipeline_rlm must be called with
        hybrid_repair_only=True and phase1_weak_clusters as explicit kwargs."""
        project_id = "hybrid_needs_repair"
        report_path = tmp_path / project_id / "final_report.json"
        _write_report(report_path, [
            {"id": "l1", "score": 0.2, "justification": "train loop broken"},
            {"id": "l2", "score": 0.9},
        ])

        rdr_result = _make_rdr_result(
            project_id, rubric_score=0.55, final_report_path=str(report_path)
        )
        rlm_result = _make_rlm_result(project_id)

        received_kwargs: list[dict] = []

        async def _capture_rlm(pid, root, wcm, **kw):  # noqa: ANN001
            received_kwargs.append(kw)
            return rlm_result

        mock_rdr = AsyncMock(return_value=rdr_result)

        result = await run_pipeline_hybrid(
            project_id, tmp_path, claim_map,
            repair_target=0.6,
            _rdr_runner=mock_rdr,
            _rlm_runner=_capture_rlm,
        )

        mock_rdr.assert_awaited_once()
        assert len(received_kwargs) == 1

        # Phase 2 must receive the repair flag as an explicit kwarg.
        kw = received_kwargs[0]
        assert kw.get("hybrid_repair_only") is True

        # Phase 2 must know which clusters are weak via explicit kwarg.
        weak = kw.get("phase1_weak_clusters", [])
        assert len(weak) == 1
        assert weak[0]["id"] == "l1"

        # Final result is Phase 2's result.
        assert result.status == "completed"
        assert result.iterations == 3

    @pytest.mark.asyncio
    async def test_phase2_null_score_restores_phase1_report(
        self, tmp_path: Path, claim_map: dict
    ) -> None:
        """A failed repair pass must not erase Phase 1's scored report."""
        project_id = "hybrid_restore_phase1"
        report_path = tmp_path / project_id / "final_report.json"
        _write_report(report_path, [
            {"id": "l1", "score": 0.0, "justification": "metricless"},
            {"id": "l2", "score": 0.2, "justification": "partial"},
        ])
        phase1_json = report_path.read_text(encoding="utf-8")
        report_path.with_suffix(".md").write_text("# Phase 1 report\n", encoding="utf-8")

        rdr_result = _make_rdr_result(
            project_id,
            rubric_score=0.1,
            final_report_path=str(report_path),
            clusters_total=2,
            clusters_failed=1,
            status="partial",
        )

        async def _failing_rlm(pid, root, wcm, **kw):  # noqa: ANN001
            report_path.write_text(
                json.dumps({"rubric": {"overall_score": None}, "mode": "rlm"}),
                encoding="utf-8",
            )
            report_path.with_suffix(".md").write_text("# Null Phase 2 report\n", encoding="utf-8")
            return RLMRunResult(
                project_id=project_id,
                status="failed",
                iterations=0,
                rubric_score=None,
                cost_usd=0.0,
                final_report_path=str(report_path),
            )

        result = await run_pipeline_hybrid(
            project_id, tmp_path, claim_map,
            repair_target=0.6,
            _rdr_runner=AsyncMock(return_value=rdr_result),
            _rlm_runner=_failing_rlm,
        )

        assert result.status == "failed"
        assert result.rubric_score == pytest.approx(0.1)
        assert report_path.read_text(encoding="utf-8") == phase1_json
        assert report_path.with_suffix(".md").read_text(encoding="utf-8") == "# Phase 1 report\n"

    @pytest.mark.asyncio
    async def test_budget_threads_to_rdr_and_preserves_pod_seconds_for_phase2(
        self, tmp_path: Path, claim_map: dict
    ) -> None:
        from backend.agents.resilience import RunBudget

        project_id = "hybrid_budget"
        report_path = tmp_path / project_id / "final_report.json"
        _write_report(report_path, [{"id": "l1", "score": 0.1}])

        rdr_result = _make_rdr_result(
            project_id,
            rubric_score=0.1,
            final_report_path=str(report_path),
            clusters_total=1,
            clusters_failed=0,
        )
        captured_rdr: list[dict] = []
        captured_rlm: list[dict] = []

        async def _capture_rdr(*args, **kwargs):  # noqa: ANN001
            captured_rdr.append(kwargs)
            return rdr_result

        async def _capture_rlm(pid, root, wcm, **kwargs):  # noqa: ANN001
            captured_rlm.append(kwargs)
            return _make_rlm_result(project_id)

        budget = RunBudget(
            max_usd=1.0,
            max_wall_clock_seconds=600,
            max_pod_seconds=90,
            max_invocations_per_agent={"rdr": 1},
        )

        await run_pipeline_hybrid(
            project_id,
            tmp_path,
            claim_map,
            run_budget=budget,
            repair_target=0.6,
            _rdr_runner=_capture_rdr,
            _rlm_runner=_capture_rlm,
        )

        assert captured_rdr[0]["run_budget"] is budget
        phase2_budget = captured_rlm[0]["run_budget"]
        assert phase2_budget.max_pod_seconds == 90
        assert phase2_budget.max_invocations_per_agent == {"rdr": 1}

    # ------------------------------------------------------------------
    # Test: Phase 2 reuses the same project_id
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_phase2_uses_same_project_id(
        self, tmp_path: Path, claim_map: dict
    ) -> None:
        """Phase 2 must be invoked with the same project_id as Phase 1."""
        project_id = "hybrid_same_pid"
        report_path = tmp_path / project_id / "final_report.json"
        _write_report(report_path, [{"id": "l1", "score": 0.1}])

        rdr_result = _make_rdr_result(
            project_id, rubric_score=0.1, final_report_path=str(report_path)
        )
        rlm_result = _make_rlm_result(project_id)

        phase2_project_ids: list[str] = []

        async def _capture_rlm(pid, root, wcm, **kw):  # noqa: ANN001
            phase2_project_ids.append(pid)
            return rlm_result

        await run_pipeline_hybrid(
            project_id, tmp_path, claim_map,
            repair_target=0.6,
            _rdr_runner=AsyncMock(return_value=rdr_result),
            _rlm_runner=_capture_rlm,
        )

        assert phase2_project_ids == [project_id]

    # ------------------------------------------------------------------
    # Test: Phase 1 failure degrades gracefully — no Phase 2
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_failure_in_phase1_degrades_to_phase1_result(
        self, tmp_path: Path, claim_map: dict
    ) -> None:
        """If Phase 1 raises, the hybrid returns a failed RLMRunResult without
        calling Phase 2."""

        async def _failing_rdr(*args, **kwargs):  # noqa: ANN001
            raise RuntimeError("RDR catastrophic failure")

        mock_rlm = AsyncMock()

        result = await run_pipeline_hybrid(
            "hybrid_fail", tmp_path, claim_map,
            repair_target=0.6,
            _rdr_runner=_failing_rdr,
            _rlm_runner=mock_rlm,
        )

        mock_rlm.assert_not_awaited()
        assert result.status == "failed"
        assert result.rubric_score is None

    # ------------------------------------------------------------------
    # Test: Phase 1 max_repair_iterations is always 0
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_phase1_uses_zero_repair_iterations(
        self, tmp_path: Path, claim_map: dict
    ) -> None:
        """Phase 1 must call run_pipeline_rdr with max_repair_iterations=0."""
        project_id = "hybrid_zero_repair"
        report_path = tmp_path / project_id / "final_report.json"
        _write_report(report_path, [{"id": "l1", "score": 0.9}])
        rdr_result = _make_rdr_result(
            project_id, rubric_score=0.9, final_report_path=str(report_path)
        )

        captured_kwargs: list[dict] = []

        async def _capture_rdr(*args, **kwargs):  # noqa: ANN001
            captured_kwargs.append(kwargs)
            return rdr_result

        await run_pipeline_hybrid(
            project_id, tmp_path, claim_map,
            repair_target=0.6,
            _rdr_runner=_capture_rdr,
            _rlm_runner=AsyncMock(),
        )

        assert captured_kwargs[0].get("max_repair_iterations") == 0

    # ------------------------------------------------------------------
    # Test: R1 — uniform Phase 1 failure skips Phase 2
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_uniform_phase1_failure_skips_phase2(
        self, tmp_path: Path, claim_map: dict
    ) -> None:
        """When every cluster fails in Phase 1, Phase 2 must NOT be called."""
        project_id = "hybrid_all_fail"
        clusters_total = 27
        rdr_result = _make_rdr_result(
            project_id,
            rubric_score=0.0,
            final_report_path="/tmp/fake_report.json",
            clusters_total=clusters_total,
            clusters_failed=clusters_total,
            status="failed",
        )

        mock_rdr = AsyncMock(return_value=rdr_result)
        mock_rlm = MagicMock(return_value=_make_rlm_result(project_id))

        result = await run_pipeline_hybrid(
            project_id, tmp_path, claim_map,
            repair_target=0.6,
            _rdr_runner=mock_rdr,
            _rlm_runner=mock_rlm,
        )

        mock_rlm.assert_not_called()
        assert result.status == "failed"
        assert result.iterations == clusters_total

    # ------------------------------------------------------------------
    # Test: R3 — budget is decremented across phases
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_budget_remaining_after_phase1(
        self, tmp_path: Path, claim_map: dict
    ) -> None:
        """Phase 2 must receive a budget reduced by Phase 1's cost."""
        from backend.agents.resilience.budget import RunBudget

        project_id = "hybrid_budget_remaining"
        report_path = tmp_path / project_id / "final_report.json"
        _write_report(report_path, [{"id": "l1", "score": 0.1}])  # weak cluster → Phase 2 runs

        rdr_result = _make_rdr_result(
            project_id,
            rubric_score=0.1,
            final_report_path=str(report_path),
            clusters_failed=0,
        )
        # Override cost_usd to a known value.
        rdr_result = dataclasses.replace(rdr_result, cost_usd=0.40)

        captured_budgets: list = []

        async def _capture_rlm(pid, root, wcm, **kw):  # noqa: ANN001
            captured_budgets.append(kw.get("run_budget"))
            return _make_rlm_result(project_id)

        budget = RunBudget(max_usd=1.00, max_wall_clock_seconds=600)

        await run_pipeline_hybrid(
            project_id, tmp_path, claim_map,
            run_budget=budget,
            repair_target=0.6,
            _rdr_runner=AsyncMock(return_value=rdr_result),
            _rlm_runner=_capture_rlm,
        )

        assert len(captured_budgets) == 1
        rb = captured_budgets[0]
        assert rb is not None
        assert rb.max_usd == pytest.approx(0.60)
        assert rb.max_wall_clock_seconds == 600

    # ------------------------------------------------------------------
    # Test: R3 — Phase 1 exhausts budget → Phase 2 skipped
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_phase1_exhausts_budget_skips_phase2(
        self, tmp_path: Path, claim_map: dict
    ) -> None:
        """When Phase 1 cost_usd >= max_usd, Phase 2 must be skipped."""
        from backend.agents.resilience.budget import RunBudget

        project_id = "hybrid_budget_exhausted"
        report_path = tmp_path / project_id / "final_report.json"
        _write_report(report_path, [{"id": "l1", "score": 0.1}])  # weak cluster

        rdr_result = _make_rdr_result(
            project_id,
            rubric_score=0.1,
            final_report_path=str(report_path),
            clusters_failed=0,
        )
        rdr_result = dataclasses.replace(rdr_result, cost_usd=0.60)

        mock_rlm = AsyncMock(return_value=_make_rlm_result(project_id))
        budget = RunBudget(max_usd=0.50, max_wall_clock_seconds=None)

        result = await run_pipeline_hybrid(
            project_id, tmp_path, claim_map,
            run_budget=budget,
            repair_target=0.6,
            _rdr_runner=AsyncMock(return_value=rdr_result),
            _rlm_runner=mock_rlm,
        )

        mock_rlm.assert_not_awaited()
        assert result.cost_usd == pytest.approx(0.60)

    # ------------------------------------------------------------------
    # Test: E1 — explicit kwargs passed; claim_map not mutated
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_explicit_kwargs_passed_to_rlm(
        self, tmp_path: Path, claim_map: dict
    ) -> None:
        """Phase 2 must receive hybrid_repair_only and phase1_weak_clusters as
        explicit kwargs; the original workspace_claim_map must not be mutated."""
        project_id = "hybrid_explicit_kwargs"
        report_path = tmp_path / project_id / "final_report.json"
        _write_report(report_path, [{"id": "l1", "score": 0.1}])

        rdr_result = _make_rdr_result(
            project_id,
            rubric_score=0.1,
            final_report_path=str(report_path),
            clusters_failed=0,
        )

        received_args: list[tuple] = []
        received_kwargs: list[dict] = []
        received_wcms: list[dict] = []

        async def _capture_rlm(pid, root, wcm, **kw):  # noqa: ANN001
            received_wcms.append(dict(wcm))
            received_kwargs.append(kw)
            return _make_rlm_result(project_id)

        original_claim_map = dict(claim_map)  # snapshot before call

        await run_pipeline_hybrid(
            project_id, tmp_path, claim_map,
            repair_target=0.6,
            _rdr_runner=AsyncMock(return_value=rdr_result),
            _rlm_runner=_capture_rlm,
        )

        assert len(received_kwargs) == 1
        kw = received_kwargs[0]
        assert kw.get("hybrid_repair_only") is True
        assert isinstance(kw.get("phase1_weak_clusters"), list)
        assert len(kw["phase1_weak_clusters"]) == 1

        # The workspace_claim_map passed to Phase 2 must not contain the
        # control-flow keys (proves we no longer mutate it).
        sent_wcm = received_wcms[0]
        assert "_hybrid_repair_only" not in sent_wcm
        assert "_phase1_weak_clusters" not in sent_wcm

        # The original claim_map object itself must also be untouched.
        assert "_hybrid_repair_only" not in claim_map
        assert "_phase1_weak_clusters" not in claim_map


    # ------------------------------------------------------------------
    # Test: no-bundle paper_id falls back to pure RLM (deployment unblock)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_bundle_falls_back_to_pure_rlm(self, tmp_path: Path) -> None:
        """For PDF/arXiv uploads (project_id=prj_*) there is no PaperBench bundle.
        The guard must skip Phase 1 RDR and dispatch directly to run_pipeline_rlm
        with the original workspace_claim_map (no _hybrid_repair_only flag)."""
        project_id = "prj_synthetic_arxiv"
        claim_map = {
            "project_id": project_id,
            "paperbench": {},  # no paper_id → falls back to project_id, no bundle
        }
        rlm_result = _make_rlm_result(project_id)

        received_args: list[tuple] = []
        received_kwargs: list[dict] = []

        async def _capture_rlm(pid, root, wcm, **kw):  # noqa: ANN001
            received_args.append((pid, root, wcm))
            received_kwargs.append(kw)
            return rlm_result

        mock_rdr = AsyncMock()  # must NOT be called

        result = await run_pipeline_hybrid(
            project_id, tmp_path, claim_map,
            _rdr_runner=mock_rdr,
            _rlm_runner=_capture_rlm,
        )

        # Phase 1 must have been completely skipped.
        mock_rdr.assert_not_awaited()

        # Pure RLM must have been called once.
        assert len(received_args) == 1
        pid_sent, _, wcm_sent = received_args[0]
        assert pid_sent == project_id
        # Must pass the original claim_map unchanged.
        assert wcm_sent is claim_map

        # Pure RLM call must NOT carry the hybrid-repair-only flag.
        kw = received_kwargs[0]
        assert kw.get("hybrid_repair_only") is not True
        assert kw.get("phase1_weak_clusters") is None

        # Return value must be what _rlm returned.
        assert result is rlm_result

    # ------------------------------------------------------------------
    # Test: bundle present → Phase 1 still runs (regression guard)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_bundle_present_still_runs_hybrid(self, tmp_path: Path) -> None:
        """When a real PaperBench bundle directory exists, the guard must NOT
        short-circuit — Phase 1 RDR is called as normal."""
        # Use a paper_id that has a real bundle in third_party/paperbench/.
        paper_id = "sequential-neural-score-estimation"
        project_id = "pb_test_seqnn"
        report_path = tmp_path / project_id / "final_report.json"
        _write_report(report_path, [
            {"id": "l1", "score": 0.8},
            {"id": "l2", "score": 0.9},
        ])
        claim_map = {
            "project_id": project_id,
            "paperbench": {"paper_id": paper_id},
        }
        rdr_result = _make_rdr_result(
            project_id, rubric_score=0.85, final_report_path=str(report_path)
        )

        mock_rdr = AsyncMock(return_value=rdr_result)
        mock_rlm = AsyncMock()  # Phase 2 should NOT run (all leaves pass)

        result = await run_pipeline_hybrid(
            project_id, tmp_path, claim_map,
            repair_target=0.6,
            _rdr_runner=mock_rdr,
            _rlm_runner=mock_rlm,
        )

        # Phase 1 must have run because the bundle dir exists.
        mock_rdr.assert_awaited_once()
        # Phase 2 must NOT have run (all leaves above 0.6).
        mock_rlm.assert_not_awaited()

        assert result.rubric_score == pytest.approx(0.85)

    # ------------------------------------------------------------------
    # Test: explicit _bundles_root in claim_map is honored
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_explicit_bundles_root_in_claim_map_honored(
        self, tmp_path: Path
    ) -> None:
        """When workspace_claim_map contains '_bundles_root' pointing to a path
        that has no bundle dir for the paper_id, the guard falls back to pure RLM."""
        project_id = "test"
        claim_map = {
            "project_id": project_id,
            "_bundles_root": "/nonexistent/path",
            "paperbench": {},
        }
        rlm_result = _make_rlm_result(project_id)

        mock_rdr = AsyncMock()  # must NOT be called
        mock_rlm = AsyncMock(return_value=rlm_result)

        result = await run_pipeline_hybrid(
            project_id, tmp_path, claim_map,
            _rdr_runner=mock_rdr,
            _rlm_runner=mock_rlm,
        )

        # The explicit bundles_root path doesn't have the bundle → pure RLM.
        mock_rdr.assert_not_awaited()
        mock_rlm.assert_awaited_once()
        assert result is rlm_result


# ---------------------------------------------------------------------------
# Module-level: run_pipeline_rlm accepts hybrid kwargs (E1 signature check)
# ---------------------------------------------------------------------------


def test_run_pipeline_rlm_accepts_hybrid_kwargs() -> None:
    """run_pipeline_rlm signature must include the explicit hybrid kwargs."""
    import inspect
    from backend.agents.rlm.run import run_pipeline_rlm

    sig = inspect.signature(run_pipeline_rlm)
    assert "hybrid_repair_only" in sig.parameters
    assert "phase1_weak_clusters" in sig.parameters
