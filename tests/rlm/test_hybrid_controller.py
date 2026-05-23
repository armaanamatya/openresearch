"""Tests for the hybrid RDR+RLM controller
(``backend/agents/hybrid/controller.py``).

All LLM / RDR / RLM I/O is monkeypatched — no network, API keys, or Docker.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

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
        _hybrid_repair_only=True in the workspace_claim_map."""
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

        received_claim_maps: list[dict] = []

        async def _capture_rlm(pid, root, wcm, **kw):  # noqa: ANN001
            received_claim_maps.append(dict(wcm))
            return rlm_result

        mock_rdr = AsyncMock(return_value=rdr_result)

        result = await run_pipeline_hybrid(
            project_id, tmp_path, claim_map,
            repair_target=0.6,
            _rdr_runner=mock_rdr,
            _rlm_runner=_capture_rlm,
        )

        mock_rdr.assert_awaited_once()
        assert len(received_claim_maps) == 1

        # Phase 2 must receive the repair flag.
        sent_map = received_claim_maps[0]
        assert sent_map.get("_hybrid_repair_only") is True

        # Phase 2 must know which clusters are weak.
        weak = sent_map.get("_phase1_weak_clusters", [])
        assert len(weak) == 1
        assert weak[0]["id"] == "l1"

        # Final result is Phase 2's result.
        assert result.status == "completed"
        assert result.iterations == 3

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
