"""Integration test: scope-evidence cross-check downgrades verdict when unverified items remain."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


from backend.agents.rlm.report import build_final_report
from rlm.core.types import RLMChatCompletion


def _ctx(tmp_path: Path):
    ledger = MagicMock()
    # Simulate run_experiment having been called once so the honesty guard
    # in build_final_report does not drop baseline_metrics before scope-check.
    entry = MagicMock()
    entry.agent_id = "run_experiment"
    ledger.entries = [entry]
    ledger.total_usd.return_value = 0.0
    return SimpleNamespace(
        project_id="t1",
        project_dir=tmp_path,
        cost_ledger=ledger,
    )


def _make_result(report_dict: dict) -> RLMChatCompletion:
    """Build an RLMChatCompletion carrying the agent's self-attested report."""
    r = MagicMock(spec=RLMChatCompletion)
    r.response = json.dumps(report_dict)
    r.metadata = {"iterations": 1}
    r.usage_summary = None
    return r


class TestVerdictDowngradeOnUnverifiedScope:
    def test_reproduced_downgrades_to_partial_on_unverified(self, tmp_path):
        # experiment_runs.jsonl has evidence for only one of two claimed items.
        (tmp_path / "experiment_runs.jsonl").write_text(
            json.dumps(
                {"timestamp": "t", "success": True, "metrics": {"acc": 0.5},
                 "model_id": "qwen3-1.7b", "eval_env": "ALFWorld"}
            ) + "\n",
            encoding="utf-8",
        )
        result = _make_result({
            "verdict": "reproduced",
            "scope": {
                "requested": "two models",
                "ran": ["qwen3-1.7b/ALFWorld", "qwen2.5-3b/ALFWorld"],
                "gaps": [],
            },
            "baseline_metrics": {"acc": 0.5},
            "rubric": {"overall_score": 0.8, "meets_target": True},
            # primitive_trace is the run_experiment-was-called gate for the
            # existing _reconcile_verdict_against_evidence path; include it
            # so we isolate the scope check.
            "primitive_trace": {"by_primitive": {"run_experiment": 1}},
        })
        report = build_final_report(result, ctx=_ctx(tmp_path))
        assert report.verdict == "partial"
        assert "qwen2.5-3b/ALFWorld" not in report.scope["ran"]
        assert any("qwen2.5-3b/ALFWorld" in g for g in report.scope["gaps"])

    def test_reproduced_stays_when_all_verified(self, tmp_path):
        (tmp_path / "experiment_runs.jsonl").write_text(
            json.dumps(
                {"timestamp": "t", "success": True, "metrics": {"acc": 0.7},
                 "model_id": "qwen3-1.7b", "eval_env": "ALFWorld"}
            ) + "\n",
            encoding="utf-8",
        )
        result = _make_result({
            "verdict": "reproduced",
            "scope": {
                "requested": "smallest only",
                "ran": ["qwen3-1.7b/ALFWorld"],
                "gaps": [],
            },
            "baseline_metrics": {"acc": 0.7},
            "rubric": {"overall_score": 0.8, "meets_target": True},
            "primitive_trace": {"by_primitive": {"run_experiment": 1}},
        })
        report = build_final_report(result, ctx=_ctx(tmp_path))
        assert report.verdict == "reproduced"
        assert report.scope["ran"] == ["qwen3-1.7b/ALFWorld"]

    def test_legacy_run_with_default_tags_not_downgraded(self, tmp_path):
        # All log rows tagged "default" → cross-check is a no-op.
        (tmp_path / "experiment_runs.jsonl").write_text(
            json.dumps(
                {"timestamp": "t", "success": True, "metrics": {"acc": 0.7},
                 "model_id": "default", "eval_env": "default"}
            ) + "\n",
            encoding="utf-8",
        )
        result = _make_result({
            "verdict": "reproduced",
            "scope": {"requested": "", "ran": ["something"], "gaps": []},
            "baseline_metrics": {"acc": 0.7},
            "rubric": {"overall_score": 0.8, "meets_target": True},
            "primitive_trace": {"by_primitive": {"run_experiment": 1}},
        })
        report = build_final_report(result, ctx=_ctx(tmp_path))
        assert report.verdict == "reproduced"
        assert "something" in report.scope["ran"]
