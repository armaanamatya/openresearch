"""experiment_runs.jsonl tagging tests for run_experiment (PR A Wave 3).

Exercises the model_id/eval_env tagging contract end-to-end via
_persist_experiment_result so PR B's cross-check can rely on the format.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.rlm.primitives import _persist_experiment_result


@pytest.fixture
def fake_ctx(tmp_path: Path):
    ctx = SimpleNamespace(
        project_dir=tmp_path,
        cost_ledger=None,
        dashboard=None,
        emit=None,
        scope_spec=None,
    )
    return ctx


class TestPersistExperimentResult:
    def test_default_tags(self, fake_ctx):
        result = {"success": True, "metrics": {"acc": 0.5}, "logs": ""}
        out = _persist_experiment_result(fake_ctx, result)
        log = (fake_ctx.project_dir / "experiment_runs.jsonl").read_text(encoding="utf-8").strip()
        row = json.loads(log)
        assert row["model_id"] == "default"
        assert row["eval_env"] == "default"
        assert row["success"] is True
        assert out is result  # function returns its argument by contract

    def test_custom_tags_persisted(self, fake_ctx):
        result = {"success": True, "metrics": {"reward": 0.4}, "logs": ""}
        _persist_experiment_result(
            fake_ctx, result, model_id="qwen3-1.7b", eval_env="ALFWorld"
        )
        row = json.loads((fake_ctx.project_dir / "experiment_runs.jsonl").read_text(encoding="utf-8").strip())
        assert row["model_id"] == "qwen3-1.7b"
        assert row["eval_env"] == "ALFWorld"

    def test_multiple_rows_appended(self, fake_ctx):
        for mid, env in [("qwen3-1.7b", "ALFWorld"), ("qwen3-1.7b", "WebShop"), ("qwen2.5-3b", "ALFWorld")]:
            _persist_experiment_result(
                fake_ctx, {"success": True, "metrics": {}, "logs": ""},
                model_id=mid, eval_env=env,
            )
        lines = (fake_ctx.project_dir / "experiment_runs.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        rows = [json.loads(line) for line in lines]
        pairs = {(r["model_id"], r["eval_env"]) for r in rows}
        assert pairs == {
            ("qwen3-1.7b", "ALFWorld"),
            ("qwen3-1.7b", "WebShop"),
            ("qwen2.5-3b", "ALFWorld"),
        }

    def test_failed_result_still_tagged(self, fake_ctx):
        result = {"success": False, "metrics": {}, "logs": "err", "error": "boom"}
        _persist_experiment_result(fake_ctx, result, model_id="m1", eval_env="e1")
        row = json.loads((fake_ctx.project_dir / "experiment_runs.jsonl").read_text(encoding="utf-8").strip())
        assert row["success"] is False
        assert row["model_id"] == "m1"
        assert row["eval_env"] == "e1"

    def test_caller_supplied_tag_in_result_wins(self, fake_ctx):
        # setdefault contract: if result carries model_id, the explicit kwarg
        # does NOT override.
        result = {"success": True, "metrics": {}, "logs": "", "model_id": "from_result"}
        _persist_experiment_result(fake_ctx, result, model_id="from_kwarg")
        row = json.loads((fake_ctx.project_dir / "experiment_runs.jsonl").read_text(encoding="utf-8").strip())
        assert row["model_id"] == "from_result"
