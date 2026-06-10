"""Tests for _verify_scope_evidence (PR B scope-evidence cross-check)."""

from __future__ import annotations

import json
from pathlib import Path


from backend.agents.rlm.report import _verify_scope_evidence


def _write_log(run_dir: Path, rows: list[dict]) -> None:
    log = run_dir / "experiment_runs.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class TestVerifyScopeEvidence:
    def test_non_dict_scope_passes_through(self, tmp_path):
        out, reason = _verify_scope_evidence("not a dict", tmp_path)
        assert out == "not a dict"
        assert reason is None

    def test_empty_ran_passes(self, tmp_path):
        scope = {"requested": "x", "ran": [], "gaps": []}
        out, reason = _verify_scope_evidence(scope, tmp_path)
        assert out == scope
        assert reason is None

    def test_missing_log_passes(self, tmp_path):
        scope = {"requested": "x", "ran": ["a"], "gaps": []}
        out, reason = _verify_scope_evidence(scope, tmp_path)
        assert out == scope
        assert reason is None

    def test_legacy_default_tags_is_no_op(self, tmp_path):
        # All rows tagged "default" → no real evidence → no enforcement.
        _write_log(tmp_path, [
            {"success": True, "model_id": "default", "eval_env": "default"},
            {"success": True, "model_id": "default", "eval_env": "default"},
        ])
        scope = {"ran": ["claimed-but-not-tagged"]}
        out, reason = _verify_scope_evidence(scope, tmp_path)
        assert out == scope
        assert reason is None

    def test_model_only_evidence(self, tmp_path):
        _write_log(tmp_path, [
            {"success": True, "model_id": "qwen3-1.7b", "eval_env": "default"},
        ])
        scope = {"ran": ["qwen3-1.7b"], "gaps": []}
        out, reason = _verify_scope_evidence(scope, tmp_path)
        assert out == scope
        assert reason is None

    def test_env_only_evidence(self, tmp_path):
        _write_log(tmp_path, [
            {"success": True, "model_id": "default", "eval_env": "ALFWorld"},
        ])
        scope = {"ran": ["ALFWorld"], "gaps": []}
        out, reason = _verify_scope_evidence(scope, tmp_path)
        assert out == scope
        assert reason is None

    def test_composite_evidence(self, tmp_path):
        _write_log(tmp_path, [
            {"success": True, "model_id": "qwen3-1.7b", "eval_env": "ALFWorld"},
            {"success": True, "model_id": "qwen3-1.7b", "eval_env": "WebShop"},
        ])
        scope = {"ran": ["qwen3-1.7b/ALFWorld", "qwen3-1.7b/WebShop"], "gaps": []}
        out, reason = _verify_scope_evidence(scope, tmp_path)
        assert out == scope
        assert reason is None

    def test_unverified_moves_to_gaps(self, tmp_path):
        _write_log(tmp_path, [
            {"success": True, "model_id": "qwen3-1.7b", "eval_env": "ALFWorld"},
        ])
        scope = {"ran": ["qwen3-1.7b/ALFWorld", "qwen3-1.7b/Search-QA"], "gaps": []}
        out, reason = _verify_scope_evidence(scope, tmp_path)
        assert "qwen3-1.7b/ALFWorld" in out["ran"]
        assert "qwen3-1.7b/Search-QA" not in out["ran"]
        assert any("qwen3-1.7b/Search-QA" in g for g in out["gaps"])
        assert reason is not None and "1 unverified" in reason

    def test_failed_rows_dont_count(self, tmp_path):
        _write_log(tmp_path, [
            {"success": False, "model_id": "qwen3-1.7b", "eval_env": "ALFWorld"},
        ])
        scope = {"ran": ["qwen3-1.7b/ALFWorld"], "gaps": []}
        out, reason = _verify_scope_evidence(scope, tmp_path)
        assert "qwen3-1.7b/ALFWorld" not in out["ran"]
        assert any("qwen3-1.7b/ALFWorld" in g for g in out["gaps"])

    def test_preserves_existing_gaps(self, tmp_path):
        _write_log(tmp_path, [
            {"success": True, "model_id": "qwen3-1.7b", "eval_env": "ALFWorld"},
        ])
        scope = {
            "ran": ["qwen3-1.7b/ALFWorld", "qwen3-1.7b/Search-QA"],
            "gaps": ["pre-existing gap: budget exhausted"],
        }
        out, _ = _verify_scope_evidence(scope, tmp_path)
        assert "pre-existing gap: budget exhausted" in out["gaps"]
        assert any("Search-QA" in g for g in out["gaps"])
        assert len(out["gaps"]) == 2
