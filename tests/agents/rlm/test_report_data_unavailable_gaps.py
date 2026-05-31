"""final_report must clearly surface datasets the agent recorded as unobtainable
(data_load_failures / status=data_unavailable) — they were excluded from the
rubric score, so the report says so plainly (2026-05-30 user requirement)."""
from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm.report import (
    _collect_data_unavailable_gaps,
    _merge_data_unavailable_gaps,
)


def _write_metrics(project_dir: Path, payload: dict) -> None:
    out = project_dir / "code" / "outputs" / "run1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_collect_from_data_load_failures(tmp_path: Path):
    _write_metrics(tmp_path, {"data_load_failures": [{"dataset": "WebShop", "error": "HTTP 404 timeout"}]})
    gaps = _collect_data_unavailable_gaps(tmp_path)
    assert len(gaps) == 1
    assert gaps[0].startswith("webshop:")
    assert "unobtainable" in gaps[0]
    assert "excluded from rubric score" in gaps[0]
    assert "HTTP 404" in gaps[0]


def test_collect_from_experiment_status(tmp_path: Path):
    _write_metrics(tmp_path, {"experiments": {"webshop_3b": {"status": "data_unavailable", "reason": "no server"}}})
    gaps = _collect_data_unavailable_gaps(tmp_path)
    assert any(g.startswith("webshop_3b:") and "no server" in g for g in gaps)


def test_collect_empty_when_no_metrics(tmp_path: Path):
    assert _collect_data_unavailable_gaps(tmp_path) == []


def test_merge_preserves_existing_and_dedups(tmp_path: Path):
    _write_metrics(tmp_path, {"data_load_failures": ["webshop"]})
    scope = {"requested": "smallest-two", "ran": ["alfworld"], "gaps": ["search-qa: timed out"]}
    merged = _merge_data_unavailable_gaps(scope, tmp_path)
    # existing gap preserved
    assert any("search-qa" in g for g in merged["gaps"])
    # webshop added
    assert any(g.startswith("webshop:") for g in merged["gaps"])
    # idempotent: a second merge does not duplicate webshop
    merged2 = _merge_data_unavailable_gaps(merged, tmp_path)
    assert sum(g.startswith("webshop:") for g in merged2["gaps"]) == 1


def test_merge_no_metrics_is_noop(tmp_path: Path):
    scope = {"gaps": ["x"]}
    assert _merge_data_unavailable_gaps(scope, tmp_path) == scope


# --- failed/skipped MODELS are surfaced too (2026-05-30 graceful degradation) ---
# Updated 2026-05-31: tests pass operator_skip_models so the intentional-skip vs
# code-bug phrasing distinction is correctly exercised.


def test_collect_model_load_failed_status(tmp_path: Path):
    # Operator intentionally de-scoped qwen3_1_7b → "model unavailable" phrasing.
    _write_metrics(tmp_path, {"per_model": {"qwen3_1_7b": {"status": "model_load_failed",
                                                            "error": "invalid HF id"}}})
    gaps = _collect_data_unavailable_gaps(tmp_path, operator_skip_models=["qwen3_1_7b"])
    assert any(g.startswith("qwen3_1_7b:") and "model unavailable" in g and "invalid HF id" in g
               for g in gaps)


def test_collect_models_skipped_scope_reduction(tmp_path: Path):
    # Operator de-scoped 7B → "model unavailable, excluded" phrasing.
    _write_metrics(tmp_path, {"scope": {"models_skipped": ["qwen2_5_7b"]}})
    gaps = _collect_data_unavailable_gaps(tmp_path, operator_skip_models=["qwen2_5_7b"])
    assert any(g.startswith("qwen2_5_7b:") and "model unavailable" in g for g in gaps)


def test_collect_model_load_failed_code_bug_phrasing(tmp_path: Path):
    # Requested model (not in operator skip) → "repairable code bug" phrasing.
    _write_metrics(tmp_path, {"per_model": {"qwen3_1_7b": {"status": "model_load_failed",
                                                            "error": "invalid HF id"}}})
    gaps = _collect_data_unavailable_gaps(tmp_path, operator_skip_models=[])
    assert any(g.startswith("qwen3_1_7b:") and "repairable code bug" in g for g in gaps)


def test_collect_mixed_dataset_and_model(tmp_path: Path):
    # Operator de-scoped 7B; webshop dataset also failed.
    _write_metrics(tmp_path, {
        "data_load_failures": [{"dataset": "webshop", "error": "404"}],
        "per_model": {"qwen2_5_7b": {"status": "model_load_failed"}},
    })
    gaps = _collect_data_unavailable_gaps(tmp_path, operator_skip_models=["qwen2_5_7b"])
    assert any(g.startswith("webshop:") and "dataset unobtainable" in g for g in gaps)
    assert any(g.startswith("qwen2_5_7b:") and "model unavailable" in g for g in gaps)
