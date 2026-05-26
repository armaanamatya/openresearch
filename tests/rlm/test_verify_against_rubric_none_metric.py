"""Regression test: verify_against_rubric must not crash when leaf_scores
contain score=None entries (PR-κ data-unavailable leaves).

Bug: _rubric_areas called float(e.get("score", 0.0)) on an entry with an
explicit score=None, crashing every VAE run whose Frey Face metrics were None.
"""
import json
import pytest

from backend.agents.rlm.primitives import verify_against_rubric, _rubric_areas


# Rubric with one leaf that will be skipped (data-unavailable) and one that won't.
RUBRIC = {
    "id": "root",
    "requirements": "reproduce the paper",
    "weight": 1.0,
    "source": "generated",
    "target_score": 0.5,
    "sub_tasks": [
        {
            "id": "mnist-area",
            "requirements": "MNIST results",
            "weight": 0.6,
            "sub_tasks": [
                {"id": "mnist-leaf", "requirements": "mnist elbo", "weight": 1.0, "sub_tasks": []},
            ],
        },
        {
            "id": "freyface-area",
            "requirements": "Frey Face results",
            "weight": 0.4,
            "sub_tasks": [
                {"id": "ff-leaf", "requirements": "frey face elbo", "weight": 1.0, "sub_tasks": []},
            ],
        },
    ],
}


def test_rubric_areas_tolerates_none_score():
    """_rubric_areas must not raise TypeError when a leaf has score=None."""
    leaf_scores_with_none = [
        {"id": "mnist-leaf", "score": 0.75, "justification": "ok"},
        {"id": "ff-leaf", "score": None, "justification": "data_unavailable: frey_face",
         "state": "skipped_data_unavailable"},
    ]
    # Must not raise
    areas = _rubric_areas(RUBRIC, leaf_scores_with_none)
    assert isinstance(areas, list)
    assert len(areas) == 2
    # The MNIST area has a real score; the Frey Face area has no graded leaf
    # so roll_up returns None → _clamp01(None) → 0.0.
    assert all(isinstance(a["score"], float) for a in areas)


def test_verify_against_rubric_none_metric_no_typeerror(make_context, tmp_path):
    """verify_against_rubric must not raise TypeError when score_reproduction
    returns some leaf_scores with score=None (data-unavailable leaves).

    The LLM batch grader is stubbed: it grades the MNIST leaf with 0.8 and
    the Frey Face leaf is pre-filtered as data-unavailable by
    _detect_data_unavailable_leaves (which reads data_load_failures from
    final_report.json). We simulate this by writing final_report.json with
    data_load_failures before calling verify_against_rubric.
    """
    # Write a final_report.json with scope.gaps declaring frey_face unavailable.
    # _detect_data_unavailable_leaves reads scope.gaps from final_report.json
    # and data_load_failures from code/outputs/metrics.json.
    import json
    from pathlib import Path
    project_dir = tmp_path / "test_proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "final_report.json").write_text(json.dumps({
        "verdict": "partial",
        "baseline_metrics": {"mnist_test_elbo": -123.2},
        "rubric": {},
        "paper": {"id": "", "title": "test"},
        "scope": {
            "gaps": ["Frey Face dataset: HTTP 403 — unavailable"],
        },
    }))
    # Also write data_load_failures into code/outputs/metrics.json (signal 1).
    outputs_dir = project_dir / "code" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "metrics.json").write_text(json.dumps({
        "mnist_test_elbo": -123.2,
        "data_load_failures": [{"dataset": "frey_face", "error": "HTTP 403"}],
    }))

    # The LLM grader sees only the eligible (MNIST) leaf — returns score 0.8 for it.
    batch_response = json.dumps([
        {"leaf_id": "mnist-leaf", "score": 0.8, "justification": "matches paper"},
    ])
    ctx = make_context(tmp_path, llm_responses=[batch_response])
    # Override project_dir to the one with final_report.json
    ctx = ctx.__class__(
        project_id=ctx.project_id,
        project_dir=project_dir,
        runs_root=tmp_path,
        dashboard=ctx.dashboard,
        emit=ctx.emit,
        cost_ledger=ctx.cost_ledger,
        llm_client=ctx.llm_client,
        provider=ctx.provider,
        model=ctx.model,
    )

    results = {"success": True, "metrics": {"mnist_test_elbo": -123.2}}
    # Must not raise TypeError
    result = verify_against_rubric(results, RUBRIC, ctx=ctx)

    assert "error" not in result or result.get("success") is not False, (
        f"verify_against_rubric returned an error: {result.get('error')}"
    )
    assert "overall_score" in result
    assert isinstance(result["overall_score"], float)
    # Frey Face leaf was skipped — its score should NOT appear in weak_leaves
    weak_ids = {e.get("id") for e in result.get("weak_leaves", [])}
    assert "ff-leaf" not in weak_ids, "data-unavailable leaf must not appear in weak_leaves"
    # Justification for skipped leaf mentions None or data_unavailable
    leaf_records = result.get("leaf_scores", [])
    ff_records = [r for r in leaf_records if r.get("id") == "ff-leaf"]
    assert ff_records, "ff-leaf should appear in leaf_scores as a skipped record"
    ff_rec = ff_records[0]
    assert ff_rec["score"] is None, "skipped leaf score must be None, not 0.0"
    assert "unavailable" in (ff_rec.get("justification") or "").lower() or \
           ff_rec.get("state") == "skipped_data_unavailable"
