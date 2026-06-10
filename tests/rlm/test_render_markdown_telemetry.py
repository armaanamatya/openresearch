"""PR-ν.2 — final_report.md renderer must consume token + timing sidecars.

Three invariants pinned here:
1. ``_render_markdown(report, project_dir)`` must surface tokens_total.json and
   timing.json content when those files exist.
2. It must fail soft (no exception, just no section) when sidecars are missing
   or corrupt — telemetry is best-effort by design.
3. Weak rubric leaves must render with truncated justifications when present
   AND must filter out score=None entries (PR-κ data-unavailable leaves).
"""
from __future__ import annotations

import json
from pathlib import Path


from backend.agents.rlm.report import RLMFinalReport, _render_markdown


def _make_report(rubric: dict | None = None, **extra) -> RLMFinalReport:
    """Minimal RLMFinalReport with safe defaults."""
    return RLMFinalReport(
        paper={"id": "test-paper", "title": "Test Paper"},
        verdict="partial",
        reproduction_summary="A test reproduction.",
        baseline_metrics={},
        paper_claims={},
        rubric=rubric or {"overall_score": 0.5, "meets_target": False, "areas": []},
        improvements=[],
        primitive_trace={},
        cost={"primitives": 0.0, "llm_usd": 0.0},
        iterations=1,
        primitive_provider="real",
        degraded=False,
        **extra,
    )


def test_render_includes_token_usage_when_sidecar_present(tmp_path: Path) -> None:
    project_dir = tmp_path / "prj_xyz"
    project_dir.mkdir()
    (project_dir / "tokens_total.json").write_text(json.dumps({
        "grand_total": {
            "calls": 42, "input_tokens": 100, "output_tokens": 12345,
            "cache_creation_input_tokens": 5000,
            "cache_read_input_tokens": 50000,
        },
        "by_primitive": {
            "plan_reproduction": {"calls": 1, "input_tokens": 6, "output_tokens": 999},
            "heartbeat": {"calls": 10, "input_tokens": 0, "output_tokens": 0},
        },
    }))

    report = _make_report()
    md = _render_markdown(report, project_dir=project_dir)

    assert "## Token Usage" in md
    assert "Total LLM calls" in md
    assert "12,345" in md  # output_tokens formatted with comma
    assert "50,000" in md  # cache reads
    # Per-primitive table includes plan_reproduction but suppresses all-zero rows.
    assert "plan_reproduction" in md
    assert "heartbeat" not in md  # zero-token primitive must be suppressed


def test_render_includes_timing_when_sidecar_present(tmp_path: Path) -> None:
    project_dir = tmp_path / "prj_xyz"
    project_dir.mkdir()
    (project_dir / "timing.json").write_text(json.dumps({
        "wall_clock_s": 3661.0,  # 1h 1m 1s
        "primitive_wall_clock_s": {
            "run_experiment": 3000.0,
            "implement_baseline": 600.0,
            "heartbeat": 0.005,  # too-small entries are suppressed
        },
        "primitive_call_counts": {"run_experiment": 2, "implement_baseline": 1, "heartbeat": 5},
        "gpu_hours": 0.85,
        "gpu_type": "L40S",
        "gpu_count": 1,
    }))

    report = _make_report()
    md = _render_markdown(report, project_dir=project_dir)

    assert "## Per-Step Timing" in md
    assert "Total wall clock" in md
    assert "1h 1m" in md
    assert "run_experiment" in md
    assert "GPU hours" in md
    assert "L40S" in md
    # Sub-0.01s rows must be suppressed (heartbeat noise).
    assert "heartbeat" not in md.split("## Per-Step Timing")[1]


def test_render_fails_soft_when_sidecars_missing(tmp_path: Path) -> None:
    """Missing sidecars must produce no Token/Timing section but no exception."""
    project_dir = tmp_path / "prj_xyz"
    project_dir.mkdir()
    report = _make_report()
    md = _render_markdown(report, project_dir=project_dir)
    assert "## Token Usage" not in md
    assert "## Per-Step Timing" not in md
    # Renderer still produced the rest of the report.
    assert "## Rubric Score" in md


def test_render_fails_soft_when_sidecars_corrupt(tmp_path: Path) -> None:
    """Corrupt JSON in sidecars must not crash the renderer."""
    project_dir = tmp_path / "prj_xyz"
    project_dir.mkdir()
    (project_dir / "tokens_total.json").write_text("this is not json {")
    (project_dir / "timing.json").write_text("{\"wall_clock_s\": ")  # truncated
    report = _make_report()
    md = _render_markdown(report, project_dir=project_dir)
    assert "## Token Usage" not in md
    assert "## Per-Step Timing" not in md


def test_render_no_project_dir_skips_telemetry_sections() -> None:
    """Legacy callers (no project_dir) must still get a valid markdown report."""
    report = _make_report()
    md = _render_markdown(report)  # legacy call shape
    assert "## Token Usage" not in md
    assert "## Per-Step Timing" not in md
    assert "## Rubric Score" in md


def test_render_weak_leaves_with_truncation_and_none_filter(tmp_path: Path) -> None:
    rubric = {
        "overall_score": 0.4,
        "meets_target": False,
        "areas": [],
        "weak_leaves": [
            {"id": "L1", "score": 0.1, "justification": "x" * 300},   # long, must truncate
            {"id": "L2", "score": 0.2, "justification": "ok"},
            {"id": "L3", "score": None, "justification": "data_unavailable"},  # must filter
            {"id": "L4", "score": 0.3, "justification": "another | with pipe"},
        ],
    }
    report = _make_report(rubric=rubric)
    md = _render_markdown(report, project_dir=tmp_path)
    assert "Weakest rubric leaves" in md
    # The 300-char justification must be truncated with an ellipsis.
    assert "…" in md
    # The None-score leaf must NOT appear (it's data-unavailable, not weak).
    assert "data_unavailable" not in md
    # Pipe chars must be escaped to not break the markdown table.
    assert "another \\| with pipe" in md


def test_render_area_field_accepts_both_name_and_area_keys() -> None:
    """The Area table must render whether the field is 'name' (legacy) or
    'area' (current _rubric_areas output) — backfilled reports use 'area'."""
    rubric = {
        "overall_score": 0.5,
        "meets_target": False,
        "areas": [
            {"area": "Method", "score": 0.7, "weight": 0.5},
            {"name": "Data", "score": 0.4, "notes": "legacy key"},
        ],
    }
    report = _make_report(rubric=rubric)
    md = _render_markdown(report)
    assert "Method" in md
    assert "0.700" in md
    assert "Data" in md
    assert "0.400" in md
