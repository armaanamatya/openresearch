"""Deterministic weak-leaf triage (leaf_triage.py).

The automated form of the 2026-06-11/12 operator steering that took Adam
0.0→0.716: classify grader justifications into repair classes, ground them
against actual disk state, order cheapest-first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm import leaf_triage


def _project(tmp_path: Path, *, history: bool = False, outputs: int = 0,
             sweep: bool = False, curves: bool = False) -> Path:
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"status": "completed"}
    if history:
        metrics["history"] = {"exp": {"adam": {"epoch": [1, 2]}}}
    if sweep:
        metrics["vae_lr_sweep"] = {"lr_0.001": {"elbo": -98.0}}
    (code / "metrics.json").write_text(json.dumps(metrics))
    for i in range(outputs):
        d = code / "outputs" / "run-x" / f"cell_{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text('{"test_accuracy": 0.9}')
        if curves:
            (d / "training_curves.json").write_text("{}")
    return tmp_path


def _leaf(score: float, justification: str, lid: str = "aabbccdd") -> dict:
    return {"id": lid, "score": score, "justification": justification}


# ---------------------------------------------------------------------------
# Classification + grounding
# ---------------------------------------------------------------------------


def test_render_artifact_when_data_exists(tmp_path):
    p = _project(tmp_path, history=True, sweep=True)
    out = leaf_triage.triage_weak_leaves(
        [_leaf(0.0, "The evidence contains no Figure-4-style artifact for the LR sweep")], p)
    d = out["plan"][0]
    assert d["repair_class"] == "render_artifact"
    assert d["cost"] == "none"
    assert "RENDER" in d["directive"]


def test_render_downgrades_to_protocol_without_data(tmp_path):
    p = _project(tmp_path)  # no history/curves/sweep/results on disk
    out = leaf_triage.triage_weak_leaves(
        [_leaf(0.0, "No figure showing the training curves is present")], p)
    assert out["plan"][0]["repair_class"] == "protocol_gap"
    assert out["plan"][0]["cost"] == "targeted_rerun"


def test_render_kept_when_measured_results_exist(tmp_path):
    # Adam fe5e7900 (2026-06-16): scalar per_model finals, NO training_curves.json.
    # A COMPARISON figure (final metric by condition) is renderable from those
    # scalars, so render_artifact must be KEPT (not demoted to protocol_gap) — the
    # demotion left it at 0.0 and the L2b sidecar backstop never fired.
    p = tmp_path
    code = p / "code"
    code.mkdir()
    (code / "metrics.json").write_text(json.dumps(
        {"status": "completed",
         "per_model": {"mnist_logreg": {"e": {"adam": {"metric": 37.06}}}}}))
    out = leaf_triage.triage_weak_leaves([_leaf(
        0.0, "the listing shows zero image or figure artifacts")], p)
    d = out["plan"][0]
    assert d["repair_class"] == "render_artifact" and d["cost"] == "none"


def test_provenance_gap(tmp_path):
    p = _project(tmp_path)
    out = leaf_triage.triage_weak_leaves(
        [_leaf(0.4, "batch size is only an assumption; epochs not confirmed anywhere")], p)
    assert out["plan"][0]["repair_class"] == "provenance_gap"
    assert "provenance.json" in out["plan"][0]["directive"]


def test_aggregation_gap_grounded_on_outputs(tmp_path):
    p = _project(tmp_path, outputs=3)
    out = leaf_triage.triage_weak_leaves(
        [_leaf(0.2, "Cell directories exist in outputs but metrics.json contains no per_model entries")], p)
    d = out["plan"][0]
    assert d["repair_class"] == "aggregation_gap"
    assert "3 per-cell metrics.json files" in d["directive"]


def test_aggregation_downgrades_to_review_without_outputs(tmp_path):
    p = _project(tmp_path, outputs=0)
    out = leaf_triage.triage_weak_leaves(
        [_leaf(0.2, "metrics.json is missing per_model entries")], p)
    assert out["plan"][0]["repair_class"] == "review"


def test_result_quality(tmp_path):
    p = _project(tmp_path)
    out = leaf_triage.triage_weak_leaves(
        [_leaf(0.0, "Adam ranked last among the five optimizers, directly contradicting the paper")], p)
    d = out["plan"][0]
    assert d["repair_class"] == "result_quality"
    # Recourse names the general fixable cause (per-condition HP tuning) AND the
    # honest-negative path (recourse-first + two-axis), not truncated.
    assert "per-condition" in d["directive"]
    assert "faithful-negative" in d["directive"]
    assert len(d["directive"]) <= leaf_triage._MAX_DIRECTIVE_CHARS


def test_protocol_gap(tmp_path):
    p = _project(tmp_path)
    out = leaf_triage.triage_weak_leaves(
        [_leaf(0.0, "Nothing in the code evidences whitening or input/FC dropout")], p)
    # 'no ... evidences' also matches provenance wording; either class must
    # carry a concrete directive — but whitening/dropout should win protocol.
    assert out["plan"][0]["repair_class"] in ("protocol_gap", "provenance_gap")
    assert out["plan"][0]["directive"]


def test_render_artifact_zero_figure_phrasing(tmp_path):
    # Adam fe5e79 (2026-06-16): the grader said "shows ZERO image or figure
    # artifacts" — a phrasing the old render regex missed, so a figure that just
    # needed rendering from on-disk data fell to "review". Data on disk → render.
    p = _project(tmp_path, history=True, sweep=True)
    out = leaf_triage.triage_weak_leaves([_leaf(
        0.0,
        "outputs/.../metrics.json and .log files are present but the listing "
        "shows zero image or figure artifacts; train.py has a fail-soft mpl guard",
    )], p)
    d = out["plan"][0]
    assert d["repair_class"] == "render_artifact"
    assert d["cost"] == "none" and "RENDER" in d["directive"]


def test_cell_failure_attempted_but_no_result(tmp_path):
    # Adam ac4006 (2026-06-16): in-scope imdb_logreg cells were ATTEMPTED
    # (provenance lists them) but produced no per_model entry — they errored.
    # Honest repair = re-run the failed cell, NOT exclude (that hides a real miss).
    p = _project(tmp_path)
    out = leaf_triage.triage_weak_leaves([_leaf(
        0.0,
        "metrics.json per_model has no 'imdb_logreg' entry and scope.models_run "
        "does not include it; provenance.json lists imdb_logreg cells but they "
        "failed to produce output",
    )], p)
    d = out["plan"][0]
    assert d["repair_class"] == "cell_failure"
    assert d["cost"] == "targeted_rerun"
    assert "RE-RUN" in d["directive"] and "exclud" in d["directive"]


def test_cell_failure_does_not_steal_result_quality(tmp_path):
    # A contradiction wins result_quality even when the same justification also
    # mentions a failed cell — cell_failure is checked LAST, only catching leaves
    # that would otherwise be bare "review".
    p = _project(tmp_path)
    out = leaf_triage.triage_weak_leaves([_leaf(
        0.0,
        "Adam ranked last, contradicting the paper; one cell also failed to run",
    )], p)
    assert out["plan"][0]["repair_class"] == "result_quality"


def test_review_fallback(tmp_path):
    p = _project(tmp_path)
    out = leaf_triage.triage_weak_leaves(
        [_leaf(0.3, "The proof in appendix B is paraphrased imprecisely")], p)
    assert out["plan"][0]["repair_class"] == "review"


# ---------------------------------------------------------------------------
# Ordering, thresholds, caps, robustness
# ---------------------------------------------------------------------------


def test_plan_orders_cheapest_then_weakest(tmp_path):
    p = _project(tmp_path, history=True, outputs=2)
    out = leaf_triage.triage_weak_leaves([
        _leaf(0.0, "Adam ranked last, contradicting the paper", "r1"),
        _leaf(0.4, "epochs not confirmed in provenance", "p1"),
        _leaf(0.1, "no figure rendered for the sweep", "f1"),
    ], p)
    costs = [d["cost"] for d in out["plan"]]
    assert costs == sorted(costs, key=lambda c: {"none": 0, "targeted_rerun": 1, "review": 2}[c])
    none_scores = [d["score"] for d in out["plan"] if d["cost"] == "none"]
    assert none_scores == sorted(none_scores)


def test_strong_leaves_excluded(tmp_path):
    p = _project(tmp_path)
    out = leaf_triage.triage_weak_leaves([_leaf(0.8, "fine"), _leaf(0.7, "ok")], p)
    assert out["plan"] == []
    assert out["summary"] == ""


def test_cap_respected(tmp_path):
    p = _project(tmp_path)
    leaves = [_leaf(0.0, f"no evidence of parameter {i} recorded", f"l{i}") for i in range(20)]
    out = leaf_triage.triage_weak_leaves(leaves, p)
    assert len(out["plan"]) == 8


def test_never_raises_on_garbage(tmp_path):
    out = leaf_triage.triage_weak_leaves(
        [None, "string", {"score": "NaN-ish"}, {}], tmp_path / "missing")
    assert isinstance(out, dict) and "plan" in out


# ---------------------------------------------------------------------------
# Persistence + guidance block + flag
# ---------------------------------------------------------------------------


def test_persist_and_guidance_block(tmp_path, monkeypatch):
    monkeypatch.delenv(leaf_triage.ENV_FLAG, raising=False)
    p = _project(tmp_path, history=True)
    triage = leaf_triage.triage_weak_leaves(
        [_leaf(0.0, "no figure rendered from the history data")], p)
    leaf_triage.persist(p, triage)
    assert (p / "rlm_state" / "leaf_triage.json").is_file()
    block = leaf_triage.guidance_block(p)
    assert "LEAF REPAIR PLAN" in block
    assert "[none]" in block


def test_guidance_block_empty_without_state(tmp_path, monkeypatch):
    monkeypatch.delenv(leaf_triage.ENV_FLAG, raising=False)
    assert leaf_triage.guidance_block(tmp_path) == ""


def test_flag_disables(tmp_path, monkeypatch):
    monkeypatch.setenv(leaf_triage.ENV_FLAG, "0")
    p = _project(tmp_path)
    leaf_triage.persist(p, {"plan": [{"leaf_id": "x", "cost": "none",
                                      "score": 0.0, "directive": "d"}]})
    assert leaf_triage.guidance_block(p) == ""
    assert leaf_triage.is_enabled() is False


def test_summary_counts(tmp_path):
    p = _project(tmp_path, history=True, outputs=1)
    out = leaf_triage.triage_weak_leaves([
        _leaf(0.0, "no figure for the curves", "a"),
        _leaf(0.0, "result contradicts the paper ordering", "b"),
    ], p)
    assert "repairable with NO" in out["summary"]
