"""Leaf-repair control loop (leaf_actuator.py, L4/L5/L6, 2026-06-16).

leaf_triage diagnoses; this actuator closes the loop — staging a synthesized
per-condition lr search (L4), a budget-gated seed plan (L5), and a declared-vs-
aggregated completeness audit (L6). Pure cores + flag contracts + the default-OFF
guarantee (unset == today byte-for-byte) + never-raise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm import leaf_actuator as la


@pytest.fixture(autouse=True)
def _clean_flags(monkeypatch):
    for f in (la.ENV_FLAG, la.MAX_COST_FLAG, la.SEEDS_FLAG, la.SEED_MAX_FLAG):
        monkeypatch.delenv(f, raising=False)
    yield


def _project(tmp_path: Path, cells=None, metrics=None) -> Path:
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)
    if cells is not None:
        (code / "cells.json").write_text(json.dumps({"cells": cells}))
    if metrics is not None:
        (code / "metrics.json").write_text(json.dumps(metrics))
    return tmp_path


def _cell(cid, mk="m", env="e", base="b", **extra):
    return {"id": cid, "model_key": mk, "env": env, "baseline": base,
            "params": {"lr": 0.05, "epochs": 200}, **extra}


# ---------------------------------------------------------------------------
# Flag accessors / default-OFF guarantee
# ---------------------------------------------------------------------------


def test_flags_default_off():
    assert la.is_enabled() is False
    assert la.max_cost() == "none"
    assert la.seeds_enabled() is False
    assert la.seed_max() == 5


def test_actuate_noop_when_master_flag_off(tmp_path):
    p = _project(tmp_path, cells=[_cell("c1", base="adam")])
    out = la.actuate([{"repair_class": "result_quality", "cost": "targeted_rerun"}], p)
    assert out == {"actuated": [], "artifact": {}, "summary": ""}
    assert not (p / "rlm_state" / la.STATE_FILE).exists()  # no file written


def test_max_cost_clamps_unknown_value(monkeypatch):
    monkeypatch.setenv(la.MAX_COST_FLAG, "bogus")
    assert la.max_cost() == "none"


# ---------------------------------------------------------------------------
# L5 — plan_seed_expansion (pure)
# ---------------------------------------------------------------------------


def test_seed_plan_fits_target():
    sp = la.plan_seed_expansion(current_seeds=1, paper_n=5, seed_max=5,
                                est_seconds_per_seed=600, remaining_s=4000)
    assert sp.affordable_seeds == 5 and sp.fits and sp.expand


def test_seed_plan_budget_capped_no_silent_cap():
    sp = la.plan_seed_expansion(current_seeds=1, paper_n=5, seed_max=5,
                                est_seconds_per_seed=600, remaining_s=1300)
    assert sp.affordable_seeds == 3 and not sp.fits and sp.expand
    assert "capped" in sp.reason  # the shortfall is named, never silent


def test_seed_plan_no_estimate_grants_target():
    sp = la.plan_seed_expansion(current_seeds=1, paper_n=5, seed_max=5,
                                est_seconds_per_seed=None, remaining_s=None)
    assert sp.affordable_seeds == 5 and sp.fits


def test_seed_plan_no_expand_when_already_at_target():
    sp = la.plan_seed_expansion(current_seeds=5, paper_n=5, seed_max=5,
                                est_seconds_per_seed=600, remaining_s=99999)
    assert not sp.expand and sp.fits


def test_seed_max_caps_paper_n():
    sp = la.plan_seed_expansion(current_seeds=1, paper_n=20, seed_max=3,
                                est_seconds_per_seed=None, remaining_s=None)
    assert sp.target_seeds == 3


def test_seed_plan_too_tight_no_expand():
    sp = la.plan_seed_expansion(current_seeds=1, paper_n=5, seed_max=5,
                                est_seconds_per_seed=9000, remaining_s=500)
    assert not sp.expand and not sp.fits


# ---------------------------------------------------------------------------
# L5 — _wants_variance + expand_cells_for_seeds (pure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "reports only a single seed; the paper averages over 5 seeds",
    "no error bars are shown across seeds",
    "mean ± std over 5 seeds is required",
    "the result uses n=5 seeds but we ran one",
])
def test_wants_variance_positive(text):
    assert la._wants_variance(text) is True


@pytest.mark.parametrize("text", [
    "the loss variance is high during training",
    "Adam ranked last, contradicting the paper",
    "the figure is missing",
    "",
])
def test_wants_variance_negative(text):
    assert la._wants_variance(text) is False


def test_expand_cells_for_seeds_replicates():
    cells = [_cell("c1", base="adam")]
    out = la.expand_cells_for_seeds(cells, 3)
    assert [c["id"] for c in out] == ["c1__seed0", "c1__seed1", "c1__seed2"]
    assert [c["seed"] for c in out] == [0, 1, 2]
    assert [c["params"]["seed"] for c in out] == [0, 1, 2]  # both shapes written


def test_expand_cells_for_seeds_respects_existing_seed():
    cells = [_cell("c1", base="adam", seed=42)]
    out = la.expand_cells_for_seeds(cells, 2)
    assert [c["seed"] for c in out] == [42, 43]


def test_expand_cells_for_seeds_noop_when_one():
    cells = [_cell("c1")]
    assert la.expand_cells_for_seeds(cells, 1) == cells
    assert la.expand_cells_for_seeds([], 5) == []


# ---------------------------------------------------------------------------
# Dispatcher — L4 / L6 / L5 staging (flag ON)
# ---------------------------------------------------------------------------


def test_actuate_l4_search_requires_cost_ceiling(tmp_path, monkeypatch):
    monkeypatch.setenv(la.ENV_FLAG, "1")  # ceiling defaults to "none"
    p = _project(tmp_path, cells=[_cell("c1", base="adam")])
    plan = [{"repair_class": "result_quality", "cost": "targeted_rerun", "leaf_id": "r1"}]
    out = la.actuate(plan, p)
    assert "result_quality" not in out["actuated"]  # dropped at ceiling=none
    monkeypatch.setenv(la.MAX_COST_FLAG, "targeted_rerun")
    out2 = la.actuate(plan, p)
    assert "result_quality" in out2["actuated"]
    assert len(out2["artifact"]["search"]) == 1


def test_actuate_l6_audit_fires_at_default_ceiling(tmp_path, monkeypatch):
    monkeypatch.setenv(la.ENV_FLAG, "1")  # ceiling none — L6 is cost none
    cells = [_cell("c1", base="adam"), _cell("c2", base="sgd")]
    metrics = {"status": "partial",
               "per_model": {"m": {"e": {"adam": {"status": "ok", "metric": 0.9}}}},
               "scope": {"gaps": []}}
    p = _project(tmp_path, cells=cells, metrics=metrics)
    out = la.actuate([{"repair_class": "aggregation_gap", "cost": "none"}], p)
    assert "aggregation_gap" in out["actuated"]
    assert out["artifact"]["aggregation_audit"]["unaccounted"] == ["c2"]


def test_actuate_l5_seed_plan_behind_subgate(tmp_path, monkeypatch):
    monkeypatch.setenv(la.ENV_FLAG, "1")
    p = _project(tmp_path, cells=[_cell("c1", base="adam")])
    weak = [{"id": "v1", "score": 0.4,
             "justification": "single seed only; paper reports mean±std over 5 seeds"}]
    # sub-gate OFF → no seed plan
    out = la.actuate([], p, weak_leaves=weak, est_seconds_per_seed=300, remaining_s=4000)
    assert "variance_gap" not in out["actuated"]
    # sub-gate ON → seed plan staged
    monkeypatch.setenv(la.SEEDS_FLAG, "1")
    out2 = la.actuate([], p, weak_leaves=weak, est_seconds_per_seed=300, remaining_s=4000)
    assert "variance_gap" in out2["actuated"]
    assert out2["artifact"]["seed_plan"]["affordable_seeds"] == 5


def test_actuate_persists_and_readers_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv(la.ENV_FLAG, "1")
    monkeypatch.setenv(la.MAX_COST_FLAG, "targeted_rerun")
    monkeypatch.setenv(la.SEEDS_FLAG, "1")
    cells = [_cell("c1", base="adam")]
    p = _project(tmp_path, cells=cells, metrics={"per_model": {}, "scope": {}})
    weak = [{"id": "v1", "score": 0.3, "justification": "no error bars across seeds"}]
    la.actuate([{"repair_class": "result_quality", "cost": "targeted_rerun"}], p,
               weak_leaves=weak, est_seconds_per_seed=None, remaining_s=None)
    assert (p / "rlm_state" / la.STATE_FILE).is_file()
    assert la.staged_search_override(p) is not None
    assert (la.seed_plan_for(p) or {}).get("affordable_seeds") == 5


def test_readers_guarded_by_flags(tmp_path, monkeypatch):
    # Persist an artifact, then prove the readers return None with flags off.
    monkeypatch.setenv(la.ENV_FLAG, "1")
    monkeypatch.setenv(la.MAX_COST_FLAG, "targeted_rerun")
    p = _project(tmp_path, cells=[_cell("c1", base="adam")])
    la.actuate([{"repair_class": "result_quality", "cost": "targeted_rerun"}], p)
    monkeypatch.delenv(la.ENV_FLAG, raising=False)
    assert la.staged_search_override(p) is None
    assert la.seed_plan_for(p) is None


# ---------------------------------------------------------------------------
# guidance_block + robustness
# ---------------------------------------------------------------------------


def test_guidance_block_empty_when_off(tmp_path, monkeypatch):
    p = _project(tmp_path)
    (p / "rlm_state").mkdir(exist_ok=True)
    (p / "rlm_state" / la.STATE_FILE).write_text(json.dumps(
        {"artifact": {"seed_plan": {"affordable_seeds": 5, "expand": True, "reason": "x"}}}))
    assert la.guidance_block(p) == ""  # master flag off
    monkeypatch.setenv(la.ENV_FLAG, "1")
    block = la.guidance_block(p)
    assert "LEAF ACTUATION" in block and "5 seeds" in block


def test_actuate_never_raises_on_garbage(tmp_path, monkeypatch):
    monkeypatch.setenv(la.ENV_FLAG, "1")
    out = la.actuate([None, "x", {}], tmp_path / "missing")  # no code/ dir
    assert isinstance(out, dict) and out["artifact"] == {}


def test_guidance_block_never_raises_on_garbage(tmp_path, monkeypatch):
    monkeypatch.setenv(la.ENV_FLAG, "1")
    (tmp_path / "rlm_state").mkdir()
    (tmp_path / "rlm_state" / la.STATE_FILE).write_text("{not json")
    assert la.guidance_block(tmp_path) == ""


# ---------------------------------------------------------------------------
# L2b — emit_figure_sidecars (deterministic render backstop, the fe5e7900 fix)
# ---------------------------------------------------------------------------

# Adam-shape per_model: scalar finals only (no per-step series on disk) — the
# exact shape that left fe5e7900 ("zero figure artifacts") at 0.0.
_ADAM_METRICS = {
    "per_model": {
        "mnist_logreg": {
            "mnist_logreg": {
                "adam": {"status": "ok", "metric": 37.06, "final_test_acc": 0.926},
                "adagrad": {"status": "ok", "metric": 37.04, "final_test_acc": 0.921},
                "sgd_nesterov": {"status": "ok", "metric": 36.94, "final_test_acc": 0.918},
            }
        }
    }
}

_RENDER_LEAF = [{"repair_class": "render_artifact", "cost": "none", "leaf_id": "fe5e7900"}]


def _read_sidecar(project: Path, rel: str) -> dict:
    return json.loads((project / rel).read_text())


def test_emit_figure_sidecars_grounded_comparison(tmp_path):
    p = _project(tmp_path, metrics=_ADAM_METRICS)
    written = la.emit_figure_sidecars(p, _RENDER_LEAF)
    assert len(written) == 1 and written[0].endswith(".json")
    assert "fig_auto_" in written[0]  # backstop-named, won't collide with agent figs
    sc = _read_sidecar(p, written[0])
    # the measured comparison reaches the (text-only) grader: series + axes present
    assert set(sc["series"]) == {"adam", "adagrad", "sgd_nesterov"}
    assert sc["series"]["adam"] == 37.06
    assert sc["y_axis"]["label"] == "metric" and "scale" in sc["y_axis"]
    assert "measured" in sc["note"].lower()  # honest: grounded, not fabricated


def test_emit_figure_sidecars_matches_grader_glob(tmp_path):
    """The written file must match the grader's ``fig_*.json`` rglob."""
    p = _project(tmp_path, metrics=_ADAM_METRICS)
    la.emit_figure_sidecars(p, _RENDER_LEAF)
    assert list((p / "code").rglob("fig_*.json"))  # _gather_figure_sidecars would find it


def test_emit_figure_sidecars_skips_when_agent_rendered(tmp_path):
    p = _project(tmp_path, metrics=_ADAM_METRICS)
    (p / "code" / "fig_loss_curves.json").write_text('{"figure":"real"}')  # agent's own
    assert la.emit_figure_sidecars(p, _RENDER_LEAF) == []  # don't pile on
    assert not list((p / "code").glob("fig_auto_*.json"))


def test_emit_figure_sidecars_empty_without_metrics(tmp_path):
    assert la.emit_figure_sidecars(_project(tmp_path), _RENDER_LEAF) == []  # no metrics
    assert la.emit_figure_sidecars(tmp_path / "missing", _RENDER_LEAF) == []  # no code/
    assert la.emit_figure_sidecars(_project(tmp_path, metrics=_ADAM_METRICS), []) == []  # no leaf


def test_emit_figure_sidecars_curve_mode_downsamples(tmp_path):
    metrics = {"per_model": {"m": {"e": {"adam": {"loss": list(range(200))}}}}}
    p = _project(tmp_path, metrics=metrics)
    written = la.emit_figure_sidecars(p, _RENDER_LEAF, max_points=40)
    sc = _read_sidecar(p, written[0])
    assert len(sc["series"]["adam"]) == 40  # downsampled to the cap
    assert sc["x_axis"]["label"] == "training step"
    assert sc["y_axis"]["scale"] == "log"  # 'loss' → log axis


def test_emit_figure_sidecars_grounded_skips_empty_group(tmp_path):
    # A group whose cells carry no numeric metric is skipped (grounded — never a
    # fabricated figure), while a sibling group with data still emits.
    metrics = {"per_model": {
        "empty": {"e": {"b": {"status": "failed", "error": "boom"}}},
        "real": {"e": {"b": {"metric": 0.9}}},
    }}
    p = _project(tmp_path, metrics=metrics)
    written = la.emit_figure_sidecars(p, _RENDER_LEAF)
    assert len(written) == 1 and "real" in written[0]


def test_actuate_l2b_render_fires_at_default_ceiling(tmp_path, monkeypatch):
    monkeypatch.setenv(la.ENV_FLAG, "1")  # ceiling none — render is cost none
    p = _project(tmp_path, metrics=_ADAM_METRICS)
    out = la.actuate(_RENDER_LEAF, p)
    assert "render_artifact" in out["actuated"]
    assert out["artifact"]["figure_sidecars"]
    assert list((p / "code").glob("fig_auto_*.json"))


def test_actuate_l2b_noop_when_master_flag_off(tmp_path):
    p = _project(tmp_path, metrics=_ADAM_METRICS)
    out = la.actuate(_RENDER_LEAF, p)  # flag off
    assert out == {"actuated": [], "artifact": {}, "summary": ""}
    assert not list((p / "code").glob("fig_auto_*.json"))


def test_emit_figure_sidecars_never_raises(tmp_path):
    assert la.emit_figure_sidecars(tmp_path, [None, "x", {}]) == []
    bad = _project(tmp_path, metrics={"per_model": {"m": "not a dict"}})
    assert la.emit_figure_sidecars(bad, _RENDER_LEAF) == []


# ---------------------------------------------------------------------------
# Seed-demand recognition + policy (2026-06-16 ResNet ceiling)
# The grader said "only 1 seed was run instead of the required 5" / "1 seed vs
# 5 runs" — DIGIT seed counts the old _VARIANCE_RE missed, so the variance
# leaves fell to "review" and the seed expansion never fired.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "only 1 seed was run for ResNet-110 instead of the required 5",
    "resnet_110 best_test_error_pct=6.82 vs paper's 6.43% (1 seed vs 5 runs)",
    "test error is reported as 'best (mean ± std)' over 5 independent runs",
    "single seed used, no error bars",
    "results averaged across 3 seeds",
    "best of 5 runs not reported",
])
def test_wants_variance_matches_real_phrasings(text):
    assert la._wants_variance(text) is True


@pytest.mark.parametrize("text", [
    "the proof in appendix B is paraphrased imprecisely",
    "batch size is only an assumption; epochs not confirmed",
    "the figure shows no training curve",
])
def test_wants_variance_rejects_unrelated(text):
    assert la._wants_variance(text) is False


def test_resolve_seed_demand_priority():
    # operator scope wins
    assert la.resolve_seed_demand(scope_seeds=[0, 1, 2], hint_seeds=[0]) == (3, "scope_spec")
    # hint when no operator scope
    assert la.resolve_seed_demand(hint_seeds=[0, 1, 2, 3, 4]) == (5, "paper_hint")
    # reactive variance leaf when neither
    n, src = la.resolve_seed_demand(
        weak_leaves=[{"score": 0.4, "justification": "only 1 seed run instead of 5"}])
    assert (n, src) == (5, "variance_leaf")
    # nothing demanded
    assert la.resolve_seed_demand() == (1, "none")
    # a single-element seed list is NOT a multi-seed demand
    assert la.resolve_seed_demand(scope_seeds=[0]) == (1, "none")
    # a STRONG variance leaf (>=0.6) does not trigger
    assert la.resolve_seed_demand(
        weak_leaves=[{"score": 0.8, "justification": "only 1 seed"}]) == (1, "none")


def test_resolve_seed_demand_never_raises_on_garbage():
    assert la.resolve_seed_demand(scope_seeds="x", hint_seeds=None,
                                  weak_leaves=[None, "s", {}]) == (1, "none")


def test_select_headline_models_deepest():
    cells = [
        {"model_key": "resnet_20", "depth": 20},
        {"model_key": "resnet_110", "depth": 110},
        {"model_key": "plain_56", "depth": 56},
    ]
    assert la.select_headline_models(cells) == {"resnet_110"}
    assert la.select_headline_models(cells, max_models=2) == {"resnet_110", "plain_56"}


def test_select_headline_models_explicit_intersect():
    cells = [{"model_key": "a", "depth": 1}, {"model_key": "b", "depth": 2}]
    assert la.select_headline_models(cells, explicit=["b", "zzz"]) == {"b"}
    # explicit with no overlap falls back to the heuristic (deepest)
    assert la.select_headline_models(cells, explicit=["zzz"]) == {"b"}


def test_select_headline_models_empty():
    assert la.select_headline_models([]) == set()
    assert la.select_headline_models([{"no_key": 1}]) == set()


def test_expand_cells_for_seeds_headline_only():
    cells = [
        {"id": "h", "model_key": "resnet_110", "seed": 42},
        {"id": "o", "model_key": "resnet_20", "seed": 42},
    ]
    out = la.expand_cells_for_seeds(cells, 3, model_keys={"resnet_110"})
    ids = [c["id"] for c in out]
    # headline replicated x3, non-headline passes through unchanged
    assert ids == ["h__seed42", "h__seed43", "h__seed44", "o"]
    assert all(c["seed"] == c["params"]["seed"] for c in out if "__seed" in c["id"])


def test_expand_cells_for_seeds_no_filter_replicates_all():
    cells = [{"id": "a", "model_key": "x", "seed": 0}, {"id": "b", "model_key": "y", "seed": 0}]
    out = la.expand_cells_for_seeds(cells, 2)
    assert len(out) == 4
