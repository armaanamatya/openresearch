"""Unit tests for the deterministic pre-GPU-lease feasibility gate (Phase 1b, Tasks 10+11).

Cost model (``est_train_seconds``/``estimate_scope_cost``) + 3-axis triage
(``FeasibilityTriage``/``TriageDecision``). Hermetic: no network, no LLM, no
real GPU.
"""

from backend.services.runtime.feasibility_triage import (
    FeasibilityTriage,
    TriageDecision,  # noqa: F401 — public return type of triage(), documents the interface under test
    est_train_seconds,
    estimate_scope_cost,
)
from backend.services.runtime.gpu_catalog import find_by_alias
from backend.services.runtime.run_plan import RunPlan, RequiredAsset
from backend.agents.resilience.budget import RunBudget
from backend.agents.schemas import ScopeSpec

_SKU = find_by_alias("l4")


# ---------------------------------------------------------------------------
# Task 10: cost model
# ---------------------------------------------------------------------------


def test_est_train_seconds_monotonic_in_steps_and_size():
    assert est_train_seconds("qwen3-1.7b", 400) < est_train_seconds("qwen3-1.7b", 800)
    assert est_train_seconds("qwen3-1.7b", 400) < est_train_seconds("qwen2.5-7b", 400)
    assert est_train_seconds("unknown-model", 100) > 0.0          # conservative default, never 0


def test_estimate_scope_cost_scales_with_cells():
    sku = find_by_alias("l4")                                # approx_usd_per_hr known
    small = ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0])
    big = ScopeSpec(models=["qwen3-1.7b", "qwen2.5-3b"],
                    datasets=[{"name": "alfworld"}, {"name": "webshop"}], seeds=[0, 1])
    gh_s, usd_s = estimate_scope_cost(small, sku, steps=400)
    gh_b, usd_b = estimate_scope_cost(big, sku, steps=400)
    assert gh_b > gh_s and usd_b > usd_s
    assert usd_s == round(gh_s * sku.approx_usd_per_hr, 4)         # usd derives from gpu_hours × $/hr
    # 8 cells (2×2×2) vs 1 cell → ~8× the compute
    assert gh_b > 6 * gh_s


def test_estimate_scope_cost_empty_scope_is_zero():
    sku = find_by_alias("l4")
    assert estimate_scope_cost(ScopeSpec(), sku, steps=400) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Task 11: FeasibilityTriage (3-axis decision)
# ---------------------------------------------------------------------------


def test_proceed_when_reachable_and_within_budget():
    plan = RunPlan(scope=ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
                   budget=RunBudget(max_gpu_hours=100.0),
                   required_assets=(RequiredAsset("dataset", "alfworld"),))
    t = FeasibilityTriage(reachability_probe=lambda a: "reachable")
    d = t.triage(plan, _SKU)
    assert d.decision == "PROCEED"


def test_plan_only_when_asset_gated_no_cred():
    plan = RunPlan(scope=ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "gated-ds"}], seeds=[0]),
                   budget=RunBudget(max_gpu_hours=100.0),
                   required_assets=(RequiredAsset("dataset", "gated-ds", gated=True),))
    t = FeasibilityTriage(reachability_probe=lambda a: "gated")
    d = t.triage(plan, _SKU)
    assert d.decision == "PLAN_ONLY" and any("gated" in r for r in d.reasons)


def test_down_scope_when_over_budget_but_trimmable():
    plan = RunPlan(
        scope=ScopeSpec(models=["qwen2.5-7b", "qwen3-1.7b"],
                        datasets=[{"name": "alfworld"}, {"name": "webshop"}], seeds=[0, 1, 2]),
        budget=RunBudget(max_gpu_hours=0.5),          # tight → must trim
        required_assets=(RequiredAsset("dataset", "alfworld"), RequiredAsset("dataset", "webshop")))
    t = FeasibilityTriage(reachability_probe=lambda a: "reachable")
    d = t.triage(plan, _SKU)
    assert d.decision == "DOWN_SCOPE"
    assert len(d.scope.models) < 2 or len(d.scope.seeds) < 3       # trimmed to fit


def test_plan_only_when_infeasible_even_minimal():
    plan = RunPlan(scope=ScopeSpec(models=["qwen2.5-7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
                   budget=RunBudget(max_gpu_hours=0.0001),          # nothing fits
                   required_assets=(RequiredAsset("dataset", "alfworld"),))
    t = FeasibilityTriage(reachability_probe=lambda a: "reachable")
    d = t.triage(plan, _SKU)
    assert d.decision == "PLAN_ONLY"
