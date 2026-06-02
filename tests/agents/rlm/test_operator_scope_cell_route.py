"""Cell-route operator-scope minting (primitives._operator_scope_exclusions /
_apply_operator_scope, 2026-06-01).

The cell route synthesises metrics.json from cells.json, so operator-de-scoped
axes (smallest-two: ALFWorld/WebShop never enter cells.json) must be re-declared
as verified operator_scope exclusions or the leaf scorer scores their leaves 0.
These helpers turn ``ScopeSpec.skip_models`` / ``skip_datasets`` into verified
Exclusions and fold them (plus the recovered capacity/dataset gate gaps) into
``metrics.scope`` via the shared contract.
"""
from __future__ import annotations

from types import SimpleNamespace

from backend.agents.rlm.primitives import _apply_operator_scope, _operator_scope_exclusions


def _ctx(skip_models=None, skip_datasets=None, has_spec=True):
    spec = None
    if has_spec:
        spec = SimpleNamespace(skip_models=skip_models or [], skip_datasets=skip_datasets or [])
    return SimpleNamespace(scope_spec=spec)


def test_operator_scope_exclusions_from_scope_spec():
    exs = _operator_scope_exclusions(_ctx(skip_models=["Qwen2.5-7B"], skip_datasets=["ALFWorld", "WebShop"]))
    pairs = {(e.axis, e.item) for e in exs}
    assert ("model", "Qwen2.5-7B") in pairs
    assert ("environment", "ALFWorld") in pairs and ("environment", "WebShop") in pairs
    assert all(e.verified and e.kind == "operator_scope" for e in exs)


def test_operator_scope_exclusions_empty_without_scope_spec():
    assert _operator_scope_exclusions(_ctx(has_spec=False)) == []
    assert _operator_scope_exclusions(SimpleNamespace()) == []  # no attr at all → []


def test_apply_operator_scope_merges_and_recovers_gate_gaps():
    ctx = _ctx(skip_datasets=["ALFWorld"])
    metrics = {
        "status": "partial",
        "per_model": {"qwen3_1_7b": {"Search-QA": {"sdar": {"status": "ok", "metric": 0.12}}}},
        "scope": {
            "models_run": ["qwen3_1_7b"],
            "environments_skipped": [],
            "gaps": [{"item": "qwen2_5_7b", "kind": "capacity", "reason": "needs ~40GB > 24GB"}],
        },
    }
    out = _apply_operator_scope(metrics, ctx)
    sc = out["scope"]
    assert "ALFWorld" in sc["environments_skipped"]            # operator env de-scope applied
    assert sc["models_run"] == ["qwen3_1_7b"]                  # preserved
    kinds = {e["kind"] for e in sc["exclusions"]}
    assert "operator_scope" in kinds and "capacity_vram" in kinds  # recovered gate gap promoted
    assert out["per_model"] == metrics["per_model"]            # untouched


def test_apply_operator_scope_noop_without_skips():
    ctx = _ctx()  # empty skip lists
    metrics = {"scope": {"models_run": ["A"]}}
    out = _apply_operator_scope(dict(metrics), ctx)
    assert out["scope"]["models_run"] == ["A"]
    assert "exclusions" not in out["scope"]  # nothing minted → scope untouched


def test_apply_operator_scope_failsoft_on_bad_metrics():
    # A non-dict scope must not raise — enrichment is best-effort.
    ctx = _ctx(skip_datasets=["ALFWorld"])
    out = _apply_operator_scope({"scope": None}, ctx)
    assert "ALFWorld" in out["scope"]["environments_skipped"]
