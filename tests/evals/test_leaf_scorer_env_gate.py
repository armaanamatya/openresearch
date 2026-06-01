"""Environment-axis anti-gaming gate in the leaf scorer (2026-06-01).

A VERIFIED env exclusion (operator scope / harness-confirmed) excludes its leaves
from numerator AND denominator; an UNVERIFIED, agent-declared env skip does NOT —
it stays scored, so a broad ``except`` cannot launder a real failure into a free
scope reduction. This mirrors the model-axis ``operator_skip_models`` behaviour
and closes the hole where ``environments_skipped`` was honoured unconditionally.

Legacy runs (no ``scope.exclusions`` and no ``operator_skip_environments``) keep
the prior lenient behaviour — pinned by test_leaf_scorer_scope_exclusion.py, which
still passes. Fixtures here build ``scope`` through the real
``exclusion.build_scope_block`` so we test the true producer→consumer contract.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm import exclusion as X
from backend.evals.paperbench.leaf_scorer import _detect_data_unavailable_leaves

LEAVES = [
    {"id": "leaf_alfworld", "requirements": "ALFWorld success rate reaches 53.9% on Qwen3-1.7B"},
    {"id": "leaf_webshop", "requirements": "WebShop training uses 1000 tasks and a 128-task val set"},
    {"id": "leaf_core", "requirements": "The sigmoid gate g_t uses beta=10 with stop-gradient on the gate"},
]


def _write(run_dir: Path, scope: dict) -> None:
    out = run_dir / "code" / "outputs" / "run1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"status": "partial", "scope": scope}), encoding="utf-8")


def _env(item: str, verified: bool, kind: str = "operator_scope") -> X.Exclusion:
    return X.Exclusion(item=item, axis="environment", kind=kind, reason="r", verified=verified)


def test_verified_env_exclusions_excluded(tmp_path: Path):
    scope = X.build_scope_block([_env("ALFWorld", True), _env("WebShop", True)])
    _write(tmp_path, scope)
    skip = _detect_data_unavailable_leaves(LEAVES, tmp_path)
    assert {"leaf_alfworld", "leaf_webshop"} <= skip
    assert "leaf_core" not in skip


def test_unverified_env_exclusion_stays_scored(tmp_path: Path):
    # Anti-gaming: an agent-declared (verified=False) env skip is recorded but NOT
    # excluded — structured exclusions present ⇒ gate enforced ⇒ WebShop stays scored.
    scope = X.build_scope_block([_env("WebShop", False, kind="env_setup_failed")])
    assert scope["environments_skipped"] == []          # not in the derived skip list
    assert {e["item"] for e in scope["exclusions"]} == {"WebShop"}  # but recorded
    _write(tmp_path, scope)
    skip = _detect_data_unavailable_leaves(LEAVES, tmp_path)
    assert "leaf_webshop" not in skip
    assert skip == set()


def test_mixed_verified_and_unverified(tmp_path: Path):
    scope = X.build_scope_block([_env("ALFWorld", True), _env("WebShop", False, kind="env_setup_failed")])
    _write(tmp_path, scope)
    skip = _detect_data_unavailable_leaves(LEAVES, tmp_path)
    assert "leaf_alfworld" in skip          # verified → excluded
    assert "leaf_webshop" not in skip       # unverified → stays scored
    assert "leaf_core" not in skip


def test_operator_skip_environments_param_gates_legacy_list(tmp_path: Path):
    # No structured exclusions, but an explicit operator list enables the gate.
    _write(tmp_path, {"environments_skipped": ["ALFWorld", "WebShop"]})
    skip = _detect_data_unavailable_leaves(LEAVES, tmp_path, operator_skip_environments=["ALFWorld"])
    assert "leaf_alfworld" in skip
    assert "leaf_webshop" not in skip       # requested-but-not-operator-de-scoped → repairable


def test_verified_model_exclusion_via_structured(tmp_path: Path):
    leaves = [
        {"id": "leaf_7b", "requirements": "Qwen2.5-7B reaches 46% accuracy on Search-QA"},
        {"id": "leaf_core", "requirements": "sigmoid gate beta=10 stop-gradient"},
    ]
    scope = X.build_scope_block([X.Exclusion(item="Qwen2.5-7B", axis="model",
                                             kind="operator_scope", reason="r", verified=True)])
    assert scope["models_skipped"] == ["Qwen2.5-7B"]
    _write(tmp_path, scope)
    skip = _detect_data_unavailable_leaves(leaves, tmp_path)
    assert "leaf_7b" in skip
    assert "leaf_core" not in skip
