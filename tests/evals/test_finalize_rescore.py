"""Phase 0B — deterministic finalize re-roll-up (``finalize_rescore``).

Re-rolls-up the ALREADY-GRADED leaves under a FINAL scope that may have been
declared AFTER the last in-loop verify, routed through the environment-axis
anti-gaming gate (Phase 4B). It NEVER re-grades (reuses persisted leaf scores)
and NEVER maxes across exclusion policies — it is the authoritative finalize
number. A non-operator-sanctioned env skip stays SCORED.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.evals.paperbench.leaf_scorer import finalize_rescore

TREE = {
    "id": "root",
    "sub_tasks": [
        {
            "id": "area_exec",
            "weight": 1.0,
            "sub_tasks": [
                {"id": "leaf_alf", "weight": 1.0,
                 "requirements": "ALFWorld success rate reaches 53.9% on Qwen3-1.7B"},
                {"id": "leaf_a", "weight": 1.0,
                 "requirements": "Search-QA exact match is logged per step"},
                {"id": "leaf_b", "weight": 1.0,
                 "requirements": "The sigmoid gate uses beta=10 with stop-gradient"},
            ],
        },
    ],
}

RECORDS = [
    {"id": "leaf_alf", "score": 0.0, "justification": "ALFWorld out of scope"},
    {"id": "leaf_a", "score": 0.6, "justification": "ok"},
    {"id": "leaf_b", "score": 0.6, "justification": "ok"},
]


def _setup(run_dir: Path, overall: float = 0.4) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rubric_tree.json").write_text(json.dumps(TREE), encoding="utf-8")
    (run_dir / "rubric_evaluation.json").write_text(
        json.dumps({"overall_score": overall, "leaf_scores": RECORDS}), encoding="utf-8")


def test_no_exclusion_reproduces_recorded_score(tmp_path: Path):
    _setup(tmp_path)
    out = finalize_rescore(tmp_path)
    assert out is not None
    assert abs(out["overall_score"] - 0.4) < 1e-9     # (0.0 + 0.6 + 0.6) / 3
    assert out["n_excluded"] == 0
    assert out["policy"] == "finalize_rescore"
    assert abs(out["prior_overall"] - 0.4) < 1e-9


def test_verified_env_exclusion_lifts_score(tmp_path: Path):
    _setup(tmp_path)
    extra = {"exclusions": [{"item": "ALFWorld", "axis": "environment",
                             "kind": "operator_scope", "verified": True, "reason": "r"}]}
    out = finalize_rescore(tmp_path, extra_scope=extra)
    assert out is not None
    assert "leaf_alf" in out["excluded_leaf_ids"]
    assert abs(out["overall_score"] - 0.6) < 1e-9     # (0.6 + 0.6) / 2 over survivors


def test_unverified_env_skip_stays_scored(tmp_path: Path):
    # Anti-gaming: an agent-declared (verified=False) env exclusion is NOT honoured
    # — the leaf stays in the denominator, score unchanged. This is the laundering
    # hole that must stay closed at finalize too.
    _setup(tmp_path)
    extra = {"exclusions": [{"item": "ALFWorld", "axis": "environment",
                             "kind": "env_setup_failed", "verified": False, "reason": "agent"}]}
    out = finalize_rescore(tmp_path, extra_scope=extra)
    assert out is not None
    assert "leaf_alf" not in out["excluded_leaf_ids"]
    assert abs(out["overall_score"] - 0.4) < 1e-9


def test_operator_skip_environments_gates_legacy_list(tmp_path: Path):
    # The finalize wiring always passes a (possibly empty) operator list, which
    # ENFORCES the gate: a legacy environments_skipped entry is honoured only when
    # operator-sanctioned, mirroring the model axis.
    _setup(tmp_path)
    extra = {"environments_skipped": ["ALFWorld"]}
    # Gate enforced (empty list, not None) but ALFWorld not sanctioned → stays scored.
    out = finalize_rescore(tmp_path, extra_scope=extra, operator_skip_environments=[])
    assert "leaf_alf" not in out["excluded_leaf_ids"]
    assert abs(out["overall_score"] - 0.4) < 1e-9
    # Operator-sanctioned (skip_datasets → operator_skip_environments) → excluded.
    out2 = finalize_rescore(tmp_path, extra_scope=extra, operator_skip_environments=["ALFWorld"])
    assert "leaf_alf" in out2["excluded_leaf_ids"]
    assert abs(out2["overall_score"] - 0.6) < 1e-9


def test_failsoft_when_no_eval_or_tree(tmp_path: Path):
    # No rubric_evaluation.json → None (caller keeps today's recorded score).
    (tmp_path / "rubric_tree.json").write_text(json.dumps(TREE), encoding="utf-8")
    assert finalize_rescore(tmp_path) is None
    # eval present but no tree recoverable → None.
    other = tmp_path / "run2"
    other.mkdir()
    (other / "rubric_evaluation.json").write_text(
        json.dumps({"overall_score": 0.4, "leaf_scores": RECORDS}), encoding="utf-8")
    assert finalize_rescore(other) is None


def test_explicit_rubric_tree_arg_is_used(tmp_path: Path):
    # When no rubric_tree.json is on disk, an explicit tree arg still works.
    run = tmp_path / "run3"
    run.mkdir()
    (run / "rubric_evaluation.json").write_text(
        json.dumps({"overall_score": 0.4, "leaf_scores": RECORDS}), encoding="utf-8")
    out = finalize_rescore(run, rubric_tree=TREE)
    assert out is not None
    assert abs(out["overall_score"] - 0.4) < 1e-9
