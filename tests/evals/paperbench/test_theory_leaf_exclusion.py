"""Layer 3 — theory-only leaf exclusion. A code reproduction cannot PROVE a
theorem, so such leaves are excluded from the roll-up denominator (like
data-unavailable leaves), not scored 0.0. Detection is on the rubric TEXT only
(no gaming surface); flag-gated (default OFF); conservative — a leaf that ALSO
asks for an empirical artifact stays gradeable.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.evals.paperbench.leaf_scorer import (
    _detect_theory_only_leaves,
    score_reproduction,
)


def _leaf(lid: str, req: str, cat: str = "") -> dict:
    return {"id": lid, "requirements": req, "task_category": cat}


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("REPROLAB_EXCLUDE_THEORY_LEAVES", raising=False)
    leaves = [_leaf("t1", "Prove the O(√T) regret bound (Theorem 4.1).")]
    assert _detect_theory_only_leaves(leaves) == set()


def test_excludes_pure_theory_when_enabled(monkeypatch):
    monkeypatch.setenv("REPROLAB_EXCLUDE_THEORY_LEAVES", "1")
    leaves = [
        _leaf("t1", "Prove the O(√T) regret bound stated in Theorem 4.1."),
        _leaf("t2", "Reproduce the lemma establishing the convergence guarantee."),
        _leaf("e1", "Implement Adam and report MNIST accuracy in metrics.json."),
    ]
    assert _detect_theory_only_leaves(leaves) == {"t1", "t2"}


def test_keeps_empirically_gradeable_theory_leaf(monkeypatch):
    monkeypatch.setenv("REPROLAB_EXCLUDE_THEORY_LEAVES", "1")
    # mentions "theorem" but ALSO asks for a loss-curve plot → a reproduction CAN satisfy it
    leaves = [_leaf("m1", "Verify the convergence theorem empirically via the loss curve plot.")]
    assert _detect_theory_only_leaves(leaves) == set()


def test_score_excludes_theory_from_denominator(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REPROLAB_EXCLUDE_THEORY_LEAVES", "1")
    run = tmp_path / "run"
    (run / "code").mkdir(parents=True)
    (run / "code" / "train.py").write_text("print('x')\n", encoding="utf-8")
    (run / "final_report.json").write_text(
        json.dumps({"baseline_metrics": {"acc": 0.9}, "verdict": "reproduced"}),
        encoding="utf-8",
    )
    tree = {
        "id": "root", "weight": 1.0,
        "sub_tasks": [
            {"id": "e1", "weight": 1.0,
             "requirements": "Implement Adam and report accuracy in metrics.json."},
            {"id": "t1", "weight": 1.0,
             "requirements": "Prove the O(√T) regret bound (Theorem 4.1)."},
        ],
    }

    class _Stub:
        def complete(self, *, system, user):
            # Only the EMPIRICAL leaf should ever be presented for grading; grade it 1.0.
            return json.dumps([{"leaf_id": "e1", "score": 1.0, "justification": "metrics.json acc=0.9"}])

    res = score_reproduction(tree, run, _Stub(), degraded=False)
    # t1 (theory) excluded from both numerator AND denominator → overall == e1's 1.0,
    # NOT (1.0 + 0.0)/2 = 0.5 which is what un-excluded scoring would give.
    assert res["overall_score"] == 1.0
