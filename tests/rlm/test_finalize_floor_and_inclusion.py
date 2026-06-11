"""Floor-after-rescore + operator-inclusion-scope exclusion (2026-06-11).

All-CNN v3 shipped 0.694 although verify#1 had recorded 0.712 on the SAME
evidence: the finalize re-roll-up (anti-gaming) unconditionally superseded the
best-of-run floor even when no scope movement existed, letting pure grader
drift bury the high-water. And the run's three 0.0 ImageNet leaves stayed in
the denominator although the OPERATOR had scoped the run to CIFAR-10/100.
"""

from __future__ import annotations

from backend.evals.paperbench.leaf_scorer import (
    _detect_out_of_inclusion_scope_leaves,
)


def _leaf(lid, req):
    return {"id": lid, "requirements": req, "task_category": "", "sub_tasks": ""}


LEAVES = [
    _leaf("L1", "Train All-CNN-B on ImageNet and report Top-1 validation error"),
    _leaf("L2", "Measure CIFAR-10 test error for model C"),
    _leaf("L3", "Compare ImageNet Top-1 against the CIFAR-10 baseline"),  # mixed
    _leaf("L4", "Report parameter counts for each architecture"),         # no dataset
]


def test_flag_off_returns_empty(monkeypatch):
    monkeypatch.delenv("REPROLAB_SCOPE_INCLUSION_EXCLUDE", raising=False)
    assert _detect_out_of_inclusion_scope_leaves(LEAVES, ["CIFAR-10", "CIFAR-100"]) == set()


def test_out_of_scope_dataset_leaf_excluded(monkeypatch):
    monkeypatch.setenv("REPROLAB_SCOPE_INCLUSION_EXCLUDE", "1")
    out = _detect_out_of_inclusion_scope_leaves(LEAVES, ["CIFAR-10", "CIFAR-100"])
    assert out == {"L1"}  # pure-ImageNet leaf only


def test_mixed_and_datasetless_leaves_kept(monkeypatch):
    monkeypatch.setenv("REPROLAB_SCOPE_INCLUSION_EXCLUDE", "1")
    out = _detect_out_of_inclusion_scope_leaves(LEAVES, ["CIFAR-10", "CIFAR-100"])
    assert "L3" not in out  # mentions an in-scope dataset → kept
    assert "L4" not in out  # mentions no dataset → kept


def test_no_inclusion_list_means_no_exclusion(monkeypatch):
    monkeypatch.setenv("REPROLAB_SCOPE_INCLUSION_EXCLUDE", "1")
    assert _detect_out_of_inclusion_scope_leaves(LEAVES, []) == set()
    assert _detect_out_of_inclusion_scope_leaves(LEAVES, None) == set()


def test_digit_aware_matching(monkeypatch):
    monkeypatch.setenv("REPROLAB_SCOPE_INCLUSION_EXCLUDE", "1")
    # scope = CIFAR-100 only; a cifar10-only leaf is OUT of scope (cifar100 must
    # not cover cifar10), an imagenet leaf is out, a cifar100 leaf is in.
    leaves = [
        _leaf("A", "CIFAR-10 test error"),
        _leaf("B", "CIFAR-100 test error"),
        _leaf("C", "ImageNet Top-1"),
    ]
    out = _detect_out_of_inclusion_scope_leaves(leaves, ["CIFAR-100"])
    assert out == {"A", "C"}
