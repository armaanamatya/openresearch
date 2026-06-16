"""Floor-after-rescore + operator-inclusion-scope exclusion (2026-06-11).

All-CNN v3 shipped 0.694 although verify#1 had recorded 0.712 on the SAME
evidence: the finalize re-roll-up (anti-gaming) unconditionally superseded the
best-of-run floor even when no scope movement existed, letting pure grader
drift bury the high-water. And the run's three 0.0 ImageNet leaves stayed in
the denominator although the OPERATOR had scoped the run to CIFAR-10/100.
"""

from __future__ import annotations

import inspect

from backend.evals.paperbench.leaf_scorer import (
    _detect_out_of_inclusion_scope_leaves,
    score_reproduction,
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
    monkeypatch.delenv("OPENRESEARCH_SCOPE_INCLUSION_EXCLUDE", raising=False)
    assert _detect_out_of_inclusion_scope_leaves(LEAVES, ["CIFAR-10", "CIFAR-100"]) == set()


def test_out_of_scope_dataset_leaf_excluded(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SCOPE_INCLUSION_EXCLUDE", "1")
    out = _detect_out_of_inclusion_scope_leaves(LEAVES, ["CIFAR-10", "CIFAR-100"])
    assert out == {"L1"}  # pure-ImageNet leaf only


def test_mixed_and_datasetless_leaves_kept(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SCOPE_INCLUSION_EXCLUDE", "1")
    out = _detect_out_of_inclusion_scope_leaves(LEAVES, ["CIFAR-10", "CIFAR-100"])
    assert "L3" not in out  # mentions an in-scope dataset → kept
    assert "L4" not in out  # mentions no dataset → kept


def test_no_inclusion_list_means_no_exclusion(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SCOPE_INCLUSION_EXCLUDE", "1")
    assert _detect_out_of_inclusion_scope_leaves(LEAVES, []) == set()
    assert _detect_out_of_inclusion_scope_leaves(LEAVES, None) == set()


def test_digit_aware_matching(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SCOPE_INCLUSION_EXCLUDE", "1")
    # scope = CIFAR-100 only; a cifar10-only leaf is OUT of scope (cifar100 must
    # not cover cifar10), an imagenet leaf is out, a cifar100 leaf is in.
    leaves = [
        _leaf("A", "CIFAR-10 test error"),
        _leaf("B", "CIFAR-100 test error"),
        _leaf("C", "ImageNet Top-1"),
    ]
    out = _detect_out_of_inclusion_scope_leaves(leaves, ["CIFAR-100"])
    assert out == {"A", "C"}


# --- the REAL ResNet (prj_4627097f8362928c) out-of-scope leaves -----------------
# Ground truth, 2026-06-16: 10 of 20 leaves were ImageNet/COCO/bottleneck on a
# CIFAR-10-scoped run, scored 0.0 in the in-loop verify (rubric_evaluation.json
# 0.368) but excluded at finalize (final_report 0.620). These assert the detector
# covers every real shape so the IN-LOOP path (now Layer-4 wired) excludes them too.
RESNET_OOS = [
    _leaf("imagenet_train", "Train ResNet on ImageNet (256 mini-batch, 60e4 iters)"),
    _leaf("imagenet_topk", "Report ImageNet top-1 and top-5 validation error"),
    _leaf("coco_detect", "Evaluate Faster R-CNN on MS COCO and report mAP"),
    _leaf("bottleneck", "Implement 3-layer bottleneck for ResNet-50/101/152 on ImageNet"),
    _leaf("tencrop", "10-crop / multi-scale ImageNet inference at {224,256,384}"),
    _leaf("cifar_ok", "Train ResNet-110 on CIFAR-10 and report test error"),  # in scope
    _leaf("params", "Report the parameter count for each architecture"),       # datasetless
]


def test_real_resnet_imagenet_coco_leaves_excluded(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SCOPE_INCLUSION_EXCLUDE", "1")
    out = _detect_out_of_inclusion_scope_leaves(RESNET_OOS, ["CIFAR-10"])
    assert out == {"imagenet_train", "imagenet_topk", "coco_detect",
                   "bottleneck", "tencrop"}
    assert "cifar_ok" not in out      # in-scope dataset → never excluded
    assert "params" not in out        # no dataset named → kept


def test_score_reproduction_wires_inclusion_param():
    # The in-loop grade path (score_reproduction) must accept the operator
    # inclusion list so Layer 4 fires in-loop, not only at finalize — otherwise
    # the agent sees un-fixable out-of-scope leaves as "weak" (the ResNet 0.368
    # vs 0.620 split). Signature-level proof the plumbing exists.
    assert "operator_dataset_inclusion" in inspect.signature(score_reproduction).parameters
