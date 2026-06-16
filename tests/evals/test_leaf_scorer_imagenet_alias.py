"""Issue #2 (2026-06-15): data-unavailable detection missed a downstream leaf.

The ImageNet load failure is logged as "ImageNet" but the All-CNN eval-protocol
leaf says "ILSVRC-2012" — {imagenet} shares ZERO tokens with the leaf, so the
token-superset pass scored it 0.0 instead of excluding it (~0.05 lost on All-CNN).

Fix = a curated true-synonym alias (imagenet↔ilsvrc) + a tight overlap tier, BOTH
gated by an evidence guard: a leaf naming a subject that successfully ran is never
excluded. These tests pin the new behavior and the guard's safety.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.evals.paperbench.leaf_scorer import _detect_data_unavailable_leaves


def _write(run_dir: Path, metrics: dict, report: dict | None = None) -> None:
    out = run_dir / "code" / "outputs" / "run-x"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if report is not None:
        (run_dir / "final_report.json").write_text(json.dumps(report), encoding="utf-8")


# Leaves: three reference ImageNet (one by the "ILSVRC" synonym, the bug), one is
# an in-scope CIFAR leaf that must stay scored.
_IMAGENET_TRAIN = {
    "id": "imagenet-train",
    "requirements": "The ImageNet run trains the upscaled All-CNN-B for 450000 iterations.",
}
_IMAGENET_RESULT = {
    "id": "imagenet-result",
    "requirements": "The reproduced ImageNet All-CNN-B achieves Top-1 validation error near 41.2%.",
}
_ILSVRC_PROTOCOL = {
    "id": "ilsvrc-protocol",
    "requirements": (
        "ILSVRC-2012 performance is measured as Top-1 validation error evaluated only "
        "on the center 224x224 patch of each validation image."
    ),
}
_CIFAR_LEAF = {
    "id": "cifar-base-a",
    "requirements": "Reproduced base model A achieves ~12.5% test error on CIFAR-10 without augmentation.",
}
ALL_LEAVES = [_IMAGENET_TRAIN, _IMAGENET_RESULT, _ILSVRC_PROTOCOL, _CIFAR_LEAF]


def test_ilsvrc_protocol_leaf_excluded_via_alias(tmp_path):
    """The ILSVRC-2012 leaf (zero token overlap with 'imagenet') is now excluded."""
    _write(
        tmp_path,
        {
            "data_load_failures": [{"dataset": "ImageNet", "error": "licence gated"}],
            "per_model": {"a_base": {"cifar10_noaug": {"base": {"status": "ok", "metric": 0.84}}}},
        },
        {"verdict": "partial", "scope": {"gaps": ["ImageNet: out of scope — licence gated"]}},
    )
    result = _detect_data_unavailable_leaves(ALL_LEAVES, tmp_path)
    # All three ImageNet/ILSVRC leaves excluded — including the synonym one.
    assert "imagenet-train" in result
    assert "imagenet-result" in result
    assert "ilsvrc-protocol" in result, "the ILSVRC-2012 synonym leaf must be excluded (the bug)"
    # The in-scope CIFAR leaf is never excluded by an ImageNet failure.
    assert "cifar-base-a" not in result


def test_ilsvrc_excluded_via_scope_gaps_prose_only(tmp_path):
    """The exact ce9caf production scenario: ImageNet declared ONLY in scope.gaps
    prose ('ImageNet All-CNN-B: requires licensed manual download …'), no
    data_load_failures entry. The alias must still bridge to the ILSVRC leaf."""
    _write(
        tmp_path,
        {"per_model": {"a_base": {"cifar10_noaug": {"base": {"status": "ok", "metric": 0.84}}}}},
        {
            "verdict": "partial",
            "scope": {
                "gaps": [
                    "ImageNet All-CNN-B: requires licensed manual download from "
                    "image-net.org — unobtainable in sandbox"
                ]
            },
        },
    )
    result = _detect_data_unavailable_leaves(ALL_LEAVES, tmp_path)
    assert "imagenet-train" in result
    assert "imagenet-result" in result
    assert "ilsvrc-protocol" in result, "alias must fire from scope.gaps prose, not just data_load_failures"
    assert "cifar-base-a" not in result


def test_reverse_alias_ilsvrc_failure_excludes_imagenet_leaf(tmp_path):
    """Symmetry: a failure logged as 'ILSVRC' excludes a leaf that says 'ImageNet'."""
    _write(
        tmp_path,
        {"data_load_failures": [{"dataset": "ILSVRC-2012", "error": "403"}]},
        {"verdict": "partial"},
    )
    result = _detect_data_unavailable_leaves([_IMAGENET_TRAIN, _CIFAR_LEAF], tmp_path)
    assert "imagenet-train" in result
    assert "cifar-base-a" not in result


def test_evidence_guard_blocks_exclusion_of_a_ran_subject(tmp_path):
    """A leaf naming a SUCCESSFUL subject is never excluded by the loosened tiers.

    Even though 'ilsvrc' is an unavailable signal, a leaf that also names a subject
    with an ok on-disk metric (here the ilsvrc cell actually produced a result) is
    scored, not skipped — the guard prevents laundering a ran result into a skip.
    """
    _write(
        tmp_path,
        {
            "data_load_failures": [{"dataset": "ImageNet", "error": "gated"}],
            # The protocol leaf's subject DID run (status ok) → must be graded.
            "per_model": {"ilsvrc": {"val": {"center_patch": {"status": "ok", "metric": 0.59}}}},
        },
        {"verdict": "partial"},
    )
    guarded_leaf = {
        "id": "ilsvrc-ran",
        "requirements": "ilsvrc val center_patch Top-1 error measured.",
    }
    result = _detect_data_unavailable_leaves([guarded_leaf], tmp_path)
    assert "ilsvrc-ran" not in result, "a leaf with a successful on-disk metric must be scored"


def test_alias_does_not_exclude_allcnn_or_cifar_leaves(tmp_path):
    """REGRESSION GUARD (the over-exclusion the re-grade caught): a gap prose that
    mixes the dataset with the model name ('ImageNet All-CNN-B') must NOT cause any
    All-CNN-C / CIFAR / ordering leaf to be excluded — only the literal-ILSVRC leaf.
    Excluding the central fidelity leaves would GAME the score."""
    _write(
        tmp_path,
        {"per_model": {"c_allcnn": {"cifar10_noaug": {"base": {"status": "ok", "metric": 0.85}}}}},
        {"verdict": "partial", "scope": {"gaps": ["ImageNet All-CNN-B: unobtainable in sandbox"]}},
    )
    leaves = [
        _ILSVRC_PROTOCOL,
        {"id": "allcnn-c-cifar", "requirements": "Reproduced All-CNN-C test error on CIFAR-10 is at or below base-model-C."},
        {"id": "table3-order", "requirements": "Reproduce the qualitative ordering of Table 3: Strided-CNN-C vs All-CNN-C."},
        {"id": "cifar100", "requirements": "The CIFAR-100 experiment reuses the identical All-CNN-C model."},
        {"id": "allcnn-impl", "requirements": "The All-CNN-C model is implemented exactly as in Table 2."},
    ]
    result = _detect_data_unavailable_leaves(leaves, tmp_path)
    assert "ilsvrc-protocol" in result, "the literal ILSVRC leaf is still excluded"
    for keep in ("allcnn-c-cifar", "table3-order", "cifar100", "allcnn-impl"):
        assert keep not in result, f"{keep} must stay scored — excluding it games the central claim"
