"""Fix G — confirming test: a harness-suffixed ablation baseline key
(e.g. ``sdar__sdar_ucb``) must NOT be falsely excluded from scoring.

The normalize_cell_axes helper suffixes the baseline key with the cell id
when three ablation cells would otherwise collide (same model_key/env/baseline
triple). The resulting key ``sdar__sdar_ucb`` tokenises to {sdar, ucb}, and a
leaf whose requirements text references the ``sdar_ucb`` variant also tokenises
to a superset of {sdar, ucb} — so the subject should NOT be marked as
"data unavailable" for that leaf.

If this test PASSES, Fix G is confirmed-handled (the harness already handles it
correctly). If it FAILS, that is a real bug — please flag before attempting a fix.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_metrics(run_dir: Path, metrics: dict) -> None:
    out = run_dir / "code" / "outputs" / "run1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


# The per_model tree the harness produces when cell axes are suffixed.
# Key: qwen2_5_3b → alfworld → sdar__sdar_ucb (harness-suffixed baseline)
def _metrics_with_suffixed_ablation() -> dict:
    return {
        "per_model": {
            "qwen2_5_3b": {
                "alfworld": {
                    "sdar__sdar_ucb": {
                        "status": "ok",
                        "sdar_reward": 0.62,
                        "baseline": "sdar_ucb",
                    }
                }
            }
        }
    }


# Leaf whose text specifically references the sdar_ucb variant — tokens: {sdar, ucb}.
SDAR_UCB_LEAF = {
    "id": "leaf_sdar_ucb",
    "requirements": (
        "The sdar_ucb ablation (UCB-based retrieval selection) achieves reward "
        "comparable to the full SDAR model on ALFWorld."
    ),
}

# A paper-wide leaf (no specific model/baseline reference) — must never be excluded.
GENERIC_LEAF = {
    "id": "leaf_generic",
    "requirements": "The SDAR training loss decreases across iterations.",
}


class TestSuffixedAblationBaselineMatching:
    """The harness-suffixed ablation key ``sdar__sdar_ucb`` must not exclude its leaf."""

    def test_sdar_ucb_leaf_not_excluded(self, tmp_path: Path):
        """_detect_data_unavailable_leaves must NOT mark the sdar_ucb leaf unavailable.

        The suffixed key sdar__sdar_ucb produces subject tokens {sdar, ucb} which are
        a subset of the leaf text tokens — the subject ran and the leaf is substantiated.

        If this FAILS, the leaf scorer falsely excludes a real ablation result and
        the sdar_ucb rubric leaf would score 0 by exclusion instead of by the grader.
        This is a real bug — do NOT attempt a silent fix; report it.
        """
        from backend.evals.paperbench.leaf_scorer import _detect_data_unavailable_leaves

        _write_metrics(tmp_path, _metrics_with_suffixed_ablation())

        leaves = [SDAR_UCB_LEAF, GENERIC_LEAF]
        unavailable = _detect_data_unavailable_leaves(leaves, tmp_path)

        # The sdar_ucb leaf must NOT be excluded — its ablation ran successfully.
        assert "leaf_sdar_ucb" not in unavailable, (
            "REAL BUG: sdar_ucb leaf was falsely excluded. The suffixed ablation key "
            "'sdar__sdar_ucb' is not being matched against the leaf's {sdar, ucb} tokens. "
            "This needs a fix in the leaf scorer — please flag before proceeding."
        )
        # The generic leaf must also not be excluded.
        assert "leaf_generic" not in unavailable

    def test_subject_tokens_include_sdar_ucb(self):
        """Token-level sanity: the suffixed key sdar__sdar_ucb yields {sdar, ucb} subject tokens.

        This is the mechanism Fix G relies on — verify it directly without I/O.
        """
        import re
        key = "sdar__sdar_ucb"
        toks = frozenset(t for t in re.split(r"[^a-z0-9]+", key.lower()) if t)
        # sdar__sdar_ucb → ["sdar", "sdar", "ucb"] → {sdar, ucb}
        assert "sdar" in toks
        assert "ucb" in toks

    def test_leaf_tokens_are_superset(self):
        """The leaf text tokens must be a superset of {sdar, ucb} so matching fires."""
        import re
        text = SDAR_UCB_LEAF["requirements"].lower()
        leaf_toks = frozenset(t for t in re.split(r"[^a-z0-9]+", text) if t)
        subject_toks = frozenset({"sdar", "ucb"})
        assert subject_toks.issubset(leaf_toks), (
            f"Leaf tokens {leaf_toks!r} must be a superset of subject tokens {subject_toks!r}"
        )
