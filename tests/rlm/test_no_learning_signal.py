"""
Tests for backend/agents/rlm/no_learning_signal.py (F3 gate).

Hermetic: uses tmp_path + monkeypatch.setenv; no network, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm.no_learning_signal import (
    _leaf_no_learning,
    detect_no_learning_signal,
    no_learning_repair_message,
    no_learning_signal_enabled,
)
from backend.agents.rlm.reproducibility_verdict import (
    FidelityCertificate,
    compute_reproducibility_verdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _green_cert() -> FidelityCertificate:
    """A fully-green fidelity certificate."""
    return FidelityCertificate(
        invariant_tests_passed=True,
        mutation_confirmed=True,
        blinded_extraction_agreed=True,
        obligation_profile="end_to_end",
        profile_satisfied=True,
        has_measured_metrics=True,
        invariant_tests_ran=True,
    )


def _write_metrics(tmp_path: Path, metrics: dict) -> Path:
    code_dir = tmp_path / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return code_dir


# ---------------------------------------------------------------------------
# 1. Default OFF — gate disabled → always (False, None)
# ---------------------------------------------------------------------------

class TestGateOff:
    def test_gate_disabled_returns_false_none_flat(self, tmp_path: Path) -> None:
        """With gate OFF, detect_no_learning_signal returns (False, None) even for
        a flat grid with all-zero reward_history leaves."""
        # Build a flat per_model with zero reward curves.
        flat_leaf = {
            "status": "ok",
            "reward_history": [0.0] * 8,
        }
        code_dir = _write_metrics(tmp_path, {"per_model": {"qwen-1.7b": flat_leaf}})
        # Gate is OFF by default (no env var set).
        veto, detail = detect_no_learning_signal(code_dir)
        assert veto is False
        assert detail is None

    def test_gate_disabled_no_env_var(self) -> None:
        assert no_learning_signal_enabled() is False


# ---------------------------------------------------------------------------
# 2. Gate ON, flat grid, all-zero reward → (True, detail)
# ---------------------------------------------------------------------------

class TestFlatGridNoLearning:
    def test_flat_no_learning_returns_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", "1")
        flat_leaf = {
            "status": "ok",
            "reward_history": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        code_dir = _write_metrics(tmp_path, {"per_model": {"qwen-1.7b": flat_leaf}})
        veto, detail = detect_no_learning_signal(code_dir)
        assert veto is True
        assert detail is not None
        assert len(detail) > 0

    def test_detail_contains_reward_info(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", "1")
        leaf = {
            "status": "success",
            "reward_history": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        }
        code_dir = _write_metrics(tmp_path, {"per_model": {"m": leaf}})
        veto, detail = detect_no_learning_signal(code_dir)
        assert veto is True
        assert "reward" in (detail or "").lower()

    def test_gate_true_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for v in ("1", "true", "yes", "on"):
            monkeypatch.setenv("OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", v)
            assert no_learning_signal_enabled() is True


# ---------------------------------------------------------------------------
# 3. Rising leaf present → (False, None) — learned somewhere
# ---------------------------------------------------------------------------

class TestRisingLeafLearned:
    def test_one_rising_leaf_returns_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", "1")
        rising_leaf = {
            "status": "ok",
            "reward_history": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
        flat_leaf = {
            "status": "ok",
            "reward_history": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        code_dir = _write_metrics(
            tmp_path,
            {"per_model": {"model_a": rising_leaf, "model_b": flat_leaf}},
        )
        veto, detail = detect_no_learning_signal(code_dir)
        assert veto is False
        assert detail is None

    def test_solo_rising_leaf(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", "1")
        leaf = {
            "status": "ok",
            "reward_history": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
        code_dir = _write_metrics(tmp_path, {"per_model": {"m": leaf}})
        veto, detail = detect_no_learning_signal(code_dir)
        assert veto is False
        assert detail is None


# ---------------------------------------------------------------------------
# 4. No judgeable curves (<5 points or absent) → (False, None)
# ---------------------------------------------------------------------------

class TestNoJudgeableCurves:
    def test_absent_curves_not_judgeable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", "1")
        # Leaf without any curve key.
        leaf = {"status": "ok", "accuracy": 0.85}
        code_dir = _write_metrics(tmp_path, {"per_model": {"m": leaf}})
        veto, detail = detect_no_learning_signal(code_dir)
        assert veto is False
        assert detail is None

    def test_too_few_points_not_judgeable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", "1")
        leaf = {
            "status": "ok",
            "reward_history": [0.0, 0.0, 0.0, 0.0],  # only 4 points, below _MIN_POINTS=5
        }
        code_dir = _write_metrics(tmp_path, {"per_model": {"m": leaf}})
        veto, detail = detect_no_learning_signal(code_dir)
        assert veto is False
        assert detail is None

    def test_exactly_min_points_judgeable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", "1")
        leaf = {
            "status": "ok",
            "reward_history": [0.0, 0.0, 0.0, 0.0, 0.0],  # exactly 5
        }
        code_dir = _write_metrics(tmp_path, {"per_model": {"m": leaf}})
        veto, detail = detect_no_learning_signal(code_dir)
        assert veto is True  # 5 points of flat zero = judgeable + no learning

    def test_no_success_leaves_not_judgeable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", "1")
        leaf = {
            "status": "failed",
            "reward_history": [0.0] * 6,
        }
        code_dir = _write_metrics(tmp_path, {"per_model": {"m": leaf}})
        veto, detail = detect_no_learning_signal(code_dir)
        assert veto is False
        assert detail is None


# ---------------------------------------------------------------------------
# 5. Nested cells-route shape — per_model[m][env][baseline]=leaf
# ---------------------------------------------------------------------------

class TestNestedCellsRouteShape:
    def test_nested_no_learning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", "1")
        leaf = {
            "status": "ok",
            "reward_history": [0.0] * 6,
        }
        metrics = {
            "per_model": {
                "qwen-1.7b": {
                    "alfworld": {
                        "grpo": leaf,
                    }
                }
            }
        }
        code_dir = _write_metrics(tmp_path, metrics)
        veto, detail = detect_no_learning_signal(code_dir)
        assert veto is True
        assert detail is not None

    def test_nested_one_rising_returns_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", "1")
        flat_leaf = {"status": "ok", "reward_history": [0.0] * 6}
        rising_leaf = {"status": "ok", "reward_history": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
        metrics = {
            "per_model": {
                "qwen-1.7b": {
                    "alfworld": {"grpo": flat_leaf},
                    "webshop": {"grpo": rising_leaf},
                }
            }
        }
        code_dir = _write_metrics(tmp_path, metrics)
        veto, detail = detect_no_learning_signal(code_dir)
        assert veto is False
        assert detail is None

    def test_nested_training_curves_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", "1")
        leaf = {
            "status": "ok",
            "training_curves": {
                "reward": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "loss": [2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
            },
        }
        metrics = {
            "per_model": {
                "qwen-3b": {
                    "alfworld": {"grpo": leaf},
                }
            }
        }
        code_dir = _write_metrics(tmp_path, metrics)
        veto, detail = detect_no_learning_signal(code_dir)
        assert veto is True

    def test_nested_descending_loss_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Loss descends but no reward curve → leaf shows learning."""
        monkeypatch.setenv("OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", "1")
        leaf = {
            "status": "ok",
            "training_curves": {
                "loss": [2.5, 2.0, 1.5, 1.0, 0.7, 0.5],
            },
        }
        code_dir = _write_metrics(tmp_path, {"per_model": {"m": leaf}})
        veto, detail = detect_no_learning_signal(code_dir)
        assert veto is False


# ---------------------------------------------------------------------------
# 6. compute_reproducibility_verdict with no_learning_signal param
# ---------------------------------------------------------------------------

class TestReproducibilityVerdictNoLearningParam:
    def test_no_learning_true_forces_inconclusive(self) -> None:
        cert = _green_cert()
        verdict = compute_reproducibility_verdict(
            fidelity_score=0.85,
            certificate=cert,
            claims=[],
            no_learning_signal=True,
        )
        assert verdict.replication_verdict == "inconclusive"
        assert verdict.implementation_verdict == "faithful"
        # Rationale should mention the no-learning-signal reason.
        joined = " ".join(verdict.rationale)
        assert "no learning signal" in joined

    def test_no_learning_false_unchanged(self) -> None:
        """no_learning_signal=False must be byte-identical to omitting the param."""
        cert = _green_cert()
        v_explicit = compute_reproducibility_verdict(
            fidelity_score=0.85,
            certificate=cert,
            claims=[],
            no_learning_signal=False,
        )
        v_default = compute_reproducibility_verdict(
            fidelity_score=0.85,
            certificate=cert,
            claims=[],
        )
        assert v_explicit.replication_verdict == v_default.replication_verdict
        assert v_explicit.implementation_verdict == v_default.implementation_verdict
        assert v_explicit.replication_credit == v_default.replication_credit

    def test_no_learning_respects_impl_gate(self) -> None:
        """When implementation is not faithful, the impl gate fires FIRST (decision 1).
        no_learning_signal should not change the rationale for that gate."""
        cert = FidelityCertificate(
            invariant_tests_passed=False,
            mutation_confirmed=False,
            blinded_extraction_agreed=False,
            obligation_profile="end_to_end",
            profile_satisfied=False,
            has_measured_metrics=True,
            invariant_tests_ran=True,
        )
        verdict = compute_reproducibility_verdict(
            fidelity_score=0.3,  # below DEFAULT_FAITHFUL_MIN_SCORE
            certificate=cert,
            claims=[],
            no_learning_signal=True,
        )
        # Still inconclusive, but the rationale should cite the non-faithful build,
        # not the no-learning-signal (the impl gate fires first).
        assert verdict.replication_verdict == "inconclusive"
        assert verdict.implementation_verdict in ("broken", "partial")
        joined = " ".join(verdict.rationale)
        assert "faithful" in joined

    def test_no_learning_true_legacy_verdict_from_fidelity(self) -> None:
        """A4 — legacy_verdict is projected from FIDELITY, not replication."""
        cert = _green_cert()
        verdict = compute_reproducibility_verdict(
            fidelity_score=0.9,
            certificate=cert,
            claims=[],
            no_learning_signal=True,
        )
        assert verdict.legacy_verdict == "reproduced"  # faithful → reproduced
        assert verdict.replication_verdict == "inconclusive"

    def test_repair_message_non_empty(self) -> None:
        msg = no_learning_repair_message("qwen-1.7b(first_reward=0.0, best_reward=0.01)")
        assert "no_learning_signal" in msg
        assert "inconclusive" in msg
        assert "under-powered" in msg or "mis-wired" in msg


# ---------------------------------------------------------------------------
# Unit tests for _leaf_no_learning internals
# ---------------------------------------------------------------------------

class TestLeafNoLearning:
    def test_flat_zeros_no_learning(self) -> None:
        leaf = {"status": "ok", "reward_history": [0.0] * 6}
        assert _leaf_no_learning(leaf) is True

    def test_rising_reward_learns(self) -> None:
        leaf = {"status": "ok", "reward_history": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
        assert _leaf_no_learning(leaf) is False

    def test_too_few_points_returns_none(self) -> None:
        leaf = {"status": "ok", "reward_history": [0.0, 0.0, 0.0]}
        assert _leaf_no_learning(leaf) is None

    def test_no_curves_returns_none(self) -> None:
        leaf = {"status": "ok", "accuracy": 0.9}
        assert _leaf_no_learning(leaf) is None

    def test_descending_loss_no_reward_returns_false(self) -> None:
        """Descending loss (without a reward curve) = learned."""
        leaf = {
            "status": "ok",
            "training_curves": {"loss": [2.5, 2.0, 1.5, 1.0, 0.7, 0.5]},
        }
        assert _leaf_no_learning(leaf) is False

    def test_flat_loss_only_no_learning(self) -> None:
        """Flat loss curve with no reward = no learning detected."""
        leaf = {
            "status": "ok",
            "training_curves": {"loss": [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]},
        }
        assert _leaf_no_learning(leaf) is True

    def test_reward_nonzero_first_below_threshold(self) -> None:
        """Reward starts at 0.2, best is 0.21 — barely above first but below eps threshold."""
        first = 0.2
        # threshold = first*(1 + 0.05) = 0.21 exactly; best=0.21 → best <= threshold → no rise
        leaf = {
            "status": "ok",
            "reward_history": [0.2, 0.20, 0.21, 0.21, 0.20, 0.21],
        }
        assert _leaf_no_learning(leaf) is True

    def test_reward_nonzero_first_above_threshold(self) -> None:
        """Reward starts at 0.2, best is 0.22 — above eps threshold → learning."""
        # threshold = 0.2 * 1.05 = 0.21 < 0.22 → rises
        leaf = {
            "status": "ok",
            "reward_history": [0.2, 0.20, 0.21, 0.22, 0.22, 0.22],
        }
        assert _leaf_no_learning(leaf) is False

    def test_training_curves_rewards_key(self) -> None:
        """training_curves.rewards (plural) is accepted."""
        leaf = {
            "status": "ok",
            "training_curves": {"rewards": [0.0] * 6},
        }
        assert _leaf_no_learning(leaf) is True

    def test_training_curves_mean_reward_key(self) -> None:
        """training_curves.mean_reward is accepted."""
        leaf = {
            "status": "ok",
            "training_curves": {"mean_reward": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        }
        assert _leaf_no_learning(leaf) is False
