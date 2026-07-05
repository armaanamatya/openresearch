"""
Tests for backend/agents/rlm/minimal_viable.py (Minimal Viable Reproduction, MVR).

Hermetic: tmp_path + monkeypatch only; no network, no GPU, no LLM calls.

MVR is a standalone opt-in flag (OPENRESEARCH_MINIMAL_VIABLE) that composes with
adapt/reference/execute modes. Its viability verdict is computed ONLY from the
deterministic evidence layer (training curves, metrics.json, honesty-guard
flags) -- never the LLM rubric grade -- and the whole module is fail-soft:
any internal error degrades to "inconclusive", never raises.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.schemas import DatasetSlice, PaperHint, ScopeSpec
from backend.agents.rlm.minimal_viable import (
    compute_viability_verdict,
    minimal_viable_enabled,
    select_viability_scope,
    viability_guidance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_code_metrics(tmp_path: Path, metrics: dict) -> Path:
    code_dir = tmp_path / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return code_dir


_RISING_LEAF = {
    "status": "ok",
    "success_rate": 0.35,
    "reward_history": [0.0, 0.05, 0.1, 0.2, 0.3, 0.35],
}

_FLAT_LEAF = {
    "status": "ok",
    "success_rate": 0.1,
    "reward_history": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
}

_CLEAN_REPORT_DICT: dict = {"verdict": "partial", "rubric": {"overall_score": 0.4}}


# ---------------------------------------------------------------------------
# 1. minimal_viable_enabled — truthy/falsy/unset parse
# ---------------------------------------------------------------------------

class TestMinimalViableEnabled:
    def test_unset_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENRESEARCH_MINIMAL_VIABLE", raising=False)
        assert minimal_viable_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "True", "YES", "on", "ON"])
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("OPENRESEARCH_MINIMAL_VIABLE", value)
        assert minimal_viable_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "False", "no", "off", "", "garbage"])
    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("OPENRESEARCH_MINIMAL_VIABLE", value)
        assert minimal_viable_enabled() is False


# ---------------------------------------------------------------------------
# 2. select_viability_scope
# ---------------------------------------------------------------------------

class TestSelectViabilityScope:
    def test_real_sdar_hint_picks_smallest_model_first_dataset_first_seed(self) -> None:
        result = select_viability_scope("2605.15155", None)
        assert result is not None
        assert result.models == ["Qwen3-1.7B"]
        assert len(result.datasets) == 1
        assert result.datasets[0].name == "ALFWorld"
        assert result.seeds == [0]

    def test_version_suffixed_arxiv_id_normalizes(self) -> None:
        result = select_viability_scope("2605.15155v2", None)
        assert result is not None
        assert result.models == ["Qwen3-1.7B"]

    def test_smallest_bucket_wins_regardless_of_list_position(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """7B listed FIRST but 1.7B is the smaller size-bucket -- must still win."""
        fake_hint = PaperHint(
            default_scope=ScopeSpec(
                models=["Qwen2.5-7B-Instruct", "Qwen3-1.7B"],
                datasets=[DatasetSlice(name="ALFWorld"), DatasetSlice(name="WebShop")],
                seeds=[0],
            )
        )
        monkeypatch.setattr(
            "backend.agents.prompts.paper_hints.lookup_paper_hint",
            lambda paper_id: fake_hint,
        )
        result = select_viability_scope("9999.00000", None)
        assert result is not None
        assert result.models == ["Qwen3-1.7B"]
        assert result.datasets[0].name == "ALFWorld"

    def test_ties_break_by_first_listed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Qwen2.5-3B-Instruct (3B) and Qwen3-1.7B are BOTH the 'small' bucket
        (< 4B); the first-listed model must win the tie."""
        fake_hint = PaperHint(
            default_scope=ScopeSpec(
                models=["Qwen2.5-3B-Instruct", "Qwen3-1.7B"],
                datasets=[DatasetSlice(name="ALFWorld")],
                seeds=[7],
            )
        )
        monkeypatch.setattr(
            "backend.agents.prompts.paper_hints.lookup_paper_hint",
            lambda paper_id: fake_hint,
        )
        result = select_viability_scope("9999.00001", None)
        assert result is not None
        assert result.models == ["Qwen2.5-3B-Instruct"]
        assert result.seeds == [7]

    def test_no_hint_returns_none(self) -> None:
        assert select_viability_scope("9999.99999", None) is None

    def test_none_arxiv_id_returns_none(self) -> None:
        assert select_viability_scope(None, None) is None

    def test_hint_without_models_returns_none(self) -> None:
        """ResNet's hint (1512.03385) declares datasets/seeds but no models --
        there is nothing to rank 'smallest' over, so MVR must not invent one."""
        assert select_viability_scope("1512.03385", None) is None

    def test_sdar_full_default_scope_narrows_flagship(self) -> None:
        """FLAGSHIP: the exact scope cmd_reproduce merges for `reproduce sdar
        --minimal-viable` (SDAR's 3 models x 3 datasets x [0], arriving as a
        populated operator_scope) must NARROW to the smallest central-claim
        cell -- NOT no-op. This is the whole point of the FIX-1 rewrite: MVR
        reduces the EFFECTIVE scope, and a pre-populated operator scope (from
        the hint merge) is no longer treated as 'operator already narrowed'."""
        merged = ScopeSpec(
            models=["Qwen3-1.7B", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"],
            datasets=[
                DatasetSlice(name="ALFWorld"),
                DatasetSlice(name="WebShop"),
                DatasetSlice(name="Search-QA"),
            ],
            seeds=[0],
        )
        result = select_viability_scope("2605.15155", merged)
        assert result is not None
        assert result.models == ["Qwen3-1.7B"]
        assert [d.name for d in result.datasets] == ["ALFWorld"]
        assert result.seeds == [0]

    def test_operator_single_model_multi_dataset_hint_narrows(self) -> None:
        """Operator pins 1 model but leaves datasets to the hint (SDAR's 3):
        the effective scope is still multi-cell, so MVR narrows to that model x
        the first (hint) dataset x the first seed -- NOT a no-op."""
        op_scope = ScopeSpec(models=["Qwen3-1.7B"])
        result = select_viability_scope("2605.15155", op_scope)
        assert result is not None
        assert result.models == ["Qwen3-1.7B"]
        assert [d.name for d in result.datasets] == ["ALFWorld"]  # first SDAR dataset
        assert result.seeds == [0]

    def test_operator_single_dataset_multi_model_hint_narrows(self) -> None:
        """Operator pins 1 dataset but leaves models to the hint (SDAR's 3):
        MVR narrows to the smallest (hint) model x that dataset x the first
        seed."""
        op_scope = ScopeSpec(datasets=[DatasetSlice(name="WebShop")])
        result = select_viability_scope("2605.15155", op_scope)
        assert result is not None
        assert result.models == ["Qwen3-1.7B"]  # smallest of the 3 SDAR models
        assert [d.name for d in result.datasets] == ["WebShop"]  # operator's explicit dataset wins
        assert result.seeds == [0]

    def test_already_minimal_scope_returns_none(self) -> None:
        """An effective scope that is ALREADY 1x1x1 -- e.g. the operator typed
        the exact single cell -- is a no-op: MVR preserves it verbatim (incl.
        any per-dataset episode counts) rather than rebuilding it."""
        op_scope = ScopeSpec(
            models=["Qwen2.5-3B-Instruct"],
            datasets=[DatasetSlice(name="WebShop", episodes=16)],
            seeds=[42],
        )
        assert select_viability_scope("2605.15155", op_scope) is None

    def test_empty_operator_scope_object_does_not_block_narrowing(self) -> None:
        """An all-default ScopeSpec() carries no operator narrowing -- MVR falls
        back to the hint default per axis and selects the smallest cell."""
        result = select_viability_scope("2605.15155", ScopeSpec())
        assert result is not None
        assert result.models == ["Qwen3-1.7B"]

    def test_operator_multi_model_no_hint_narrows(self) -> None:
        """No hint at all, but the operator opted into --minimal-viable AND
        supplied a multi-model scope: MVR still reduces the effective scope
        (models_src is non-empty), datasets stay empty (no hint to fall back
        to)."""
        op_scope = ScopeSpec(models=["Qwen2.5-7B-Instruct", "Qwen3-1.7B"])
        result = select_viability_scope("9999.99999", op_scope)  # no hint
        assert result is not None
        assert result.models == ["Qwen3-1.7B"]  # smallest
        assert result.datasets == []
        assert result.seeds == [0]

    def test_fail_soft_bad_operator_scope_object(self) -> None:
        class Garbage:
            @property
            def models(self) -> list[str]:
                raise RuntimeError("boom")

        assert select_viability_scope("2605.15155", Garbage()) is None

    def test_fail_soft_wrong_type_arxiv_id(self) -> None:
        assert select_viability_scope(12345, None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. viability_guidance — shape only (content is prose)
# ---------------------------------------------------------------------------

class TestViabilityGuidance:
    def test_returns_nonempty_brace_free_string(self) -> None:
        text = viability_guidance()
        assert isinstance(text, str)
        assert len(text) > 0
        assert "{" not in text and "}" not in text

    def test_mentions_single_cell_and_short_budget(self) -> None:
        text = viability_guidance().lower()
        assert "one model" in text or "single" in text
        assert "short" in text or "budget" in text


# ---------------------------------------------------------------------------
# 4. compute_viability_verdict
# ---------------------------------------------------------------------------

class TestComputeViabilityVerdict:
    def test_viable(self, tmp_path: Path) -> None:
        code_dir = _write_code_metrics(
            tmp_path, {"per_model": {"qwen3-1.7b": {"alfworld": {"sdar": _RISING_LEAF}}}}
        )
        result = compute_viability_verdict(
            code_dir,
            arxiv_id="2605.15155",
            scope=ScopeSpec(models=["Qwen3-1.7B"], datasets=[DatasetSlice(name="ALFWorld")], seeds=[0]),
            report_dict=dict(_CLEAN_REPORT_DICT),
        )
        assert result["verdict"] == "viable"
        assert result["learning_signal"] is True
        assert result["directional"] is True
        assert result["guards_clean"] is True
        assert result["central_claim"] == {"model": "Qwen3-1.7B", "env": "ALFWorld", "seed": 0}
        assert result["measured_metric"] is not None
        assert isinstance(result["rationale"], str) and result["rationale"]
        assert isinstance(result["evidence_refs"], list) and result["evidence_refs"]
        # No structured headline-value store exists in PaperHint today.
        assert result["headline_reference"] is None
        assert result["gap"] is None

    def test_not_viable_flat_curve_gate_off_learning_signal_faithful(self, tmp_path: Path) -> None:
        """FIX-2: with OPENRESEARCH_NO_LEARNING_SIGNAL_GATE OFF (default), a
        genuinely flat judged curve is `not_viable` AND `learning_signal` is
        faithfully False -- the sibling gate no longer masks it into a spurious
        True. The flat curve also makes `directional` False."""
        code_dir = _write_code_metrics(tmp_path, {"per_model": {"m": _FLAT_LEAF}})
        result = compute_viability_verdict(
            code_dir, arxiv_id=None, scope=None, report_dict=dict(_CLEAN_REPORT_DICT),
        )
        assert result["verdict"] == "not_viable"
        assert result["learning_signal"] is False  # flag-independent, faithful with gate OFF
        assert result["directional"] is False
        assert result["guards_clean"] is True

    def test_not_viable_flat_curve_with_no_learning_gate_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The verdict is identical with OPENRESEARCH_NO_LEARNING_SIGNAL_GATE=1 --
        MVR's learning signal is flag-independent, so the sibling gate's state
        does not change the outcome."""
        monkeypatch.setenv("OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", "1")
        code_dir = _write_code_metrics(tmp_path, {"per_model": {"m": _FLAT_LEAF}})
        result = compute_viability_verdict(
            code_dir, arxiv_id=None, scope=None, report_dict=dict(_CLEAN_REPORT_DICT),
        )
        assert result["verdict"] == "not_viable"
        assert result["learning_signal"] is False
        assert result["guards_clean"] is True

    def test_inconclusive_scalar_only_no_curve(self, tmp_path: Path) -> None:
        """FIX-2 tri-state: a scalar-only leaf (real, non-zero, clean, but NO
        training curve) is `inconclusive` -- there is no learning curve to judge
        viability. (Under the old flag-gated logic this was spuriously graded.)"""
        code_dir = _write_code_metrics(
            tmp_path, {"per_model": {"m": {"status": "ok", "accuracy": 0.85}}}
        )
        result = compute_viability_verdict(
            code_dir, arxiv_id=None, scope=None, report_dict=dict(_CLEAN_REPORT_DICT),
        )
        assert result["verdict"] == "inconclusive"
        assert result["learning_signal"] is False  # None tri-state -> bool() False
        assert result["guards_clean"] is True
        assert result["measured_metric"] == {"name": "accuracy", "value": 0.85}

    @pytest.mark.parametrize(
        "veto_report_dict",
        [
            {"replication_verdict": "inconclusive"},
            {"evidence_gate_passed": False},
            {"validation": {"status": "vetoed"}},
            {"per_model": {"m": {"e": {"b": {"failure_class": "fabrication_suspected"}}}}},
        ],
        ids=["replication_verdict", "evidence_gate_passed", "validation_vetoed", "fabrication_marker"],
    )
    def test_inconclusive_on_guard_veto(self, tmp_path: Path, veto_report_dict: dict) -> None:
        """Even with clean, rising, real evidence, any honesty-guard veto in
        report_dict forces inconclusive (never a false viable/not_viable)."""
        code_dir = _write_code_metrics(
            tmp_path, {"per_model": {"qwen3-1.7b": {"alfworld": {"sdar": _RISING_LEAF}}}}
        )
        result = compute_viability_verdict(
            code_dir, arxiv_id=None, scope=None, report_dict=veto_report_dict,
        )
        assert result["verdict"] == "inconclusive"
        assert result["guards_clean"] is False

    def test_inconclusive_no_evidence_missing_metrics_file(self, tmp_path: Path) -> None:
        code_dir = tmp_path / "code"
        code_dir.mkdir(parents=True, exist_ok=True)  # no metrics.json written
        result = compute_viability_verdict(
            code_dir, arxiv_id=None, scope=None, report_dict=dict(_CLEAN_REPORT_DICT),
        )
        assert result["verdict"] == "inconclusive"

    def test_inconclusive_no_evidence_all_zero_metrics(self, tmp_path: Path) -> None:
        code_dir = _write_code_metrics(
            tmp_path,
            {"per_model": {"m": {"status": "ok", "success_rate": 0.0, "reward": 0.0}}},
        )
        result = compute_viability_verdict(
            code_dir, arxiv_id=None, scope=None, report_dict=dict(_CLEAN_REPORT_DICT),
        )
        assert result["verdict"] == "inconclusive"

    def test_fail_soft_garbage_input_never_raises(self, tmp_path: Path) -> None:
        result = compute_viability_verdict(
            tmp_path / "does" / "not" / "exist",
            arxiv_id=12345,  # type: ignore[arg-type]
            scope=object(),
            report_dict=None,  # type: ignore[arg-type]
        )
        assert result["verdict"] == "inconclusive"
        assert "rationale" in result

    def test_fail_soft_unreadable_metrics_json(self, tmp_path: Path) -> None:
        code_dir = tmp_path / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "metrics.json").write_text("{not valid json", encoding="utf-8")
        result = compute_viability_verdict(
            code_dir, arxiv_id=None, scope=None, report_dict=dict(_CLEAN_REPORT_DICT),
        )
        assert result["verdict"] == "inconclusive"

    def test_no_scope_central_claim_is_none(self, tmp_path: Path) -> None:
        code_dir = _write_code_metrics(tmp_path, {"per_model": {"m": _RISING_LEAF}})
        result = compute_viability_verdict(
            code_dir, arxiv_id=None, scope=None, report_dict=dict(_CLEAN_REPORT_DICT),
        )
        assert result["central_claim"] is None


# ---------------------------------------------------------------------------
# 5. config.py wiring — Settings.minimal_viable
# ---------------------------------------------------------------------------

class TestSettingsWiring:
    def test_settings_default_false(self) -> None:
        from backend.config import Settings
        assert Settings().minimal_viable is False

    def test_settings_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.config import Settings
        monkeypatch.setenv("OPENRESEARCH_MINIMAL_VIABLE", "true")
        assert Settings().minimal_viable is True


# ---------------------------------------------------------------------------
# 6. OFF-parity — run.py wiring is a byte-identical no-op when the flag is unset
# ---------------------------------------------------------------------------

class TestRunWiringOffParity:
    def test_scope_hook_noop_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even for a hinted paper with an unscoped operator, the flag being OFF
        must leave the scope + guidance completely untouched."""
        monkeypatch.delenv("OPENRESEARCH_MINIMAL_VIABLE", raising=False)
        from backend.agents.rlm.run import _minimal_viable_scope_hook
        original_scope = ScopeSpec()
        scope, guidance = _minimal_viable_scope_hook("2605.15155", original_scope)
        assert scope is original_scope
        assert guidance is None

    def test_scope_hook_narrows_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_MINIMAL_VIABLE", "1")
        from backend.agents.rlm.run import _minimal_viable_scope_hook
        scope, guidance = _minimal_viable_scope_hook("2605.15155", ScopeSpec())
        assert scope.models == ["Qwen3-1.7B"]
        assert guidance == viability_guidance()

    def test_scope_hook_enabled_but_no_hint_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_MINIMAL_VIABLE", "1")
        from backend.agents.rlm.run import _minimal_viable_scope_hook
        original_scope = ScopeSpec()
        scope, guidance = _minimal_viable_scope_hook("9999.99999", original_scope)
        assert scope is original_scope
        assert guidance is None

    def test_finalize_hook_noop_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("OPENRESEARCH_MINIMAL_VIABLE", raising=False)
        from backend.agents.rlm.run import _apply_minimal_viable_reproduction
        report_path = tmp_path / "final_report.json"
        original = json.dumps({"verdict": "partial"}, indent=2)
        report_path.write_text(original, encoding="utf-8")
        ctx = SimpleNamespace(arxiv_id="2605.15155", scope_spec=None)
        _apply_minimal_viable_reproduction(tmp_path, ctx)
        assert report_path.read_text(encoding="utf-8") == original

    def test_finalize_hook_noop_when_disabled_no_report_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("OPENRESEARCH_MINIMAL_VIABLE", raising=False)
        from backend.agents.rlm.run import _apply_minimal_viable_reproduction
        ctx = SimpleNamespace(arxiv_id=None, scope_spec=None)
        _apply_minimal_viable_reproduction(tmp_path, ctx)  # must not raise
        assert not (tmp_path / "final_report.json").exists()

    def test_finalize_hook_stamps_verdict_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_MINIMAL_VIABLE", "1")
        from backend.agents.rlm.run import _apply_minimal_viable_reproduction
        (tmp_path / "final_report.json").write_text(
            json.dumps({"verdict": "partial"}, indent=2), encoding="utf-8"
        )
        _write_code_metrics(
            tmp_path, {"per_model": {"qwen3-1.7b": {"alfworld": {"sdar": _RISING_LEAF}}}}
        )
        ctx = SimpleNamespace(
            arxiv_id="2605.15155",
            scope_spec=ScopeSpec(models=["Qwen3-1.7B"], datasets=[DatasetSlice(name="ALFWorld")], seeds=[0]),
        )
        _apply_minimal_viable_reproduction(tmp_path, ctx)
        written = json.loads((tmp_path / "final_report.json").read_text(encoding="utf-8"))
        assert written["verdict"] == "partial"  # untouched — MVR is additive only
        assert "minimal_viable_reproduction" in written
        assert written["minimal_viable_reproduction"]["verdict"] == "viable"

    def test_finalize_hook_fail_soft_corrupt_report_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_MINIMAL_VIABLE", "1")
        from backend.agents.rlm.run import _apply_minimal_viable_reproduction
        (tmp_path / "final_report.json").write_text("{not valid json", encoding="utf-8")
        ctx = SimpleNamespace(arxiv_id=None, scope_spec=None)
        _apply_minimal_viable_reproduction(tmp_path, ctx)  # must not raise

    def test_finalize_hook_handles_none_ctx(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The hard-stop path (_hard_stop_with_report) may call with ctx=None."""
        monkeypatch.setenv("OPENRESEARCH_MINIMAL_VIABLE", "1")
        from backend.agents.rlm.run import _apply_minimal_viable_reproduction
        (tmp_path / "final_report.json").write_text(
            json.dumps({"verdict": "failed"}, indent=2), encoding="utf-8"
        )
        _apply_minimal_viable_reproduction(tmp_path, None)  # must not raise
        written = json.loads((tmp_path / "final_report.json").read_text(encoding="utf-8"))
        assert "minimal_viable_reproduction" in written
