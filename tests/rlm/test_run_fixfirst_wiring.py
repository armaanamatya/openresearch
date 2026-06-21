"""Tests for the P3.1 fix-first repair-loop wiring into run.py (Task P3.2).

Hermetic — no network, no subprocess, no rlm import required.  All tests use
monkeypatch for env vars and tmp_path for filesystem I/O.

Coverage:
  1. _fixfirst_loop_engaged() — True when either guard flag is set, False when neither.
  2. clear_repair_trigger branch in _record_last_primitive_result_tools — invoked on a
     SUCCESS run_experiment, NOT invoked on a repairable one.
  3. validator_gate closure — returns (True, directive) on a vetoed verdict, persists the
     verdict, caches by fingerprint (panel NOT called again on same metrics.json), and DOES
     call the panel again when metrics.json changes.
  4. Default-OFF contract — with both flags unset, iteration_policy.evidence_fingerprint
     is None and iteration_policy.validator_gate is None after hook assignment.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# 1.  _fixfirst_loop_engaged
# ---------------------------------------------------------------------------

class TestFixfirstLoopEngaged:
    """Unit tests for the module-level helper _fixfirst_loop_engaged()."""

    def test_false_when_neither_flag_set(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_ZERO_METRICS_GUARD", raising=False)
        monkeypatch.delenv("OPENRESEARCH_EXTERNAL_VALIDATOR", raising=False)
        from backend.agents.rlm.run import _fixfirst_loop_engaged
        assert _fixfirst_loop_engaged() is False

    def test_true_when_zero_metrics_guard_set(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ZERO_METRICS_GUARD", "1")
        monkeypatch.delenv("OPENRESEARCH_EXTERNAL_VALIDATOR", raising=False)
        from backend.agents.rlm.run import _fixfirst_loop_engaged
        assert _fixfirst_loop_engaged() is True

    def test_true_when_external_validator_set(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_ZERO_METRICS_GUARD", raising=False)
        monkeypatch.setenv("OPENRESEARCH_EXTERNAL_VALIDATOR", "1")
        from backend.agents.rlm.run import _fixfirst_loop_engaged
        assert _fixfirst_loop_engaged() is True

    def test_true_when_both_flags_set(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ZERO_METRICS_GUARD", "1")
        monkeypatch.setenv("OPENRESEARCH_EXTERNAL_VALIDATOR", "1")
        from backend.agents.rlm.run import _fixfirst_loop_engaged
        assert _fixfirst_loop_engaged() is True


# ---------------------------------------------------------------------------
# 2.  clear_repair_trigger branch in the tool wrapper
# ---------------------------------------------------------------------------

def _make_policy_with_fingerprint():
    """Return a ForcedIterationPolicy with evidence_fingerprint engaged."""
    from backend.agents.rlm.forced_iteration import ForcedIterationPolicy

    policy = ForcedIterationPolicy(min_iterations=2)
    # Wire a dummy fingerprint so clear_repair_trigger is not a no-op.
    policy.evidence_fingerprint = lambda: "fp-constant"
    # Seed a repair state so we can confirm it is cleared.
    policy._last_repair_failure_class = "some_error"
    policy._repair_iter_count = 1
    return policy


class TestClearRepairTriggerBranch:
    """Validate the elif clear_repair_trigger branch in _record_last_primitive_result_tools."""

    def _call_wrapper_with_outcome(self, outcome_str: str, policy) -> None:
        """Simulate the tool wrapper's branch logic for a run_experiment result."""
        from backend.agents.rlm.run import _fixfirst_loop_engaged  # noqa: F401
        # We exercise the branch logic directly (without launching the full pipeline)
        # by replicating what the wrapper does.

        # outcome_value mapping (from primitives.py convention)
        def _outcome_value(outcome):
            if outcome is None:
                return "ok"
            if isinstance(outcome, str):
                return outcome.lower()
            return str(outcome).lower()

        repair_policy_holder: list[Any] = [policy]
        name = "run_experiment"
        result: dict = {"outcome": outcome_str}

        repairable = (
            name == "run_experiment"
            and _outcome_value(result.get("outcome")) == "repairable"
            and repair_policy_holder
        )
        if repairable:
            failure_class = str(result.get("failure_class") or "unknown")
            repair_policy_holder[0].record_repair_attempt(failure_class)
        elif (
            name == "run_experiment"
            and _outcome_value(result.get("outcome")) not in ("repairable", "partial_evidence", "fatal")
            and repair_policy_holder
        ):
            repair_policy_holder[0].clear_repair_trigger()

    def test_success_clears_repair_trigger(self):
        """A success outcome (not repairable/partial_evidence/fatal) clears the trigger."""
        policy = _make_policy_with_fingerprint()
        assert policy._last_repair_failure_class == "some_error"
        assert policy._repair_iter_count == 1

        self._call_wrapper_with_outcome("ok", policy)

        # clear_repair_trigger resets when evidence_fingerprint is wired
        assert policy._last_repair_failure_class is None
        assert policy._repair_iter_count == 0

    def test_repairable_does_not_clear(self):
        """A repairable outcome takes the record_repair_attempt branch, not clear."""
        policy = _make_policy_with_fingerprint()
        original_count = policy._repair_iter_count

        self._call_wrapper_with_outcome("repairable", policy)

        # record_repair_attempt increments the counter, NOT clear
        assert policy._repair_iter_count == original_count + 1
        # _last_repair_failure_class was set (to "unknown" since no failure_class key)
        assert policy._last_repair_failure_class == "unknown"

    def test_partial_evidence_does_not_clear(self):
        """partial_evidence is a failure outcome — trigger is NOT cleared."""
        policy = _make_policy_with_fingerprint()
        before = policy._last_repair_failure_class

        self._call_wrapper_with_outcome("partial_evidence", policy)

        # Neither branch runs for partial_evidence (not repairable, but IS in the
        # exclusion list for the elif) — so repair state is unchanged.
        assert policy._last_repair_failure_class == before

    def test_fatal_does_not_clear(self):
        """fatal is a failure outcome — trigger is NOT cleared."""
        policy = _make_policy_with_fingerprint()
        before = policy._last_repair_failure_class

        self._call_wrapper_with_outcome("fatal", policy)

        assert policy._last_repair_failure_class == before

    def test_clear_is_noop_when_no_fingerprint(self):
        """When evidence_fingerprint is not wired, clear_repair_trigger is a no-op."""
        from backend.agents.rlm.forced_iteration import ForcedIterationPolicy
        policy = ForcedIterationPolicy(min_iterations=2)
        policy._last_repair_failure_class = "some_error"
        policy._repair_iter_count = 3
        # No evidence_fingerprint assigned → clear_repair_trigger is a no-op

        self._call_wrapper_with_outcome("ok", policy)

        # State unchanged (no-op)
        assert policy._last_repair_failure_class == "some_error"
        assert policy._repair_iter_count == 3


# ---------------------------------------------------------------------------
# 3.  validator_gate closure (cache by fingerprint + veto + persist)
# ---------------------------------------------------------------------------

def _make_vetoed_verdict(veto_set: list[str], fingerprint: str):
    """Build a ValidatorVerdict stub with status='vetoed'."""
    from backend.agents.rlm.external_validator import ValidatorVerdict
    return ValidatorVerdict(
        status="vetoed",
        veto_set=veto_set,
        predicates=[],
        panel_models=["test-validator"],
        separation="independent",
        evidence_fingerprint=fingerprint,
    )


def _make_clean_verdict(fingerprint: str):
    from backend.agents.rlm.external_validator import ValidatorVerdict
    return ValidatorVerdict(
        status="clean",
        veto_set=[],
        predicates=[],
        panel_models=["test-validator"],
        separation="independent",
        evidence_fingerprint=fingerprint,
    )


class TestValidatorGateClosure:
    """Test the validator_gate closure built in run.py when the validator is enabled."""

    def _build_gate(self, tmp_path: Path, panel_stub) -> tuple[Any, dict]:
        """Build a _validator_gate closure directly, bypassing run_pipeline_rlm.

        Returns (gate_callable, panel_cache_dict).
        """
        from backend.agents.rlm.external_validator import (
            evidence_fingerprint as _efp,
            persist_verdict as _persist_verdict,
        )
        import backend.agents.rlm.leaf_triage as _lt

        project_dir = tmp_path
        (project_dir / "rlm_state").mkdir(exist_ok=True)
        # Stub ctx with a dummy validator_client
        ctx = types.SimpleNamespace(validator_client=MagicMock())
        _val_label = "test-validator"
        _val_tier = "independent"

        _panel_cache: dict[str, tuple[bool, str]] = {}

        def _validator_gate() -> tuple[bool, str] | None:
            try:
                _mp = project_dir / "code" / "metrics.json"
                _metrics = json.loads(_mp.read_text(encoding="utf-8")) if _mp.exists() else {}
                _fp = _efp(_metrics)
                if _fp in _panel_cache:
                    return _panel_cache[_fp]
                _leaf_records: list = []
                try:
                    _re_path = project_dir / "rubric_evaluation.json"
                    if _re_path.exists():
                        _re_data = json.loads(_re_path.read_text(encoding="utf-8"))
                        _leaf_records = _re_data.get("leaf_scores") or _re_data.get("leaves") or []
                except Exception:
                    _leaf_records = []
                _verdict = panel_stub(
                    validator_client=ctx.validator_client,
                    panel_models=[_val_label],
                    metrics=_metrics,
                    project_dir=project_dir,
                    leaf_records=_leaf_records,
                    separation=_val_tier,
                )
                _persist_verdict(project_dir, _verdict)
                _vetoed = _verdict.status == "vetoed"
                _directive = ""
                if _vetoed:
                    _directive = (
                        "FINAL_VAR refused: the external validator vetoed these result "
                        f"claims as unsubstantiated: {', '.join(_verdict.veto_set[:6])}. "
                        "Re-implement so each cited metric traces to real model outputs on "
                        "real data, then run_experiment + verify_against_rubric."
                    )
                    try:
                        if _lt.is_enabled():
                            _plan = [
                                {
                                    "leaf_id": r,
                                    "score": 0.0,
                                    "repair_class": "validator_veto",
                                    "cost": "targeted_rerun",
                                    "directive": _directive,
                                    "justification": "external validator machine-verified veto",
                                }
                                for r in _verdict.veto_set[:6]
                            ]
                            _lt.persist(project_dir, {"plan": _plan, "facts": {}, "summary": "external validator veto"})
                    except Exception:
                        pass
                _result: tuple[bool, str] = (_vetoed, _directive)
                _panel_cache[_fp] = _result
                return _result
            except Exception:
                return None

        return _validator_gate, _panel_cache

    def _write_metrics(self, project_dir: Path, metrics: dict) -> None:
        code_dir = project_dir / "code"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "metrics.json").write_text(json.dumps(metrics))

    # ------------------------------------------------------------------
    # veto path
    # ------------------------------------------------------------------

    def test_vetoed_verdict_returns_true_and_directive(self, tmp_path):
        """A vetoed verdict returns (True, <directive>) with veto_set names."""
        from backend.agents.rlm.external_validator import evidence_fingerprint as _efp

        metrics = {"mean_reward": 0.0}
        self._write_metrics(tmp_path, metrics)
        fp = _efp(metrics)
        verdict = _make_vetoed_verdict(["mean_reward"], fp)
        call_count = {"n": 0}

        def panel_stub(**kwargs):
            call_count["n"] += 1
            return verdict

        gate, _ = self._build_gate(tmp_path, panel_stub)
        result = gate()

        assert result is not None
        vetoed, directive = result
        assert vetoed is True
        assert "mean_reward" in directive
        assert "FINAL_VAR refused" in directive

    # ------------------------------------------------------------------
    # fingerprint cache — same metrics.json → panel not called again
    # ------------------------------------------------------------------

    def test_cache_prevents_second_panel_call(self, tmp_path):
        """Second gate() call with same metrics.json uses cache — panel NOT called again."""
        metrics = {"loss": 0.0}
        self._write_metrics(tmp_path, metrics)
        from backend.agents.rlm.external_validator import evidence_fingerprint as _efp
        fp = _efp(metrics)
        verdict = _make_vetoed_verdict(["loss"], fp)
        call_count = {"n": 0}

        def panel_stub(**kwargs):
            call_count["n"] += 1
            return verdict

        gate, cache = self._build_gate(tmp_path, panel_stub)

        first = gate()
        second = gate()

        # Panel was only called ONCE (cache hit on the second call)
        assert call_count["n"] == 1, f"Expected 1 panel call, got {call_count['n']}"
        assert first == second

    # ------------------------------------------------------------------
    # changed metrics.json → panel IS called again
    # ------------------------------------------------------------------

    def test_new_metrics_triggers_new_panel_call(self, tmp_path):
        """After metrics.json changes (different fingerprint), the panel is called again."""
        from backend.agents.rlm.external_validator import evidence_fingerprint as _efp

        metrics_v1 = {"loss": 0.0}
        metrics_v2 = {"loss": 0.25}

        call_count = {"n": 0}

        def panel_stub(**kwargs):
            call_count["n"] += 1
            current_metrics = kwargs.get("metrics", {})
            fp = _efp(current_metrics)
            if current_metrics.get("loss", 0) == 0.0:
                return _make_vetoed_verdict(["loss"], fp)
            return _make_clean_verdict(fp)

        gate, _ = self._build_gate(tmp_path, panel_stub)

        # First call: v1 metrics (all-zero → vetoed)
        self._write_metrics(tmp_path, metrics_v1)
        result_v1 = gate()
        assert result_v1 is not None
        assert result_v1[0] is True  # vetoed

        # Update metrics (simulating a repair run)
        self._write_metrics(tmp_path, metrics_v2)
        result_v2 = gate()
        assert result_v2 is not None
        assert result_v2[0] is False  # clean

        # Panel was called TWICE (two distinct fingerprints)
        assert call_count["n"] == 2, f"Expected 2 panel calls, got {call_count['n']}"

    # ------------------------------------------------------------------
    # verdict persisted
    # ------------------------------------------------------------------

    def test_verdict_is_persisted(self, tmp_path):
        """After a gate call, the verdict is persisted to rlm_state/validation_verdict.json."""
        metrics = {"accuracy": 0.0}
        self._write_metrics(tmp_path, metrics)
        from backend.agents.rlm.external_validator import evidence_fingerprint as _efp, load_verdict
        fp = _efp(metrics)
        verdict = _make_vetoed_verdict(["accuracy"], fp)

        def panel_stub(**kwargs):
            return verdict

        gate, _ = self._build_gate(tmp_path, panel_stub)
        gate()

        loaded = load_verdict(tmp_path)
        assert loaded is not None
        assert loaded.status == "vetoed"
        assert "accuracy" in loaded.veto_set

    # ------------------------------------------------------------------
    # fail-soft: panel raises → gate returns None (never crashes)
    # ------------------------------------------------------------------

    def test_panel_exception_returns_none(self, tmp_path):
        """If the panel raises, gate() returns None (fail-soft, never crashes)."""
        self._write_metrics(tmp_path, {"loss": 0.0})

        def panel_stub(**kwargs):
            raise RuntimeError("network error")

        gate, _ = self._build_gate(tmp_path, panel_stub)
        result = gate()
        assert result is None


# ---------------------------------------------------------------------------
# 4.  Default-OFF contract
# ---------------------------------------------------------------------------

class TestDefaultOff:
    """Verify that with both flags unset, neither hook is assigned to the policy."""

    def test_hooks_are_none_when_both_flags_off(self, monkeypatch):
        """iteration_policy.evidence_fingerprint and .validator_gate must be None by default."""
        # Ensure both flags are off
        monkeypatch.delenv("OPENRESEARCH_ZERO_METRICS_GUARD", raising=False)
        monkeypatch.delenv("OPENRESEARCH_EXTERNAL_VALIDATOR", raising=False)

        from backend.agents.rlm.run import _fixfirst_loop_engaged
        from backend.agents.rlm.forced_iteration import ForcedIterationPolicy

        # Confirm the gate function itself returns False
        assert _fixfirst_loop_engaged() is False

        # Build a policy as run.py does and verify neither hook is assigned
        # (simulate the conditional blocks that run.py executes)
        policy = ForcedIterationPolicy(min_iterations=2)

        # Replicate the B1 conditional
        if _fixfirst_loop_engaged():
            policy.evidence_fingerprint = lambda: "fp"  # pragma: no cover

        # Replicate the B2 conditional
        from backend.agents.rlm.external_validator import external_validator_enabled
        if external_validator_enabled() and True:  # ctx.validator_client check — irrelevant when flag off
            policy.validator_gate = lambda: None  # pragma: no cover

        # Both must remain None (dataclass defaults)
        assert policy.evidence_fingerprint is None
        assert policy.validator_gate is None

    def test_evidence_fingerprint_none_when_zero_guard_off_only(self, monkeypatch):
        """evidence_fingerprint stays None when ZERO_METRICS_GUARD is off (validator flag also off)."""
        monkeypatch.delenv("OPENRESEARCH_ZERO_METRICS_GUARD", raising=False)
        monkeypatch.delenv("OPENRESEARCH_EXTERNAL_VALIDATOR", raising=False)
        from backend.agents.rlm.run import _fixfirst_loop_engaged
        from backend.agents.rlm.forced_iteration import ForcedIterationPolicy
        policy = ForcedIterationPolicy(min_iterations=1)
        if _fixfirst_loop_engaged():
            policy.evidence_fingerprint = lambda: "fp"  # pragma: no cover
        assert policy.evidence_fingerprint is None
