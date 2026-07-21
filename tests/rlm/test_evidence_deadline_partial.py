"""Fix C — credit a DEADLINE/SIGNAL-killed cell that has REAL metrics as PARTIAL
evidence (cap the verdict at 'partial'), never downgrade it to 'failed'.

Incident: a GKE training cell ran real training (WRN-28-10 baseline 3.92% /
cutout 3.43%), uploaded a valid metrics.json, but was SIGTERM'd at its 6h
wall-clock deadline BEFORE its script wrote a terminal status:"completed". Its
run_experiment row was recorded ``success:false, exit_code:null (None)``. The
evidence gate (report._has_experiment_evidence) requires a ``success==True``
row, found none, and downgraded the verdict to "failed" — a false "failed",
because real metrics existed on disk.

The fix extends ``_has_partial_timeout_evidence`` to ALSO accept a
signal/deadline-killed row (``exit_code is None``) that carries at least one
real metric key. This caps the verdict at "partial" — it NEVER enables a full
"reproduced" verdict (that still needs a clean ``success==True`` row via
``_has_experiment_evidence``), so the fail-closed "evidence not grade" invariant
is preserved.

Guardrails covered here:
  * a code-error row (real non-None ``exit_code`` like 41) is NOT credited;
  * a killed row with only bookkeeping metrics (no real metric key) is NOT
    credited;
  * the existing ``partial_timeout`` behavior is not regressed;
  * at the verdict level the gate caps at "partial", never "failed", and this
    path can NEVER yield "reproduced".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_exp_runs(project_dir: Path, rows: list[dict]) -> None:
    """Write one JSONL line per row to experiment_runs.jsonl."""
    project_dir.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    (project_dir / "experiment_runs.jsonl").write_text(text, encoding="utf-8")


def _make_report(verdict: str = "partial"):
    from backend.agents.rlm.report import RLMFinalReport

    return RLMFinalReport(
        verdict=verdict,
        reproduction_summary="test summary",
        baseline_metrics={"test_err": 3.43},
    )


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    # The verdict gate reads OPENRESEARCH_EVIDENCE_GATE (default ON) and the
    # unified critic reads OPENRESEARCH_EVIDENCE_AUDIT; keep both hermetic.
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_GATE", raising=False)
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_AUDIT", raising=False)


# ---------------------------------------------------------------------------
# _has_partial_timeout_evidence — predicate-level
# ---------------------------------------------------------------------------


class TestHasPartialTimeoutEvidence:
    def test_deadline_killed_row_with_nested_metrics_counts(self, tmp_path):
        """success:false + exit_code:null + real nested metrics → True."""
        from backend.agents.rlm.report import _has_partial_timeout_evidence

        _write_exp_runs(
            tmp_path,
            [{
                "success": False,
                "exit_code": None,
                "metrics": {"per_model": {"wrn_28_10": {"cifar10": {
                    "baseline": {"status": "ok", "test_error_percent": 3.92}}}}},
            }],
        )
        assert _has_partial_timeout_evidence(tmp_path) is True

    def test_deadline_killed_row_with_real_toplevel_metric_counts(self, tmp_path):
        """exit_code:null + a real top-level metric key → True."""
        from backend.agents.rlm.report import _has_partial_timeout_evidence

        _write_exp_runs(
            tmp_path,
            [{
                "success": False,
                "exit_code": None,
                "metrics": {"metric": 0.96, "test_err": 3.43},
            }],
        )
        assert _has_partial_timeout_evidence(tmp_path) is True

    def test_code_error_with_real_exit_code_not_credited(self, tmp_path):
        """GUARD: a real non-None exit_code (code bug) → NOT credited."""
        from backend.agents.rlm.report import _has_partial_timeout_evidence

        _write_exp_runs(
            tmp_path,
            [{"success": False, "exit_code": 41, "metrics": {}}],
        )
        assert _has_partial_timeout_evidence(tmp_path) is False

    def test_killed_row_with_only_bookkeeping_metrics_not_credited(self, tmp_path):
        """GUARD: exit_code:null but no REAL metric key (only status/error) → False."""
        from backend.agents.rlm.report import _has_partial_timeout_evidence

        _write_exp_runs(
            tmp_path,
            [{
                "success": False,
                "exit_code": None,
                "metrics": {"status": "running", "error": "x"},
            }],
        )
        assert _has_partial_timeout_evidence(tmp_path) is False

    def test_killed_row_with_empty_metrics_not_credited(self, tmp_path):
        """GUARD: exit_code:null + empty metrics dict → False."""
        from backend.agents.rlm.report import _has_partial_timeout_evidence

        _write_exp_runs(
            tmp_path,
            [{"success": False, "exit_code": None, "metrics": {}}],
        )
        assert _has_partial_timeout_evidence(tmp_path) is False

    def test_existing_partial_timeout_behavior_preserved(self, tmp_path):
        """Regression: the existing failure_class=='partial_timeout' branch still True."""
        from backend.agents.rlm.report import _has_partial_timeout_evidence

        _write_exp_runs(
            tmp_path,
            [{
                "success": False,
                "failure_class": "partial_timeout",
                "partial_timeout": True,
                "exit_code": 137,  # a partial_timeout may carry a signal exit code
                "metrics": {"accuracy": 0.9},
            }],
        )
        assert _has_partial_timeout_evidence(tmp_path) is True

    def test_missing_file_failsoft(self, tmp_path):
        from backend.agents.rlm.report import _has_partial_timeout_evidence

        assert _has_partial_timeout_evidence(tmp_path) is False


# ---------------------------------------------------------------------------
# _row_has_real_metrics — helper unit tests
# ---------------------------------------------------------------------------


class TestRowHasRealMetrics:
    def test_real_toplevel_metric(self):
        from backend.agents.rlm.report import _row_has_real_metrics

        assert _row_has_real_metrics({"test_err": 3.43}) is True

    def test_real_nested_metric(self):
        from backend.agents.rlm.report import _row_has_real_metrics

        assert _row_has_real_metrics({"per_model": {"m": {}}}) is True

    def test_only_bookkeeping_keys(self):
        from backend.agents.rlm.report import _row_has_real_metrics

        assert _row_has_real_metrics(
            {"status": "x", "error": "y", "timestamp": "z",
             "artifact_dir": "d", "artifact_paths": []}
        ) is False

    def test_empty_and_non_dict(self):
        from backend.agents.rlm.report import _row_has_real_metrics

        assert _row_has_real_metrics({}) is False
        assert _row_has_real_metrics(None) is False
        assert _row_has_real_metrics([1, 2]) is False


# ---------------------------------------------------------------------------
# Verdict-level: the gate caps at 'partial', NEVER 'failed', NEVER 'reproduced'
# ---------------------------------------------------------------------------


class TestVerdictCappedAtPartial:
    def test_deadline_killed_metrics_caps_partial_not_failed(self, tmp_path):
        """The only evidence is a deadline-killed-with-metrics row. A real
        run_experiment call backs it (calls=1) but no clean success row exists.
        The gate must CAP at 'partial', not downgrade to 'failed'.

        We pass run_experiment_partial_timeout_calls=None to model the real
        hard-stop/watchdog finalize path where the killed cell was NOT
        harness-finalized as a partial_timeout (so no partial_timeout ledger
        stamp exists) — content-only trust, backed by a real experiment call.
        """
        from backend.agents.rlm.report import _apply_evidence_gate

        _write_exp_runs(
            tmp_path,
            [{
                "success": False,
                "exit_code": None,
                "metrics": {"per_model": {"wrn_28_10": {"cifar10": {
                    "baseline": {"status": "ok", "test_error_percent": 3.92},
                    "cutout": {"status": "ok", "test_error_percent": 3.43}}}}},
            }],
        )
        report = _make_report("partial")
        result = _apply_evidence_gate(
            report,
            tmp_path,
            run_experiment_calls=1,
            run_experiment_ok_calls=1,
            run_experiment_partial_timeout_calls=None,
        )
        assert result.verdict == "partial"
        assert "evidence_cap" in (result.reproduction_summary or "")

    def test_reproduced_capped_down_to_partial_never_stays_reproduced(self, tmp_path):
        """This evidence path can only yield 'partial' — a starting 'reproduced'
        verdict backed ONLY by a deadline-killed row is capped to 'partial'."""
        from backend.agents.rlm.report import _apply_evidence_gate

        _write_exp_runs(
            tmp_path,
            [{
                "success": False,
                "exit_code": None,
                "metrics": {"test_err": 3.43},
            }],
        )
        report = _make_report("reproduced")
        result = _apply_evidence_gate(
            report,
            tmp_path,
            run_experiment_calls=1,
            run_experiment_ok_calls=1,
            run_experiment_partial_timeout_calls=None,
        )
        assert result.verdict == "partial"
        assert result.verdict != "reproduced"

    def test_code_error_row_still_downgrades_to_failed(self, tmp_path):
        """GUARD at the verdict level: a code-error row (real exit_code) with no
        clean success row is NOT partial-credited — it still downgrades to
        'failed'."""
        from backend.agents.rlm.report import _apply_evidence_gate

        _write_exp_runs(
            tmp_path,
            [{"success": False, "exit_code": 41, "metrics": {}}],
        )
        report = _make_report("partial")
        result = _apply_evidence_gate(
            report,
            tmp_path,
            run_experiment_calls=1,
            run_experiment_ok_calls=1,
            run_experiment_partial_timeout_calls=None,
        )
        assert result.verdict == "failed"

    def test_forged_deadline_row_zero_calls_downgrades_to_failed(self, tmp_path):
        """GUARD: a deadline-shaped row with ZERO in-process run_experiment calls
        is forged (REPL-appended) — it must NOT be partial-credited."""
        from backend.agents.rlm.report import _apply_evidence_gate

        _write_exp_runs(
            tmp_path,
            [{
                "success": False,
                "exit_code": None,
                "metrics": {"test_err": 3.43},
            }],
        )
        # A killed-with-metrics row does not itself satisfy _has_experiment_evidence
        # (success is False), so the forged-branch (which requires content_evidence
        # = a success row) does not fire; instead the no-partial-cap path must apply
        # because there is no real run_experiment call to back the partial cap.
        report = _make_report("partial")
        result = _apply_evidence_gate(
            report,
            tmp_path,
            run_experiment_calls=0,
            run_experiment_ok_calls=0,
            run_experiment_partial_timeout_calls=0,
        )
        assert result.verdict == "failed"
