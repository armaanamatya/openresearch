"""
Tests for backend/agents/rlm/eval_provenance.py.

All tests are hermetic — no network I/O.  pytest-socket blocks non-loopback;
monkeypatch.setenv controls the feature flag.  Disk-based guard tests use
tmp_path.

Test cases:
  1. Default OFF — guard returns (False, None) even with a bad cell present.
  2. SDAR bug case (headline) — reward×100 in [0,1] but mismatch with records.
  3. Absent sidecar — no eval_provenance.json alongside a rate metric.
  4. Honest pass — metrics.json + sidecar both agree at 0.5.
  5. Out-of-range per-example outcome — outcome=83.0 is not in [0,1].
  6. RL-reward-only exempt — no rate key → conservative exemption.
  7. Producer round-trip — record_eval writes sidecar; guard passes on consistent data.
  8. Non-ok cell skipped — status "failed" with bad metric → no veto.
"""

from __future__ import annotations

import json

import pytest

from backend.agents.rlm.eval_provenance import (
    RATE_METRIC_KEYS,
    SCHEMA_VERSION,
    _RECORDS_SAMPLE_CAP,
    eval_provenance_guard_enabled,
    eval_provenance_repair_message,
    eval_provenance_should_veto,
    record_eval,
)


# ---------------------------------------------------------------------------
# Helper — build a standard cell directory under tmp_path/outputs/run/cell/
# ---------------------------------------------------------------------------

def _make_cell(
    base: "pytest.Path",
    metrics: dict,
    provenance: dict | None = None,
    *,
    subdir: str = "outputs/run_01/cell_0",
) -> "pytest.Path":
    """Write metrics.json (and optionally eval_provenance.json) into a cell dir."""
    cell_dir = base / subdir
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if provenance is not None:
        (cell_dir / "eval_provenance.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )
    return cell_dir


def _write_repo_spec(project_dir, mode: str) -> None:
    """Write ``rlm_state/repo_spec.json`` with the given RESOLVED mode.

    The aggregate-provenance branch is execute-mode-only (fail-closed): the
    guard reads the spec from ``code_dir.parent / "rlm_state"``, so aggregate
    tests pass ``project_dir / "code"`` as the code_dir."""
    state = project_dir / "rlm_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "repo_spec.json").write_text(
        json.dumps({"mode": mode}), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

class TestEvalProvenanceGuardEnabled:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", raising=False)
        assert eval_provenance_guard_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "YES", "ON", " 1 "])
    def test_enabled_values(self, monkeypatch, val):
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", val)
        assert eval_provenance_guard_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "2"])
    def test_disabled_values(self, monkeypatch, val):
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", val)
        assert eval_provenance_guard_enabled() is False


# ---------------------------------------------------------------------------
# Test 1: Default OFF — guard is byte-identical when env unset
# ---------------------------------------------------------------------------

class TestDefaultOff:
    def test_off_even_with_bad_cell(self, tmp_path, monkeypatch):
        """With env var unset, guard returns (False, None) even when a veto-worthy
        cell exists.  This is the byte-identical invariant."""
        monkeypatch.delenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", raising=False)
        # Build a cell that WOULD veto if the guard were on: accuracy=0.83 but no sidecar
        _make_cell(tmp_path, {"status": "ok", "accuracy": 0.83})
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False
        assert detail is None


# ---------------------------------------------------------------------------
# Test 2: SDAR bug case (headline) — reward×100 mismatch
# ---------------------------------------------------------------------------

class TestSdarBugCase:
    def test_reward_times_100_is_vetoed(self, tmp_path, monkeypatch):
        """SDAR: accuracy = mean_success × 100 where success ≡ reward (~0.0083).
        The sidecar records honest per-example outcomes averaging ~0.0083.
        The reported accuracy=0.83 != mean(records)=0.0083 → veto."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")

        # 1 success out of 120 episodes → mean ≈ 0.00833
        n_eps = 120
        records = [{"id": str(i), "outcome": 1.0 if i == 0 else 0.0}
                   for i in range(n_eps)]
        recomputed = 1.0 / n_eps  # ≈ 0.00833

        provenance = {
            "schema_version": SCHEMA_VERSION,
            "model_key": "qwen3-1.7b",
            "env": "alfworld",
            "baseline": "grpo",
            "metric_name": "accuracy",
            "metric_value": recomputed,
            "n_eval": n_eps,
            "held_out": False,
            "records": records[:_RECORDS_SAMPLE_CAP],
            "records_truncated": n_eps > _RECORDS_SAMPLE_CAP,
        }
        _make_cell(
            tmp_path,
            {"status": "ok", "accuracy": 0.83, "reward": recomputed},
            provenance=provenance,
        )

        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is True
        assert detail is not None
        # Message should mention the mismatch
        assert "0.83" in detail or "accuracy" in detail.lower()

    def test_mismatch_detail_mentions_rate_key_and_values(self, tmp_path, monkeypatch):
        """Detail string names the rate key and both reported/recomputed values."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        records = [{"id": "0", "outcome": 0.1}]
        prov = {
            "schema_version": SCHEMA_VERSION,
            "metric_value": 0.1,
            "records": records,
        }
        _make_cell(
            tmp_path,
            {"status": "ok", "accuracy": 0.9},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is True
        assert detail is not None
        assert "accuracy" in detail.lower() or "0.9" in detail


# ---------------------------------------------------------------------------
# Test 3: Absent sidecar — no eval_provenance.json
# ---------------------------------------------------------------------------

class TestAbsentSidecar:
    def test_absent_sidecar_vetoed(self, tmp_path, monkeypatch):
        """status ok + accuracy=0.5 but no eval_provenance.json → veto True."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _make_cell(tmp_path, {"status": "ok", "accuracy": 0.5})
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is True
        assert detail is not None
        assert "eval_provenance" in detail.lower() or "no eval_provenance" in detail.lower()

    def test_absent_sidecar_detail_names_rate_key(self, tmp_path, monkeypatch):
        """Detail should mention the rate key and metric value."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _make_cell(tmp_path, {"status": "ok", "success_rate": 0.72})
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is True
        assert "success_rate" in detail or "0.72" in detail


# ---------------------------------------------------------------------------
# Test 4: Honest pass — consistent sidecar and metrics.json
# ---------------------------------------------------------------------------

class TestHonestPass:
    def test_consistent_data_passes(self, tmp_path, monkeypatch):
        """metrics.json={status:ok, success_rate:0.5} + sidecar records averaging 0.5
        → veto False (the metric is verifiable and correct)."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        records = [{"id": str(i), "outcome": float(i % 2)} for i in range(10)]
        # 5 out of 10 = 0.5
        prov = {
            "schema_version": SCHEMA_VERSION,
            "metric_name": "success_rate",
            "metric_value": 0.5,
            "n_eval": 10,
            "held_out": True,
            "records": records,
            "records_truncated": False,
        }
        _make_cell(
            tmp_path,
            {"status": "ok", "success_rate": 0.5},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False
        assert detail is None

    def test_within_tolerance_passes(self, tmp_path, monkeypatch):
        """A tiny floating-point rounding difference within _RECOMPUTE_TOL must not veto."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        records = [{"id": str(i), "outcome": 1.0 / 3} for i in range(3)]
        reported = 0.3333  # within 1e-3 of 0.33333...
        prov = {
            "schema_version": SCHEMA_VERSION,
            "metric_value": 1.0 / 3,
            "records": records,
        }
        _make_cell(
            tmp_path,
            {"status": "ok", "accuracy": reported},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False

    def test_exactly_1_0_passes(self, tmp_path, monkeypatch):
        """A perfect score of 1.0 with all-1 outcomes must pass."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        records = [{"id": str(i), "outcome": 1.0} for i in range(5)]
        prov = {"schema_version": SCHEMA_VERSION, "metric_value": 1.0, "records": records}
        _make_cell(
            tmp_path,
            {"status": "ok", "accuracy": 1.0},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False


# ---------------------------------------------------------------------------
# Test 5: Out-of-range per-example outcome
# ---------------------------------------------------------------------------

class TestOutOfRangeOutcome:
    def test_outcome_100x_scaled_vetoed(self, tmp_path, monkeypatch):
        """outcome=83.0 is not in [0, 1+eps] → veto True (catches ×100 scaling)."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        records = [{"id": "0", "outcome": 83.0}]
        prov = {
            "schema_version": SCHEMA_VERSION,
            "metric_value": 83.0,
            "records": records,
        }
        _make_cell(
            tmp_path,
            {"status": "ok", "accuracy": 0.83},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is True
        assert detail is not None
        assert "out of" in detail.lower() or "[0,1]" in detail or "outcome" in detail.lower()

    def test_outcome_slightly_above_1_vetoed(self, tmp_path, monkeypatch):
        """outcome=1.5 is out of [0, 1+eps] → veto."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        records = [{"id": "0", "outcome": 1.5}]
        prov = {"schema_version": SCHEMA_VERSION, "metric_value": 1.5, "records": records}
        _make_cell(tmp_path, {"status": "ok", "accuracy": 0.5}, provenance=prov)
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is True

    def test_outcome_exactly_1_passes(self, tmp_path, monkeypatch):
        """outcome=1.0 is in [0, 1] — must pass."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        records = [{"id": "0", "outcome": 1.0}]
        prov = {"schema_version": SCHEMA_VERSION, "metric_value": 1.0, "records": records}
        _make_cell(tmp_path, {"status": "ok", "accuracy": 1.0}, provenance=prov)
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False


# ---------------------------------------------------------------------------
# Test 6: RL-reward-only cells are conservatively exempt
# ---------------------------------------------------------------------------

class TestRlRewardOnlyExempt:
    def test_reward_only_no_rate_key_exempt(self, tmp_path, monkeypatch):
        """metrics.json with reward=0.0083 and no rate key, no sidecar → veto False.
        The guard is conservative: RL-reward-only cells are exempt (they don't claim
        a rate metric)."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _make_cell(tmp_path, {"status": "ok", "reward": 0.0083, "loss": 1.2})
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False
        assert detail is None

    def test_zero_rate_metric_exempt(self, tmp_path, monkeypatch):
        """A rate metric of 0.0 is not > 0 → _find_rate_metric returns None → exempt."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        # accuracy=0.0 is not > 0 → conservative: no veto
        _make_cell(tmp_path, {"status": "ok", "accuracy": 0.0})
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False

    def test_boolean_success_flag_exempt(self, tmp_path, monkeypatch):
        """A boolean `success: true` is a per-cell STATUS flag, not a rate metric —
        float(True)==1.0 must not demand a sidecar (regression: no false veto)."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _make_cell(tmp_path, {"status": "ok", "success": True, "reward": 0.0083})
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False
        assert detail is None


# ---------------------------------------------------------------------------
# Test 7: Producer round-trip — record_eval writes sidecar, guard passes
# ---------------------------------------------------------------------------

class TestProducerRoundTrip:
    def test_round_trip_passes_guard(self, tmp_path, monkeypatch):
        """record_eval(records=[1.0, 0.0]) returns 0.5, writes a sidecar.
        metrics.json with success_rate=0.5 in the same dir → guard veto False."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")

        # Simulate a cell dir under outputs/
        cell_dir = tmp_path / "outputs" / "run_01" / "cell_0"
        cell_dir.mkdir(parents=True, exist_ok=True)

        records = [
            {"id": "1", "outcome": 1.0},
            {"id": "2", "outcome": 0.0},
        ]
        returned_value = record_eval(
            cell_dir,
            model_key="qwen3-1.7b",
            env="alfworld",
            baseline="grpo",
            metric_name="success_rate",
            records=records,
        )

        assert abs(returned_value - 0.5) < 1e-9

        # Write metrics.json with the returned value
        (cell_dir / "metrics.json").write_text(
            json.dumps({"status": "ok", "success_rate": returned_value}),
            encoding="utf-8",
        )

        # Guard must pass: sidecar records average to the reported value
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False, f"Expected no veto but got: {detail}"

    def test_record_eval_returns_correct_mean(self, tmp_path):
        """record_eval returns mean(valid_outcomes) including float coercion."""
        cell_dir = tmp_path / "cell"
        cell_dir.mkdir()
        records = [{"id": str(i), "outcome": float(i) / 10} for i in range(5)]
        # outcomes: 0.0, 0.1, 0.2, 0.3, 0.4 → mean = 0.2
        val = record_eval(
            cell_dir,
            model_key="m", env="e", baseline="b",
            metric_name="accuracy",
            records=records,
        )
        assert abs(val - 0.2) < 1e-9

    def test_record_eval_writes_sidecar(self, tmp_path):
        """record_eval writes eval_provenance.json with expected fields."""
        cell_dir = tmp_path / "cell"
        cell_dir.mkdir()
        records = [{"id": "1", "outcome": 1.0, "prediction": "yes", "gold": "yes"}]
        record_eval(
            cell_dir,
            model_key="model_x", env="env_y", baseline="base_z",
            metric_name="exact_match",
            records=records,
            held_out=True,
            seed=42,
        )
        sidecar_path = cell_dir / "eval_provenance.json"
        assert sidecar_path.exists()
        sc = json.loads(sidecar_path.read_text())
        assert sc["schema_version"] == SCHEMA_VERSION
        assert sc["model_key"] == "model_x"
        assert sc["env"] == "env_y"
        assert sc["baseline"] == "base_z"
        assert sc["metric_name"] == "exact_match"
        assert abs(sc["metric_value"] - 1.0) < 1e-9
        assert sc["held_out"] is True
        assert sc["seed"] == 42
        assert isinstance(sc["records"], list)
        assert len(sc["records"]) == 1
        assert sc["records"][0]["id"] == "1"
        assert abs(sc["records"][0]["outcome"] - 1.0) < 1e-9

    def test_record_eval_empty_records_returns_zero(self, tmp_path):
        """Empty records → value=0.0, sidecar written with n_eval=0."""
        cell_dir = tmp_path / "cell"
        cell_dir.mkdir()
        val = record_eval(
            cell_dir,
            model_key="m", env="e", baseline="b",
            metric_name="accuracy",
            records=[],
        )
        assert val == 0.0
        sc = json.loads((cell_dir / "eval_provenance.json").read_text())
        assert sc["n_eval"] == 0

    def test_record_eval_keeps_all_outcomes_caps_text(self, tmp_path):
        """Every outcome is kept (so the guard can recompute the EXACT mean at any N);
        only the heavy prediction/gold text is dropped beyond _RECORDS_SAMPLE_CAP."""
        cell_dir = tmp_path / "cell"
        cell_dir.mkdir()
        n = _RECORDS_SAMPLE_CAP + 10
        records = [{"id": str(i), "outcome": 1.0, "prediction": "p", "gold": "g"}
                   for i in range(n)]
        record_eval(
            cell_dir,
            model_key="m", env="e", baseline="b",
            metric_name="accuracy",
            records=records,
        )
        sc = json.loads((cell_dir / "eval_provenance.json").read_text())
        # All outcomes retained — exact recompute is possible.
        assert len(sc["records"]) == n
        assert sc["records_truncated"] is True
        # Heavy text kept within the cap, dropped beyond it; the outcome always stays.
        assert "prediction" in sc["records"][0]
        assert "prediction" not in sc["records"][_RECORDS_SAMPLE_CAP + 1]
        assert "outcome" in sc["records"][_RECORDS_SAMPLE_CAP + 1]

    def test_large_eval_round_trip_passes_guard(self, tmp_path, monkeypatch):
        """A >64-example eval through record_eval must NOT false-veto: all outcomes are
        stored so the guard recomputes the exact full-set mean (regression for the
        truncation edge)."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        cell_dir = tmp_path / "outputs" / "run_01" / "cell_0"
        cell_dir.mkdir(parents=True, exist_ok=True)
        # 100 examples, exactly half correct → mean 0.5
        records = [{"id": str(i), "outcome": float(i % 2),
                    "prediction": f"p{i}", "gold": f"g{i}"} for i in range(100)]
        val = record_eval(
            cell_dir, model_key="m", env="e", baseline="b",
            metric_name="success_rate", records=records,
        )
        assert abs(val - 0.5) < 1e-9
        (cell_dir / "metrics.json").write_text(
            json.dumps({"status": "ok", "success_rate": val}), encoding="utf-8",
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False, f"Expected no veto but got: {detail}"

    def test_record_eval_fail_soft_on_bad_records(self, tmp_path):
        """Non-numeric records are skipped; no exception is raised."""
        cell_dir = tmp_path / "cell"
        cell_dir.mkdir()
        val = record_eval(
            cell_dir,
            model_key="m", env="e", baseline="b",
            metric_name="accuracy",
            records=[{"id": "x", "outcome": "not_a_number"}, {"id": "y", "outcome": 0.5}],
        )
        # Only the 0.5 survives → mean = 0.5
        assert abs(val - 0.5) < 1e-9

    def test_record_eval_fail_soft_on_write_error(self, tmp_path):
        """Even if writing fails, record_eval must not raise."""
        # Pass a file as output_dir so mkdir will fail on an existing non-dir path
        bogus_dir = tmp_path / "bogus_file"
        bogus_dir.write_text("not a dir")
        try:
            val = record_eval(
                bogus_dir,
                model_key="m", env="e", baseline="b",
                metric_name="accuracy",
                records=[{"id": "0", "outcome": 0.7}],
            )
            # Return value is still computed correctly even if write failed
            assert abs(val - 0.7) < 1e-9
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"record_eval raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# Test 8: Non-ok cell skipped — status "failed" with bad metric
# ---------------------------------------------------------------------------

class TestNonOkCellSkipped:
    def test_failed_status_not_vetoed(self, tmp_path, monkeypatch):
        """status="failed" + accuracy=0.99 with no sidecar → veto False.
        Non-success cells are not subject to the provenance check."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _make_cell(tmp_path, {"status": "failed", "accuracy": 0.99})
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False

    def test_error_status_not_vetoed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _make_cell(tmp_path, {"status": "error", "success_rate": 0.8})
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False

    @pytest.mark.parametrize("status", ["ok", "success", "succeeded", "completed"])
    def test_success_statuses_are_checked(self, tmp_path, monkeypatch, status):
        """All success-equivalent statuses trigger the check."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        subdir = f"outputs/run/{status}_cell"
        _make_cell(
            tmp_path,
            {"status": status, "accuracy": 0.75},
            subdir=subdir,
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is True  # no sidecar → absent → veto


# ---------------------------------------------------------------------------
# Misc guard behaviours
# ---------------------------------------------------------------------------

class TestGuardMiscBehaviours:
    def test_nonexistent_dir_returns_false(self, monkeypatch):
        """A non-existent code_dir is handled gracefully."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        veto, detail = eval_provenance_should_veto("/nonexistent/path/xyz_12345")
        assert veto is False
        assert detail is None

    def test_no_metrics_files_returns_false(self, tmp_path, monkeypatch):
        """An empty code_dir with no metrics.json → veto False."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False

    def test_top_level_metrics_json_skipped(self, tmp_path, monkeypatch):
        """A metrics.json at the top level (outside outputs/) is skipped."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        (tmp_path / "metrics.json").write_text(
            json.dumps({"status": "ok", "accuracy": 0.9}), encoding="utf-8"
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False

    def test_top_level_metrics_no_status_key_skipped(self, tmp_path, monkeypatch):
        """A metrics.json inside outputs/ but WITHOUT a status key is skipped."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        subdir = tmp_path / "outputs" / "run" / "cell"
        subdir.mkdir(parents=True)
        (subdir / "metrics.json").write_text(
            json.dumps({"accuracy": 0.9}), encoding="utf-8"
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False

    def test_multiple_cells_collects_up_to_3_violations(self, tmp_path, monkeypatch):
        """Violations are capped at 3; the message joins them."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        for i in range(5):
            _make_cell(
                tmp_path,
                {"status": "ok", "accuracy": 0.8},
                subdir=f"outputs/run_01/cell_{i}",
            )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is True
        # Detail should combine up to 3 cells
        parts = detail.split(";") if detail else []
        assert 1 <= len(parts) <= 3

    def test_sidecar_with_mismatching_metric_value_field(self, tmp_path, monkeypatch):
        """Sidecar metric_value disagrees with metrics.json → veto."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        records = [{"id": "0", "outcome": 0.4}]
        prov = {
            "schema_version": SCHEMA_VERSION,
            "metric_value": 0.9,   # disagrees with both records mean AND metrics.json
            "records": records,
        }
        _make_cell(
            tmp_path,
            # reported=0.9, sidecar metric_value=0.9, but recomputed=0.4
            {"status": "ok", "accuracy": 0.9},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is True

    def test_empty_records_list_vetoed(self, tmp_path, monkeypatch):
        """Sidecar with empty records list → veto (no evidence)."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        prov = {"schema_version": SCHEMA_VERSION, "metric_value": 0.5, "records": []}
        _make_cell(tmp_path, {"status": "ok", "accuracy": 0.5}, provenance=prov)
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is True
        assert detail is not None
        assert "empty" in detail.lower() or "records" in detail.lower()


# ---------------------------------------------------------------------------
# Repair message
# ---------------------------------------------------------------------------

class TestRepairMessage:
    def test_returns_string(self):
        msg = eval_provenance_repair_message("some detail")
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_prefixed_with_fabrication_suspected(self):
        msg = eval_provenance_repair_message("some detail")
        assert msg.startswith("fabrication_suspected:")

    def test_detail_embedded(self):
        msg = eval_provenance_repair_message("reported accuracy=0.83 != 0.0083")
        assert "0.83" in msg or "reported" in msg.lower()

    def test_mentions_record_eval(self):
        msg = eval_provenance_repair_message("missing sidecar")
        assert "record_eval" in msg

    def test_mentions_held_out(self):
        msg = eval_provenance_repair_message("missing sidecar")
        assert "held-out" in msg.lower() or "held_out" in msg

    def test_do_not_scale_guidance(self):
        msg = eval_provenance_repair_message("mismatch")
        assert "100" in msg or "scale" in msg.lower()

    def test_fail_soft_on_bad_input(self):
        """Must not raise even on pathological input."""
        try:
            msg = eval_provenance_repair_message(None)  # type: ignore[arg-type]
            assert isinstance(msg, str)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"repair_message raised on None: {exc}")


# ---------------------------------------------------------------------------
# Rate metric key membership
# ---------------------------------------------------------------------------

class TestRateMetricKeys:
    """Verify the RATE_METRIC_KEYS set contains the expected members."""

    @pytest.mark.parametrize("key", [
        "accuracy", "success_rate", "success", "f1", "em",
        "exact_match", "precision", "recall", "pass_rate",
    ])
    def test_expected_keys_present(self, key):
        assert key in RATE_METRIC_KEYS

    def test_reward_not_a_rate_key(self):
        """reward is RL-specific, not a rate metric."""
        assert "reward" not in RATE_METRIC_KEYS

    def test_loss_not_a_rate_key(self):
        assert "loss" not in RATE_METRIC_KEYS


# ---------------------------------------------------------------------------
# Disjointness / held-out-ness checks (new tests)
# ---------------------------------------------------------------------------

class TestDisjointnessCheck:
    """Tests for the eval/train disjointness guard added to eval_provenance_should_veto."""

    def test_overlap_vetoed(self, tmp_path, monkeypatch):
        """Eval ids {"q1","q2"} ∩ train_ids {"q2","q9"} = {"q2"} → veto True,
        detail mentions 'overlaps' and 'held-out'."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        records = [
            {"id": "q1", "outcome": 0.5},
            {"id": "q2", "outcome": 0.5},
        ]
        prov = {
            "schema_version": SCHEMA_VERSION,
            "metric_name": "success_rate",
            "metric_value": 0.5,
            "records": records,
            "train_ids": ["q2", "q9"],
            "n_train": 2,
        }
        _make_cell(
            tmp_path,
            {"status": "ok", "success_rate": 0.5},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is True
        assert detail is not None
        assert "overlap" in detail.lower() or "held-out" in detail.lower()

    def test_disjoint_no_veto(self, tmp_path, monkeypatch):
        """Eval ids {"e1","e2"} ∩ train_ids {"t1","t2"} = {} → veto False
        (value consistent → no veto from recompute check either)."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        records = [
            {"id": "e1", "outcome": 0.5},
            {"id": "e2", "outcome": 0.5},
        ]
        prov = {
            "schema_version": SCHEMA_VERSION,
            "metric_name": "success_rate",
            "metric_value": 0.5,
            "records": records,
            "train_ids": ["t1", "t2"],
            "n_train": 2,
        }
        _make_cell(
            tmp_path,
            {"status": "ok", "success_rate": 0.5},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False
        assert detail is None

    def test_absent_train_ids_no_disjointness_veto(self, tmp_path, monkeypatch):
        """A consistent cell with records but no train_ids in the sidecar → veto False.
        This is the conservative fallback: old runs without train_ids are not penalised."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        records = [
            {"id": "q1", "outcome": 0.5},
            {"id": "q2", "outcome": 0.5},
        ]
        prov = {
            "schema_version": SCHEMA_VERSION,
            "metric_name": "success_rate",
            "metric_value": 0.5,
            "records": records,
            # deliberately no "train_ids" key
        }
        _make_cell(
            tmp_path,
            {"status": "ok", "success_rate": 0.5},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False
        assert detail is None

    def test_record_eval_persists_train_ids(self, tmp_path):
        """record_eval(train_ids=['a','b']) writes train_ids and n_train to the sidecar."""
        cell_dir = tmp_path / "cell"
        cell_dir.mkdir()
        records = [{"id": "e1", "outcome": 1.0}, {"id": "e2", "outcome": 0.0}]
        record_eval(
            cell_dir,
            model_key="m",
            env="alfworld",
            baseline="grpo",
            metric_name="success_rate",
            records=records,
            train_ids=["a", "b"],
        )
        sc = json.loads((cell_dir / "eval_provenance.json").read_text())
        assert sc["train_ids"] == ["a", "b"]
        assert sc["n_train"] == 2

    def test_record_eval_no_train_ids_key_absent(self, tmp_path):
        """When train_ids is None (default), the sidecar must NOT contain the key."""
        cell_dir = tmp_path / "cell"
        cell_dir.mkdir()
        records = [{"id": "e1", "outcome": 1.0}]
        record_eval(
            cell_dir,
            model_key="m",
            env="alfworld",
            baseline="grpo",
            metric_name="success_rate",
            records=records,
            # train_ids omitted → None by default
        )
        sc = json.loads((cell_dir / "eval_provenance.json").read_text())
        assert "train_ids" not in sc


# ---------------------------------------------------------------------------
# Aggregate-provenance schema (verl_metrics_adapter) — distinct valid schema
# ---------------------------------------------------------------------------

class TestAggregateProvenance:
    """Tests for the aggregate-provenance branch: execute-mode's
    ``verl_metrics_adapter`` writes a sidecar with a verbatim-copied AGGREGATE
    ``metric_value`` and NO per-example ``records`` (verl exposes only an
    aggregate — synthesizing fake per-example records would violate the
    no-fabrication red line). This schema is verified by a value-preserving
    cross-check against ``metrics.json`` instead of a records-recompute, and
    is accepted ONLY when ``rlm_state/repo_spec.json`` mode == "execute"."""

    def test_aggregate_matching_value_passes(self, tmp_path, monkeypatch):
        """Execute mode: provenance_kind=aggregate, metric_value matches
        metrics.json, source exists on disk → veto False."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _write_repo_spec(tmp_path, "execute")
        code = tmp_path / "code"
        source_file = code / "val_log.log"
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text("val/success_rate: 0.456", encoding="utf-8")
        prov = {
            "schema_version": 1,
            "adapter": "verl_metrics_adapter",
            "provenance_kind": "aggregate",
            "metric_value": 0.456,
            "source": str(source_file),
            "source_line": "val/success_rate: 0.456",
        }
        _make_cell(
            code,
            {"status": "success", "success_rate": 0.456},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(code)
        assert veto is False
        assert detail is None

    def test_aggregate_mismatched_value_vetoed(self, tmp_path, monkeypatch):
        """metrics.json success_rate disagrees with the sidecar's aggregate
        metric_value (e.g. metrics.json tampered after the fact) → veto."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _write_repo_spec(tmp_path, "execute")
        code = tmp_path / "code"
        source_file = code / "val_log.log"
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text("val/success_rate: 0.456", encoding="utf-8")
        prov = {
            "provenance_kind": "aggregate",
            "adapter": "verl_metrics_adapter",
            "metric_value": 0.456,
            "source": str(source_file),
        }
        _make_cell(
            code,
            {"status": "success", "success_rate": 0.9},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(code)
        assert veto is True
        assert detail is not None
        assert "aggregate" in detail.lower()

    def test_aggregate_metric_value_out_of_range_vetoed(self, tmp_path, monkeypatch):
        """metric_value=45.6 (×100-scaled) is out of [0,1] → veto (even in
        execute mode — the value checks stay authoritative)."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _write_repo_spec(tmp_path, "execute")
        code = tmp_path / "code"
        prov = {
            "provenance_kind": "aggregate",
            "metric_value": 45.6,
        }
        _make_cell(
            code,
            {"status": "success", "success_rate": 45.6},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(code)
        assert veto is True
        assert detail is not None
        assert "[0,1]" in detail or "out of" in detail.lower()

    def test_aggregate_nonexistent_source_not_vetoed_round_trip(self, tmp_path, monkeypatch):
        """source points at a file that does not exist on the host → NOT a veto.

        On the GKE path the ``source`` is a POD-side absolute path that
        legitimately does not exist on the orchestrator host after the
        pod→GCS→host round-trip, and the host guard cannot distinguish that
        from a fabricated path. Source existence is therefore best-effort only;
        the value cross-checks (metric_value finite, in [0,1], reported ==
        metric_value) are the real honesty guarantee, and they pass here."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _write_repo_spec(tmp_path, "execute")
        code = tmp_path / "code"
        prov = {
            "provenance_kind": "aggregate",
            "adapter": "verl_metrics_adapter",  # the real adapter always stamps this
            "metric_value": 0.456,
            "source": str(code / "does_not_exist.log"),
        }
        _make_cell(
            code,
            {"status": "success", "success_rate": 0.456},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(code)
        assert veto is False
        assert detail is None

    def test_aggregate_no_source_no_marker_is_unverifiable_vetoed(self, tmp_path, monkeypatch):
        """EXECUTE mode, but the aggregate is NEITHER stamped by the trusted
        verl_metrics_adapter NOR backed by a resolvable source file →
        unverifiable self-attestation → veto. Internal consistency alone is not
        a real held-out measurement, even in execute mode (closes the
        self-attestation loophole a pure-inline aggregate would otherwise open)."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _write_repo_spec(tmp_path, "execute")
        code = tmp_path / "code"
        prov = {
            "provenance_kind": "aggregate",
            "metric_value": 0.456,
        }
        _make_cell(
            code,
            {"status": "success", "success_rate": 0.456},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(code)
        assert veto is True
        assert detail is not None
        assert "unverifiable" in detail.lower()

    def test_aggregate_resolvable_source_no_marker_passes(self, tmp_path, monkeypatch):
        """The local/docker path: a source file that resolves on disk binds the
        aggregate to a real artifact even without the producer marker → pass
        (execute mode)."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _write_repo_spec(tmp_path, "execute")
        code = tmp_path / "code"
        src = code / "outputs" / "run_01" / "cell_0" / "train.log"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("val/success_rate: 0.456", encoding="utf-8")
        prov = {
            "provenance_kind": "aggregate",
            "metric_value": 0.456,
            "source": str(src),
        }
        _make_cell(
            code,
            {"status": "success", "success_rate": 0.456},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(code)
        assert veto is False
        assert detail is None

    def test_adapter_marker_alone_detected_as_aggregate(self, tmp_path, monkeypatch):
        """adapter == 'verl_metrics_adapter' alone (no explicit provenance_kind)
        is also detected as an aggregate sidecar (execute mode)."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _write_repo_spec(tmp_path, "execute")
        code = tmp_path / "code"
        prov = {
            "adapter": "verl_metrics_adapter",
            "metric_value": 0.456,
        }
        _make_cell(
            code,
            {"status": "success", "success_rate": 0.456},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(code)
        assert veto is False
        assert detail is None

    def test_regression_records_based_sidecar_unaffected(self, tmp_path, monkeypatch):
        """A normal records-based sidecar (no provenance_kind/adapter marker)
        still goes through the pre-existing records-recompute path unchanged —
        no repo_spec.json needed (the mode gate applies only to aggregates)."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        records = [{"id": str(i), "outcome": float(i % 2)} for i in range(10)]
        prov = {
            "schema_version": SCHEMA_VERSION,
            "metric_value": 0.5,
            "records": records,
        }
        _make_cell(
            tmp_path,
            {"status": "ok", "success_rate": 0.5},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False
        assert detail is None


# ---------------------------------------------------------------------------
# Aggregate-provenance mode gate — execute-only, fail-closed
# ---------------------------------------------------------------------------

class TestAggregateModeGate:
    """The aggregate schema is legal ONLY for execute-mode runs (resolved
    ``rlm_state/repo_spec.json`` mode == "execute"). Adapt-mode / mode-unknown
    runs must persist per-example records — an adapter-stamped aggregate there
    is vetoed (fail-closed)."""

    _AGG_PROV = {
        "schema_version": 1,
        "adapter": "verl_metrics_adapter",
        "provenance_kind": "aggregate",
        "metric_value": 0.456,
        "source": "/pod/only/train.log",
        "source_line": "val/success_rate: 0.456",
    }

    def test_execute_mode_aggregate_passes(self, tmp_path, monkeypatch):
        """(a) execute-mode aggregate sidecar (trusted adapter stamp) → pass."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _write_repo_spec(tmp_path, "execute")
        code = tmp_path / "code"
        _make_cell(
            code,
            {"status": "success", "success_rate": 0.456},
            provenance=dict(self._AGG_PROV),
        )
        veto, detail = eval_provenance_should_veto(code)
        assert veto is False
        assert detail is None

    def test_adapt_mode_aggregate_vetoed(self, tmp_path, monkeypatch):
        """(b) adapt-mode aggregate sidecar → veto, even with the trusted
        adapter stamp: adapt-mode trainers must record per-example outcomes."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _write_repo_spec(tmp_path, "adapt")
        code = tmp_path / "code"
        _make_cell(
            code,
            {"status": "success", "success_rate": 0.456},
            provenance=dict(self._AGG_PROV),
        )
        veto, detail = eval_provenance_should_veto(code)
        assert veto is True
        assert detail is not None
        assert "execute" in detail.lower()
        assert "record_eval" in detail

    def test_missing_repo_spec_aggregate_vetoed(self, tmp_path, monkeypatch):
        """Mode-unknown (no rlm_state/repo_spec.json at all) → fail-closed veto.
        An unknown mode never widens acceptance."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        code = tmp_path / "code"
        _make_cell(
            code,
            {"status": "success", "success_rate": 0.456},
            provenance=dict(self._AGG_PROV),
        )
        veto, detail = eval_provenance_should_veto(code)
        assert veto is True
        assert detail is not None
        assert "unknown" in detail.lower()

    @pytest.mark.parametrize("mode", ["reference", "scratch", "", None])
    def test_other_modes_aggregate_vetoed(self, tmp_path, monkeypatch, mode):
        """Any non-execute resolved mode (reference/scratch/blank/null) → veto."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        state = tmp_path / "rlm_state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "repo_spec.json").write_text(
            json.dumps({"mode": mode}), encoding="utf-8"
        )
        code = tmp_path / "code"
        _make_cell(
            code,
            {"status": "success", "success_rate": 0.456},
            provenance=dict(self._AGG_PROV),
        )
        veto, detail = eval_provenance_should_veto(code)
        assert veto is True
        assert detail is not None

    def test_adapt_mode_records_sidecar_still_passes(self, tmp_path, monkeypatch):
        """Adapt mode with a proper per-example records sidecar is unaffected
        by the mode gate — the records path stays byte-identical."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _write_repo_spec(tmp_path, "adapt")
        code = tmp_path / "code"
        records = [{"id": str(i), "outcome": float(i % 2)} for i in range(10)]
        prov = {
            "schema_version": SCHEMA_VERSION,
            "metric_value": 0.5,
            "records": records,
        }
        _make_cell(
            code,
            {"status": "ok", "success_rate": 0.5},
            provenance=prov,
        )
        veto, detail = eval_provenance_should_veto(code)
        assert veto is False
        assert detail is None

    def test_unreadable_repo_spec_aggregate_vetoed(self, tmp_path, monkeypatch):
        """A corrupt repo_spec.json resolves to mode-unknown → fail-closed veto."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        state = tmp_path / "rlm_state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "repo_spec.json").write_text("{not json", encoding="utf-8")
        code = tmp_path / "code"
        _make_cell(
            code,
            {"status": "success", "success_rate": 0.456},
            provenance=dict(self._AGG_PROV),
        )
        veto, detail = eval_provenance_should_veto(code)
        assert veto is True


# ---------------------------------------------------------------------------
# Inline provenance (GKE round-trip) + honest reward_mean exemption
# ---------------------------------------------------------------------------

class TestInlineProvenanceRoundTrip:
    """The GKE pod entrypoint uploads only metrics.json — a standalone
    eval_provenance.json sidecar never survives the pod→GCS→host round-trip.
    The adapter therefore also embeds the aggregate provenance INLINE under
    metrics.json['eval_provenance'], and the guard credits it identically."""

    def test_inline_aggregate_provenance_credited(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _write_repo_spec(tmp_path, "execute")
        code = tmp_path / "code"
        metrics = {
            "status": "success",
            "success_rate": 0.42,
            "eval_provenance": {
                "provenance_kind": "aggregate",
                "adapter": "verl_metrics_adapter",
                "metric_name": "success_rate",
                "metric_value": 0.42,
                "source": "/pod/only/train.log",  # a pod path, absent on host
            },
        }
        _make_cell(code, metrics, provenance=None)  # NO sidecar
        veto, detail = eval_provenance_should_veto(code)
        assert veto is False
        assert detail is None

    def test_inline_provenance_value_mismatch_still_vetoed(self, tmp_path, monkeypatch):
        """Inline provenance whose metric_value disagrees with the reported rate
        metric is still vetoed — the value cross-check stays authoritative."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _write_repo_spec(tmp_path, "execute")
        code = tmp_path / "code"
        metrics = {
            "status": "success",
            "success_rate": 0.90,
            "eval_provenance": {"provenance_kind": "aggregate", "metric_value": 0.42},
        }
        _make_cell(code, metrics, provenance=None)
        veto, detail = eval_provenance_should_veto(code)
        assert veto is True
        assert detail is not None

    def test_sidecar_still_wins_when_present(self, tmp_path, monkeypatch):
        """A standalone sidecar (local/docker path) is used when present, even if
        an inline block also exists — sidecar read is tried first. Both carry the
        trusted producer marker, so the honesty binding is satisfied."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _write_repo_spec(tmp_path, "execute")
        code = tmp_path / "code"
        prov = {
            "provenance_kind": "aggregate",
            "adapter": "verl_metrics_adapter",
            "metric_value": 0.42,
        }
        metrics = {"status": "success", "success_rate": 0.42, "eval_provenance": prov}
        _make_cell(code, metrics, provenance=prov)
        veto, detail = eval_provenance_should_veto(code)
        assert veto is False
        assert detail is None

    def test_reward_mean_cell_is_exempt(self, tmp_path, monkeypatch):
        """A train-reward cell writes reward_mean (NOT a rate-metric key), so the
        guard treats it as RL-reward-only and never vetoes for a missing sidecar —
        the honest fallback when no held-out val score is available."""
        monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
        _make_cell(
            tmp_path,
            {"status": "success", "reward_mean": 0.141, "reward": 0.141, "metric": 0.141},
            provenance=None,
        )
        veto, detail = eval_provenance_should_veto(tmp_path)
        assert veto is False
        assert detail is None
