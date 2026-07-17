"""Tests for verl_metrics_adapter — value-preserving verl log/JSON → metrics.json.

Motivating case: a cell running the authors' verl trainer verbatim (execute
mode Change #1) writes its OWN log format, not the harness's flat
``metrics.json`` contract. This adapter parses the authors' output and
produces the canonical shape WITHOUT scaling/recomputing the measured value.

Key regex-robustness requirement: a per-dataset sub-key
(``val/success_rate/nq:0.418``) must NOT be confused with the aggregate key
(``val/success_rate:0.456``) — only the aggregate, whole-token match wins,
and among multiple aggregate occurrences the LAST one in the log wins.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm.verl_metrics_adapter import write_cell_metrics_from_verl


def _write_log(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Log-regex path
# ---------------------------------------------------------------------------

class TestVerlMetricsAdapterLogRegex:
    def test_aggregate_key_wins_over_per_dataset_subkey(self, tmp_path):
        _write_log(
            tmp_path / "train.log",
            "step 100: val/success_rate/nq:0.418\n"
            "step 100: val/success_rate:0.456\n"
            "step 100: sdar/gate_mean:0.478\n",
        )
        result = write_cell_metrics_from_verl(
            tmp_path, model_key="qwen3-1.7b", env="search-qa", baseline="grpo",
        )
        assert result["success_rate"] == 0.456
        assert result["status"] == "success"

    def test_value_is_preserved_verbatim_not_scaled(self, tmp_path):
        _write_log(tmp_path / "train.log", "val/success_rate: 0.456\n")
        result = write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        assert result["success_rate"] == 0.456  # NOT ×100, NOT recomputed

    def test_last_occurrence_across_log_wins(self, tmp_path):
        _write_log(
            tmp_path / "train.log",
            "val/success_rate:0.1\nval/success_rate:0.2\nval/success_rate:0.456\n",
        )
        result = write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        assert result["success_rate"] == 0.456

    def test_equals_separator_supported(self, tmp_path):
        _write_log(tmp_path / "train.log", "val/success_rate=0.456\n")
        result = write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        assert result["success_rate"] == 0.456

    def test_writes_metrics_json(self, tmp_path):
        _write_log(tmp_path / "train.log", "val/success_rate:0.456\n")
        write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        metrics = json.loads((tmp_path / "metrics.json").read_text())
        assert metrics["success_rate"] == 0.456
        assert metrics["status"] == "success"

    def test_writes_eval_provenance_sidecar_naming_source(self, tmp_path):
        log_path = tmp_path / "train.log"
        _write_log(log_path, "val/success_rate:0.456\n")
        write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        sidecar = json.loads((tmp_path / "eval_provenance.json").read_text())
        assert sidecar["source"] == str(log_path)
        assert "0.456" in sidecar["source_line"]

    def test_extra_keys_copied_verbatim(self, tmp_path):
        _write_log(
            tmp_path / "train.log",
            "val/success_rate:0.456\nsdar/gate_mean:0.478\n",
        )
        result = write_cell_metrics_from_verl(
            tmp_path, model_key="m", env="e", baseline="b",
            extra_keys=("sdar/gate_mean",),
        )
        assert result["sdar/gate_mean"] == 0.478

    def test_custom_success_rate_key(self, tmp_path):
        _write_log(tmp_path / "train.log", "val/exact_match:0.321\n")
        result = write_cell_metrics_from_verl(
            tmp_path, model_key="m", env="e", baseline="b",
            success_rate_key="val/exact_match",
        )
        assert result["success_rate"] == 0.321

    def test_custom_log_glob_with_output_dir_placeholder(self, tmp_path):
        subdir = tmp_path / "logs"
        subdir.mkdir()
        _write_log(subdir / "run.txt", "val/success_rate:0.456\n")
        result = write_cell_metrics_from_verl(
            tmp_path, model_key="m", env="e", baseline="b",
            log_glob="$OUTPUT_DIR/logs/*.txt",
        )
        assert result["success_rate"] == 0.456


# ---------------------------------------------------------------------------
# Fail-honest
# ---------------------------------------------------------------------------

class TestVerlMetricsAdapterFailHonest:
    def test_missing_source_gives_failed_status_no_success_rate_key(self, tmp_path):
        result = write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        assert result["status"] == "failed"
        assert "success_rate" not in result

    def test_log_present_but_no_matching_key_gives_failed_status(self, tmp_path):
        _write_log(tmp_path / "train.log", "irrelevant line\nsome/other_metric:1.0\n")
        result = write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        assert result["status"] == "failed"
        assert "success_rate" not in result

    def test_failed_result_written_to_metrics_json(self, tmp_path):
        write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        metrics = json.loads((tmp_path / "metrics.json").read_text())
        assert metrics == {"status": "failed"}

    def test_failed_result_writes_no_eval_provenance_sidecar(self, tmp_path):
        write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        assert not (tmp_path / "eval_provenance.json").exists()


# ---------------------------------------------------------------------------
# JSON preference (machine-readable summary wins over log regex)
# ---------------------------------------------------------------------------

class TestVerlMetricsAdapterJsonPreference:
    def test_prefers_json_summary_over_log_when_both_present(self, tmp_path):
        (tmp_path / "val_summary.json").write_text(
            json.dumps({"val/success_rate": 0.789}), encoding="utf-8",
        )
        _write_log(tmp_path / "train.log", "val/success_rate:0.456\n")
        result = write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        assert result["success_rate"] == 0.789

    def test_nested_json_shape_also_resolves(self, tmp_path):
        (tmp_path / "run_summary.json").write_text(
            json.dumps({"val": {"success_rate": 0.654}}), encoding="utf-8",
        )
        result = write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        assert result["success_rate"] == 0.654


class TestVerlMetricsAdapterRealVerlDictRepr:
    """verl logs val metrics as a Python dict repr on the console, e.g.
    ``'val/success_rate': np.float64(0.4562844669117647)`` — confirmed against a
    real /mnt/sdar-cache/logs/run_search_3b.log (Search-3B proof, 0.456). These
    lock in that exact format so the adapter never silently misses the metric on
    a live run."""

    _REAL = (
        "(SDARTaskRunner pid=101401) validation metrics: {\n"
        "  \"'val/nq_success_rate': np.float64(0.512), \"\n"
        "  \"'val/musique_success_rate': np.float64(0.14740774842642354), \"\n"
        "  \"'val/2wikimultihopqa_success_rate': np.float64(0.399184718833248), \"\n"
        "  \"'val/success_rate': np.float64(0.4562844669117647), \"\n"
        "  \"'val/bamboogle_success_rate': np.float64(0.6693548387096774)}\"\n"
    )

    def test_real_np_float64_dict_repr_parses_aggregate_only(self, tmp_path):
        _write_log(tmp_path / "cell_stdout.log", self._REAL)
        result = write_cell_metrics_from_verl(
            tmp_path, model_key="Qwen2.5-3B-Instruct", env="search_qa",
            baseline="sdar", success_rate_key="val/success_rate",
        )
        # The aggregate, verbatim — never a per-dataset sub-key, never scaled.
        assert result["success_rate"] == 0.4562844669117647
        assert result["status"] == "success"

    def test_np_float32_wrapper(self, tmp_path):
        _write_log(tmp_path / "t.log", "'val/success_rate': np.float32(0.456)\n")
        result = write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        assert result["success_rate"] == 0.456

    def test_torch_tensor_wrapper(self, tmp_path):
        _write_log(tmp_path / "t.log", "val/success_rate: tensor(0.456)\n")
        result = write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        assert result["success_rate"] == 0.456

    def test_double_quoted_json_ish_form(self, tmp_path):
        _write_log(tmp_path / "t.log", '"val/success_rate": 0.456\n')
        result = write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        assert result["success_rate"] == 0.456

    def test_per_dataset_only_no_aggregate_fails_honest(self, tmp_path):
        # Only per-dataset keys, no aggregate → honest failure, never a fabricated
        # number or a mis-picked per-dataset value.
        _write_log(tmp_path / "t.log", "'val/musique_success_rate': np.float64(0.147)\n")
        result = write_cell_metrics_from_verl(
            tmp_path, model_key="m", env="e", baseline="b",
            success_rate_key="val/success_rate",
        )
        assert result == {"status": "failed"}
        assert "success_rate" not in result


# ---------------------------------------------------------------------------
# Held-out eval keys: prefix match (verl per-data_source val), metric_name,
# inline provenance
# ---------------------------------------------------------------------------

class TestHeldOutEvalKeys:
    def test_prefix_key_resolves_single_data_source(self, tmp_path):
        """verl logs one val key PER data_source (val/test_score/math) and no bare
        aggregate; the 'val/test_score/' PREFIX resolves the single concrete
        sub-key, written as success_rate with aggregate provenance."""
        _write_log(
            tmp_path / "train.log",
            "step 4: 'val/test_score/math': np.float64(0.575)\n",
        )
        result = write_cell_metrics_from_verl(
            tmp_path, model_key="m", env="verl", baseline="execute",
            success_rate_key="val/test_score/", metric_name="success_rate",
        )
        assert result["status"] == "success"
        assert result["success_rate"] == 0.575
        assert result["eval_provenance"]["metric_value"] == 0.575
        assert result["eval_provenance"]["provenance_kind"] == "aggregate"

    def test_prefix_key_data_source_with_slash(self, tmp_path):
        """A data_source that itself contains '/' (openai/gsm8k) is captured whole."""
        _write_log(tmp_path / "train.log", "val/test_score/openai/gsm8k:0.61\n")
        result = write_cell_metrics_from_verl(
            tmp_path, success_rate_key="val/test_score/",
            model_key="m", env="e", baseline="b",
        )
        assert result["status"] == "success"
        assert result["success_rate"] == 0.61

    def test_prefix_key_ambiguous_multi_source_falls_through(self, tmp_path):
        """Two distinct data_sources under the prefix → ambiguous → no value
        (never averages across sources — not value-preserving)."""
        _write_log(
            tmp_path / "train.log",
            "val/test_score/math:0.5\nval/test_score/gsm8k:0.7\n",
        )
        result = write_cell_metrics_from_verl(
            tmp_path, success_rate_key="val/test_score/",
            model_key="m", env="e", baseline="b",
        )
        assert result == {"status": "failed"}
        assert "success_rate" not in result

    def test_prefix_key_last_value_wins(self, tmp_path):
        """The LAST logged value of the single data_source key wins (final val)."""
        _write_log(
            tmp_path / "train.log",
            "val/test_score/math:0.30\nval/test_score/math:0.575\n",
        )
        result = write_cell_metrics_from_verl(
            tmp_path, success_rate_key="val/test_score/",
            model_key="m", env="e", baseline="b",
        )
        assert result["success_rate"] == 0.575

    def test_metric_name_reward_mean_not_success_rate(self, tmp_path):
        """metric_name='reward_mean' writes the value under reward_mean, NOT
        success_rate — a train reward is not a held-out rate metric."""
        _write_log(tmp_path / "train.log", "critic/rewards/mean:0.141\n")
        result = write_cell_metrics_from_verl(
            tmp_path, success_rate_key="critic/rewards/mean",
            metric_name="reward_mean", model_key="m", env="e", baseline="b",
        )
        assert result["status"] == "success"
        assert result["reward_mean"] == 0.141
        assert "success_rate" not in result

    def test_inline_provenance_written_into_metrics_json(self, tmp_path):
        """Provenance is embedded inline in metrics.json (GKE round-trip safe)."""
        import json as _json
        _write_log(tmp_path / "train.log", "val/success_rate:0.456\n")
        write_cell_metrics_from_verl(tmp_path, model_key="m", env="e", baseline="b")
        on_disk = _json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert on_disk["eval_provenance"]["metric_value"] == 0.456
        assert on_disk["eval_provenance"]["metric_name"] == "success_rate"
        assert on_disk["eval_provenance"]["provenance_kind"] == "aggregate"

    def test_exact_key_still_rejects_per_dataset_subkey(self, tmp_path):
        """Regression: the exact (non-prefix) val/success_rate still wins over a
        per-dataset sub-key — prefix mode is only for keys ending in '/'."""
        _write_log(
            tmp_path / "train.log",
            "val/success_rate/nq:0.418\nval/success_rate:0.456\n",
        )
        result = write_cell_metrics_from_verl(
            tmp_path, success_rate_key="val/success_rate",
            model_key="m", env="e", baseline="b",
        )
        assert result["success_rate"] == 0.456

    def test_prefix_json_conflicting_values_across_files_falls_through(self, tmp_path):
        """Two val/summary JSONs carrying the SAME prefix key with DIFFERENT
        values → ambiguous → no value (never silently pick the first sorted
        file's stale value; JSON files have no temporal 'last wins' order)."""
        import json as _json
        (tmp_path / "a_val.json").write_text(
            _json.dumps({"val/test_score/math": 0.1}), encoding="utf-8"
        )
        (tmp_path / "z_val.json").write_text(
            _json.dumps({"val/test_score/math": 0.9}), encoding="utf-8"
        )
        result = write_cell_metrics_from_verl(
            tmp_path, success_rate_key="val/test_score/",
            model_key="m", env="e", baseline="b",
        )
        assert result == {"status": "failed"}
        assert "success_rate" not in result

    def test_prefix_json_single_consistent_key_resolves(self, tmp_path):
        """One distinct prefix key with a consistent value across files → used."""
        import json as _json
        (tmp_path / "summary.json").write_text(
            _json.dumps({"val/test_score/math": 0.575}), encoding="utf-8"
        )
        result = write_cell_metrics_from_verl(
            tmp_path, success_rate_key="val/test_score/",
            model_key="m", env="e", baseline="b",
        )
        assert result["status"] == "success"
        assert result["success_rate"] == 0.575
