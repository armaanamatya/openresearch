"""
Unit tests for G2 stub-metrics guard (backend/agents/rlm/stub_detection.py).
Pure unit tests — no network, no filesystem, no subprocess.
"""



from backend.agents.rlm.stub_detection import (
    looks_like_stub_metrics,
    stub_metrics_guard_enabled,
    stub_repair_message,
)


# ---------------------------------------------------------------------------
# looks_like_stub_metrics — core detector
# ---------------------------------------------------------------------------

class TestLooksLikeStubMetrics:
    def test_pure_placeholder_keys_fires(self):
        assert looks_like_stub_metrics({"total_length": 1234, "chunk_count": 5}) is True

    def test_real_metric_only_does_not_fire(self):
        assert looks_like_stub_metrics({"accuracy": 0.91}) is False

    def test_real_metric_present_with_placeholder_does_not_fire(self):
        # success_rate is a real metric hint ("rate" / "success") → guard stays silent
        assert looks_like_stub_metrics({"success_rate": 0.4, "total_length": 10}) is False

    def test_empty_dict_does_not_fire(self):
        assert looks_like_stub_metrics({}) is False

    def test_none_does_not_fire(self):
        assert looks_like_stub_metrics(None) is False

    def test_non_dict_string_does_not_fire(self):
        assert looks_like_stub_metrics("notadict") is False

    def test_unknown_non_placeholder_keys_conservative(self):
        # Unknown keys that are neither placeholder nor real-metric → False
        assert looks_like_stub_metrics({"weird_custom_key": 1}) is False

    def test_nested_placeholder_fires(self):
        assert looks_like_stub_metrics({"per_model": {"qwen": {"chunk_count": 3}}}) is True

    def test_nested_real_metric_does_not_fire(self):
        assert looks_like_stub_metrics({"per_model": {"qwen": {"reward": 0.5}}}) is False

    def test_loss_hint_does_not_fire(self):
        assert looks_like_stub_metrics({"train_loss": 0.25, "chunk_count": 5}) is False

    def test_placeholder_key_alone_fires(self):
        assert looks_like_stub_metrics({"placeholder_metric": 0.0}) is True

    def test_dummy_key_alone_fires(self):
        assert looks_like_stub_metrics({"dummy": 42}) is True

    def test_list_values_nested_placeholder(self):
        # List containing dicts — recurse into dict elements
        assert looks_like_stub_metrics({"results": [{"chunk_count": 1}]}) is True

    def test_list_values_nested_real_metric(self):
        assert looks_like_stub_metrics({"results": [{"accuracy": 0.9}]}) is False

    def test_fail_soft_on_unhashable(self):
        # Should not raise even on weird inputs
        assert looks_like_stub_metrics({"a": {"b": {"c": []}}}) is False

    def test_count_key_fires(self):
        assert looks_like_stub_metrics({"count": 100}) is True

    def test_n_examples_key_fires(self):
        assert looks_like_stub_metrics({"n_examples": 50}) is True

    def test_f1_real_metric(self):
        assert looks_like_stub_metrics({"f1_score": 0.88}) is False

    def test_bleu_real_metric(self):
        assert looks_like_stub_metrics({"bleu": 22.3}) is False

    def test_ppl_real_metric(self):
        assert looks_like_stub_metrics({"ppl": 14.5}) is False

    def test_pass_at_k_real_metric(self):
        assert looks_like_stub_metrics({"pass@1": 0.72}) is False

    def test_throughput_real_metric(self):
        assert looks_like_stub_metrics({"throughput": 120.0}) is False


# ---------------------------------------------------------------------------
# stub_repair_message
# ---------------------------------------------------------------------------

class TestStubRepairMessage:
    def test_non_empty_string(self):
        msg = stub_repair_message({"total_length": 1})
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_mentions_stub(self):
        msg = stub_repair_message({"total_length": 1})
        assert "stub" in msg.lower()

    def test_mentions_reimplement_or_metric(self):
        msg = stub_repair_message({"chunk_count": 5})
        assert "re-implement" in msg.lower() or "metric" in msg.lower()

    def test_names_offending_keys(self):
        msg = stub_repair_message({"total_length": 1, "chunk_count": 5})
        assert "total_length" in msg or "chunk_count" in msg

    def test_capped_at_six_keys(self):
        many_keys = {k: i for i, k in enumerate([
            "total_length", "chunk_count", "placeholder_metric",
            "dummy", "todo", "n_chunks", "num_chunks",
        ])}
        msg = stub_repair_message(many_keys)
        # Should not crash and should be a non-empty string
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_none_input_does_not_raise(self):
        msg = stub_repair_message(None)
        assert isinstance(msg, str)

    def test_empty_dict_does_not_raise(self):
        msg = stub_repair_message({})
        assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# stub_metrics_guard_enabled — env-var flag
# ---------------------------------------------------------------------------

class TestStubMetricsGuardEnabled:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_STUB_METRICS_GUARD", raising=False)
        assert stub_metrics_guard_enabled() is False

    def test_enabled_by_1(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_STUB_METRICS_GUARD", "1")
        assert stub_metrics_guard_enabled() is True

    def test_enabled_by_true(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_STUB_METRICS_GUARD", "true")
        assert stub_metrics_guard_enabled() is True

    def test_enabled_by_yes(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_STUB_METRICS_GUARD", "yes")
        assert stub_metrics_guard_enabled() is True

    def test_enabled_by_on(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_STUB_METRICS_GUARD", "on")
        assert stub_metrics_guard_enabled() is True

    def test_disabled_by_0(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_STUB_METRICS_GUARD", "0")
        assert stub_metrics_guard_enabled() is False

    def test_disabled_by_false(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_STUB_METRICS_GUARD", "false")
        assert stub_metrics_guard_enabled() is False

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_STUB_METRICS_GUARD", "  TRUE  ")
        assert stub_metrics_guard_enabled() is True
