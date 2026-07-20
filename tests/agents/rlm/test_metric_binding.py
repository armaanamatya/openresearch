"""Tests for metric_binding.bind_claims (Track A, closes gap D2).

Pure deterministic tests — no LLM, no network, no filesystem. See
docs/history/specs/2026-07-09-eval-integrity-track-a-design.md §4.1 for the
design this locks down: a candidate bind is accepted only when the resolved
scope keys deterministically match the claim's declared scope; anything
ambiguous or mismatched stays unbound rather than risk a wrong number.
"""
from __future__ import annotations

from backend.agents.rlm.metric_binding import bind_claims

_METRICS = {"per_model": {"adam": {"mnist": {"test": {"accuracy": 0.991}}}},
            "accuracy": 0.991}


def test_unambiguous_scope_match_binds_path():
    spec = {"claims": [{"claim_id": "primary_0", "metric_name": "accuracy",
                        "direction": "higher_is_better",
                        "scope": {"model": "adam", "dataset": "mnist", "split": "test"}}]}
    out = bind_claims(spec, _METRICS)
    b = out["claims"][0]["metric_binding"]
    assert b["bound"] is True
    assert b["metric_key"] == "accuracy" and b["model_key"] == "adam" and b["env_key"] == "mnist"


def test_scope_mismatch_does_not_bind():
    # path 'accuracy' exists but the claim's split is 'train' — must NOT bind to the test-split number
    spec = {"claims": [{"claim_id": "primary_0", "metric_name": "accuracy",
                        "direction": "higher_is_better",
                        "scope": {"model": "adam", "dataset": "mnist", "split": "train"}}]}
    out = bind_claims(spec, _METRICS)
    assert out["claims"][0]["metric_binding"]["bound"] is False


def test_ambiguous_metric_name_does_not_bind():
    metrics = {"loss_a": 0.1, "loss_b": 0.2}
    spec = {"claims": [{"claim_id": "primary_0", "metric_name": "loss", "scope": {}}]}
    out = bind_claims(spec, metrics)
    assert out["claims"][0]["metric_binding"]["bound"] is False


def test_missing_metric_stays_unbound_never_raises():
    spec = {"claims": [{"claim_id": "primary_0", "metric_name": "perplexity", "scope": {}}]}
    out = bind_claims(spec, _METRICS)
    assert out["claims"][0]["metric_binding"]["bound"] is False
