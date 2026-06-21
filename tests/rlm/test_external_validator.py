"""Tests for backend.agents.rlm.external_validator (Task P2.2).

All tests are hermetic — no network calls, no real LLM calls.
``grader_transport.sample_completions`` is monkeypatched to return canned JSON.
pytest-socket blocks non-loopback anyway.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from backend.agents.rlm.external_validator import (
    PredicateVerdict,
    ValidatorVerdict,
    check_gpu_claim_plausible,
    check_not_all_constant,
    check_provenance_present,
    check_rerun_agrees,
    external_validator_enabled,
    load_verdict,
    persist_verdict,
    run_validation_panel,
    validator_panel_n,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# SDAR v6 all-zero flat metrics (the canonical hallucination fixture).
V6_ALL_ZERO = {
    "loss": 0.0,
    "l_grpo": 0.0,
    "mean_reward": 0.0,
    "accuracy_avg": 0.0,
    "f1_avg": 0.0,
    "teacher_gap_mean": 0.0,
    "gate_activation_ratio": 0.0,
}

# Legitimate mixed metrics
LEGIT_MIXED = {"loss": 1.2, "return": 31.1, "accuracy": 0.74}

# Nested aggregated metrics (per_model shape)
NESTED_REAL = {
    "status": "ok",
    "per_model": {
        "qwen2.5-3b": {
            "alfworld": {
                "grpo": {"metric": 0.086, "reward_mean": 0.086, "status": "ok"}
            }
        }
    },
}


# ---------------------------------------------------------------------------
# Feature flag tests
# ---------------------------------------------------------------------------


def test_external_validator_disabled_by_default():
    assert external_validator_enabled() is False


def test_external_validator_enabled_when_set(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXTERNAL_VALIDATOR", "1")
    assert external_validator_enabled() is True


def test_validator_panel_n_default():
    assert validator_panel_n() == 2


def test_validator_panel_n_custom(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_PANEL_N", "4")
    assert validator_panel_n() == 4


def test_validator_panel_n_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_PANEL_N", "notanint")
    assert validator_panel_n() == 2


def test_validator_panel_n_min_1(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_PANEL_N", "0")
    assert validator_panel_n() >= 1


# ---------------------------------------------------------------------------
# check_provenance_present
# ---------------------------------------------------------------------------


def test_check_provenance_present_absent(tmp_path):
    """Without a provenance.json under code/, returns False."""
    (tmp_path / "code").mkdir()
    assert check_provenance_present({}, tmp_path) is False


def test_check_provenance_present_present(tmp_path):
    """With a provenance.json under code/, returns True."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "provenance.json").write_text('{"runs": []}')
    assert check_provenance_present({}, tmp_path) is True


def test_check_provenance_present_nested(tmp_path):
    """provenance.json nested inside a sub-dir is still found."""
    code_dir = tmp_path / "code" / "outputs" / "run_001"
    code_dir.mkdir(parents=True)
    (code_dir / "provenance.json").write_text("{}")
    assert check_provenance_present({}, tmp_path) is True


def test_check_provenance_present_no_code_dir(tmp_path):
    """Missing code/ dir → False (fail-soft)."""
    assert check_provenance_present({}, tmp_path) is False


# ---------------------------------------------------------------------------
# check_not_all_constant
# ---------------------------------------------------------------------------


def test_check_not_all_constant_v6_all_zero():
    """v6 all-zero metrics → NOT healthy → False."""
    # Directly pass simple flat dict so zero_metrics_detection is not required
    # (the lazy import path still works for all-zero flat dicts).
    result = check_not_all_constant({"loss": 0.0, "reward": 0.0})
    assert result is False


def test_check_not_all_constant_legit():
    """Mixed-value metrics → healthy → True."""
    result = check_not_all_constant({"loss": 1.2, "reward": 0.3})
    assert result is True


def test_check_not_all_constant_empty_dict():
    """Empty dict → nothing to check → True (conservative)."""
    result = check_not_all_constant({})
    assert result is True


def test_check_not_all_constant_constant_across():
    """All identical non-zero values (constant) → False."""
    result = check_not_all_constant({"loss": 0.5, "reward": 0.5, "acc": 0.5})
    assert result is False


def test_check_not_all_constant_single_nonzero_is_healthy():
    """Single non-zero value is a normal partial result → healthy (True).

    Regression: the constant branch requires >= 2 values. A lone value is NOT
    "constant across cells", so it must not be flagged; only the all-zero branch
    fires on a single value.
    """
    assert check_not_all_constant({"loss": 0.5}) is True


def test_check_not_all_constant_single_zero_is_unhealthy():
    """A single 0.0 still trips the all-zero branch (not healthy)."""
    assert check_not_all_constant({"loss": 0.0}) is False


# ---------------------------------------------------------------------------
# check_gpu_claim_plausible
# ---------------------------------------------------------------------------


def test_check_gpu_claim_plausible_no_gpu_claim():
    """Metrics without a GPU claim → plausible (True)."""
    metrics = {"loss": 1.0, "accuracy": 0.5}
    assert check_gpu_claim_plausible(metrics, {}) is True


def test_check_gpu_claim_plausible_gpu_with_time():
    """GPU claim + wall_time_s > 0 → plausible."""
    metrics = {
        "device": "cuda",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "wall_time_s": 120.0,
        "loss": 0.0,
    }
    assert check_gpu_claim_plausible(metrics, {}) is True


def test_check_gpu_claim_plausible_gpu_no_corroboration():
    """GPU claim without wall_time or steps → NOT plausible."""
    metrics = {
        "device": "cuda",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "loss": 0.0,
    }
    assert check_gpu_claim_plausible(metrics, {}) is False


def test_check_gpu_claim_plausible_evidence_context():
    """GPU claim without time in metrics but time in evidence dict → plausible."""
    metrics = {"device": "cuda", "model_id": "Qwen/Qwen2.5-1.7B"}
    evidence = {"wall_time_s": 300.0}
    assert check_gpu_claim_plausible(metrics, evidence) is True


# ---------------------------------------------------------------------------
# check_rerun_agrees (P2 stub)
# ---------------------------------------------------------------------------


def test_check_rerun_agrees_returns_none():
    """P2 stub always returns None (skipped)."""
    result = check_rerun_agrees()
    assert result is None


def test_check_rerun_agrees_any_args():
    result = check_rerun_agrees(metrics={}, metric_ref="loss", anything=True)
    assert result is None


# ---------------------------------------------------------------------------
# run_validation_panel — None client → unavailable
# ---------------------------------------------------------------------------


def test_run_validation_panel_none_client(tmp_path):
    """validator_client=None → status=='unavailable'."""
    verdict = run_validation_panel(
        validator_client=None,
        panel_models=[],
        metrics=V6_ALL_ZERO,
        project_dir=tmp_path,
        leaf_records=[],
        separation="unavailable",
    )
    assert verdict.status == "unavailable"
    assert verdict.veto_set == []
    assert verdict.predicates == []


# ---------------------------------------------------------------------------
# run_validation_panel — stubbed panel, provenance suspicion fires
# ---------------------------------------------------------------------------


def test_run_validation_panel_provenance_suspicion_vetoed(tmp_path, monkeypatch):
    """Panel flags provenance_present on all-zero, no-provenance fixture → vetoed."""
    # No provenance.json exists in tmp_path/code
    (tmp_path / "code").mkdir()

    # Stub sample_completions to return a suspicion list
    canned_response = json.dumps([
        {"predicate": "provenance_present", "metric_ref": "mean_reward"}
    ])

    def _fake_sample_completions(client, *, system, user, n, **kwargs):
        return [canned_response] * n

    monkeypatch.setattr(
        "backend.agents.rlm.grader_transport.sample_completions",
        _fake_sample_completions,
    )

    fake_client = MagicMock()
    verdict = run_validation_panel(
        validator_client=fake_client,
        panel_models=["gpt-4o"],
        metrics=V6_ALL_ZERO,
        project_dir=tmp_path,
        leaf_records=[],
        separation="independent",
    )

    assert verdict.status == "vetoed", f"Expected vetoed, got: {verdict.status}"
    assert "mean_reward" in verdict.veto_set
    # Check that machine-check confirmed the violation
    pv = verdict.predicates[0]
    assert pv.predicate == "provenance_present"
    assert pv.violated is True


def test_run_validation_panel_provenance_suspicion_clean_when_present(tmp_path, monkeypatch):
    """Same fixture WITH provenance.json present → machine-check finds it → clean."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "provenance.json").write_text('{"runs": ["run_001"]}')

    canned_response = json.dumps([
        {"predicate": "provenance_present", "metric_ref": "mean_reward"}
    ])

    def _fake_sample_completions(client, *, system, user, n, **kwargs):
        return [canned_response] * n

    monkeypatch.setattr(
        "backend.agents.rlm.grader_transport.sample_completions",
        _fake_sample_completions,
    )

    fake_client = MagicMock()
    verdict = run_validation_panel(
        validator_client=fake_client,
        panel_models=["gpt-4o"],
        metrics=V6_ALL_ZERO,
        project_dir=tmp_path,
        leaf_records=[],
        separation="independent",
    )

    assert verdict.status == "clean", f"Expected clean, got: {verdict.status}"
    assert verdict.veto_set == []
    pv = verdict.predicates[0]
    assert pv.predicate == "provenance_present"
    assert pv.violated is False


# ---------------------------------------------------------------------------
# run_validation_panel — not_all_constant suspicion
# ---------------------------------------------------------------------------


def test_run_validation_panel_constant_metrics_vetoed(tmp_path, monkeypatch):
    """Panel flags not_all_constant on all-zero metrics → vetoed."""
    (tmp_path / "code").mkdir()

    canned_response = json.dumps([
        {"predicate": "not_all_constant", "metric_ref": "loss"}
    ])

    def _fake_sample_completions(client, *, system, user, n, **kwargs):
        return [canned_response] * n

    monkeypatch.setattr(
        "backend.agents.rlm.grader_transport.sample_completions",
        _fake_sample_completions,
    )

    fake_client = MagicMock()
    verdict = run_validation_panel(
        validator_client=fake_client,
        panel_models=["test-model"],
        metrics={"loss": 0.0, "reward": 0.0},
        project_dir=tmp_path,
        leaf_records=[],
        separation="weak",
    )

    assert verdict.status == "vetoed"
    assert "loss" in verdict.veto_set


def test_run_validation_panel_constant_metrics_clean_for_real_data(tmp_path, monkeypatch):
    """Same predicate on legit metrics → machine-check passes → clean."""
    (tmp_path / "code").mkdir()

    canned_response = json.dumps([
        {"predicate": "not_all_constant", "metric_ref": "loss"}
    ])

    def _fake_sample_completions(client, *, system, user, n, **kwargs):
        return [canned_response] * n

    monkeypatch.setattr(
        "backend.agents.rlm.grader_transport.sample_completions",
        _fake_sample_completions,
    )

    fake_client = MagicMock()
    verdict = run_validation_panel(
        validator_client=fake_client,
        panel_models=["test-model"],
        metrics={"loss": 1.2, "reward": 0.3},
        project_dir=tmp_path,
        leaf_records=[],
        separation="independent",
    )

    assert verdict.status == "clean"
    assert verdict.veto_set == []


# ---------------------------------------------------------------------------
# Min-aggregation: one panelist violated + one clean → vetoed
# ---------------------------------------------------------------------------


def test_run_validation_panel_min_aggregation(tmp_path, monkeypatch):
    """One panelist flags a violation, the other finds nothing → min-aggregation → vetoed."""
    (tmp_path / "code").mkdir()

    # Panelist 1: flags provenance_present violation
    # Panelist 2: reports empty (no suspicion)
    responses = [
        json.dumps([{"predicate": "provenance_present", "metric_ref": "accuracy_avg"}]),
        json.dumps([]),
    ]
    call_idx = [0]

    def _fake_sample_completions(client, *, system, user, n, **kwargs):
        return responses[:n] if n <= len(responses) else responses + [responses[-1]] * (n - len(responses))

    monkeypatch.setattr(
        "backend.agents.rlm.grader_transport.sample_completions",
        _fake_sample_completions,
    )
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_PANEL_N", "2")

    fake_client = MagicMock()
    verdict = run_validation_panel(
        validator_client=fake_client,
        panel_models=["model-a", "model-b"],
        metrics=V6_ALL_ZERO,  # no provenance.json → check will confirm violated
        project_dir=tmp_path,
        leaf_records=[],
        separation="weak",
    )

    # Even though one panelist was clean, the one violated suspicion → vetoed
    assert verdict.status == "vetoed"
    assert "accuracy_avg" in verdict.veto_set


def test_run_validation_panel_no_suspicions_clean(tmp_path, monkeypatch):
    """Panel returns empty suspicion list → no predicates to check → clean."""
    def _fake_sample_completions(client, *, system, user, n, **kwargs):
        return [json.dumps([])] * n

    monkeypatch.setattr(
        "backend.agents.rlm.grader_transport.sample_completions",
        _fake_sample_completions,
    )

    fake_client = MagicMock()
    verdict = run_validation_panel(
        validator_client=fake_client,
        panel_models=["test-model"],
        metrics=LEGIT_MIXED,
        project_dir=tmp_path,
        leaf_records=[],
        separation="independent",
    )
    assert verdict.status == "clean"
    assert verdict.veto_set == []


# ---------------------------------------------------------------------------
# Invalid predicates are silently dropped
# ---------------------------------------------------------------------------


def test_run_validation_panel_unknown_predicate_ignored(tmp_path, monkeypatch):
    """An LLM returning an unknown predicate name → silently ignored (not vetoed)."""
    canned = json.dumps([
        {"predicate": "made_up_predicate", "metric_ref": "loss"}
    ])

    def _fake_sample_completions(client, *, system, user, n, **kwargs):
        return [canned] * n

    monkeypatch.setattr(
        "backend.agents.rlm.grader_transport.sample_completions",
        _fake_sample_completions,
    )

    fake_client = MagicMock()
    verdict = run_validation_panel(
        validator_client=fake_client,
        panel_models=["test-model"],
        metrics=LEGIT_MIXED,
        project_dir=tmp_path,
        leaf_records=[],
        separation="independent",
    )
    assert verdict.status == "clean"


# ---------------------------------------------------------------------------
# Panel call failure → unavailable
# ---------------------------------------------------------------------------


def test_run_validation_panel_call_failure_returns_unavailable(tmp_path, monkeypatch):
    """If sample_completions raises, the panel returns 'unavailable' (fail-soft)."""
    def _bad_sample_completions(client, *, system, user, n, **kwargs):
        raise RuntimeError("network error")

    monkeypatch.setattr(
        "backend.agents.rlm.grader_transport.sample_completions",
        _bad_sample_completions,
    )

    fake_client = MagicMock()
    verdict = run_validation_panel(
        validator_client=fake_client,
        panel_models=["test-model"],
        metrics=V6_ALL_ZERO,
        project_dir=tmp_path,
        leaf_records=[],
        separation="independent",
    )
    assert verdict.status == "unavailable"


# ---------------------------------------------------------------------------
# ValidatorVerdict fields
# ---------------------------------------------------------------------------


def test_run_validation_panel_carries_separation_and_models(tmp_path, monkeypatch):
    """The returned verdict echoes back the separation string and panel_models."""
    def _fake_sample_completions(client, *, system, user, n, **kwargs):
        return [json.dumps([])] * n

    monkeypatch.setattr(
        "backend.agents.rlm.grader_transport.sample_completions",
        _fake_sample_completions,
    )

    fake_client = MagicMock()
    verdict = run_validation_panel(
        validator_client=fake_client,
        panel_models=["oauth-sonnet", "azure-gpt4o"],
        metrics=LEGIT_MIXED,
        project_dir=tmp_path,
        leaf_records=[],
        separation="independent",
    )
    assert verdict.separation == "independent"
    assert verdict.panel_models == ["oauth-sonnet", "azure-gpt4o"]
    assert verdict.evidence_fingerprint != ""


# ---------------------------------------------------------------------------
# persist_verdict / load_verdict round-trip
# ---------------------------------------------------------------------------


def test_persist_and_load_verdict_round_trip(tmp_path):
    """persist_verdict + load_verdict restore the verdict identically."""
    verdict = ValidatorVerdict(
        status="vetoed",
        veto_set=["mean_reward"],
        predicates=[
            PredicateVerdict(
                predicate="provenance_present",
                metric_ref="mean_reward",
                violated=True,
                detail="no provenance.json",
            )
        ],
        panel_models=["gpt-4o"],
        separation="independent",
        evidence_fingerprint="abc123",
    )
    persist_verdict(tmp_path, verdict)
    loaded = load_verdict(tmp_path)
    assert loaded is not None
    assert loaded.status == "vetoed"
    assert loaded.veto_set == ["mean_reward"]
    assert loaded.separation == "independent"
    assert loaded.evidence_fingerprint == "abc123"
    assert len(loaded.predicates) == 1
    pv = loaded.predicates[0]
    assert pv.predicate == "provenance_present"
    assert pv.violated is True
    assert pv.metric_ref == "mean_reward"


def test_load_verdict_returns_none_when_absent(tmp_path):
    """load_verdict on a fresh project_dir returns None."""
    assert load_verdict(tmp_path) is None


def test_load_verdict_stale_fingerprint_ignored(tmp_path):
    """load_verdict(expect_fingerprint=X) returns None when stored fp != X."""
    verdict = ValidatorVerdict(
        status="clean",
        veto_set=[],
        predicates=[],
        panel_models=["gpt-4o"],
        separation="independent",
        evidence_fingerprint="stored_fp_value",
    )
    persist_verdict(tmp_path, verdict)
    # A different fingerprint → stale, ignored
    result = load_verdict(tmp_path, expect_fingerprint="different_fp")
    assert result is None


def test_load_verdict_matching_fingerprint_returned(tmp_path):
    """load_verdict(expect_fingerprint=X) returns the verdict when fp matches."""
    fp = "matching_fp_value"
    verdict = ValidatorVerdict(
        status="clean",
        veto_set=[],
        predicates=[],
        panel_models=["test"],
        separation="weak",
        evidence_fingerprint=fp,
    )
    persist_verdict(tmp_path, verdict)
    result = load_verdict(tmp_path, expect_fingerprint=fp)
    assert result is not None
    assert result.evidence_fingerprint == fp


def test_load_verdict_no_expect_fingerprint_always_returns(tmp_path):
    """load_verdict(expect_fingerprint=None) always returns the stored verdict."""
    verdict = ValidatorVerdict(
        status="clean",
        veto_set=[],
        predicates=[],
        panel_models=["test"],
        separation="degraded",
        evidence_fingerprint="anything",
    )
    persist_verdict(tmp_path, verdict)
    result = load_verdict(tmp_path, expect_fingerprint=None)
    assert result is not None
    assert result.evidence_fingerprint == "anything"


def test_persist_verdict_atomic_creates_dir(tmp_path):
    """persist_verdict creates rlm_state/ if missing (no pre-creation needed)."""
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    verdict = ValidatorVerdict(
        status="unavailable",
        veto_set=[],
        predicates=[],
        panel_models=[],
        separation="unavailable",
        evidence_fingerprint="",
    )
    persist_verdict(project_dir, verdict)
    target = project_dir / "rlm_state" / "validation_verdict.json"
    assert target.exists()


def test_persist_verdict_file_is_valid_json(tmp_path):
    """The persisted file is parseable JSON."""
    verdict = ValidatorVerdict(
        status="vetoed",
        veto_set=["loss"],
        predicates=[
            PredicateVerdict(
                predicate="not_all_constant",
                metric_ref="loss",
                violated=True,
                detail="all zero",
            )
        ],
        panel_models=["model-x"],
        separation="independent",
        evidence_fingerprint="fp_xyz",
    )
    persist_verdict(tmp_path, verdict)
    target = tmp_path / "rlm_state" / "validation_verdict.json"
    data = json.loads(target.read_text())
    assert data["status"] == "vetoed"
    assert data["veto_set"] == ["loss"]
    assert data["evidence_fingerprint"] == "fp_xyz"


# ---------------------------------------------------------------------------
# Evidence fingerprint stability
# ---------------------------------------------------------------------------


def test_evidence_fingerprint_deterministic(tmp_path, monkeypatch):
    """Same metrics → same fingerprint across two panel calls."""
    def _fake_sample_completions(client, *, system, user, n, **kwargs):
        return [json.dumps([])] * n

    monkeypatch.setattr(
        "backend.agents.rlm.grader_transport.sample_completions",
        _fake_sample_completions,
    )

    fake_client = MagicMock()
    kwargs = dict(
        validator_client=fake_client,
        panel_models=["m"],
        metrics=V6_ALL_ZERO,
        project_dir=tmp_path,
        leaf_records=[],
        separation="independent",
    )
    v1 = run_validation_panel(**kwargs)
    v2 = run_validation_panel(**kwargs)
    assert v1.evidence_fingerprint == v2.evidence_fingerprint


def test_evidence_fingerprint_changes_with_metrics(tmp_path, monkeypatch):
    """Different metrics → different fingerprints."""
    def _fake_sample_completions(client, *, system, user, n, **kwargs):
        return [json.dumps([])] * n

    monkeypatch.setattr(
        "backend.agents.rlm.grader_transport.sample_completions",
        _fake_sample_completions,
    )

    fake_client = MagicMock()
    base = dict(
        validator_client=fake_client,
        panel_models=["m"],
        project_dir=tmp_path,
        leaf_records=[],
        separation="independent",
    )
    v1 = run_validation_panel(**base, metrics=V6_ALL_ZERO)
    v2 = run_validation_panel(**base, metrics=LEGIT_MIXED)
    assert v1.evidence_fingerprint != v2.evidence_fingerprint


# ---------------------------------------------------------------------------
# Default-OFF contract: with flag unset, panel is inert (unavailable)
# ---------------------------------------------------------------------------


def test_flag_off_panel_unavailable_with_none_client(tmp_path):
    """Flag OFF + client=None → status==unavailable (flag not needed to reach this state)."""
    assert not external_validator_enabled()
    verdict = run_validation_panel(
        validator_client=None,
        panel_models=[],
        metrics=V6_ALL_ZERO,
        project_dir=tmp_path,
        leaf_records=[],
        separation="unavailable",
    )
    assert verdict.status == "unavailable"


# ---------------------------------------------------------------------------
# Malformed LLM responses are handled gracefully
# ---------------------------------------------------------------------------


def test_malformed_json_from_panel_handled(tmp_path, monkeypatch):
    """Completely unparseable LLM response → no predicate verdicts, clean (fail-soft)."""
    def _fake_sample_completions(client, *, system, user, n, **kwargs):
        return ["this is not json at all"] * n

    monkeypatch.setattr(
        "backend.agents.rlm.grader_transport.sample_completions",
        _fake_sample_completions,
    )

    fake_client = MagicMock()
    verdict = run_validation_panel(
        validator_client=fake_client,
        panel_models=["m"],
        metrics=LEGIT_MIXED,
        project_dir=tmp_path,
        leaf_records=[],
        separation="independent",
    )
    assert verdict.status == "clean"
    assert verdict.predicates == []


def test_partial_json_in_prose_handled(tmp_path, monkeypatch):
    """JSON array embedded in prose is extracted."""
    response = 'The metrics look suspicious. Here are my findings: [{"predicate": "not_all_constant", "metric_ref": "reward"}] That is all.'

    def _fake_sample_completions(client, *, system, user, n, **kwargs):
        return [response] * n

    monkeypatch.setattr(
        "backend.agents.rlm.grader_transport.sample_completions",
        _fake_sample_completions,
    )

    fake_client = MagicMock()
    verdict = run_validation_panel(
        validator_client=fake_client,
        panel_models=["m"],
        metrics={"loss": 0.0, "reward": 0.0},  # all-zero → violation confirmed
        project_dir=tmp_path,
        leaf_records=[],
        separation="independent",
    )
    assert verdict.status == "vetoed"
    assert "reward" in verdict.veto_set
