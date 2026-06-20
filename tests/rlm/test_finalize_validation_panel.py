"""Tests for _run_finalize_validation_panel — the shared finalize validator panel."""
from types import SimpleNamespace


def _report():
    return SimpleNamespace(baseline_metrics={}, reproduction_summary="", reported_metrics=None)


def test_panel_noop_when_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENRESEARCH_EXTERNAL_VALIDATOR", raising=False)
    from backend.agents.rlm.run import _run_finalize_validation_panel
    ctx = SimpleNamespace(validator_client=object(), role_selection=None)
    _run_finalize_validation_panel(ctx, _report(), tmp_path)  # must not raise
    assert not (tmp_path / "rlm_state" / "validation_verdict.json").exists()


def test_panel_noop_when_no_client(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EXTERNAL_VALIDATOR", "1")
    from backend.agents.rlm.run import _run_finalize_validation_panel
    ctx = SimpleNamespace(validator_client=None, role_selection=None)
    _run_finalize_validation_panel(ctx, _report(), tmp_path)  # no client -> no-op
    assert not (tmp_path / "rlm_state" / "validation_verdict.json").exists()


def test_panel_runs_when_enabled_with_client(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EXTERNAL_VALIDATOR", "1")
    from backend.agents.rlm.run import _run_finalize_validation_panel
    panel_calls, persist_calls = [], []
    fake_verdict = SimpleNamespace(status="clean", veto_set=[], separation="independent")
    monkeypatch.setattr(
        "backend.agents.rlm.external_validator.run_validation_panel",
        lambda **k: panel_calls.append(k) or fake_verdict,
    )
    monkeypatch.setattr(
        "backend.agents.rlm.external_validator.persist_verdict",
        lambda pd, v: persist_calls.append((pd, v)),
    )
    monkeypatch.setattr(
        "backend.agents.rlm.external_validator.load_verdict",
        lambda pd, expect_fingerprint=None: None,  # not already validated
    )
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(
        '{"per_model": {"m": {"e": {"b": {"status": "ok", "accuracy": 0.8}}}}}'
    )
    ctx = SimpleNamespace(validator_client=object(), role_selection=None)
    _run_finalize_validation_panel(ctx, _report(), tmp_path)
    assert panel_calls, "panel should have run"
    assert persist_calls, "verdict should have been persisted"


def test_panel_reuses_existing_verdict(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EXTERNAL_VALIDATOR", "1")
    from backend.agents.rlm.run import _run_finalize_validation_panel
    panel_calls = []
    monkeypatch.setattr(
        "backend.agents.rlm.external_validator.run_validation_panel",
        lambda **k: panel_calls.append(k),
    )
    monkeypatch.setattr(
        "backend.agents.rlm.external_validator.load_verdict",
        lambda pd, expect_fingerprint=None: SimpleNamespace(status="clean"),  # already validated
    )
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text('{"per_model": {}}')
    ctx = SimpleNamespace(validator_client=object(), role_selection=None)
    _run_finalize_validation_panel(ctx, _report(), tmp_path)
    assert not panel_calls, "should reuse the persisted verdict, not re-run the panel"


def test_panel_failsoft_on_panel_error(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EXTERNAL_VALIDATOR", "1")
    from backend.agents.rlm.run import _run_finalize_validation_panel

    def boom(**k):
        raise RuntimeError("panel exploded")

    monkeypatch.setattr("backend.agents.rlm.external_validator.run_validation_panel", boom)
    monkeypatch.setattr(
        "backend.agents.rlm.external_validator.load_verdict",
        lambda pd, expect_fingerprint=None: None,
    )
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text('{"per_model": {}}')
    ctx = SimpleNamespace(validator_client=object(), role_selection=None)
    _run_finalize_validation_panel(ctx, _report(), tmp_path)  # must not raise
