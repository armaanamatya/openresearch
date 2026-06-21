"""Tests for backend.agents.rlm.evidence_audit (Task 1–4, TDD)."""
import dataclasses
import json
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Task 1: evidence_audit_enabled() + _provenance_on_disk()
# ---------------------------------------------------------------------------


def test_evidence_audit_disabled_by_default(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_AUDIT", raising=False)
    from backend.agents.rlm.evidence_audit import evidence_audit_enabled
    assert evidence_audit_enabled() is False


def test_evidence_audit_enabled_truthy(monkeypatch):
    from backend.agents.rlm.evidence_audit import evidence_audit_enabled
    for v in ("1", "true", "on", "yes", "TRUE", " On "):
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", v)
        assert evidence_audit_enabled() is True


def test_evidence_audit_disabled_falsey(monkeypatch):
    from backend.agents.rlm.evidence_audit import evidence_audit_enabled
    for v in ("0", "false", "off", "no", ""):
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", v)
        assert evidence_audit_enabled() is False


def test_provenance_on_disk_true_when_file_present(tmp_path):
    from backend.agents.rlm.evidence_audit import _provenance_on_disk
    code = tmp_path / "code"
    code.mkdir()
    (code / "provenance.json").write_text('{"schema_version": 1}')
    assert _provenance_on_disk(SimpleNamespace(project_dir=tmp_path)) is True


def test_provenance_on_disk_false_when_absent(tmp_path):
    from backend.agents.rlm.evidence_audit import _provenance_on_disk
    (tmp_path / "code").mkdir()
    assert _provenance_on_disk(SimpleNamespace(project_dir=tmp_path)) is False


def test_provenance_on_disk_failsoft_on_bad_ctx():
    from backend.agents.rlm.evidence_audit import _provenance_on_disk
    assert _provenance_on_disk(SimpleNamespace()) is False  # no project_dir -> False, no raise


# ---------------------------------------------------------------------------
# Task 2: EvidenceAudit dataclass + run_level_clean
# ---------------------------------------------------------------------------


def _audit(**kw):
    from backend.agents.rlm.evidence_audit import EvidenceAudit
    base = dict(backed_by_ledger=True, provenance_present=True,
                metrics_non_degenerate=True, metric_keys_real=True, fingerprint="fp")
    base.update(kw)
    return EvidenceAudit(**base)


def test_run_level_clean_all_true():
    assert _audit().run_level_clean is True


def test_run_level_clean_false_when_unbacked():
    assert _audit(backed_by_ledger=False).run_level_clean is False


def test_run_level_clean_false_when_degenerate():
    assert _audit(metrics_non_degenerate=False).run_level_clean is False


def test_run_level_clean_false_when_keys_not_real():
    assert _audit(metric_keys_real=False).run_level_clean is False


def test_run_level_clean_ignores_provenance():
    # provenance absent but everything else real -> still clean
    assert _audit(provenance_present=False).run_level_clean is True


def test_run_level_clean_false_when_rerun_disagrees():
    assert _audit(rerun_agrees=False).run_level_clean is False


def test_run_level_clean_true_when_rerun_none_or_agrees():
    assert _audit(rerun_agrees=None).run_level_clean is True
    assert _audit(rerun_agrees=True).run_level_clean is True


def test_evidence_audit_is_frozen():
    a = _audit()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.backed_by_ledger = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Task 3: result_is_fabricated()
# ---------------------------------------------------------------------------


def _ctx(tmp_path):
    return SimpleNamespace(project_dir=tmp_path)


def test_result_is_fabricated_none_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_AUDIT", raising=False)
    from backend.agents.rlm.evidence_audit import result_is_fabricated
    result = {"success": True, "metrics": {"per_model": {"m": {"e": {"b":
             {"status": "ok", "device": "cuda", "reward": 0.0, "success_rate": 0.0}}}}}}
    assert result_is_fabricated(result, _ctx(tmp_path)) is None  # dormant


def test_result_is_fabricated_none_when_not_success(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import result_is_fabricated
    assert result_is_fabricated({"success": False, "metrics": {}}, _ctx(tmp_path)) is None
    assert result_is_fabricated("not a dict", _ctx(tmp_path)) is None


def test_result_is_fabricated_vetoes_sdar_v6_zero_metrics(monkeypatch, tmp_path):
    # The SDAR-v6 hallucination: real metric keys, all 0.0, GPU claim, NO provenance.
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import result_is_fabricated
    result = {"success": True, "metrics": {"per_model": {"Qwen/Qwen3-1.7B": {"alfworld": {"sdar":
             {"status": "ok", "device": "cuda", "success_rate": 0.0, "reward": 0.0}}}}}}
    reason = result_is_fabricated(result, _ctx(tmp_path))
    assert reason is not None
    assert "zero" in reason.lower() or "constant" in reason.lower()


def test_result_is_fabricated_vetoes_stub(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import result_is_fabricated
    result = {"success": True, "metrics": {"total_length": 5, "chunk_count": 2}}
    assert result_is_fabricated(result, _ctx(tmp_path)) is not None


def test_result_is_fabricated_passes_clean(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import result_is_fabricated
    result = {"success": True, "metrics": {"per_model": {"m": {"e": {"b":
             {"status": "ok", "accuracy": 0.83}}}}}}
    assert result_is_fabricated(result, _ctx(tmp_path)) is None


def test_result_is_fabricated_legit_zero_with_provenance(monkeypatch, tmp_path):
    # all-zero + GPU claim BUT provenance.json present -> real 0 baseline, NOT vetoed.
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import result_is_fabricated
    code = tmp_path / "code"
    code.mkdir()
    (code / "provenance.json").write_text('{"schema_version": 1}')
    result = {"success": True, "metrics": {"per_model": {"m": {"e": {"b":
             {"status": "ok", "device": "cuda", "reward": 0.0, "success_rate": 0.0}}}}}}
    assert result_is_fabricated(result, _ctx(tmp_path)) is None


def test_result_is_fabricated_vetoes_low_vram(monkeypatch, tmp_path):
    # GPU claimed, non-zero metric, but peak VRAM below the 1.5 GiB floor.
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
    from backend.agents.rlm.evidence_audit import result_is_fabricated
    result = {"success": True, "metrics": {"per_model": {"m": {"e": {"b":
             {"status": "ok", "device": "cuda", "accuracy": 0.42}}}}}}
    reason = result_is_fabricated(result, _ctx(tmp_path), peak_vram_gb=0.2)
    assert reason is not None and "vram" in reason.lower()


# ---------------------------------------------------------------------------
# Task 4: audit_evidence()
# ---------------------------------------------------------------------------


def _ctx_dir(tmp_path, ledger_ok=None):
    ledger = None
    if ledger_ok is not None:
        ledger = SimpleNamespace(
            session_success_compatible_count=lambda agent_id: ledger_ok,
            session_call_count=lambda agent_id: max(ledger_ok, 1),
        )
    return SimpleNamespace(project_dir=tmp_path, cost_ledger=ledger)


def _write_metrics(tmp_path, metrics, provenance=False):
    code = tmp_path / "code"
    code.mkdir(exist_ok=True)
    (code / "metrics.json").write_text(json.dumps(metrics))
    if provenance:
        (code / "provenance.json").write_text('{"schema_version": 1}')


def test_audit_evidence_clean(tmp_path):
    from backend.agents.rlm.evidence_audit import audit_evidence
    _write_metrics(tmp_path, {"per_model": {"m": {"e": {"b":
                  {"status": "ok", "accuracy": 0.8}}}}}, provenance=True)
    a = audit_evidence(_ctx_dir(tmp_path, ledger_ok=1))
    assert a.backed_by_ledger and a.metrics_non_degenerate and a.metric_keys_real
    assert a.provenance_present and a.run_level_clean and a.fingerprint


def test_audit_evidence_zero_metrics_not_clean(tmp_path):
    from backend.agents.rlm.evidence_audit import audit_evidence
    _write_metrics(tmp_path, {"per_model": {"m": {"e": {"b":
                  {"status": "ok", "reward": 0.0, "success_rate": 0.0}}}}})
    a = audit_evidence(_ctx_dir(tmp_path, ledger_ok=1))
    assert a.metrics_non_degenerate is False
    assert a.run_level_clean is False
    assert any("zero" in r.lower() or "constant" in r.lower() for r in a.reasons)


def test_audit_evidence_unbacked_when_ok_count_zero(tmp_path):
    from backend.agents.rlm.evidence_audit import audit_evidence
    _write_metrics(tmp_path, {"per_model": {"m": {"e": {"b":
                  {"status": "ok", "accuracy": 0.8}}}}})
    a = audit_evidence(_ctx_dir(tmp_path, ledger_ok=0))
    assert a.backed_by_ledger is False
    assert a.run_level_clean is False


def test_audit_evidence_backed_when_no_ledger(tmp_path):
    # None ledger (replay/postmortem) -> trust content, do not fail closed.
    from backend.agents.rlm.evidence_audit import audit_evidence
    _write_metrics(tmp_path, {"per_model": {"m": {"e": {"b":
                  {"status": "ok", "accuracy": 0.8}}}}})
    a = audit_evidence(_ctx_dir(tmp_path, ledger_ok=None))
    assert a.backed_by_ledger is True


def test_audit_evidence_fingerprint_deterministic_and_sensitive(tmp_path):
    from backend.agents.rlm.evidence_audit import audit_evidence
    _write_metrics(tmp_path, {"per_model": {"m": {"e": {"b":
                  {"status": "ok", "accuracy": 0.8}}}}})
    ctx = _ctx_dir(tmp_path, ledger_ok=1)
    fp1 = audit_evidence(ctx).fingerprint
    fp2 = audit_evidence(ctx).fingerprint
    assert fp1 and fp1 == fp2
    _write_metrics(tmp_path, {"per_model": {"m": {"e": {"b":
                  {"status": "ok", "accuracy": 0.9}}}}})
    assert audit_evidence(ctx).fingerprint != fp1


def test_audit_evidence_failsoft_on_missing_dir(tmp_path):
    # No code/ dir at all -> empty metrics, no raise, backed defaults True.
    from backend.agents.rlm.evidence_audit import audit_evidence
    a = audit_evidence(_ctx_dir(tmp_path, ledger_ok=None))
    assert isinstance(a.fingerprint, str)
    assert a.provenance_present is False


# ---------------------------------------------------------------------------
# Plan 2: apply_result_veto()
# ---------------------------------------------------------------------------

_SDAR_V6 = {"success": True, "metrics": {"per_model": {"Qwen/Qwen3-1.7B": {"alfworld": {"sdar":
           {"status": "ok", "device": "cuda", "success_rate": 0.0, "reward": 0.0}}}}}}


def test_apply_result_veto_unchanged_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_AUDIT", raising=False)
    from backend.agents.rlm.evidence_audit import apply_result_veto
    src = dict(_SDAR_V6)
    out = apply_result_veto(src, _ctx(tmp_path))
    assert out is src  # byte-identical: same object returned, untouched


def test_apply_result_veto_degrades_when_flag_on(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import apply_result_veto
    emitted = []
    out = apply_result_veto(dict(_SDAR_V6), _ctx(tmp_path), emit=emitted.append)
    assert out["success"] is False
    assert out["failure_class"] == "fabrication_suspected"
    assert out["error"]
    assert emitted and ("zero" in emitted[0].lower() or "constant" in emitted[0].lower())


def test_apply_result_veto_clean_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import apply_result_veto
    clean = {"success": True, "metrics": {"per_model": {"m": {"e": {"b":
            {"status": "ok", "accuracy": 0.83}}}}}}
    out = apply_result_veto(clean, _ctx(tmp_path))
    assert out is clean  # identity: untouched


def test_apply_result_veto_preserves_other_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import apply_result_veto
    src = dict(_SDAR_V6, logs="x", wall_time_s=1.5)
    out = apply_result_veto(src, _ctx(tmp_path))
    assert out["success"] is False
    assert out["logs"] == "x" and out["wall_time_s"] == 1.5


def test_apply_result_veto_emit_failsoft(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import apply_result_veto

    def bad_emit(_msg):
        raise RuntimeError("boom")

    out = apply_result_veto(dict(_SDAR_V6), _ctx(tmp_path), emit=bad_emit)
    assert out["success"] is False  # degraded despite emit failure
