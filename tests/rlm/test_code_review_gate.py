"""Tests for backend/agents/rlm/code_review_gate.py (P1 — §4.1)."""

from __future__ import annotations

import json

import backend.agents.rlm.code_review_gate as crg


class TestEnabled:
    def test_requires_both_flags(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_CODE_REVIEW_GATE", "1")
        monkeypatch.delenv("OPENRESEARCH_EXTERNAL_VALIDATOR", raising=False)
        assert crg.code_review_gate_enabled() is False
        monkeypatch.setenv("OPENRESEARCH_EXTERNAL_VALIDATOR", "1")
        assert crg.code_review_gate_enabled() is True

    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_CODE_REVIEW_GATE", raising=False)
        monkeypatch.setenv("OPENRESEARCH_EXTERNAL_VALIDATOR", "1")
        assert crg.code_review_gate_enabled() is False


def _write_train(tmp_path, lines=20):
    p = tmp_path / "train.py"
    p.write_text("\n".join(f"x = {i}" for i in range(lines)), encoding="utf-8")
    return p


def _patch_completion(monkeypatch, payload):
    def _fake(client, *, system, user, n):  # noqa: ARG001
        return [payload]
    monkeypatch.setattr(crg, "sample_completions", _fake, raising=False)
    # the gate imports sample_completions lazily from grader_transport
    import backend.agents.rlm.grader_transport as gt
    monkeypatch.setattr(gt, "sample_completions", _fake, raising=False)


class TestReview:
    def test_client_none_fail_open(self, tmp_path):
        _write_train(tmp_path)
        v = crg.review_executor_code(validator_client=None, code_dir=tmp_path, method_context="m")
        assert v.blocking is False

    def test_no_files_fail_open(self, tmp_path):
        v = crg.review_executor_code(validator_client=object(), code_dir=tmp_path, method_context="m")
        assert v.blocking is False

    def test_blocking_high_severity_grounded(self, tmp_path, monkeypatch):
        _write_train(tmp_path, lines=20)
        _patch_completion(monkeypatch, json.dumps([{
            "file": "train.py", "line": 5, "severity": "will_produce_fake_metrics",
            "anti_pattern": "constant_loss", "detail": "loss hardcoded to 0",
        }]))
        v = crg.review_executor_code(validator_client=object(), code_dir=tmp_path, method_context="m")
        assert v.blocking is True

    def test_advisory_not_blocking(self, tmp_path, monkeypatch):
        _write_train(tmp_path)
        _patch_completion(monkeypatch, json.dumps([{
            "file": "train.py", "line": 5, "severity": "style",
            "anti_pattern": "naming", "detail": "rename var",
        }]))
        v = crg.review_executor_code(validator_client=object(), code_dir=tmp_path, method_context="m")
        assert v.blocking is False

    def test_high_severity_ungrounded_file_not_blocking(self, tmp_path, monkeypatch):
        _write_train(tmp_path)
        _patch_completion(monkeypatch, json.dumps([{
            "file": "nonexistent.py", "line": 5, "severity": "will_produce_fake_metrics",
            "anti_pattern": "x", "detail": "y",
        }]))
        v = crg.review_executor_code(validator_client=object(), code_dir=tmp_path, method_context="m")
        assert v.blocking is False

    def test_high_severity_line_past_eof_not_blocking(self, tmp_path, monkeypatch):
        _write_train(tmp_path, lines=10)
        _patch_completion(monkeypatch, json.dumps([{
            "file": "train.py", "line": 9999, "severity": "will_produce_fake_metrics",
            "anti_pattern": "x", "detail": "y",
        }]))
        v = crg.review_executor_code(validator_client=object(), code_dir=tmp_path, method_context="m")
        assert v.blocking is False

    def test_empty_findings_not_blocking(self, tmp_path, monkeypatch):
        _write_train(tmp_path)
        _patch_completion(monkeypatch, "[]")
        v = crg.review_executor_code(validator_client=object(), code_dir=tmp_path, method_context="m")
        assert v.blocking is False

    def test_unparseable_response_fail_open(self, tmp_path, monkeypatch):
        _write_train(tmp_path)
        _patch_completion(monkeypatch, "not json at all")
        v = crg.review_executor_code(validator_client=object(), code_dir=tmp_path, method_context="m")
        assert v.blocking is False

    def test_call_raises_fail_open(self, tmp_path, monkeypatch):
        _write_train(tmp_path)

        def _boom(client, *, system, user, n):  # noqa: ARG001
            raise RuntimeError("transport down")
        import backend.agents.rlm.grader_transport as gt
        monkeypatch.setattr(gt, "sample_completions", _boom, raising=False)
        v = crg.review_executor_code(validator_client=object(), code_dir=tmp_path, method_context="m")
        assert v.blocking is False
