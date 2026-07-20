"""Evidence-decision audit log — OPENRESEARCH_EVIDENCE_DECISION_LOG (default OFF).

A structured, append-only record of every evidence-gate decision (which gate,
what outcome, on what evidence) so a verdict is auditable from logs alone.
Default-OFF ⇒ no file written, byte-identical. Fail-soft — never raises.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm.evidence_log import (
    read_evidence_decisions,
    record_evidence_decision,
)


def test_off_writes_no_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_DECISION_LOG", raising=False)
    record_evidence_decision(tmp_path, gate="grader_integrity", outcome="evidence_tampered")
    assert not (tmp_path / "rlm_state" / "evidence_decisions.jsonl").exists()


def test_on_appends_structured_row(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_DECISION_LOG", "1")
    record_evidence_decision(
        tmp_path, gate="grader_integrity", outcome="evidence_tampered",
        detail="rubric_mismatch", extra={"pinned_sha": "abc", "current_sha": "def"},
    )
    path = tmp_path / "rlm_state" / "evidence_decisions.jsonl"
    assert path.exists()
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln]
    assert len(rows) == 1
    r = rows[0]
    assert r["gate"] == "grader_integrity"
    assert r["outcome"] == "evidence_tampered"
    assert r["detail"] == "rubric_mismatch"
    assert r["pinned_sha"] == "abc"
    assert "ts" in r  # a timestamp is stamped for auditability


def test_on_appends_multiple_and_reads_back(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_DECISION_LOG", "1")
    record_evidence_decision(tmp_path, gate="grader_integrity", outcome="ok")
    record_evidence_decision(tmp_path, gate="eval_coverage", outcome="veto", detail="n_eval=2<100")
    rows = read_evidence_decisions(tmp_path)
    assert [r["gate"] for r in rows] == ["grader_integrity", "eval_coverage"]
    assert rows[1]["detail"] == "n_eval=2<100"


def test_read_missing_is_empty(tmp_path: Path):
    assert read_evidence_decisions(tmp_path) == []


def test_fail_soft_on_bad_run_dir(monkeypatch):
    """A record call must never raise, even with an unwritable/garbage path."""
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_DECISION_LOG", "1")
    # A path whose parent is a file, not a dir — mkdir would fail; must fail soft.
    record_evidence_decision("/dev/null/nope", gate="x", outcome="y")  # no raise
