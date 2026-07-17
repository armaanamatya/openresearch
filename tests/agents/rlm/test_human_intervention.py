"""Track E Task 2: deterministic HumanIntervention capture (autonomy telemetry).

Hermetic OFF+ON pair (tests/CLAUDE.md): the writer is default-OFF + byte-identical
off, and the autonomy metric is display-only (never gates). No network, tmp_path
fixtures only.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm.human_intervention import (
    autonomy_metric,
    human_intervention_enabled,
    record_intervention,
)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_HUMAN_INTERVENTION_LOG", raising=False)
    assert human_intervention_enabled() is False


def test_record_appends_row(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_HUMAN_INTERVENTION_LOG", "1")
    assert (
        record_intervention(
            tmp_path, kind="credentials", what="added HF token", blocking=True
        )
        is True
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "human_interventions.jsonl").read_text().splitlines()
    ]
    assert rows[0]["kind"] == "credentials"
    assert rows[0]["blocking"] is True
    assert "ts" in rows[0]


def test_off_state_no_write(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_HUMAN_INTERVENTION_LOG", raising=False)
    assert record_intervention(tmp_path, kind="clarification", what="x") is False
    assert not (tmp_path / "human_interventions.jsonl").exists()


def test_autonomy_metric_weights_credentials_lower(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_HUMAN_INTERVENTION_LOG", "1")
    record_intervention(tmp_path, kind="credentials", what="token")
    record_intervention(tmp_path, kind="scientific-judgment", what="pick")
    m = autonomy_metric(tmp_path)
    assert m["n_interventions"] == 2
    assert set(m["by_kind"]) == {"credentials", "scientific-judgment"}
    assert 0.0 <= m["autonomy_score"] <= 1.0


def test_credentials_cost_less_autonomy_than_code_fix(tmp_path, monkeypatch):
    # The weighting asymmetry: a run whose only intervention was supplying a
    # credential is scored MORE autonomous than one where a human had to fix code.
    monkeypatch.setenv("OPENRESEARCH_HUMAN_INTERVENTION_LOG", "1")
    creds_dir = tmp_path / "creds"
    code_dir = tmp_path / "code"
    creds_dir.mkdir()
    code_dir.mkdir()
    record_intervention(creds_dir, kind="credentials", what="token")
    record_intervention(code_dir, kind="code-fix", what="patched train.py")
    assert (
        autonomy_metric(creds_dir)["autonomy_score"]
        > autonomy_metric(code_dir)["autonomy_score"]
    )


def test_record_never_raises_on_bad_dir(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_HUMAN_INTERVENTION_LOG", "1")
    assert (
        record_intervention(Path("/nonexistent/xyz"), kind="clarification", what="x")
        is False
    )
