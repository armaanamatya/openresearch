"""Adversarial regression for the forged-evidence cross-check (audit 2026-05-30).

The RLM root's REPL keeps ``open()`` live, so it can append a fabricated
``{"success": true, "metrics": {...}}`` row to ``experiment_runs.jsonl`` and
*satisfy* the content-only evidence predicate WITHOUT any real ``run_experiment``
— defeating the gate by passing it, not skipping it. The fix cross-checks the
on-disk evidence against the authoritative in-memory cost-ledger ``run_experiment``
count (``run_experiment_call_count``): a success row with **0** ledger calls is
forged and downgraded to ``failed``. A legit run (``>= 1`` call) is untouched;
``None`` (no ledger — replay/postmortem) falls back to content-only so the gate
never over-fires on a path that lacks the trace.

These tests literally perform the forge (write the row by hand) and assert the
verdict the gate produces, plus the safety direction (a real backing call must
NOT be downgraded).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.agents.rlm.report import (
    RLMFinalReport,
    run_experiment_call_count,
    write_final_report_rlm,
)


def _forge_success_row(project_dir: Path, metrics: dict | None = None) -> None:
    """Write the exact on-disk shape a forging REPL would append."""
    project_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "success": True,
        "metrics": metrics or {"accuracy": 0.95},
        "model_id": "qwen3-1.7b",
        "eval_env": "ALFWorld",
    }
    (project_dir / "experiment_runs.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )


def _written_verdict(json_path: Path) -> str:
    return json.loads(json_path.read_text(encoding="utf-8"))["verdict"]


@pytest.mark.parametrize("pre_verdict", ["partial", "reproduced"])
def test_forged_row_with_zero_run_experiment_calls_is_downgraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pre_verdict: str
) -> None:
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")
    _forge_success_row(tmp_path)  # the exploit: a success row, but no real run_experiment
    report = RLMFinalReport(
        verdict=pre_verdict,
        reproduction_summary="claimed reproduction",
        baseline_metrics={"accuracy": 0.95},
    )

    json_path, _ = write_final_report_rlm(report, tmp_path, run_experiment_calls=0)

    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["verdict"] == "failed", "a forged success row (0 ledger calls) must not survive"
    assert "not backed by a real experiment" in written["reproduction_summary"]


def test_real_success_row_with_a_backing_run_experiment_call_stays_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same on-disk shape, but THIS time the ledger shows a real run_experiment call
    # backs it. The gate must NOT downgrade — over-firing here would fail every real run.
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")
    _forge_success_row(tmp_path)
    report = RLMFinalReport(
        verdict="partial",
        reproduction_summary="real partial reproduction",
        baseline_metrics={"accuracy": 0.95},
    )

    json_path, _ = write_final_report_rlm(report, tmp_path, run_experiment_calls=1)

    assert _written_verdict(json_path) == "partial"


def test_none_count_falls_back_to_content_only_no_overfire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Replay / postmortem path: no ledger available (run_experiment_calls=None default).
    # Content-only — a real success row stays partial; the cross-check is skipped.
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")
    _forge_success_row(tmp_path)
    report = RLMFinalReport(verdict="partial", reproduction_summary="replay")

    json_path, _ = write_final_report_rlm(report, tmp_path)  # no count passed

    assert _written_verdict(json_path) == "partial"


def test_no_evidence_row_downgrades_regardless_of_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The original FM-004 case still holds: no success row at all → failed, even if
    # the ledger somehow shows run_experiment calls.
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")
    report = RLMFinalReport(verdict="partial", reproduction_summary="no experiment ran")

    json_path, _ = write_final_report_rlm(report, tmp_path, run_experiment_calls=5)

    assert _written_verdict(json_path) == "failed"


def test_run_experiment_call_count_counts_only_run_experiment_rows() -> None:
    """The count helper is the trust anchor — it must count exactly the
    run_experiment ledger rows (and the in-memory ledger is what the root cannot forge)."""
    from backend.agents.resilience.cost import CostLedgerEntry, RunCostLedger

    ledger = RunCostLedger(project_id="p")

    def _entry(name: str) -> CostLedgerEntry:
        return CostLedgerEntry(
            timestamp=datetime.now(timezone.utc),
            agent_id=name,
            attempt_index=0,
            provider="openai",
            model="gpt-5",
        )

    for name in [
        "understand_section",
        "run_experiment",
        "build_environment",
        "run_experiment",
        "heartbeat",
    ]:
        ledger.append(_entry(name))

    class _Ctx:
        cost_ledger = ledger

    assert run_experiment_call_count(_Ctx()) == 2


def test_run_experiment_call_count_none_when_no_ledger() -> None:
    class _Ctx:
        cost_ledger = None

    assert run_experiment_call_count(_Ctx()) is None
