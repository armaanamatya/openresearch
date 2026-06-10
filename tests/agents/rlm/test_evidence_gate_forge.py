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
    run_experiment_success_count,
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
    assert "not backed by a real successful experiment" in written["reproduction_summary"]


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


def _forge_ledger_row(project_dir: Path) -> None:
    """Write the exact cost_ledger.jsonl shape a forging REPL would append
    (the to_json field names CostLedgerEntry.from_json round-trips)."""
    from backend.agents.resilience.cost import CostLedgerEntry

    project_dir.mkdir(parents=True, exist_ok=True)
    entry = CostLedgerEntry(
        timestamp=datetime.now(timezone.utc),
        agent_id="run_experiment",
        attempt_index=0,
        provider="openai",
        model="gpt-5",
    )
    with (project_dir / "cost_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_json()) + "\n")


def test_warm_retry_does_not_trust_disk_seeded_ledger_rows(tmp_path: Path) -> None:
    """Audit 2026-06-09: the cross-check counter must be session-scoped.

    Exploit it closes: in run 1 the root forges BOTH a success row in
    experiment_runs.jsonl AND a run_experiment row in cost_ledger.jsonl via its
    live ``open()``; run 1 dies without final_report.json (SIGKILL/OOM), so
    attempt isolation archives nothing; run 2 warm-retries the same project dir
    and ``RunCostLedger.load_jsonl`` re-ingests the forged ledger row. Counting
    seeded rows would yield 1 and let the forged verdict ship.
    """
    from backend.agents.resilience.cost import RunCostLedger

    _forge_ledger_row(tmp_path)
    ledger = RunCostLedger.load_jsonl(tmp_path / "cost_ledger.jsonl", project_id="p")
    assert len(ledger.entries) == 1  # seeding itself still works (budget continuity)

    class _Ctx:
        cost_ledger = ledger

    assert run_experiment_call_count(_Ctx()) == 0


def test_warm_retry_forged_files_yield_failed_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end warm-retry forge: forged evidence row + forged (re-ingested)
    ledger row must still downgrade to 'failed'."""
    from backend.agents.resilience.cost import RunCostLedger

    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")
    _forge_success_row(tmp_path)
    _forge_ledger_row(tmp_path)
    ledger = RunCostLedger.load_jsonl(tmp_path / "cost_ledger.jsonl", project_id="p")

    class _Ctx:
        cost_ledger = ledger

    report = RLMFinalReport(
        verdict="reproduced",
        reproduction_summary="claimed reproduction",
        baseline_metrics={"accuracy": 0.95},
    )
    json_path, _ = write_final_report_rlm(
        report, tmp_path, run_experiment_calls=run_experiment_call_count(_Ctx())
    )

    assert _written_verdict(json_path) == "failed"


def test_session_appends_after_seeding_still_count(tmp_path: Path) -> None:
    """Budget seeding must not break the legit path: a REAL in-process
    run_experiment call after warm-retry seeding counts."""
    from backend.agents.resilience.cost import CostLedgerEntry, RunCostLedger

    _forge_ledger_row(tmp_path)  # stale/seeded row (untrusted)
    ledger = RunCostLedger.load_jsonl(tmp_path / "cost_ledger.jsonl", project_id="p")
    ledger.append(
        CostLedgerEntry(
            timestamp=datetime.now(timezone.utc),
            agent_id="run_experiment",
            attempt_index=1,
            provider="openai",
            model="gpt-5",
        )
    )

    class _Ctx:
        cost_ledger = ledger

    assert run_experiment_call_count(_Ctx()) == 1
    assert ledger.total_usd() == 0.0  # cumulative cost math untouched by scoping


# ---------------------------------------------------------------------------
# Per-row provenance (audit 2026-06-10) — closes the former KNOWN RESIDUAL #2:
# one real-but-FAILED run_experiment call + a forged success row passed the
# >=1 total-count check. The outcome stamp written by binding.wrap_primitive
# ("ok"/"failed"/"raised") now distinguishes them.
# ---------------------------------------------------------------------------


def _ledger_with_one_call(outcome: str):
    from backend.agents.resilience.cost import CostLedgerEntry, RunCostLedger

    ledger = RunCostLedger(project_id="p")
    ledger.append(
        CostLedgerEntry(
            timestamp=datetime.now(timezone.utc),
            agent_id="run_experiment",
            attempt_index=0,
            provider="openai",
            model="gpt-5",
            outcome=outcome,
        )
    )

    class _Ctx:
        cost_ledger = ledger

    return _Ctx()


def test_one_real_failed_call_plus_forged_success_row_is_downgraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE residual: run a real container that FAILS (1 ledger call, outcome
    'failed'), then forge a success row via the REPL's open(). The >=1 total
    count used to bless it; the success-compatible count now rejects it."""
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")
    _forge_success_row(tmp_path)
    ctx = _ledger_with_one_call("failed")

    assert run_experiment_call_count(ctx) == 1  # the old check alone would pass
    assert run_experiment_success_count(ctx) == 0

    report = RLMFinalReport(
        verdict="reproduced",
        reproduction_summary="claimed reproduction",
        baseline_metrics={"accuracy": 0.95},
    )
    json_path, _ = write_final_report_rlm(
        report,
        tmp_path,
        run_experiment_calls=run_experiment_call_count(ctx),
        run_experiment_ok_calls=run_experiment_success_count(ctx),
    )

    assert _written_verdict(json_path) == "failed"


def test_real_successful_call_still_backs_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")
    _forge_success_row(tmp_path)  # here it stands in for the REAL success row
    ctx = _ledger_with_one_call("ok")

    report = RLMFinalReport(
        verdict="partial",
        reproduction_summary="real run",
        baseline_metrics={"accuracy": 0.95},
    )
    json_path, _ = write_final_report_rlm(
        report,
        tmp_path,
        run_experiment_calls=run_experiment_call_count(ctx),
        run_experiment_ok_calls=run_experiment_success_count(ctx),
    )

    assert _written_verdict(json_path) == "partial"


def test_unstamped_legacy_rows_stay_success_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Back-compat: outcome='' (legacy doubles, pre-stamp artifacts) must never
    over-downgrade — only explicit failed/raised stamps refuse to back a row."""
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")
    _forge_success_row(tmp_path)
    ctx = _ledger_with_one_call("")

    assert run_experiment_success_count(ctx) == 1

    report = RLMFinalReport(
        verdict="partial",
        reproduction_summary="legacy path",
        baseline_metrics={"accuracy": 0.95},
    )
    json_path, _ = write_final_report_rlm(
        report,
        tmp_path,
        run_experiment_calls=run_experiment_call_count(ctx),
        run_experiment_ok_calls=run_experiment_success_count(ctx),
    )

    assert _written_verdict(json_path) == "partial"


def test_outcome_round_trips_through_jsonl() -> None:
    """to_json/from_json must carry the stamp; old rows without it parse fine."""
    from backend.agents.resilience.cost import CostLedgerEntry

    entry = CostLedgerEntry(
        timestamp=datetime.now(timezone.utc),
        agent_id="run_experiment",
        attempt_index=0,
        provider="openai",
        model="gpt-5",
        outcome="failed",
    )
    assert CostLedgerEntry.from_json(entry.to_json()).outcome == "failed"

    legacy = entry.to_json()
    del legacy["outcome"]
    assert CostLedgerEntry.from_json(legacy).outcome == ""
