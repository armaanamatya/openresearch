"""Replay regression for the final-report evidence gate (review #5, 2026-05-30).

`scripts/replay_run.py` is the read-only postmortem harness; this test promotes
its core check to CI. It replays a committed catalog of known-broken /runs shapes
(`tests/fixtures/evidence_gate_replay/cases.json`) — the minimal sanitized report
inputs + experiment rows, NO paper corpus — through the real report writer and
asserts the gate produces the corrected verdict.

Why a catalog and not just inline cases: when a real run ships a hollow verdict,
the fix is to add its shape to cases.json (pre_gate_verdict + baseline_metrics +
experiment_runs rows). That turns every observed leak into a permanent regression
without touching this file. The `positive_control_real_evidence` case must stay
`partial` — it fails loudly if the gate ever starts over-firing (downgrading runs
that DO have real evidence).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm

_CASES_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "evidence_gate_replay" / "cases.json"


def _load_cases() -> list[dict]:
    data = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    return list(data["cases"])


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["id"])
def test_evidence_gate_replay(case: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")

    rows = case.get("experiment_rows") or []
    if rows:
        (tmp_path / "experiment_runs.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

    report = RLMFinalReport(
        verdict=case["pre_gate_verdict"],
        reproduction_summary=case.get("note", "replayed run"),
        baseline_metrics=case.get("baseline_metrics") or {},
    )
    # ``run_experiment_calls`` (the authoritative in-memory ledger count) is the
    # forged-evidence cross-check input. Absent in a case → None → content-only.
    json_path, _ = write_final_report_rlm(
        report, tmp_path, run_experiment_calls=case.get("run_experiment_calls")
    )
    written = json.loads(json_path.read_text(encoding="utf-8"))

    assert written["verdict"] == case["expected_verdict"], (
        f"replay case {case['id']!r} ({case['source']}): "
        f"expected {case['expected_verdict']!r}, got {written['verdict']!r}"
    )


def test_catalog_has_a_positive_control() -> None:
    """A replay catalog of only 'must downgrade to failed' cases would pass even if
    the gate downgraded EVERYTHING. Require at least one 'must stay success-ish' case."""
    cases = _load_cases()
    assert any(c["expected_verdict"] in {"partial", "reproduced"} for c in cases), (
        "evidence_gate_replay/cases.json needs a positive control (a case the gate must NOT downgrade)"
    )
