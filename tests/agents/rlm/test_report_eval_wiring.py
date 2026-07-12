"""Phase-2a wiring tests: the two report.py finalize hooks for Track E.

Covers the OFF+ON pair for both hooks (tests/CLAUDE.md convention) and proves
neither can move the verdict surface:

  * **E-6 scorecard sidecar** — ``write_final_report_rlm`` emits
    ``evaluation_report.{json,md}`` when ``OPENRESEARCH_EVAL_SCORECARD`` is on,
    and writes nothing when off (byte-identical). The sidecar copies the
    verdict read-only; it never mutates ``final_report.json``'s verdict surface,
    so the existing single-writer tripwire in ``write_final_report_rlm`` still
    passes with the flag on and the authority active.
  * **E-4 ok-receipt fallback** — ``run_experiment_success_count`` returns the
    forge-resistant on-disk receipt count ONLY when the in-memory ledger is
    absent AND ``OPENRESEARCH_OK_RECEIPT`` is on AND genuine receipts exist;
    otherwise it stays ``None`` (byte-identical to today).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from backend.agents.rlm import ok_receipt
from backend.agents.rlm.report import (
    RLMFinalReport,
    run_experiment_success_count,
    write_final_report_rlm,
)


# --------------------------------------------------------------------------- #
# E-6: the EvaluationReport scorecard sidecar at finalize
# --------------------------------------------------------------------------- #


def _shipped_verdict(project_dir: Path) -> str:
    return json.loads((project_dir / "final_report.json").read_text(encoding="utf-8"))["verdict"]


def test_scorecard_sidecar_written_when_flag_on(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EVAL_SCORECARD", "1")
    project_dir = tmp_path / "run"
    project_dir.mkdir()
    write_final_report_rlm(RLMFinalReport(verdict="partial"), project_dir)

    sidecar = project_dir / "evaluation_report.json"
    assert sidecar.exists()
    assert (project_dir / "evaluation_report.md").exists()
    er = json.loads(sidecar.read_text(encoding="utf-8"))
    # verdict copied read-only from the just-written final_report.json (whatever
    # value landed there — here the default-on evidence gate downgrades the
    # evidence-less "partial" to "failed"; the sidecar faithfully mirrors it).
    assert er["verdict"] == _shipped_verdict(project_dir)
    # 11 dimensions, all at their safe defaults on an empty run dir
    assert len(er["scorecard"]) == 11


def test_no_sidecar_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EVAL_SCORECARD", raising=False)
    project_dir = tmp_path / "run"
    project_dir.mkdir()
    write_final_report_rlm(RLMFinalReport(verdict="partial"), project_dir)

    assert (project_dir / "final_report.json").exists()  # base artifact still written
    assert not (project_dir / "evaluation_report.json").exists()
    assert not (project_dir / "evaluation_report.md").exists()


def test_scorecard_sidecar_does_not_move_the_verdict_under_active_authority(tmp_path, monkeypatch):
    """With the authority active AND the scorecard flag on, the shipped verdict
    is exactly what decide() strikes — the sidecar (written after both verdict
    tripwires) cannot perturb it, and the tripwire does not fire.
    """
    monkeypatch.setenv("OPENRESEARCH_TWO_AXIS_VERDICT", "1")
    monkeypatch.setenv("OPENRESEARCH_VERDICT_AUTHORITY", "1")
    monkeypatch.setenv("OPENRESEARCH_EVAL_SCORECARD", "1")
    project_dir = tmp_path / "run"
    (project_dir / "code").mkdir(parents=True)
    (project_dir / "rlm_state").mkdir(parents=True)
    (project_dir / "code" / "metrics.json").write_text(json.dumps({"accuracy": 0.991}), encoding="utf-8")
    (project_dir / "rlm_state" / "repro_spec.json").write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "primary_0",
                        "is_primary": True,
                        "kind": "numeric",
                        "metric_name": "accuracy",
                        "claimed_effect": 0.99,
                        "equivalence_margin": 0.01,
                        "direction": "higher_is_better",
                        "scope": {},
                        "ambiguous": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (project_dir / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True, "metrics": {"accuracy": 0.991}}) + "\n", encoding="utf-8"
    )

    # A passing primary claim + real evidence => decide() strikes "reproduced";
    # if the sidecar had tripped the single-writer guard this call would raise.
    write_final_report_rlm(
        RLMFinalReport(verdict="partial", rubric={"overall_score": 0.0, "target_score": 0.6, "areas": []}),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )
    assert _shipped_verdict(project_dir) == "reproduced"
    # sidecar mirrors the authoritative verdict read-only
    er = json.loads((project_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    assert er["verdict"] == "reproduced"


# --------------------------------------------------------------------------- #
# E-4: the forge-resistant ok-receipt fallback for an out-of-process re-grade
# --------------------------------------------------------------------------- #


def _write_receipts(project_dir: Path, monkeypatch, n: int) -> None:
    monkeypatch.setenv("OPENRESEARCH_OK_RECEIPT", "1")
    (project_dir / "rlm_state").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        assert ok_receipt.write_ok_receipt(
            project_dir, experiment_run_id=f"e{i}", ok=True, metrics_sha256=f"sha{i}", ts=f"t{i}"
        )


def test_fallback_lifts_count_when_receipts_present(tmp_path, monkeypatch):
    _write_receipts(tmp_path, monkeypatch, 2)
    ctx = SimpleNamespace(cost_ledger=None, project_dir=tmp_path)
    assert run_experiment_success_count(ctx) == 2


def test_fallback_is_none_when_flag_off(tmp_path, monkeypatch):
    _write_receipts(tmp_path, monkeypatch, 2)  # receipts exist on disk...
    monkeypatch.delenv("OPENRESEARCH_OK_RECEIPT", raising=False)  # ...but flag is off
    ctx = SimpleNamespace(cost_ledger=None, project_dir=tmp_path)
    assert run_experiment_success_count(ctx) is None  # byte-identical to today


def test_fallback_is_none_when_no_receipts(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_OK_RECEIPT", "1")
    ctx = SimpleNamespace(cost_ledger=None, project_dir=tmp_path)
    # flag on but zero forge-resistant receipts => stays None (never a spurious 0
    # that would newly cap the verdict); the gate skips the check exactly as today
    assert run_experiment_success_count(ctx) is None


def test_present_ledger_ignores_receipts(tmp_path, monkeypatch):
    """When an in-memory ledger IS present, the fallback is never consulted —
    the ledger stays authoritative regardless of on-disk receipts."""
    _write_receipts(tmp_path, monkeypatch, 5)
    ledger = SimpleNamespace(session_success_compatible_count=lambda _agent: 1)
    ctx = SimpleNamespace(cost_ledger=ledger, project_dir=tmp_path)
    assert run_experiment_success_count(ctx) == 1  # from the ledger, not the 5 receipts
