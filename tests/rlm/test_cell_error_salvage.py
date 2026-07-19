"""Cell-error salvage (OPENRESEARCH_CELL_ERROR_SALVAGE): a run whose cells
EXECUTED then errored with real graded metrics salvages to 'partial' — but only
when a harness-owned receipt (cell_manifest error status) + a session-scoped
partial_cell_error ledger stamp prove an OBSERVED cell run. A REPL-forged
cell_execution_error row (no session stamp) still fails closed."""
import json
from pathlib import Path

from backend.agents.rlm.report import (
    RLMFinalReport,
    _has_cell_manifest_error_receipt,
    write_final_report_rlm,
)


def _written_verdict(json_path) -> str:
    return json.loads(Path(json_path).read_text(encoding="utf-8"))["verdict"]


def _write_manifest(project_dir: Path, run_id: str, cell_id: str, status: str) -> None:
    d = project_dir / "code" / "outputs" / run_id / cell_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "cell_manifest.json").write_text(
        json.dumps({"cell_id": cell_id, "status": status, "fingerprint": "abc"}),
        encoding="utf-8",
    )


def test_manifest_error_receipt_true(tmp_path):
    _write_manifest(tmp_path, "run-1", "monolithic", "error")
    assert _has_cell_manifest_error_receipt(tmp_path) is True


def test_manifest_oom_receipt_true(tmp_path):
    _write_manifest(tmp_path, "run-1", "monolithic", "oom_failed")
    assert _has_cell_manifest_error_receipt(tmp_path) is True


def test_manifest_ok_receipt_false(tmp_path):
    _write_manifest(tmp_path, "run-1", "monolithic", "ok")
    assert _has_cell_manifest_error_receipt(tmp_path) is False


def test_no_manifest_false(tmp_path):
    assert _has_cell_manifest_error_receipt(tmp_path) is False


# --- gate-tier integration (end-to-end through write_final_report_rlm) ---

def test_cell_error_run_salvages_to_partial(tmp_path, monkeypatch):
    """cutout-shaped: reproduced-from-best-of-run, NO success row, but a real
    session partial_cell_error stamp + an error cell_manifest → salvages to
    'partial' (capped, never a full reproduction claim)."""
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")
    monkeypatch.setenv("OPENRESEARCH_CELL_ERROR_SALVAGE", "1")
    _write_manifest(tmp_path, "run-1", "monolithic", "error")
    report = RLMFinalReport(verdict="reproduced", reproduction_summary="cell errored")

    json_path, _ = write_final_report_rlm(
        report, tmp_path,
        run_experiment_calls=1,
        run_experiment_ok_calls=0,
        run_experiment_partial_timeout_calls=0,
        run_experiment_partial_cell_error_calls=1,
    )

    written = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert written["verdict"] == "partial"
    assert "[evidence_cap]" in written["reproduction_summary"]


def test_forged_cell_error_stays_failed(tmp_path, monkeypatch):
    """FORGERY guard: an error cell_manifest on disk but 0 session
    partial_cell_error stamps → stays 'failed' (a REPL cannot mint a session
    stamp, so the salvage tier is unreachable)."""
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")
    monkeypatch.setenv("OPENRESEARCH_CELL_ERROR_SALVAGE", "1")
    _write_manifest(tmp_path, "run-1", "monolithic", "error")
    report = RLMFinalReport(verdict="reproduced", reproduction_summary="forged")

    json_path, _ = write_final_report_rlm(
        report, tmp_path,
        run_experiment_calls=1,
        run_experiment_ok_calls=0,
        run_experiment_partial_timeout_calls=0,
        run_experiment_partial_cell_error_calls=0,  # no session stamp → no salvage
    )

    assert _written_verdict(json_path) == "failed"


def test_cell_error_salvage_flag_off_stays_failed(tmp_path, monkeypatch):
    """Flag OFF → byte-identical hard-downgrade (stays 'failed')."""
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")
    monkeypatch.delenv("OPENRESEARCH_CELL_ERROR_SALVAGE", raising=False)
    _write_manifest(tmp_path, "run-1", "monolithic", "error")
    report = RLMFinalReport(verdict="reproduced", reproduction_summary="flag off")

    json_path, _ = write_final_report_rlm(
        report, tmp_path,
        run_experiment_calls=1,
        run_experiment_ok_calls=0,
        run_experiment_partial_timeout_calls=0,
        run_experiment_partial_cell_error_calls=1,
    )

    assert _written_verdict(json_path) == "failed"
