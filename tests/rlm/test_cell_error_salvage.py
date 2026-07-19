"""Cell-error salvage (OPENRESEARCH_CELL_ERROR_SALVAGE): a run whose cells
EXECUTED then errored with real graded metrics salvages to 'partial' — but only
when a harness-owned receipt (cell_manifest error status) + a session-scoped
partial_cell_error ledger stamp prove an OBSERVED cell run. A REPL-forged
cell_execution_error row (no session stamp) still fails closed."""
import json
from pathlib import Path

from backend.agents.rlm.report import _has_cell_manifest_error_receipt


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
