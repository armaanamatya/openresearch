"""Track E Task 8 — acceptance battery on REAL frozen run dirs.

Drives the full Track E chain (``EvaluationReport.from_run`` +
``build_scorecard`` + ``write_evaluation_report``) on the operator's actual run
dirs — Adam (``reproduced``), UCPO (``partial`` and a sparse ``failed``) — and
asserts the §8 invariants: the scorecard is coherent, the verdict is copied
read-only (never recomputed, never lifted), the sidecar write never mutates
``final_report.json``'s verdict surface, missing metrics read ``unmeasured``
(never auto-``pass``), and no display row can contribute a gate cap.

Runs are COPIED to ``tmp_path`` first — ``runs/`` is never mutated. SDAR is the
canonical stress goal, never the correctness oracle here (no SDAR run dir is
required for this battery).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from backend.evals.evaluation_report import EvaluationReport
from backend.evals.scorecard import (
    DISPLAY_DIMENSIONS,
    GATE_DIMENSIONS,
    build_scorecard,
    write_evaluation_report,
)

_RUNS = Path(__file__).resolve().parents[2] / "runs"

# (run dir, expected verdict already on disk) — spans the verdict range.
_CASES = [
    ("prj_adam_local_1", "reproduced"),
    ("prj_ucpo_optA_2", "partial"),
    ("prj_ucpo_optA_1", "failed"),
]

# Keep the copy small + fast: skip weights/datasets/vendored trees (the scorecard
# only reads small JSON/JSONL artifacts).
_IGNORE_DIRS = {
    "datasets", "outputs", ".venv", "venv", "__pycache__", "node_modules",
    "wandb", "checkpoints", ".git", "weights", ".preserved",
}
_IGNORE_EXT = (".pt", ".safetensors", ".bin", ".ckpt", ".pth", ".gz", ".zip", ".tar", ".log")


def _ignore(_dir, names):
    return {
        n for n in names
        if n in _IGNORE_DIRS or n.lower().endswith(_IGNORE_EXT)
    }


def _copy_run(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=_ignore, ignore_dangling_symlinks=True)


@pytest.mark.parametrize("run_name,expected_verdict", _CASES)
def test_scorecard_coherent_and_verdict_preserving(tmp_path, monkeypatch, run_name, expected_verdict):
    src = _RUNS / run_name
    if not src.exists():
        pytest.skip(f"{run_name} not present in runs/")
    dst = tmp_path / run_name
    _copy_run(src, dst)

    before = json.loads((dst / "final_report.json").read_text(encoding="utf-8"))
    assert before.get("verdict") == expected_verdict  # the frozen truth on disk

    monkeypatch.setenv("OPENRESEARCH_EVAL_SCORECARD", "1")
    out = write_evaluation_report(dst)
    assert out is not None and out.exists()
    assert (dst / "evaluation_report.md").exists()

    # North-star: emitting the scorecard must NOT mutate final_report.json's verdict.
    after = json.loads((dst / "final_report.json").read_text(encoding="utf-8"))
    assert after.get("verdict") == expected_verdict

    er = EvaluationReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert er.verdict == expected_verdict  # copied read-only, never recomputed/lifted

    # Exactly the 11 dimensions, in the stable order, coherent per family.
    dims = [r.dimension for r in er.scorecard]
    assert dims == list(GATE_DIMENSIONS) + list(DISPLAY_DIMENSIONS)
    for row in er.scorecard:
        if row.dimension in GATE_DIMENSIONS:
            assert row.gates is True
            assert row.status in {"pass", "fail", "unmeasured", "excluded"}
        else:
            assert row.gates is False
            assert row.status == "display"

    # A display-only scorecard can never produce a gate cap.
    display_only = er.model_copy(update={"scorecard": [r for r in er.scorecard if not r.gates]})
    assert display_only.gate_caps() is None


def test_missing_metrics_is_unmeasured_never_autopass(tmp_path):
    """The sparse UCPO run (no metrics.json / repro_spec.json) must read the
    numerical dimension as ``unmeasured`` — never a fabricated ``pass``."""
    src = _RUNS / "prj_ucpo_optA_1"
    if not src.exists():
        pytest.skip("prj_ucpo_optA_1 not present in runs/")
    dst = tmp_path / "sparse"
    _copy_run(src, dst)
    rows = {r.dimension: r for r in build_scorecard(dst)}
    assert rows["numerical_reproduction"].status == "unmeasured"


def test_serialization_round_trip(tmp_path, monkeypatch):
    src = _RUNS / "prj_adam_local_1"
    if not src.exists():
        pytest.skip("prj_adam_local_1 not present in runs/")
    dst = tmp_path / "adam"
    _copy_run(src, dst)
    monkeypatch.setenv("OPENRESEARCH_EVAL_SCORECARD", "1")
    out = write_evaluation_report(dst)
    er = EvaluationReport.model_validate_json(out.read_text(encoding="utf-8"))
    # dump -> reload is stable (no lossy field)
    assert EvaluationReport.model_validate_json(er.model_dump_json()) == er
