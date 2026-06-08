"""D2 — the grader's ``_gather_evidence`` must surface the agent-emitted provenance
manifest + per-figure sidecars (the dominant lever on the "legibility" docks:
"45-epoch not confirmed", "log-axis not verifiable"), and must no longer truncate
code at 6 KB. These are deterministic, LLM-free guards for the consume side; the
emission side is covered by ``tests/agents/rlm/test_provenance.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.evals.paperbench.leaf_scorer import (
    _MAX_FILE_BYTES,
    _gather_evidence,
    _gather_figure_sidecars,
    _provenance_paths,
)


def _mk_run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    out = run / "code" / "outputs" / "abc123"
    out.mkdir(parents=True)
    (run / "code" / "train.py").write_text("import torch\nprint('train')\n", encoding="utf-8")
    return run, out


def test_gather_evidence_includes_provenance_manifest(tmp_path: Path):
    run, out = _mk_run(tmp_path)
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiments": {
                    "mnist_lr": {
                        "epochs": 45,
                        "batch_size": 128,
                        "per_optimizer": {"adam": {"lr": 0.001}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    ev = _gather_evidence(run)
    assert "provenance.json" in ev
    # epochs + batch size are now CONFIRMED evidence, not "an assumption".
    assert "45" in ev and "128" in ev


def test_gather_evidence_includes_figure_sidecars(tmp_path: Path):
    run, out = _mk_run(tmp_path)
    (out / "fig_mnist_lr.png").write_bytes(b"\x89PNG fake-bytes")
    (out / "fig_mnist_lr.json").write_text(
        json.dumps(
            {
                "shows": "NLL vs iteration",
                "axis": {
                    "x": {"label": "iter", "scale": "log"},
                    "y": {"label": "nll", "scale": "linear"},
                },
            }
        ),
        encoding="utf-8",
    )
    ev = _gather_evidence(run)
    assert "fig_mnist_lr.json" in ev
    # the log-scale axis the text-only grader can never read off the PNG itself.
    assert "log" in ev


def test_provenance_paths_finds_both_levels(tmp_path: Path):
    run, out = _mk_run(tmp_path)
    (run / "code" / "provenance.json").write_text("{}", encoding="utf-8")
    (out / "provenance.json").write_text("{}", encoding="utf-8")
    paths = _provenance_paths(run)
    assert len(paths) == 2  # code/ root (monolithic) + outputs/<id>/ (promoted)


def test_missing_manifest_is_fail_soft(tmp_path: Path):
    run, _ = _mk_run(tmp_path)
    # No provenance.json / sidecars present — must not raise, must still return code.
    assert _provenance_paths(run) == []
    assert _gather_figure_sidecars(run) == ""
    ev = _gather_evidence(run)
    assert "train.py" in ev


def test_code_no_longer_truncated_at_6kb(tmp_path: Path):
    run, _ = _mk_run(tmp_path)
    big = "# header\n" + ("y = 1\n" * 3000)  # ~18 KB, well over the old 6 KB/file cap
    (run / "code" / "big.py").write_text(big, encoding="utf-8")
    ev = _gather_evidence(run)
    assert _MAX_FILE_BYTES >= 32 * 1024
    # the whole 18 KB file survives the new 32 KB/file cap (old cap kept ~1000 lines).
    assert ev.count("y = 1") > 2500
