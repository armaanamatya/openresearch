"""Tests for the 2026-06-08 execution-reliability redesign — Pillar 3 (finalize-on-
timeout) and Pillar 5 (honest disk attribution) in ``backend.agents.rlm.primitives``.

Three pure(ish) helpers are pinned here:

  * ``_finalize_timeout_result`` — loads the newest results-bearing ``metrics.json``
    from ``ctx.project_dir`` (via the leaf scorer's ``_latest_metrics_path``); if
    >= 1 family carries a measured value it attaches the metrics and flags a
    repairable ``partial_timeout`` instead of zeroing completed work; an empty
    placeholder (or no file) stays the empty-fail tagged ``exec_timeout``/``exec_stalled``.
  * ``_dir_footprint_gb`` — approximate on-disk footprint, skipping the harness
    per-run ``.venv`` (and ``__pycache__``/``.git``).
  * ``_disk_floor_violation`` — fires ``disk_exhausted`` below ``REPROLAB_DISK_FLOOR_GB``,
    but attributes a small-run-on-a-full-volume to OTHER runs' caches (GC advice).

Style mirrors tests/rlm/test_run_experiment_timeout.py: sync test functions,
``tmp_path``/``monkeypatch``, ctx built from a lightweight namespace.  ``_latest_metrics_path``
resolves ``<project_dir>/code/outputs/**/metrics.json`` + ``<project_dir>/code/metrics.json``,
so the fixtures write metrics under ``tmp_path/code/outputs/run1/``.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

import backend.agents.rlm.primitives as P


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ctx(tmp_path: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(project_dir=tmp_path)


def _write_metrics(tmp_path: Path, payload: dict, run_id: str = "run1") -> Path:
    """Write ``tmp_path/code/outputs/<run_id>/metrics.json`` — the layout
    ``_latest_metrics_path`` (and therefore ``_finalize_timeout_result``) scans."""
    out = tmp_path / "code" / "outputs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "metrics.json"
    mpath.write_text(json.dumps(payload), encoding="utf-8")
    return mpath


# ---------------------------------------------------------------------------
# _finalize_timeout_result
# ---------------------------------------------------------------------------


def test_finalize_populated_partial_is_preserved_and_scored(tmp_path: Path) -> None:
    """A status:running metrics.json with measured per_model values → partial_timeout
    (repairable), metrics attached, success False, error names the partial."""
    _write_metrics(
        tmp_path,
        {
            "status": "running",
            "per_model": {
                "mnist_mlp": {"test_error": 1.69},
                "logreg": {"nll": 0.2395},
            },
        },
    )
    result_in = {
        "success": False,
        "metrics": {},
        "cause_kind": "exec_timeout",
    }
    out = P._finalize_timeout_result(
        _ctx(tmp_path),
        str(tmp_path / "code"),
        "run1",
        result_in,
        reason="exec_timeout",
    )

    assert out["failure_class"] == "partial_timeout"
    assert out["partial_timeout"] is True
    assert out["success"] is False
    # The completed families' metrics survived and were attached.
    per_model = out["metrics"]["per_model"]
    assert "mnist_mlp" in per_model
    assert "logreg" in per_model
    assert per_model["mnist_mlp"]["test_error"] == 1.69
    # The repair_context-style error names the partial.
    assert out["error"].startswith("partial_timeout")


def test_finalize_empty_placeholder_stays_empty_fail(tmp_path: Path) -> None:
    """A metrics.json with an empty per_model placeholder → NOT a partial; keeps the
    empty-fail, tagged from cause_kind (exec_stalled here). Metrics stay empty."""
    _write_metrics(tmp_path, {"status": "running", "per_model": {}})
    result_in = {
        "success": False,
        "metrics": {},
        "cause_kind": "exec_stalled",
    }
    out = P._finalize_timeout_result(
        _ctx(tmp_path),
        str(tmp_path / "code"),
        "run1",
        result_in,
        reason="exec_stalled",
    )

    assert out["failure_class"] in {"exec_timeout", "exec_stalled"}
    # cause_kind contained "stall" → exec_stalled specifically.
    assert out["failure_class"] == "exec_stalled"
    # Empty placeholder is not a recoverable partial.
    assert not out.get("partial_timeout")
    # The empty-fail branch does not swap in the (still-empty) on-disk metrics.
    assert out["metrics"] == {}


def test_finalize_no_metrics_file_at_all_returns_empty_fail(tmp_path: Path) -> None:
    """No metrics.json anywhere under project_dir → empty-fail, classified, never raises."""
    result_in = {
        "success": False,
        "metrics": {},
        "cause_kind": "exec_timeout",
    }
    out = P._finalize_timeout_result(
        _ctx(tmp_path),
        str(tmp_path / "code"),
        "run1",
        result_in,
        reason="exec_timeout",
    )

    assert out["failure_class"] in {"exec_timeout", "exec_stalled"}
    assert out["failure_class"] == "exec_timeout"  # cause_kind has no "stall"
    assert not out.get("partial_timeout")
    assert out["metrics"] == {}


# ---------------------------------------------------------------------------
# _dir_footprint_gb
# ---------------------------------------------------------------------------


def test_dir_footprint_skips_venv_and_counts_real_files(tmp_path: Path) -> None:
    """_dir_footprint_gb skips a .venv subdir but counts a sibling real file."""
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)

    # A few small files → ~0.0 GB (rounds to 0.0 at 2 decimals).
    (code / "metrics.json").write_text("{}", encoding="utf-8")
    (code / "train.py").write_text("print('x')\n", encoding="utf-8")
    assert P._dir_footprint_gb(code) == pytest.approx(0.0, abs=1e-9)

    # A large file INSIDE .venv must be skipped — footprint stays ~0.0 even though
    # the byte count, if counted, would round above zero.
    venv_big = code / ".venv" / "big"
    venv_big.parent.mkdir(parents=True, exist_ok=True)
    venv_big.write_bytes(b"\0" * (20 * 1024 * 1024))  # 20 MB
    assert P._dir_footprint_gb(code) == pytest.approx(0.0, abs=1e-9), (
        "the .venv subtree must not contribute to the footprint"
    )

    # A sibling real file (outside .venv) IS counted — confirms counting works and
    # the .venv content is still excluded (result reflects only the sibling).
    (code / "real.bin").write_bytes(b"\0" * (20 * 1024 * 1024))  # 20 MB ≈ 0.02 GB
    footprint = P._dir_footprint_gb(code)
    assert footprint == pytest.approx(0.02, abs=5e-3), (
        f"only the 20 MB sibling should count (not the 20 MB .venv/big); got {footprint}"
    )


# ---------------------------------------------------------------------------
# _disk_floor_violation
# ---------------------------------------------------------------------------


def test_disk_floor_small_run_on_full_volume_blames_other_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the volume is below the floor but THIS run's footprint is tiny, the
    message attributes it to OTHER runs' caches (GC advice), not this run."""
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "metrics.json").write_text("{}", encoding="utf-8")  # tiny footprint

    # _disk_floor_violation does a local ``import shutil``; patch the real module
    # attribute so the bound name resolves to our fake (only 2 GB free).
    monkeypatch.setattr(
        "shutil.disk_usage",
        lambda p: types.SimpleNamespace(
            total=11 * 10**12, used=11 * 10**12 - 2 * 10**9, free=2 * 10**9
        ),
    )
    monkeypatch.setenv("REPROLAB_DISK_FLOOR_GB", "15")

    out = P._disk_floor_violation([str(tmp_path)])
    assert out is not None
    klass, msg = out
    assert klass == "disk_exhausted"
    assert "OTHER runs" in msg
    assert "rm -rf runs/.cache" in msg


def test_disk_floor_above_floor_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Plenty of free space (500 GB) → no violation."""
    monkeypatch.setattr(
        "shutil.disk_usage",
        lambda p: types.SimpleNamespace(
            total=11 * 10**12, used=10 * 10**12, free=500 * 10**9
        ),
    )
    monkeypatch.setenv("REPROLAB_DISK_FLOOR_GB", "15")
    assert P._disk_floor_violation([str(tmp_path)]) is None


def test_disk_floor_disabled_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REPROLAB_DISK_FLOOR_GB=0 disables the check regardless of free space."""
    monkeypatch.setattr(
        "shutil.disk_usage",
        lambda p: types.SimpleNamespace(
            total=11 * 10**12, used=11 * 10**12 - 1 * 10**9, free=1 * 10**9
        ),
    )
    monkeypatch.setenv("REPROLAB_DISK_FLOOR_GB", "0")
    assert P._disk_floor_violation([str(tmp_path)]) is None
