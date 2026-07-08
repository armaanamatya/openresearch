"""OPENRESEARCH_GKE_SYNTH_CELL — synthesizing a single-cell manifest so a monolithic
``commands.json`` project routes through the validated GKE cell-matrix instead of the
monolithic exec path (which never downloads the GCS-uploaded code into a gcp pod).

Covers: synthesis fires under the exact scope (gcp + flag on + commands-only dir),
every no-op branch returns None and writes nothing, default-OFF is byte-identical, and
the generated ``train_cell.py`` shim actually bridges metrics.json to the
entrypoint-provided output dir without ever fabricating one.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from backend.agents.rlm import gke_cell_synth


def _write_commands(code_dir: Path, commands: list) -> None:
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "commands.json").write_text(json.dumps(commands), encoding="utf-8")


def _assert_untouched(code_dir: Path) -> None:
    assert not (code_dir / "cells.json").exists()
    assert not (code_dir / "train_cell.py").exists()


# --- synthesis fires ---------------------------------------------------------------

def test_synthesis_fires_on_gcp_with_flag_on(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GKE_SYNTH_CELL", "1")
    code = tmp_path / "code"
    _write_commands(code, ["python train.py"])

    result = gke_cell_synth.maybe_synthesize_gke_cell(str(code), "gcp")

    assert isinstance(result, dict)
    cells_path = code / "cells.json"
    train_cell_path = code / "train_cell.py"
    assert cells_path.is_file()
    assert train_cell_path.is_file()

    manifest = json.loads(cells_path.read_text(encoding="utf-8"))
    cells = manifest.get("cells")
    assert isinstance(cells, list) and len(cells) >= 1
    for cell in cells:
        assert isinstance(cell, dict)
        assert cell.get("id")

    shim_text = train_cell_path.read_text(encoding="utf-8")
    assert repr("python train.py") in shim_text or "'python train.py'" in shim_text


# --- no-ops: each returns None and writes nothing ------------------------------------

def test_noop_flag_off(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_GKE_SYNTH_CELL", raising=False)
    code = tmp_path / "code"
    _write_commands(code, ["python train.py"])

    result = gke_cell_synth.maybe_synthesize_gke_cell(str(code), "gcp")

    assert result is None
    _assert_untouched(code)


@pytest.mark.parametrize("backend_kind", ["local", "docker", "runpod", "azure"])
def test_noop_non_gcp_backend(tmp_path, monkeypatch, backend_kind):
    monkeypatch.setenv("OPENRESEARCH_GKE_SYNTH_CELL", "1")
    code = tmp_path / "code"
    _write_commands(code, ["python train.py"])

    result = gke_cell_synth.maybe_synthesize_gke_cell(str(code), backend_kind)

    assert result is None
    _assert_untouched(code)


def test_noop_cells_json_already_present(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GKE_SYNTH_CELL", "1")
    code = tmp_path / "code"
    _write_commands(code, ["python train.py"])
    existing = json.dumps({"cells": [{"id": "hand_written"}]})
    (code / "cells.json").write_text(existing, encoding="utf-8")

    result = gke_cell_synth.maybe_synthesize_gke_cell(str(code), "gcp")

    assert result is None
    # The existing manifest must survive untouched (never clobbered).
    assert (code / "cells.json").read_text(encoding="utf-8") == existing
    assert not (code / "train_cell.py").exists()


def test_noop_train_cell_already_present(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GKE_SYNTH_CELL", "1")
    code = tmp_path / "code"
    _write_commands(code, ["python train.py"])
    existing = "# hand-written trainer\n"
    (code / "train_cell.py").write_text(existing, encoding="utf-8")

    result = gke_cell_synth.maybe_synthesize_gke_cell(str(code), "gcp")

    assert result is None
    assert (code / "train_cell.py").read_text(encoding="utf-8") == existing
    assert not (code / "cells.json").exists()


def test_noop_commands_json_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GKE_SYNTH_CELL", "1")
    code = tmp_path / "code"
    code.mkdir(parents=True)
    # no commands.json written

    result = gke_cell_synth.maybe_synthesize_gke_cell(str(code), "gcp")

    assert result is None
    _assert_untouched(code)


def test_noop_commands_json_empty_list(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GKE_SYNTH_CELL", "1")
    code = tmp_path / "code"
    _write_commands(code, [])

    result = gke_cell_synth.maybe_synthesize_gke_cell(str(code), "gcp")

    assert result is None
    _assert_untouched(code)


# --- off-state byte-identical --------------------------------------------------------

def test_off_state_leaves_gcp_commands_only_dir_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_GKE_SYNTH_CELL", raising=False)
    code = tmp_path / "code"
    _write_commands(code, ["python train.py", "python eval.py"])
    before = sorted(p.name for p in code.iterdir())

    result = gke_cell_synth.maybe_synthesize_gke_cell(str(code), "gcp")

    assert result is None
    after = sorted(p.name for p in code.iterdir())
    assert before == after
    assert (code / "commands.json").read_text(encoding="utf-8") == json.dumps(
        ["python train.py", "python eval.py"]
    )


# --- shim behavior: metrics bridging (evidence-not-grade invariant) ------------------

def _run_shim(code_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    train_cell = code_dir / "train_cell.py"
    out_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, str(train_cell), "--output-dir", str(out_dir)],
        cwd=str(code_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_shim_bridges_metrics_written_into_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GKE_SYNTH_CELL", "1")
    code = tmp_path / "code"
    _write_commands(code, [f"{sys.executable} inner.py"])
    (code / "inner.py").write_text(
        "import json\n"
        "with open('metrics.json', 'w') as f:\n"
        "    json.dump({'metric': 0.42}, f)\n",
        encoding="utf-8",
    )
    result = gke_cell_synth.maybe_synthesize_gke_cell(str(code), "gcp")
    assert result is not None

    out_dir = tmp_path / "cell_out_cwd"
    proc = _run_shim(code, out_dir)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    dest = out_dir / "metrics.json"
    assert dest.is_file()
    assert json.loads(dest.read_text(encoding="utf-8")) == {"metric": 0.42}


def test_shim_finds_metrics_written_directly_to_output_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GKE_SYNTH_CELL", "1")
    code = tmp_path / "code"
    _write_commands(code, [f"{sys.executable} inner.py"])
    (code / "inner.py").write_text(
        "import json\n"
        "import os\n"
        "out = os.environ['OUTPUT_DIR']\n"
        "with open(os.path.join(out, 'metrics.json'), 'w') as f:\n"
        "    json.dump({'metric': 0.77}, f)\n",
        encoding="utf-8",
    )
    result = gke_cell_synth.maybe_synthesize_gke_cell(str(code), "gcp")
    assert result is not None

    out_dir = tmp_path / "cell_out_env"
    proc = _run_shim(code, out_dir)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    dest = out_dir / "metrics.json"
    assert dest.is_file()
    assert json.loads(dest.read_text(encoding="utf-8")) == {"metric": 0.77}


def test_shim_never_fabricates_metrics_when_inner_writes_none(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GKE_SYNTH_CELL", "1")
    code = tmp_path / "code"
    _write_commands(code, [f"{sys.executable} inner.py"])
    (code / "inner.py").write_text("print('no metrics written')\n", encoding="utf-8")
    result = gke_cell_synth.maybe_synthesize_gke_cell(str(code), "gcp")
    assert result is not None

    out_dir = tmp_path / "cell_out_none"
    proc = _run_shim(code, out_dir)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    dest = out_dir / "metrics.json"
    assert not dest.is_file()  # honest no-metrics — never fabricated
