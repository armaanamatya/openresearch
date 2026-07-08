"""OPENRESEARCH_EXECUTE_SYNTH — synthesizing a deterministic execute-mode cell
(cells.json + train_cell.py + execute_spec.json + a verl_metrics_adapter.py
copy) from a planned ExecuteSpec, mirroring gke_cell_synth's style.

Covers: synthesis fires under the exact scope (flag on + a confidently
verl-planned repo), every no-op branch returns None and writes nothing,
default-OFF is byte-identical, an existing manifest is never clobbered, and
the generated train_cell.py shim is syntactically valid Python.
"""
from __future__ import annotations

import json
import py_compile
from pathlib import Path

from backend.agents.rlm import execute_cell_synth


def _write_verl_fixture(code_dir: Path) -> None:
    verl_dir = code_dir / "verl"
    verl_dir.mkdir(parents=True)
    (verl_dir / "setup.py").write_text(
        "from setuptools import setup\nsetup(name='verl')\n", encoding="utf-8"
    )

    scripts_dir = code_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "run.sh").write_text(
        "#!/bin/bash\n"
        "python3 -m demo.main_run \\\n"
        "    algorithm.adv_estimator=grpo \\\n"
        "    actor_rollout_ref.rollout.n=8 \\\n"
        "    data.max_response_length=3072 \\\n"
        "    data.train_files=dataset/train.parquet \\\n"
        "    trainer.n_gpus_per_node=4 \\\n"
        "    trainer.logger=['console','tensorboard']\n",
        encoding="utf-8",
    )

    demo_dir = code_dir / "demo"
    demo_dir.mkdir(parents=True)
    (demo_dir / "main_run.py").write_text("# stub entrypoint\n", encoding="utf-8")

    dataset_dir = code_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.parquet").write_bytes(b"")


def _all_files(code_dir: Path) -> list[str]:
    return sorted(p.relative_to(code_dir).as_posix() for p in code_dir.rglob("*") if p.is_file())


def _assert_untouched(code_dir: Path, before: list[str]) -> None:
    assert not (code_dir / "cells.json").exists()
    assert not (code_dir / "train_cell.py").exists()
    assert not (code_dir / "execute_spec.json").exists()
    assert not (code_dir / "verl_metrics_adapter.py").exists()
    assert _all_files(code_dir) == before


# --- off-state byte-identical --------------------------------------------------------

def test_noop_flag_off(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EXECUTE_SYNTH", raising=False)
    code_dir = tmp_path / "code"
    _write_verl_fixture(code_dir)
    before = _all_files(code_dir)

    result = execute_cell_synth.maybe_synthesize_execute_cells(code_dir)

    assert result is None
    _assert_untouched(code_dir, before)


# --- synthesis fires -----------------------------------------------------------------

def test_synthesis_fires_on_verl_fixture_with_flag_on(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    code_dir = tmp_path / "code"
    _write_verl_fixture(code_dir)

    result = execute_cell_synth.maybe_synthesize_execute_cells(code_dir)

    assert isinstance(result, dict)
    assert result["framework"] == "verl"
    assert result["image_key"] == "verl"

    cells_path = code_dir / "cells.json"
    train_cell_path = code_dir / "train_cell.py"
    spec_path = code_dir / "execute_spec.json"
    adapter_path = code_dir / "verl_metrics_adapter.py"
    assert cells_path.is_file()
    assert train_cell_path.is_file()
    assert spec_path.is_file()
    assert adapter_path.is_file()

    manifest = json.loads(cells_path.read_text(encoding="utf-8"))
    cells = manifest.get("cells")
    assert isinstance(cells, list) and len(cells) == 1
    cell = cells[0]
    assert cell["framework"] == "verl"
    assert cell["image_key"] == "verl"
    assert cell["id"]

    spec_json = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec_json["framework"] == "verl"
    assert "demo.main_run" in spec_json["launch"]["command"]

    # the shim byte-compiles (syntactically valid, even though it isn't run here)
    py_compile.compile(str(train_cell_path), doraise=True)

    # the embedded spec JSON round-trips inside the generated shim's own source
    shim_text = train_cell_path.read_text(encoding="utf-8")
    assert "SPEC_JSON = " in shim_text
    assert "verl_metrics_adapter" in shim_text

    # verl_metrics_adapter.py copy matches the source module byte-for-byte
    from backend.agents.rlm import verl_metrics_adapter as _adapter_module

    adapter_src = Path(_adapter_module.__file__).read_text(encoding="utf-8")
    assert adapter_path.read_text(encoding="utf-8") == adapter_src


# --- no-ops: each returns None and writes nothing -------------------------------------

def test_noop_cells_json_already_present(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    code_dir = tmp_path / "code"
    _write_verl_fixture(code_dir)
    existing = json.dumps({"cells": [{"id": "hand_written"}]})
    (code_dir / "cells.json").write_text(existing, encoding="utf-8")

    result = execute_cell_synth.maybe_synthesize_execute_cells(code_dir)

    assert result is None
    # the existing manifest must survive untouched (never clobbered)
    assert (code_dir / "cells.json").read_text(encoding="utf-8") == existing
    assert not (code_dir / "train_cell.py").exists()
    assert not (code_dir / "execute_spec.json").exists()


def test_noop_train_cell_already_present(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    code_dir = tmp_path / "code"
    _write_verl_fixture(code_dir)
    existing = "# hand-written trainer\n"
    (code_dir / "train_cell.py").write_text(existing, encoding="utf-8")

    result = execute_cell_synth.maybe_synthesize_execute_cells(code_dir)

    assert result is None
    assert (code_dir / "train_cell.py").read_text(encoding="utf-8") == existing
    assert not (code_dir / "cells.json").exists()


def test_noop_non_verl_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    code_dir = tmp_path / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "train.py").write_text("print('hi')\n", encoding="utf-8")
    before = _all_files(code_dir)

    result = execute_cell_synth.maybe_synthesize_execute_cells(code_dir)

    assert result is None
    _assert_untouched(code_dir, before)


# --- shim hermeticity: checkpoint dir pinned under out_dir ---------------------------

def test_shim_pins_checkpoint_dir_under_out_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    code_dir = tmp_path / "code"
    _write_verl_fixture(code_dir)

    result = execute_cell_synth.maybe_synthesize_execute_cells(code_dir)
    assert result is not None

    shim = (code_dir / "train_cell.py").read_text(encoding="utf-8")
    # the generated shim rewrites/appends trainer.default_local_dir to an
    # out_dir-relative path (never the authors' ../checkpoint_ds/... escape).
    assert 'trainer.default_local_dir=' in shim
    assert 'ckpt_dir = out_dir / "checkpoint"' in shim
    assert 'f"trainer.default_local_dir={ckpt_dir}"' in shim
