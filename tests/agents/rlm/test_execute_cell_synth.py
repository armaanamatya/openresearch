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


# --- OPENRESEARCH_TRAIN_CHECKPOINT_STEPS: mid-training checkpoint/resume injection ----

def test_checkpoint_flag_off_artifacts_byte_identical(tmp_path, monkeypatch):
    """Flag unset AND flag empty/whitespace -> byte-identical artifacts with the
    pre-flag baseline: save_freq stays the -1 downscale default and no
    resume_mode appears anywhere in the spec or the shim."""
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    monkeypatch.delenv("OPENRESEARCH_TRAIN_CHECKPOINT_STEPS", raising=False)
    code_dir = tmp_path / "off"
    _write_verl_fixture(code_dir)
    assert execute_cell_synth.maybe_synthesize_execute_cells(code_dir) is not None

    spec_off = (code_dir / "execute_spec.json").read_text(encoding="utf-8")
    shim_off = (code_dir / "train_cell.py").read_text(encoding="utf-8")
    cells_off = (code_dir / "cells.json").read_text(encoding="utf-8")

    overrides = json.loads(spec_off)["launch"]["overrides"]
    assert overrides["trainer.save_freq"] == "-1"
    assert "trainer.resume_mode" not in overrides
    assert "resume_mode" not in spec_off
    assert "resume_mode" not in shim_off

    # empty/whitespace value is OFF too -> byte-identical artifacts
    monkeypatch.setenv("OPENRESEARCH_TRAIN_CHECKPOINT_STEPS", "   ")
    code_dir2 = tmp_path / "empty"
    _write_verl_fixture(code_dir2)
    assert execute_cell_synth.maybe_synthesize_execute_cells(code_dir2) is not None
    assert (code_dir2 / "execute_spec.json").read_text(encoding="utf-8") == spec_off
    assert (code_dir2 / "train_cell.py").read_text(encoding="utf-8") == shim_off
    assert (code_dir2 / "cells.json").read_text(encoding="utf-8") == cells_off


def test_checkpoint_flag_on_appends_both_overrides_exactly_once(tmp_path, monkeypatch):
    """Flag on -> trainer.save_freq=<N> replaces the -1 default and
    trainer.resume_mode=auto is added, each EXACTLY once; every other override
    (esp. data.max_response_length, which must never be touched) is unchanged."""
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    monkeypatch.setenv("OPENRESEARCH_TRAIN_CHECKPOINT_STEPS", "50")
    code_dir = tmp_path / "code"
    _write_verl_fixture(code_dir)
    assert execute_cell_synth.maybe_synthesize_execute_cells(code_dir) is not None

    spec_json = json.loads((code_dir / "execute_spec.json").read_text(encoding="utf-8"))
    overrides = spec_json["launch"]["overrides"]
    assert overrides["trainer.save_freq"] == "50"
    assert overrides["trainer.resume_mode"] == "auto"

    # exactly once each in the embedded shim spec (dict keys guarantee once in
    # the overrides; the shim carries the spec JSON a single time).
    shim_text = (code_dir / "train_cell.py").read_text(encoding="utf-8")
    assert shim_text.count("trainer.resume_mode") == 1
    assert shim_text.count("trainer.save_freq=-1") == 0

    # execute_planner invariant: the authors' data.max_response_length survives
    # verbatim in the command and is never an override.
    assert "data.max_response_length=3072" in spec_json["launch"]["command"]
    assert "data.max_response_length" not in overrides

    # only the two checkpoint keys differ from the flag-off override set
    monkeypatch.delenv("OPENRESEARCH_TRAIN_CHECKPOINT_STEPS")
    baseline_dir = tmp_path / "baseline"
    _write_verl_fixture(baseline_dir)
    assert execute_cell_synth.maybe_synthesize_execute_cells(baseline_dir) is not None
    baseline_overrides = json.loads(
        (baseline_dir / "execute_spec.json").read_text(encoding="utf-8")
    )["launch"]["overrides"]
    assert {
        k: v for k, v in overrides.items() if k not in ("trainer.save_freq", "trainer.resume_mode")
    } == {k: v for k, v in baseline_overrides.items() if k != "trainer.save_freq"}

    py_compile.compile(str(code_dir / "train_cell.py"), doraise=True)


def test_checkpoint_flag_invalid_or_nonpositive_stays_off(tmp_path, monkeypatch):
    """A non-integer or non-positive value is treated as OFF (fail-safe,
    byte-identical) -- never a crash, never a partial injection."""
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    monkeypatch.delenv("OPENRESEARCH_TRAIN_CHECKPOINT_STEPS", raising=False)
    baseline_dir = tmp_path / "baseline"
    _write_verl_fixture(baseline_dir)
    assert execute_cell_synth.maybe_synthesize_execute_cells(baseline_dir) is not None
    spec_baseline = (baseline_dir / "execute_spec.json").read_text(encoding="utf-8")

    for i, bad in enumerate(("abc", "0", "-5")):
        monkeypatch.setenv("OPENRESEARCH_TRAIN_CHECKPOINT_STEPS", bad)
        code_dir = tmp_path / f"bad{i}"
        _write_verl_fixture(code_dir)
        assert execute_cell_synth.maybe_synthesize_execute_cells(code_dir) is not None
        assert (
            code_dir / "execute_spec.json"
        ).read_text(encoding="utf-8") == spec_baseline


# --- Option A: held-out eval wiring surfaces in the synthesized cell -------------------

def test_synth_shim_carries_dual_key_bridge_and_val_slice(tmp_path, monkeypatch):
    """The generated shim prefers the HELD-OUT eval key (metric_name=success_rate)
    and falls back to the TRAIN reward (metric_name=reward_mean), and slices the
    val set when the repo ships one — so the eval-provenance guard credits a real
    held-out score instead of vetoing a mislabeled train reward."""
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    code_dir = tmp_path / "code"
    _write_verl_fixture(code_dir)
    # give the fixture a real val parquet + a data.val_files reference so the
    # planner enables held-out validation.
    (code_dir / "scripts" / "run.sh").write_text(
        "#!/bin/bash\n"
        "python3 -m demo.main_run \\\n"
        "    algorithm.adv_estimator=grpo \\\n"
        "    data.max_response_length=3072 \\\n"
        "    data.train_files=dataset/train.parquet \\\n"
        "    data.val_files=dataset/valid.parquet \\\n"
        "    trainer.test_freq=10 \\\n"
        "    trainer.n_gpus_per_node=4\n",
        encoding="utf-8",
    )
    (code_dir / "dataset" / "valid.parquet").write_bytes(b"")

    result = execute_cell_synth.maybe_synthesize_execute_cells(code_dir)
    assert isinstance(result, dict)

    spec_json = json.loads((code_dir / "execute_spec.json").read_text(encoding="utf-8"))
    # held-out eval keys offered + validation enabled + val slice present
    assert spec_json["reward"]["eval_keys"], "eval_keys should be populated when a val set exists"
    assert spec_json["launch"]["overrides"]["trainer.test_freq"] != "-1"
    assert spec_json["data_slice"]["val_file"] == "dataset/valid.parquet"

    shim_text = (code_dir / "train_cell.py").read_text(encoding="utf-8")
    # dual-key bridge markers
    assert 'metric_name="success_rate"' in shim_text
    assert 'metric_name="reward_mean"' in shim_text
    # val-slice logic present
    assert "val_slice" in shim_text
    assert "data.val_files=" in shim_text
    py_compile.compile(str(code_dir / "train_cell.py"), doraise=True)
