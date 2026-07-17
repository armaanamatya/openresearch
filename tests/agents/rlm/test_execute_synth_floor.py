"""OPENRESEARCH_EXECUTE_SYNTH wired into implement_baseline (E4 — the execute
floor). Covers: `primitives._maybe_execute_synth_floor` is the single call site
that lets a confidently-detected known-framework (verl) execute-mode repo be
RUN via the authors' own launch — deterministically, with NO LLM implementer
call — while every gate (flag off, mode != execute, a repair pass, a
non-confident framework) falls through to the SDK path unchanged.

Fixture mirrors `test_execute_cell_synth.py::_write_verl_fixture` verbatim so
the real planner/detector (not a stub) drives every case.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

from backend.agents.rlm import primitives as P


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


def _make_ctx(tmp_path: Path, *, mode: str = "execute") -> types.SimpleNamespace:
    """A minimal ctx: project_dir/rlm_state/repo_spec.json is what
    `_load_repo_spec` reads (primitives.py:2157-2168)."""
    project_dir = tmp_path / "prj_test"
    rlm_state = project_dir / "rlm_state"
    rlm_state.mkdir(parents=True)
    (rlm_state / "repo_spec.json").write_text(
        json.dumps({"mode": mode}), encoding="utf-8"
    )
    return types.SimpleNamespace(
        project_dir=project_dir, project_id="prj_test", runs_root=project_dir.parent, emit=None
    )


def _synth_files_absent(code_dir: Path) -> bool:
    return not (
        (code_dir / "cells.json").exists()
        or (code_dir / "train_cell.py").exists()
        or (code_dir / "execute_spec.json").exists()
    )


# --- 1. off (flag unset) ------------------------------------------------------------

def test_flag_off_returns_none_byte_identical(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EXECUTE_SYNTH", raising=False)
    ctx = _make_ctx(tmp_path, mode="execute")
    code_dir = ctx.project_dir / "code"
    _write_verl_fixture(code_dir)

    result = P._maybe_execute_synth_floor(ctx, code_dir, repair_context=None)

    assert result is None
    assert _synth_files_absent(code_dir)


# --- 2. mode gate (adapt) ------------------------------------------------------------

def test_mode_adapt_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    ctx = _make_ctx(tmp_path, mode="adapt")
    code_dir = ctx.project_dir / "code"
    _write_verl_fixture(code_dir)

    result = P._maybe_execute_synth_floor(ctx, code_dir, repair_context=None)

    assert result is None
    assert _synth_files_absent(code_dir)


# --- 3. repair fall-through ----------------------------------------------------------

def test_repair_context_falls_through(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    ctx = _make_ctx(tmp_path, mode="execute")
    code_dir = ctx.project_dir / "code"
    _write_verl_fixture(code_dir)

    result = P._maybe_execute_synth_floor(ctx, code_dir, repair_context={"error": "x"})

    assert result is None
    assert _synth_files_absent(code_dir)


# --- 4. first call: synthesis fires + ok envelope ------------------------------------

def test_first_call_synthesizes_and_returns_ok_envelope(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    ctx = _make_ctx(tmp_path, mode="execute")
    code_dir = ctx.project_dir / "code"
    _write_verl_fixture(code_dir)

    result = P._maybe_execute_synth_floor(ctx, code_dir, repair_context=None)

    assert result is not None
    assert result["ok"] is True
    assert result["execute_synth"] is True
    assert result["framework"] == "verl"
    assert (code_dir / "cells.json").is_file()
    assert (code_dir / "train_cell.py").is_file()
    assert (code_dir / "execute_spec.json").is_file()


# --- 5. warm-retry idempotence --------------------------------------------------------

def test_second_call_recognizes_existing_synth_and_stays_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    ctx = _make_ctx(tmp_path, mode="execute")
    code_dir = ctx.project_dir / "code"
    _write_verl_fixture(code_dir)

    first = P._maybe_execute_synth_floor(ctx, code_dir, repair_context=None)
    assert first is not None

    second = P._maybe_execute_synth_floor(ctx, code_dir, repair_context=None)

    assert second is not None
    assert second["ok"] is True
    assert second["execute_synth"] is True
    assert second["framework"] == "verl"


# --- 6. non-verl repo: no confident framework -----------------------------------------

def test_non_verl_repo_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    ctx = _make_ctx(tmp_path, mode="execute")
    code_dir = ctx.project_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "train.py").write_text("print('hi')\n", encoding="utf-8")

    result = P._maybe_execute_synth_floor(ctx, code_dir, repair_context=None)

    assert result is None
    assert not (code_dir / "execute_spec.json").exists()


# --- Codex P0: orphaned/non-synth marker must NOT skip the SDK -------------------------

def test_orphaned_spec_marker_without_cells_falls_through(tmp_path, monkeypatch):
    # A bare execute_spec.json with no cells.json/train_cell.py is NOT a complete
    # synth-owned cell — the floor must fall through to the SDK, not vouch for it.
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    ctx = _make_ctx(tmp_path, mode="execute")
    code_dir = ctx.project_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "execute_spec.json").write_text('{"framework": "verl"}', encoding="utf-8")

    assert P._maybe_execute_synth_floor(ctx, code_dir, repair_context=None) is None


def test_non_synth_cells_with_spec_marker_falls_through(tmp_path, monkeypatch):
    # A NON-synth cells.json (missing the synth _comment marker) beside a stale
    # execute_spec.json + train_cell.py must fall through to the SDK — E4 may never
    # skip the SDK on a manifest it did not write (idempotence red line).
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "1")
    ctx = _make_ctx(tmp_path, mode="execute")
    code_dir = ctx.project_dir / "code"
    code_dir.mkdir(parents=True)
    # a hand-authored manifest — NO synth marker
    (code_dir / "cells.json").write_text('{"cells": [{"id": "hand"}]}', encoding="utf-8")
    (code_dir / "train_cell.py").write_text("# hand-authored\n", encoding="utf-8")
    (code_dir / "execute_spec.json").write_text('{"framework": "verl"}', encoding="utf-8")

    assert P._maybe_execute_synth_floor(ctx, code_dir, repair_context=None) is None


def test_is_synth_owned_requires_complete_marked_set(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    # only the marker file
    (code_dir / "execute_spec.json").write_text("{}", encoding="utf-8")
    assert P._is_synth_owned_execute_cell(code_dir) is False
    # add train_cell.py + a MARKED cells.json + spec -> synth-owned
    (code_dir / "train_cell.py").write_text("x\n", encoding="utf-8")
    (code_dir / "cells.json").write_text(
        '{"_comment": "synthesized by OPENRESEARCH_EXECUTE_SYNTH", "cells": []}',
        encoding="utf-8",
    )
    assert P._is_synth_owned_execute_cell(code_dir) is True
    # a non-synth manifest (no marker) -> not owned
    (code_dir / "cells.json").write_text('{"cells": []}', encoding="utf-8")
    assert P._is_synth_owned_execute_cell(code_dir) is False


# --- Codex P1: the "on" truthy token must fire the floor (gate consistency) ------------

def test_flag_on_token_synthesizes(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_SYNTH", "on")
    ctx = _make_ctx(tmp_path, mode="execute")
    code_dir = ctx.project_dir / "code"
    _write_verl_fixture(code_dir)

    result = P._maybe_execute_synth_floor(ctx, code_dir, repair_context=None)

    assert result is not None and result["ok"] is True and result["execute_synth"] is True
