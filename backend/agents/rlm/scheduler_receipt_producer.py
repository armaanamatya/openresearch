"""Harness-owned producer of authority receipts from a completed LOCAL cell.

This module lives OUTSIDE gpu_cell_runner (which is stdlib-only, copied into the
agent sandbox). It reads the deterministic on-disk evidence a cell produced and
assembles the exact ``raw_receipt`` mapping ``scheduler_evidence.write_verified_receipt``
expects. It NEVER reads an LLM grade. The 5-field checkpoint is materialized here
(the harness owns the hashing + manifest); the trainer only produces raw component
bytes.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

_CHECKPOINT_COMPONENTS = ("model", "optimizer", "lr_scheduler", "rng", "data_order")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path, run_dir: Path) -> str:
    return str(path.resolve().relative_to(run_dir.resolve()))


def materialize_checkpoint(
    *, run_dir: Path, cell_output_dir: Path, checkpoint_components_dir: Path,
) -> dict[str, Any]:
    """Bundle a 5-component checkpoint dir into one resumable blob + a state manifest.

    Returns the ``checkpoint`` sub-object for a raw_receipt: path/sha256/state/
    state_path/state_sha256 (all paths relative to run_dir). Fails closed
    (ValueError) if any of the five components is missing.
    """
    run_dir = Path(run_dir)
    components_dir = Path(checkpoint_components_dir)
    state: dict[str, str] = {}
    for name in _CHECKPOINT_COMPONENTS:
        comp = components_dir / name
        if not comp.is_file():
            raise ValueError(f"checkpoint component missing: {name}")
        state[f"{name}_sha256"] = _sha256_file(comp)

    bundle_path = Path(cell_output_dir) / "checkpoint.tar"
    with tarfile.open(bundle_path, "w") as tar:
        for name in _CHECKPOINT_COMPONENTS:            # fixed order → deterministic
            info = tar.gettarinfo(str(components_dir / name), arcname=name)
            info.mtime = 0                             # pin mtime → stable bundle sha
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with (components_dir / name).open("rb") as fh:
                tar.addfile(info, fh)

    state_path = Path(cell_output_dir) / "checkpoint-state.json"
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

    return {
        "path": _rel(bundle_path, run_dir),
        "sha256": _sha256_file(bundle_path),
        "state": state,
        "state_path": _rel(state_path, run_dir),
        "state_sha256": _sha256_file(state_path),
    }
