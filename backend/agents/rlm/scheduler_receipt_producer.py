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
import math
import tarfile
from pathlib import Path
from typing import Any

_CHECKPOINT_COMPONENTS = ("model", "optimizer", "lr_scheduler", "rng", "data_order")

_PRODUCER_ARTIFACTS = {"checkpoint.tar", "checkpoint-state.json", "evidence_bundle.json"}


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


def _code_tree_bytes(cell_output_dir: Path) -> bytes:
    """Deterministic digest over the cell's CODE files only.

    Excludes this producer's own materialized artifacts (checkpoint.tar,
    checkpoint-state.json, evidence_bundle.json) and any checkpoint-components
    directory, so code_tree_digest reflects code, never trained weights — two
    code-identical cells hash identically, and re-running the producer on a
    populated cell dir is stable.
    """
    root = Path(cell_output_dir)
    parts = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name in _PRODUCER_ARTIFACTS:
            continue
        # skip anything under a checkpoint-components dir (raw weight/optimizer bytes)
        if "checkpoint_components" in p.relative_to(root).parts:
            continue
        parts.append(p.name.encode() + b"\0" + hashlib.sha256(p.read_bytes()).hexdigest().encode())
    return b"\n".join(parts)


def build_raw_receipt(
    *,
    run_dir: Path,
    cell_output_dir: Path,
    checkpoint_components_dir: Path,
    ladder: Any,
    campaign_id: str,
    branch_id: str,
    parent_branch_id: str | None,
    attempt_n: int,
    cell_id: str,
    from_step: int,
    to_step: int,
    seed: int,
    termination_cause: str | None,
    dataset_manifest_path: Path,
    run_spec_path: Path,
    code_sha256: str | None = None,
) -> dict[str, Any]:
    """Assemble the exact ``raw_receipt`` ``write_verified_receipt`` expects.

    The ladder metric is read ONLY from ``metrics.json[ladder.metric_id]`` — never
    an LLM grade (``final_report.score``). Fails closed (ValueError) when the metric
    key is absent or is not a finite number, so a receipt is never manufactured from
    a grade. All ``*_path`` values are relative to ``run_dir``.
    """
    run_dir = Path(run_dir)
    cell_output_dir = Path(cell_output_dir)
    metrics_path = cell_output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict) or ladder.metric_id not in metrics:
        raise ValueError(
            f"metric {ladder.metric_id!r} absent from metrics.json — never fall back to a grade"
        )
    value = metrics[ladder.metric_id]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"metric {ladder.metric_id!r} is not a finite number")
    value = float(value)
    metric_sha = _sha256_file(metrics_path)

    checkpoint = materialize_checkpoint(
        run_dir=run_dir,
        cell_output_dir=cell_output_dir,
        checkpoint_components_dir=checkpoint_components_dir,
    )

    # ``_code_tree_bytes`` excludes this producer's own materialized artifacts
    # (checkpoint.tar / checkpoint-state.json / evidence_bundle.json) and the
    # checkpoint-components dir, so the digest is CODE-only and stable regardless
    # of call order — the earlier ``materialize_checkpoint`` writes cannot alter it.
    code_digest = code_sha256 or hashlib.sha256(_code_tree_bytes(cell_output_dir)).hexdigest()

    bundle_path = cell_output_dir / "evidence_bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "coherent": True,
                "metrics_sha256": metric_sha,
                "code_tree_digest": code_digest,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "branch_id": branch_id,
        "parent_branch_id": parent_branch_id,
        "attempt_n": attempt_n,
        "cell_id": cell_id,
        "paper_ref": ladder.paper_ref,
        "ladder_sha256": ladder.sha256,
        "from_step": from_step,
        "to_step": to_step,
        "seed": seed,
        "termination_cause": termination_cause,
        "metric": {
            "id": ladder.metric_id,
            "direction": ladder.direction,
            "value": value,
            "artifact_path": _rel(metrics_path, run_dir),
            "sha256": metric_sha,
        },
        "checkpoint": checkpoint,
        "evidence_bundle": {
            "path": _rel(bundle_path, run_dir),
            "sha256": _sha256_file(bundle_path),
        },
        "fingerprints": {
            "code_sha256": code_digest,
            "dataset_sha256": _sha256_file(Path(dataset_manifest_path)),
            "dataset_manifest_path": _rel(Path(dataset_manifest_path), run_dir),
            "run_spec_sha256": _sha256_file(Path(run_spec_path)),
            "run_spec_path": _rel(Path(run_spec_path), run_dir),
        },
    }
