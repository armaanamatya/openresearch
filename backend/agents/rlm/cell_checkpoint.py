"""Harness-owned WRITER for the 5-field receipt-ready checkpoint contract.

The receipt-gated scheduler (Phase B) verifies branch-rung receipts whose
``checkpoint`` sub-object is materialized by
:func:`scheduler_receipt_producer.materialize_checkpoint`, which reads a
components dir containing EXACTLY five files —
``model``, ``optimizer``, ``lr_scheduler``, ``rng``, ``data_order`` — and hashes
each into the 5-field state manifest.  Nothing PRODUCED that dir from a real
trainer.  This module is that writer: a framework-agnostic, stdlib-only helper a
cell can call to emit receipt-ready checkpoints.

It is stdlib-only (like ``gpu_cell_runner.py`` / ``scheduler_receipt_producer``)
so it can be copied verbatim into the agent cell sandbox — no torch/numpy at
module scope (the optional ``capture_rng_state`` uses guarded LOCAL imports).

Every write is deterministic + atomic (temp file + ``os.replace``): the five
component files are written into ``<checkpoint_dir>/step_<step>/`` and the
``<checkpoint_dir>/LATEST`` pointer is updated LAST, so a preemption mid-write
never leaves a torn checkpoint the reader will trust — the previous good step
stays authoritative and a half-written ``step_<step>`` is simply never pointed
at.  Input validation runs BEFORE any directory is created, so a bad component
leaves no partial ``step_<step>`` dir behind (fail-closed).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# The exact five component names the receipt producer + verifier require.
COMPONENTS = ("model", "optimizer", "lr_scheduler", "rng", "data_order")

_LATEST = "LATEST"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file in the same dir + replace)."""
    directory = path.parent
    fd, tmp = tempfile.mkstemp(dir=str(directory), prefix=".tmp-", suffix=path.name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_checkpoint(
    checkpoint_dir: Path | str,
    step: int,
    *,
    model: bytes,
    optimizer: bytes,
    lr_scheduler: bytes,
    rng: bytes,
    data_order: bytes,
) -> Path:
    """Write a receipt-ready 5-component checkpoint for ``step`` and advance ``LATEST``.

    Each component is raw ``bytes`` — the caller serializes with whatever it uses
    (``torch.save`` to a buffer, pickle, msgpack, ...); this writer is
    framework-agnostic.  Files land in ``<checkpoint_dir>/step_<step>/`` under the
    exact names the receipt producer reads.  Returns the ``step_<step>`` components
    dir path (feed it to ``scheduler_receipt_producer`` as
    ``checkpoint_components_dir``, or read it back with :func:`read_checkpoint`).

    Fails closed: ``step`` must be a non-negative int and every component must be
    ``bytes`` — validated BEFORE any directory is created, so a bad call never
    leaves a partial ``step_<step>`` dir.
    """
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError(f"step must be a non-negative int, got {type(step).__name__}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    payloads = {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "rng": rng,
        "data_order": data_order,
    }
    for name in COMPONENTS:
        if not isinstance(payloads[name], bytes):
            raise TypeError(
                f"checkpoint component {name!r} must be bytes, got {type(payloads[name]).__name__}"
            )

    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    components_dir = root / f"step_{step}"
    components_dir.mkdir(parents=True, exist_ok=True)

    # Fixed order → deterministic; each component written atomically.
    for name in COMPONENTS:
        _atomic_write_bytes(components_dir / name, payloads[name])

    # LATEST is the commit point — written LAST + atomically, so a crash mid-write
    # leaves the previous good step authoritative and a torn step is never pointed at.
    _atomic_write_bytes(root / _LATEST, str(step).encode("utf-8"))
    return components_dir


def _iter_step_dirs(root: Path) -> list[tuple[int, Path]]:
    steps: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("step_"):
            continue
        suffix = child.name[len("step_"):]
        try:
            steps.append((int(suffix), child))
        except ValueError:
            continue
    return steps


def latest_checkpoint_dir(checkpoint_dir: Path | str) -> Path | None:
    """Return the newest step's components dir, or ``None`` if there is none.

    Reads the ``LATEST`` pointer first; falls back to the NUMERICALLY-max
    ``step_<n>`` dir (so ``step_10`` beats ``step_2``) when the pointer is
    missing/stale.  This is what the cohort loop / receipt producer points
    ``checkpoint_components_dir`` at.
    """
    root = Path(checkpoint_dir)
    if not root.is_dir():
        return None

    pointer = root / _LATEST
    if pointer.is_file():
        try:
            step = int(pointer.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            step = None
        if step is not None:
            candidate = root / f"step_{step}"
            if candidate.is_dir():
                return candidate

    steps = _iter_step_dirs(root)
    if not steps:
        return None
    return max(steps, key=lambda pair: pair[0])[1]


def read_checkpoint(components_dir: Path | str) -> dict[str, bytes]:
    """Return the 5 component byte blobs from a step components dir (for a resume).

    Fails closed (``ValueError``) if any of the five components is missing.
    """
    root = Path(components_dir)
    out: dict[str, bytes] = {}
    for name in COMPONENTS:
        comp = root / name
        if not comp.is_file():
            raise ValueError(f"checkpoint component missing: {name}")
        out[name] = comp.read_bytes()
    return out


def capture_rng_state() -> bytes:
    """Serialize python ``random`` + (if importable) numpy + torch RNG state to bytes.

    Convenience for a real trainer producing the ``rng`` component.  The numpy /
    torch imports are LOCAL + optional so this module stays stdlib-only at scope;
    absent frameworks are simply omitted.  Deterministic: the same RNG state
    serializes to the same bytes.
    """
    import pickle
    import random

    state: dict[str, object] = {"python_random": random.getstate()}

    try:  # numpy is optional
        import numpy as _np

        state["numpy"] = _np.random.get_state()
    except Exception:  # pragma: no cover - depends on env
        pass

    try:  # torch is optional
        import torch as _torch

        state["torch_cpu"] = _torch.random.get_rng_state()
        if _torch.cuda.is_available():  # pragma: no cover - needs a GPU
            state["torch_cuda"] = _torch.cuda.get_rng_state_all()
    except Exception:  # pragma: no cover - depends on env
        pass

    return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
