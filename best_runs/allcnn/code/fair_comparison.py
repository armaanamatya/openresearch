"""fair_comparison — identical initialization across compared methods, with evidence.

Module B of the fidelity-evidence layer. When a paper compares N methods (optimizers,
schedulers, configs) the comparison is only meaningful if every method starts from the SAME
initial weights — otherwise the ordering is initialization noise. The 2026-06-09 Adam run lost
leaf ``aa97209f`` (0.6, "identical initialization across optimizers are not evidenced") AND
showed inverted orderings (``mlp_sgd_nesterov_acc > mlp_adam_acc``) precisely because each
optimizer got its own init.

This helper does two things:

1. ``snapshot_init_state`` / ``restore_init_state`` — capture the model's initial ``state_dict``
   once and reload it before training each method, so every method trains from byte-identical
   weights (torch-guarded; the model object is only touched on the agent side where torch lives).
2. ``init_fingerprint`` — a deterministic hash of the snapshot the agent records per method in
   ``provenance.json``. Identical fingerprints across methods are the *evidence* the grader was
   missing — "identical init" becomes verifiable rather than asserted.

Pure-stdlib + torch/numpy optional (imported lazily, guarded) so it copies into an agent
sandbox like ``rubric_guard.py``. Flag-gated on ``REPROLAB_FIDELITY_EVIDENCE`` — but the
snapshot/restore/fingerprint functions are useful regardless and never depend on the flag for
correctness (the flag only governs whether the harness *requires* the evidence).
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

__all__ = [
    "is_enabled",
    "snapshot_init_state",
    "restore_init_state",
    "init_fingerprint",
]

ENV_FLAG = "REPROLAB_FIDELITY_EVIDENCE"


def is_enabled() -> bool:
    """True when the fidelity-evidence layer is armed (``REPROLAB_FIDELITY_EVIDENCE`` truthy)."""
    val = os.environ.get(ENV_FLAG, "").strip().lower()
    return val not in ("", "0", "false", "no", "off")


def snapshot_init_state(model: Any) -> dict[str, Any]:
    """Return a deep CPU copy of ``model.state_dict()`` to reuse as a shared init.

    Call ONCE on a freshly-constructed model, then ``restore_init_state`` before training each
    compared method. Works with any object exposing ``state_dict()`` (torch ``nn.Module`` or a
    test double). Tensors are detached + cloned to CPU so the snapshot is immune to later
    in-place training updates. Never raises — a model without ``state_dict`` yields ``{}``.
    """
    get = getattr(model, "state_dict", None)
    if not callable(get):
        return {}
    try:
        sd = get()
    except Exception:
        return {}
    out: dict[str, Any] = {}
    for k, v in dict(sd).items():
        out[k] = _clone_value(v)
    return out


def restore_init_state(model: Any, snapshot: dict[str, Any]) -> bool:
    """Reload ``snapshot`` into ``model`` (``load_state_dict``); True on success.

    Reuse before training each method so every method starts from byte-identical weights.
    Fail-soft: returns False (never raises) if the model has no ``load_state_dict`` or the load
    fails — the caller can fall back to a fresh init, losing only the fairness guarantee.
    """
    load = getattr(model, "load_state_dict", None)
    if not callable(load) or not isinstance(snapshot, dict):
        return False
    try:
        load({k: _clone_value(v) for k, v in snapshot.items()})
        return True
    except Exception:
        return False


def init_fingerprint(snapshot: dict[str, Any]) -> str:
    """Deterministic 16-hex-char fingerprint of an init snapshot, for provenance evidence.

    Two models snapshotted from the same initial weights hash identically; record this per
    method in ``provenance.json`` so the grader can confirm identical initialization. Stable
    across processes (sorted keys, raw tensor/array bytes when available, else ``repr``). Total
    — an empty/None snapshot fingerprints a fixed sentinel rather than raising.
    """
    h = hashlib.sha256()
    if not isinstance(snapshot, dict) or not snapshot:
        h.update(b"__empty_init__")
        return h.hexdigest()[:16]
    for key in sorted(snapshot, key=str):
        h.update(str(key).encode("utf-8"))
        h.update(b"\x00")
        h.update(_value_bytes(snapshot[key]))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# internals — torch / numpy handled lazily so the module imports without them
# ---------------------------------------------------------------------------

def _clone_value(v: Any) -> Any:
    """Detach + CPU-clone a tensor; deep-copy other containers; pass scalars through."""
    try:
        import torch  # noqa: PLC0415
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().clone()
    except Exception:
        pass
    if isinstance(v, dict):
        return {k: _clone_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clone_value(x) for x in v]
    return v


def _value_bytes(v: Any) -> bytes:
    """Stable byte representation of a single state-dict value for hashing."""
    try:
        import torch  # noqa: PLC0415
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            return bytes(str(tuple(t.shape)).encode()) + t.numpy().tobytes()
    except Exception:
        pass
    try:
        import numpy as np  # noqa: PLC0415
        if isinstance(v, np.ndarray):
            return str(v.shape).encode() + v.tobytes()
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return repr([_value_bytes(x) for x in v]).encode()
    if isinstance(v, dict):
        return repr({k: _value_bytes(v[k]) for k in sorted(v, key=str)}).encode()
    return repr(v).encode()
