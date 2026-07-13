"""CPU-class cell classification for Phase D of the reproduction harness.

Pure stdlib (no imports beyond stdlib) — deliberately dependency-free so it
can be imported from any cell-runner module without pulling in K8s/blob/GCS
SDKs. Decides, per cell, whether a cell can run on a cheap CPU pool instead
of a GPU node, and whether an entire matrix result is CPU-class + wholly
infra-failed (the signal that triggers the local in-process fallback in
``k8s_job_cell_runner.run_matrix``, gated on
``OPENRESEARCH_CPU_CLOUD_CELLS``).

Contract (locked by the design spec):

* A **hard GPU signal** (non-trivial VRAM estimate, a known GPU-only
  framework/image, or any declared distributed/multi-process launch) always
  wins over a *soft* ``accelerator="cpu"`` declaration — a cell is never
  silently downgraded off a GPU it actually needs.
* Absent any signal at all, ``requires_gpu`` is conservative and returns
  ``True`` (unknown ⇒ GPU) — never assume a paper's cell is cheap.
"""
from __future__ import annotations

# Frameworks/image keys that are GPU-only by construction — a cell declaring
# one of these is hard-GPU regardless of any accelerator="cpu" hint.
_GPU_FRAMEWORKS = {"verl"}


def _hard_gpu(cell: dict) -> bool:
    """True iff ``cell`` carries a hard GPU signal that overrides any soft hint."""
    if float(cell.get("est_vram_gb") or 0) > 0:
        return True
    if str(cell.get("framework") or cell.get("image_key") or "").lower() in _GPU_FRAMEWORKS:
        return True
    if cell.get("distributed") or cell.get("nproc_per_node"):
        return True
    return False


def requires_gpu(cell: dict, *, trusted_cpu: bool = False) -> bool:
    """Return True iff ``cell`` must run on a GPU node.

    Hard GPU signals (see ``_hard_gpu``) always win, even over an explicit
    ``accelerator="cpu"`` declaration (the caller may choose to warn on that
    conflict; this function just refuses to downgrade). Otherwise an explicit
    ``accelerator`` of ``"cpu"``/``"gpu"`` is honored; with no signal at all
    the unknown case is conservative and returns True (GPU).
    """
    if _hard_gpu(cell):
        return True  # hard signal wins (caller may warn on conflict)
    acc = str(cell.get("accelerator") or "").lower()
    if acc == "cpu":
        return False
    if acc == "gpu":
        return True
    return True  # unknown ⇒ conservative GPU


def run_is_cpu_class(cells: list[dict], *, trusted_cpu: bool = False) -> bool:
    """True iff ``cells`` is non-empty AND every cell is CPU-class."""
    return bool(cells) and all(
        not requires_gpu(c, trusted_cpu=trusted_cpu) for c in cells
    )


def all_cells_infra_failed(results: dict) -> bool:
    """True iff every cell result errored with an infra-shaped reason.

    ``results`` is ``{cell_id: {"status": ..., "error": ...}}``. A single
    non-error cell result (any status other than the ``STATUS_ERROR`` string
    ``"error"``) means the matrix produced at least one real result, so the
    fallback must never discard it — returns False.
    """
    if not results:
        return False
    for r in results.values():
        if (r or {}).get("status") != "error":  # STATUS_ERROR is the string "error"
            return False
    return True
