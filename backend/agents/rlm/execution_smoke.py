"""Phase 2B — preflight EXECUTION smoke (the 1-step dry-run half of preflight "TDD").

Where :mod:`preflight_smoke` runs an IMPORT-only probe (every declared third-party
dependency resolves), this runs the agent's *entry script* for ONE training step on
tiny data, INSIDE the sandbox, BEFORE the full GPU training command. That catches the
class of bug that import resolution cannot: a runtime crash that only manifests once a
tensor actually hits the kernel — the canonical case being an asynchronous
``CUDA error: device-side assert triggered`` (e.g. an out-of-range index/label feeding
a VAE) that otherwise surfaces only AFTER a 25-minute full run, with an obscured,
async-deferred traceback pointing at the wrong line.

Design (robust + low-false-positive):
  * It exports ``CUDA_LAUNCH_BLOCKING=1`` so CUDA kernels run synchronously and the
    crash surfaces AT THE REAL LINE in seconds, not deferred to some later allocation.
  * It exports ``REPROLAB_SMOKE_STEPS=<steps>`` (default 1) so a cooperating entry
    script does one optimizer step on tiny data and exits 0 — the whole point is that
    this is cheap (seconds), not a real run.
  * It wraps the entry script in ``timeout <timeout_s>``. A script that IGNORES the
    smoke env and tries the full run is KILLED (exit 124) rather than hung forever.
    A timeout is treated as a SOFT pass ("smoke not honored → skip, don't block"): a
    non-honoring script is not necessarily broken, and blocking it would be a false
    positive that costs more than the bug it might catch.
  * It inherits the sandbox's isolation (network/mem/fs controls, hard command
    timeout) — the sandbox is the isolation boundary, exactly as for the import smoke.
  * Gated behind ``REPROLAB_EXECUTION_SMOKE`` (default OFF) — opt-in, since it actually
    executes the agent's training code (one step). Off by default; the sandbox bounds it.

Unlike the import smoke, this does NOT emit a generated helper script — it runs the
agent's OWN entry script (``train.py`` by default) with a step-cap env var. The
contract is "honor ``REPROLAB_SMOKE_STEPS`` and exit, or be timed out and skipped".
The emitted command is a single ``sh -c '...'`` string (stdlib-only to build), the
same shape :mod:`preflight_smoke` uses, so the wiring can treat both identically.
"""
from __future__ import annotations

import os
from pathlib import Path

# A command carrying this marker short-circuits the sandbox command loop on a BLOCKING
# failure (skip the remaining training commands) — same role as
# ``preflight_smoke.MARKER``. See _execute_in_sandbox / interpret_exit: only a real
# crash (not a timeout-kill) is blocking.
MARKER = "# reprolab:execution-smoke"

# The entry script reads this to cap itself to N steps on tiny data for the dry-run.
SMOKE_STEPS_ENV = "REPROLAB_SMOKE_STEPS"


def is_enabled() -> bool:
    """True when ``REPROLAB_EXECUTION_SMOKE`` is truthy (default OFF — opt-in).

    Runs the agent's entry for ONE step on tiny data with ``CUDA_LAUNCH_BLOCKING=1``
    before the full GPU run, so a runtime crash (e.g. the Adam VAE BCELoss device-side
    assert) surfaces at the real line in seconds. Fail-soft: a script that ignores the
    step cap is killed by ``timeout`` (exit 124) → SOFT PASS; only a genuine crash
    blocks. Kept OPT-IN (vs the always-on import smoke) because it changes the cell
    route for every multi-cell local run (runs a pre-grid smoke) — enable it on the
    validated GPU launch via ``REPROLAB_EXECUTION_SMOKE=1`` rather than globally
    (issue #5, 2026-06-15: default-ON had un-validated blast radius across the suite).
    """
    return os.environ.get("REPROLAB_EXECUTION_SMOKE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def smoke_command(
    code_dir: Path,
    *,
    entry_script: str = "train.py",
    steps: int = 1,
    timeout_s: int = 300,
) -> str:
    """Return the sandbox command that runs the 1-step execution smoke (carries :data:`MARKER`).

    The command:
      * ``cd``s into ``code_dir``,
      * resolves a working interpreter at run time (``python3`` then ``python``) so the
        step is robust across the runpod image and the local per-run venv,
      * exports ``REPROLAB_SMOKE_STEPS=<steps>`` (cap the run to N steps on tiny data)
        and ``CUDA_LAUNCH_BLOCKING=1`` (synchronous kernels → real crash line),
      * runs ``<entry_script>`` under ``timeout <timeout_s>`` so a script that ignores
        the smoke env and tries the full run is killed (exit 124), not hung forever.

    Exit-code semantics (see :func:`interpret_exit`): 0 → honored & clean; 124 → timed
    out (not honored → soft pass, skip); any other non-zero → a REAL crash of the entry
    script (e.g. the device-side assert) → blocking.
    """
    code = str(code_dir)
    return (
        f'sh -c \'cd "{code}" && P=$(command -v python3 || command -v python) && '
        f'{SMOKE_STEPS_ENV}={int(steps)} CUDA_LAUNCH_BLOCKING=1 '
        f'timeout {int(timeout_s)} "$P" {entry_script}\'  {MARKER}'
    )


def interpret_exit(code: int) -> tuple[str, bool]:
    """Map the smoke command's exit code to ``(status, is_blocking_crash)``.

    Rationale:
      * ``0`` → ``("ok", False)`` — the entry script honored the step cap and exited
        cleanly; the one-step run did not crash. Proceed to the full training run.
      * ``124`` → ``("not_honored", False)`` — ``timeout`` killed the script (it ignored
        ``REPROLAB_SMOKE_STEPS`` and attempted the full run). A non-honoring script is
        NOT necessarily broken, so this is a SOFT pass: skip the smoke, do NOT block.
      * any other non-zero → ``("crash", True)`` — the entry script genuinely failed
        (a real runtime error like ``CUDA error: device-side assert triggered``). This
        is the bug the smoke exists to catch in seconds; it BLOCKS the GPU training run.
    """
    if code == 0:
        return ("ok", False)
    if code == 124:  # GNU coreutils ``timeout`` SIGTERM exit code
        return ("not_honored", False)
    return ("crash", True)


__all__ = [
    "MARKER",
    "SMOKE_STEPS_ENV",
    "is_enabled",
    "smoke_command",
    "interpret_exit",
]
