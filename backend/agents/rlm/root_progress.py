"""Pure lifecycle-stage helper for degenerate-loop detection.

Given the observed state of a run (what the root has actually done), returns
the single next mandatory step the root must complete before being allowed to
call ``FINAL_VAR``.

The function is **pure**: no I/O, no environment reads, no globals, no side
effects.  It is consumed by the forced-iteration refusal text and the harness
backstop (oauth-root-reliability plan, Task 1).
"""
from __future__ import annotations

__all__ = ["REQUIRED_STAGES", "infer_required_stage"]

# ---------------------------------------------------------------------------
# Public contract — canonical stage names
# ---------------------------------------------------------------------------

REQUIRED_STAGES: frozenset[str] = frozenset(
    {
        "need_baseline",      # no implementation yet
        "need_environment",   # code exists, environment not built
        "need_experiment",    # environment ready, nothing run
        "need_verification",  # experiment ran, not yet scored
        "can_finalize",       # scored — FINAL_VAR is allowed
    }
)


def infer_required_stage(
    *,
    primitives: list[str],  # ordered list of observed domain-primitive names
    code_path_exists: bool,
    env_built: bool,
    total_run_experiments: int,
    total_verifications: int,
) -> str:
    """Infer the next mandatory lifecycle stage from observed run state.

    Precedence ladder (authoritative):

    1. ``not code_path_exists``       → ``"need_baseline"``
    2. ``not env_built``              → ``"need_environment"``
    3. ``total_run_experiments == 0`` → ``"need_experiment"``
    4. ``total_verifications == 0``   → ``"need_verification"``
    5. otherwise                      → ``"can_finalize"``

    The ``primitives`` list is accepted for caller convenience and may carry
    future corroborating signals, but the explicit boolean/count arguments are
    authoritative for the ladder above.  The return value is always a member
    of :data:`REQUIRED_STAGES`.
    """
    # NOTE: primitives is intentionally kept in the signature per the plan
    # contract; it is accepted for caller convenience / future signals but is
    # not used in the ladder (the explicit booleans/counts are authoritative).
    if not code_path_exists:
        return "need_baseline"
    if not env_built:
        return "need_environment"
    if total_run_experiments == 0:
        return "need_experiment"
    if total_verifications == 0:
        return "need_verification"
    return "can_finalize"
