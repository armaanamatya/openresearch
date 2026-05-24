"""Forced-iteration policy — refuse FINAL_VAR until the rubric is honest (Lane H).

The root model is supposed to:

  1. ``run_experiment`` → ``verify_against_rubric``
  2. If ``overall_score < target_score`` and iterations remain, call
     ``propose_improvements`` + ``implement_baseline`` with ``repair_context``,
     then ``run_experiment`` again.
  3. ``FINAL_VAR`` only when the rubric is satisfied OR the budget is gone.

But until this module shipped, step 2 was *suggested* by the system prompt and
not enforced — the root could call ``FINAL_VAR`` immediately after a low score
and ship the partial report.  This module installs a one-time monkey-patch on
``rlm.environments.local_repl.LocalREPL._final_var`` that:

  * consults a thread-local policy
  * if the policy says "block", short-circuits ``_final_var`` to return an
    error string in the exact shape that ``rlm.utils.parsing.find_final_answer``
    treats as "no final answer yet" — so the rlm root-loop continues to the
    next iteration without ever surfacing a final answer.

The block message is the run_warning the operator sees in the UI; the root
model sees it as the FINAL_VAR return value, which is the natural place for
"keep going" guidance.

Wall-clock takes precedence: when less than :data:`_WALL_CLOCK_FLOOR_S` remain,
the policy is bypassed.  Shipping a partial report is always more useful than
timing out with nothing.

Idempotent.  Side effects scoped per-run via a thread-local stack.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False
_PATCH_LOCK = threading.Lock()

# Below this many seconds remaining, the policy unconditionally accepts
# FINAL_VAR.  A partial report shipped on a near-timeout is more useful to
# the operator than no report at all.
_WALL_CLOCK_FLOOR_S = 60.0


@dataclass
class ForcedIterationPolicy:
    """Per-run policy state read by the FINAL_VAR interceptor.

    The policy is intentionally simple: read latest rubric score / target /
    iteration off the RunContext, compare against `min_iterations`, and either
    accept or refuse the FINAL_VAR.  The interceptor never inspects RLM
    internals — it only consults this object.
    """

    min_iterations: int
    # Callable that returns the latest (score, target, iteration_when_recorded)
    # tuple.  Reading via callable rather than direct field access lets tests
    # supply lambdas and lets production code read the live RunContext.
    rubric_snapshot: Callable[[], tuple[float | None, float | None, int]]
    # Callable that returns the current root-loop iteration index (1-based).
    current_iteration: Callable[[], int]
    # Callable that returns remaining wall-clock seconds, or None if no budget.
    remaining_s: Callable[[], float | None]
    # Callable invoked when the policy refuses a FINAL_VAR.  Used to emit the
    # `run_warning` SSE event.  Receives a single `message: str` argument.
    on_refusal: Callable[[str], None]
    # Counter — how many FINAL_VAR refusals have been issued for this run.
    # The runtime stops refusing past `_MAX_REFUSALS_PER_RUN` so a stubborn
    # root model can still terminate the run.  A defensive bound, not a
    # primary correctness lever.
    refusal_count: int = 0

    def should_refuse(self) -> tuple[bool, str | None]:
        """Return (refuse, message). When refuse=True, message is non-None.

        Order of checks (each takes precedence over the next):

          0. Wall-clock floor — never refuse if remaining_s <= floor.
          1. min_iterations==0 — policy disabled, always accept.
          2. No rubric data yet — accept (the run is rubric-less).
          3. Score >= target — accept (the rubric is satisfied).
          4. current_iteration < min_iterations — refuse.
          5. Otherwise — accept (best-effort exit; ran the floor of attempts).

        The message returned on refuse=True is a single-line, plain-English
        sentence the root model can act on directly.
        """
        # 0. Wall-clock floor — always honored. Better to ship partial than
        # to time out with nothing.
        remaining = self.remaining_s()
        if remaining is not None and remaining <= _WALL_CLOCK_FLOOR_S:
            return (False, None)

        # 1. Disabled.
        if self.min_iterations <= 0:
            return (False, None)

        # Stop after a defensive max refusals — a stubborn root can still ship.
        if self.refusal_count >= _MAX_REFUSALS_PER_RUN:
            return (False, None)

        score, target, _score_iter = self.rubric_snapshot()

        # 2. No rubric data — accept honestly.
        if score is None or target is None:
            return (False, None)

        # 3. Score satisfies target — accept.
        if score >= target:
            return (False, None)

        # 4. Below target AND haven't hit the iteration floor — refuse.
        cur = self.current_iteration()
        if cur < self.min_iterations:
            msg = (
                f"rubric overall_score={score:.3f} is below target_score={target:.3f} "
                f"after iteration {cur} (min_rubric_iterations={self.min_iterations}); "
                "call propose_improvements + implement_baseline with repair_context "
                "set to your latest verify_against_rubric result, then run_experiment "
                "again — do NOT call FINAL_VAR until the rubric is satisfied or the "
                "iteration floor is reached."
            )
            return (True, msg)

        # 5. Iteration floor reached — accept the partial result.
        return (False, None)


# A run can stubbornly call FINAL_VAR every iteration.  Past this many
# refusals we let it through so the run still terminates.  Picked at 8x the
# default min_iterations=2 so a healthy run never hits it, but a wedged
# loop still drains within the wall-clock budget.
_MAX_REFUSALS_PER_RUN = 16


# Thread-local stack of active policies.  Each call to `forced_iteration_policy`
# pushes a policy and pops on exit; the interceptor reads the top of the stack.
# Stack semantics support nested runs (concurrent tests, sub-runs).
_LOCAL = threading.local()


def _policy_stack() -> list[ForcedIterationPolicy]:
    stack = getattr(_LOCAL, "stack", None)
    if stack is None:
        stack = []
        _LOCAL.stack = stack
    return stack


def _current_policy() -> ForcedIterationPolicy | None:
    stack = _policy_stack()
    return stack[-1] if stack else None


# The exact prefix find_final_answer treats as "no final answer yet".  Built
# from the rlm.utils.parsing.find_final_answer regex literals — when the
# returned string contains all three of "Variable '", "' not found", and
# "FINAL_VAR", the rlm core loop continues to the next iteration.
_BLOCK_PREFIX_TEMPLATE = (
    "Error: Variable '{var}' not found — RLM forced-iteration policy is "
    "blocking FINAL_VAR. {msg} The FINAL_VAR call has been refused."
)


def _build_block_message(variable_name: str, policy_msg: str) -> str:
    """Build the string `_final_var` returns when the policy refuses.

    Must contain the three substrings ``Variable '``, ``' not found``, and
    ``FINAL_VAR`` so the rlm core's find_final_answer treats it as "no final
    answer yet" rather than an actual answer (see
    `rlm/utils/parsing.py:find_final_answer`).
    """
    safe = variable_name if isinstance(variable_name, str) else str(variable_name)
    return _BLOCK_PREFIX_TEMPLATE.format(var=safe, msg=policy_msg)


def apply_forced_iteration_patch() -> None:
    """Install the FINAL_VAR interceptor on ``LocalRepl._final_var``.

    Idempotent.  Safe to call multiple times.  Should be called once at
    module import — `run.py` does this alongside the other rlm patches.
    """
    global _PATCH_APPLIED
    with _PATCH_LOCK:
        if _PATCH_APPLIED:
            return

        try:
            from rlm.environments.local_repl import LocalREPL
        except ImportError as exc:  # pragma: no cover — rlm always installed
            logger.warning(
                "apply_forced_iteration_patch: rlm.environments.local_repl not "
                "importable (%s); forced-iteration policy will not be active",
                exc,
            )
            return

        _original_final_var = LocalREPL._final_var

        def _intercepted_final_var(self: Any, variable_name: Any) -> str:
            policy = _current_policy()
            if policy is None:
                return _original_final_var(self, variable_name)

            refuse, message = policy.should_refuse()
            if not refuse:
                return _original_final_var(self, variable_name)

            assert message is not None  # invariant from should_refuse contract
            policy.refusal_count += 1

            # Notify the policy's on_refusal callback so the orchestrator can
            # surface a run_warning SSE event. The callback must not raise.
            try:
                policy.on_refusal(message)
            except Exception:  # noqa: BLE001 — defensive; emit failures must not block
                logger.exception("forced_iteration: on_refusal callback raised")

            return _build_block_message(variable_name, message)

        LocalREPL._final_var = _intercepted_final_var  # type: ignore[method-assign]
        _PATCH_APPLIED = True
        logger.info("rlm LocalREPL._final_var forced-iteration interceptor installed")


@contextmanager
def forced_iteration_policy(policy: ForcedIterationPolicy) -> Iterator[None]:
    """Context manager — install the policy for the wrapped block.

    Pushes ``policy`` onto the thread-local stack on enter, pops on exit.
    The interceptor reads only the top of the stack, so concurrent runs in
    other threads each get their own policy.
    """
    stack = _policy_stack()
    stack.append(policy)
    try:
        yield
    finally:
        # Defensive pop — if a test inserts other entries we still pop our own.
        try:
            stack.remove(policy)
        except ValueError:  # pragma: no cover — only when caller already popped
            pass


__all__ = [
    "ForcedIterationPolicy",
    "apply_forced_iteration_patch",
    "forced_iteration_policy",
]
