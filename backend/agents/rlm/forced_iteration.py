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
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


@dataclass
class PolicyDecision:
    """Return type for should_refuse_final_var — carries refuse flag + reason."""
    refuse: bool
    reason: str = ""

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False
_PATCH_LOCK = threading.Lock()

# Below this many seconds remaining, the policy unconditionally accepts
# FINAL_VAR.  A partial report shipped on a near-timeout is more useful to
# the operator than no report at all.
_WALL_CLOCK_FLOOR_S = 60.0

# Terminal failure classes that must NOT be force-iterated.  A shrink-exhausted
# OOM (gpu_cell_runner spent its per-cell batch-scale ladder), an explicit
# capacity-exhausted stop, or a per-run GPU-budget exhaustion cannot be fixed by
# re-running the same config — the only honest move is to stop and ship the
# structured stop report.  Refusing FINAL_VAR here just re-OOMs (or re-burns the
# already-exceeded budget on) the next iteration (the 2026-05-31 death spiral).
# ``root_degenerate_loop`` is the analogous root-side terminal: the root has
# called FINAL_VAR repeatedly with NO lifecycle progress (the degenerate
# refusal loop, Task 4) — continuing to refuse only churns to the 16-refusal
# cap / wall clock, so we accept the next FINAL_VAR and ship the report.
_TERMINAL_FAILURE_CLASSES = frozenset(
    {
        "oom_shrink_exhausted",
        "capacity_exhausted",
        "budget_exhausted",
        "root_degenerate_loop",
    }
)


def _default_degenerate_threshold() -> int:
    """Read OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD (default 3).

    Defensive parse (mirrors the iteration-budget ``.isdigit()`` guard): a
    non-numeric / non-positive value falls back to the default 3 rather than
    raising at import.
    """
    raw = os.environ.get("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else 3


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
    # None disables rubric-snapshot checks (used in PR-μ simplified tests).
    rubric_snapshot: Callable[[], tuple[float | None, float | None, int]] | None = None
    # Callable that returns the current root-loop iteration index (1-based).
    current_iteration: Callable[[], int] | None = None
    # Callable that returns remaining wall-clock seconds, or None if no budget.
    remaining_s: Callable[[], float | None] | None = None
    # Callable invoked when the policy refuses a FINAL_VAR.  Used to emit the
    # `run_warning` SSE event.  Receives a single `message: str` argument.
    on_refusal: Callable[[str], None] | None = None
    # PR-ι.1 — hard per-run iteration cap.  When current_iteration() >=
    # max_rlm_iterations, FINAL_VAR is ACCEPTED unconditionally (the budget is
    # exhausted and we must not loop indefinitely).  None disables this check.
    # Read from OPENRESEARCH_MAX_RLM_ITERATIONS env var (default 5).
    max_rlm_iterations: int | None = None
    # Callable invoked when the iteration budget is exceeded.
    on_budget_exceeded: Callable[[str], None] | None = None
    # Lane O — count of "honest" candidate outcomes recorded so far. An
    # honest outcome is "promoted", "failed", or "marginal" — the agent
    # actually ran the candidate's experiment. "declined" and "skipped"
    # are NOT honest. When None, this check is disabled (back-compat with
    # tests that don't supply it).
    #
    # Pinned by the 2026-05-25 Adam regression: agent reached iter 2,
    # called propose_improvements(), then blanket-declined all 3 in a
    # for-loop without running any, then FINAL_VAR'd with rubric=0/0.6.
    # The min_iterations floor alone wasn't enough — the agent satisfied
    # iteration count by ingesting + planning + a single failed experiment,
    # never honestly testing an improvement.
    honest_candidate_outcomes: Callable[[], int] | None = None
    # Counter — how many FINAL_VAR refusals have been issued for this run.
    # The runtime stops refusing past `_MAX_REFUSALS_PER_RUN` so a stubborn
    # root model can still terminate the run.  A defensive bound, not a
    # primary correctness lever.
    refusal_count: int = 0
    # PR-α followup — repair-iteration counter.  Incremented each time
    # run_experiment returns outcome="repairable" (preflight_blocked, code
    # error, etc.).  When _repair_iter_count < OPENRESEARCH_MIN_REPAIR_ITERATIONS
    # the policy refuses FINAL_VAR, forcing the root to attempt another repair.
    _repair_iter_count: int = field(default=0, compare=False, repr=False)
    _last_repair_failure_class: str | None = field(default=None, compare=False, repr=False)
    # Terminal stop signal — set by run_experiment (via note_terminal_failure)
    # when an experiment failed un-repairably (shrink-exhausted OOM / capacity
    # exhaustion). When in _TERMINAL_FAILURE_CLASSES, should_refuse ACCEPTS the
    # next FINAL_VAR so the run stops cleanly instead of re-OOMing the same config.
    _terminal_failure_class: str | None = field(default=None, compare=False, repr=False)
    # Optional separate callback for the repair-refusal path so the SSE event
    # can carry code="forced_repair_iteration" distinct from "forced_iteration".
    # Defaults to None; when None the existing on_refusal is used as fallback.
    on_repair_refusal: Callable[[str], None] | None = None
    # Task 2 — no-progress refusal-loop detection.  ``degenerate_threshold`` is
    # the number of consecutive same-signature refusals with NO intervening
    # state-changing primitive that constitutes a degenerate loop (default from
    # OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD, else 3).  When the counter
    # first reaches it, ``on_degenerate_refusal_loop`` fires once with
    # ``{signature, count, required_stage}``.  ``required_stage`` returns the
    # current required lifecycle stage (Task 1 enum value), injected by run.py
    # in Task 4; None until then → payload required_stage is None.  All
    # additive + inert when on_degenerate_refusal_loop is None (the default).
    degenerate_threshold: int = field(default_factory=_default_degenerate_threshold)
    on_degenerate_refusal_loop: Callable[[dict], None] | None = None
    required_stage: Callable[[], str] | None = None
    # Task 3 — set by run.py (Task 4) when the root model is claude-oauth.
    # Sonnet recovers better with fewer degrees of freedom, so the escalated
    # degenerate-loop message appends a one-line command skeleton for it.
    # Default False keeps non-oauth / unset runs skeleton-free.
    oauth_root: bool = False
    # Internal: set by should_refuse() to signal which SSE event code the
    # interceptor should use when calling on_refusal / on_repair_refusal.
    _pending_refusal_code: str = field(default="forced_iteration", compare=False, repr=False)
    # Task 2 — no-progress refusal-loop state.  ``_pending_refusal_signature``
    # is stamped by every refusing return of should_refuse()/_build_repair_refusal()
    # and read by the interceptor when calling register_refusal().  The counter
    # is deliberately CROSS-turn: it resets only on a real state-changing
    # primitive (record_state_change), never in on_iteration_advance().
    _noprogress_refusals: int = field(default=0, compare=False, repr=False)
    _last_refusal_signature: str | None = field(default=None, compare=False, repr=False)
    _degenerate_fired: bool = field(default=False, compare=False, repr=False)
    _pending_refusal_signature: str | None = field(default=None, compare=False, repr=False)
    # PR-μ Solution C — per-iteration run_experiment outcome sequence.
    # Tracks outcomes of all run_experiment calls in the current root turn;
    # reset by on_iteration_advance() at each turn boundary.
    _experiments_in_iteration: list[str] = field(default_factory=list, compare=False, repr=False)
    # BUG-NEW-046: total run_experiment calls across ALL iterations.  Unlike
    # _experiments_in_iteration (reset per turn), this is monotonically
    # increasing.  should_refuse() uses it to block FINAL_VAR when the root
    # has never run any experiment — even if the iteration floor is satisfied.
    _total_run_experiments: int = field(default=0, compare=False, repr=False)
    # PR-μ Solution C — simplified constructor fields for test ergonomics and
    # future direct instantiation without callables.  Production code continues
    # to use the callable-based fields (rubric_snapshot, current_iteration,
    # remaining_s, on_refusal) which are still required.
    target_score: float | None = field(default=None, compare=False, repr=False)
    run_id: str | None = field(default=None, compare=False, repr=False)
    ctx: Any | None = field(default=None, compare=False, repr=False)

    # Task 3 — map an inferred lifecycle stage (root_progress.infer_required_stage
    # enum value) to the concrete next-call directive the refusal text should name.
    # An unmapped / None stage (e.g. ``can_finalize``) falls back to the legacy text.
    _STAGE_DIRECTIVES = {
        "need_baseline": "call plan_reproduction, then implement_baseline(plan)",
        "need_environment": "call build_environment",
        "need_experiment": "call run_experiment(code_path, env_id)",
        "need_verification": "call verify_against_rubric on your latest run_experiment result",
    }

    def _safe_required_stage(self) -> str | None:
        """Return the current required lifecycle stage, or None (fail-soft).

        Wraps the injected ``required_stage`` callable; a missing callable or a
        raising one degrades to None so the refusal text falls back to the
        legacy form rather than crashing the policy.
        """
        if self.required_stage is None:
            return None
        try:
            return self.required_stage()
        except Exception:  # noqa: BLE001 — never crash the policy
            return None

    def _no_experiment_message(self, cur: int) -> str:
        """Build the no-experiment refusal text, stage-specific when a stage is known.

        When ``required_stage`` resolves to a mapped stage, the message NAMES
        that stage's concrete next call (so ``need_baseline`` mentions
        ``implement_baseline``, not just ``run_experiment``).  When the stage is
        None or unmapped, returns the legacy text byte-for-byte — the single
        source of truth for that string.
        """
        stage = self._safe_required_stage()
        directive = self._STAGE_DIRECTIVES.get(stage or "") if stage else None
        if directive is not None:
            return (
                f"FINAL_VAR refused at iteration {cur}: run_experiment has never "
                "been called and there is no executed evidence to report. Your "
                f"next step in the reproduction lifecycle is to {directive}, then "
                "run_experiment to execute the code, then verify_against_rubric to "
                "score it, then FINAL_VAR."
            )
        # Legacy text — preserved byte-for-byte (required_stage None / unmapped).
        return (
            f"FINAL_VAR refused at iteration {cur}: run_experiment has never "
            "been called. You must execute the baseline code at least once "
            "before terminating. Next steps: call build_environment (if the "
            "image is not built yet), then run_experiment(code_path, env_id) "
            "to execute the code, then verify_against_rubric to score it, "
            "then FINAL_VAR."
        )

    def escalate_refusal_message(self, signature: str) -> str:
        """Strong loop-naming message once the no-progress counter hit the threshold.

        Called by the interceptor AFTER ``register_refusal`` so
        ``_noprogress_refusals`` is the true post-registration count.  This
        REPLACES (does not append to) the base refusal text — a weak root
        recovers better from one unambiguous directive.  For ``claude-oauth``
        roots a one-line recovery skeleton is appended.
        """
        stage = self._safe_required_stage()
        action = self._STAGE_DIRECTIVES.get(
            stage or "", "call plan_reproduction, then implement_baseline(plan), then run_experiment"
        )
        msg = (
            f"You have called FINAL_VAR {self._noprogress_refusals}× with zero progress "
            f"(signature={signature}). STOP reading and analyzing — your only valid "
            f"next call is to {action}."
        )
        if self.oauth_root:
            msg += " " + self._oauth_command_skeleton(stage)
        return msg

    @staticmethod
    def _oauth_command_skeleton(stage: str | None) -> str:
        """A ONE-LINE stage-appropriate command skeleton for claude-oauth recovery."""
        skeletons = {
            "need_baseline": (
                "Recovery: plan = plan_reproduction(); "
                "impl = implement_baseline(plan); "
                "run_experiment(impl['code_path'], env_id)"
            ),
            "need_environment": "Recovery: env = build_environment(); run_experiment(code_path, env['env_id'])",
            "need_experiment": "Recovery: run_experiment(code_path, env_id)",
            "need_verification": "Recovery: verify_against_rubric(metrics_path)",
        }
        return skeletons.get(
            stage or "",
            "Recovery: plan = plan_reproduction(); impl = implement_baseline(plan); "
            "run_experiment(impl['code_path'], env_id)",
        )

    def record_repair_attempt(self, failure_class: str) -> None:
        """Record that run_experiment returned a repairable outcome.

        Called from the tool wrapper in run.py whenever run_experiment yields
        outcome="repairable". The count is consulted by should_refuse() to
        decide whether to block the next FINAL_VAR call.
        """
        self._repair_iter_count += 1
        self._last_repair_failure_class = failure_class

    def note_terminal_failure(self, failure_class: str) -> None:
        """Record an un-repairable terminal failure (e.g. ``oom_shrink_exhausted``).

        Called from run.py when run_experiment returns a terminal capacity stop.
        When ``failure_class`` is in :data:`_TERMINAL_FAILURE_CLASSES`,
        :meth:`should_refuse` accepts the next FINAL_VAR so the run stops cleanly
        and ships its structured stop report rather than looping on the same OOM.
        """
        self._terminal_failure_class = failure_class

    def should_refuse(self) -> tuple[bool, str | None]:
        """Return (refuse, message). When refuse=True, message is non-None.

        Order of checks (each takes precedence over the next):

          0. Wall-clock floor — never refuse if remaining_s <= floor.
          0.5. Defensive max-refusals cap — a stubborn root still terminates.
          1. min_iterations==0 — rubric-iteration policy disabled; skip to 4.6.
          2. No rubric data yet — accept (the run is rubric-less) unless
             repair floor fires (4.6).
          3. Score >= target — accept the result.
          4. current_iteration < min_iterations — refuse (rubric floor).
          4.5. Lane O — iteration floor reached BUT no candidate was
               honestly tested (everything was declined/skipped) — refuse.
          4.6. PR-α followup — repair iteration floor.  Last run_experiment
               returned repairable AND repair_iter < MIN_REPAIR — refuse.
          5. Otherwise — accept (best-effort exit; ran the floor of attempts).

        The message returned on refuse=True is a single-line, plain-English
        sentence the root model can act on directly.
        """
        # Clear the per-decision signature handoff up front: every refusing
        # branch below MUST re-stamp self._pending_refusal_signature. Resetting
        # here makes a future un-stamped refusal degrade gracefully to None ->
        # the interceptor's generic "forced_iteration" bucket, rather than
        # silently inheriting the previous decision's signature (which could
        # mask or fabricate a degenerate-loop trip — the field now feeds the
        # no-progress counter, not just an SSE label).
        self._pending_refusal_signature = None

        # 0. Wall-clock floor — always honored. Better to ship partial than
        # to time out with nothing.
        remaining = self.remaining_s() if self.remaining_s is not None else None
        if remaining is not None and remaining <= _WALL_CLOCK_FLOOR_S:
            return (False, None)

        # 0.4. Terminal stop — accept FINAL_VAR, ship report, no re-loop. A
        # shrink-exhausted OOM / capacity-exhausted stop is NOT repairable by
        # re-running the same config (refusing only re-OOMs), and a
        # root_degenerate_loop is a root that keeps calling FINAL_VAR with no
        # lifecycle progress (refusing only churns to the 16-refusal cap).
        # Either way the honest move is to stop and ship the structured stop
        # report (2026-05-31 OOM remediation + Task 4 degenerate-loop early-abort).
        # Robust to both wiring styles: note_terminal_failure() OR a terminal
        # class threaded through record_repair_attempt().
        _terminal = self._terminal_failure_class or self._last_repair_failure_class
        if _terminal in _TERMINAL_FAILURE_CLASSES:
            logger.info(
                "forced_iteration: terminal stop '%s' — accepting FINAL_VAR "
                "(stop + report, no re-loop)", _terminal,
            )
            return (False, None)

        # 0.3. PR-ι.1 — per-run iteration budget cap.  When the iteration
        # budget is exhausted, ACCEPT FINAL_VAR unconditionally — no further
        # refusing is possible, and the run must terminate.  Wall-clock floor
        # (check 0) already catches the near-timeout case, so this fires only
        # when the root has consumed its full iteration allowance.
        _max_iter = self.max_rlm_iterations
        if _max_iter is None:
            # Fall back to env var so callers that predate max_rlm_iterations
            # still get the cap without re-constructing the policy object.
            _raw = os.environ.get("OPENRESEARCH_MAX_RLM_ITERATIONS", "").strip()
            _max_iter = int(_raw) if _raw.isdigit() and int(_raw) > 0 else None
        if _max_iter is not None and _max_iter > 0 and self.current_iteration is not None:
            cur = self.current_iteration()
            if cur >= _max_iter:
                msg = (
                    f"Iteration budget exhausted: current_iteration={cur} has "
                    f"reached max_rlm_iterations={_max_iter}. "
                    "Accepting FINAL_VAR and shipping the best partial report available."
                )
                logger.info("forced_iteration: iteration_budget_exceeded — %s", msg)
                cb = self.on_budget_exceeded or self.on_refusal
                if cb is not None:
                    try:
                        cb(msg)
                    except Exception:  # noqa: BLE001
                        logger.exception("forced_iteration: on_budget_exceeded callback raised")
                self._pending_refusal_code = "iteration_budget_exceeded"
                return (False, None)

        # 0.5. Defensive max-refusals cap.
        if self.refusal_count >= _MAX_REFUSALS_PER_RUN:
            return (False, None)

        # 0.7. BUG-NEW-046: no experiment ever run — refuse FINAL_VAR.
        # A run that never called run_experiment has done no reproducible work,
        # regardless of how many iterations it consumed on planning/implementing.
        if self._total_run_experiments == 0:
            cur = self.current_iteration() if self.current_iteration is not None else 0
            self._pending_refusal_signature = "no_experiment"
            return (True, self._no_experiment_message(cur))

        # Compute repair-iteration refusal eagerly — this check is independent
        # of min_iterations (rubric floor) so it fires even when the rubric
        # policy is disabled via OPENRESEARCH_MIN_RUBRIC_ITERATIONS=0.
        min_repair = int(os.environ.get("OPENRESEARCH_MIN_REPAIR_ITERATIONS", "2"))
        _repair_refuse = (
            min_repair > 0
            and self._last_repair_failure_class is not None
            and self._repair_iter_count < min_repair
        )

        # 1. Rubric-iteration policy disabled — skip rubric checks; only the
        # repair floor (4.6) can still refuse.
        if self.min_iterations <= 0:
            if _repair_refuse:
                return self._build_repair_refusal(min_repair)
            return (False, None)

        if self.rubric_snapshot is None:
            score, target = None, None
        else:
            score, target, _score_iter = self.rubric_snapshot()

        # 2. No rubric data — repair floor OR iteration floor can refuse.
        if score is None or target is None:
            if _repair_refuse:
                return self._build_repair_refusal(min_repair)
            # BUG-LR-013: a model that has never called verify_against_rubric
            # has done strictly less work than one that scored 0.0 — refuse if
            # we haven't reached the iteration floor yet.
            cur = self.current_iteration() if self.current_iteration is not None else 0
            if cur < self.min_iterations:
                msg = (
                    f"FINAL_VAR refused at iteration {cur}: no rubric score recorded yet "
                    f"(min_rubric_iterations={self.min_iterations}). "
                    "Call verify_against_rubric on your latest run_experiment result, "
                    "or call run_experiment if you have not run the baseline yet. "
                    "Do NOT call FINAL_VAR until verify_against_rubric has returned a score."
                )
                self._pending_refusal_signature = "no_rubric"
                return (True, msg)
            # 2.1 Hard-floor mode also closes the no-verify exit (2026-06-12
            # OmniZip attempt 3): a root that NEVER calls verify_against_rubric
            # carries no score/target, sails past the iteration floor, and the
            # rubric-less accept below would ship a fabricated report under a
            # 0.656 best-attempt floor. With REPROLAB_FLOOR_HARD on, at least
            # one real verification is required before any finalize; checks
            # 0/0.3/0.4/0.5 (wall clock, budget, terminal, refusal cap) still
            # dominate above.
            if os.environ.get("REPROLAB_FLOOR_HARD", "").strip() in ("1", "true", "yes"):
                msg = (
                    f"FINAL_VAR refused (hard floor) at iteration {cur}: this run has "
                    "NEVER recorded a rubric score — there is no evidence to report. "
                    "Call verify_against_rubric (it scores the on-disk evidence) to "
                    "establish the real score, then continue the loop until it reaches "
                    "the best-attempt floor. Claims in a final report must trace to "
                    "experiments that actually ran."
                )
                self._pending_refusal_code = "floor_hard"
                self._pending_refusal_signature = "no_rubric"
                return (True, msg)
            return (False, None)

        # 3. Score satisfies target — accept.
        if score >= target:
            return (False, None)

        # 4. Below target AND haven't hit the iteration floor — refuse.
        cur = self.current_iteration() if self.current_iteration is not None else 0
        if cur < self.min_iterations:
            msg = (
                f"rubric overall_score={score:.3f} is below target_score={target:.3f} "
                f"after iteration {cur} (min_rubric_iterations={self.min_iterations}); "
                "call propose_improvements + implement_baseline with repair_context "
                "set to your latest verify_against_rubric result, then run_experiment "
                "again — do NOT call FINAL_VAR until the rubric is satisfied or the "
                "iteration floor is reached."
            )
            self._pending_refusal_signature = "below_target"
            return (True, msg)

        # 4.5. Lane O — iteration floor reached AND rubric below target AND
        # no candidate has been honestly tested. Blanket-declining everything
        # is observer bias, not triage.
        if self.honest_candidate_outcomes is not None:
            try:
                tested = self.honest_candidate_outcomes()
            except Exception:  # noqa: BLE001 — never crash the policy
                tested = 0
            if tested == 0:
                msg = (
                    f"rubric overall_score={score:.3f} is below target_score={target:.3f} "
                    f"and no improvement candidate has been honestly tested yet "
                    f"(zero outcomes with 'promoted'/'failed'/'marginal'). "
                    "Pick ONE candidate from your latest propose_improvements result, "
                    "implement_baseline with that candidate's hypothesis in repair_context, "
                    "run_experiment, verify_against_rubric, then record_candidate_outcome "
                    "with the truthful outcome ('promoted' if rubric improved, 'failed' if "
                    "it didn't). Blanket-declining all candidates without running any is "
                    "observer bias and FINAL_VAR remains refused."
                )
                self._pending_refusal_signature = "below_target"
                return (True, msg)

        # 4.6. PR-α followup — repair iteration floor.  Even when the rubric
        # iteration floor is satisfied, refuse FINAL_VAR if the last
        # run_experiment returned a repairable outcome AND fewer than
        # OPENRESEARCH_MIN_REPAIR_ITERATIONS repair attempts have been made.
        # This prevents the root from giving up after a single preflight
        # failure (e.g. AST-caught code bugs) without trying to fix them.
        if _repair_refuse:
            return self._build_repair_refusal(min_repair)

        # 4.7. Hard best-attempt floor (REPROLAB_FLOOR_HARD=1, default off).
        # Ratchet semantics for multi-attempt climbs: when the target is the
        # prior best attempt's score, the iteration-floor escape hatch below
        # let the 2026-06-12 OmniZip attempt 2 ship a 0.0 report under a
        # 0.656 floor with 7 h of wall clock left. With the flag on, FINAL_VAR
        # stays refused while score < target and time remains. Checks 0 / 0.3
        # / 0.4 / 0.5 still dominate, so a genuinely stuck or out-of-time run
        # ships its best partial instead of never terminating.
        if os.environ.get("REPROLAB_FLOOR_HARD", "").strip() in ("1", "true", "yes"):
            msg = (
                f"FINAL_VAR refused (hard floor): rubric overall_score={score:.3f} is "
                f"below target_score={target:.3f} and wall clock remains. This run's "
                "target is the prior best attempt — shipping below it discards the "
                "attempt. Continue the loop: propose_improvements → implement_baseline "
                "(repair_context = your latest verify_against_rubric result) → "
                "run_experiment → verify_against_rubric. The refusal lifts when the "
                "score reaches the floor or wall clock runs out."
            )
            self._pending_refusal_code = "floor_hard"
            self._pending_refusal_signature = "below_target"
            return (True, msg)

        # 5. Iteration floor reached — accept the partial result.
        return (False, None)

    def _build_repair_refusal(self, min_repair: int) -> tuple[bool, str]:
        """Return the repair-refusal tuple and set the pending SSE event code."""
        failure_class = self._last_repair_failure_class or "unknown"
        msg = (
            f"FINAL_VAR refused: last primitive returned repairable outcome "
            f"'{failure_class}'; {self._repair_iter_count}/{min_repair} repair "
            "iterations completed. Next step: implement_baseline + run_experiment "
            "to fix the violations — do NOT call FINAL_VAR until the repair "
            "floor is reached or the rubric is satisfied."
        )
        self._pending_refusal_code = "forced_repair_iteration"
        self._pending_refusal_signature = "repair_floor"
        return (True, msg)

    # PR-μ Solution C — per-iteration run_experiment tracking.

    _FAILURE_OUTCOMES = frozenset({"repairable", "partial_evidence", "fatal"})

    def record_run_experiment(self, outcome: str) -> None:
        """Append an outcome to the current iteration's run_experiment sequence.
        Called from the run_experiment primitive after computing its outcome."""
        self._experiments_in_iteration.append(outcome)
        self._total_run_experiments += 1
        # run_experiment is a state-changing primitive — clear the no-progress
        # refusal counter so a subsequent refusal does not accumulate toward the
        # degenerate-loop trip (Task 2).
        self.record_state_change()

    def record_state_change(self) -> None:
        """Reset the no-progress refusal counter — a state-changing primitive ran.

        Called by ``record_run_experiment`` and (Task 4) by run.py for
        ``implement_baseline`` / ``build_environment``.  Re-arms the
        degenerate-loop detector so a healthy root that does real work between
        refusals never trips it.
        """
        self._noprogress_refusals = 0
        self._last_refusal_signature = None
        self._degenerate_fired = False

    def register_refusal(self, signature: str) -> None:
        """Update the no-progress refusal counter after a refusal was issued.

        Increments only while consecutive refusals share a signature with no
        intervening state-changing primitive; a new signature resets the run to
        1; a state-changing primitive (record_state_change) resets to 0. Fires
        ``on_degenerate_refusal_loop`` exactly once when the counter first
        reaches ``degenerate_threshold``.  Inert (counter-only, no callback)
        when ``on_degenerate_refusal_loop`` is None.
        """
        if signature == self._last_refusal_signature:
            self._noprogress_refusals += 1
        else:
            self._last_refusal_signature = signature
            self._noprogress_refusals = 1
        if (
            self.on_degenerate_refusal_loop is not None
            and not self._degenerate_fired
            and self._noprogress_refusals >= self.degenerate_threshold
        ):
            self._degenerate_fired = True
            stage: str | None = None
            if self.required_stage is not None:
                try:
                    stage = self.required_stage()
                except Exception:  # noqa: BLE001 — never crash the policy
                    stage = None
            try:
                self.on_degenerate_refusal_loop(
                    {
                        "signature": signature,
                        "count": self._noprogress_refusals,
                        "required_stage": stage,
                    }
                )
            except Exception:  # noqa: BLE001 — emit must never block the policy
                logger.exception(
                    "forced_iteration: on_degenerate_refusal_loop raised"
                )

    def on_iteration_advance(self) -> None:
        """Reset per-iteration trackers when a new REPL turn starts.

        NOTE: deliberately does NOT touch the no-progress refusal counter — that
        counter is cross-turn and resets only on a real state-changing primitive
        (``record_state_change``).  on_iteration_advance fires after EVERY
        refusal, so resetting here would defeat the degenerate-loop detector.
        """
        self._experiments_in_iteration = []

    def should_refuse_final_var(self, current_score: float, iteration_count: int) -> PolicyDecision:
        """Check the two-experiment anti-pattern only; return a PolicyDecision.

        Refuses when the same iteration contains >=2 run_experiment calls and
        the LAST one returned a failure outcome (repairable/partial_evidence/
        fatal). This is the 0.305 Adam pattern: root chained both attempts into
        one turn and tried to FINAL_VAR without a fresh iteration.

        This method is independent of the existing should_refuse() callable-based
        checks; it is called from the FINAL_VAR interceptor BEFORE should_refuse().
        """
        if (
            len(self._experiments_in_iteration) >= 2
            and self._experiments_in_iteration[-1] in self._FAILURE_OUTCOMES
        ):
            last = self._experiments_in_iteration[-1]
            return PolicyDecision(
                refuse=True,
                reason=(
                    f"two run_experiment calls in this iteration; the latter returned "
                    f"'{last}'. End this iteration so the failure surfaces as fresh "
                    f"next-turn context."
                ),
            )
        return PolicyDecision(refuse=False)


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

            # PR-μ Solution C: two-experiment-per-iteration anti-pattern check.
            # Refuses BEFORE the existing should_refuse() so the message is precise.
            current_score = 0.0
            try:
                snap = policy.rubric_snapshot() if policy.rubric_snapshot else None
                if snap and len(snap) >= 1 and snap[0] is not None:
                    current_score = float(snap[0])
            except Exception:
                pass
            current_iter = 0
            try:
                current_iter = int(policy.current_iteration()) if policy.current_iteration else 0
            except Exception:
                pass
            two_exp_decision = policy.should_refuse_final_var(current_score, current_iter)
            if two_exp_decision.refuse:
                policy.refusal_count += 1
                policy.register_refusal("two_experiment_same_iteration")
                message = two_exp_decision.reason
                if policy._noprogress_refusals >= policy.degenerate_threshold:
                    message = policy.escalate_refusal_message(
                        "two_experiment_same_iteration"
                    )
                if policy.on_refusal is not None:
                    try:
                        policy.on_refusal(message)
                    except Exception:
                        logger.exception("forced_iteration: on_refusal (two-exp) raised")
                policy.on_iteration_advance()
                return _build_block_message(variable_name, message)

            refuse, message = policy.should_refuse()
            if not refuse:
                return _original_final_var(self, variable_name)

            assert message is not None  # invariant from should_refuse contract
            policy.refusal_count += 1
            _signature = policy._pending_refusal_signature or "forced_iteration"
            policy.register_refusal(_signature)

            # Task 3 — escalate once the no-progress counter has reached the
            # degenerate threshold (must be AFTER register_refusal so the count
            # is the true post-registration value). Healthy runs (1-2 refusals)
            # never reach it; no accept/refuse decision changes.
            if policy._noprogress_refusals >= policy.degenerate_threshold:
                message = policy.escalate_refusal_message(_signature)

            # Notify the policy's callback so the orchestrator can surface a
            # run_warning SSE event. Route to on_repair_refusal when the
            # pending code signals a repair refusal and the callback is set;
            # otherwise fall back to the standard on_refusal. Must not raise.
            _code = getattr(policy, "_pending_refusal_code", "forced_iteration")
            if _code == "forced_repair_iteration" and policy.on_repair_refusal is not None:
                _cb = policy.on_repair_refusal
            else:
                _cb = policy.on_refusal
            if _cb is not None:
                try:
                    _cb(message)
                except Exception:  # noqa: BLE001 — defensive; emit failures must not block
                    logger.exception("forced_iteration: on_refusal callback raised")

            # PR-μ Solution C: reset per-iteration trackers since the root will
            # start a fresh iteration after the refusal block message lands.
            policy.on_iteration_advance()
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
    "PolicyDecision",
    "apply_forced_iteration_patch",
    "forced_iteration_policy",
]
