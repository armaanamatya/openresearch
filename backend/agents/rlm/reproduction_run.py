"""Checkpointed, fail-soft whole-run lifecycle controller (Phase 1c, Unit C).

``ReproductionRun`` drives a :class:`~backend.services.runtime.compute_provider.ComputeProvider`
through the full reproduction lifecycle -- RESOLVE -> TRIAGE -> PROVISION_CPU
-> PROVISION_ASSETS -> GREEN_GATE -> ACQUIRE_GPU -> RUN -> WATCH -> COLLECT ->
RELEASE_GPU -> FINALIZE -> TEARDOWN -- wiring the Phase-1b feasibility gates
(:class:`~backend.services.runtime.feasibility_triage.FeasibilityTriage`,
:class:`~backend.agents.resilience.budget.RunBudget`) so that
``provider.acquire_gpu`` -- the only method that starts GPU billing (see the
``ComputeProvider`` docstring) -- is reachable ONLY once triage has decided
the scope is feasible (``PROCEED``/``DOWN_SCOPE``), an operator-supplied
"green gate" (e.g. an environment-standability / asset-presence check) has
passed, and the run budget still has headroom.

Two distinct terminal vocabularies are in play, intentionally: ``STATES`` is
the ordered list of state-MACHINE positions a normal run passes through (used
for checkpointing "how far did we get"); ``ReproductionOutcome.state`` is the
OUTCOME classification returned by :meth:`ReproductionRun.run` -- one of
``"FINALIZE"`` (success), ``"PLAN_ONLY"`` (infeasible even after triage's own
scope-trim, so nothing was ever provisioned), ``"TEARDOWN"`` (aborted or
released without a usable result), or ``"RECOVERED"`` (compute vanished
mid-run but artifacts were salvaged via ``provider.recover()``). The
checkpoint file mirrors both vocabularies: it records the ``STATES`` value on
every normal transition, and the outcome label itself as the very last write
in a terminal branch (e.g. ``"PLAN_ONLY"``/``"RECOVERED"``), so a resuming
supervisor can tell "still mid-flight in state X" from "already terminal with
outcome Y" without inventing a fourth vocabulary.

Fail-soft throughout, per the Phase 1c plan's Global Constraints: every
``ComputeProvider`` call is wrapped (see ``_safe_call``) so a raise degrades
to a safe ``TEARDOWN`` (or, mid-run, an emergency-shutdown attempt at
``RECOVERED``) terminal -- ``run()`` itself never propagates an exception to
its caller, not even a bug in this module's own orchestration code (the
outer ``try/except`` in ``run()`` is the last-ditch net for that). Checkpoint
writes are best-effort: a write error is swallowed and never aborts the run.

Standalone in this unit: this controller is a peer entrypoint, not (yet)
wired into the live ``backend/agents/rlm/run.py`` path -- it is exercised
here only via the injected ``FakeComputeProvider`` test harness
(``tests/rlm/test_reproduction_run.py``). Live-cloud wiring is a later,
explicitly opt-in step (``OPENRESEARCH_UNIFIED_RUN``).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from backend.agents.resilience.failures import BudgetExhausted

if TYPE_CHECKING:
    from backend.services.runtime.compute_provider import (
        ComputeLease,
        ComputeProvider,
        ReportBundle,
    )
    from backend.services.runtime.feasibility_triage import FeasibilityTriage
    from backend.services.runtime.gpu_catalog import GpuSku
    from backend.services.runtime.run_plan import RunPlan


_CHECKPOINT_FILENAME = "reproduction_run.json"


@dataclass(frozen=True)
class ReproductionOutcome:
    """The terminal result of a :class:`ReproductionRun`."""

    state: str  # "FINALIZE" | "PLAN_ONLY" | "TEARDOWN" | "RECOVERED"
    decision: str  # the triage decision: "PROCEED" | "DOWN_SCOPE" | "PLAN_ONLY" | ""
    report: "ReportBundle | None"
    reasons: tuple[str, ...]
    gpu_acquired: bool  # invariant witness: True only if the gates passed


class ReproductionRun:
    """Checkpointed, fail-soft state machine driving a ``ComputeProvider``.

    See the module docstring for the full lifecycle + terminal-vocabulary
    design. ``green_gate`` defaults to always-pass (``lambda lease: True``)
    so a caller that has no environment-standability check yet still gets a
    working (if permissive) state machine; a real gate is injected by the
    caller once one exists. ``clock`` defaults to ``time.monotonic`` and is
    injectable so tests can control elapsed-GPU-time budget accounting
    deterministically.
    """

    STATES = (
        "RESOLVE",
        "TRIAGE",
        "PROVISION_CPU",
        "PROVISION_ASSETS",
        "GREEN_GATE",
        "ACQUIRE_GPU",
        "RUN",
        "WATCH",
        "COLLECT",
        "RELEASE_GPU",
        "FINALIZE",
        "TEARDOWN",
    )

    def __init__(
        self,
        *,
        plan: "RunPlan",
        provider: "ComputeProvider",
        triage: "FeasibilityTriage",
        sku: "GpuSku",
        state_dir: "Path",
        green_gate: "Callable[[ComputeLease], bool] | None" = None,
        sync_each_watch: bool = True,
        clock: "Callable[[], float]" = time.monotonic,
    ) -> None:
        self._plan = plan
        self._provider = provider
        self._triage = triage
        self._sku = sku
        self._state_dir = Path(state_dir)
        self._green_gate = green_gate if green_gate is not None else (lambda _lease: True)
        self._sync_each_watch = sync_each_watch
        self._clock = clock

        # Checkpoint witnesses; ``sync_log`` also doubles as the directly
        # inspectable "periodic sync witness" (invariant 5) for tests/
        # supervisors -- the checkpoint file's fixed 3-field schema has no
        # room for a per-poll history, so it is tracked here instead.
        self._decision = ""
        self._lease_ref = ""
        self.sync_log: list[bool] = []

    # -- checkpointing --------------------------------------------------

    def _checkpoint(self, state: str) -> None:
        """Atomically persist ``{state, decision, lease_ref}``.

        Fail-soft (invariant 7): any error -- the directory can't be
        created, the temp file can't be written, the rename fails -- is
        swallowed. Checkpointing is an observability aid, never a run
        precondition.
        """
        tmp_path: str | None = None
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "state": state,
                "decision": self._decision,
                "lease_ref": self._lease_ref,
            }
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._state_dir), prefix=".reproduction_run.", suffix=".tmp"
            )
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh)
            os.replace(tmp_path, str(self._state_dir / _CHECKPOINT_FILENAME))
        except Exception:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # -- fail-soft provider-call boundary --------------------------------

    def _safe_call(self, fn: "Callable[[], object]", *, label: str) -> "tuple[object | None, str | None]":
        """Call ``fn()``; on any exception return ``(None, '<label>_error:<exc>')`` instead of raising."""
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001 - deliberate fail-soft boundary
            return None, f"{label}_error:{exc}"

    # -- the state machine -----------------------------------------------

    def run(self) -> ReproductionOutcome:
        """Drive the machine to a terminal outcome. Never raises."""
        try:
            return self._run()
        except Exception as exc:  # last-ditch net: a bug in this module's own
            # orchestration code (not a provider call, which _safe_call
            # already isolates) must still never escape run().
            self._checkpoint("TEARDOWN")
            return ReproductionOutcome(
                state="TEARDOWN",
                decision=self._decision,
                report=None,
                reasons=(f"unhandled_error:{exc}",),
                gpu_acquired=False,
            )

    def _run(self) -> ReproductionOutcome:
        self._checkpoint("RESOLVE")

        decision = self._triage.triage(self._plan, self._sku)
        self._decision = decision.decision
        self._checkpoint("TRIAGE")

        if decision.decision == "PLAN_ONLY":
            # Invariant 1: an infeasible triage never provisions anything.
            self._checkpoint("PLAN_ONLY")
            return ReproductionOutcome(
                state="PLAN_ONLY",
                decision=decision.decision,
                report=None,
                reasons=tuple(decision.reasons),
                gpu_acquired=False,
            )

        # Invariant 2: DOWN_SCOPE adopts the trimmed scope before any
        # provider call runs, so every downstream call sees the fitted scope.
        active_plan = self._plan
        if decision.decision == "DOWN_SCOPE" and decision.scope is not None:
            active_plan = replace(self._plan, scope=decision.scope)

        cap, err = self._safe_call(lambda: self._provider.preflight(active_plan), label="preflight")
        if err is not None:
            return self._abort_before_gpu(None, reason=err, decision=decision.decision)
        if cap is not None and not cap.available:
            return self._abort_before_gpu(
                None,
                reason=f"capacity_unavailable:{cap.reason}",
                decision=decision.decision,
            )

        lease, err = self._safe_call(
            lambda: self._provider.provision_cpu(active_plan), label="provision_cpu"
        )
        if err is not None or lease is None:
            return self._abort_before_gpu(
                None, reason=err or "provision_cpu_returned_none", decision=decision.decision
            )
        self._lease_ref = lease.ref
        self._checkpoint("PROVISION_CPU")

        _, err = self._safe_call(
            lambda: self._provider.stage(lease, None, active_plan), label="stage"
        )
        self._checkpoint("PROVISION_ASSETS")
        if err is not None:
            return self._abort_before_gpu(lease, reason=err, decision=decision.decision)

        try:
            green = bool(self._green_gate(lease))
            gate_err = None
        except Exception as exc:  # noqa: BLE001 - the injected gate is caller code, not trusted
            green = False
            gate_err = f"green_gate_error:{exc}"
        self._checkpoint("GREEN_GATE")
        if not green:
            return self._abort_before_gpu(
                lease, reason=gate_err or "green_gate_failed", decision=decision.decision
            )

        budget = getattr(active_plan, "budget", None)
        if budget is not None:
            try:
                budget.check_gpu_hours(gpu_hours_used=0.0, agent_id="reproduction_run")
                budget.check_run_gpu_usd(cumulative_pod_usd=0.0, agent_id="reproduction_run")
            except BudgetExhausted as exc:
                return self._abort_before_gpu(
                    lease,
                    reason=f"budget_exhausted_pre_gpu:{exc}",
                    decision=decision.decision,
                )

        # Invariant 1: acquire_gpu -- the ONLY method that starts GPU billing
        # -- is reachable exactly here: TRIAGE in {PROCEED, DOWN_SCOPE} AND
        # GREEN_GATE passed AND budget ok, all gated above.
        lease, err = self._safe_call(lambda: self._provider.acquire_gpu(lease), label="acquire_gpu")
        if err is not None or lease is None:
            return self._abort_before_gpu(
                lease, reason=err or "acquire_gpu_returned_none", decision=decision.decision
            )
        self._lease_ref = lease.ref or self._lease_ref
        self._checkpoint("ACQUIRE_GPU")

        handle, err = self._safe_call(
            lambda: self._provider.launch(lease, active_plan), label="launch"
        )
        self._checkpoint("RUN")
        if err is not None or handle is None:
            return self._emergency_shutdown(
                lease, reason=err or "launch_returned_none", decision=decision.decision
            )

        self._checkpoint("WATCH")
        gpu_start = self._clock()
        try:
            for status in self._provider.watch(handle):
                # Invariant 5: every poll's synced flag is recorded, in
                # order, BEFORE any decision is made on it -- so an
                # emergency shutdown always has the "last synced" witness.
                self.sync_log.append(bool(status.synced))

                if status.state == "stopped_uncollected":
                    # Invariant 4: compute vanished before COLLECT could run.
                    return self._emergency_shutdown(
                        lease, reason="stopped_uncollected", decision=decision.decision
                    )

                if status.state == "completed":
                    # The in-VM child wrote final_report.json: the run finished
                    # cleanly. Break BEFORE the budget block so a coincident
                    # budget-ceiling observation cannot re-route this poll to
                    # _emergency_shutdown -> RECOVERED. Takes the graceful
                    # COLLECT -> RELEASE_GPU -> FINALIZE path (invariant 3).
                    break

                if budget is not None:
                    # Invariant 6: a budget breach observed at WATCH time
                    # (real elapsed GPU seconds via the injected clock, not
                    # the triage's up-front estimate) is also an emergency.
                    now = self._clock()
                    elapsed_h = max(0.0, (now - gpu_start) / 3600.0)
                    try:
                        budget.check_gpu_hours(gpu_hours_used=elapsed_h, agent_id="reproduction_run")
                        usd_rate = float(getattr(self._sku, "approx_usd_per_hr", 0.0) or 0.0)
                        budget.check_run_gpu_usd(
                            cumulative_pod_usd=elapsed_h * usd_rate, agent_id="reproduction_run"
                        )
                    except BudgetExhausted as exc:
                        return self._emergency_shutdown(
                            lease, reason=f"budget_exhausted:{exc}", decision=decision.decision
                        )
        except Exception as exc:  # noqa: BLE001 - provider.watch is a generator; a mid-stream raise counts too
            return self._emergency_shutdown(
                lease, reason=f"watch_error:{exc}", decision=decision.decision
            )

        # Invariant 3: graceful terminal order COLLECT -> RELEASE_GPU -> TEARDOWN.
        report, _ = self._safe_call(lambda: self._provider.collect(handle), label="collect")
        self._checkpoint("COLLECT")

        self._safe_call(lambda: self._provider.release_gpu(lease), label="release_gpu")
        self._checkpoint("RELEASE_GPU")

        self._checkpoint("FINALIZE")

        self._safe_call(lambda: self._provider.teardown(lease, reason="finalize"), label="teardown")
        self._checkpoint("TEARDOWN")

        return ReproductionOutcome(
            state="FINALIZE",
            decision=decision.decision,
            report=report,
            reasons=tuple(decision.reasons),
            gpu_acquired=True,
        )

    # -- terminal helpers --------------------------------------------------

    def _abort_before_gpu(
        self, lease: "ComputeLease | None", *, reason: str, decision: str = ""
    ) -> ReproductionOutcome:
        """Graceful abort before ``acquire_gpu`` ever ran (invariant 1's negative side)."""
        self._checkpoint("TEARDOWN")
        if lease is not None:
            self._safe_call(lambda: self._provider.teardown(lease, reason=reason), label="teardown")
        return ReproductionOutcome(
            state="TEARDOWN",
            decision=decision or self._decision,
            report=None,
            reasons=(reason,),
            gpu_acquired=False,
        )

    def _emergency_shutdown(
        self, lease: "ComputeLease", *, reason: str, decision: str = ""
    ) -> ReproductionOutcome:
        """Invariants 4 + 6: bypass COLLECT, attempt ``recover()``, degrade to TEARDOWN.

        A non-``None`` recovered bundle wins as ``RECOVERED``; otherwise the
        run ends at ``TEARDOWN`` with the last synced state already captured
        in ``sync_log`` (invariant 5) -- "last synced," never "stranded."
        """
        report, _ = self._safe_call(lambda: self._provider.recover(lease.ref), label="recover")
        self._safe_call(lambda: self._provider.release_gpu(lease), label="release_gpu")

        if report is not None:
            self._checkpoint("RECOVERED")
            self._safe_call(
                lambda: self._provider.teardown(lease, reason="recovered"), label="teardown"
            )
            return ReproductionOutcome(
                state="RECOVERED",
                decision=decision or self._decision,
                report=report,
                reasons=(reason,),
                gpu_acquired=True,
            )

        self._checkpoint("TEARDOWN")
        self._safe_call(lambda: self._provider.teardown(lease, reason=reason), label="teardown")
        return ReproductionOutcome(
            state="TEARDOWN",
            decision=decision or self._decision,
            report=None,
            reasons=(reason,),
            gpu_acquired=True,
        )


__all__ = ["ReproductionOutcome", "ReproductionRun"]
