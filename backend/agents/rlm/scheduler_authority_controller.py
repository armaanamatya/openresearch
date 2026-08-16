"""Controller facade for the receipt-gated authoritative branch tree.

This module is deliberately smaller than the reproduction executor.  It owns
the authority-specific lifecycle around that executor:

``spec -> durable branches -> atomic launch claim -> verified receipt -> ASHA``.

The actual runner remains responsible for materialising metrics, checkpoint
state, evidence bundles, and provider-accounted GPU cost.  The controller never
uses a final-report grade and never launches work merely because a score looks
promising.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.agents.rlm.scheduler_evidence import BranchRungReceipt, write_verified_receipt
from backend.agents.rlm.scheduler_runtime import (
    AuthorityResult,
    BranchTemplate,
    EventSink,
    SchedulerAuthoritySpec,
    SchedulerBranch,
    SchedulerRuntimeError,
    SchedulerTreeRuntime,
)


@dataclass(frozen=True)
class SchedulerLaunch:
    """An atomically claimed, evidence-bound branch execution request."""

    branch_id: str
    branch_type: str
    hypothesis_fingerprint: str
    rung: int
    from_step: int
    to_step: int
    seed: int
    parent_branch_id: str | None
    resume_checkpoint_path: str | None
    is_safety_bracket: bool


class SchedulerAuthorityController:
    """Authoritative tree lifecycle over one explicit paper-owned spec."""

    def __init__(
        self,
        run_dir: Path | str,
        *,
        campaign_id: str,
        spec: SchedulerAuthoritySpec,
        event_sink: EventSink | None = None,
    ) -> None:
        if spec.ladder.paper_ref == "":
            raise ValueError("authority spec must name a paper")
        self.spec = spec
        self.runtime = SchedulerTreeRuntime(
            run_dir, campaign_id=campaign_id, ladder=spec.ladder, event_sink=event_sink,
        )

    @property
    def branches(self) -> Mapping[str, SchedulerBranch]:
        return self.runtime.branches

    def bootstrap(self) -> int:
        """Idempotently materialise the declared mixed branch portfolio."""
        pending = list(self.spec.branches)
        registered = 0
        while pending:
            next_pending: list[BranchTemplate] = []
            progressed = False
            present = self.runtime.branches
            for template in pending:
                if template.parent_branch_id is not None and template.parent_branch_id not in present:
                    next_pending.append(template)
                    continue
                if self.runtime.register_branch(
                    branch_id=template.branch_id,
                    branch_type=template.branch_type,
                    hypothesis_fingerprint=template.hypothesis_fingerprint,
                    parent_branch_id=template.parent_branch_id,
                    rung=0,
                    is_safety_bracket=template.is_safety_bracket,
                    seed=template.seed,
                ):
                    registered += 1
                progressed = True
            if not next_pending:
                return registered
            if not progressed:
                raise SchedulerRuntimeError("authority spec contains an unresolvable branch parent")
            pending = next_pending
        return registered

    def claim_launches(self, *, max_parallel: int | None = None) -> tuple[SchedulerLaunch, ...]:
        """Claim launch slots without exceeding the hard branch/GPU cap."""
        limit = self.spec.a100_cap if max_parallel is None else max_parallel
        if type(limit) is not int or limit < 1:
            raise SchedulerRuntimeError("max_parallel must be a positive integer")
        outstanding = sum(1 for branch in self.runtime.branches.values() if branch.state == "launching")
        available = self.spec.a100_cap - outstanding
        if available <= 0:
            return ()
        branches = self.runtime.claim_queued_branches(limit=min(limit, available))
        launches: list[SchedulerLaunch] = []
        for branch in branches:
            if branch.seed is None:
                # A launch without an explicit seed would break deterministic
                # receipt provenance; return its durable claim before raising.
                self.runtime.release_launch_claim(branch.branch_id)
                raise SchedulerRuntimeError("authority branch lacks a pinned seed")
            from_step = 0 if branch.rung == 0 else self.spec.ladder.rung_steps[branch.rung - 1]
            launches.append(SchedulerLaunch(
                branch_id=branch.branch_id,
                branch_type=branch.branch_type,
                hypothesis_fingerprint=branch.hypothesis_fingerprint,
                rung=branch.rung,
                from_step=from_step,
                to_step=self.spec.ladder.rung_steps[branch.rung],
                seed=branch.seed,
                parent_branch_id=branch.parent_branch_id,
                resume_checkpoint_path=branch.checkpoint_path,
                is_safety_bracket=branch.is_safety_bracket,
            ))
        return tuple(launches)

    def release_launch(self, branch_id: str) -> SchedulerBranch:
        return self.runtime.release_launch_claim(branch_id)

    def record_cell_receipt(
        self,
        raw_receipt: Mapping[str, Any],
        *,
        attest: Callable[[Mapping[str, Any]], None],
    ) -> BranchRungReceipt:
        """Publish an evidence-complete cell boundary to the tree runtime."""
        branch_id = raw_receipt.get("branch_id")
        if not isinstance(branch_id, str) or branch_id not in self.runtime.branches:
            raise SchedulerRuntimeError("receipt names an unknown authority branch")
        path = write_verified_receipt(
            raw_receipt,
            ladder=self.spec.ladder,
            run_dir=self.runtime.run_dir,
            campaign_id=self.runtime.campaign_id,
            attest=attest,
        )
        return self.runtime.record_receipt(branch_id, path)

    def decide_rung(
        self,
        *,
        rung: int,
        provider_gpu_usd_by_branch: Mapping[str, float],
    ) -> AuthorityResult:
        """Apply the explicit, two-meter authority policy to one ready cohort."""
        expected = tuple(sorted(
            branch.branch_id
            for branch in self.runtime.branches.values()
            if branch.rung == rung and branch.state in {"queued", "launching", "awaiting_receipt"}
        ))
        return self.runtime.decide_ready_rung(
            rung=rung,
            provider_gpu_usd_by_branch=provider_gpu_usd_by_branch,
            gpu_usd_budget=self.spec.gpu_usd_budget,
            a100_cap=self.spec.a100_cap,
            safety_gpu_usd_budget=self.spec.safety_gpu_usd_budget,
            discovery_gpu_usd_budget=self.spec.discovery_gpu_usd_budget,
            expected_branch_ids=expected,
            eta=self.spec.eta,
            noise_floor=self.spec.noise_floor,
        )

    def revive(self, branch_id: str) -> SchedulerBranch:
        return self.runtime.revive_branch(branch_id)

    def reconcile_lineage(self) -> int:
        return self.runtime.reconcile_events()
