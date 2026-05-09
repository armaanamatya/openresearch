"""Spawn-policy guards for delegation depth and fan-out limits."""

from __future__ import annotations

from pydantic import BaseModel

from backend.persistence.database import Database
from backend.services.orchestration.delegation import DelegationService


class SpawnCheckResult(BaseModel):
    allowed: bool
    reason: str = ""


class EscalationResult(BaseModel):
    should_escalate: bool
    reason: str = ""


class SpawnPolicy:
    def __init__(
        self,
        db: Database,
        max_depth: int = 3,
        max_fanout: int = 5,
        escalation_threshold: int = 3,
    ) -> None:
        self._db = db
        self._max_depth = max_depth
        self._max_fanout = max_fanout
        self._escalation_threshold = escalation_threshold

    def check_can_spawn(
        self,
        parent_task_id: str,
        delegation_service: DelegationService,
    ) -> bool:
        result = self.check_can_spawn_detailed(parent_task_id, delegation_service)
        return result.allowed

    def check_can_spawn_detailed(
        self,
        parent_task_id: str,
        delegation_service: DelegationService,
    ) -> SpawnCheckResult:
        # Check depth
        depth = delegation_service.get_depth(parent_task_id)
        if depth >= self._max_depth:
            return SpawnCheckResult(
                allowed=False,
                reason=f"Max depth exceeded: current depth is {depth}, max is {self._max_depth}",
            )

        # Check fan-out
        children = delegation_service.get_children(parent_task_id)
        if len(children) >= self._max_fanout:
            return SpawnCheckResult(
                allowed=False,
                reason=f"Max fan-out exceeded: {len(children)} children, max is {self._max_fanout}",
            )

        return SpawnCheckResult(allowed=True)

    def check_escalation(
        self,
        parent_task_id: str,
        child_failure_count: int,
    ) -> EscalationResult:
        if child_failure_count >= self._escalation_threshold:
            return EscalationResult(
                should_escalate=True,
                reason=f"{child_failure_count} child failures exceeded threshold of {self._escalation_threshold}",
            )
        return EscalationResult(should_escalate=False)
