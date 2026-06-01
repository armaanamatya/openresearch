"""SSEGepaCallback: bridges GEPA callback events → ctx.emit() → dashboard_events.jsonl."""
from __future__ import annotations

import logging
from typing import Any

from backend.agents.rlm.sse_bridge import (
    build_gepa_candidate_accepted,
    build_gepa_candidate_proposed,
    build_gepa_candidate_rejected,
)

logger = logging.getLogger(__name__)


class SSEGepaCallback:
    """GEPA callback that forwards optimization events to the run's SSE stream.

    Pass to gepa.optimize(callbacks=[SSEGepaCallback(ctx, primitive_name)]).
    Implements duck-typed GEPA callback protocol (no base class needed).
    """

    def __init__(self, *, ctx: Any, primitive_name: str) -> None:
        self._ctx = ctx
        self._name = primitive_name
        self._seed_score: float = 0.0
        self._iteration: int = 0

    def _emit(self, event: dict) -> None:
        if self._ctx.emit is not None:
            try:
                self._ctx.emit(event)
            except Exception as exc:
                logger.debug("SSEGepaCallback emit error: %s", exc)

    def on_optimization_start(self, event: dict) -> None:
        pass  # gepa_phase_start emitted by gepa_pre_call before optimize() is called

    def on_iteration_start(self, event: dict) -> None:
        self._iteration = event.get("iteration", self._iteration + 1)

    def on_proposal_end(self, event: dict) -> None:
        new_instructions = event.get("new_instructions", {})
        prompt_text = next(iter(new_instructions.values()), "") if new_instructions else ""
        self._emit(build_gepa_candidate_proposed(
            primitive_name=self._name,
            iteration=self._iteration,
            candidate_id=f"{self._name}-{self._iteration}",
            prompt_preview=prompt_text,
            parent_id=None,
        ))

    def on_candidate_accepted(self, event: dict) -> None:
        new_score = event.get("new_score", 0.0)
        self._emit(build_gepa_candidate_accepted(
            primitive_name=self._name,
            candidate_id=f"{self._name}-{self._iteration}",
            score=new_score,
            score_delta=new_score - self._seed_score,
        ))

    def on_candidate_rejected(self, event: dict) -> None:
        self._emit(build_gepa_candidate_rejected(
            primitive_name=self._name,
            candidate_id=f"{self._name}-{self._iteration}",
            reason=event.get("reason", "no_improvement"),
            score=event.get("new_score", 0.0),
        ))

    def set_seed_score(self, score: float) -> None:
        self._seed_score = score
