"""Citation primitives shared by the workspace and any agent claim.

Defined locally for now; proposed upstream to #8 in a follow-up PR.
Import from this module across the codebase so the move to #8 is one
import change.

Defense in depth for the citation invariant (spec §5.6):
  1. `NonEmptyCitations` Pydantic alias rejects empty tuples at validation.
  2. `Citation.model_construct` is overridden to raise — bypass attempts
     fail at construction time too.
  3. `EventStore.append` re-validates payloads against their registered
     Pydantic class on the way in.
  4. `StoredEvent.into()` re-validates on the way out (catches storage
     tampering and dict-bypass attempts).
"""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field

from backend.messaging.event import InvariantBypassError


class Citation(BaseModel):
    """A single piece of evidence backing an agent claim.

    `source_id` and (optional) `chunk_id` resolve to entries in the
    SourcesProjection / chunks store. `quote` is the exact evidence text
    the agent grounded its claim in. `locator` is a human-readable pointer
    rendered by the dashboard ("PPO §3.2", "github.com/openai/baselines/README.md:42").
    `confidence` is 0..1; default 1.0 means "directly cited from primary source".
    """

    model_config = ConfigDict(frozen=True)

    source_id: str
    chunk_id: str | None = None
    quote: str
    locator: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: Any) -> Self:
        """Banned: bypassing validation on a Citation undermines the
        invariant that every citation has a real source pointer.

        Use `Citation(...)` for validated construction or
        `Citation.model_validate(...)` for validated deserialization."""
        raise InvariantBypassError(
            "Citation.model_construct is banned to preserve evidence invariants. "
            "Use Citation(...) or Citation.model_validate(...) instead."
        )


# Pydantic-validated typed alias: every list[Citation] in an event payload
# must be non-empty. Enforced at construction; bypasses (model_construct on
# the surrounding event class) are blocked by DomainEvent.model_construct
# raising InvariantBypassError.
NonEmptyCitations = Annotated[tuple[Citation, ...], Field(min_length=1)]


__all__ = ["Citation", "NonEmptyCitations"]
