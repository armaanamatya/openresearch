"""Command base class + CommandId.

Commands carry intent into application services. Unlike events, they
are not stored — their *outcome* (the resulting events) is what gets
recorded. Commands carry a `command_id` for idempotency: re-issuing
the same command on the same aggregate is a no-op (see IdempotencyTable).
"""

from __future__ import annotations

from typing import NewType

from pydantic import BaseModel, ConfigDict, Field

from backend.messaging.envelope import _ulid

CommandId = NewType("CommandId", str)


def new_command_id() -> CommandId:
    return CommandId(f"cmd_{_ulid()}")


class Command(BaseModel):
    """Base class for every command in our system.

    Commands are immutable. Subclasses are Pydantic models with
    payload fields plus an inherited `command_id` (auto-generated
    on construction unless the caller pins it for idempotency tests).
    """

    model_config = ConfigDict(frozen=True)

    # Field(default_factory=...) is the Pydantic-recommended pattern;
    # mypy sees the type-narrowed CommandId and the runtime default
    # comes from new_command_id().
    command_id: CommandId = Field(default_factory=new_command_id)  # type: ignore[assignment]


__all__ = ["Command", "CommandId", "new_command_id"]
