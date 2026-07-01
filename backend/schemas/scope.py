"""Blackboard scope visibility levels.

Mirrors PRD §1078-1082. Defined as a typed enum for our event payloads.
(The original `backend.services.orchestration.blackboard` service was
removed as dead code; the enum stays because event schemas use it.)
"""

from __future__ import annotations

from enum import Enum


class Scope(str, Enum):
    """Visibility of a workspace variable / blackboard record."""

    private_to_parent = "private_to_parent"
    """Only the spawning agent and its direct parent see it."""

    branch_shared = "branch_shared"
    """Shared across an improvement branch (one delegation lineage)."""

    global_verified = "global_verified"
    """Promoted after verifier confirmation; visible globally."""


__all__ = ["Scope"]
