"""resolve_adapter — name-routing across a list of EnvironmentAdapter instances.

Part of the provisioning-seam refactor (see
``docs/history/plans/2026-07-01-phase-1a-1b-provisioning-seam-and-gates.md``):
replaces ``EnvCacheManager.setup``'s ``if``/``elif`` alias ladder with
adapter-owned ``applies()`` matching, so adding a new paper-specific
environment means adding a new adapter to the list — no ladder to extend.
"""

from __future__ import annotations

from typing import Sequence

from backend.services.runtime.env_adapters.base import EnvironmentAdapter

__all__ = ["resolve_adapter"]


def resolve_adapter(
    env_name: str, adapters: "Sequence[EnvironmentAdapter]"
) -> "EnvironmentAdapter | None":
    """Return the first adapter in ``adapters`` whose ``applies(env_name)`` is True."""
    for adapter in adapters:
        if adapter.applies(env_name):
            return adapter
    return None
