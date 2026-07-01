"""env_adapters — the paper-agnostic environment-provisioning registry (Phase 1a).

Package exports the base adapter contract plus the three SDAR-specific
concrete adapters (``AlfworldAdapter`` / ``WebShopAdapter`` / ``SearchQaAdapter``)
and ``resolve_adapter``, the name-routing helper that replaces
``EnvCacheManager.setup``'s alias ladder.
"""

from __future__ import annotations

from backend.services.runtime.env_adapters.alfworld import AlfworldAdapter
from backend.services.runtime.env_adapters.base import (
    EnvironmentAdapter,
    EnvSetupResult,
    HealthReport,
    ProvisionCtx,
    SmokeResult,
)
from backend.services.runtime.env_adapters.registry import resolve_adapter
from backend.services.runtime.env_adapters.search_qa import SearchQaAdapter
from backend.services.runtime.env_adapters.webshop import WebShopAdapter

__all__ = [
    "EnvironmentAdapter",
    "EnvSetupResult",
    "SmokeResult",
    "HealthReport",
    "ProvisionCtx",
    "AlfworldAdapter",
    "WebShopAdapter",
    "SearchQaAdapter",
    "resolve_adapter",
]
