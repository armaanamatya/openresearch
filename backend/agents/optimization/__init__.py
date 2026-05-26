"""GEPA prompt optimization — registry + override seam.

Phase 1 foundation (no GEPA adapter yet). See
``docs/superpowers/specs/2026-05-25-gepa-prompt-optimization-design.md`` and
``docs/superpowers/specs/2026-05-26-gepa-phase0-audit.md``.

Public surface:
- ``MutableRegion`` / ``REGIONS`` — addressable mutable prompt regions.
- ``AGENT_TO_COMPONENT`` — maps registry ``agent_id`` → ``component_id``.
- ``PromptOverrideContext`` / ``PROMPT_OVERRIDES`` — thread-local override stack.

By default (no active override), every consumer reads ``REGIONS[cid].default_text``
which is the original module-level prompt string. Production runs see exactly
today's prompts.
"""

from backend.agents.optimization.mutable_regions import (
    AGENT_TO_COMPONENT,
    REGIONS,
    MutableRegion,
)
from backend.agents.optimization.prompt_overrides import (
    PROMPT_OVERRIDES,
    PromptOverrideContext,
)

__all__ = [
    "AGENT_TO_COMPONENT",
    "MutableRegion",
    "PROMPT_OVERRIDES",
    "PromptOverrideContext",
    "REGIONS",
]
