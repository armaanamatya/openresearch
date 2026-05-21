"""RLM root-orchestrator package.

This package implements Algorithm 1 from arXiv 2512.24601 (Zhang, Kraska,
Khattab — *Recursive Language Models*) as the orchestrator for paper
reproduction runs. The 14-stage `PipelineStage` enum-driven loop is
superseded by a Python REPL the root LLM writes code in, with stage agents
exposed as callable primitives.

Phase 1 (#58) artifact — these modules are stubs. Phase 2 (#59) will
implement them following the contract in `docs/rlm-pivot-mapping.md` and
the brief at `docs/design/rlm-pivot-brief.md`.
"""

# Phase-1 skeleton modules (repl_host / root_loop / sub_call) are superseded by
# the rlms library — dead code, NOT re-exported here; deleted in Phase 6 (#63).

# --- primitive layer (#59) --------------------------------------------------
from backend.agents.rlm.primitives import PRIMITIVE_DESCRIPTIONS, PRIMITIVE_REGISTRY
from backend.agents.rlm.context import RunContext
from backend.agents.rlm.binding import build_custom_tools

# --- orchestrator (#60) -----------------------------------------------------
from backend.agents.rlm.models import resolve_root_model
from backend.agents.rlm.run import RLMRunResult, run_pipeline_rlm
from backend.agents.rlm.system_prompt import build_system_prompt

__all__ = [
    # primitive layer (#59)
    "PRIMITIVE_REGISTRY",
    "PRIMITIVE_DESCRIPTIONS",
    "RunContext",
    "build_custom_tools",
    # orchestrator (#60)
    "build_system_prompt",
    "resolve_root_model",
    "RLMRunResult",
    "run_pipeline_rlm",
]
