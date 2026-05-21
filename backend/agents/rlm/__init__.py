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

# Phase-1 skeleton re-exports kept (repl_host / root_loop / sub_call are
# superseded by the rlms library but are NOT deleted until Phase 6 cleanup).
# NOTE: the hand-built ReplHost/RootLoop/llm_query/rlm_query are dead code;
# they are omitted from re-exports here to keep the public surface clean for
# the #60 API.  Delete in Phase 6 (#63).

from backend.agents.rlm.primitives import PRIMITIVE_REGISTRY

# --- Phase 3 (#60) public API -----------------------------------------------
from backend.agents.rlm.models import resolve_root_model
from backend.agents.rlm.run import RLMRunResult, run_pipeline_rlm
from backend.agents.rlm.system_prompt import build_system_prompt

__all__ = [
    # primitives (#59)
    "PRIMITIVE_REGISTRY",
    # orchestrator (#60)
    "build_system_prompt",
    "resolve_root_model",
    "RLMRunResult",
    "run_pipeline_rlm",
]
