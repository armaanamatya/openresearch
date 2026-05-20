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

from backend.agents.rlm.primitives import PRIMITIVE_REGISTRY
from backend.agents.rlm.repl_host import ReplHost
from backend.agents.rlm.root_loop import RootLoop, RootIterationCapExceeded
from backend.agents.rlm.sub_call import llm_query, rlm_query
from backend.agents.rlm.system_prompt import build_system_prompt

__all__ = [
    "PRIMITIVE_REGISTRY",
    "ReplHost",
    "RootLoop",
    "RootIterationCapExceeded",
    "llm_query",
    "rlm_query",
    "build_system_prompt",
]
