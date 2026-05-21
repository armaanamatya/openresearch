"""RLM root-orchestrator package — Phase 2 (issue #59).

The 14-stage `PipelineStage` loop is superseded by an RLM orchestrator built
on the `rlms` library (arXiv 2512.24601): the root LLM writes Python in a REPL
and calls each surviving stage agent's core logic as a callable "primitive".

Phase 2 ships the primitive layer:
  - `RunContext` — run-scoped dependencies threaded into every primitive.
  - `PRIMITIVE_REGISTRY` / `PRIMITIVE_DESCRIPTIONS` — the nine primitives and
    their root-facing signatures (`primitives.py`).
  - `build_custom_tools(ctx)` — binds the primitives to a `RunContext` and
    assembles the `custom_tools` dict `rlm.RLM(...)` consumes (`binding.py`).

Phase 3 (#60) constructs `rlm.RLM(...)` itself. The root REPL MUST be
`environment="local"` — `rlm`'s `DockerREPL` silently drops `custom_tools`,
and `"local"` is also a security boundary (the root model's REPL code runs via
`exec` on the host); see the threat model in `docs/design/rlm-pivot-brief.md` §7.
"""

# --- primitive layer (#59) --------------------------------------------------
from backend.agents.rlm.context import RunContext
from backend.agents.rlm.primitives import PRIMITIVE_DESCRIPTIONS, PRIMITIVE_REGISTRY
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
