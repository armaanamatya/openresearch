"""Hybrid RDR+RLM orchestration package.

``run_pipeline_hybrid`` is the default ``--mode rlm`` entry point.  It runs
a deterministic Phase 1 (RDR, ``max_repair_iterations=0``) to get structural
coverage, then invokes Phase 2 (RLM adaptive repair) only on weak clusters.

See ``docs/superpowers/specs/`` for the design rationale.
"""

from backend.agents.hybrid.controller import run_pipeline_hybrid

__all__ = ["run_pipeline_hybrid"]
