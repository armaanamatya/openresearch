"""RunContext — run-scoped dependencies threaded into every RLM primitive.

Phase 2 (issue #59). The root RLM model passes only slices/specs as primitive
arguments (the Algorithm-2 guard). Everything else a primitive needs — paths,
the event emitter, the cost ledger, the LLM client, the agent runtime — lives
here and is closed over by `backend.agents.rlm.binding.build_custom_tools`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    """Everything a primitive needs that the root model does not pass.

    `llm_client` is the synchronous `LlmClient` protocol from
    `backend/services/context/workspace/tools/rlm_query.py` — `.complete(*,
    system, user) -> str`. `runtime` is an `AgentRuntime`; only
    `implement_baseline` needs it, so it defaults to None. `agent_model` is the
    model a primitive-spawned agent runs on (the code-writing agent in
    `implement_baseline`); `None` falls back to the agent registry's default.

    `deadline_utc` is set by `run.py` from the wall-clock budget at run start
    (M-DEADLINE, WS-H Batch P).  Primitives call `remaining_s()` to get seconds
    left; `None` means no wall-clock budget was configured.
    """

    project_id: str
    project_dir: Path
    runs_root: Path
    dashboard: Any            # DashboardEmitter
    cost_ledger: Any          # RunCostLedger
    llm_client: Any           # LlmClient protocol: .complete(*, system, user) -> str
    provider: str             # "anthropic" | "openai"
    model: str
    runtime: Any = None       # AgentRuntime — only implement_baseline uses it
    agent_model: str | None = None  # model for primitive-spawned agents (implement_baseline)
    workspace_service: Any = None
    workspace_id: str | None = None
    deadline_utc: datetime | None = field(default=None)  # M-DEADLINE — set by run.py
    sandbox_mode: Any = None  # SandboxMode — threaded from --sandbox CLI flag (I7)
    gpu_mode: Any = None     # GpuMode — threaded from ExecutionProfile.gpu_mode so
                             # _compute_constraint_guidance (in baseline_implementation.py)
                             # can decide CPU-vs-GPU baseline strategy dynamically rather
                             # than assuming docker = CPU-only. (2026-05-23 user mandate:
                             # "sandbox shouldn't be cpu only it should be dynamic since
                             # we can use runpod etc.")
    gpu_device_ids: tuple[str, ...] = ()  # host GPU UUIDs leased to this run (local sandbox); set from REPROLAB_GPU_DEVICE_IDS
    gpu_parallelism: str = "auto"  # "auto"|"single"|"multi"; from REPROLAB_GPU_PARALLELISM
    gpu_visible_count: int | None = None  # GPUs visible to this run (from CUDA_VISIBLE_DEVICES / lease); hints the code-writing agent
    run_budget: Any = None   # RunBudget — threaded from --max-pod-seconds / --max-usd etc.
    current_iteration: int = 0  # root-loop iteration index, incremented by ReproLabRLMLogger.log
    propose_round: int = 0      # per-run count of propose_improvements calls, incremented in wrap_primitive
    emit: Any = None          # thread-safe emit callable from sse_bridge.make_emit — set by run.py / conftest
    vram_override: int | None = None  # --vram-gb CLI flag; bypasses LLM VRAM estimate in resolve_gpu_requirements
    scope_spec: Any = None  # ScopeSpec — typed via Any to avoid a top-level import cycle;
                            # set by run.py / rdr/run.py from REPROLAB_SCOPE_SPEC_JSON.
    arxiv_id: str | None = None  # Bare arXiv ID (e.g. "2605.15155") when known; set by
                                 # run_pipeline_rlm from artifact_index.json / demo_status.json
                                 # so implement_baseline can route docs/papers/<id>.yaml even
                                 # when project_id is a hashed `prj_<digest>` string that the
                                 # _extract_arxiv_id regex cannot parse.
    minimize_compute: bool = False  # Lane Q — --minimize-compute / lab UI checkbox. When True,
                                    # implement_baseline prompt gets the substitution rules
                                    # (modern fast equivalents for slow paper schedules) and
                                    # the scope.declared_reductions contract.
    # θ: agent-declared metric paths, set by plan_reproduction once the planning
    # LLM responds with a metrics_shape list. run_experiment reads this to
    # validate that the emitted metrics.json matches the declared contract.
    # Typed as Any to avoid a top-level import cycle (schemas.ReproductionContract).
    reproduction_contract: Any = None  # ReproductionContract | None

    # --- Forced-iteration policy state (Lane H, spec 2026-05-24) ---
    # The most recent verify_against_rubric result the root has observed.
    # Set by binding._emit_supplemental on every successful rubric event so
    # the FINAL_VAR interceptor can read score-vs-target without re-scoring.
    # `None` means no rubric verification has happened yet — the policy
    # accepts FINAL_VAR honestly in that case (the run is rubric-less).
    latest_rubric_score: float | None = None
    latest_rubric_target: float | None = None
    latest_rubric_iteration: int = 0  # the iteration in which the score above was recorded

    def remaining_s(self) -> float | None:
        """Seconds until `deadline_utc`, clamped ≥ 0; None if no deadline set.

        Always returns a timezone-aware comparison: if `deadline_utc` is naive
        it is treated as UTC.
        """
        if self.deadline_utc is None:
            return None
        dl = self.deadline_utc
        if dl.tzinfo is None:
            dl = dl.replace(tzinfo=timezone.utc)
        return max(0.0, (dl - datetime.now(tz=timezone.utc)).total_seconds())
