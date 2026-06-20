"""run.py — the RLM run entry.

``run_pipeline_rlm()`` is the single run entry point.  It:

  1. builds a run-scoped :class:`RunContext`,
  2. resolves the primitive layer — the real ``build_custom_tools`` from
     ``binding.py`` if importable, else the deterministic stub provider,
  3. constructs an ``rlm.RLM`` (the Recursive Language Model engine),
  4. runs ``.completion()`` on a worker thread, streaming + checkpointing every
     iteration through :class:`OpenResearchRLMLogger`,
  5. writes ``final_report.{json,md}`` and returns an :class:`RLMRunResult`.

Time is bounded three ways (design spec §8): ``rlm``'s ``max_timeout`` (soft,
between iterations), per-primitive deadlines carried on :class:`RunContext`
(the real bound on a hung primitive), and a process-level wall-clock watchdog
here (the hard backstop — a thread cannot be killed, only the process).

Design contract: ``docs/superpowers/specs/2026-05-21-rlm-phase3-orchestrator-design.md`` §8.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import socket
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from rlm import RLM

from backend.agents.dashboard_emitter import DashboardEmitter
from backend.agents.execution import DEFAULT_SANDBOX_MODE
from backend.agents.resilience.budget import RunBudget
from backend.agents.resilience.cost import RunCostLedger
from backend.config import get_settings
from backend.eventstore.sqlite_store import SqliteEventStore

from backend.agents.rlm.checkpoint import IterationCheckpointer
from backend.agents.rlm.context import RunContext
from backend.agents.rlm.repl_snapshot import ReplSnapshotWriter
from backend.agents.rlm.models import (
    AZURE_FOUNDRY_KEY,
    RootModel,
    register_featherless_context_limits,
    resolve_root_model,
)
from backend.agents.rlm.report import (
    RLMFinalReport,
    build_final_report,
    run_experiment_call_count,
    run_experiment_partial_timeout_count,
    run_experiment_success_count,
    write_final_report_rlm,
)
from backend.agents.rlm.sse_bridge import (
    OpenResearchRLMLogger,
    build_run_complete_event,
    build_run_warning_event,
    make_emit,
    make_on_subcall_complete,
    make_on_subcall_start,
    redact_corpus,
)
from backend.agents.rlm.stub_primitives import build_stub_custom_tools
from backend.agents.rlm.system_prompt import build_system_prompt

# Register the anthropic-oauth backend with rlm.clients.get_client — must run
# before RLM(backend="anthropic-oauth", ...) is constructed below.
# Also install the prompt-caching wrapper for the anthropic API-key path so that
# the stable system prompt is cached across iterations (~50% input-token saving).
from backend.agents.rlm._oauth_backend_patch import (
    apply_oauth_backend_patch,
    apply_anthropic_caching_patch,
)
from backend.agents.rlm.forced_iteration import (
    _TERMINAL_FAILURE_CLASSES,
    _WALL_CLOCK_FLOOR_S,
    ForcedIterationPolicy,
    _current_policy,
    _default_degenerate_threshold,
    apply_forced_iteration_patch,
    forced_iteration_policy,
)
from backend.agents.rlm.root_progress import infer_required_stage
from backend.agents.rlm.root_validation import classify_root_model
# BUG-LR-011: restore globals()/locals() inside rlm's LocalREPL sandbox
# (upstream blacklists them alongside eval/exec/compile/input — incorrect).
from backend.agents.rlm import safe_builtins_patch as _safe_builtins_patch  # noqa: F401
# Harden the rlm vendored AzureOpenAIClient: rebuild openai.AzureOpenAI /
# AsyncAzureOpenAI with max_retries=6 so root completions survive AOAI 429 bursts.
from backend.agents.rlm import azure_root_hardening_patch as _azure_root_hardening_patch  # noqa: F401
# BUG-LR-012: include traceback.format_exc() in REPL exception stderr so the
# root model can diagnose failures rather than concluding primitives unavailable.
from backend.agents.rlm import safe_repl_traceback_patch as _safe_repl_traceback_patch  # noqa: F401
# BUG-NEW-033 (ported 2026-06-10 from pipeline-validation-mech-understanding):
# auto-recover from (slice, question) misuse of rlm_query/llm_query — the
# library API is single-prompt; the misuse routed the question as a model name
# and the CLI error string leaked into paper_claims (SDAR attempt 4 post-mortem).
from backend.agents.rlm import rlm_query_misuse_patch as _rlm_query_misuse_patch  # noqa: F401
# BUG-NEW-043 (ported 2026-06-09): surface real traceback when rlm._subcall's
# child completion raises; upstream catches with `str(e)` and we get only
# "maximum recursion depth exceeded" with no file/line. Mech-understanding
# 2026-05-29 lost two sub-RLMs to this. (The branch's BUG-NEW-033
# rlm_query_misuse_patch is ported too — imported above, 2026-06-10.)
from backend.agents.rlm import safe_subcall_traceback_patch as _safe_subcall_traceback_patch  # noqa: F401
# 2026-06-18: accept ```python / ```py fences (not only ```repl) in root
# responses — the upstream parser dropped grok-4.3's ```python blocks, so nothing
# executed and the empty-code-block degenerate detector killed the run at iter 3.
from backend.agents.rlm import code_fence_patch as _code_fence_patch  # noqa: F401
# BUG-NEW-043 (belt+braces): the default recursion limit is 1000; the
# mech-understanding paper's LaTeX-dense prompt blew it via some unknown deep
# recursion path in the rlms stack. 10000 is defensive against the same kind
# of regex/templater walker.
import sys as _sys_for_recursion
_sys_for_recursion.setrecursionlimit(10000)
apply_oauth_backend_patch()
apply_anthropic_caching_patch()
# Lane H — install the FINAL_VAR interceptor once. Per-run policies are
# pushed via the forced_iteration_policy context manager around rlm.completion.
apply_forced_iteration_patch()

logger = logging.getLogger(__name__)


def _fixfirst_loop_engaged() -> bool:
    """True iff the P3 fix-first repair loop should be engaged for this run.

    The loop activates when EITHER the zero-metrics guard OR the external
    adversarial validator is enabled, or any of the pre-GPU/report gates that
    feed the same repair loop.  When none of these flags are set this returns
    False and the policy is byte-identical to the pre-P3 behavior (no
    evidence_fingerprint / validator_gate hooks are assigned).
    """
    from backend.agents.rlm.zero_metrics_detection import zero_metrics_guard_enabled  # noqa: PLC0415
    from backend.agents.rlm.external_validator import external_validator_enabled  # noqa: PLC0415
    from backend.agents.rlm.report_claim_gate import report_claim_gate_enabled  # noqa: PLC0415
    _code_review = os.environ.get("OPENRESEARCH_CODE_REVIEW_GATE", "").strip() == "1"
    _smoke = os.environ.get("OPENRESEARCH_METRIC_REALITY_SMOKE", "").strip() == "1"
    return (
        zero_metrics_guard_enabled()
        or external_validator_enabled()
        or report_claim_gate_enabled()
        or _code_review
        or _smoke
    )


# --- Tuning constants ------------------------------------------------------
_MAX_ITERATIONS = 20          # paper Appendix A
_MAX_DEPTH = 2                # brief §3 — depth-2 enables real rlm_query recursion
# 2026-05-25 — the rlm/ctx wall-clock ceiling is FULLY USER-CONTROLLED. When no
# ``--max-wall-clock`` flag (or OPENRESEARCH_MAX_WALL_CLOCK_S env var) is set, rlm's
# own between-iteration timeout and every per-primitive deadline are unbounded —
# the user mandate is "no truncation of a long reproduction unless the operator
# opts in." 2026-06-01 — but "unbounded" must NOT mean "can hang forever with no
# report": a hard-ceiling watchdog backstop (``_watchdog_hard_ceiling_s``) is now
# armed even when no explicit ceiling is requested, so a wedged run always ships a
# partial report and hard-exits. The 2026-06-01 SDAR run hung ~3h inside one
# synchronous run_experiment (every soft bound collapsed to None, watchdog unarmed),
# the user killed it, and NO final_report was written — this backstop closes that
# gap. Opt fully out with OPENRESEARCH_WATCHDOG_HARD_CEILING_S=0.
_DEFAULT_WALL_CLOCK_S: float | None = None
_WATCHDOG_GRACE_S = 120.0     # watchdog fires only past rlm's own max_timeout
_WATCHDOG_EXIT_CODE = 75      # EX_TEMPFAIL — "the run was hard-stopped"
_WATCHDOG_HARD_CEILING_DEFAULT_S = 50400.0  # 14h — generous backstop (operator preference 2026-06-02)


def _watchdog_hard_ceiling_s() -> float:
    """Always-on watchdog backstop (seconds), used when no explicit wall-clock is set.

    Read at arm-time (not import-time) so tests and operators can tune it via
    ``OPENRESEARCH_WATCHDOG_HARD_CEILING_S``. ``0`` (or empty) disables the backstop
    entirely, restoring the pre-2026-06-01 fully-unbounded behaviour. A malformed
    value falls back to the 14h default rather than crashing the run.
    """
    raw = os.environ.get("OPENRESEARCH_WATCHDOG_HARD_CEILING_S", "").strip()
    if raw == "":
        return _WATCHDOG_HARD_CEILING_DEFAULT_S
    try:
        return float(raw)
    except ValueError:
        return _WATCHDOG_HARD_CEILING_DEFAULT_S


_WATCHDOG_POLL_S = 30.0       # wall-clock poll cadence (sleep-robust; see _arm_watchdog)

_ROOT_PROMPT = (
    "Reproduce the research paper offloaded in the REPL variable `context`. "
    "Navigate it with REPL code and sub-calls (llm_query / rlm_query) over slices "
    "you construct, use the domain primitives to detect the environment, plan, "
    "implement and run a baseline, then score it against the rubric and propose "
    "improvements. Accumulate the reproduction as REPL state. "
    "You MUST call run_experiment to actually execute the baseline — a "
    "reproduction is not complete until the baseline has run. Every metric in "
    "your final report must come from a real run_experiment result; never "
    "estimate, guess, or invent a metric. If run_experiment did not run or did "
    "not succeed, report no baseline metrics and an honest 'partial' verdict. "
    "If run_experiment returns success=False, do not give up: call "
    "implement_baseline again with plan['repair_context'] set to that failed "
    "run_experiment result, then call run_experiment again — the code-writing "
    "agent will diagnose the error and fix the code in place. Repeat this "
    "repair cycle up to 2 times before accepting a partial result. "
    "When finished, build your report dict, json.dumps it into a variable, and "
    "call FINAL_VAR on that JSON-string variable — exactly as the system "
    "prompt's termination contract describes."
)


@dataclass
class RLMRunResult:
    """Lightweight outcome of one ``rlm``-mode run — returned to the CLI / caller."""

    project_id: str
    status: str                       # "completed" | "partial" | "failed"
    iterations: int
    rubric_score: float | None
    cost_usd: float | None
    final_report_path: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _accelerator_grader_offloaded(scope: str | None) -> bool:
    """Whether ``OPENRESEARCH_ACCELERATOR_SCOPE`` routes the rubric grader to the accelerator.

    The grader is ``ctx.llm_client`` (used by ``verify_against_rubric`` and
    ``propose_improvements`` — quality-critical judgment/generation). Default
    ``"navigation"`` keeps it on the strong root model (e.g. Sonnet) so a small
    accelerator never decides the score; only ``"all"`` offloads it (sensible only with a
    strong accelerator). Context navigation (``rlm_query``/``llm_query``) always uses the
    accelerator when one is active, independent of this setting.
    """
    return (scope or "navigation").strip().lower() == "all"


def _build_llm_client(provider: str | None, root_model: RootModel) -> tuple[Any, str]:
    """Build the ``LlmClient`` for ``RunContext.llm_client`` and its model label.

    Primitives (build_environment, plan_reproduction, implement_baseline, etc.)
    AND rubric-generation share this one client, so the choice must follow the
    selected root model. Dispatch order:

      1. ``root_model.rlm_backend == "anthropic-oauth"``  → ``ClaudeLlmClient``
         (OAuth-capable; no API key required — auth resolved by ``claude-agent-sdk``).
      2. ``root_model.rlm_backend == "openai"`` AND a custom ``base_url`` is set
         (e.g. Featherless) → ``OpenAILlmClient(model, api_key, base_url)`` mirroring
         the root endpoint.
      3. ``root_model.rlm_backend == "openrouter"`` → ``OpenAILlmClient(model, api_key,
         base_url="https://openrouter.ai/api/v1")`` using ``OPENROUTER_API_KEY``.
      4. ``root_model.rlm_backend == "anthropic"`` (raw HTTP, paid API key) →
         ``ClaudeLlmClient`` (uses ANTHROPIC_API_KEY via the SDK auth resolution).
      5. ``root_model.rlm_backend == "openai"`` (plain OpenAI) → ``OpenAILlmClient``
         using ``OPENAI_API_KEY``.
      6. Last resort — explicit ``provider`` arg wins over a guessed fallback:
         ``provider == "openai"`` → ``OpenAILlmClient``; else → ``ClaudeLlmClient``.

    Rationale: the previous implementation hard-coded "OpenAI when OPENAI_API_KEY is
    set" — but a stale/invalid OPENAI_API_KEY (common in shared dev envs) silently
    routed every primitive to OpenAI even when the user explicitly selected
    claude-oauth. Dispatching on ``root_model.rlm_backend`` first respects the
    user's intent.

    The ``LlmClient`` protocol is ``.complete(*, system, user) -> str`` — it
    returns no token usage; primitive-internal LLM cost is therefore not captured
    here (the dominant cost is the root + sub-call accounted for in
    ``report._cost_dict`` via ``rlm``'s ``usage_summary``). Real per-primitive
    usage needs the ``LlmClient`` protocol to carry usage — deferred.
    """
    backend = root_model.rlm_backend
    bk = root_model.backend_kwargs
    sub_bk = root_model.sub_backend_kwargs

    # Optional quality pin (2026-06-11): OPENRESEARCH_PRIMITIVE_LLM_MODEL routes the
    # shared primitive LlmClient (plan_reproduction, propose_improvements,
    # generate_rubric_tree, repro-spec extraction, tool-recommendation) to an
    # explicit Claude model id (e.g. an Opus id) instead of the claude CLI's
    # configured default. Scope is ONLY the Claude client paths below —
    # navigation sub-calls (llm_query/rlm_query) ride the rlm sub-backend /
    # accelerator and the root loop rides backend_kwargs, both unaffected.
    # Unset/empty → ClaudeLlmClient falls back to default_oauth_model() (Sonnet),
    # NOT the bundled CLI's mutable default (the 2026-06-14 Fable-5 wedge fix).
    _pinned_model = os.environ.get("OPENRESEARCH_PRIMITIVE_LLM_MODEL", "").strip() or None

    # 1. claude-oauth — explicit OAuth path, no api_key in kwargs
    if backend == "anthropic-oauth":
        from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient
        return ClaudeLlmClient(model=_pinned_model), (_pinned_model or "claude-oauth")

    # 2. OpenAI-compatible custom endpoint (Featherless, vLLM-via-OpenAI, etc.)
    if backend == "openai" and bk.get("base_url"):
        if not bk.get("api_key"):
            raise ValueError(
                f"Root model {root_model.key!r} uses a custom base_url but its "
                f"api_key was not resolved — _build_llm_client requires a RootModel "
                f"from resolve_root_model()."
            )
        from backend.services.context.workspace.tools.openai_client import OpenAILlmClient
        model = sub_bk.get("model_name") or bk.get("model_name", "")
        # Reasoning models served via Azure Foundry (e.g. Kimi-K2.6) emit
        # reasoning_content BEFORE content; a 4096 cap shared with the thinking
        # truncates a large rubric/plan into unparseable output (observed
        # 2026-06-18: generate_rubric_tree failed all 3 attempts → rubric-less
        # run). Give the Foundry primitive client ample headroom (a cap only —
        # unused tokens cost nothing). Featherless / other openai+base_url roots
        # keep the 4096 default → byte-identical.
        _client_max_tokens = 32768 if root_model.key == AZURE_FOUNDRY_KEY else 4096
        return (
            OpenAILlmClient(
                model=model,
                api_key=bk["api_key"],
                base_url=bk["base_url"],
                max_tokens=_client_max_tokens,
            ),
            model,
        )

    # 3. OpenRouter — also OpenAI-compatible; api_key from backend_kwargs (injected by resolve_root_model)
    if backend == "openrouter":
        from backend.services.context.workspace.tools.openai_client import OpenAILlmClient
        api_key = bk.get("api_key")
        if not api_key:
            raise ValueError(
                f"Root model {root_model.key!r} uses backend 'openrouter' but its "
                f"api_key was not resolved — _build_llm_client requires a RootModel "
                f"from resolve_root_model()."
            )
        model = sub_bk.get("model_name") or bk.get("model_name", "")
        return OpenAILlmClient(model=model, api_key=api_key, base_url="https://openrouter.ai/api/v1"), model

    # 4. Azure OpenAI — rlm has a built-in AzureOpenAIClient; wrap it in an
    #    OpenAILlmClient-compatible shim so primitives get the same .complete()
    #    interface.  The shim delegates to openai.AzureOpenAI directly, mirroring
    #    how OpenAILlmClient wraps openai.OpenAI.
    if backend == "azure_openai":
        from backend.services.context.workspace.tools.azure_openai_client import AzureOpenAILlmClient
        api_key = bk.get("api_key")
        azure_endpoint = bk.get("azure_endpoint")
        azure_deployment = bk.get("azure_deployment")
        model = bk.get("model_name", "gpt-4o")
        if not azure_endpoint:
            raise ValueError(
                f"Root model {root_model.key!r} uses backend 'azure_openai' but "
                "azure_endpoint was not resolved — _build_llm_client requires a "
                "RootModel from resolve_root_model()."
            )
        return AzureOpenAILlmClient(
            model=model,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
        ), model

    # 5. Anthropic raw HTTP — uses ANTHROPIC_API_KEY through claude-agent-sdk's resolution
    if backend == "anthropic":
        from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient
        return ClaudeLlmClient(model=_pinned_model), (_pinned_model or "claude")

    # 6. Plain OpenAI
    if backend == "openai":
        from backend.services.context.workspace.tools.openai_client import OpenAILlmClient
        return OpenAILlmClient(), "gpt-4o-mini"

    # 7. Unknown backend — respect explicit `provider` arg, else default to Claude.
    #    Removed the old "OPENAI_API_KEY env → assume OpenAI" heuristic; it
    #    misrouted claude-oauth runs when a stale key was in env.
    if (provider or "").lower() == "openai":
        from backend.services.context.workspace.tools.openai_client import OpenAILlmClient
        return OpenAILlmClient(), "gpt-4o-mini"
    from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient
    return ClaudeLlmClient(model=_pinned_model), (_pinned_model or "claude")


def _resolve_agent_runtime(
    runtime: Any, provider: str | None, role_selection: Any = None
) -> tuple[Any, str | None, str]:
    """Resolve the sub-agent runtime + model for primitives (``implement_baseline``).

    An RLM run has two independent LLM layers with separate credentials:

      * the **root-model loop** — the ``rlm`` library's backend; makes raw HTTP
        completion calls, so it needs a real API key (Featherless);
      * the **sub-agent runtime** — used by ``implement_baseline`` to drive a
        code-writing agent.

    For the sub-agent layer, ``claude-agent-sdk`` resolves auth itself, and the
    same runtime serves both deployment modes:

      * **production / API mode** — ``ANTHROPIC_API_KEY`` takes priority;
      * **dev** — the Claude Code subscription's OAuth login (no key needed).

    The resolved ``agent_model`` is ``settings.anthropic_default_model`` (Sonnet
    by default). It is threaded onto ``RunContext.agent_model`` and passed by
    ``implement_baseline`` as the per-invocation ``model_override`` — which is
    the *only* knob that beats the agent registry's heavier default
    (``baseline-implementation`` is registered as Opus). Without this override
    the code-writing agent runs Opus-rate and exhausts the OAuth quota, exactly
    as RLM run 3 did.

    Resolution order (most-preferred first):

      1. an explicitly-passed ``runtime`` (caller override — e.g. a test);
      2. **Claude** — whenever anthropic credentials resolve (API key *or* the
         logged-in ``claude`` CLI), independent of any ``.env`` provider
         setting that might point at a dead OpenAI key;
      3. whatever ``make_runtime`` resolves from env/settings.

    Returns ``(runtime_or_None, agent_model_or_None, label)``. ``agent_model``
    is ``None`` outside the Claude path (the registry default then applies). A
    ``None`` runtime is not fatal — ``implement_baseline`` falls through to
    ``make_runtime`` itself and fails honestly there.
    """
    if runtime is not None:
        return runtime, None, "caller-supplied"

    # Per-role executor override (2026-06-17): an explicit unified-surface
    # executor pick builds the matching runtime directly (Azure now reachable via
    # make_runtime). A legacy OPENRESEARCH_EXECUTOR=qwen/vllm/etc. stays None in the
    # selection, so the executor-tier path below still handles it unchanged.
    _exec_spec = getattr(role_selection, "executor", None) if role_selection is not None else None
    if _exec_spec is not None:
        from backend.agents.runtime.factory import make_runtime as _make_runtime
        _exec_provider = {
            "anthropic-oauth": "anthropic", "anthropic": "anthropic",
            "openai": "openai", "azure": "azure",
            "azure-foundry": "azure-foundry",
        }[_exec_spec.provider]
        return (
            _make_runtime(_exec_provider, require_api_key=True),
            _exec_spec.model,
            f"role:executor:{_exec_spec.stamp}",
        )

    # Executor tier (OPENRESEARCH_EXECUTOR): run the code-writing agent on a local Qwen
    # (vLLM) instead of Sonnet to save Sonnet usage. Health-probed with graceful
    # fallback to the default below when the endpoint is unset/unreachable.
    try:
        from backend.agents.rlm.executor import resolve_executor

        _plan = resolve_executor()
        if _plan is not None:
            return _plan.runtime, _plan.model, f"executor:{_plan.label}"
    except Exception as exc:  # noqa: BLE001 — never block on the optional tier
        logger.warning("executor-tier resolution failed (%s); using default executor", exc)

    from backend.agents.runtime.factory import (
        has_provider_credentials,
        make_runtime,
    )

    if has_provider_credentials("anthropic"):
        model = get_settings().anthropic_default_model
        return (
            make_runtime("anthropic"),
            model,
            f"claude / {model} (SDK-resolved auth: API key or OAuth)",
        )
    try:
        # require_api_key=True so a missing-credentials environment fails here,
        # at resolution, rather than opaquely at the primitive's call site.
        return (
            make_runtime(provider, require_api_key=True),
            None,
            f"make_runtime({provider or 'env-default'})",
        )
    except Exception as exc:  # noqa: BLE001 — degrade; the primitive fails honestly
        logger.warning(
            "run_pipeline_rlm: no agent runtime could be resolved (%s: %s) — "
            "implement_baseline will surface the credential error itself",
            type(exc).__name__,
            exc,
        )
        return None, None, f"unresolved ({type(exc).__name__})"


def _resolve_custom_tools(ctx: RunContext) -> tuple[dict, str]:
    """Return ``(custom_tools, provider_label)`` for ``RLM(custom_tools=...)``.

    Prefers the real ``binding.build_custom_tools`` (the domain primitives).
    Falls back to the deterministic stub provider (§13) when the primitive
    layer is **absent** *or* **present-but-failing** — if ``binding.py`` cannot
    be imported, or ``build_custom_tools(ctx)`` raises. The run must proceed
    regardless; the fallback is loud (a WARNING with the underlying exception)
    — it degrades, it never silently masks.
    """
    if os.environ.get("OPENRESEARCH_RLM_STUB_PRIMITIVES") == "1":
        return build_stub_custom_tools(ctx), "stub (OPENRESEARCH_RLM_STUB_PRIMITIVES=1)"
    try:
        from backend.agents.rlm.binding import build_custom_tools

        tools = build_custom_tools(ctx)
    except ImportError:
        logger.info(
            "run_pipeline_rlm: backend.agents.rlm.binding not importable — "
            "using stub primitives"
        )
        return build_stub_custom_tools(ctx), "stub (binding.py absent)"
    except Exception as exc:  # noqa: BLE001 — degrade loudly, don't crash the run
        logger.warning(
            "run_pipeline_rlm: build_custom_tools raised (%s: %s) — "
            "primitive layer incomplete; falling back to stub primitives.",
            type(exc).__name__,
            exc,
        )
        return (
            build_stub_custom_tools(ctx),
            f"stub (binding failed: {type(exc).__name__})",
        )
    return tools, "real (binding)"


def _extract_arxiv_id_from_project_dir(project_dir: Path) -> str | None:
    """Derive the bare arXiv ID (e.g. ``"2605.15155"``) from on-disk artifacts.

    arXiv-sourced runs receive a hashed project_id (``prj_<sha256[:16]>``)
    that encodes no ID-shaped string, so the ``_extract_arxiv_id`` regex in
    ``baseline_implementation.py`` always returns ``None`` for them.  This
    helper reads the on-disk files produced during ingest to recover the
    real ID so ``docs/papers/<id>.yaml`` overrides can fire.

    Resolution order (most-authoritative first):
    1. ``artifact_index.json`` → ``paper.arxiv_id``
    2. ``demo_status.json``    → ``sourceUrl`` (``arxiv.org/abs/<id>`` URL)
    3. ``demo_status.json``    → ``sourceLabel`` (``arxiv_2605.15155.pdf`` pattern)
    4. ``None`` — no arXiv ID recoverable; caller falls back to regex.

    The regex used here is the same ``DDDD.DDDDD?`` pattern as
    ``baseline_implementation._ARXIV_ID_RE``.
    """
    import re as _re
    _ARXIV_RE = _re.compile(r"(\d{4,5}\.\d{4,5})")

    # 1. artifact_index.json → paper.arxiv_id (most authoritative)
    ai_path = project_dir / "artifact_index.json"
    if ai_path.exists():
        try:
            data = json.loads(ai_path.read_text(encoding="utf-8", errors="replace"))
            aid = (data.get("paper") or {}).get("arxiv_id")
            if aid and _ARXIV_RE.search(str(aid)):
                return str(aid).strip()
        except Exception:  # noqa: BLE001 — corrupt JSON, skip
            pass

    # 2 & 3. demo_status.json → sourceUrl or sourceLabel
    ds_path = project_dir / "demo_status.json"
    if ds_path.exists():
        try:
            data = json.loads(ds_path.read_text(encoding="utf-8", errors="replace"))
            # 2. sourceUrl: "https://arxiv.org/abs/2605.15155"
            url = data.get("sourceUrl", "") or ""
            m = _re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4,5}\.\d{4,5})", url)
            if m:
                return m.group(1)
            # 3. sourceLabel: "arxiv_2604.01733.pdf" or similar
            label = data.get("sourceLabel", "") or ""
            m = _ARXIV_RE.search(label)
            if m:
                return m.group(1)
        except Exception:  # noqa: BLE001 — corrupt JSON, skip
            pass

    return None


def _build_context(workspace_claim_map: dict[str, Any]) -> dict[str, Any]:
    """Assemble the offloaded RLM ``context`` dict from the workspace claim map.

    ``workspace_claim_map`` shape (see ``pipeline._write_workspace_claim_map``):
    ``{"project_id": str, "entries": [{"source_id", "title", "excerpt"}, ...]}``.

    For PaperBench bundle runs the map also carries a ``"paperbench"`` sub-dict
    with ``{"paper_id": ..., "metadata": {"id": ..., "title": ...}, ...}``.
    When present, ``paper_metadata.id`` and ``paper_metadata.title`` are sourced
    from the bundle's real metadata rather than the first entry's generic title
    (e.g. ``"PaperBench paper markdown"``).

    The full corpus — supplementary text, repo files, prior-work refs — is
    populated by later phases (#62).  For now ``paper_text`` / ``paper_metadata``
    are assembled from the claim-map entries; ``rubric_spec`` is passed through
    if present.
    """
    entries = workspace_claim_map.get("entries") or []
    sections: list[str] = []
    for entry in entries:
        title = entry.get("title", "")
        excerpt = entry.get("excerpt", "")
        sections.append(f"## {title}\n\n{excerpt}" if title else excerpt)

    # PaperBench bundle runs carry real paper identity in the "paperbench" block.
    # Use it when available so the root model sees the actual paper id + title
    # rather than the generic entry title produced by bundle_to_workspace_claim_map.
    pb = workspace_claim_map.get("paperbench") or {}
    pb_meta = pb.get("metadata") or {}
    paper_id = pb_meta.get("id") or pb.get("paper_id") or ""
    paper_title = pb_meta.get("title") or (entries[0].get("title", "") if entries else "")

    return {
        "paper_text": "\n\n".join(sections),
        "paper_metadata": {
            "id": paper_id,
            "title": paper_title,
            "sections": [e.get("title", "") for e in entries],
            "source_ids": [e.get("source_id", "") for e in entries],
        },
        "supplementary_text": None,
        "repo_files": None,
        "prior_work_refs": [],
        "rubric_spec": workspace_claim_map.get("rubric_spec") or {},
    }


def _context_metadata(context_dict: dict[str, Any]) -> dict[str, dict]:
    """Build ``{key: {"type", "length"}}`` metadata for the system prompt.

    Only the name / type / length of each ``context`` value is exposed to the
    root model — never the value itself (RLM property 1).
    """
    meta: dict[str, dict] = {}
    for key, value in context_dict.items():
        if isinstance(value, (str, list, dict)):
            length: int = len(value)
        else:
            length = 0
        meta[key] = {"type": type(value).__name__, "length": length}
    return meta


def _corpus_sentinels(context_dict: dict[str, Any]) -> list[str]:
    """Build the M-REDACT sentinel list from ``context_dict`` corpus values.

    Returns the first 200 chars of each string corpus value — enough to detect
    verbatim leakage at egress (stdout/stderr prefixes, final report summary)
    without storing the full corpus in memory twice.
    """
    sentinels: list[str] = []
    for value in context_dict.values():
        if isinstance(value, str) and value:
            sentinels.append(value[:200])
    return sentinels


def _verdict_to_status(verdict: str) -> str:
    """Map an ``RLMFinalReport`` verdict to a run status."""
    return "completed" if verdict == "reproduced" else verdict


def _assert_paper_text_precondition(project_dir: Path, *, allow_lossy: bool) -> str | None:
    """PR-π Module E — fail-fast gate for missing/degraded parsed_full_text.txt.

    Raises ``RuntimeError`` when ``allow_lossy=False`` and the parsed paper
    text is absent or suspiciously small (<1 KB), indicating that the ingestion
    parser cascade failed. When ``allow_lossy=True`` (default) a WARNING is
    logged and the run proceeds in degraded mode.

    This guard runs at the START of ``run_pipeline_rlm`` — before any RLM
    loop iteration — so the user gets an actionable failure message instead of
    a silent lossy-workspace fallback that defeats paper-grounding.

    Returns the human-readable *degraded reason* when the run proceeds in lossy
    mode, so the caller can surface it as an operator-visible warning in
    ``demo_status.json`` (F-29) rather than only a buried log line; returns
    ``None`` when the paper text is intact.
    """
    parsed_path = project_dir / "parsed_full_text.txt"
    degraded = not parsed_path.exists() or parsed_path.stat().st_size < 1024
    if not degraded:
        return None
    if not allow_lossy:
        raise RuntimeError(
            f"parsed_full_text.txt missing or <1KB at {parsed_path}. "
            f"Parser likely failed. Re-run ingestion or set "
            f"OPENRESEARCH_ALLOW_LOSSY_PAPER_TEXT=true."
        )
    logger.warning(
        "paper text degraded — proceeding with lossy workspace fallback "
        "(parsed_full_text.txt missing or <1KB at %s)",
        parsed_path,
    )
    return (
        "paper text degraded — proceeding with lossy workspace fallback "
        f"(parsed_full_text.txt missing or <1KB at {parsed_path})"
    )


def _assert_disk_headroom(runs_root: Path, *, min_gb: float) -> str | None:
    """Fail-fast gate for low disk on the runs root (2026-06-15).

    A run that starts on a near-full disk cannot write checkpoints / metrics → its
    GPU cells HANG and ORPHAN (the 2026-06-15 incident: ``/home`` hit 100%, SDAR's
    training subprocesses hung holding 17 GB of VRAM each, the run died with NO final
    report). Aborting at run start — before any GPU work — is far cheaper than an
    orphaned 14h run. Raises ``RuntimeError`` when critically low; returns a warning
    reason when headroom is thin-but-OK; ``min_gb <= 0`` disables the gate. Paper-
    agnostic: every run gets the same protection.
    """
    if min_gb <= 0:
        return None
    try:
        # f_bavail (blocks available to NON-root), NOT shutil.disk_usage().free /
        # f_bfree, which counts root-reserved space a non-root run cannot write to —
        # on a near-full disk that gap is the difference between "passes the gate" and
        # "actually has room to write checkpoints".
        st = os.statvfs(runs_root)
        free_gb = st.f_bavail * st.f_frsize / (1024 ** 3)
    except OSError:
        return None  # cannot stat the mount → never block on a bookkeeping failure
    if free_gb < min_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free on the runs root ({runs_root}) — below the "
            f"{min_gb:.0f} GB floor. A run on a near-full disk hangs/orphans on "
            f"checkpoint writes. Free space, point --runs-root at a disk with room "
            f"(e.g. --runs-root /scratch/runs), or set OPENRESEARCH_MIN_DISK_GB=0 to override."
        )
    if free_gb < min_gb * 2:
        return (
            f"low disk headroom: {free_gb:.1f} GB free on the runs root ({runs_root}); "
            f"floor is {min_gb:.0f} GB — a long run may exhaust it."
        )
    return None


# Exact-name denylist for the run_config.json env snapshot: knobs whose VALUES
# routinely carry credentials even though their names lack KEY/SECRET/TOKEN/
# PASSWORD. OPENRESEARCH_DATABASE_URL is a deployment knob (a credentialed
# postgres DSN embeds user:pass), not a launch parameter; the bootstrap
# command is arbitrary shell that may inline tokens (e.g. an hf login).
_ENV_SNAPSHOT_DENY_EXACT = frozenset({
    "OPENRESEARCH_DATABASE_URL",
    "OPENRESEARCH_RUNPOD_BOOTSTRAP_COMMAND",
})


def _redact_env_value(value: str) -> str:
    """Strip URL userinfo (``user:pass@``) before persisting a snapshot value.

    Durably covers the URL-shaped class — OPENRESEARCH_LOCAL_TORCH_INDEX_URL,
    OPENRESEARCH_ACCELERATOR_BASE_URL, and any future ``*_URL`` knob pointing
    at a private index with embedded credentials.
    """
    return re.sub(r"(?<=://)[^/@\s]+@", "***@", value)


def _write_demo_status(
    project_dir: Path,
    status: str,
    *,
    error: Any | None = None,
    primitive_provider: str = "real",  # T21 / review I8
    process_status: str | None = None,
    verdict: str | None = None,
    warnings: list[str] | None = None,
    root_model_validated: bool | None = None,
    root_model_risk: str | None = None,
) -> None:
    """Write (merge) ``runs/<id>/demo_status.json`` so the run is REST-retrievable.

    The HTTP layer's ``GET /runs/{id}`` reads this snapshot via
    ``live_runs._read_status``; without it a CLI- or script-launched RLM run
    404s. The payload carries ``LiveRunState``'s required fields (``projectId``,
    ``outputDir``, ``runMode``, ``status``). Any pre-existing file is merged, not
    overwritten, so an earlier ``startedAt`` survives the terminal write.

    ``status`` must be a valid ``RunStatus`` (``running`` | ``completed`` |
    ``failed`` | ``stopped``). Two related axes are recorded alongside it:
    ``process_status`` — the run-subprocess lifecycle (``running`` while the
    process is alive, ``completed`` once it has exited) — and ``verdict`` — the
    reproduction outcome (``reproduced`` | ``partial`` | ``failed`` |
    ``unknown``, also mirrored in ``final_report.json``). Both are derived from
    ``status`` when not passed explicitly. ``LiveRunState`` ignores these extra
    keys (pydantic ``extra='ignore'``); they exist for richer status consumers,
    and the CLI sanity/reproduce paths pass them explicitly.
    """
    path = project_dir / "demo_status.json"
    now = datetime.now(timezone.utc).isoformat()
    terminal = status in ("completed", "failed", "stopped")
    # Derive the lifecycle/verdict axes from RunStatus when not supplied so the
    # snapshot schema is consistent regardless of which caller wrote it.
    if process_status is None:
        process_status = "completed" if terminal else "running"
    if verdict is None:
        verdict = "failed" if status == "failed" else "unknown"
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a corrupt prior snapshot is replaced
            existing = {}
    payload: dict[str, Any] = {
        **existing,
        "projectId": project_dir.name,
        "outputDir": str(project_dir),
        "runMode": "rlm",
        "status": status,
        "updatedAt": now,
    }
    payload.setdefault("startedAt", now)
    if terminal:
        payload["completedAt"] = now
    if error is not None:
        payload["error"] = error
    # Operator-visible warnings (e.g. degraded paper text, F-29). Merged via
    # ``**existing`` on later writes, so a run-start warning survives the
    # terminal write; only replaced when this call explicitly supplies one.
    if warnings:
        payload["warnings"] = list(warnings)
    # Root-validation gate stamp (oauth-root-reliability plan, P2). Written only
    # when supplied so existing call sites stay byte-for-byte unchanged; the
    # ``**existing`` merge carries them forward across later status writes.
    if root_model_validated is not None:
        payload["root_model_validated"] = bool(root_model_validated)
    if root_model_risk is not None:
        payload["root_model_risk"] = root_model_risk
    payload["primitiveProvider"] = primitive_provider  # T21 / review I8
    payload["process_status"] = process_status
    payload["verdict"] = verdict
    # Liveness prereq: run_liveness.sweep_orphaned_runs deliberately skips
    # runs without a pid ("absent-pid means unknown, not dead"), so a
    # SIGKILLed CLI/batch run used to show status=running forever — only the
    # API spawn path stamped one (live_runs.py). run.py always executes inside
    # the run process itself, so an unconditional overwrite is always correct:
    # on the API path the parent recorded this same subprocess pid
    # (live_runs.py spawn, no shell), and on CLI re-runs of a reused project
    # dir a setdefault would inherit a DEAD prior attempt's pid (demo_status
    # is not in attempt_isolation's archive set), getting a live run falsely
    # swept as orphaned. pidHost scopes the pid to the namespace that minted
    # it — a containerized sweeper must not os.kill-probe host pids through
    # the bind-mounted runs/ (see run_liveness.sweep_orphaned_runs).
    payload["pid"] = os.getpid()
    payload["pidHost"] = socket.gethostname()
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 — status is best-effort; never crash the run
        logger.exception("run_pipeline_rlm: could not write demo_status.json")


class _FatalPrimitiveAbort(RuntimeError):
    """Controlled abort when a primitive reports a non-recoverable backend state."""

    def __init__(self, *, primitive_name: str, result: dict) -> None:
        super().__init__(str(result.get("error") or "fatal primitive outcome"))
        self.primitive_name = primitive_name
        self.result = result


def _outcome_value(value: object) -> str:
    return str(getattr(value, "value", value or ""))


def _autodrive_enabled() -> bool:
    """Whether the OAuth auto-drive behaviour is enabled (Task 6).

    Reads ``OPENRESEARCH_OAUTH_AUTODRIVE`` (truthy ``1``/``true``/``yes``);
    default OFF.  When ON, the degenerate-loop callback only emits the warning
    and does NOT early-abort — Task 6 fills the auto-drive recovery branch.
    """
    return os.environ.get("OPENRESEARCH_OAUTH_AUTODRIVE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


# Lifecycle stages the harness can DRIVE itself (Task 6). A degenerate root
# stuck before one of these has done strictly less than the harness can do for
# it, so the backstop drives exactly ONE step and hands control back.
_AUTODRIVE_DRIVABLE_STAGES = frozenset(
    {"need_baseline", "need_environment", "need_experiment"}
)

# Stage -> the structured directive the auto-drive backstop hands the root.
# Sibling maps (kept separate by design — different audiences/formats): the
# refusal-text map and the oauth REPL skeleton both live in
# forced_iteration.ForcedIterationPolicy (_STAGE_DIRECTIVES / _oauth_command_skeleton).
# If a primitive's call shape changes, update all three.
_AUTODRIVE_DIRECTIVES = {
    "need_baseline": (
        "The reproduction loop is stuck before implement_baseline. Assemble the "
        "plan (paper_claim_map + environment_spec + reproduction_contract) and "
        "call implement_baseline(plan) next — do NOT call FINAL_VAR again."
    ),
    "need_environment": (
        "Code exists but the environment is not built. Call "
        "build_environment(env_spec) next — do NOT call FINAL_VAR again."
    ),
    "need_experiment": (
        "Code and environment are ready but no experiment has run. Call "
        "run_experiment(code_path, env_id) next — do NOT call FINAL_VAR again."
    ),
}


def _autodrive_one_step(
    *,
    stage: str,
    tools: dict,
    ctx: "RunContext",
    emit: "Callable[[dict], None]",
    payload: dict,
) -> None:
    """Drive ONE missing lifecycle step on the root's behalf (Task 6 backstop).

    Marks a postmortem trail (``rlm_state/root_autodrive.json`` + a
    ``root_autodrive`` run_warning) and issues exactly ONE structured,
    stage-specific directive via ``recommend_next_tool`` — then returns so
    control flows back to the root.  See the inline v1-limitation note below for
    why the harness issues a directive rather than executing
    implement_baseline / build_environment / run_experiment itself.

    Fires at most once per no-progress streak: the policy's ``_degenerate_fired``
    latch is reset only by a state-changing primitive (implement_baseline /
    build_environment / run_experiment), and ``recommend_next_tool`` is NOT one —
    so issuing a directive does not itself re-arm the detector. (If the root
    subsequently does real work and then degenerates again, that state-change
    re-arms the latch and a fresh streak can fire a second directive — by
    design.)

    Every side effect (marker write, emit, directive dispatch) is independently
    fail-soft: a backstop that crashes the run is worse than no backstop.  The
    wrapped tool is called WITHOUT ``ctx`` — binding's wrapper pops/re-supplies
    it — so only the ``situation`` arg is passed.
    """
    signature = payload.get("signature")
    count = payload.get("count")
    required_stage = payload.get("required_stage")

    # 1a. Marker — postmortem trail.
    try:
        state_dir = ctx.project_dir / "rlm_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        marker = state_dir / "root_autodrive.json"
        marker.write_text(
            json.dumps(
                {
                    "stage": stage,
                    "signature": signature,
                    "count": count,
                    "required_stage": required_stage,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001 — marker is best-effort
        logger.exception("_autodrive_one_step: marker write failed")

    # 1b. Event — surface the backstop in the SSE stream.
    try:
        emit(
            build_run_warning_event(
                level="warn",
                code="root_autodrive",
                message=(
                    f"Auto-drive backstop: issued a structured directive for "
                    f"missing stage '{stage}'."
                ),
                data={
                    "stage": stage,
                    "signature": signature,
                    "count": count,
                    "required_stage": required_stage,
                },
            )
        )
    except Exception:  # noqa: BLE001 — emit must never block the drive
        logger.exception("_autodrive_one_step: event emit failed")

    # 2. Drive ONE step via a stage-specific structured directive.
    #
    # v1 LIMITATION (honest): implement_baseline(plan, *, ctx),
    # build_environment(env_spec, *, ctx) and run_experiment(code_path, env_id,
    # *, ...) all require root-assembled context (the plan / env_spec / code+env
    # ids) that the root builds in the REPL and that is NOT persisted to a fixed
    # disk location the harness can reconstruct. So the harness cannot faithfully
    # synthesize their arguments and CANNOT truly execute them in v1 — calling
    # them with no args would simply TypeError. The one primitive the harness can
    # fully drive is `recommend_next_tool` (it takes only a `situation` string),
    # so v1 issues exactly ONE structured, stage-specific directive that names the
    # concrete next call the root must make. This is the "one final structured
    # step" the plan permits; a TRUE harness-driven primitive execution needs the
    # lifecycle-state-machine refactor in the plan's Follow-on (persist the
    # plan/specs, then call the primitive directly here). The marker + event above
    # plus the Task-3 escalated refusal message remain the operative backstop.
    try:
        entry = tools.get("recommend_next_tool")
        if entry is not None and callable(entry.get("tool")):
            entry["tool"](
                situation=_AUTODRIVE_DIRECTIVES.get(stage, _AUTODRIVE_DIRECTIVES["need_baseline"])
            )
    except Exception:  # noqa: BLE001 — a failed drive must not crash the run
        logger.exception("_autodrive_one_step: drive dispatch failed (stage=%s)", stage)


def _make_degenerate_loop_callback(
    *,
    emit: "Callable[[dict], None]",
    ctx: "RunContext",
    policy: "ForcedIterationPolicy",
    autodrive_enabled: bool,
    tools: dict | None = None,
    oauth_root: bool = False,
):
    """Return the ``on_degenerate_refusal_loop`` callback.

    Default (autodrive OFF): emit a ``root_degenerate_refusal_loop``
    run_warning + mark a terminal stop so the run finalizes fast (the next
    FINAL_VAR is accepted via the ``root_degenerate_loop`` terminal class)
    instead of churning to the 16-refusal cap / wall clock.  Wall-clock floor
    and a pre-existing terminal stop take precedence.

    AUTODRIVE ON (Task 6, ``OPENRESEARCH_OAUTH_AUTODRIVE=1``, experimental):
    instead of early-aborting, for an oauth root stuck on a drivable lifecycle
    stage the harness DRIVES exactly one missing step (``implement_baseline`` /
    ``build_environment`` / ``run_experiment``) itself and hands control back to
    the root.  Inert (emit-only) for a non-oauth root, a non-drivable stage,
    near-wall-clock, an already-terminal stop, or when ``tools`` is unavailable.
    """

    def _on_degenerate(payload: dict) -> None:
        signature = payload.get("signature")
        count = payload.get("count")
        stage = payload.get("required_stage")
        try:
            emit(
                build_run_warning_event(
                    level="warn",
                    code="root_degenerate_refusal_loop",
                    message=(
                        f"Degenerate refusal loop: {count} no-progress FINAL_VAR "
                        f"refusals (signature={signature}, required_stage={stage}). "
                        "The root is looping without making lifecycle progress."
                    ),
                    data={
                        "signature": signature,
                        "count": count,
                        # `required_stage` is the canonical payload key (Tasks 2/4);
                        # `stage` is the alias the plan's Task 3 "payload.stage"
                        # names — both carry the same inferred lifecycle stage.
                        "required_stage": stage,
                        "stage": stage,
                    },
                )
            )
        except Exception:  # noqa: BLE001 — emit must never block the policy
            logger.exception(
                "run_pipeline_rlm: degenerate-loop warning emit failed"
            )

        # Task 6 — flag-gated OAuth auto-drive backstop. Drive ONE missing
        # lifecycle step (then hand control back to the root) ONLY when the flag
        # is ON *and every drive-gate holds* (oauth root, a harness-drivable
        # stage, wrapped tools present, not near the wall clock, no existing
        # terminal stop). In EVERY other case control MUST fall through to the
        # Task-4 early-abort below — autodrive=ON must never be LESS safe than
        # autodrive=OFF. (A guard-fail used to ``return`` here, leaving the
        # latched ``_degenerate_fired`` detector unable to re-fire, so the run
        # churned on to the refusal cap: the exact degenerate behaviour this
        # feature fixes.)
        if (
            autodrive_enabled
            and oauth_root
            and stage in _AUTODRIVE_DRIVABLE_STAGES
            and tools is not None
        ):
            remaining = None
            try:
                remaining = ctx.remaining_s()
            except Exception:  # noqa: BLE001
                remaining = None
            near_wall_clock = (
                remaining is not None and remaining <= _WALL_CLOCK_FLOOR_S
            )
            already_terminal = bool(getattr(ctx, "_terminal_stop_reason", None))
            if not near_wall_clock and not already_terminal:
                _autodrive_one_step(
                    stage=stage,
                    tools=tools,
                    ctx=ctx,
                    emit=emit,
                    payload=payload,
                )
                return
            # Near the wall clock or already terminal → do NOT drive; fall
            # through to the early-abort (it no-ops on those same conditions).

        # Task-4 early-abort — reached when autodrive is OFF, OR autodrive is ON
        # but a drive-gate failed (non-oauth root, un-drivable stage, no tools,
        # near wall clock, or already terminal).
        # Precedence guard: never override a near-wall-clock or an
        # already-terminal stop.
        remaining = None
        try:
            remaining = ctx.remaining_s()
        except Exception:  # noqa: BLE001
            remaining = None
        if remaining is not None and remaining <= _WALL_CLOCK_FLOOR_S:
            return
        if getattr(ctx, "_terminal_stop_reason", None):
            return

        # Early-abort: mark terminal so the next FINAL_VAR is accepted and the
        # run finalizes via the existing hard-stop path with a clear reason.
        try:
            ctx._terminal_stop_reason = {
                "kind": "root_degenerate_loop",
                "failure_class": "root_degenerate_loop",
                "signature": signature,
                "count": count,
                "required_stage": stage,
            }
        except Exception:  # noqa: BLE001
            logger.exception(
                "run_pipeline_rlm: could not stamp degenerate terminal stop"
            )
        try:
            policy.note_terminal_failure("root_degenerate_loop")
        except Exception:  # noqa: BLE001
            logger.exception(
                "run_pipeline_rlm: note_terminal_failure(root_degenerate_loop) failed"
            )

    return _on_degenerate


def _record_last_primitive_result_tools(
    custom_tools: dict,
    ctx: RunContext,
    repair_policy_holder: list | None = None,
) -> dict:
    """Wrap RLM tools so run.py can gate on the last primitive outcome.

    This sits outside binding.py to keep PR-alpha's fatal policy local to the
    orchestrator. Tool behavior is otherwise unchanged.

    ``repair_policy_holder``, when supplied, must be a single-element list
    ``[ForcedIterationPolicy]`` that is populated after the policy is created
    (the tools are wrapped before the policy exists). When run_experiment
    returns outcome="repairable", the wrapper calls
    ``policy.record_repair_attempt(failure_class)`` on the policy in the
    holder. This is the PR-α-followup repair-iteration accounting hook.
    """
    wrapped_tools: dict = {}

    def _wrap_tool(name: str, tool: Any) -> Any:
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            result = tool(*args, **kwargs)
            # Task 4: implement_baseline / build_environment are state-changing
            # primitives — calling either is genuine lifecycle progress, so reset
            # the no-progress refusal counter (the plan treats implement_baseline /
            # build_environment / run_experiment all as progress; run_experiment
            # resets via record_run_experiment()). This prevents a root that does
            # real work between premature FINAL_VARs from falsely tripping the
            # degenerate-loop detector. These two primitives do not carry an
            # "outcome" key, so this sits ABOVE the outcome gate below.
            if name in ("implement_baseline", "build_environment") and repair_policy_holder:
                try:
                    repair_policy_holder[0].record_state_change()
                except Exception:  # noqa: BLE001 — never crash a tool wrapper
                    logger.exception(
                        "_record_last_primitive_result_tools: record_state_change failed"
                    )
            if isinstance(result, dict) and "outcome" in result:
                setattr(ctx, "_last_primitive_name", name)
                setattr(ctx, "_last_primitive_result", result)
                # PR-α followup: when run_experiment yields a repairable
                # outcome, notify the forced-iteration policy so it can
                # enforce the repair-iteration floor.
                if (
                    name == "run_experiment"
                    and _outcome_value(result.get("outcome")) == "repairable"
                    and repair_policy_holder
                ):
                    policy = repair_policy_holder[0]
                    failure_class = str(result.get("failure_class") or "unknown")
                    try:
                        policy.record_repair_attempt(failure_class)
                    except Exception:  # noqa: BLE001 — never crash a tool wrapper
                        logger.exception(
                            "_record_last_primitive_result_tools: record_repair_attempt failed"
                        )
                elif (
                    name == "run_experiment"
                    and _outcome_value(result.get("outcome")) not in ("repairable", "partial_evidence", "fatal")
                    and repair_policy_holder
                ):
                    # P3 fix-first: a SUCCESS run_experiment clears the repair trigger so
                    # the loop stops forcing repairs once real evidence lands on disk.
                    # INERT (clear_repair_trigger is a no-op) unless evidence_fingerprint
                    # is wired — byte-identical to today when both flags are off.
                    try:
                        repair_policy_holder[0].clear_repair_trigger()
                    except Exception:  # noqa: BLE001 — never crash a tool wrapper
                        logger.exception(
                            "_record_last_primitive_result_tools: clear_repair_trigger failed"
                        )
                # comp 4b (2026-05-31): a terminal capacity/OOM stop is NOT repairable
                # by re-running the same config — notify the policy so forced_iteration
                # accepts the next FINAL_VAR (stop + report) instead of re-OOMing the
                # same matrix. This is INDEPENDENT of the repairable branch above: a
                # terminal cell-matrix result carries aggregated metrics, so it
                # classifies as partial_evidence (not repairable), and the
                # record_repair_attempt path would never fire for it.
                if name == "run_experiment" and repair_policy_holder:
                    _stop = result.get("stop_reason")
                    _stop_kind = _stop.get("kind") if isinstance(_stop, dict) else None
                    _terminal_class = _stop_kind or result.get("failure_class")
                    if _terminal_class in _TERMINAL_FAILURE_CLASSES:
                        policy = repair_policy_holder[0]
                        # Stash for build_final_report so final_report.json carries
                        # the structured stop_reason (done-criteria #3).
                        setattr(
                            ctx, "_terminal_stop_reason",
                            _stop if isinstance(_stop, dict) and _stop.get("kind")
                            else {"kind": str(_terminal_class)},
                        )
                        try:
                            policy.note_terminal_failure(str(_terminal_class))
                            logger.warning(
                                "run_experiment returned terminal stop '%s' — accepting "
                                "the next FINAL_VAR (stop + report, no re-OOM loop)",
                                _terminal_class,
                            )
                        except Exception:  # noqa: BLE001 — never crash a tool wrapper
                            logger.exception(
                                "_record_last_primitive_result_tools: note_terminal_failure failed"
                            )
            return result

        _wrapped.__name__ = getattr(tool, "__name__", name)
        return _wrapped

    for name, entry in custom_tools.items():
        if isinstance(entry, dict) and callable(entry.get("tool")):
            wrapped_tools[name] = {**entry, "tool": _wrap_tool(name, entry["tool"])}
        else:
            wrapped_tools[name] = entry
    return wrapped_tools


def _fatal_primitive_result(ctx: RunContext | None) -> tuple[str, dict] | None:
    if ctx is None:
        return None
    result = getattr(ctx, "_last_primitive_result", None)
    if not isinstance(result, dict):
        return None
    if _outcome_value(result.get("outcome")) != "fatal":
        return None
    return (str(getattr(ctx, "_last_primitive_name", "unknown")), result)


class _FatalBackendGateLogger(OpenResearchRLMLogger):
    """Logger hook that aborts before RLM appends fatal REPL output to history."""

    def log(self, iteration: Any) -> None:
        super().log(iteration)
        self._register_iteration_progress(bool(getattr(iteration, "code_blocks", None)))
        fatal = _fatal_primitive_result(getattr(self, "_ctx", None))
        if fatal is not None:
            primitive_name, result = fatal
            raise _FatalPrimitiveAbort(
                primitive_name=primitive_name,
                result=result,
            )

    def _register_iteration_progress(self, has_code_block: bool) -> None:
        """Abort a degenerate no-code-block loop (a root not driving the REPL).

        A root that emits only prose (no ```repl``` block) for
        OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD consecutive iterations is not
        making progress and would otherwise churn to the rlm iteration cap. Any
        iteration with a code block resets the streak, so healthy runs are
        byte-identical.
        """
        if has_code_block:
            self._empty_iter_streak = 0
            return
        streak = getattr(self, "_empty_iter_streak", 0) + 1
        self._empty_iter_streak = streak
        if streak < _default_degenerate_threshold():
            return
        # best-effort: reuse the existing degenerate-loop run_warning emission
        policy = _current_policy()
        cb = getattr(policy, "on_degenerate_refusal_loop", None) if policy else None
        if cb is not None:
            try:
                cb({"signature": "empty_code_block", "count": streak, "required_stage": None})
            except Exception:  # noqa: BLE001 — emit must never block the abort
                logger.exception("empty-iteration degenerate callback raised")
        raise _FatalPrimitiveAbort(
            primitive_name="root_degenerate_loop",
            result={
                "error": (
                    f"root emitted {streak} consecutive iterations with no code "
                    "block (pure prose) — it is not driving the RLM REPL loop"
                ),
                "failure_class": "root_degenerate_loop",
                "suggested_fix": (
                    "Use a validated agentic root (gpt-5 / claude / grok-4.3). "
                    "Chat-only models (e.g. ChatGPT-latest) refuse the RLM REPL premise."
                ),
            },
        )


def _fatal_error_payload(abort: _FatalPrimitiveAbort) -> dict[str, Any]:
    result = abort.result
    metrics = result.get("metrics")
    return {
        "primitive": abort.primitive_name,
        "outcome": "fatal",
        "error": result.get("error") or "fatal primitive outcome",
        "failure_class": result.get("failure_class"),
        "suggested_fix": result.get("suggested_fix"),
        "metrics_present": isinstance(metrics, dict) and bool(metrics),
    }


def _partial_evidence_from_experiment_runs(project_dir: Path) -> list[dict[str, Any]]:
    path = project_dir / "experiment_runs.jsonl"
    if not path.exists():
        return []
    evidence: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            metrics = entry.get("metrics")
            if isinstance(metrics, dict) and metrics:
                evidence.append({
                    "timestamp": entry.get("timestamp"),
                    "success": entry.get("success"),
                    "metrics": metrics,
                    "failure_class": entry.get("failure_class"),
                    "model_id": entry.get("model_id"),
                    "eval_env": entry.get("eval_env"),
                })
    except OSError:
        return []
    return evidence


def _notify_run_terminal(project_dir: Path) -> None:
    """Fire the opt-in terminal webhook (no-op unless OPENRESEARCH_NOTIFY_WEBHOOK_URL is set)."""
    try:
        from backend.agents.rlm.run_notify import notify_run_terminal
        notify_run_terminal(project_dir)
    except Exception:  # noqa: BLE001 — a notification must never affect run outcome
        logger.debug("run-notify: terminal helper raised", exc_info=True)


def _run_finalize_validation_panel(ctx: Any, report: Any, project_dir: Path) -> None:
    """OFFLINE adversarial validation panel (report-stamping only), shared by every
    ctx-bearing finalize path so the grok validator runs on the normal, fatal-abort,
    AND hard-stop paths — not just the happy one. Gated by OPENRESEARCH_EXTERNAL_VALIDATOR
    + ctx.validator_client (unset/None -> no-op -> byte-identical). Reuses a verdict the
    P3 FINAL_VAR gate already persisted for this evidence (no duplicate panel). Fail-soft:
    a panel failure must NEVER break finalize."""
    try:
        from backend.agents.rlm.external_validator import (  # noqa: PLC0415
            external_validator_enabled,
            run_validation_panel,
            persist_verdict,
        )
        _val_client = getattr(ctx, "validator_client", None)
        if external_validator_enabled() and _val_client is not None:
            # Gather metrics from the report's baseline_metrics if present, else on-disk.
            _val_metrics: dict = dict(report.baseline_metrics) if report.baseline_metrics else {}
            if not _val_metrics:
                _mpath = project_dir / "code" / "metrics.json"
                if _mpath.exists():
                    try:
                        _val_metrics = json.loads(_mpath.read_text(encoding="utf-8"))
                    except Exception:  # noqa: BLE001
                        _val_metrics = {}
            # Consume a verdict the P3 validator gate ALREADY persisted for THIS
            # evidence (spec §7.1: the panel is invoked once at the FINAL_VAR-attempt
            # and consumed — NOT re-run — by _finalize). Skipping the re-run avoids a
            # duplicate LLM panel and a later stochastic verdict overwriting the gate's.
            from backend.agents.rlm.external_validator import (  # noqa: PLC0415
                evidence_fingerprint as _val_efp,
                load_verdict as _load_verdict,
            )
            _already_validated = (
                _load_verdict(project_dir, expect_fingerprint=_val_efp(_val_metrics)) is not None
            )
            if _already_validated:
                logger.info("finalize-validation: reusing the validator verdict from the FINAL_VAR gate (no re-run)")
            else:
                # Gather leaf records from rubric_evaluation.json (best-effort).
                _leaf_records: list[dict] = []
                _eval_p = project_dir / "rubric_evaluation.json"
                if _eval_p.exists():
                    try:
                        _eval_data = json.loads(_eval_p.read_text(encoding="utf-8"))
                        _leaf_records = list(_eval_data.get("leaf_scores", {}).values())
                    except Exception:  # noqa: BLE001
                        _leaf_records = []
                _val_tier = _validator_separation_tier(getattr(ctx, "role_selection", None))
                _val_label = os.environ.get("OPENRESEARCH_VALIDATOR_MODEL", "").strip() or \
                             os.environ.get("OPENRESEARCH_VALIDATOR_BACKEND", "").strip() or "validator"
                # §4.5: thread report claims into the panel when OPENRESEARCH_VALIDATOR_CHECK_REPORT=1.
                _report_claims_for_panel = None
                if os.environ.get("OPENRESEARCH_VALIDATOR_CHECK_REPORT", "").strip() == "1":
                    try:
                        from backend.agents.rlm.claim_grounding import extract_result_claims as _erc  # noqa: PLC0415
                        _summary = getattr(report, "reproduction_summary", "") or ""
                        _rm = getattr(report, "reported_metrics", None)
                        _rm_text = (
                            _rm if isinstance(_rm, str) else
                            (__import__("json").dumps(_rm, default=str) if _rm else "")
                        )
                        _report_claims_for_panel = _erc(_summary + "\n" + _rm_text)
                    except Exception:  # noqa: BLE001
                        _report_claims_for_panel = None
                _verdict = run_validation_panel(
                    validator_client=_val_client,
                    panel_models=[_val_label],
                    metrics=_val_metrics,
                    project_dir=project_dir,
                    leaf_records=_leaf_records,
                    separation=_val_tier,
                    report_claims=_report_claims_for_panel,
                )
                persist_verdict(project_dir, _verdict)
                logger.info(
                    "finalize-validation: validation panel complete — status=%s veto_set=%r separation=%s",
                    _verdict.status, _verdict.veto_set, _verdict.separation,
                )
    except Exception:  # noqa: BLE001 — panel failure must never break finalize
        logger.warning("finalize-validation: external validation panel failed (non-fatal)", exc_info=True)


def _finalize_fatal_primitive_abort(
    *,
    abort: _FatalPrimitiveAbort,
    ctx: RunContext,
    iterations: int,
    project_dir: Path,
    emit: Any,
    tools_label: str = "real",
) -> RLMRunResult:
    error_payload = _fatal_error_payload(abort)
    try:
        emit({
            "event": "run_fatal",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iteration": iterations,
            "error": error_payload,
        })
    except Exception:  # noqa: BLE001 — finalization should continue
        logger.exception("run_pipeline_rlm: could not emit run_fatal event")

    primitive_provider = "stub" if "stub" in tools_label.lower() else "real"
    _write_demo_status(
        project_dir,
        "failed",
        error=error_payload,
        primitive_provider=primitive_provider,
    )

    evidence = _partial_evidence_from_experiment_runs(project_dir)
    baseline_metrics: dict[str, Any] = {}
    if len(evidence) == 1:
        baseline_metrics = evidence[0]["metrics"]
    elif evidence:
        baseline_metrics = {"partial_experiment_runs": evidence}
    report = RLMFinalReport(
        verdict="partial" if evidence else "failed",
        reproduction_summary=(
            "The run stopped at an orchestrator fatal-backend gate: "
            f"{error_payload['error']}"
        ),
        baseline_metrics=baseline_metrics,
        iterations=iterations,
        primitive_provider=primitive_provider,
        degraded=True,
        mode="rlm",
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    # Re-grade the completed grid before shipping a fatal-abort partial — the
    # abort may follow a finished grid that was never graded (same coverage as
    # _finalize; this path also has ctx). Best-effort.
    try:
        from backend.agents.rlm import finalize_regrade as _fr
        _fr.regrade_and_emit(ctx, report, emit)
    except Exception:  # noqa: BLE001
        logger.warning("_finalize_fatal_primitive_abort: regrade failed (non-fatal)", exc_info=True)
    # Run the adversarial validator on the abort path too (gated; byte-identical when off) —
    # a fabrication-guard abort must not skip the critic. Before write so the stamp sees it.
    _run_finalize_validation_panel(ctx, report, project_dir)
    json_path, _md_path = write_final_report_rlm(
        report, project_dir, run_experiment_calls=run_experiment_call_count(ctx),
        run_experiment_ok_calls=run_experiment_success_count(ctx),
        run_experiment_partial_timeout_calls=run_experiment_partial_timeout_count(ctx)
    )
    _notify_run_terminal(project_dir)

    # Positive recipe admission (OPENRESEARCH_POSITIVE_RECIPES, default-OFF).
    # Mirrors the mine_lessons pattern; fail-soft so a bookkeeping error never
    # interrupts this finalize path.
    try:
        from backend.agents.rlm import recipe_library as _rl
        from backend.agents.prompts.paper_hints import PAPER_HINTS as _PH
        from backend.agents.rlm.external_validator import load_verdict as _load_verdict
        if _rl.positive_recipes_enabled():
            _report_dict = report.model_dump() if hasattr(report, "model_dump") else (report if isinstance(report, dict) else {})
            _verdict = _load_verdict(project_dir)
            _paper_class = _rl.derive_paper_class(
                arxiv_id=getattr(ctx, "arxiv_id", None), paper_hints=_PH, rubric=_report_dict.get("rubric"),
            )
            _rl.admit_recipe(project_dir, project_dir.parent, report=_report_dict, validator_verdict=_verdict, paper_class=_paper_class)
    except Exception:  # noqa: BLE001 — recipe bookkeeping must never break finalize
        logger.debug("_finalize_fatal_primitive_abort: recipe admission skipped", exc_info=True)

    try:
        emit(
            build_run_complete_event(
                status="failed",
                iterations=iterations,
                rubric_score=None,
                cost_usd=None,
                final_report_path=str(json_path),
            )
        )
    except Exception:  # noqa: BLE001
        logger.exception("run_pipeline_rlm: could not emit fatal run_complete event")

    return RLMRunResult(
        project_id=ctx.project_id,
        status="failed",
        iterations=iterations,
        rubric_score=None,
        cost_usd=None,
        final_report_path=str(json_path),
    )


def _salvage_partial_report(
    report: "RLMFinalReport", project_dir: Path, *, stop_kind: str, stop_detail: str,
) -> "float | None":
    """Fold the run's already-earned evidence into a hard-stop report (in place).

    2026-06-09 All-CNN: the watchdog shipped ``rubric.overall_score=None`` +
    ``verdict="failed"`` while the run had recorded a real 0.49 rubric score —
    the hard-stop path bypassed ``build_final_report`` and with it the
    best-of-run floor. This helper applies the floor from dashboard events,
    reconciles the verdict against the salvaged score (downgrade-only ceiling,
    starting from the honest "partial" claim), and attaches a structured
    ``stop_reason``. Fail-soft: any error leaves the bare report intact.
    Returns the salvaged overall score (or None).
    """
    score: float | None = None
    try:
        from backend.agents.rlm.report import (
            _apply_best_of_run_floor,
            reconcile_verdict_with_score,
        )
        rubric = _apply_best_of_run_floor(dict(report.rubric or {}), project_dir)
        raw = rubric.get("overall_score")
        score = float(raw) if raw is not None else None
        if score is not None:
            report.rubric = rubric
            report.verdict = reconcile_verdict_with_score("partial", score)
            report.reproduction_summary = (
                (report.reproduction_summary or "")
                + f"\n\n[salvage] Best rubric score recorded before the stop: "
                f"{score:.3f} (best_of_run) — preserved in this report."
            ).strip()
    except Exception:  # noqa: BLE001 — salvage must never block the report write
        logger.exception("run_pipeline_rlm: hard-stop salvage failed (shipping bare report)")
    try:
        report.stop_reason = {"kind": stop_kind, "detail": stop_detail}
    except Exception:  # noqa: BLE001 — schema drift must not block the write
        logger.debug("run_pipeline_rlm: could not attach stop_reason", exc_info=True)
    return score


def _hard_stop_with_report(
    *,
    project_dir: Path,
    emit: Any,
    done: int,
    summary: str,
    status_error: str,
    exit_code: int,
    stop_kind: str = "hard_stop",
    ctx: Any = None,
) -> None:
    """Ship a partial report, emit ``run_complete``, flip demo_status, and
    hard-exit — the single "never die without a report" path shared by the wall-clock
    watchdog and the SIGTERM finalizer (2026-06-01). Every step is best-effort so a
    failure writing one artifact never blocks the others or the exit. The report
    carries the run's best recorded rubric score + a reconciled verdict (salvage,
    2026-06-09) instead of an unconditional scoreless ``failed``.

    ``ctx`` (captured at run start) carries the llm_client that lets salvage RE-GRADE the
    completed-but-never-verified grid before flooring — Adam's long runs hit
    the wall-clock here, and a grid that finished without a verify has a best
    RECORDED score of zero, so without this the watchdog ships 0 over a grid
    that earned its score (2026-06-13).
    """
    report = RLMFinalReport(verdict="failed", reproduction_summary=summary, iterations=done)
    # Re-grade the completed grid FIRST so the salvage floor can see it (writes
    # rubric_evaluation.json, which _salvage_partial_report's best-of-run floor
    # and write_final_report_rlm's merge both read).
    try:
        from backend.agents.rlm import finalize_regrade as _fr
        _fresh = _fr.regrade_for_hard_stop(project_dir, getattr(ctx, "llm_client", None))
        if _fresh is not None:
            _fr_emit = emit if callable(emit) else (lambda *a, **k: None)
            _fr_emit("run_warning", {
                "code": "finalize_regrade_hardstop",
                "message": (
                    "hard-stop re-graded the completed grid → "
                    f"{_fresh.get('overall_score')} (was un-graded)."
                ),
            })
    except Exception:  # noqa: BLE001 — salvage re-grade is best-effort
        logger.warning("hard-stop: regrade_for_hard_stop failed (non-fatal)", exc_info=True)
    salvaged_score = _salvage_partial_report(
        report, project_dir, stop_kind=stop_kind, stop_detail=status_error,
    )
    # Run the adversarial validator on the wall-clock/SIGTERM hard-stop path too (gated;
    # byte-identical when off). ctx may be None here (the watchdog can fire pre-bind).
    if ctx is not None:
        _run_finalize_validation_panel(ctx, report, project_dir)
    try:
        # Evidence-gate trust counts (audit 2026-06-11): without these the
        # watchdog/SIGTERM path fell back to content-only trust — a forging
        # root could wedge past the deadline and ship a forged 'partial'
        # (the exact class the gate closes on the FINAL_VAR/fatal paths).
        write_final_report_rlm(
            report,
            project_dir,
            run_experiment_calls=run_experiment_call_count(ctx) if ctx is not None else None,
            run_experiment_ok_calls=run_experiment_success_count(ctx),
        run_experiment_partial_timeout_calls=run_experiment_partial_timeout_count(ctx) if ctx is not None else None,
        )
    except Exception:  # noqa: BLE001
        logger.exception("run_pipeline_rlm: hard-stop could not write final report")
    _notify_run_terminal(project_dir)
    # Positive recipe admission (OPENRESEARCH_POSITIVE_RECIPES, default-OFF).
    # The ctx guard is load-bearing here: _hard_stop_with_report has ctx: Any = None.
    try:
        from backend.agents.rlm import recipe_library as _rl
        from backend.agents.prompts.paper_hints import PAPER_HINTS as _PH
        from backend.agents.rlm.external_validator import load_verdict as _load_verdict
        if ctx is not None and _rl.positive_recipes_enabled():
            _report_dict = report.model_dump() if hasattr(report, "model_dump") else (report if isinstance(report, dict) else {})
            _verdict = _load_verdict(project_dir)
            _paper_class = _rl.derive_paper_class(
                arxiv_id=getattr(ctx, "arxiv_id", None), paper_hints=_PH, rubric=_report_dict.get("rubric"),
            )
            _rl.admit_recipe(project_dir, project_dir.parent, report=_report_dict, validator_verdict=_verdict, paper_class=_paper_class)
    except Exception:  # noqa: BLE001 — recipe bookkeeping must never break finalize
        logger.debug("_hard_stop_with_report: recipe admission skipped", exc_info=True)
    try:
        emit(
            build_run_complete_event(
                status="failed",
                iterations=done,
                rubric_score=salvaged_score,
                cost_usd=None,
                final_report_path=str(project_dir / "final_report.json"),
            )
        )
    except Exception:  # noqa: BLE001
        logger.exception("run_pipeline_rlm: hard-stop could not emit run_complete event")
    try:
        _write_demo_status(project_dir, "failed", error=status_error)
    except Exception:  # noqa: BLE001
        logger.exception("run_pipeline_rlm: hard-stop could not write demo_status")
    os._exit(exit_code)


def _arm_watchdog(
    deadline_s: float | None,
    *,
    project_dir: Path,
    emit: Any,
    iteration_count: Any,
    ctx: Any = None,
) -> threading.Timer | None:
    """Arm the process-level wall-clock backstop (design spec §8, Codex H2).

    ``rlm``'s ``max_timeout`` only checks between iterations; a primitive wedged
    inside ``execute_code`` can overrun it indefinitely, and a Python thread
    cannot be killed.  This timer fires ``_WATCHDOG_GRACE_S`` past the deadline
    (so it only triggers when ``rlm``'s own timeout failed), writes an honest
    partial report, and hard-exits the process — the OS then reclaims the wedged
    worker thread.

    ``iteration_count`` is a zero-arg callable returning the iterations done so
    far.  Returns a handle exposing ``.cancel()`` — the caller must call it on
    normal completion. When ``deadline_s`` is ``None`` (no explicit
    ``--max-wall-clock``), the watchdog falls back to the always-on hard-ceiling
    backstop (``_watchdog_hard_ceiling_s``) so a wedged/hung run still ships a
    partial report; it returns ``None`` (fully bypassed) only when that backstop
    is disabled via ``OPENRESEARCH_WATCHDOG_HARD_CEILING_S=0``.

    Sleep-robust (2026-05-30, ported): a ``threading.Timer`` waits on a
    MONOTONIC clock that PAUSES during macOS system sleep, so a closed lid
    stretched a 2h deadline to ~5h before the timer fired. This polls real
    wall-clock ``time.time()`` (which counts sleep) against an absolute
    deadline, so on wake it fires within one poll interval regardless of how
    long the machine slept.
    """
    if deadline_s is None:
        ceiling = _watchdog_hard_ceiling_s()
        if ceiling <= 0:
            return None  # operator opted fully out — truly unbounded
        deadline_s = ceiling
        logger.warning(
            "run_pipeline_rlm: no explicit wall-clock ceiling; arming the always-on "
            "watchdog backstop at %.0fs (OPENRESEARCH_WATCHDOG_HARD_CEILING_S=0 disables)",
            deadline_s,
        )
    def _fire() -> None:
        logger.error(
            "run_pipeline_rlm: wall-clock watchdog fired (%.0fs + %.0fs grace) — "
            "hard-stopping a wedged run",
            deadline_s,
            _WATCHDOG_GRACE_S,
        )
        done = iteration_count()
        _hard_stop_with_report(
            project_dir=project_dir,
            emit=emit,
            done=done,
            summary=(
                f"Wall-clock watchdog: the run exceeded {deadline_s:.0f}s "
                f"and was hard-stopped after {done} iteration(s)."
            ),
            status_error=(
                f"wall-clock watchdog: run hard-stopped past its {deadline_s:.0f}s deadline"
            ),
            exit_code=_WATCHDOG_EXIT_CODE,
            ctx=ctx,
            stop_kind="wall_clock_watchdog",
        )

    import time as _time
    fire_at = _time.time() + deadline_s + _WATCHDOG_GRACE_S
    stop_event = threading.Event()

    def _poll() -> None:
        # stop_event.wait() also waits on a monotonic clock, but only for one
        # poll interval at a time — on wake the in-flight wait finishes within
        # <= _WATCHDOG_POLL_S of real post-wake time, then the time.time() check
        # sees the full elapsed wall clock and fires.
        while not stop_event.wait(_WATCHDOG_POLL_S):
            if _time.time() >= fire_at:
                _fire()
                return

    threading.Thread(
        target=_poll, name="rlm-wallclock-watchdog", daemon=True
    ).start()

    class _WatchdogHandle:
        """Cancel handle for the polling watchdog. ``interval`` mirrors the
        old ``threading.Timer.interval`` (armed delay in seconds) so callers
        and tests can introspect what was armed."""

        __slots__ = ("interval",)

        def __init__(self, interval: float) -> None:
            self.interval = interval

        def cancel(self) -> None:
            stop_event.set()

    return _WatchdogHandle(deadline_s + _WATCHDOG_GRACE_S)


def _install_sigterm_finalizer(
    *,
    project_dir: Path,
    emit: Any,
    iteration_count: Any,
    ctx: Any = None,
) -> Any:
    """On SIGTERM, ship a partial report before exiting instead of dying silently.

    The 2026-06-01 SDAR run was KILLED by the operator after it appeared stuck and
    left NO ``final_report``. A run launched by the batch scheduler is stopped with
    SIGTERM then SIGKILL-after-grace; catching SIGTERM lets us write the report
    during that grace window. Returns the previously-installed handler (to restore
    on clean completion) or ``None`` when not installed — off the main thread (where
    signals cannot be set) or on a platform/handler error. ``SIGKILL`` (kill -9)
    stays uncatchable by design; that case is covered by the batch scheduler.
    """
    if threading.current_thread() is not threading.main_thread():
        return None
    try:
        prev = signal.getsignal(signal.SIGTERM)
    except (ValueError, OSError):
        return None

    def _on_sigterm(signum: int, frame: Any) -> None:  # noqa: ARG001
        logger.error(
            "run_pipeline_rlm: SIGTERM received — shipping a partial report before exit"
        )
        done = iteration_count() if callable(iteration_count) else 0
        _hard_stop_with_report(
            project_dir=project_dir,
            emit=emit,
            done=done,
            summary=(
                f"Run terminated by SIGTERM after {done} iteration(s); a partial "
                f"report was written from last-known state."
            ),
            status_error="run terminated by SIGTERM",
            exit_code=143,  # 128 + SIGTERM(15)
            stop_kind="sigterm",
            ctx=ctx,
        )

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        logger.warning(
            "run_pipeline_rlm: could not install SIGTERM finalizer", exc_info=True
        )
        return None
    return prev


# ---------------------------------------------------------------------------
# PR-ι.3 — Rolling cost surfacing
# ---------------------------------------------------------------------------


def _compute_cost_summary(project_dir: Path, iteration_count: int) -> dict:
    """Aggregate cost_ledger.jsonl into a cost_summary dict.

    Reads cost entries from disk and computes:
    - usd_total: sum of all entries
    - usd_this_iter: sum of entries for the current iteration
    - iter_count: current iteration number
    - usd_per_iter_p50: median USD spend per iteration (over completed iterations)

    Fail-soft: returns a minimal dict on any I/O / parse error.
    """
    import json as _json
    import statistics as _stats

    ledger_path = project_dir / "cost_ledger.jsonl"
    if not ledger_path.exists():
        return {
            "usd_total": 0.0,
            "usd_this_iter": 0.0,
            "iter_count": iteration_count,
            "usd_per_iter_p50": 0.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    try:
        lines = ledger_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {
            "usd_total": 0.0,
            "usd_this_iter": 0.0,
            "iter_count": iteration_count,
            "usd_per_iter_p50": 0.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    entries: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(_json.loads(line))
        except _json.JSONDecodeError:
            continue

    if not entries:
        return {
            "usd_total": 0.0,
            "usd_this_iter": 0.0,
            "iter_count": iteration_count,
            "usd_per_iter_p50": 0.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    usd_total = sum(float(e.get("cost_usd") or e.get("estimated_usd") or 0.0) for e in entries)

    # Group by iteration index (approximated via timestamp order).
    # The ledger has no iteration tag, so we split costs into per-iteration
    # buckets by dividing the entry list into `iteration_count` equal slices.
    per_iter_usd: list[float] = []
    if iteration_count > 0:
        slice_size = max(1, len(entries) // iteration_count)
        for i in range(iteration_count):
            start = i * slice_size
            end = start + slice_size if i < iteration_count - 1 else len(entries)
            bucket = entries[start:end]
            per_iter_usd.append(
                sum(float(e.get("cost_usd") or e.get("estimated_usd") or 0.0) for e in bucket)
            )
        usd_this_iter = per_iter_usd[-1] if per_iter_usd else 0.0
        p50 = float(_stats.median(per_iter_usd)) if len(per_iter_usd) >= 2 else usd_this_iter
    else:
        usd_this_iter = usd_total
        p50 = 0.0

    return {
        "usd_total": round(usd_total, 6),
        "usd_this_iter": round(usd_this_iter, 6),
        "iter_count": iteration_count,
        "usd_per_iter_p50": round(p50, 6),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _update_cost_summary_loop(
    project_dir: Path,
    stop_event: threading.Event,
    iteration_count: Any,  # zero-arg callable
    interval_s: float = 30.0,
) -> None:
    """Background daemon: update demo_status.json::cost_summary every ``interval_s``s.

    Reads cost_ledger.jsonl and merges ``cost_summary`` into the existing
    demo_status.json via an atomic tmp-write. Fail-soft — never crashes the run.

    Also refreshes ``updatedAt`` (audit 2026-06-09): this loop is the only
    periodic writer during a run, so the refresh turns the orphan sweeper's
    staleness gate (run_liveness, default 120s) into a genuine process-written
    heartbeat — a LIVE run never looks stale, regardless of pid-namespace or
    user-id visibility from the sweeping process. When this process dies the
    refresh stops, staleness accrues, and the sweep proceeds as designed.
    """
    while not stop_event.wait(timeout=interval_s):
        try:
            cur_iter = iteration_count()
            summary = _compute_cost_summary(project_dir, cur_iter)
            status_path = project_dir / "demo_status.json"
            existing: dict = {}
            if status_path.exists():
                try:
                    existing = json.loads(status_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    existing = {}
            existing["cost_summary"] = summary
            if str(existing.get("status") or "") == "running":
                existing["updatedAt"] = (
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                )
            tmp = status_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            os.replace(tmp, status_path)
        except Exception:  # noqa: BLE001 — cost surfacing must never crash the run
            logger.debug("cost_summary_loop: update failed (will retry)", exc_info=True)


def _parse_gpu_device_ids() -> tuple[str, ...]:
    """Parse OPENRESEARCH_GPU_DEVICE_IDS (CSV of GPU UUIDs/indices) into a tuple.

    Empty / unset => () meaning "no explicit pin" (backend default). A batch
    launcher exports this per run so the experiment subprocess is pinned to a
    disjoint GPU subset; CUDA_VISIBLE_DEVICES is also set by the launcher, so
    this is the explicit, testable companion that flows into SandboxConfig.
    """
    raw = (os.environ.get("OPENRESEARCH_GPU_DEVICE_IDS") or "").strip()
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_gpu_parallelism() -> str:
    """OPENRESEARCH_GPU_PARALLELISM -> one of {auto,single,multi}; default 'auto'."""
    val = (os.environ.get("OPENRESEARCH_GPU_PARALLELISM") or "auto").strip().lower()
    return val if val in {"auto", "single", "multi"} else "auto"


def _visible_gpu_count() -> int | None:
    """Best-effort count of GPUs visible to this run, for the multi-GPU guidance
    hint. Prefer the explicit lease (OPENRESEARCH_GPU_DEVICE_IDS), then
    CUDA_VISIBLE_DEVICES; None when neither is set (agent relies on runtime
    torch.cuda.device_count() inside the sandbox)."""
    ids = _parse_gpu_device_ids()
    if ids:
        return len(ids)
    raw = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not raw:
        return None
    parts = [p for p in raw.split(",") if p.strip()]
    return len(parts) or None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _ensure_local_data_root(sandbox_mode: object, runs_root: Path) -> None:
    """Point the volume-mount data root at a writable dir for LOCAL sandboxes.

    Local hosts have no ``/workspace`` volume (that path is RunPod-only), yet
    ``config.runpod_volume_mount_path`` defaults to ``/workspace`` and the baseline
    DATASET-SETUP guidance defaults every dataset dir to ``/workspace/data/<env>``.
    On a local box ``os.makedirs('/workspace/...')`` raises PermissionError, the
    agent's env loader swallows it, and every algorithm reports ``env_load_failed``
    with zero reward while the GPUs sit idle (the 2026-05-29 SDAR local failure).
    Repoint ``OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH`` at a writable, SHARED (download-once)
    cache dir so ALFWorld/WebShop/HF setup actually succeeds.  No-op for runpod/docker
    (they keep the real ``/workspace`` volume); an explicit non-default operator
    override always wins.
    """
    import os as _os

    key = getattr(sandbox_mode, "value", str(sandbox_mode or "")).lower()
    if "local" not in key:
        return
    current = (_os.environ.get("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH") or "").strip()
    if current and current != "/workspace":
        return  # operator pinned an explicit writable root — respect it
    data_root = (runs_root / ".cache" / "data").resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    _os.environ["OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH"] = str(data_root)
    logger.info(
        "local sandbox: volume-mount data root → %s (writable shared cache; "
        "/workspace is RunPod-only)", data_root,
    )


def _maybe_auto_arm_cell_resume(project_dir: Path) -> bool:
    """Auto-arm cell-resume on a re-invoke of an UNFINISHED project (spot-restart safe).

    On a spot VM a preemption stops the run; restarting + re-invoking with the
    SAME --project-id should SKIP already-completed cells instead of redoing
    them. ``gpu_cell_runner.run_matrix`` already does that when
    ``OPENRESEARCH_RESUME_CELLS`` is truthy (it only skips a cell whose prior
    ``cell_manifest.json`` is ``status=ok`` AND fingerprint-matches — so a
    changed cells.json safely re-runs). This fills the gap: when a prior attempt
    left state under *project_dir* but never produced ``final_report.json``,
    arm resume automatically.

    An explicit ``OPENRESEARCH_RESUME_CELLS`` (set to ANY value, incl. "0")
    always wins — operator intent is never overridden. Returns True iff it
    armed (set the env var to "1") this call.
    """
    if os.environ.get("OPENRESEARCH_RESUME_CELLS") is not None:
        return False  # explicit operator setting wins (including "0")
    if (project_dir / "final_report.json").exists():
        return False  # already finished — nothing to resume
    prior_attempt = (
        (project_dir / "rlm_state").exists()
        or (project_dir / "experiment_runs.jsonl").exists()
    )
    if not prior_attempt:
        return False  # first attempt — no completed cells to skip
    os.environ["OPENRESEARCH_RESUME_CELLS"] = "1"
    return True


def _load_reusable_rubric(project_dir: Path) -> dict | None:
    """OPENRESEARCH_REUSE_RUBRIC=1 → the pre-seeded generated_rubric.json, else None.

    Rubric generation is an LLM call, so every re-run otherwise grades against
    a slightly different rubric — rubric drift alone moves scores. A/B arms
    (and rubric-stable re-run campaigns) pre-seed the project dir with the
    reference rubric and set the flag so score deltas measure the HARNESS
    change, not rubric variance. Default OFF; fail-soft — a missing/corrupt
    file returns None and the caller falls through to generation as before.
    """
    enabled = os.environ.get("OPENRESEARCH_REUSE_RUBRIC", "").strip().lower() not in (
        "", "0", "false", "off",
    )
    if not enabled:
        return None
    try:
        existing = json.loads(
            (project_dir / "generated_rubric.json").read_text(encoding="utf-8")
        )
        if isinstance(existing, dict) and existing:
            return existing
    except (OSError, json.JSONDecodeError, ValueError):
        logger.warning(
            "OPENRESEARCH_REUSE_RUBRIC set but no readable generated_rubric.json — "
            "falling through to rubric generation"
        )
    return None


def _validator_separation_tier(role_selection: Any) -> str:
    """Compute the model-lineage separation tier between executor and validator.

    Reads ``OPENRESEARCH_VALIDATOR_BACKEND`` and ``OPENRESEARCH_VALIDATOR_MODEL``
    from the environment (no side effects) and constructs a synthetic
    :class:`~backend.agents.rlm.role_models.RoleSpec` for the validator, then
    delegates to :func:`~backend.agents.rlm.role_models.separation_strength`.

    For the Azure×Azure case the executor's ACTUAL deployment
    (``AZURE_OPENAI_DEPLOYMENT``) is substituted so that two different Azure
    deployments read as "weak" rather than "degraded".

    Returns one of: ``"independent"`` | ``"weak"`` | ``"degraded"`` | ``"unavailable"``.
    """
    backend = os.environ.get("OPENRESEARCH_VALIDATOR_BACKEND", "").strip().lower()
    if not backend:
        return "unavailable"
    try:
        from backend.agents.rlm.role_models import (  # noqa: PLC0415
            RoleSpec,
            _classify_model_family,
            separation_strength,
            PROVIDER_AZURE,
            PROVIDER_ANTHROPIC_OAUTH,
            PROVIDER_ANTHROPIC,
            PROVIDER_OPENAI,
            PROVIDER_AZURE_FOUNDRY,
        )
        import dataclasses  # noqa: PLC0415

        _BACKEND_PROVIDER: dict[str, str] = {
            "azure": PROVIDER_AZURE,
            "oauth": PROVIDER_ANTHROPIC_OAUTH,
            "claude-oauth": PROVIDER_ANTHROPIC_OAUTH,
            "anthropic-oauth": PROVIDER_ANTHROPIC_OAUTH,
            "anthropic": PROVIDER_ANTHROPIC,
            "openai": PROVIDER_OPENAI,
            "azure-foundry": PROVIDER_AZURE_FOUNDRY,
            "foundry": PROVIDER_AZURE_FOUNDRY,
            "grok": PROVIDER_AZURE_FOUNDRY,
        }
        val_provider = _BACKEND_PROVIDER.get(backend, backend)
        # The actual validator model/deployment for the tier comparison.
        # For the azure backend, OPENRESEARCH_VALIDATOR_MODEL overrides the
        # deployment — same logic as build_transport_client's azure branch.
        val_model = (
            os.environ.get("OPENRESEARCH_VALIDATOR_MODEL", "").strip()
            or (
                os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
                if val_provider == PROVIDER_AZURE
                else ""
            )
            or None
        )
        val_spec = RoleSpec(
            role="validator",
            token=backend,
            provider=val_provider,
            model=val_model,
            family=_classify_model_family(val_provider, val_model),
        )
        exec_spec = getattr(role_selection, "executor", None) if role_selection is not None else None
        # For the Azure executor, substitute the ACTUAL deployment so that
        # azure(deployA) × azure(deployB) reads as "weak", not "degraded".
        if exec_spec is not None and exec_spec.provider == PROVIDER_AZURE:
            exec_spec = dataclasses.replace(
                exec_spec,
                model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip() or exec_spec.model,
            )
        return separation_strength(exec_spec, val_spec)
    except Exception:  # noqa: BLE001 — tier computation is advisory; never crashes a run
        logger.debug("_validator_separation_tier: failed, defaulting to unavailable", exc_info=True)
        return "unavailable"


async def run_pipeline_rlm(
    project_id: str,
    runs_root: Path,
    workspace_claim_map: dict[str, Any],
    *,
    model: str | None = None,
    provider: str | None = None,
    runtime: Any = None,
    run_budget: RunBudget | None = None,
    sandbox_mode: Any = DEFAULT_SANDBOX_MODE,
    seed: int | None = None,
    execution_profile: Any = None,
    attempt_id: str | None = None,
    run_group_id: str | None = None,
    workspace_service: Any = None,
    workspace_id: str | None = None,
    hybrid_repair_only: bool = False,
    phase1_weak_clusters: list | None = None,
) -> RLMRunResult:
    """Run one paper reproduction in ``rlm`` mode.

    ``seed`` / ``execution_profile`` / ``attempt_id`` / ``run_group_id`` are
    accepted for call-site parity; the RLM engine owns its own iteration loop,
    so they are not all load-bearing yet.

    Returns an :class:`RLMRunResult`; never raises for an in-run failure — a
    crashed or timed-out run yields an honest ``partial`` / ``failed`` report.
    """
    # Resolve to an absolute path. Primitives execute inside the RLM REPL,
    # whose working directory is NOT the repo root — a relative runs_root makes
    # every primitive's artifact write (dashboard_events.jsonl, code/, ...) fail
    # with FileNotFoundError. Absolute paths are CWD-independent.
    runs_root = Path(runs_root).resolve()
    project_dir = runs_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    # PR-π Module E — paper-text precondition gate.
    # Check before archiving so a fresh ingest failure is caught immediately,
    # not buried behind a new partial run. Uses the default from Settings
    # (allow_lossy_paper_text=True) so all existing callers proceed unchanged.
    _settings_for_gate = get_settings()
    # Bridge Azure OpenAI creds (canonical names + the portal KEY1/KEY2 aliases +
    # .env) into the process env before any Azure consumer reads os.environ
    # directly: the executor's make_runtime("azure"), the navigation accelerator,
    # and grader_transport. No-op (byte-identical) when no AZURE_OPENAI_* is set.
    from backend.agents.runtime.factory import configure_azure_openai_credentials
    configure_azure_openai_credentials(_settings_for_gate)
    _allow_lossy = getattr(_settings_for_gate, "allow_lossy_paper_text", True)
    _paper_degraded_reason = _assert_paper_text_precondition(project_dir, allow_lossy=_allow_lossy)

    # Disk-headroom preflight (2026-06-15): abort before any GPU work when the runs
    # root is near-full, so a run can't hang/orphan on checkpoint writes (the SDAR
    # disk-100% incident). Paper-agnostic; OPENRESEARCH_MIN_DISK_GB=0 disables it.
    try:
        _min_disk_gb = float(os.environ.get("OPENRESEARCH_MIN_DISK_GB", "10") or "10")
    except ValueError:
        _min_disk_gb = 10.0
    _disk_warn_reason = _assert_disk_headroom(runs_root, min_gb=_min_disk_gb)
    if _disk_warn_reason:
        logger.warning("%s", _disk_warn_reason)

    # Archive prior-attempt artifacts before touching anything else.
    # Fires only when final_report.json exists (a completed prior run);
    # first-ever runs and incomplete-but-failed runs are handled gracefully.
    from backend.services.runs.attempt_isolation import maybe_archive_prior_attempt
    _archived = maybe_archive_prior_attempt(project_id, runs_root)
    if _archived:
        logger.info(
            "run_pipeline_rlm[%s]: prior attempt archived to %s (%d item(s))",
            project_id, _archived["attempt_dir"], len(_archived["moved"]),
        )

    # Anti-regression seeding (2026-06-11, OPENRESEARCH_SEED_BEST_ATTEMPT): copy the
    # best prior attempt's working code into code/_best_attempt/ so the
    # implementer starts FROM the proven solution. Each Adam attempt used to
    # re-derive everything from scratch and routinely landed below the 0.831
    # baseline it had already achieved. Fail-soft + flag-gated inside the module.
    try:
        from backend.agents.rlm.best_attempt import seed_reference_code
        _seeded = seed_reference_code(runs_root / project_id)
        if _seeded:
            logger.info("run_pipeline_rlm[%s]: best-attempt reference seeded at %s",
                        project_id, _seeded)
    except Exception:  # noqa: BLE001 — seeding must never block the run
        logger.debug("run_pipeline_rlm: best-attempt seeding skipped", exc_info=True)

    # Status snapshot at run start — GET /runs/{id} reads this; without it a
    # CLI- or script-launched RLM run 404s. Terminal status is set in _finalize.
    # Surface a degraded-paper-text warning here (F-29) so an operator sees the
    # run is non-faithful; the merge in _write_demo_status carries it forward.
    _write_demo_status(
        project_dir,
        "running",
        warnings=[_paper_degraded_reason] if _paper_degraded_reason else None,
    )

    # Relaunchable config snapshot (audit 2026-06-09, cap-10): before this,
    # sandbox/model/provider/budgets/seed had to be reconstructed by hand to
    # re-launch an identical run (final_report carries only mode/models/scope).
    # Secrets are deliberately NOT written — these are launch parameters only.
    try:
        _snapshot = {
            "schema_version": 1,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "mode": "rlm",
            "project_id": project_id,
            "model": model,
            "provider": provider,
            "sandbox_mode": str(getattr(sandbox_mode, "value", sandbox_mode)),
            "seed": seed,
            "attempt_id": attempt_id,
            "run_group_id": run_group_id,
            "hybrid_repair_only": hybrid_repair_only,
            "max_usd": getattr(run_budget, "max_usd", None) if run_budget is not None else None,
            "max_wall_clock_seconds": (
                getattr(run_budget, "max_wall_clock_seconds", None) if run_budget is not None else None
            ),
            "max_pod_seconds": getattr(run_budget, "max_pod_seconds", None) if run_budget is not None else None,
            "env_flags": {
                k: _redact_env_value(v)
                for k, v in sorted(os.environ.items())
                if k.startswith("OPENRESEARCH_")
                and k not in _ENV_SNAPSHOT_DENY_EXACT
                and not any(t in k for t in ("KEY", "SECRET", "TOKEN", "PASSWORD"))
            },
        }
        _cfg_tmp = project_dir / "run_config.json.tmp"
        _cfg_tmp.write_text(json.dumps(_snapshot, indent=2, default=str), encoding="utf-8")
        os.replace(_cfg_tmp, project_dir / "run_config.json")
    except Exception:  # noqa: BLE001 — the snapshot must never block a run
        logger.exception("run_pipeline_rlm: could not write run_config.json")

    # Local sandboxes have no /workspace volume — repoint the dataset root at a
    # writable shared cache BEFORE any primitive (implement_baseline / run_experiment)
    # reads it, so dataset/env setup does not die at os.makedirs. See the helper.
    _ensure_local_data_root(sandbox_mode, runs_root)

    # 1. Observability + budget.
    cost_ledger = RunCostLedger.load_jsonl(
        project_dir / "cost_ledger.jsonl", project_id=project_id, attach_path=True
    )
    dashboard = DashboardEmitter(project_id, runs_root)
    emit = make_emit(dashboard)
    # wall_clock_s is intentionally Optional[float] — None means unbounded
    # (no watchdog, no rlm max_timeout, no ctx deadline). The user mandates
    # the operator must opt-in to a ceiling via --max-wall-clock or
    # OPENRESEARCH_MAX_WALL_CLOCK_S; otherwise long-running paper reproductions
    # are not artificially truncated. See _DEFAULT_WALL_CLOCK_S.
    wall_clock_s: float | None = _DEFAULT_WALL_CLOCK_S
    if run_budget is not None and run_budget.max_wall_clock_seconds:
        wall_clock_s = float(run_budget.max_wall_clock_seconds)

    # 2. Root model (resolved before the primitive LLM client so the client
    #    can mirror a custom endpoint when the root uses one).
    # The unified surface's planner pick drives the ACTUAL root model when
    # --model is unset, so `--models planner=opus` == `--model opus` and the
    # planner stamp matches what ran (no split between --model and --models).
    # Explicit --model wins. Resolution still goes through resolve_root_model, so
    # the root alias collapse (opus→claude-oauth) applies uniformly to both.
    if not model:
        from backend.agents.rlm.role_models import planner_token_from_surface
        _planner_tok = planner_token_from_surface(
            os.environ.get("OPENRESEARCH_ROLE_MODELS", "").strip() or None
        )
        if _planner_tok:
            model = _planner_tok
    root_model = resolve_root_model(model)
    if not root_model.paper_validated:
        logger.warning(
            "run_pipeline_rlm: root model %r is NOT paper-validated as an RLM root "
            "(root_model_unvalidated) — results may not match paper expectations",
            root_model.key,
        )
    # Root-validation gate stamp (oauth-root-reliability plan, P2): record the
    # validated/risk verdict into demo_status.json so an operator can predict
    # the degenerate-loop failure for this run. Fail-soft — a stamp failure
    # must never abort the run; the merge forward-fills onto the running file.
    try:
        _root_validation = classify_root_model(root_model)
        _write_demo_status(
            project_dir,
            "running",
            root_model_validated=_root_validation.validated,
            root_model_risk=_root_validation.risk,
        )
    except Exception:  # noqa: BLE001 — stamping must never block a run
        logger.debug("run_pipeline_rlm: root-validation stamp skipped", exc_info=True)

    # 3. Primitive LLM client (see _build_llm_client on the usage caveat).
    llm_client, llm_model = _build_llm_client(provider, root_model)

    # 3a. Accelerator override — route cheap calls to a fast endpoint when
    # OPENRESEARCH_ACCELERATOR is set to anything other than "off".
    import os as _os
    _accel_mode = (_os.environ.get("OPENRESEARCH_ACCELERATOR") or "off").strip().lower()
    _accel_ep = None
    if _accel_mode != "off":
        try:
            from backend.agents.rlm.accelerator import resolve_accelerator, build_accelerator_client
            _accel_ep = resolve_accelerator(_accel_mode, sandbox_mode=sandbox_mode, settings=get_settings())
        except Exception as _exc:  # explicit-provider failure (AcceleratorError) etc.
            logger.warning(
                "accelerator %r could not be resolved (%s); using default Sonnet/OAuth for cheap calls",
                _accel_mode, _exc,
            )
            _accel_ep = None
        if _accel_ep is not None:
            # OPENRESEARCH_ACCELERATOR_SCOPE — which call tiers the accelerator serves:
            #   "navigation" (default): only the rlms rlm_query/llm_query context-navigation
            #     calls (high-volume, low-judgment) route to the accelerator (see
            #     other_backends below). The quality-critical GRADER + improvement calls
            #     (ctx.llm_client → verify_against_rubric / propose_improvements) stay on the
            #     strong root model (e.g. Sonnet), so a small accelerator never decides the
            #     rubric score — it only speeds up paper navigation.
            #   "all": also route ctx.llm_client to the accelerator (max offload). Only
            #     sensible when the accelerator is itself strong (e.g. a 32B), else grading
            #     quality drops.
            _accel_scope = (_os.environ.get("OPENRESEARCH_ACCELERATOR_SCOPE") or "navigation").strip().lower()
            if _accelerator_grader_offloaded(_accel_scope):
                llm_client = build_accelerator_client(_accel_ep)   # grader + nav both on accel
                llm_model = _accel_ep.model
            logger.info(
                "accelerator: %s (%s, model=%s); scope=%s — navigation routes to the "
                "accelerator; grader/improvements use %s",
                _accel_ep.base_url, _accel_ep.kind, _accel_ep.model, _accel_scope, llm_model,
            )
        elif _accel_mode != "auto":
            logger.warning(
                "accelerator=%r requested but unavailable; cheap calls fall back to Sonnet/OAuth",
                _accel_mode,
            )

    bk = root_model.backend_kwargs
    if root_model.rlm_backend == "openai" and bk.get("base_url"):
        import urllib.parse
        provider_label = urllib.parse.urlparse(bk["base_url"]).hostname or root_model.key
    else:
        provider_label = (provider or "").lower() or (
            "openai" if os.environ.get("OPENAI_API_KEY") else "anthropic"
        )

    # 3b. Per-role model selection (2026-06-17, dynamic Sonnet/Opus ⇄ gpt-4/gpt-5):
    #     resolve the unified surface (OPENRESEARCH_ROLE_MODELS / --models) + legacy
    #     per-role feeders into one RoleSelection. Unset → every sub-role inherits
    #     today's behaviour, so this is byte-identical when unused.
    from backend.agents.rlm.role_models import resolve_role_models, RoleModelError
    try:
        role_selection = resolve_role_models(
            planner_token=root_model.key,
            role_models_json=os.environ.get("OPENRESEARCH_ROLE_MODELS", "").strip() or None,
            grader_backend_env=os.environ.get("OPENRESEARCH_GRADER_BACKEND", "").strip() or None,
            grader_model_env=os.environ.get("OPENRESEARCH_GRADER_MODEL", "").strip() or None,
            verifier_model_setting=(
                getattr(get_settings(), "rubric_verifier_model", "")
                or os.environ.get("OPENRESEARCH_RUBRIC_VERIFIER_MODEL", "")
            ).strip() or None,
            validator_model_setting=os.environ.get("OPENRESEARCH_VALIDATOR_MODEL", "").strip() or None,
        )
    except RoleModelError as _exc:
        raise RuntimeError(f"invalid per-role model selection: {_exc}") from _exc

    # Verifier transport: an overridden verifier role gets a dedicated
    # sampler-capable client (reusing the grader transport); else None → the
    # rubric judge inherits the planner client (today's behaviour). A Claude pick
    # auto-resolves OAuth-vs-API-key by availability (api key ⇄ OAuth seamlessly).
    from backend.agents.rlm.grader_transport import (
        build_transport_client,
        resolve_anthropic_subrole_backend,
    )

    def _subrole_backend(spec: Any) -> str:
        # Claude tokens (opus/sonnet/haiku) → whichever Anthropic auth is present
        # (honours llm_auth_strategy); openai/azure are key-only, pass through.
        return resolve_anthropic_subrole_backend() if spec.is_claude else spec.provider

    verifier_client = None
    if role_selection.verifier is not None:
        verifier_client, _verifier_label = build_transport_client(
            backend=_subrole_backend(role_selection.verifier),
            model=role_selection.verifier.model,
            fallback_client=llm_client,
            fallback_label=provider_label,
            role_label="verifier",
        )
        logger.info("run_pipeline_rlm: verifier transport=%s", _verifier_label)

    # Validator unified surface → feed OPENRESEARCH_VALIDATOR_BACKEND/_MODEL that
    # build_validator_client + _validator_separation_tier read (only when the operator
    # did not set VALIDATOR_BACKEND directly). Without this bridge, `--models
    # validator=gpt-4o-azure` resolves a RoleSelection.validator but silently builds
    # NO validator client — this makes the unified role surface actually wire one.
    if role_selection.validator is not None and not os.environ.get("OPENRESEARCH_VALIDATOR_BACKEND", "").strip():
        os.environ["OPENRESEARCH_VALIDATOR_BACKEND"] = _subrole_backend(role_selection.validator)
        if role_selection.validator.model:
            os.environ.setdefault("OPENRESEARCH_VALIDATOR_MODEL", role_selection.validator.model)

    # Validator transport (P2.3 — fail-closed): when OPENRESEARCH_VALIDATOR_BACKEND
    # is set, build an independent adversarial-panel client.  build_validator_client
    # raises ValueError when the requested backend cannot be constructed (missing
    # credential / unknown backend) — we convert to RuntimeError (fail-closed).
    # When OPENRESEARCH_VALIDATOR_BACKEND is unset, build_validator_client returns the
    # fallback unchanged; in that case the separation tier is "unavailable" and we
    # leave validator_client=None to signal "no independent validator".
    validator_client = None
    _validator_label = provider_label
    try:
        from backend.agents.rlm.grader_transport import build_validator_client
        _vc, _vlabel = build_validator_client(
            fallback_client=llm_client,
            fallback_label=provider_label,
        )
        if _vc is not llm_client:
            validator_client = _vc
            _validator_label = _vlabel
            logger.info("run_pipeline_rlm: validator transport=%s", _validator_label)
    except ValueError as _exc:
        raise RuntimeError(f"validator setup failed (fail-closed): {_exc}") from _exc

    # Separation tier (emitted as a run_warning when degraded/weak).
    _validator_tier = _validator_separation_tier(role_selection)
    if _validator_tier == "degraded":
        emit(build_run_warning_event(
            level="warn",
            code="validator_separation_degraded",
            message=(
                "External validator and executor share the same model/deployment — "
                "the LLM-suspicion portion is not independently grounded (separation=degraded). "
                "The harness-side machine-check veto still stands."
            ),
        ))
    elif _validator_tier == "weak":
        emit(build_run_warning_event(
            level="info",
            code="validator_separation_weak",
            message=(
                "External validator and executor share the same model family but use "
                "different deployments/models (separation=weak). "
                "Same-provider weak separation is supported; the operator requested it explicitly."
            ),
        ))

    # Grader unified surface → feed the existing OPENRESEARCH_GRADER_* path that
    # leaf_scorer.build_grader_client reads (only when the operator did not set
    # GRADER_BACKEND directly — then resolve_role_models already derived from it).
    # A Claude grader pick auto-resolves OAuth-vs-key the same way the verifier does.
    if role_selection.grader is not None and not os.environ.get("OPENRESEARCH_GRADER_BACKEND", "").strip():
        os.environ["OPENRESEARCH_GRADER_BACKEND"] = _subrole_backend(role_selection.grader)
        if role_selection.grader.model:
            os.environ.setdefault("OPENRESEARCH_GRADER_MODEL", role_selection.grader.model)

    # 4. RunContext. The sub-agent runtime + model are resolved here so
    #    implement_baseline never falls through to a dead env-default key,
    #    and runs Sonnet rather than the registry's Opus default.
    agent_runtime, agent_model, runtime_label = _resolve_agent_runtime(runtime, provider, role_selection)
    logger.info("run_pipeline_rlm: sub-agent runtime=%s", runtime_label)
    # Per-run VRAM override from --vram-gb CLI flag (set as env var by cli.py
    # before Settings construction; consumed here so RunContext carries it and
    # resolve_gpu_requirements can bypass the LLM VRAM estimate).
    _vram_override_env = os.environ.get("OPENRESEARCH_VRAM_OVERRIDE_GB")
    _vram_override: int | None = int(_vram_override_env) if _vram_override_env else None

    # Per-run ScopeSpec from OPENRESEARCH_SCOPE_SPEC_JSON (set by cli.cmd_reproduce
    # from --scope-spec + --paper-hint merge). Empty/unset → None (no constraint).
    _scope_json = os.environ.get("OPENRESEARCH_SCOPE_SPEC_JSON", "").strip()
    if _scope_json:
        from backend.agents.schemas import ScopeSpec as _ScopeSpec
        _scope_spec = _ScopeSpec.model_validate_json(_scope_json)
    else:
        _scope_spec = None

    # Recover the arXiv ID from on-disk artifacts so docs/papers/<id>.yaml
    # overrides fire even when project_id is a hashed `prj_<digest>` string.
    # Falls back to the regex over project_id for legacy non-hashed IDs.
    # See _extract_arxiv_id_from_project_dir for resolution order.
    from backend.agents.baseline_implementation import _extract_arxiv_id as _regex_extract
    _arxiv_id: str | None = (
        _extract_arxiv_id_from_project_dir(project_dir)
        or _regex_extract(project_id)
    )
    if _arxiv_id:
        logger.info("run_pipeline_rlm[%s]: arxiv_id=%s", project_id, _arxiv_id)

    ctx = RunContext(
        project_id=project_id,
        project_dir=project_dir,
        runs_root=runs_root,
        dashboard=dashboard,
        emit=emit,           # thread-safe emit chokepoint from make_emit above
        cost_ledger=cost_ledger,
        llm_client=llm_client,
        provider=provider_label,
        model=llm_model,
        runtime=agent_runtime,
        agent_model=agent_model,
        role_selection=role_selection,
        verifier_client=verifier_client,
        validator_client=validator_client,
        workspace_service=workspace_service,
        workspace_id=workspace_id,
        sandbox_mode=sandbox_mode,
        # 2026-05-23: thread execution_profile.gpu_mode through so the
        # baseline-implementation agent's _compute_constraint_guidance can
        # decide CPU-vs-GPU baseline strategy dynamically. Without this,
        # ctx.gpu_mode is always None and the helper falls back to its
        # most-conservative branch regardless of what the user actually
        # configured (--gpu-mode max on runpod should NOT trigger smoke-test).
        gpu_mode=(
            execution_profile.gpu_mode
            if execution_profile is not None and hasattr(execution_profile, "gpu_mode")
            else None
        ),
        run_budget=run_budget,
        deadline_utc=(
            datetime.now(timezone.utc) + timedelta(seconds=wall_clock_s)
            if wall_clock_s is not None
            else None
        ),  # M-DEADLINE — None when no wall-clock ceiling was requested.
        vram_override=_vram_override,
        scope_spec=_scope_spec,
        arxiv_id=_arxiv_id,  # P0: thread arXiv ID so implement_baseline can load
                             # docs/papers/<id>.yaml even on hashed project IDs.
        # Lane Q — --minimize-compute / lab UI checkbox. Threaded onto ctx so the
        # implement_baseline primitive can pass it into run_with_sdk.
        minimize_compute=(
            bool(getattr(execution_profile, "minimize_compute", False))
            if execution_profile is not None
            else False
        ),
        # Execution mode (C1): thread ExecutionProfile.mode so
        # resolve_experiment_timeout_s honors --execution-mode max instead of
        # silently capping long papers at the 2h default. mode is an enum
        # (.value = "efficient"/"max", matching EXPERIMENT_TIMEOUT_BY_MODE);
        # tolerate a plain-string mode too.
        execution_mode=(
            getattr(execution_profile.mode, "value", execution_profile.mode)
            if execution_profile is not None
            and getattr(execution_profile, "mode", None) is not None
            else None
        ),
        gpu_device_ids=_parse_gpu_device_ids(),
        gpu_parallelism=_parse_gpu_parallelism(),
        gpu_visible_count=_visible_gpu_count(),
    )

    # Fidelity advisories (never block): warn when a non-Claude model drives a
    # sub-role on a fidelity-critical paper (a paper hint with invariants present).
    for _msg in role_selection.fidelity_warnings(
        fidelity_critical=bool(getattr(ctx, "paper_hint_invariants", None))
    ):
        emit(build_run_warning_event(level="warn", code="role_model_fidelity", message=_msg))

    # 4a. Asset pre-provisioning (2026-06-18). For papers that declare heavy ML
    # assets (pip stack, HF weights, datasets, WebShop), ensure they are warm in
    # the shared cache BEFORE any GPU work. Gated on: local sandbox only, the
    # paper hint carries an AssetSpec, and OPENRESEARCH_PRELOAD_ASSETS != "0".
    # Required assets (requirements_files, models, webshop) abort the run on
    # failure; datasets are best-effort and only logged. Complete no-op for every
    # non-SDAR paper and every non-local sandbox.
    _preload_hint = None
    if (
        getattr(ctx, "arxiv_id", None)
        and getattr(ctx.sandbox_mode, "value", str(ctx.sandbox_mode or "")).lower() == "local"
        and os.environ.get("OPENRESEARCH_PRELOAD_ASSETS", "1") != "0"
    ):
        try:
            from backend.agents.prompts.paper_hints import lookup_paper_hint as _lookup_hint
            _preload_hint = _lookup_hint(ctx.arxiv_id)
        except Exception:  # noqa: BLE001 — hint lookup must never crash a run
            logger.debug("run_pipeline_rlm: paper hint lookup failed", exc_info=True)
            _preload_hint = None

    if _preload_hint is not None and getattr(_preload_hint, "assets", None) is not None:
        _asset_cache_root = runs_root / ".cache"
        from backend.services.runtime.asset_provisioning import (
            AssetProvisionError as _AssetProvisionError,
            ensure_assets as _ensure_assets,
        )
        try:
            _asset_report = _ensure_assets(
                _preload_hint.assets,
                cache_root=_asset_cache_root,
            )
            emit(build_run_warning_event(
                level="info",
                code="asset_preload",
                message=(
                    f"asset pre-provisioning complete: "
                    f"ensured={_asset_report.ensured}, "
                    f"skipped={_asset_report.skipped}, "
                    f"failed={_asset_report.failed}"
                ),
            ))
            logger.info(
                "run_pipeline_rlm[%s]: asset preload — ensured=%s skipped=%s failed=%s",
                project_id,
                _asset_report.ensured,
                _asset_report.skipped,
                _asset_report.failed,
            )
        except _AssetProvisionError as _ape:
            # A missing required asset is fatal: abort before GPU work starts.
            logger.error(
                "run_pipeline_rlm[%s]: asset preload FAILED (fatal): %s",
                project_id,
                _ape,
                exc_info=True,
            )
            emit(build_run_warning_event(
                level="error",
                code="asset_preload_failed",
                message=f"Required asset could not be provisioned: {_ape}",
            ))
            _write_demo_status(project_dir, "failed", error={"error": str(_ape)})
            return RLMRunResult(
                project_id=project_id,
                status="failed",
                iterations=0,
                rubric_score=None,
                cost_usd=None,
                final_report_path=None,
            )
        except Exception:  # noqa: BLE001
            # Unexpected errors are logged but must not abort (conservative fallback).
            logger.warning(
                "run_pipeline_rlm[%s]: asset preload raised unexpectedly (non-fatal)",
                project_id,
                exc_info=True,
            )

    # 4a-bis. Cell-resume auto-arm (2026-06-18, T1). On a spot-restart re-invoke
    # with the same --project-id, skip already-completed cells instead of redoing
    # them. Correctness-safe (resume only skips fingerprint-matched ok cells);
    # an explicit OPENRESEARCH_RESUME_CELLS always wins. No-op on a first run.
    if _maybe_auto_arm_cell_resume(project_dir):
        emit(build_run_warning_event(
            level="info",
            code="resume_auto_armed",
            message=(
                "prior incomplete attempt detected for this project_id — auto-armed "
                "OPENRESEARCH_RESUME_CELLS=1 so fingerprint-matched completed cells are "
                "skipped on this resume"
            ),
        ))
        logger.info(
            "run_pipeline_rlm[%s]: cell-resume auto-armed (spot-restart resume)",
            project_id,
        )

    # 4b. Full-scope environment provisioning (2026-06-01). When the scope names
    # heavy RL envs (ALFWorld / WebShop) or Search-QA, stand them up ONCE in the
    # host-shared cache and splice their locations (ALFWORLD_DATA / WEBSHOP_URL /
    # SEARCH_QA_INDEX_DIR / SEARCH_QA_RETRIEVER) into os.environ so every cell
    # subprocess inherits them. Fail-soft: an ALFWorld/WebShop that cannot be stood
    # up becomes a VERIFIED exclusion on ctx (folded into metrics.scope → excluded,
    # not zeroed). A no-op for non-SDAR papers (setup() ignores unknown dataset
    # names) and for Search-QA when dense is off (it just runs BM25).
    _provision_envs = (
        [d.normalized_id() for d in (getattr(_scope_spec, "datasets", None) or [])]
        if _scope_spec is not None else []
    )
    if _provision_envs:
        try:
            import atexit as _atexit
            from backend.services.runtime.env_cache import (
                EnvCacheManager as _EnvCacheManager,
                provision_scope as _provision_scope,
            )
            _prov = _provision_scope(_provision_envs, _EnvCacheManager())
            if _prov.env_vars:
                os.environ.update(_prov.env_vars)
            ctx.env_setup_exclusions = list(_prov.exclusions)
            _atexit.register(_prov.release)
            logger.info(
                "run_pipeline_rlm[%s]: env provisioning — vars=%s, exclusions=%s",
                project_id, sorted(_prov.env_vars), [e.item for e in _prov.exclusions],
            )
        except Exception:  # noqa: BLE001 — provisioning must never abort the run
            logger.warning(
                "run_pipeline_rlm: env provisioning failed (non-fatal)", exc_info=True
            )

    # 5. Primitives — the real binding or the stub provider.
    # repair_policy_holder is a late-binding 1-slot list: the tool wrappers
    # close over it, and run.py populates slot 0 after the ForcedIterationPolicy
    # is constructed below.  This lets _record_last_primitive_result_tools notify
    # the policy of repairable run_experiment outcomes without circular deps.
    repair_policy_holder: list = []
    custom_tools, tools_label = _resolve_custom_tools(ctx)
    custom_tools = _record_last_primitive_result_tools(custom_tools, ctx, repair_policy_holder)
    logger.info(
        "run_pipeline_rlm: project=%s root=%s primitives=%s",
        project_id,
        root_model.key,
        tools_label,
    )

    # Hybrid-repair-only mode (set by backend.agents.hybrid.controller when
    # run_pipeline_rlm is called as Phase 2 of a hybrid run).
    # The RDR Phase 1 already produced a code_dir; we skip full reproduction
    # and focus the root model on repairing the weak clusters identified by
    # Phase 1 scoring.
    # Explicit kwargs are preferred; legacy claim_map keys remain for
    # back-compat with any external caller still using the old contract.
    _hybrid_repair_only: bool = bool(
        hybrid_repair_only or workspace_claim_map.get("_hybrid_repair_only", False)
    )
    _phase1_weak_clusters: list[Any] = (
        phase1_weak_clusters
        if phase1_weak_clusters is not None
        else (workspace_claim_map.get("_phase1_weak_clusters") or [])
    )

    # 6. The offloaded corpus + 7. the system prompt.
    context_dict = _build_context(workspace_claim_map)

    # arXiv runs arrive with no rubric_spec — derive a PaperBench-shaped rubric
    # from the paper so the run is scorable (bundle runs already carry one).
    # Stub-primitive runs skip GENERATION only: rubric generation is a REAL paid
    # LLM call (the one non-stubbed network path) — the 862s-suite-stall fix
    # (audit 2026-06-09). OPENRESEARCH_REUSE_RUBRIC (LLM-free, see
    # _load_reusable_rubric) stays active in both modes so A/B arms grade
    # against the SAME pre-seeded rubric.
    _stub_mode = os.environ.get("OPENRESEARCH_RLM_STUB_PRIMITIVES") == "1"
    if not context_dict.get("rubric_spec") and context_dict.get("paper_text"):
        _reused_rubric = _load_reusable_rubric(project_dir)
        if _reused_rubric is not None:
            context_dict["rubric_spec"] = _reused_rubric
            logger.info(
                "run_pipeline_rlm: OPENRESEARCH_REUSE_RUBRIC — reusing on-disk "
                "generated_rubric.json (no regeneration)"
            )
    if _stub_mode and not context_dict.get("rubric_spec") and context_dict.get("paper_text"):
        logger.info("run_pipeline_rlm: stub mode — skipping LLM rubric generation (run proceeds rubric-less)")
    if not _stub_mode and not context_dict.get("rubric_spec") and context_dict.get("paper_text"):
        from backend.agents.rlm.rubric_gen import generate_rubric_tree

        generated = generate_rubric_tree(
            context_dict["paper_text"],
            llm_client,
            paper_title=context_dict.get("paper_metadata", {}).get("title", ""),
        )
        if generated is not None:
            context_dict["rubric_spec"] = generated
            (project_dir / "generated_rubric.json").write_text(
                json.dumps(generated, indent=2), encoding="utf-8"
            )
            logger.info("run_pipeline_rlm: using a self-generated rubric (persisted to generated_rubric.json)")
        else:
            logger.warning("run_pipeline_rlm: rubric generation failed — run proceeds rubric-less")

    # Hybrid Phase 2: seed context with Phase 1 code path + weak cluster list
    # so the root model repairs rather than reproduces from scratch.
    active_prompt = _ROOT_PROMPT
    if _hybrid_repair_only:
        code_dir = project_dir / "code"
        context_dict["_hybrid_repair_only"] = True
        context_dict["_phase1_code_dir"] = str(code_dir)
        context_dict["_phase1_weak_clusters"] = _phase1_weak_clusters
        logger.info(
            "run_pipeline_rlm[%s]: hybrid-repair-only mode — "
            "%d weak cluster(s); code_dir=%s",
            project_id, len(_phase1_weak_clusters), code_dir,
        )
        active_prompt = (
            "You are the repair agent for a hybrid RDR+RLM reproduction run. "
            "Phase 1 (RDR) already produced a code directory at "
            "`context['_phase1_code_dir']`. "
            "The weak clusters that need repair are listed in "
            "`context['_phase1_weak_clusters']` — each entry has "
            "{'id', 'score', 'justification'}. "
            "Your goal is to improve the reproduction for those specific clusters. "
            "Inspect the existing code, understand what each weak cluster requires "
            "(see the rubric in context['rubric_spec']), implement targeted fixes, "
            "re-run the experiment with run_experiment, and re-score with "
            "verify_against_rubric. Do NOT rewrite passing clusters. "
            "When finished, call FINAL_VAR on the updated report dict — "
            "exactly as the system prompt's termination contract describes."
        )

    # M-REDACT: build sentinels once; threaded into logger + _finalize (A1-M2, A1-C1).
    corpus_sentinels = _corpus_sentinels(context_dict)
    system_prompt = build_system_prompt(
        context_metadata=_context_metadata(context_dict),
        root_model=root_model,
    )

    # 8. Checkpoint + the streaming logger.
    event_store = SqliteEventStore(get_settings().database_url)
    checkpointer = IterationCheckpointer(
        project_id=project_id,
        event_store=event_store,
        snapshot_dir=project_dir / "rlm_state",
    )
    snapshot_writer = ReplSnapshotWriter(
        project_dir=project_dir,
        sentinels=corpus_sentinels,
    )
    rlm_logger = _FatalBackendGateLogger(
        emit=emit,
        checkpointer=checkpointer,
        sentinels=corpus_sentinels,
        snapshot_writer=snapshot_writer,
        ctx=ctx,
    )

    # Resolve cost cap — passed directly to RLM(max_budget=...) so the library
    # itself raises BudgetExceededError between iterations (T2/M-BUDGET).
    max_usd: float | None = None
    if run_budget is not None and run_budget.max_usd:
        max_usd = float(run_budget.max_usd)

    # Featherless's plan caps input context far below the model's native window.
    # Register that cap with rlm so compaction fires before the provider 400s
    # the run on context_length_exceeded, and tighten the compaction threshold
    # for extra margin against token-count drift on a non-tiktoken model.
    is_featherless = root_model.rlm_backend == "openai" and "featherless" in str(
        bk.get("base_url", "")
    )
    if is_featherless:
        register_featherless_context_limits()
    compaction_threshold_pct = 0.7 if is_featherless else 0.85

    # 9. Construct the RLM engine.
    # Accelerator sub-backend override: when an accelerator endpoint is active,
    # redirect rlm_query/llm_query navigation to the same fast endpoint so
    # context-navigation calls also benefit from the accelerator.
    _other_backends = [root_model.sub_backend]
    _other_backend_kwargs = [root_model.sub_backend_kwargs]
    if _accel_ep is not None and not _accel_ep.is_azure:
        _other_backends = ["openai"]
        _other_backend_kwargs = [
            {
                "model_name": _accel_ep.model,
                "base_url": _accel_ep.base_url,
                "api_key": _accel_ep.api_key,
            }
        ]
    elif _accel_ep is not None and _accel_ep.is_azure:
        from backend.services.context.workspace.tools.azure_openai_client import (
            DEFAULT_AZURE_OPENAI_API_VERSION,
        )
        _other_backends = ["azure_openai"]
        _other_backend_kwargs = [
            {
                "model_name": _accel_ep.model,
                "azure_endpoint": _accel_ep.base_url,
                "azure_deployment": _accel_ep.model,
                "api_key": _accel_ep.api_key,
                "api_version": (
                    os.environ.get("AZURE_OPENAI_API_VERSION")
                    or DEFAULT_AZURE_OPENAI_API_VERSION
                ),
            }
        ]

    rlm = RLM(
        backend=root_model.rlm_backend,
        backend_kwargs=root_model.backend_kwargs,
        environment="local",                       # mandatory — DockerREPL drops custom_tools
        max_depth=_MAX_DEPTH,
        max_iterations=_MAX_ITERATIONS,
        max_timeout=wall_clock_s,
        max_budget=max_usd,                        # T2/M-BUDGET: enforced by rlm between iterations
        compaction=True,
        compaction_threshold_pct=compaction_threshold_pct,
        other_backends=_other_backends,
        other_backend_kwargs=_other_backend_kwargs,
        custom_tools=custom_tools,
        custom_sub_tools={},                       # sub-calls navigate text, not primitives
        custom_system_prompt=system_prompt,
        logger=rlm_logger,
        on_subcall_start=make_on_subcall_start(emit),
        on_subcall_complete=make_on_subcall_complete(emit),
    )

    # 10. Arm the wall-clock backstop, then 11. run .completion() on a worker thread.
    watchdog = _arm_watchdog(
        wall_clock_s,
        project_dir=project_dir,
        emit=emit,
        iteration_count=lambda: rlm_logger.iteration_count,
        ctx=ctx,
    )
    # Ship a partial report on a graceful SIGTERM kill too (not just on a hang).
    _prev_sigterm = _install_sigterm_finalizer(
        project_dir=project_dir,
        emit=emit,
        iteration_count=lambda: rlm_logger.iteration_count,
        ctx=ctx,
    )

    # 10.5. Lane H — wire the forced-iteration policy so FINAL_VAR is refused
    # while the latest rubric score is below target AND the iteration floor
    # has not been hit. The interceptor was installed at module load via
    # apply_forced_iteration_patch(); here we push a per-run policy onto the
    # thread-local stack so the patched _final_var consults this run's state.
    settings = get_settings()
    min_iterations = int(getattr(settings, "min_rubric_iterations", 2))
    # PR-ι.1: per-run iteration budget from env var (CLI sets this before calling us).
    _raw_max_iter = os.environ.get("OPENRESEARCH_MAX_RLM_ITERATIONS", "").strip()
    _max_rlm_iterations: int | None = int(_raw_max_iter) if _raw_max_iter.isdigit() and int(_raw_max_iter) > 0 else None

    def _emit_forced_iteration_warning(message: str) -> None:
        try:
            emit(build_run_warning_event(
                level="warn",
                code="forced_iteration",
                message=message,
            ))
        except Exception:  # noqa: BLE001 — emit must never block the policy
            logger.exception("run_pipeline_rlm: forced-iteration warning emit failed")

    def _emit_iteration_budget_exceeded(message: str) -> None:
        try:
            emit(build_run_warning_event(
                level="warn",
                code="iteration_budget_exceeded",
                message=message,
            ))
        except Exception:  # noqa: BLE001 — emit must never block the policy
            logger.exception("run_pipeline_rlm: iteration-budget-exceeded warning emit failed")

    def _emit_forced_repair_warning(message: str) -> None:
        try:
            emit(build_run_warning_event(
                level="warn",
                code="forced_repair_iteration",
                message=message,
            ))
        except Exception:  # noqa: BLE001 — emit must never block the policy
            logger.exception("run_pipeline_rlm: forced-repair-iteration warning emit failed")

    def _count_honest_candidate_outcomes() -> int:
        """Count candidate_outcome events with truthful outcomes.

        Lane O — "honest" means the agent actually ran the candidate's
        experiment and reported the result. Declined / skipped don't count.
        Reads dashboard_events.jsonl since the events flow through that
        single egress chokepoint regardless of which primitive recorded them.
        """
        ev_path = ctx.project_dir / "dashboard_events.jsonl"
        if not ev_path.exists():
            return 0
        honest = {"promoted", "failed", "marginal"}
        count = 0
        try:
            with ev_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or '"candidate_outcome"' not in line:
                        continue
                    try:
                        import json as _json
                        e = _json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if e.get("event") != "candidate_outcome":
                        continue
                    if str(e.get("outcome") or "").lower() in honest:
                        count += 1
        except OSError:
            return 0
        return count

    def _code_path_exists() -> bool:
        """Whether a usable baseline implementation exists on disk.

        Minimal predicate (the degenerate case has NO code so this returns
        False then): a non-empty ``code/commands.json`` JSON list AND ≥1
        runnable source file under ``code/``.  Inlined rather than importing
        the heavier ``primitives._harvest_baseline_artifacts`` to keep this
        closure import-light and side-effect-free.
        """
        import json as _json

        code_dir = ctx.project_dir / "code"
        commands_path = code_dir / "commands.json"
        try:
            commands = _json.loads(commands_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(commands, list) or not commands:
            return False
        runnable_suffixes = {".py", ".sh", ".bash", ".ps1"}
        for file in code_dir.rglob("*"):
            if not file.is_file() or file.name == "commands.json":
                continue
            if file.suffix.lower() in runnable_suffixes or file.name in {
                "Dockerfile",
                "Makefile",
            }:
                return True
        return False

    def _env_built() -> bool:
        """Whether the environment build has succeeded for this run.

        For ``docker``/``auto`` an explicit ``build_environment`` ok-row is
        required; for ``local``/``runpod``/``azure``/``gcp`` env-build is a
        no-op success → treat as built.
        """
        mode = getattr(ctx.sandbox_mode, "value", str(ctx.sandbox_mode or "")).lower()
        if mode in ("docker", "auto"):
            try:
                return ctx.cost_ledger.session_call_count("build_environment") > 0
            except Exception:  # noqa: BLE001
                return False
        return True

    def _required_stage() -> str:
        """Infer the next mandatory lifecycle stage (fail-soft).

        Any error → ``"need_baseline"`` (the safe default — the degenerate
        case has no code).
        """
        try:
            return infer_required_stage(
                primitives=[],
                code_path_exists=_code_path_exists(),
                env_built=_env_built(),
                total_run_experiments=run_experiment_call_count(ctx),
                total_verifications=1 if ctx.latest_rubric_score is not None else 0,
            )
        except Exception:  # noqa: BLE001 — never crash the policy
            return "need_baseline"

    oauth_root = root_model.rlm_backend == "anthropic-oauth"

    iteration_policy = ForcedIterationPolicy(
        min_iterations=min_iterations,
        rubric_snapshot=lambda: (
            ctx.latest_rubric_score,
            ctx.latest_rubric_target,
            ctx.latest_rubric_iteration,
        ),
        current_iteration=lambda: rlm_logger.iteration_count,
        remaining_s=lambda: ctx.remaining_s(),
        on_refusal=_emit_forced_iteration_warning,
        honest_candidate_outcomes=_count_honest_candidate_outcomes,
        on_repair_refusal=_emit_forced_repair_warning,
        max_rlm_iterations=_max_rlm_iterations,
        on_budget_exceeded=_emit_iteration_budget_exceeded,
        required_stage=_required_stage,
        oauth_root=oauth_root,
    )
    # Task 4 — register the degenerate-refusal-loop callback. Built after the
    # policy exists so it can close over it (note_terminal_failure). Default
    # (autodrive OFF): emit the warning + mark a terminal stop so the run
    # finalizes fast instead of churning to the 16-refusal cap.
    iteration_policy.on_degenerate_refusal_loop = _make_degenerate_loop_callback(
        emit=emit,
        ctx=ctx,
        policy=iteration_policy,
        autodrive_enabled=_autodrive_enabled(),
        # `custom_tools` here is the WRAPPED dict (binding tools wrapped by
        # `_record_last_primitive_result_tools`) — so the backstop dispatches the
        # same wrapped primitives the root uses (ctx re-supplied by the wrapper).
        tools=custom_tools,
        oauth_root=oauth_root,
    )
    # P3 fix-first repair loop (2026-06-20): wire the two new hooks that engage
    # the evidence-fingerprint-based repair loop and the external validator gate.
    # Both are DEFAULT-OFF: when neither OPENRESEARCH_ZERO_METRICS_GUARD nor
    # OPENRESEARCH_EXTERNAL_VALIDATOR is set, _fixfirst_loop_engaged() is False
    # and neither hook is assigned — the policy is byte-identical to today.
    # B1: evidence_fingerprint — engage the loop when either guard flag is on.
    if _fixfirst_loop_engaged():
        def _current_evidence_fingerprint() -> str:
            """Hash the measured on-disk metrics. Changes on real progress; fail-soft -> ''."""
            try:
                from backend.agents.rlm.external_validator import evidence_fingerprint as _efp  # noqa: PLC0415
                _mp = project_dir / "code" / "metrics.json"
                _m = json.loads(_mp.read_text(encoding="utf-8")) if _mp.exists() else {}
                return _efp(_m)
            except Exception:  # noqa: BLE001 — never crash the policy
                return ""
        iteration_policy.evidence_fingerprint = _current_evidence_fingerprint

    # B2: validator_gate — only when the external validator flag is on AND a
    # validator client was built.  The closure caches the panel result by evidence
    # fingerprint so the LLM panel runs AT MOST ONCE per distinct evidence state
    # (cost guard — should_refuse may call the gate on every FINAL_VAR attempt).
    from backend.agents.rlm.external_validator import external_validator_enabled as _ext_validator_enabled  # noqa: PLC0415
    if _ext_validator_enabled() and getattr(ctx, "validator_client", None) is not None:
        from backend.agents.rlm.external_validator import (  # noqa: PLC0415
            run_validation_panel as _run_validation_panel,
            persist_verdict as _persist_verdict,
        )
        _panel_cache: dict[str, tuple[bool, str]] = {}
        _val_tier = _validator_separation_tier(role_selection)
        _val_label = _validator_label  # captured from enclosing scope (built at line ~2312)

        def _validator_gate() -> tuple[bool, str] | None:
            """Run the adversarial panel, cached by evidence fingerprint (one run per state).

            Returns (vetoed, directive) on a machine-verified veto, else None.
            Fail-soft: any error returns None (panel never crashes the policy).
            """
            try:
                from backend.agents.rlm.external_validator import evidence_fingerprint as _efp  # noqa: PLC0415
                _mp = project_dir / "code" / "metrics.json"
                _metrics = json.loads(_mp.read_text(encoding="utf-8")) if _mp.exists() else {}
                _fp = _efp(_metrics)
                if _fp in _panel_cache:  # cost guard — one panel run per evidence state
                    return _panel_cache[_fp]
                # Gather leaf records from rubric_evaluation.json (best-effort).
                _leaf_records: list[dict] = []
                try:
                    _re_path = project_dir / "rubric_evaluation.json"
                    if _re_path.exists():
                        _re_data = json.loads(_re_path.read_text(encoding="utf-8"))
                        _leaf_records = _re_data.get("leaf_scores") or _re_data.get("leaves") or []
                except Exception:  # noqa: BLE001
                    _leaf_records = []
                _verdict = _run_validation_panel(
                    validator_client=ctx.validator_client,
                    panel_models=[_val_label],
                    metrics=_metrics,
                    project_dir=project_dir,
                    leaf_records=_leaf_records,
                    separation=_val_tier,
                )
                _persist_verdict(project_dir, _verdict)
                _vetoed = _verdict.status == "vetoed"
                _directive = ""
                if _vetoed:
                    _directive = (
                        "FINAL_VAR refused: the external validator vetoed these result "
                        f"claims as unsubstantiated: {', '.join(_verdict.veto_set[:6])}. "
                        "Re-implement so each cited metric traces to real model outputs on "
                        "real data (loss backprops from the model, reward reads env outcomes, "
                        "eval scores against gold), then run_experiment + verify_against_rubric."
                    )
                    # Route the cited directive to leaf_triage so the next implementer
                    # prompt carries the specific repair guidance (best-effort, fail-soft).
                    try:
                        from backend.agents.rlm import leaf_triage as _lt  # noqa: PLC0415
                        if _lt.is_enabled():
                            _plan = [
                                {
                                    "leaf_id": r,
                                    "score": 0.0,
                                    "repair_class": "validator_veto",
                                    "cost": "targeted_rerun",
                                    "directive": _directive,
                                    "justification": "external validator machine-verified veto",
                                }
                                for r in _verdict.veto_set[:6]
                            ]
                            _lt.persist(project_dir, {"plan": _plan, "facts": {}, "summary": "external validator veto"})
                    except Exception:  # noqa: BLE001 — leaf_triage is advisory only
                        pass
                _result: tuple[bool, str] = (_vetoed, _directive)
                _panel_cache[_fp] = _result
                return _result
            except Exception:  # noqa: BLE001 — the validator must never crash the policy
                logger.debug("_validator_gate: failed; treating as clean", exc_info=True)
                return None

        iteration_policy.validator_gate = _validator_gate

    # §4.4 B — claim gate: wire only when the fix-first loop is engaged AND the
    # report-claim gate flag is on. Stringifies the pending FINAL_VAR value
    # (stashed by _intercepted_final_var), extracts result claims, and checks
    # them against on-disk measured values. Returns (vetoed, directive) when
    # ungrounded claims exist, else None. Fail-soft: any exception → None.
    from backend.agents.rlm.report_claim_gate import report_claim_gate_enabled as _rcg_enabled  # noqa: PLC0415
    if _rcg_enabled():
        from backend.agents.rlm.claim_grounding import (  # noqa: PLC0415
            check_claims_grounded as _check_claims_grounded,
            extract_result_claims as _extract_result_claims,
            flatten_measured_values as _flatten_measured_values,
        )

        def _claim_gate() -> tuple[bool, str] | None:
            """Check the pending FINAL_VAR value's claims vs on-disk metrics.

            Returns (True, directive) when ≥1 ungrounded result claim is found
            and measured evidence is present, else None (clean or unverifiable).
            Fail-soft: any error → None.
            """
            try:
                _pending = iteration_policy.pending_final_value
                if not _pending:
                    return None
                _claims = _extract_result_claims(_pending)
                if not _claims:
                    return None
                _measured = _flatten_measured_values(project_dir)
                if not _measured:
                    return None  # no evidence → unverifiable, not ungrounded
                _grounding = _check_claims_grounded(_claims, _measured)
                _ungrounded = _grounding.get("ungrounded", [])
                if not _ungrounded:
                    return None
                # Build a concise cited directive.
                _first = _ungrounded[0]
                _directive = (
                    f"FINAL_VAR refused: your report claims {_first.value} "
                    f"({_first.term}) but code/metrics.json has no matching "
                    "measured value within 5%. Either run_experiment to produce "
                    "that evidence or correct the report, then FINAL_VAR again."
                )
                return (True, _directive)
            except Exception:  # noqa: BLE001 — never crash the policy
                logger.debug("_claim_gate: failed; treating as clean", exc_info=True)
                return None

        iteration_policy.claim_gate = _claim_gate

    # PR-α followup: populate the late-binding holder so the tool wrapper
    # can call policy.record_repair_attempt() when run_experiment returns
    # a repairable outcome.  Slot 0 is set here; wrappers close over the list.
    repair_policy_holder.append(iteration_policy)
    # PR-μ Solution C: expose the policy on ctx so run_experiment can feed
    # per-iteration outcomes via record_run_experiment(); the consumer side
    # at primitives.py:_emit_iteration_boundary_warning is fail-soft when
    # ctx._forced_iteration_policy is None.
    ctx._forced_iteration_policy = iteration_policy

    # PR-ι.3 — rolling cost surfacing.  A background daemon thread updates
    # demo_status.json::cost_summary every 30 s while the RLM loop runs.
    _cost_stop_event = threading.Event()
    _cost_thread = threading.Thread(
        target=_update_cost_summary_loop,
        kwargs={
            "project_dir": project_dir,
            "stop_event": _cost_stop_event,
            "iteration_count": lambda: rlm_logger.iteration_count,
        },
        daemon=True,
        name=f"cost-summary-{project_id}",
    )
    _cost_thread.start()

    result_obj: Any = None
    run_failed = False
    fatal_abort: _FatalPrimitiveAbort | None = None
    try:
        # Boot model-availability preflight (2026-06-14): fail FAST with a clean
        # fatal report if the primitive/grader model is unavailable, instead of
        # wedging on every primitive call for hours. The default_oauth_model pin
        # already prevents the CLI-default deferral that caused the Fable-5 outage;
        # this is the backstop for the rarer case where the PINNED model is itself
        # down. Fail-soft on ambiguous/transient probe errors (a network blip never
        # aborts a good run); abort ONLY on the definitive 'unavailable' block.
        if _os.environ.get("OPENRESEARCH_MODEL_PREFLIGHT", "1").strip() not in ("0", "false", ""):
            from backend.services.context.workspace.tools.rlm_query import (
                preflight_model_available,
            )
            _pf_ok, _pf_detail = preflight_model_available(llm_client)
            if not _pf_ok:
                raise _FatalPrimitiveAbort(
                    primitive_name="model_preflight",
                    result={
                        "error": (
                            f"primitive/grader model {llm_model!r} is unavailable — "
                            f"aborted before GPU work: {_pf_detail}"
                        ),
                        "failure_class": "model_unavailable",
                        "suggested_fix": (
                            "Set OPENRESEARCH_OAUTH_FALLBACK_MODEL to an available Claude "
                            "model id, or OPENRESEARCH_MODEL_PREFLIGHT=0 to bypass."
                        ),
                    },
                )
            logger.info("model preflight: %r available", llm_model)

        def _run_completion_on_worker() -> Any:
            # The forced-iteration policy stack is THREAD-LOCAL (forced_iteration._LOCAL),
            # and the FINAL_VAR interceptor (LocalREPL._final_var) executes on whatever
            # thread runs rlm.completion. asyncio.to_thread dispatches completion to a
            # SEPARATE worker thread, so the policy MUST be pushed on THAT thread — entering
            # the context manager on the asyncio loop thread (as this code did until
            # 2026-05-31) leaves the interceptor's _current_policy() empty on the worker
            # thread, silently disabling the entire premature-exit guard (Lane H /
            # BUG-LR-013): the root could FINAL_VAR after one sub-target iteration and
            # nothing refused it. Enter the policy INSIDE the worker callable.
            with forced_iteration_policy(iteration_policy):
                return rlm.completion(context_dict, active_prompt)

        result_obj = await asyncio.to_thread(_run_completion_on_worker)

        # C3 — Drain the module-level ClaudeOauthClient root-usage sink and
        # ledger cache tokens for the root reasoning turns.
        #
        # Double-count design (documented here, tested in tests/rlm/test_root_usage_ledger.py):
        #
        #   - tokens_total.json is generated exclusively from cost_ledger.jsonl via
        #     _aggregate_tokens_total(); the rlm usage_summary NEVER feeds tokens_total.
        #   - final_report.cost.llm_usd is sourced from result.usage_summary
        #     (in _cost_dict()) — it does NOT read the cost ledger for the root.
        #   - Therefore: adding a rlm_root row to the ledger with the full
        #     input/output/cache tokens does NOT double-count in final_report.cost.llm_usd.
        #     It does add these tokens to tokens_total.json, which is the desired
        #     behaviour — the root's tokens were previously absent from tokens_total.
        #   - For OAuth runs estimated_usd stays $0 (correct: real cost is $0).
        #     Use equivalent_cost_usd() from pricing.py for the hypothetical API cost.
        if root_model.rlm_backend == "anthropic-oauth":
            try:
                from backend.agents.rlm.claude_oauth_client import drain_root_usage
                from backend.agents.resilience.cost import CostLedgerEntry

                root_usage_by_model = drain_root_usage()
                for _model, _u in root_usage_by_model.items():
                    if _u.get("calls", 0) == 0:
                        continue
                    _entry = CostLedgerEntry.from_usage(
                        agent_id="rlm_root",
                        attempt_index=0,
                        provider="anthropic",  # type: ignore[arg-type]
                        model=_model,
                        usage={
                            "input_tokens": _u.get("input_tokens", 0),
                            "output_tokens": _u.get("output_tokens", 0),
                            "cache_creation_input_tokens": _u.get("cache_creation_input_tokens", 0),
                            "cache_read_input_tokens": _u.get("cache_read_input_tokens", 0),
                            "reasoning_tokens": 0,
                        },
                    )
                    if ctx.cost_ledger is not None:
                        ctx.cost_ledger.append(_entry)
                    # Also write directly to the ledger file (mirrors
                    # record_subagent_usage_to_path for resilience when
                    # ctx.cost_ledger has no path attached).
                    _ledger_path = project_dir / "cost_ledger.jsonl"
                    if ctx.cost_ledger is None or ctx.cost_ledger.path is None:
                        import json as _json_ledger
                        _ledger_path.parent.mkdir(parents=True, exist_ok=True)
                        with _ledger_path.open("a", encoding="utf-8") as _fh:
                            _fh.write(_json_ledger.dumps(_entry.to_json(), sort_keys=True) + "\n")
                    else:
                        ctx.cost_ledger.flush()
            except Exception:  # noqa: BLE001 — ledgering is best-effort
                logger.warning("run_pipeline_rlm: drain_root_usage ledger failed", exc_info=True)

    except _FatalPrimitiveAbort as exc:
        fatal_abort = exc
        run_failed = True
        logger.error(
            "run_pipeline_rlm: fatal primitive outcome from %s: %s",
            exc.primitive_name,
            exc.result.get("error") or exc.result,
        )
    except Exception as exc:  # noqa: BLE001 — an honest failure is data, not a crash
        run_failed = True
        logger.exception("run_pipeline_rlm: rlm.completion failed: %s", exc)
    finally:
        # Watchdog is None only when the hard-ceiling backstop is disabled (=0).
        if watchdog is not None:
            watchdog.cancel()
        # Restore the prior SIGTERM handler so we don't leak our finalizer into a
        # long-lived host process (CLI exits anyway; matters for server/tests).
        if _prev_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, _prev_sigterm)
            except (ValueError, OSError):
                pass
        # Stop the cost-summary background thread.
        _cost_stop_event.set()
        _cost_thread.join(timeout=5.0)

    # 12. Build, write, and report (close event_store in finally — A4-9).
    try:
        if fatal_abort is not None:
            return _finalize_fatal_primitive_abort(
                abort=fatal_abort,
                ctx=ctx,
                iterations=rlm_logger.iteration_count,
                project_dir=project_dir,
                emit=emit,
                tools_label=tools_label,
            )
        return _finalize(
            result_obj=result_obj,
            run_failed=run_failed,
            ctx=ctx,
            iterations=rlm_logger.iteration_count,
            project_dir=project_dir,
            emit=emit,
            corpus_sentinels=corpus_sentinels,
            tools_label=tools_label,  # T21 / review I8
            llm_model=llm_model,
        )
    finally:
        try:
            event_store.close()
        except Exception:  # noqa: BLE001
            logger.exception("run_pipeline_rlm: could not close SqliteEventStore")


def _finalize(
    *,
    result_obj: Any,
    run_failed: bool,
    ctx: RunContext,
    iterations: int,
    project_dir: Path,
    emit: Any,
    corpus_sentinels: list[str] | None = None,
    tools_label: str = "real",  # T21 / review I8
    llm_model: str | None = None,
) -> RLMRunResult:
    """Convert the RLM result into a written report + an :class:`RLMRunResult`."""
    if result_obj is not None:
        report = build_final_report(result_obj, ctx=ctx)
    else:
        report = RLMFinalReport(
            verdict="failed",
            reproduction_summary="The RLM run produced no result (see run logs).",
        )

    # The real iteration count is authoritative over the root's self-report.
    report.iterations = iterations
    # A crashed or budget-exhausted run is never a clean completion — force "failed".
    if run_failed:
        report.verdict = "failed"

    # T21 / review I8: stub-run honesty — mark degraded, cap verdict.
    if "stub" in tools_label.lower():
        report.primitive_provider = "stub"
        report.degraded = True
        # A stub run cannot honestly claim "reproduced" — the primitives are
        # deterministic placeholders, not real ML training or evaluation.
        if report.verdict == "reproduced":
            report.verdict = "partial"

    # M-REDACT (A1-C1): scrub corpus content from the report summary before writing
    # to disk or streaming — the root model may copy context["paper_text"] slices
    # verbatim into reproduction_summary.
    if corpus_sentinels and report.reproduction_summary:
        report.reproduction_summary = redact_corpus(
            report.reproduction_summary, corpus_sentinels
        )

    # --- Phase-4-forward-compat metadata (spec 2026-05-23-rubric-climb-leaderboard §4.5)
    # Lifts started_at from demo_status.json (written at run start); stamps
    # completed_at at write time; records per-role models for leaderboard ranking.
    report.mode = "rlm"
    # verifier == the rubric-scoring client (ctx.llm_client → llm_model). Under the
    # default accelerator scope="navigation" this stays the strong root model
    # (Sonnet) even when a small accelerator serves rlm_query/llm_query nav.
    # F5: the GRADER (leaf scorer) rides the same client by default, but A5's
    # decoupled transport (OPENRESEARCH_GRADER_BACKEND/_MODEL) can move it onto an
    # independent sampler-capable model — and ACCELERATOR_SCOPE=all routes it onto
    # the accelerator. Stamp what ACTUALLY graded so the leaderboard is honest;
    # default (no override) → llm_model, byte-for-byte today.
    _grader_stamp = (
        os.environ.get("OPENRESEARCH_GRADER_MODEL", "").strip()
        or os.environ.get("OPENRESEARCH_GRADER_BACKEND", "").strip()
        or (
            os.environ.get("OPENRESEARCH_ACCELERATOR_MODEL", "").strip()
            if (
                os.environ.get("OPENRESEARCH_ACCELERATOR_SCOPE", "").strip().lower() == "all"
                and os.environ.get("OPENRESEARCH_ACCELERATOR", "").strip().lower() == "endpoint"
            )
            else ""
        )
        or llm_model
    )
    # Selection-aware stamping: an overridden executor/verifier reports its
    # resolved provider:model; planner + grader stay byte-identical when no
    # unified surface is used (role_selection is None / those sub-roles inherit).
    _role_selection = getattr(ctx, "role_selection", None)
    _sel_executor = getattr(_role_selection, "executor", None)
    _sel_verifier = getattr(_role_selection, "verifier", None)
    _sel_grader = getattr(_role_selection, "grader", None)
    report.models = {
        "planner": llm_model,
        "executor": (
            _sel_executor.stamp
            if _sel_executor is not None
            else getattr(ctx, "agent_model", None)
        ),
        "verifier": (
            _sel_verifier.stamp
            if _sel_verifier is not None
            else llm_model
        ),
        # Prefer the explicit grader RoleSpec stamp (mirrors executor/verifier) so
        # an explicit foundry grader names its real deployment; fall back to the
        # env/llm_model-derived stamp when the grader inherits.
        "grader": (_sel_grader.stamp if _sel_grader is not None else _grader_stamp),
    }
    started_at: str | None = None
    demo_status_path = project_dir / "demo_status.json"
    if demo_status_path.exists():
        try:
            started_at = json.loads(demo_status_path.read_text()).get("startedAt")
        except (OSError, json.JSONDecodeError):
            started_at = None
    report.started_at = started_at
    report.completed_at = datetime.now(timezone.utc).isoformat()

    # BUG-LR-015: diagnostic heuristic — detect "model gave up before doing work".
    # Fires a run_warning with code="suspicious_partial" when a partial verdict is
    # accompanied by two or more of: (a) no essential primitive executed,
    # (b) very few iterations used, (c) rubric never scored.
    # Suppressed under wall-clock pressure (≤60s remaining) — in that case a
    # truncated partial is expected and correct.
    if report.verdict == "partial" and not run_failed:
        try:
            _remaining = ctx.remaining_s() if ctx is not None else None
            _wall_pressure = (_remaining is not None and _remaining <= 60)
            if not _wall_pressure:
                _by_prim = report.primitive_trace.get("by_primitive", {})
                _essential = {"implement_baseline", "run_experiment", "verify_against_rubric"}
                _called = set(_by_prim.keys())
                _essential_missed = not bool(_essential & _called)
                _iter_underutilized = iterations < max(1, int(_MAX_ITERATIONS * 0.25))
                _rubric_never_scored = "verify_against_rubric" not in _called
                _signal_count = sum([_essential_missed, _iter_underutilized, _rubric_never_scored])
                if _signal_count >= 2:
                    _missed_names = sorted(_essential - _called)
                    _msg = (
                        f"suspicious_partial: run completed with verdict='partial' after only "
                        f"{iterations} iteration(s) without executing key primitives "
                        f"({', '.join(_missed_names) if _missed_names else 'some'}). "
                        f"Signals: essential_primitives_missed={_essential_missed}, "
                        f"iteration_underutilization={_iter_underutilized} "
                        f"(iterations={iterations}, floor={int(_MAX_ITERATIONS * 0.25)}), "
                        f"rubric_never_scored={_rubric_never_scored}. "
                        "This may indicate the model concluded primitives were unavailable "
                        "(see BUG-LR-011/012 in rlm-stability-remediation-design.md)."
                    )
                    emit(build_run_warning_event(
                        level="warn",
                        code="suspicious_partial",
                        message=_msg,
                    ))
        except Exception:  # noqa: BLE001 — diagnostic only; never block report write
            logger.debug("_finalize: suspicious_partial check raised", exc_info=True)

    # Two-axis reproducibility verdict producers (U14/U16, flag-gated + fail-soft):
    # write repro_spec.json (claims) + fidelity_certificate.json so write_final_report_rlm
    # can attach the verdict.  No-ops entirely unless OPENRESEARCH_TWO_AXIS_VERDICT is set.
    try:
        from backend.agents.rlm import repro_spec_extractor, fidelity_certificate_builder
        if repro_spec_extractor.is_enabled():
            _paper_txt = ""
            _ptxt = project_dir / "parsed_full_text.txt"
            if _ptxt.exists():
                _paper_txt = _ptxt.read_text(encoding="utf-8", errors="replace")
            repro_spec_extractor.extract_and_write(
                None, project_dir,
                llm_client=getattr(ctx, "llm_client", None),
                paper_text=_paper_txt,
            )
            fidelity_certificate_builder.build_certificate(
                project_dir, arxiv_id=getattr(ctx, "arxiv_id", None),
            )
    except Exception:  # noqa: BLE001 — producers are best-effort; never block finalize
        logger.warning("_finalize: two-axis producers failed (non-fatal)", exc_info=True)

    # E1 (NEGATIVE_LESSONS): mine agent-correctable failure_class rows from this
    # run's experiment_runs.jsonl into runs/_lessons/<arxiv_id>.json so the NEXT
    # run of the same paper injects them into implementer guidance. Flag-gated
    # (OPENRESEARCH_NEGATIVE_LESSONS) + fail-soft; off / no arxiv_id → no-op (today).
    try:
        from backend.agents.rlm import lesson_distiller as _ld
        _ld.mine_lessons(project_dir, project_dir.parent, getattr(ctx, "arxiv_id", None))
    except Exception:  # noqa: BLE001 — lesson mining must never block finalize
        logger.warning("_finalize: negative-lessons mining failed (non-fatal)", exc_info=True)

    # Finalize-time freshness re-grade (2026-06-13): if the complete on-disk
    # grid landed AFTER the last verify_against_rubric and the recorded grade is
    # below target, re-grade the full evidence and adopt it only if higher
    # (best-of-run MAX). Closes the stale-partial-grade gap that shipped All-CNN
    # v5 at 0.558 when its 13/14-converged grid had earned ~0.73 ungraded.
    # Flag-gated default ON; fail-soft. Runs BEFORE write so the report ships
    # the recovered score (and write_final_report_rlm re-applies the floor).
    # Finalize-time freshness re-grade (single shared entry point; always emits
    # the fire/skip reason for observability). Wired into every ctx-bearing
    # finalize path so a long run that grades late — or never — still ships the
    # score its completed grid earned (2026-06-13 v5/v6/v10).
    try:
        from backend.agents.rlm import finalize_regrade as _fr
        if not run_failed:
            _fr.regrade_and_emit(ctx, report, emit)
    except Exception:  # noqa: BLE001 — re-grade is advisory, never blocks finalize
        logger.warning("_finalize: finalize_regrade failed (non-fatal)", exc_info=True)

    # P2.3 — OFFLINE adversarial validation panel (report-stamping only). Extracted to
    # _run_finalize_validation_panel so the fatal-abort + hard-stop paths run it too.
    # Runs BEFORE write_final_report_rlm so the verdict is on disk for the stamp chokepoint.
    if not run_failed:
        _run_finalize_validation_panel(ctx, report, project_dir)

    json_path, _md_path = write_final_report_rlm(
        report, project_dir, run_experiment_calls=run_experiment_call_count(ctx),
        run_experiment_ok_calls=run_experiment_success_count(ctx),
        run_experiment_partial_timeout_calls=run_experiment_partial_timeout_count(ctx)
    )
    _notify_run_terminal(project_dir)

    # Per-paper negative lessons (MUSE-lite, OPENRESEARCH_NEGATIVE_LESSONS): mine
    # agent-correctable failures from experiment_runs.jsonl into
    # runs/_lessons/<arxiv_id>.json for the next run of the same paper.
    # Flag-gated + fail-soft; no-op when arxiv_id is unknown.
    try:
        from backend.agents.rlm.lesson_distiller import mine_lessons
        mine_lessons(project_dir, project_dir.parent, getattr(ctx, "arxiv_id", None))
    except Exception:  # noqa: BLE001 — lesson bookkeeping must never break finalize
        logger.debug("_finalize: mine_lessons raised", exc_info=True)

    # Positive recipe admission (OPENRESEARCH_POSITIVE_RECIPES, default-OFF).
    # Mirrors mine_lessons; fired after write_final_report_rlm so the report is
    # on disk when admit_recipe reads experiment_runs.jsonl alongside it.
    try:
        from backend.agents.rlm import recipe_library as _rl
        from backend.agents.prompts.paper_hints import PAPER_HINTS as _PH
        from backend.agents.rlm.external_validator import load_verdict as _load_verdict
        if _rl.positive_recipes_enabled():
            _report_dict = report.model_dump() if hasattr(report, "model_dump") else (report if isinstance(report, dict) else {})
            _verdict = _load_verdict(project_dir)
            _paper_class = _rl.derive_paper_class(
                arxiv_id=getattr(ctx, "arxiv_id", None), paper_hints=_PH, rubric=_report_dict.get("rubric"),
            )
            _rl.admit_recipe(project_dir, project_dir.parent, report=_report_dict, validator_verdict=_verdict, paper_class=_paper_class)
    except Exception:  # noqa: BLE001 — recipe bookkeeping must never break finalize
        logger.debug("_finalize: recipe admission skipped", exc_info=True)

    # Write worker reports summary at run finalization
    try:
        from backend.agents.worker_reports import write_summary_report
        write_summary_report(project_dir)
    except Exception:  # noqa: BLE001
        logger.debug("run_pipeline_rlm: could not write summary_report.json")

    rubric_score = report.rubric.get("overall_score")
    cost_usd = report.cost.get("llm_usd")
    status = _verdict_to_status(report.verdict)

    emit(
        build_run_complete_event(
            status=status,
            iterations=iterations,
            rubric_score=rubric_score,
            cost_usd=cost_usd,
            final_report_path=str(json_path),
        )
    )
    logger.info(
        "run_pipeline_rlm: %s — verdict=%s iterations=%d cost=$%.4f",
        ctx.project_id,
        report.verdict,
        iterations,
        cost_usd or 0.0,
    )
    # demo_status.json terminal write: a produced report means the run-process
    # completed; a crash means it failed. The reproduction verdict (incl.
    # "partial") is a separate axis recorded in final_report.json.
    _write_demo_status(
        project_dir,
        "failed" if run_failed else "completed",
        primitive_provider=report.primitive_provider,  # T21 / review I8
    )
    return RLMRunResult(
        project_id=ctx.project_id,
        status=status,
        iterations=iterations,
        rubric_score=rubric_score,
        cost_usd=cost_usd,
        final_report_path=str(json_path),
    )


__all__ = ["RLMRunResult", "run_pipeline_rlm"]
