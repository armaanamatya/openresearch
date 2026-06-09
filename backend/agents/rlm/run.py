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
import signal
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
    RootModel,
    register_featherless_context_limits,
    resolve_root_model,
)
from backend.agents.rlm.report import (
    RLMFinalReport,
    build_final_report,
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
    ForcedIterationPolicy,
    apply_forced_iteration_patch,
    forced_iteration_policy,
)
# BUG-LR-011: restore globals()/locals() inside rlm's LocalREPL sandbox
# (upstream blacklists them alongside eval/exec/compile/input — incorrect).
from backend.agents.rlm import safe_builtins_patch as _safe_builtins_patch  # noqa: F401
# BUG-LR-012: include traceback.format_exc() in REPL exception stderr so the
# root model can diagnose failures rather than concluding primitives unavailable.
from backend.agents.rlm import safe_repl_traceback_patch as _safe_repl_traceback_patch  # noqa: F401
apply_oauth_backend_patch()
apply_anthropic_caching_patch()
# Lane H — install the FINAL_VAR interceptor once. Per-run policies are
# pushed via the forced_iteration_policy context manager around rlm.completion.
apply_forced_iteration_patch()

logger = logging.getLogger(__name__)

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

    # 1. claude-oauth — explicit OAuth path, no api_key in kwargs
    if backend == "anthropic-oauth":
        from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient
        return ClaudeLlmClient(), "claude-oauth"

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
        return OpenAILlmClient(model=model, api_key=bk["api_key"], base_url=bk["base_url"]), model

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
        return ClaudeLlmClient(), "claude"

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
    return ClaudeLlmClient(), "claude"


def _resolve_agent_runtime(
    runtime: Any, provider: str | None
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


def _write_demo_status(
    project_dir: Path,
    status: str,
    *,
    error: Any | None = None,
    primitive_provider: str = "real",  # T21 / review I8
    process_status: str | None = None,
    verdict: str | None = None,
    warnings: list[str] | None = None,
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
    payload["primitiveProvider"] = primitive_provider  # T21 / review I8
    payload["process_status"] = process_status
    payload["verdict"] = verdict
    # Liveness prereq: run_liveness.sweep_orphaned_runs deliberately skips
    # runs without a pid ("absent-pid means unknown, not dead"), so a
    # SIGKILLed CLI/batch run used to show status=running forever — only the
    # API spawn path stamped one (live_runs.py). For CLI runs this process IS
    # the run; for API runs the parent already wrote the same subprocess pid
    # and **existing keeps it.
    payload.setdefault("pid", os.getpid())
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
        fatal = _fatal_primitive_result(getattr(self, "_ctx", None))
        if fatal is not None:
            primitive_name, result = fatal
            raise _FatalPrimitiveAbort(
                primitive_name=primitive_name,
                result=result,
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
    json_path, _md_path = write_final_report_rlm(report, project_dir)

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


def _hard_stop_with_report(
    *,
    project_dir: Path,
    emit: Any,
    done: int,
    summary: str,
    status_error: str,
    exit_code: int,
) -> None:
    """Ship a partial ``failed`` report, emit ``run_complete``, flip demo_status, and
    hard-exit — the single "never die without a report" path shared by the wall-clock
    watchdog and the SIGTERM finalizer (2026-06-01). Every step is best-effort so a
    failure writing one artifact never blocks the others or the exit.
    """
    report = RLMFinalReport(verdict="failed", reproduction_summary=summary, iterations=done)
    try:
        write_final_report_rlm(report, project_dir)
    except Exception:  # noqa: BLE001
        logger.exception("run_pipeline_rlm: hard-stop could not write final report")
    try:
        emit(
            build_run_complete_event(
                status="failed",
                iterations=done,
                rubric_score=None,
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
) -> threading.Timer | None:
    """Arm the process-level wall-clock backstop (design spec §8, Codex H2).

    ``rlm``'s ``max_timeout`` only checks between iterations; a primitive wedged
    inside ``execute_code`` can overrun it indefinitely, and a Python thread
    cannot be killed.  This timer fires ``_WATCHDOG_GRACE_S`` past the deadline
    (so it only triggers when ``rlm``'s own timeout failed), writes an honest
    partial report, and hard-exits the process — the OS then reclaims the wedged
    worker thread.

    ``iteration_count`` is a zero-arg callable returning the iterations done so
    far.  Returns the armed (daemon) ``Timer`` — the caller must ``.cancel()``
    it on normal completion. When ``deadline_s`` is ``None`` (no explicit
    ``--max-wall-clock``), the watchdog falls back to the always-on hard-ceiling
    backstop (``_watchdog_hard_ceiling_s``) so a wedged/hung run still ships a
    partial report; it returns ``None`` (fully bypassed) only when that backstop
    is disabled via ``OPENRESEARCH_WATCHDOG_HARD_CEILING_S=0``.
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
        )

    timer = threading.Timer(deadline_s + _WATCHDOG_GRACE_S, _fire)
    timer.daemon = True
    timer.start()
    return timer


def _install_sigterm_finalizer(
    *,
    project_dir: Path,
    emit: Any,
    iteration_count: Any,
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
    _allow_lossy = getattr(_settings_for_gate, "allow_lossy_paper_text", True)
    _paper_degraded_reason = _assert_paper_text_precondition(project_dir, allow_lossy=_allow_lossy)

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

    # Status snapshot at run start — GET /runs/{id} reads this; without it a
    # CLI- or script-launched RLM run 404s. Terminal status is set in _finalize.
    # Surface a degraded-paper-text warning here (F-29) so an operator sees the
    # run is non-faithful; the merge in _write_demo_status carries it forward.
    _write_demo_status(
        project_dir,
        "running",
        warnings=[_paper_degraded_reason] if _paper_degraded_reason else None,
    )

    # Local sandboxes have no /workspace volume — repoint the dataset root at a
    # writable shared cache BEFORE any primitive (implement_baseline / run_experiment)
    # reads it, so dataset/env setup does not die at os.makedirs. See the helper.
    _ensure_local_data_root(sandbox_mode, runs_root)

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
    root_model = resolve_root_model(model)
    if not root_model.paper_validated:
        logger.warning(
            "run_pipeline_rlm: root model %r is NOT paper-validated as an RLM root "
            "(root_model_unvalidated) — results may not match paper expectations",
            root_model.key,
        )

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

    # 4. RunContext. The sub-agent runtime + model are resolved here so
    #    implement_baseline never falls through to a dead env-default key,
    #    and runs Sonnet rather than the registry's Opus default.
    agent_runtime, agent_model, runtime_label = _resolve_agent_runtime(runtime, provider)
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
        gpu_device_ids=_parse_gpu_device_ids(),
        gpu_parallelism=_parse_gpu_parallelism(),
        gpu_visible_count=_visible_gpu_count(),
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
    # Stub-primitive runs skip this: rubric generation is a REAL paid LLM call
    # (the one non-stubbed network path), and under pytest it turned the
    # leaked .env OPENAI_API_KEY into minutes of 429-retry sleep per run —
    # the suite's 862s-test stall (audit 2026-06-09).
    _stub_mode = os.environ.get("OPENRESEARCH_RLM_STUB_PRIMITIVES") == "1"
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
    # Accelerator sub-backend override: when an accelerator endpoint is active and
    # not Azure, redirect rlm_query/llm_query navigation to the same fast endpoint
    # so context-navigation calls also benefit from the accelerator.  Azure endpoints
    # require their own backend type and are left unchanged.
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
    )
    # Ship a partial report on a graceful SIGTERM kill too (not just on a hang).
    _prev_sigterm = _install_sigterm_finalizer(
        project_dir=project_dir,
        emit=emit,
        iteration_count=lambda: rlm_logger.iteration_count,
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
    )
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
    # verifier == grader: both are the rubric-scoring client (ctx.llm_client), whose model
    # is llm_model. Under the default accelerator scope="navigation" this stays the strong
    # root model (Sonnet) even when a small accelerator serves rlm_query/llm_query nav.
    report.models = {
        "planner": llm_model,
        "executor": getattr(ctx, "agent_model", None),
        "verifier": llm_model,
        "grader": llm_model,
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

    json_path, _md_path = write_final_report_rlm(report, project_dir)

    # Per-paper negative lessons (MUSE-lite, OPENRESEARCH_NEGATIVE_LESSONS): mine
    # agent-correctable failures from experiment_runs.jsonl into
    # runs/_lessons/<arxiv_id>.json for the next run of the same paper.
    # Flag-gated + fail-soft; no-op when arxiv_id is unknown.
    try:
        from backend.agents.rlm.lesson_distiller import mine_lessons
        mine_lessons(project_dir, project_dir.parent, getattr(ctx, "arxiv_id", None))
    except Exception:  # noqa: BLE001 — lesson bookkeeping must never break finalize
        logger.debug("_finalize: mine_lessons raised", exc_info=True)

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
