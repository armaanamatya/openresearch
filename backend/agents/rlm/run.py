"""run.py — the RLM run entry.

``run_pipeline_rlm()`` is the single run entry point.  It:

  1. builds a run-scoped :class:`RunContext`,
  2. resolves the primitive layer — the real ``build_custom_tools`` from
     ``binding.py`` if importable, else the deterministic stub provider,
  3. constructs an ``rlm.RLM`` (the Recursive Language Model engine),
  4. runs ``.completion()`` on a worker thread, streaming + checkpointing every
     iteration through :class:`ReproLabRLMLogger`,
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
    ReproLabRLMLogger,
    build_run_complete_event,
    make_emit,
    make_on_subcall_complete,
    make_on_subcall_start,
    redact_corpus,
)
from backend.agents.rlm.stub_primitives import build_stub_custom_tools
from backend.agents.rlm.system_prompt import build_system_prompt

# Register the anthropic-oauth backend with rlm.clients.get_client — must run
# before RLM(backend="anthropic-oauth", ...) is constructed below.
from backend.agents.rlm._oauth_backend_patch import apply_oauth_backend_patch
apply_oauth_backend_patch()

logger = logging.getLogger(__name__)

# --- Tuning constants ------------------------------------------------------
_MAX_ITERATIONS = 20          # paper Appendix A
_MAX_DEPTH = 2                # brief §3 — depth-2 enables real rlm_query recursion
_DEFAULT_WALL_CLOCK_S = 3600.0
_WATCHDOG_GRACE_S = 120.0     # watchdog fires only past rlm's own max_timeout
_WATCHDOG_EXIT_CODE = 75      # EX_TEMPFAIL — "the run was hard-stopped"

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

    # 4. Anthropic raw HTTP — uses ANTHROPIC_API_KEY through claude-agent-sdk's resolution
    if backend == "anthropic":
        from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient
        return ClaudeLlmClient(), "claude"

    # 5. Plain OpenAI
    if backend == "openai":
        from backend.services.context.workspace.tools.openai_client import OpenAILlmClient
        return OpenAILlmClient(), "gpt-4o-mini"

    # 6. Unknown backend — respect explicit `provider` arg, else default to Claude.
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
    if os.environ.get("REPROLAB_RLM_STUB_PRIMITIVES") == "1":
        return build_stub_custom_tools(ctx), "stub (REPROLAB_RLM_STUB_PRIMITIVES=1)"
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


def _write_demo_status(
    project_dir: Path,
    status: str,
    *,
    error: str | None = None,
    primitive_provider: str = "real",  # T21 / review I8
) -> None:
    """Write (merge) ``runs/<id>/demo_status.json`` so the run is REST-retrievable.

    The HTTP layer's ``GET /runs/{id}`` reads this snapshot via
    ``live_runs._read_status``; without it a CLI- or script-launched RLM run
    404s. The payload carries ``LiveRunState``'s required fields (``projectId``,
    ``outputDir``, ``runMode``, ``status``). Any pre-existing file is merged, not
    overwritten, so an earlier ``startedAt`` survives the terminal write.

    ``status`` must be a valid ``RunStatus`` (``running`` | ``completed`` |
    ``failed`` | ``stopped``) — the reproduction *verdict* (which may be
    ``partial``) is a separate axis and lives in ``final_report.json``.
    """
    path = project_dir / "demo_status.json"
    now = datetime.now(timezone.utc).isoformat()
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
    if status in ("completed", "failed", "stopped"):
        payload["completedAt"] = now
    if error is not None:
        payload["error"] = error
    payload["primitiveProvider"] = primitive_provider  # T21 / review I8
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 — status is best-effort; never crash the run
        logger.exception("run_pipeline_rlm: could not write demo_status.json")


def _arm_watchdog(
    deadline_s: float,
    *,
    project_dir: Path,
    emit: Any,
    iteration_count: Any,
) -> threading.Timer:
    """Arm the process-level wall-clock backstop (design spec §8, Codex H2).

    ``rlm``'s ``max_timeout`` only checks between iterations; a primitive wedged
    inside ``execute_code`` can overrun it indefinitely, and a Python thread
    cannot be killed.  This timer fires ``_WATCHDOG_GRACE_S`` past the deadline
    (so it only triggers when ``rlm``'s own timeout failed), writes an honest
    partial report, and hard-exits the process — the OS then reclaims the wedged
    worker thread.

    ``iteration_count`` is a zero-arg callable returning the iterations done so
    far.  Returns the armed (daemon) ``Timer`` — the caller must ``.cancel()``
    it on normal completion.
    """
    def _fire() -> None:
        logger.error(
            "run_pipeline_rlm: wall-clock watchdog fired (%.0fs + %.0fs grace) — "
            "hard-stopping a wedged run",
            deadline_s,
            _WATCHDOG_GRACE_S,
        )
        done = iteration_count()
        report = RLMFinalReport(
            verdict="failed",
            reproduction_summary=(
                f"Wall-clock watchdog: the run exceeded {deadline_s:.0f}s "
                f"and was hard-stopped after {done} iteration(s)."
            ),
            iterations=done,
        )
        try:
            write_final_report_rlm(report, project_dir)
        except Exception:  # noqa: BLE001
            logger.exception("run_pipeline_rlm: watchdog could not write final report")
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
            logger.exception("run_pipeline_rlm: watchdog could not emit run_complete event")
        _write_demo_status(
            project_dir,
            "failed",
            error=f"wall-clock watchdog: run hard-stopped past its {deadline_s:.0f}s deadline",
        )
        os._exit(_WATCHDOG_EXIT_CODE)

    timer = threading.Timer(deadline_s + _WATCHDOG_GRACE_S, _fire)
    timer.daemon = True
    timer.start()
    return timer


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


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
    # Status snapshot at run start — GET /runs/{id} reads this; without it a
    # CLI- or script-launched RLM run 404s. Terminal status is set in _finalize.
    _write_demo_status(project_dir, "running")

    # 1. Observability + budget.
    cost_ledger = RunCostLedger.load_jsonl(
        project_dir / "cost_ledger.jsonl", project_id=project_id, attach_path=True
    )
    dashboard = DashboardEmitter(project_id, runs_root)
    emit = make_emit(dashboard)
    wall_clock_s = _DEFAULT_WALL_CLOCK_S
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
        run_budget=run_budget,
        deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=wall_clock_s),  # M-DEADLINE
    )

    # 5. Primitives — the real binding or the stub provider.
    custom_tools, tools_label = _resolve_custom_tools(ctx)
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
    if not context_dict.get("rubric_spec") and context_dict.get("paper_text"):
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
            "score_against_rubric. Do NOT rewrite passing clusters. "
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
    rlm_logger = ReproLabRLMLogger(
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
        other_backends=[root_model.sub_backend],
        other_backend_kwargs=[root_model.sub_backend_kwargs],
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
    result_obj: Any = None
    run_failed = False
    try:
        result_obj = await asyncio.to_thread(rlm.completion, context_dict, active_prompt)
    except Exception as exc:  # noqa: BLE001 — an honest failure is data, not a crash
        run_failed = True
        logger.exception("run_pipeline_rlm: rlm.completion failed: %s", exc)
    finally:
        watchdog.cancel()

    # 12. Build, write, and report (close event_store in finally — A4-9).
    try:
        return _finalize(
            result_obj=result_obj,
            run_failed=run_failed,
            ctx=ctx,
            iterations=rlm_logger.iteration_count,
            project_dir=project_dir,
            emit=emit,
            corpus_sentinels=corpus_sentinels,
            tools_label=tools_label,  # T21 / review I8
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

    json_path, _md_path = write_final_report_rlm(report, project_dir)

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
