"""Phase-4 RDR run entry — thin entry point that builds a RunContext and calls
``run_rdr``.

Usage::

    from backend.agents.rdr.run import run_pipeline_rdr
    result = await run_pipeline_rdr(project_id, runs_root, paper_id="sequential-neural-score-estimation")

Mirrors ``backend.agents.rlm.run.run_pipeline_rlm``'s construction pattern:
``DashboardEmitter``, ``RunCostLedger``, dynamic LLM client, runtime resolution.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.agents.dashboard_emitter import DashboardEmitter
from backend.agents.resilience.cost import RunCostLedger
from backend.agents.rlm.context import RunContext
from backend.agents.rdr.controller import run_rdr
from backend.agents.rdr.models import RdrResult

logger = logging.getLogger(__name__)

# The canonical paperbench bundle root. run.py is at backend/agents/rdr/run.py,
# so parents[3] is the repo root.
_PAPERBENCH_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "paperbench"


def _resolve_bundle_path(
    paper_id: str,
    bundles_root: "str | Path | None" = None,
) -> Path:
    """Resolve the absolute path to a paperbench bundle directory.

    Supports:
    - a bare ``paper_id`` relative to ``bundles_root`` (defaults to
      ``third_party/paperbench/`` when ``None``).
    - an absolute path (returned as-is).
    """
    p = Path(paper_id)
    if p.is_absolute():
        return p
    root = Path(bundles_root).resolve() if bundles_root is not None else _PAPERBENCH_ROOT
    candidate = root / paper_id
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"PaperBench bundle not found for paper_id={paper_id!r}. "
        f"Looked at {candidate}. "
        f"Bundles live under {root}."
    )


def _effective_provider(
    provider: str | None,
    model: str | None = None,
) -> str | None:
    """Resolve the effective LLM provider.

    Resolution order (most-explicit-first):

      1. Explicit ``provider`` arg wins.
      2. Model name implies provider — ``claude-*`` / ``sonnet`` / ``opus`` /
         ``haiku`` → anthropic; ``gpt-*`` / contains ``openai`` → openai.
      3. ``has_provider_credentials("anthropic")`` (covers API key, Claude
         OAuth subscription, macOS Keychain) — preferred over a possibly-
         stale ``OPENAI_API_KEY``. Same root-cause class as commit 005e3b6
         fixed for ``_build_llm_client`` in rlm: a revoked/invalid
         ``OPENAI_API_KEY=sk-svcacct...`` in ``.env`` silently misrouted
         every cluster agent to OpenAI (live smoke
         ``rlm_hybrid_smoke_1779521608`` failed 27/27 clusters this way).
      4. ``OPENAI_API_KEY`` present (fall-through; will still 401 if invalid
         but at least the explicit-model path bypasses it).
      5. ``None`` — fall back to whatever ``collect_agent_text`` resolves
         internally.

    Returns ``"anthropic"``, ``"openai"``, or ``None``.
    """
    if provider is not None:
        return provider.lower()
    # 2. Model implies provider.
    if model:
        m = model.lower()
        if (
            m.startswith("claude")
            or "sonnet" in m
            or "opus" in m
            or "haiku" in m
        ):
            return "anthropic"
        if m.startswith("gpt") or "openai" in m:
            return "openai"
    # 3. Valid Anthropic credentials win over a possibly-stale OPENAI key.
    try:
        from backend.agents.runtime.factory import has_provider_credentials
        if has_provider_credentials("anthropic"):
            return "anthropic"
    except Exception:  # noqa: BLE001 — defensive: factory import failure
        pass
    # 4. OPENAI_API_KEY presence as last resort.
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY_PATH"):
        return "anthropic"
    return None


def _build_llm_client(
    effective_prov: str | None,
    model: str | None,
) -> tuple[Any, str, str]:
    """Build an ``LlmClient`` for primitives.  Returns ``(client, model, provider_label)``.

    Resolution is dynamic so it works with:
    - Claude OAuth / API key (``anthropic`` provider)
    - Azure / standard OpenAI (``openai`` provider, reads ``OPENAI_BASE_URL`` /
      ``AZURE_OPENAI_ENDPOINT`` for Azure endpoints)

    ``model`` is passed at construction time (not patched after).
    Falls back to ``ClaudeLlmClient`` when no explicit provider or OpenAI key
    is detected (matching the pattern in ``rlm/run.py``).
    """
    if effective_prov == "openai":
        from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

        resolved_model = model or "gpt-4o-mini"
        # Azure OpenAI: honour OPENAI_BASE_URL or AZURE_OPENAI_ENDPOINT
        base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
            or None
        )
        api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or None
        client = OpenAILlmClient(resolved_model, base_url=base_url, api_key=api_key)
        return client, resolved_model, "openai"

    from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient

    resolved_model = model or "claude"
    return ClaudeLlmClient(), resolved_model, "anthropic"


async def run_pipeline_rdr(
    project_id: str,
    runs_root: Path,
    *,
    paper_id: str,
    provider: str | None = None,
    model: str | None = None,
    sandbox_mode: Any = None,
    max_repair_iterations: int = 2,
    repair_target: float = 0.6,
    bundles_root: "str | Path | None" = None,
    resume: bool = False,
    run_budget: Any = None,
) -> RdrResult:
    """Run one paper reproduction using the rubric-driven ``rdr`` harness.

    Loads the PaperBench bundle, builds a :class:`~backend.agents.rlm.context.RunContext`,
    then calls :func:`~backend.agents.rdr.controller.run_rdr`.

    Args:
        project_id: Unique run identifier; determines the output directory.
        runs_root: Root directory under which ``<project_id>/`` is created.
        paper_id: PaperBench bundle name (e.g. ``"sequential-neural-score-estimation"``)
            or an absolute path to the bundle directory.
        provider: LLM provider hint (``"anthropic"`` | ``"openai"``); ``None`` → auto-detect.
        model: Model override passed to the LLM client at construction time.
        sandbox_mode: Forwarded to ``RunContext.sandbox_mode``; selects Docker/local/etc.
        max_repair_iterations: Maximum repair loops passed to ``run_rdr``.
        repair_target: Cluster-level score threshold passed to ``run_rdr``.
        bundles_root: Override the default ``third_party/paperbench/`` root when
            resolving ``paper_id``.  ``None`` → use the canonical vendored root.
        resume: When True, reuse the existing project_dir and resume from
            cluster checkpoints rather than starting fresh.
        run_budget: Optional budget object threaded into primitive calls,
            runtime sandboxes, and controller watchdog metadata.

    Returns:
        An :class:`~backend.agents.rdr.models.RdrResult`.
    """
    from backend.evals.paperbench.bundle import load_paperbench_bundle

    runs_root = Path(runs_root).resolve()
    project_dir = runs_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    # Resolve bundle — bundles_root threads through to _resolve_bundle_path so a
    # custom --bundles-root flag is respected at run time, not just for validation.
    bundle_path = _resolve_bundle_path(paper_id, bundles_root=bundles_root)
    bundle = load_paperbench_bundle(bundle_path)

    # Observability
    cost_ledger = RunCostLedger.load_jsonl(
        project_dir / "cost_ledger.jsonl",
        project_id=project_id,
        attach_path=True,
    )
    dashboard = DashboardEmitter(project_id, runs_root)

    # Compute a single effective provider — explicit arg wins; auto-detect otherwise.
    eff_provider = _effective_provider(provider, model=model)

    # LLM client — dynamic; works with Claude OAuth, standard OpenAI, and Azure OpenAI.
    # model is passed at construction time so it is honoured immediately.
    llm_client, resolved_model, provider_label = _build_llm_client(eff_provider, model)

    # Agent runtime — only build the runtime that matches eff_provider.
    # When eff_provider is "openai", skip Anthropic runtime so the Azure path works.
    agent_runtime = None
    agent_model = None
    try:
        from backend.agents.runtime.factory import (
            has_provider_credentials,
            make_runtime,
        )
        from backend.config import get_settings

        if eff_provider in (None, "anthropic") and has_provider_credentials("anthropic"):
            agent_model = get_settings().anthropic_default_model
            agent_runtime = make_runtime("anthropic")
        elif eff_provider == "openai":
            try:
                agent_runtime = make_runtime("openai")
            except Exception:  # noqa: BLE001 — openai runtime is optional
                pass
    except Exception as exc:  # noqa: BLE001 — runtime resolution is optional
        logger.warning(
            "run_pipeline_rdr: agent runtime resolution failed (%s: %s) — "
            "implement_baseline will surface the credential error if needed",
            type(exc).__name__,
            exc,
        )

    deadline_utc = None
    wall_clock_s = getattr(run_budget, "max_wall_clock_seconds", None)
    if wall_clock_s is not None:
        deadline_utc = datetime.now(timezone.utc) + timedelta(seconds=float(wall_clock_s))

    ctx = RunContext(
        project_id=project_id,
        project_dir=project_dir,
        runs_root=runs_root,
        dashboard=dashboard,
        cost_ledger=cost_ledger,
        llm_client=llm_client,
        provider=provider_label,
        model=resolved_model,
        agent_model=agent_model,
        runtime=agent_runtime,
        sandbox_mode=sandbox_mode,
        run_budget=run_budget,
        deadline_utc=deadline_utc,
    )

    logger.info(
        "run_pipeline_rdr: project=%s paper=%s provider=%s model=%s",
        project_id, paper_id, provider_label, resolved_model,
    )

    return await run_rdr(
        bundle,
        ctx=ctx,
        max_repair_iterations=max_repair_iterations,
        repair_target=repair_target,
        resume=resume,
    )


__all__ = ["run_pipeline_rdr"]
