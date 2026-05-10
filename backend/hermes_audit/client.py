"""Adapter for the Nous Hermes Python runtime with Claude Code SDK fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from backend.hermes_audit.models import (
    HermesAuditReport,
    HermesAuditScope,
    HermesAuditStatus,
    HermesInterventionType,
)

logger = logging.getLogger(__name__)


def _resolve_hermes_config() -> tuple[str, str, str]:
    """Return (model, api_key, provider) based on available credentials.

    Checks ANTHROPIC_API_KEY first, then OPENAI_API_KEY.  Returns the
    Hermes-style ``provider/model`` string, the raw key, and the provider
    name so callers can pass all three to ``AIAgent``.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        return "anthropic/claude-sonnet-4", anthropic_key, "anthropic"

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return "openai/gpt-4o", openai_key, "openai"

    return "", "", ""


class NousHermesClient:
    """Audit client: Hermes Agent primary, Claude Code SDK fallback.

    Resolution order for each ``audit()`` call:

    1. **Hermes Agent** (``run_agent.AIAgent``) — full agent with tool use.
       Requires the ``hermes-agent`` package *and* a valid API key.
    2. **Claude Code SDK** (``claude_agent_sdk.query``) — lightweight
       single-turn query.  Requires ``claude-agent-sdk`` (already used by
       the main pipeline).
    3. **Unavailable report** — returned when both backends fail so the
       pipeline can continue without crashing.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        enabled: bool = True,
    ) -> None:
        self._explicit_model = model
        self._explicit_api_key = api_key
        self.enabled = enabled

    def _effective_config(self) -> tuple[str, str, str]:
        """Return (model, api_key, provider) honouring explicit overrides."""
        auto_model, auto_key, auto_provider = _resolve_hermes_config()
        model = self._explicit_model or auto_model
        api_key = self._explicit_api_key or auto_key
        provider = auto_provider
        return model, api_key, provider

    def audit(self, *, scope: HermesAuditScope, target: str, payload: dict[str, Any]) -> HermesAuditReport:
        if not self.enabled:
            return HermesAuditReport(
                target=target,
                scope=scope,
                status=HermesAuditStatus.unavailable,
                summary="Nous Hermes audit disabled",
                recommended_intervention=HermesInterventionType.annotate,
            )

        prompt = self._build_prompt(scope=scope, target=target, payload=payload)
        hermes_error = ""
        sdk_error = ""

        # --- Primary: Hermes Agent ---
        try:
            response = self._run_hermes_agent(prompt)
            return self._parse_response(response, target=target, scope=scope, provider="nous-hermes")
        except Exception as exc:
            hermes_error = str(exc)
            logger.warning("Hermes Agent unavailable (%s), trying Claude Code SDK fallback", exc)

        # --- Fallback: Claude Code SDK ---
        try:
            response = self._run_claude_sdk(prompt)
            return self._parse_response(response, target=target, scope=scope, provider="claude-code-sdk")
        except Exception as exc:
            sdk_error = str(exc)
            logger.warning("Claude Code SDK fallback also failed: %s", exc)

        # --- Both failed ---
        return HermesAuditReport(
            target=target,
            scope=scope,
            status=HermesAuditStatus.unavailable,
            summary="All audit backends unavailable",
            recommended_intervention=HermesInterventionType.annotate,
            error_message=f"hermes: {hermes_error}; claude-sdk: {sdk_error}",
        )

    def _parse_response(
        self,
        response: str,
        *,
        target: str,
        scope: HermesAuditScope,
        provider: str,
    ) -> HermesAuditReport:
        data = self._extract_json(response)
        data.setdefault("target", target)
        data.setdefault("scope", scope.value)
        data.setdefault("provider", provider)
        return HermesAuditReport(**data)

    # ------------------------------------------------------------------
    # Backend 1: Hermes Agent (run_agent.AIAgent)
    # ------------------------------------------------------------------

    def _run_hermes_agent(self, prompt: str) -> str:
        from run_agent import AIAgent  # hermes-agent package

        model, api_key, provider = self._effective_config()
        if not api_key:
            raise RuntimeError("No API key available for Hermes Agent")

        agent = AIAgent(
            model=model,
            api_key=api_key,
            provider=provider,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        if hasattr(agent, "chat"):
            return str(agent.chat(prompt))
        if hasattr(agent, "run"):
            return str(agent.run(prompt))
        if callable(agent):
            return str(agent(prompt))
        raise RuntimeError("Unsupported Hermes Agent interface")

    # ------------------------------------------------------------------
    # Backend 2: Claude Code SDK (claude_agent_sdk.query)
    # ------------------------------------------------------------------

    def _run_claude_sdk(self, prompt: str) -> str:
        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

        options = ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            max_turns=1,
        )

        collected: list[str] = []

        # query() is async — run it in an event loop
        async def _collect() -> None:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    text = getattr(message, "text", "")
                    if text:
                        collected.append(str(text))
                else:
                    for block in getattr(message, "content", []):
                        text = getattr(block, "text", "")
                        if text:
                            collected.append(str(text))

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(lambda: asyncio.run(_collect())).result(timeout=120)
        else:
            asyncio.run(_collect())

        result = "\n".join(collected)
        if not result.strip():
            raise RuntimeError("Claude Code SDK returned empty response")
        return result

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(*, scope: HermesAuditScope, target: str, payload: dict[str, Any]) -> str:
        return (
            "You are auditing a research reproduction pipeline for unsupported claims.\n"
            f"Scope: {scope.value}\n"
            f"Target: {target}\n"
            "Return only JSON with fields: target, scope, status, summary, findings, "
            "unsupported_claims, evidence_refs, recommended_intervention, corrective_note, confidence.\n"
            f"Payload:\n```json\n{json.dumps(payload, indent=2)}\n```"
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if fence_match:
            return json.loads(fence_match.group(1))
        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for idx in range(brace_start, len(text)):
                if text[idx] == "{":
                    depth += 1
                elif text[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[brace_start : idx + 1])
        raise ValueError("No JSON found in audit response")
