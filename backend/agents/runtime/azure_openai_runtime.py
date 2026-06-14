"""Azure OpenAI executor runtime — opt-in sub-agent (executor tier) backed by Azure OpenAI.

This is the Stream D executor runtime: ``implement_baseline`` (and other
executor-tier agents) run against an Azure OpenAI deployment instead of the
default Sonnet (claude-agent-sdk).

EXPERIMENTAL: gpt-4o is not paper-validated for the executor tier — this
runtime exists for client-owned all-Azure fan-out scenarios where every
inference hop must stay within a tenant's Azure subscription.  The default
executor remains Sonnet; set ``OPENRESEARCH_EXECUTOR=azure`` to opt in.

Fan-out stays sequential best-of-N (decision D9 — no parallelism change).
"""

from __future__ import annotations

import os
from typing import Any

from backend.agents.runtime.openai_runtime import OpenAiAgentRuntime
from backend.services.context.workspace.tools.azure_openai_client import (
    DEFAULT_AZURE_OPENAI_API_VERSION,
)


class AzureOpenAiAgentRuntime(OpenAiAgentRuntime):
    """OpenAI Agents SDK pointed at an Azure OpenAI deployment.

    Resolves credentials from constructor args falling back to env vars:
      - ``azure_endpoint``  → ``AZURE_OPENAI_ENDPOINT``
      - ``api_key``         → ``AZURE_OPENAI_API_KEY``
      - ``api_version``     → ``AZURE_OPENAI_API_VERSION`` → ``DEFAULT_AZURE_OPENAI_API_VERSION``
      - ``deployment``      → ``AZURE_OPENAI_DEPLOYMENT``

    ``provider_name`` is ``"openai"`` (inherited) — Azure routes through the
    same OpenAI Agents SDK; callers that branch on provider_name need no change.
    """

    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        deployment: str | None = None,
    ) -> None:
        resolved_endpoint = (azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()
        resolved_key = (api_key or os.environ.get("AZURE_OPENAI_API_KEY") or "").strip()
        resolved_version = (
            api_version
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or DEFAULT_AZURE_OPENAI_API_VERSION
        )
        resolved_deployment = (deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or "").strip()

        # Pass the resolved key up; base_url=None so the base class does not
        # enter its vLLM/custom-endpoint branch in _configure_sdk_client.
        super().__init__(base_url=None, api_key=resolved_key, use_chat_completions=True)

        self._azure_endpoint = resolved_endpoint
        self._api_version = resolved_version
        self._deployment = resolved_deployment

    def _configure_sdk_client(self, agents_module: Any) -> tuple[Any, Any]:
        """Build an ``AsyncAzureOpenAI`` client and return it with the chat-completions model class."""
        from openai import AsyncAzureOpenAI

        client = AsyncAzureOpenAI(
            azure_endpoint=self._azure_endpoint,
            api_key=self._api_key or None,
            api_version=self._api_version,
            azure_deployment=self._deployment,
        )
        chat_model_cls = getattr(agents_module, "OpenAIChatCompletionsModel", None)
        # Disable SDK tracing — avoids the SDK POSTing traces to api.openai.com
        # with Azure credentials, which would cause 401 noise.
        _set_td = getattr(agents_module, "set_tracing_disabled", None)
        if _set_td is not None:
            _set_td(True)
        return client, chat_model_cls

    def _model_override(self) -> str | None:
        """Return the Azure deployment name as the model id.

        Azure OpenAI routes by deployment name, not by the underlying model id
        (e.g. ``gpt-4o``).  Returning the deployment here ensures the Agents
        SDK sends ``"model": "<deployment>"`` in every chat completions request.
        """
        return self._deployment or None


__all__ = ["AzureOpenAiAgentRuntime"]
