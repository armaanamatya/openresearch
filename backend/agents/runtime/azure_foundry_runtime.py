"""Azure AI Foundry executor runtime — OpenAI-compatible custom endpoint (e.g. Grok).

This is the sub-role (executor-tier) twin of the root ``azure-foundry`` model:
``implement_baseline`` (and other executor-tier agents) run against an Azure AI
Foundry deployment served on a ``*.services.ai.azure.com/openai/v1`` endpoint
instead of the default Sonnet (claude-agent-sdk). Combined with the foundry
grader/verifier transport, a run can be fully OAuth-free and any sub-role is
interchangeable grok⇄openai⇄claude.

Unlike ``AzureOpenAiAgentRuntime`` (classic Azure OpenAI
``/openai/deployments/{name}?api-version=`` path, ``AsyncAzureOpenAI``), Foundry
is a v1 *OpenAI-compatible* surface (Bearer auth, ``base_url=…/openai/v1``,
``model=deployment``) — so it rides the plain OpenAI SDK via
``OpenAiAgentRuntime``'s custom-base_url branch, NOT ``AsyncAzureOpenAI``.

Credentials resolve through the single canonical
``foundry_endpoint.resolve_foundry_credentials`` resolver (env then Settings/.env),
never read ad hoc.

EXPERIMENTAL for the executor tier: a non-Claude executor is not paper-validated
(``role_models.RoleSelection.fidelity_warnings`` surfaces this, advisory only).
"""

from __future__ import annotations

from backend.agents.runtime.foundry_endpoint import resolve_foundry_credentials
from backend.agents.runtime.openai_runtime import OpenAiAgentRuntime


class AzureFoundryAgentRuntime(OpenAiAgentRuntime):
    """OpenAI Agents SDK pointed at an Azure AI Foundry OpenAI-compatible endpoint.

    Resolves ``(base_url, deployment, api_key)`` from
    ``foundry_endpoint.resolve_foundry_credentials()`` and configures the base
    ``OpenAiAgentRuntime`` for the custom-endpoint (chat-completions) path. The
    Foundry deployment name is the model id — Foundry routes by deployment, like
    Azure OpenAI — so ``_model_override`` returns it (mirrors
    ``AzureOpenAiAgentRuntime._model_override``).

    ``provider_name`` is ``"openai"`` (inherited) — Foundry rides the OpenAI SDK,
    so callers that branch on provider_name need no change.
    """

    def __init__(self) -> None:
        base_url, deployment, api_key = resolve_foundry_credentials()
        super().__init__(base_url=base_url, api_key=api_key, use_chat_completions=True)
        self._deployment = deployment

    def _model_override(self) -> str | None:
        """Return the Foundry deployment name as the model id (Foundry routes by it)."""
        return self._deployment or None


__all__ = ["AzureFoundryAgentRuntime"]
