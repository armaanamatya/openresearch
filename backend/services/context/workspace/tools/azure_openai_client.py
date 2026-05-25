"""Azure OpenAI-backed LlmClient for RLM queries.

Mirrors the interface of OpenAILlmClient but uses openai.AzureOpenAI so
primitives work transparently with an Azure-hosted endpoint.  Pins
temperature=0 for deterministic recursion replay.
"""

from __future__ import annotations

import os

from backend.services.context.workspace.tools._retry import with_429_backoff

# Current Azure OpenAI Chat Completions GA used as the default when the
# operator (or BYO body) doesn't specify one. 2024-10-21 is the
# longest-stable GA that supports gpt-4o, gpt-4o-mini, o1, and
# structured-output JSON mode — picked over the older 2024-02-01 that
# the rlm library still ships with. Override per-deployment by setting
# ``AZURE_OPENAI_API_VERSION`` in the run env, or by passing
# ``azure_openai_api_version`` in the upload form's BYO credentials.
DEFAULT_AZURE_OPENAI_API_VERSION = "2024-10-21"


class AzureOpenAILlmClient:
    """LlmClient backed by Azure OpenAI Chat Completions.

    Required constructor args (all resolved from env by resolve_root_model):
      - ``azure_endpoint``: Azure OpenAI resource URL, e.g.
        ``https://<resource>.openai.azure.com``.
      - ``azure_deployment``: deployment name (may differ from model name).

    ``api_version`` resolves in this order: constructor arg → the
    ``AZURE_OPENAI_API_VERSION`` env var → ``DEFAULT_AZURE_OPENAI_API_VERSION``.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        azure_endpoint: str,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 300.0,
    ) -> None:
        from openai import AzureOpenAI

        resolved_version = (
            api_version
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or DEFAULT_AZURE_OPENAI_API_VERSION
        )
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=resolved_version,
            timeout=timeout,
            max_retries=6,
        )
        self._model = model
        self._max_tokens = max_tokens

    @with_429_backoff
    def complete(self, *, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content or ""


__all__ = ["AzureOpenAILlmClient"]
