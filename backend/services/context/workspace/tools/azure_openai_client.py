"""Azure OpenAI-backed LlmClient for RLM queries.

Mirrors the interface of OpenAILlmClient but uses openai.AzureOpenAI so
primitives work transparently with an Azure-hosted endpoint.  Pins
temperature=0 for deterministic recursion replay.
"""

from __future__ import annotations

from backend.services.context.workspace.tools._retry import with_429_backoff


class AzureOpenAILlmClient:
    """LlmClient backed by Azure OpenAI Chat Completions.

    Required constructor args (all resolved from env by resolve_root_model):
      - ``azure_endpoint``: Azure OpenAI resource URL, e.g.
        ``https://<resource>.openai.azure.com``.
      - ``azure_deployment``: deployment name (may differ from model name).

    ``api_version`` defaults to ``"2024-02-01"`` — the same default the rlm
    library's own AzureOpenAIClient uses.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        azure_endpoint: str,
        azure_deployment: str | None = None,
        api_version: str = "2024-02-01",
        max_tokens: int = 4096,
        timeout: float = 300.0,
    ) -> None:
        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
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
