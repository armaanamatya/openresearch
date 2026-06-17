"""Executor-tier resolver — run the code-writing agent (``implement_baseline``) on a
local Qwen via vLLM instead of Sonnet, to save Sonnet usage / rate-limit budget.

Mirrors :func:`backend.agents.rlm.accelerator.resolve_accelerator`: env-driven,
pluggable, health-probed with graceful fallback to the default Sonnet executor.

Selection (``OPENRESEARCH_EXECUTOR``):
  - ``sonnet`` / unset / ``off``      → None (default Claude/Sonnet executor)
  - ``qwen`` / ``local`` / ``vllm``   → OpenAI runtime on a local OpenAI-compatible
                                        endpoint (vLLM-served Qwen)
  - ``endpoint``                      → same, against ``OPENRESEARCH_EXECUTOR_BASE_URL``
  - ``azure`` / ``azure-openai`` /
    ``azure_openai`` / ``aoai``       → Azure OpenAI deployment
                                        (creds: ``AZURE_OPENAI_API_KEY``,
                                        ``AZURE_OPENAI_ENDPOINT``,
                                        ``AZURE_OPENAI_DEPLOYMENT``; no health-probe —
                                        Azure is a managed endpoint, credential presence
                                        is the gate; default stays Sonnet; EXPERIMENTAL)

Env: ``OPENRESEARCH_EXECUTOR_BASE_URL`` (default ``http://127.0.0.1:8001/v1``),
``OPENRESEARCH_EXECUTOR_MODEL`` (default ``Qwen/Qwen2.5-Coder-32B-Instruct``),
``OPENRESEARCH_EXECUTOR_API_KEY`` (default ``local``).
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
_DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
_DEFAULT_KEY = "local"
_DEFAULT_MODES = {"", "sonnet", "claude", "off", "default", "anthropic"}
_AZURE_MODES = {"azure", "azure-openai", "azure_openai", "aoai"}
_VLLM_MODES = {"qwen", "local", "vllm", "endpoint", "openai-endpoint", "on"}


@dataclass(frozen=True)
class ExecutorPlan:
    """Resolved executor runtime + model for ``implement_baseline``."""

    runtime: Any  # AgentRuntime
    model: str
    label: str


def _probe(base_url: str, api_key: str, timeout: float = 4.0) -> bool:
    """A served OpenAI-compatible endpoint answers GET /models. 200 ⇒ up; 401/403
    ⇒ up (port bound, auth-gated); connection error ⇒ down. Mirrors accelerator probe."""
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except urllib.error.HTTPError as exc:  # 401/403 = bound but auth-gated → up
        return exc.code in (401, 403)
    except Exception:  # noqa: BLE001 — connection refused / DNS / timeout = down
        return False


def resolve_executor() -> ExecutorPlan | None:
    """Resolve the executor tier from env. ``None`` ⇒ default Sonnet executor.

    Returns an :class:`ExecutorPlan` (OpenAI runtime → vLLM Qwen) when a non-default
    tier is selected AND the endpoint health-probes OK; otherwise falls back to the
    default (``None``) with a warning so a missing/booting vLLM never breaks the run.
    """
    mode = (os.environ.get("OPENRESEARCH_EXECUTOR") or "").strip().lower()
    if mode in _DEFAULT_MODES:
        return None

    if mode in _AZURE_MODES:
        from backend.agents.runtime.factory import _has_azure_openai_credentials
        if not _has_azure_openai_credentials():
            logger.warning(
                "OPENRESEARCH_EXECUTOR=%r but AZURE_OPENAI_API_KEY/AZURE_OPENAI_ENDPOINT missing"
                " — falling back to the default Sonnet executor",
                mode,
            )
            return None
        deployment = (os.environ.get("AZURE_OPENAI_DEPLOYMENT") or "").strip()
        if not deployment:
            logger.warning(
                "OPENRESEARCH_EXECUTOR=%r but AZURE_OPENAI_DEPLOYMENT is unset"
                " — falling back to the default Sonnet executor",
                mode,
            )
            return None
        from backend.agents.runtime.azure_openai_runtime import AzureOpenAiAgentRuntime
        runtime = AzureOpenAiAgentRuntime()
        logger.info("executor tier active: azure-openai → deployment=%s", deployment)
        return ExecutorPlan(runtime=runtime, model=deployment, label=f"azure:{deployment}")

    if mode not in _VLLM_MODES:
        logger.warning("unknown OPENRESEARCH_EXECUTOR=%r; using the default Sonnet executor", mode)
        return None

    base_url = (os.environ.get("OPENRESEARCH_EXECUTOR_BASE_URL") or _DEFAULT_BASE_URL).strip()
    api_key = (os.environ.get("OPENRESEARCH_EXECUTOR_API_KEY") or _DEFAULT_KEY).strip()
    model = (os.environ.get("OPENRESEARCH_EXECUTOR_MODEL") or _DEFAULT_MODEL).strip()

    if not _probe(base_url, api_key):
        logger.warning(
            "executor tier %r requested but %s did not health-probe — falling back to "
            "the default Sonnet executor (set OPENRESEARCH_EXECUTOR_BASE_URL or start vLLM)",
            mode, base_url,
        )
        return None

    from backend.agents.runtime.openai_runtime import OpenAiAgentRuntime

    runtime = OpenAiAgentRuntime(base_url=base_url, api_key=api_key, use_chat_completions=True)
    logger.info("executor tier active: %s → %s (model=%s)", mode, base_url, model)
    return ExecutorPlan(runtime=runtime, model=model, label=f"{mode}:{model}")


__all__ = ["ExecutorPlan", "resolve_executor"]
