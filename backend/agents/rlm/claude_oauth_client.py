"""rlm ``BaseLM`` subclass that drives the RLM root via Claude Agent SDK + OAuth.

The standard ``rlm.clients.anthropic.AnthropicClient`` requires a literal
``api_key`` constructor argument because it calls ``anthropic.Anthropic(api_key=...)``
directly — incompatible with Claude Code OAuth (no extractable bearer token).

``ClaudeOauthClient`` replaces that path: each ``completion()`` delegates to
``ClaudeLlmClient`` (rlm_query.py), which calls ``claude_agent_sdk.query()`` with
``ClaudeAgentOptions(tools=[], ...)`` — no tool use possible, so the model must
emit text on turn 1. This is correct for the RLM root, which is a pure
text-generation task. ``ClaudeLlmClient`` instances are cached per model to avoid
per-call ThreadPoolExecutor overhead.

Usage tracking is best-effort — the SDK does not return token counts, so per-call
input/output tokens are recorded as 0. The dominant cost signal comes from
``rlm``'s own ``usage_summary`` if a cost-tracking backend is used elsewhere;
for OAuth runs there is no per-token cost to track. See
``ROOT_MODELS["claude-oauth"]`` in ``backend/agents/rlm/models.py``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 1800.0  # 30 minutes per completion — bounded

# ---------------------------------------------------------------------------
# Module-level root-usage sink (C3)
# ---------------------------------------------------------------------------
# Accumulates per-model token counts for ALL ClaudeOauthClient.completion()
# calls made in this process since the last drain_root_usage() call.  Each
# completion() call increments this dict IN ADDITION to its per-instance
# counters so that run.py can ledger the root's cache tokens after
# rlm.completion() finishes — without having access to the ClaudeOauthClient
# instance (which is owned by the rlm library and not exposed).
#
# Thread safety: writes are protected by _ROOT_USAGE_LOCK; drain is atomic
# (swap + reset under the lock).
#
# Structure: {model_name: {calls, input_tokens, output_tokens,
#                           cache_creation_input_tokens, cache_read_input_tokens}}

import threading as _threading

_ROOT_USAGE_LOCK = _threading.Lock()
_ROOT_USAGE: dict[str, dict[str, int]] = {}


def drain_root_usage() -> dict[str, dict[str, int]]:
    """Return and clear the accumulated root-model usage since the last drain.

    Returns a dict of ``{model_name: {calls, input_tokens, output_tokens,
    cache_creation_input_tokens, cache_read_input_tokens}}``.  The returned
    snapshot is a fresh copy — the module-level accumulator is reset to empty.

    Safe to call from any thread; uses a lock to ensure the swap is atomic.
    """
    global _ROOT_USAGE
    with _ROOT_USAGE_LOCK:
        snapshot = _ROOT_USAGE
        _ROOT_USAGE = {}
    return snapshot


def _empty_root_turn_fallback() -> str:
    """A parseable no-op REPL turn for when the root SDK call returns empty.

    An EMPTY root completion ends the rlm loop: the library parses no ```repl
    block and no ``FINAL_VAR`` (``rlm/utils/parsing.py`` matches
    ``r"```repl\\s*\\n(.*?)\\n```"``), so it treats the turn as terminal and
    ships a partial report mid-reproduction — exactly the 2026-05-30
    ``prj_09047604e591d969`` iteration-1 death after the SDK aclose race
    exhausted ``complete()``'s retries. Returning a single no-op ```repl block
    instead keeps the loop alive for another iteration; it is still bounded by
    the max-iteration / wall-clock / forced-iteration policies. Opt out with
    ``REPROLAB_RLM_EMPTY_TURN_FALLBACK=0`` (then a true empty string is
    returned, restoring the pre-2026-05-30 behavior).
    """
    import os

    if os.environ.get("REPROLAB_RLM_EMPTY_TURN_FALLBACK", "1").strip().lower() in {
        "0",
        "false",
        "no",
    }:
        return ""
    return (
        "```repl\n"
        "# transient model-transport error — the previous turn was lost in the\n"
        "# SDK teardown race and could not be recovered after retries. Continuing\n"
        "# to the next iteration; re-issue your intended next step below.\n"
        "pass\n"
        "```"
    )


_CLI_DEFAULT_TIMEOUT_S = 600.0  # per-completion cap for the CLI subprocess


def _claude_cli_bin() -> str:
    """Resolve the ``claude`` CLI binary (override with ``REPROLAB_CLAUDE_CLI_BIN``)."""
    import os
    import shutil

    return (
        os.environ.get("REPROLAB_CLAUDE_CLI_BIN", "").strip()
        or shutil.which("claude")
        or "claude"
    )


def _root_transport() -> str:
    """RLM-root completion transport: ``cli`` (default) | ``sdk`` | ``auto``.

    ``cli``/``auto`` use the reliable ``claude`` CLI subprocess and fall back to
    the legacy claude-agent-sdk path only if the CLI is unavailable or errors.
    ``sdk`` forces the legacy path. Override with ``REPROLAB_RLM_ROOT_TRANSPORT``.
    """
    import os

    val = os.environ.get("REPROLAB_RLM_ROOT_TRANSPORT", "cli").strip().lower()
    return val if val in {"cli", "sdk", "auto"} else "cli"


def _cli_timeout_s() -> float:
    import os

    raw = os.environ.get("REPROLAB_RLM_CLI_TIMEOUT_S", "").strip()
    if not raw:
        return _CLI_DEFAULT_TIMEOUT_S
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _CLI_DEFAULT_TIMEOUT_S


# Tools disallowed for the root completion — the RLM root is a pure
# text-generation task (mirrors the SDK path's ``tools=[]`` contract). Belt-and-
# suspenders alongside the root system prompt's "emit a repl block" instruction.
_ROOT_DISALLOWED_TOOLS = (
    "Bash", "Read", "Write", "Edit", "Glob", "Grep",
    "WebFetch", "WebSearch", "Task", "NotebookEdit",
)


class ClaudeOauthClient(BaseLM):
    """rlm ``BaseLM`` that routes completions through Claude Agent SDK + OAuth.

    Unlike ``AnthropicClient``, this client requires NO ``api_key`` — auth is
    resolved by ``claude-agent-sdk`` from either ``ANTHROPIC_API_KEY`` or the
    Claude Code subscription's OAuth login. Each completion is thread-isolated
    inside ``ClaudeLlmClient.complete()`` (rlm_query.py) via a dedicated
    ThreadPoolExecutor, keeping the SDK's ``aclose()`` race in the worker
    thread. Sub-agent calls via ``collect_agent_text`` additionally route
    through ``backend.agents.runtime.sdk_isolation.run_isolated`` (PR-μ
    Solution A).

    Constructor args:
        model_name: Claude model to use (default ``claude-sonnet-4-6``).
        max_tokens: Honored only for ``BaseLM`` ABI compat — ignored by SDK.
        timeout_s: Per-completion wall-clock cap; default 1800s.
    """

    def __init__(
        self,
        model_name: str | None = "claude-sonnet-4-6",
        max_tokens: int = 32768,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        **kwargs: Any,
    ):
        # NOTE: do NOT call super().__init__(**kwargs) with api_key — BaseLM doesn't take it,
        # and the wrapper provides no api_key at all.
        super().__init__(model_name=model_name or "claude-sonnet-4-6", **kwargs)
        self.model_name = model_name or "claude-sonnet-4-6"
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        # Per-model usage tracking. The CLI transport (--output-format json)
        # returns REAL token counts incl. cache tokens, so OAuth-root usage is no
        # longer "best-effort" — capture all of input/output/cache so per-run
        # token logging and the cost calibration loop see the true root cost.
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)
        self.model_cache_creation_tokens: dict[str, int] = defaultdict(int)
        self.model_cache_read_tokens: dict[str, int] = defaultdict(int)
        # For get_last_usage compatibility
        self._last_model: str = self.model_name
        # CostLedgerEntry.from_usage-shaped dict for the MOST RECENT completion —
        # mirrors the workspace LlmClient contract so a per-turn ledgering hook
        # (see run.py) can record each root reasoning turn, not just primitives.
        self._last_usage: dict[str, int] = {}
        # Cached per-model ClaudeLlmClient instances — avoids per-call
        # ThreadPoolExecutor overhead and reuses the same SDK session per model.
        self._claude_clients: dict[str, Any] = {}

    def completion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        """Sync completion. The rlm root loop calls this synchronously.

        Delegates to ``ClaudeLlmClient`` (rlm_query.py) which calls
        ``claude_agent_sdk.query()`` with ``ClaudeAgentOptions(tools=[], ...)`` —
        no tool use possible, so the model must emit text on turn 1. This is
        correct for the RLM root, which is a pure text-generation task (the
        rdr-specific ``_run_sdk_in_thread`` path is for the tool-using
        ``baseline-implementation`` agent and is not appropriate here).

        ``ClaudeLlmClient`` is OAuth-capable (uses Claude Agent SDK, which
        resolves auth from API key OR the ``claude`` CLI's OAuth login) AND
        loop-safe (commit ``0c5fe4d`` added a running-loop guard).
        """
        text_prompt, system = self._render_prompt(prompt)
        resolved_model = model or self.model_name or "claude-sonnet-4-6"

        # PRIMARY transport: the `claude` CLI subprocess. It is a synchronous,
        # single-shot process with NONE of the claude-agent-sdk nested-async-
        # generator teardown race that empties ~80-90% of root completions on a
        # contended host (each SDK attempt spanned 3-4 threads + 2 event loops and
        # raced the SDK's background `_read_messages` task against loop teardown,
        # so the generator usually died before yielding any AssistantMessage).
        # Falls through to the SDK path only if the CLI is unavailable / errors.
        if _root_transport() in ("cli", "auto"):
            cli = self._cli_complete(
                system=system or "", user=text_prompt, model=resolved_model
            )
            if cli is not None:
                text, usage = cli
                in_tok = usage.get("input_tokens", 0)
                out_tok = usage.get("output_tokens", 0)
                cc_tok = usage.get("cache_creation_input_tokens", 0)
                cr_tok = usage.get("cache_read_input_tokens", 0)
                self.model_call_counts[resolved_model] += 1
                self.model_input_tokens[resolved_model] += in_tok
                self.model_output_tokens[resolved_model] += out_tok
                self.model_total_tokens[resolved_model] += in_tok + out_tok
                self.model_cache_creation_tokens[resolved_model] += cc_tok
                self.model_cache_read_tokens[resolved_model] += cr_tok
                self._last_model = resolved_model
                # CostLedgerEntry.from_usage-shaped — the per-turn ledger hook reads this.
                self._last_usage = {
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cache_creation_input_tokens": cc_tok,
                    "cache_read_input_tokens": cr_tok,
                    "reasoning_tokens": 0,
                }
                # Module-level sink: accumulate for post-completion ledgering in run.py
                # (C3 — allows drain_root_usage() to capture root cache tokens without
                # needing access to this instance).
                with _ROOT_USAGE_LOCK:
                    rec = _ROOT_USAGE.setdefault(
                        resolved_model,
                        {
                            "calls": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        },
                    )
                    rec["calls"] += 1
                    rec["input_tokens"] += in_tok
                    rec["output_tokens"] += out_tok
                    rec["cache_creation_input_tokens"] += cc_tok
                    rec["cache_read_input_tokens"] += cr_tok
                if (text or "").strip():
                    return text
                return _empty_root_turn_fallback()
            logger.warning(
                "ClaudeOauthClient.completion: CLI transport unavailable/failed — "
                "falling back to claude-agent-sdk for this call."
            )

        # FALLBACK / legacy: claude-agent-sdk path. Lazily build a per-model
        # ClaudeLlmClient and cache it — avoids per-call ThreadPoolExecutor
        # overhead. Retains the SDK aclose-retry resilience for environments
        # where the CLI binary is not present.
        client = self._claude_clients.get(resolved_model)
        if client is None:
            from backend.services.context.workspace.tools.rlm_query import (
                ClaudeLlmClient,
            )
            # max_turns=1 is correct here because tools=[] is enforced inside
            # ClaudeLlmClient — the model must emit text on turn 1.
            client = ClaudeLlmClient(model=resolved_model, max_turns=1)
            self._claude_clients[resolved_model] = client

        try:
            text = client.complete(system=system or "", user=text_prompt)
        except Exception as exc:
            logger.warning(
                "ClaudeOauthClient.completion: ClaudeLlmClient.complete failed — %s",
                exc,
            )
            raise

        # Update usage tracking (best-effort — no token counts from SDK).
        self.model_call_counts[resolved_model] += 1
        self._last_model = resolved_model

        # Premature-exit guard: complete() already retries the SDK aclose race;
        # if it STILL returns empty, emit a no-op REPL turn so the rlm root loop
        # survives to the next iteration instead of terminating the whole
        # reproduction on a transient transport error (see
        # _empty_root_turn_fallback).
        if not (text or "").strip():
            return _empty_root_turn_fallback()
        return text

    def _cli_complete(
        self, *, system: str, user: str, model: str
    ) -> tuple[str, dict[str, int]] | None:
        """Root completion via the ``claude`` CLI subprocess (reliable transport).

        Returns ``(text, usage)`` on success, or ``None`` to signal the caller to
        fall through to the legacy SDK path (CLI missing / non-zero exit / error
        result / unparseable output).

        The CLI is a synchronous, single-shot process: ``claude --print
        --output-format json`` emits one JSON object carrying ``result`` (the
        model's text) and ``usage`` (token counts). The user prompt is piped via
        STDIN — never argv — so an arbitrarily large REPL-state prompt cannot hit
        ``ARG_MAX``. Tools are disallowed so the root emits text on turn 1,
        mirroring the SDK path's ``tools=[]`` contract.
        """
        import json as _json
        import subprocess

        cmd = [
            _claude_cli_bin(),
            "--print",
            "--output-format", "json",
            "--model", model,
            "--disallowed-tools", *_ROOT_DISALLOWED_TOOLS,
        ]
        if system:
            cmd += ["--append-system-prompt", system]

        try:
            proc = subprocess.run(
                cmd,
                input=user,           # STDIN — ARG_MAX-safe for large prompts
                capture_output=True,
                text=True,
                timeout=_cli_timeout_s(),
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "ClaudeOauthClient._cli_complete: timed out after %.0fs",
                _cli_timeout_s(),
            )
            return None
        except (FileNotFoundError, OSError) as exc:
            logger.warning(
                "ClaudeOauthClient._cli_complete: claude CLI unavailable (%s)", exc
            )
            return None

        if proc.returncode != 0:
            logger.warning(
                "ClaudeOauthClient._cli_complete: exit=%d stderr=%.200s",
                proc.returncode,
                proc.stderr or "",
            )
            return None
        try:
            data = _json.loads(proc.stdout)
        except (ValueError, TypeError):
            logger.warning(
                "ClaudeOauthClient._cli_complete: non-JSON stdout (%.200s)",
                proc.stdout or "",
            )
            return None
        if not isinstance(data, dict) or data.get("is_error") or data.get("subtype") != "success":
            logger.warning(
                "ClaudeOauthClient._cli_complete: error result is_error=%s subtype=%s",
                (data or {}).get("is_error") if isinstance(data, dict) else "?",
                (data or {}).get("subtype") if isinstance(data, dict) else "?",
            )
            return None

        text = str(data.get("result") or "")
        usage_raw = data.get("usage") or {}
        usage = {
            "input_tokens": int(usage_raw.get("input_tokens", 0) or 0),
            "output_tokens": int(usage_raw.get("output_tokens", 0) or 0),
            "cache_creation_input_tokens": int(
                usage_raw.get("cache_creation_input_tokens", 0) or 0
            ),
            "cache_read_input_tokens": int(
                usage_raw.get("cache_read_input_tokens", 0) or 0
            ),
        }
        return text, usage

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        """Async path — rlm calls this for parallel sub-completions if any."""
        import asyncio
        return await asyncio.to_thread(self.completion, prompt, model)

    def get_usage_summary(self) -> UsageSummary:
        """Return best-effort usage summary (no token counts from SDK)."""
        model_summaries: dict[str, ModelUsageSummary] = {}
        for m, calls in self.model_call_counts.items():
            model_summaries[m] = ModelUsageSummary(
                total_calls=calls,
                total_input_tokens=self.model_input_tokens[m],
                total_output_tokens=self.model_output_tokens[m],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        """Return last call usage (no token counts from SDK)."""
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=0,
            total_output_tokens=0,
        )

    @staticmethod
    def _render_prompt(
        prompt: str | list[dict[str, Any]],
    ) -> tuple[str, str | None]:
        """Reduce list-of-messages prompts to (text, system) for the SDK."""
        if isinstance(prompt, str):
            return prompt, None
        if isinstance(prompt, list) and all(isinstance(m, dict) for m in prompt):
            system_parts: list[str] = []
            user_parts: list[str] = []
            for msg in prompt:
                role = msg.get("role")
                content = msg.get("content") or ""
                if role == "system":
                    system_parts.append(content)
                else:
                    user_parts.append(content)
            return "\n\n".join(user_parts), "\n\n".join(system_parts) or None
        raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")


__all__ = ["ClaudeOauthClient", "drain_root_usage"]
