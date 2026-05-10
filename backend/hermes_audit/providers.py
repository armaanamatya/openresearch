"""Audit provider implementations + stable Protocol.

Each provider is a thin sync function that takes the same prompt and
returns either a parseable JSON string or raises. The client picks one
based on learned memory; new providers plug in by registration, never
by editing the client's branching.
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
import subprocess
from typing import Any, Protocol

from backend.config import get_settings


# --------------------------------------------------------------------------- #
# Robust JSON extraction
# --------------------------------------------------------------------------- #

# Tried in order. First strategy that returns a dict wins. Each must
# either return a dict or raise — never silently substitute {}.
_PROSE_PREFIXES = (
    "here's the json",
    "here is the json",
    "json:",
    "result:",
    "audit report:",
    "the audit report",
    "below is the json",
)


def extract_audit_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction tolerant of common LLM output shapes.

    Strategies (first match wins, last error wins on total failure):
      1. Triple-backtick fenced JSON block (with or without ``json`` tag)
      2. First top-level ``{ ... }`` balanced span found in the text
      3. The whole text after stripping a chatty prose prefix
    """

    if not text:
        raise ValueError("empty response from audit provider")
    last_error: Exception | None = None

    # Strategy 1: fenced block
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return _coerce_dict(json.loads(fence_match.group(1)))
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

    # Strategy 2: balanced top-level braces
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        in_string = False
        escape = False
        for idx in range(brace_start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start : idx + 1]
                    try:
                        return _coerce_dict(json.loads(candidate))
                    except (json.JSONDecodeError, ValueError) as exc:
                        last_error = exc
                        break

    # Strategy 3: strip prose prefix and try again
    stripped = text.lstrip()
    lower = stripped.lower()
    for prefix in _PROSE_PREFIXES:
        if lower.startswith(prefix):
            tail = stripped[len(prefix) :].lstrip(" :\n")
            try:
                return _coerce_dict(json.loads(tail))
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                break

    raise ValueError(
        f"no parseable JSON in audit response (last error: {last_error or 'no candidates'})"
    )


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"expected JSON object, got {type(value).__name__}")


# --------------------------------------------------------------------------- #
# Provider Protocol + implementations
# --------------------------------------------------------------------------- #

class AuditProvider(Protocol):
    """A single auditor backend. Sync, raises on any failure."""

    name: str

    def is_available(self) -> bool:
        """Cheap precheck. Skip the provider if False."""
        ...

    def call(self, prompt: str) -> str:
        """Send the prompt and return the raw response text. Raise on failure."""
        ...


class NousHermesProvider:
    """Hermes Agent — supports both in-venv and out-of-venv installations.

    The Hermes Agent ships in two shapes:

    * **In-venv** — ``pip install hermes-agent`` into the current Python
      environment exposes a ``run_agent`` module. We call ``AIAgent``
      directly. Fastest (no subprocess), but requires a deliberate pip
      install into our venv.
    * **Out-of-venv (npm install)** — the npm wrapper drops a ``hermes``
      binary at ``~/.local/bin/hermes`` that execs Hermes's own
      bundled venv at ``~/.hermes/hermes-agent/venv``. ``run_agent`` is
      NOT importable from our Python; we shell out to the CLI's
      one-shot mode (``hermes -z <prompt> --ignore-rules
      --ignore-user-config``) and capture stdout.

    Detection precedence at ``is_available()``:

    1. Module is importable → use it (fast path).
    2. CLI is on ``$PATH`` → use it (subprocess fallback).
    3. Neither → unavailable.

    The CLI path passes ``--ignore-rules --ignore-user-config`` so the
    operator's local Hermes config / project rules don't leak into the
    audit prompt and contaminate the JSON output.
    """

    name = "nous_hermes"

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        *,
        cli_path: str | None = None,
        cli_timeout_seconds: float = 120.0,
    ) -> None:
        self.model = model
        self.cli_timeout_seconds = cli_timeout_seconds
        self._cli_override = cli_path

    # ----- backend selection ------------------------------------------------

    def _module_available(self) -> bool:
        try:
            importlib.import_module("run_agent")
            return True
        except ImportError:
            return False

    def _cli_path(self) -> str | None:
        if self._cli_override is not None:
            return self._cli_override or None
        return shutil.which("hermes")

    def is_available(self) -> bool:
        return self._module_available() or self._cli_path() is not None

    # ----- call --------------------------------------------------------------

    def call(self, prompt: str) -> str:
        if self._module_available():
            return self._call_via_module(prompt)
        cli = self._cli_path()
        if cli is not None:
            return self._call_via_cli(cli, prompt)
        raise RuntimeError(
            "Hermes Agent is unavailable: neither `run_agent` module nor "
            "`hermes` CLI is reachable. Install with `pip install hermes-agent` "
            "or `npm install -g hermes-agent`."
        )

    def _call_via_module(self, prompt: str) -> str:
        module = importlib.import_module("run_agent")
        agent_cls = getattr(module, "AIAgent")
        agent = agent_cls(
            model=self.model,
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
        raise RuntimeError("unsupported Nous Hermes runtime interface")

    def _call_via_cli(self, cli_path: str, prompt: str) -> str:
        result = subprocess.run(
            [
                cli_path,
                "-z",
                prompt,
                "--ignore-rules",
                "--ignore-user-config",
            ],
            capture_output=True,
            text=True,
            timeout=self.cli_timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            stderr_excerpt = (result.stderr or "").strip()[:500]
            raise RuntimeError(
                f"hermes CLI exited {result.returncode}: {stderr_excerpt}"
            )
        output = (result.stdout or "").strip()
        if not output:
            raise RuntimeError("hermes CLI returned empty stdout")
        return output


class ClaudeAuditProvider:
    """Direct Anthropic SDK call — bypasses our agent runtime so we don't
    need an event loop or tool plumbing for a one-shot JSON request.

    The API key is sourced from ``Settings.anthropic_api_key`` (which
    pydantic-settings loads from ``.env``), NOT ``os.environ``. This is
    deliberate: a previous version read os.environ directly and was
    silently skipped whenever the parent process never sourced .env
    (Lab UI spawn, pytest from a clean shell, …). Settings is the
    single source of truth — it reads disk on construction regardless
    of os.environ state.
    """

    name = "claude"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 2000,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key_override = api_key

    def _resolve_api_key(self) -> str:
        if self._api_key_override is not None:
            return self._api_key_override
        return get_settings().anthropic_api_key

    def is_available(self) -> bool:
        if not self._resolve_api_key():
            return False
        try:
            importlib.import_module("anthropic")
            return True
        except ImportError:
            return False

    def call(self, prompt: str) -> str:
        anthropic = importlib.import_module("anthropic")
        client = anthropic.Anthropic(api_key=self._resolve_api_key())
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # Concatenate all text blocks; ignore non-text (Hermes audits don't
        # use tools so this should always be one text block in practice).
        parts = [getattr(b, "text", "") for b in response.content]
        return "".join(parts)


class OpenAIAuditProvider:
    """Direct OpenAI SDK call — same shape as ClaudeAuditProvider.

    See ``ClaudeAuditProvider`` for why the key is resolved through
    ``Settings`` rather than ``os.environ``.
    """

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 2000,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key_override = api_key

    def _resolve_api_key(self) -> str:
        if self._api_key_override is not None:
            return self._api_key_override
        return get_settings().openai_api_key

    def is_available(self) -> bool:
        if not self._resolve_api_key():
            return False
        try:
            importlib.import_module("openai")
            return True
        except ImportError:
            return False

    def call(self, prompt: str) -> str:
        openai_mod = importlib.import_module("openai")
        client = openai_mod.OpenAI(api_key=self._resolve_api_key())
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


class ClaudeCodeSdkProvider:
    """Claude via the ``claude_agent_sdk`` package.

    Why this exists alongside ``ClaudeAuditProvider``:

    * ``ClaudeAuditProvider`` requires an Anthropic *API* key (paid
      per-token, billed to your Anthropic console).
    * ``ClaudeCodeSdkProvider`` uses ``claude_agent_sdk.query``, which
      authenticates via your Claude Code session — i.e. the same auth
      backing ``claude`` in your terminal. If you're on Claude Pro / Max,
      the audit is included in your subscription rather than charged
      per-token.

    The SDK is async (returns an async iterator of messages); this
    provider hides that behind the sync ``call()`` Protocol contract by
    running the consumer in either ``asyncio.run`` (no loop) or a
    short-lived thread-pool worker (when the caller is itself inside a
    running loop, e.g. ``HermesAuditService`` invoked from FastAPI).

    Availability requires the ``claude-agent-sdk`` package (already a
    project dependency); no env-var check, since the SDK reads its own
    auth context.
    """

    name = "claude_code_sdk"

    def __init__(self, *, max_turns: int = 1, timeout_seconds: float = 120.0) -> None:
        self.max_turns = max_turns
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        try:
            importlib.import_module("claude_agent_sdk")
            return True
        except ImportError:
            return False

    def call(self, prompt: str) -> str:
        import asyncio

        sdk = importlib.import_module("claude_agent_sdk")
        ClaudeAgentOptions = sdk.ClaudeAgentOptions
        ResultMessage = sdk.ResultMessage
        query = sdk.query

        options = ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            max_turns=self.max_turns,
        )

        async def _collect() -> str:
            chunks: list[str] = []
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    text = getattr(message, "text", "")
                    if text:
                        chunks.append(str(text))
                else:
                    for block in getattr(message, "content", []) or []:
                        text = getattr(block, "text", "")
                        if text:
                            chunks.append(str(text))
            return "\n".join(chunks)

        # Two cases: caller is sync (no loop) → asyncio.run; caller is
        # already inside an event loop (e.g. FastAPI request) → run in a
        # short-lived thread that owns its own loop. The thread bound is
        # 1 because each audit is a single short LLM call.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running and running.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(lambda: asyncio.run(_collect()))
                result = future.result(timeout=self.timeout_seconds)
        else:
            result = asyncio.run(_collect())

        if not result or not result.strip():
            raise RuntimeError("claude_agent_sdk returned empty response")
        return result


__all__ = [
    "AuditProvider",
    "ClaudeAuditProvider",
    "ClaudeCodeSdkProvider",
    "NousHermesProvider",
    "OpenAIAuditProvider",
    "extract_audit_json",
]
