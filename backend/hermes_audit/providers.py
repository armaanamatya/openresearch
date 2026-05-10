"""Audit provider implementations + stable Protocol.

Each provider is a thin sync function that takes the same prompt and
returns either a parseable JSON string or raises. The client picks one
based on learned memory; new providers plug in by registration, never
by editing the client's branching.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from typing import Any, Protocol


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
    """Wraps the official Nous Hermes Python runtime via importlib."""

    name = "nous_hermes"

    def __init__(self, model: str = "anthropic/claude-sonnet-4") -> None:
        self.model = model

    def is_available(self) -> bool:
        try:
            importlib.import_module("run_agent")
            return True
        except ImportError:
            return False

    def call(self, prompt: str) -> str:
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


class ClaudeAuditProvider:
    """Direct Anthropic SDK call — bypasses our agent runtime so we don't
    need an event loop or tool plumbing for a one-shot JSON request."""

    name = "claude"

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 2000) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def is_available(self) -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            importlib.import_module("anthropic")
            return True
        except ImportError:
            return False

    def call(self, prompt: str) -> str:
        anthropic = importlib.import_module("anthropic")
        client = anthropic.Anthropic()
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
    """Direct OpenAI SDK call — same shape as ClaudeAuditProvider."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 2000) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def is_available(self) -> bool:
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        try:
            importlib.import_module("openai")
            return True
        except ImportError:
            return False

    def call(self, prompt: str) -> str:
        openai_mod = importlib.import_module("openai")
        client = openai_mod.OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


__all__ = [
    "AuditProvider",
    "ClaudeAuditProvider",
    "NousHermesProvider",
    "OpenAIAuditProvider",
    "extract_audit_json",
]
