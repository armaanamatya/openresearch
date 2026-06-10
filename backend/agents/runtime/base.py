"""Provider-agnostic agent runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Protocol
from urllib.parse import urlparse


ProviderName = Literal["anthropic", "openai"]


class RuntimeGuardViolation(RuntimeError):
    """Raised when a provider tool call violates runtime guardrails."""


class AgentLimitExceeded(RuntimeError):
    """Raised when an agent invocation hits a configured budget cap.

    Carries enough structured info that the orchestrator (or a UI) can
    decide how to react — fail loudly, retry with a bumped budget, or
    continue with the partial output. ``kind`` is one of:

      * ``"turns"``       — provider SDK returned its turn-cap error
      * ``"tool_calls"``  — orchestrator-side counter exceeded the cap
      * ``"wall_clock"``  — agent invocation exceeded agent_wall_clock_seconds
    """

    def __init__(
        self,
        *,
        agent_id: str,
        kind: Literal["turns", "tool_calls", "wall_clock"],
        limit_value: int,
        elapsed_seconds: float,
        partial_output: str = "",
    ) -> None:
        self.agent_id = agent_id
        self.kind = kind
        self.limit_value = limit_value
        self.elapsed_seconds = elapsed_seconds
        self.partial_output = partial_output
        super().__init__(
            f"Agent {agent_id!r} hit {kind} cap of {limit_value} after "
            f"{elapsed_seconds:.1f}s "
            f"({len(partial_output)} chars of partial output preserved)"
        )


@dataclass(frozen=True)
class RuntimeGuard:
    """Runtime policy shared across provider adapters."""

    blocked_terms: tuple[str, ...] = ()
    max_tool_calls: int | None = None

    def normalized_blocked_terms(self) -> tuple[str, ...]:
        terms: list[str] = []
        for term in self.blocked_terms:
            raw = term.strip()
            if not raw:
                continue
            lowered = raw.lower()
            if lowered not in terms:
                terms.append(lowered)
            canonical = _canonicalize_url_term(raw)
            if canonical and canonical not in terms:
                terms.append(canonical)
        return tuple(terms)

    def find_blocked_term(self, text: str) -> str | None:
        if not text:
            return None
        haystack = text.lower()
        for term in self.normalized_blocked_terms():
            if term and term in haystack:
                return term
        return None

    def raise_if_blocked(self, text: str, surface: str) -> None:
        blocked = self.find_blocked_term(text)
        if blocked is not None:
            raise RuntimeGuardViolation(
                f"{surface} references blocked PaperBench resource: {blocked}"
            )


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentRuntimeSpec:
    name: str
    instructions: str
    model: str
    description: str = ""
    tools: tuple[ToolSpec, ...] = ()
    sub_agents: tuple["AgentRuntimeSpec", ...] = ()
    max_turns: int | None = None
    thinking_budget_tokens: int | None = None
    cache_static_blocks: bool = True
    permission_mode: str = "bypassPermissions"
    working_directory: Path | None = None
    guard: RuntimeGuard = field(default_factory=RuntimeGuard)


@dataclass(frozen=True)
class StreamText:
    text: str


@dataclass(frozen=True)
class StreamToolCall:
    tool_id: str
    tool_name: str
    tool_input: dict[str, Any]


@dataclass(frozen=True)
class StreamUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


StreamEvent = StreamText | StreamToolCall | StreamUsage


class ProviderConfigurationError(RuntimeError):
    def __init__(self, *, provider: str, reason: str) -> None:
        super().__init__(f"Provider {provider!r} is not configured: {reason}")
        self.provider = provider
        self.reason = reason


class ProviderFeatureUnsupported(RuntimeError):
    def __init__(self, *, provider: str, feature_name: str) -> None:
        super().__init__(f"Provider {provider!r} does not support {feature_name!r}")
        self.provider = provider
        self.feature_name = feature_name


class AgentRuntime(Protocol):
    @property
    def provider_name(self) -> ProviderName: ...

    async def run_agent(
        self,
        *,
        agent: AgentRuntimeSpec,
        user_input: str,
    ) -> AsyncIterator[StreamEvent]: ...


__all__ = [
    "AgentLimitExceeded",
    "AgentRuntime",
    "AgentRuntimeSpec",
    "blocked_terms_from_env",
    "ProviderConfigurationError",
    "ProviderFeatureUnsupported",
    "ProviderName",
    "RuntimeGuard",
    "RuntimeGuardViolation",
    "StreamEvent",
    "StreamText",
    "StreamToolCall",
    "StreamUsage",
    "ToolSpec",
]


def _canonicalize_url_term(value: str) -> str:
    """Canonicalize a configured blocked term that is *expected* to be URL-like.

    Returns a lowercase ``host/path`` form so that
    ``https://github.com/foo/bar.git`` and ``GITHUB.COM/foo/bar`` match the
    same way. Never raises: free-form input that is not a URL falls back to
    the lowercased, stripped string.
    """

    text = value.lower().strip()
    if not text:
        return ""
    text = text.replace("http://", "").replace("https://", "")
    candidate = text if "://" in text else "https://" + text
    try:
        parsed = urlparse(candidate)
    except ValueError:
        # Stray brackets, unbalanced quotes, etc. The raw lowercased form is
        # still useful for substring matching, so return it unchanged.
        return text
    if parsed.netloc:
        path = parsed.path.rstrip("/")
        text = f"{parsed.netloc}{path}".lower()
    if text.endswith(".git"):
        text = text[:-4]
    return text


# Public alias: callers should prefer the explicit name.
_normalize_guard_text = _canonicalize_url_term


def blocked_terms_from_env(env_var: str = "REPROLAB_BLOCKED_TERMS_JSON") -> tuple[str, ...]:
    """Parse the #7 benchmark-integrity blocklist from its JSON-list env var.

    The single subprocess seam: ``cli.py`` unions the curated sources (bundle
    ``blacklist.txt`` + ``--blacklist`` + ``paper_hints``) and sets the env var;
    both ``RunContext`` and ``collect_agent_text`` read it through this one
    parser so the ``RuntimeGuard`` is seeded identically everywhere. Returns a
    tuple of non-empty, stripped terms. Never raises — a malformed value degrades
    to an empty blocklist so a parse error can never crash a run.
    """
    import json as _json
    import os as _os

    raw = _os.environ.get(env_var, "").strip()
    if not raw:
        return ()
    try:
        parsed = _json.loads(raw)
    except Exception:  # noqa: BLE001 — env-var parse failure must never crash a run
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(term).strip() for term in parsed if str(term).strip())
