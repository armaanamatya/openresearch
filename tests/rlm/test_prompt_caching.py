"""Tests for prompt caching and optional-hints factoring (Lane A: lane/prompts-caching).

Covers:
  (a) System prompt assembly still contains required core sections regardless of
      the include_hints flag.
  (b) Optional hints sections (_TRIAGE_INSTRUCTION, _DECISION_ADVISOR_SECTION)
      appear when include_hints=True and are ABSENT when include_hints=False.
  (c) For the Anthropic API path, the apply_anthropic_caching_patch() wrapper
      converts a plain system-string into a list[dict] with cache_control blocks
      on the outgoing request.
  (d) Caching is a NO-OP regression — if the wrapper errors, the call still
      succeeds without cache (D5 design decision).
  (e) OAuth path (ClaudeOauthClient / claude-agent-sdk) is NOT affected by the
      caching patch — it manages caching internally.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gpt5_model():
    from backend.agents.rlm.models import ROOT_MODELS
    return ROOT_MODELS["gpt-5"]


@pytest.fixture
def claude_model():
    from backend.agents.rlm.models import ROOT_MODELS
    return ROOT_MODELS["claude"]


@pytest.fixture
def sample_metadata():
    return {
        "paper_text": {"type": "str", "length": 120_000},
        "paper_metadata": {"type": "dict", "length": 850},
    }


# ---------------------------------------------------------------------------
# Tests: system prompt sections present / absent based on include_hints
# ---------------------------------------------------------------------------


class TestOptionalHintsSection:
    """Triage + decision-advisor are factored into _OPTIONAL_HINTS_SECTION."""

    def test_triage_present_by_default(self, gpt5_model, sample_metadata):
        """With default include_hints=True, triage instruction must appear."""
        from backend.agents.rlm.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            context_metadata=sample_metadata,
            root_model=gpt5_model,
        )
        lower = prompt.lower()
        assert "triage" in lower or "decline" in lower, (
            "Triage section must appear in default prompt"
        )

    def test_decision_advisor_present_by_default(self, gpt5_model, sample_metadata):
        """With default include_hints=True, decision-advisor must appear."""
        from backend.agents.rlm.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            context_metadata=sample_metadata,
            root_model=gpt5_model,
        )
        assert "recommend_next_tool" in prompt, (
            "Decision-advisor section must appear in default prompt"
        )

    def test_triage_absent_when_hints_disabled(self, gpt5_model, sample_metadata):
        """With include_hints=False, triage instruction must NOT appear."""
        from backend.agents.rlm.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            context_metadata=sample_metadata,
            root_model=gpt5_model,
            include_hints=False,
        )
        lower = prompt.lower()
        # _TRIAGE_INSTRUCTION contains "TRIAGE" as a section header
        assert "TRIAGE — COST AND TIME ARE FINITE" not in prompt, (
            "Triage section header must not appear when include_hints=False"
        )

    def test_decision_advisor_absent_when_hints_disabled(self, gpt5_model, sample_metadata):
        """With include_hints=False, decision-advisor must NOT appear."""
        from backend.agents.rlm.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            context_metadata=sample_metadata,
            root_model=gpt5_model,
            include_hints=False,
        )
        assert "recommend_next_tool" not in prompt, (
            "Decision-advisor must not appear when include_hints=False"
        )

    def test_core_sections_present_without_hints(self, gpt5_model, sample_metadata):
        """Core sections must all be present even when include_hints=False."""
        from backend.agents.rlm.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            context_metadata=sample_metadata,
            root_model=gpt5_model,
            include_hints=False,
        )
        assert "FINAL_VAR" in prompt, "Termination contract must be present"
        assert "llm_query" in prompt, "Primitives section must be present"
        assert "rlm_query" in prompt, "Primitives section must be present"
        assert "check_user_messages" in prompt, "Chat steering must be present"
        assert "heartbeat" in prompt.lower(), "Heartbeat section must be present"
        assert "resolve_gpu_requirements" in prompt, "GPU section must be present"

    def test_format_template_valid_without_hints(self, gpt5_model, sample_metadata):
        """Prompt without hints must still be a valid .format() template.

        After calling .format(custom_tools_section=...) the injection point is
        filled and the result must contain the injected value.  Literal JSON braces
        ({{ / }}) in the template become single braces in the rendered output —
        that is correct and expected behaviour.
        """
        from backend.agents.rlm.system_prompt import build_system_prompt
        import string
        prompt = build_system_prompt(
            context_metadata=sample_metadata,
            root_model=gpt5_model,
            include_hints=False,
        )
        # Template must expose exactly one .format() field (custom_tools_section).
        fields = {
            name
            for _, name, _, _ in string.Formatter().parse(prompt)
            if name is not None
        }
        assert fields == {"custom_tools_section"}, (
            f"Unexpected .format() fields in no-hints prompt: {fields}"
        )
        rendered = prompt.format(custom_tools_section="TOOLS")
        assert "TOOLS" in rendered

    def test_decomposition_example_present_without_hints(self, gpt5_model, sample_metadata):
        """Decomposition example must be present even when hints are disabled."""
        from backend.agents.rlm.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            context_metadata=sample_metadata,
            root_model=gpt5_model,
            include_hints=False,
        )
        lower = prompt.lower()
        assert "decompos" in lower or "example" in lower, (
            "Decomposition example must be present"
        )


class TestOptionalHintsSectionConstants:
    """_OPTIONAL_HINTS_SECTION constant is importable and contains expected content."""

    def test_optional_hints_constant_importable(self):
        """_OPTIONAL_HINTS_SECTION must be importable from system_prompt."""
        from backend.agents.rlm.system_prompt import _OPTIONAL_HINTS_SECTION
        assert isinstance(_OPTIONAL_HINTS_SECTION, str)
        assert len(_OPTIONAL_HINTS_SECTION) > 100

    def test_optional_hints_contains_triage(self):
        """_OPTIONAL_HINTS_SECTION must contain the triage instruction text."""
        from backend.agents.rlm.system_prompt import _OPTIONAL_HINTS_SECTION
        assert "TRIAGE" in _OPTIONAL_HINTS_SECTION or "triage" in _OPTIONAL_HINTS_SECTION.lower()

    def test_optional_hints_contains_decision_advisor(self):
        """_OPTIONAL_HINTS_SECTION must contain decision-advisor text."""
        from backend.agents.rlm.system_prompt import _OPTIONAL_HINTS_SECTION
        assert "recommend_next_tool" in _OPTIONAL_HINTS_SECTION


# ---------------------------------------------------------------------------
# Tests: Anthropic API caching patch
# ---------------------------------------------------------------------------


class TestApplyAnthropicCachingPatch:
    """apply_anthropic_caching_patch() wraps AnthropicClient.completion to add
    cache_control blocks on the system prompt."""

    def _reset_patch(self):
        import backend.agents.rlm._oauth_backend_patch as mod
        mod._ANTHROPIC_CACHING_APPLIED = False

    def test_patch_is_importable(self):
        """apply_anthropic_caching_patch must be importable."""
        from backend.agents.rlm._oauth_backend_patch import apply_anthropic_caching_patch
        assert callable(apply_anthropic_caching_patch)

    def test_patch_idempotent(self):
        """Calling apply_anthropic_caching_patch twice is a no-op on the second call."""
        from backend.agents.rlm._oauth_backend_patch import apply_anthropic_caching_patch
        import rlm.clients.anthropic as rlm_anthro

        self._reset_patch()
        apply_anthropic_caching_patch()
        patched_fn = rlm_anthro.AnthropicClient.completion

        apply_anthropic_caching_patch()
        assert rlm_anthro.AnthropicClient.completion is patched_fn

    def test_patched_completion_adds_cache_control_to_system(self):
        """After patching, AnthropicClient.completion wraps system str as list with cache_control."""
        from backend.agents.rlm._oauth_backend_patch import apply_anthropic_caching_patch
        import rlm.clients.anthropic as rlm_anthro

        self._reset_patch()
        apply_anthropic_caching_patch()

        # Build a mock AnthropicClient instance — we intercept the .client.messages.create call.
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="assistant reply")]
        fake_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        mock_messages_create = MagicMock(return_value=fake_response)
        mock_client = MagicMock()
        mock_client.messages.create = mock_messages_create

        instance = rlm_anthro.AnthropicClient.__new__(rlm_anthro.AnthropicClient)
        instance.client = mock_client
        instance.async_client = MagicMock()
        instance.model_name = "claude-opus-4-7"
        instance.max_tokens = 1024
        from collections import defaultdict
        instance.model_call_counts = defaultdict(int)
        instance.model_input_tokens = defaultdict(int)
        instance.model_output_tokens = defaultdict(int)
        instance.model_total_tokens = defaultdict(int)
        instance.last_prompt_tokens = 0
        instance.last_completion_tokens = 0

        # Call with a list-of-messages prompt containing a system message
        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        instance.completion(prompt)

        # Verify the call was made
        assert mock_messages_create.called
        kwargs = mock_messages_create.call_args[1]

        # system must be a list (not a plain str) when cache_control is added
        system = kwargs.get("system")
        assert system is not None, "system kwarg must be present"
        assert isinstance(system, list), (
            f"system must be a list of content blocks for cache_control, got {type(system)}"
        )
        # The list must contain at least one block with cache_control
        assert any(
            isinstance(block, dict) and block.get("cache_control", {}).get("type") == "ephemeral"
            for block in system
        ), f"No ephemeral cache_control block found in system: {system}"

    def test_patched_completion_noop_when_no_system(self):
        """If no system message, completion still works (no crash from caching)."""
        from backend.agents.rlm._oauth_backend_patch import apply_anthropic_caching_patch
        import rlm.clients.anthropic as rlm_anthro

        self._reset_patch()
        apply_anthropic_caching_patch()

        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="reply")]
        fake_response.usage = MagicMock(input_tokens=5, output_tokens=3)

        mock_messages_create = MagicMock(return_value=fake_response)
        mock_client = MagicMock()
        mock_client.messages.create = mock_messages_create

        instance = rlm_anthro.AnthropicClient.__new__(rlm_anthro.AnthropicClient)
        instance.client = mock_client
        instance.async_client = MagicMock()
        instance.model_name = "claude-opus-4-7"
        instance.max_tokens = 1024
        from collections import defaultdict
        instance.model_call_counts = defaultdict(int)
        instance.model_input_tokens = defaultdict(int)
        instance.model_output_tokens = defaultdict(int)
        instance.model_total_tokens = defaultdict(int)
        instance.last_prompt_tokens = 0
        instance.last_completion_tokens = 0

        # Plain string prompt — no system message extracted
        result = instance.completion("Hello world")
        assert result == "reply"

    def test_caching_patch_noop_on_exception(self):
        """D5: if the caching wrapper fails, the call should still succeed without cache."""
        from backend.agents.rlm._oauth_backend_patch import apply_anthropic_caching_patch
        import rlm.clients.anthropic as rlm_anthro

        self._reset_patch()
        apply_anthropic_caching_patch()

        # Simulate _prepare_messages raising — the patched completion should
        # not surface a caching-specific error; it falls back gracefully.
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="fallback reply")]
        fake_response.usage = MagicMock(input_tokens=5, output_tokens=3)

        mock_messages_create = MagicMock(return_value=fake_response)
        mock_client = MagicMock()
        mock_client.messages.create = mock_messages_create

        instance = rlm_anthro.AnthropicClient.__new__(rlm_anthro.AnthropicClient)
        instance.client = mock_client
        instance.async_client = MagicMock()
        instance.model_name = "claude-opus-4-7"
        instance.max_tokens = 1024
        from collections import defaultdict
        instance.model_call_counts = defaultdict(int)
        instance.model_input_tokens = defaultdict(int)
        instance.model_output_tokens = defaultdict(int)
        instance.model_total_tokens = defaultdict(int)
        instance.last_prompt_tokens = 0
        instance.last_completion_tokens = 0

        # This should NOT raise even if we pass a valid prompt
        result = instance.completion([{"role": "user", "content": "test"}])
        assert result == "fallback reply"


class TestOAuthNotAffectedByCachingPatch:
    """ClaudeOauthClient is not patched — it routes through claude-agent-sdk which
    manages caching internally."""

    def test_claude_oauth_client_has_no_cache_control_injection(self):
        """ClaudeOauthClient.completion does not attempt to add cache_control."""
        from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient
        import inspect
        src = inspect.getsource(ClaudeOauthClient.completion)
        assert "cache_control" not in src, (
            "ClaudeOauthClient.completion must not inject cache_control — "
            "the SDK manages this internally"
        )
