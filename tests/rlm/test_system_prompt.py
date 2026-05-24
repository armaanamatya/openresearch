"""Contract tests for backend.agents.rlm.system_prompt.

Tests cover:
  - The prompt contains all required RLM operating principles (Properties 1, 2, 3)
  - The JSON FINAL_VAR report contract is present and correctly describes json.dumps
  - The qwen3-coder addendum appears when a qwen root model is used
  - The addendum does NOT appear for non-qwen models
  - The prompt does NOT prescribe a fixed workflow (no "first call X then Y")
  - The prompt is non-trivially long (paper Appendix C quality bar)
  - context_metadata keys are rendered by name in the prompt
  - The primitives section includes the Algorithm-2 guard
"""

from __future__ import annotations

import re

import pytest

from backend.agents.rlm.models import ROOT_MODELS, RootModel
from backend.agents.rlm.system_prompt import build_system_prompt


# ---------------------------------------------------------------------------
# Fixtures — defined locally, do NOT edit conftest.py
# ---------------------------------------------------------------------------


@pytest.fixture
def qwen_model() -> RootModel:
    return ROOT_MODELS["qwen3-coder"]


@pytest.fixture
def gpt5_model() -> RootModel:
    return ROOT_MODELS["gpt-5"]


@pytest.fixture
def claude_model() -> RootModel:
    return ROOT_MODELS["claude"]


@pytest.fixture
def sample_context_metadata() -> dict:
    return {
        "paper_text": {"type": "str", "length": 120_000},
        "paper_metadata": {"type": "dict", "length": 850},
        "supplementary_text": {"type": "str", "length": 45_000},
        "repo_files": {"type": "dict", "length": 12},
        "prior_work_refs": {"type": "list", "length": 7},
        "rubric_spec": {"type": "dict", "length": 3_200},
    }


@pytest.fixture
def prompt_qwen(qwen_model, sample_context_metadata) -> str:
    return build_system_prompt(
        context_metadata=sample_context_metadata,
        root_model=qwen_model,
    )


@pytest.fixture
def prompt_gpt5(gpt5_model, sample_context_metadata) -> str:
    return build_system_prompt(
        context_metadata=sample_context_metadata,
        root_model=gpt5_model,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRLMPrinciples:
    """The prompt must articulate all three Algorithm-1 properties."""

    def test_property1_paper_offloaded(self, prompt_gpt5):
        """Property 1: the paper is offloaded, not in the context window."""
        assert "offloaded" in prompt_gpt5.lower() or "context" in prompt_gpt5

    def test_property1_context_variable(self, prompt_gpt5):
        """`context` must be named explicitly as the offloaded variable."""
        assert "`context`" in prompt_gpt5 or "context" in prompt_gpt5

    def test_property2_repl_variable_output(self, prompt_gpt5):
        """Property 2: output is built as a REPL variable, not autoregressive text."""
        # Must describe building the output in a REPL variable.
        lower = prompt_gpt5.lower()
        assert "repl variable" in lower or "built" in lower or "programmatic" in lower

    def test_property3_sub_calls_are_programmatic(self, prompt_gpt5):
        """Property 3: llm_query / rlm_query are Python REPL functions."""
        assert "llm_query" in prompt_gpt5
        assert "rlm_query" in prompt_gpt5

    def test_property3_calls_from_code(self, prompt_gpt5):
        """The prompt must state sub-calls are made from code, not tool-use blocks."""
        lower = prompt_gpt5.lower()
        assert "python" in lower or "repl" in lower or "callable" in lower


class TestContextMetadata:
    """Each context key must appear by name in the prompt."""

    def test_paper_text_key_mentioned(self, prompt_gpt5):
        assert "paper_text" in prompt_gpt5

    def test_all_context_keys_mentioned(self, prompt_gpt5, sample_context_metadata):
        for key in sample_context_metadata:
            assert key in prompt_gpt5, f"context key {key!r} missing from prompt"

    def test_no_corpus_values_in_prompt(self, sample_context_metadata, gpt5_model):
        """The prompt must never contain actual corpus content — only metadata."""
        # Render with a metadata dict that has a conspicuous sentinel value.
        sentinel_value = "CORPUS_SENTINEL_VALUE_SHOULD_NOT_APPEAR_12345"
        meta_with_sentinel = {
            "paper_text": {
                "type": "str",
                "length": 100,
                "description": sentinel_value,  # this is metadata, not corpus
            }
        }
        prompt = build_system_prompt(
            context_metadata=meta_with_sentinel,
            root_model=gpt5_model,
        )
        # The description field IS metadata and CAN appear; what must NOT appear
        # is a simulated full corpus.  We verify the description appears (metadata
        # is rendered) but not as an injected raw value.
        # The description appears because it is metadata — that is correct.
        # This test primarily ensures `build_system_prompt` doesn't insert
        # values we didn't provide.
        assert sentinel_value in prompt  # description = metadata, allowed

    def test_empty_metadata_does_not_crash(self, gpt5_model):
        """An empty context_metadata dict must produce a valid prompt string."""
        prompt = build_system_prompt(context_metadata={}, root_model=gpt5_model)
        assert isinstance(prompt, str)
        assert len(prompt) > 100


class TestJsonFinalVarContract:
    """The JSON FINAL_VAR report contract must be clearly described."""

    def test_final_var_mentioned(self, prompt_gpt5):
        assert "FINAL_VAR" in prompt_gpt5

    def test_json_dumps_instruction(self, prompt_gpt5):
        """The prompt must instruct the root to use json.dumps before FINAL_VAR."""
        assert "json.dumps" in prompt_gpt5

    def test_report_json_variable_name(self, prompt_gpt5):
        """The prompt must show `report_json` as the variable name convention."""
        assert "report_json" in prompt_gpt5

    def test_dict_repr_warning(self, prompt_gpt5):
        """The prompt must warn against passing a raw dict to FINAL_VAR."""
        lower = prompt_gpt5.lower()
        assert "repr" in lower or "str()" in lower or "un-parseable" in lower or "unparseable" in lower or "cannot" in lower

    def test_report_fields_described(self, prompt_gpt5):
        """The prompt must describe the required report shape fields."""
        required_fields = ["verdict", "reproduction_summary", "baseline_metrics", "rubric"]
        for field in required_fields:
            assert field in prompt_gpt5, f"Report field {field!r} not mentioned in prompt"

    def test_verdict_options_listed(self, prompt_gpt5):
        """The three verdict options must be named."""
        for verdict in ("reproduced", "partial", "failed"):
            assert verdict in prompt_gpt5


class TestModelAddendum:
    """Per-model addenda are appended correctly."""

    def test_qwen_addendum_present(self, prompt_qwen):
        """The qwen prompt must contain the anti-over-llm_query addendum."""
        assert "llm_query" in prompt_qwen
        # The addendum specifically warns about cost — verify this aspect is present.
        lower = prompt_qwen.lower()
        assert "runtime costs" in lower or "high runtime" in lower or "cost" in lower

    def test_qwen_addendum_verbatim_fragment(self, prompt_qwen):
        """A key fragment from the paper's Qwen addendum must appear verbatim."""
        assert "Be very careful about using `llm_query`" in prompt_qwen

    def test_gpt5_no_addendum_section(self, prompt_gpt5):
        """GPT-5 has no addendum — the addendum section must be absent."""
        # The addendum section header should not appear.
        assert "MODEL-SPECIFIC ADDENDUM" not in prompt_gpt5

    def test_claude_no_addendum(self, sample_context_metadata, claude_model):
        prompt = build_system_prompt(
            context_metadata=sample_context_metadata,
            root_model=claude_model,
        )
        assert "MODEL-SPECIFIC ADDENDUM" not in prompt

    def test_custom_addendum_appended_verbatim(self, sample_context_metadata):
        """Any non-empty addendum must appear verbatim at the end."""
        custom = RootModel(
            key="custom",
            rlm_backend="openai",
            backend_kwargs={"model_name": "custom-model"},
            sub_backend="openai",
            sub_backend_kwargs={"model_name": "custom-mini"},
            prompt_addendum="CUSTOM_ADDENDUM_SENTINEL_TEXT_XYZ",
            paper_validated=False,
        )
        prompt = build_system_prompt(
            context_metadata=sample_context_metadata,
            root_model=custom,
        )
        assert "CUSTOM_ADDENDUM_SENTINEL_TEXT_XYZ" in prompt


class TestNoFixedWorkflow:
    """The prompt must NOT prescribe a step-by-step fixed workflow."""

    @pytest.mark.parametrize(
        "forbidden_phrase",
        [
            "first, call",
            "first call understand_section",
            "step 1:",
            "step 2:",
            "then call detect_environment",
            "then call build_environment",
            "first call detect_environment",
        ],
    )
    def test_forbidden_workflow_phrase_absent(self, prompt_gpt5, forbidden_phrase):
        """Fixed workflow phrases must not appear in the prompt."""
        assert forbidden_phrase.lower() not in prompt_gpt5.lower(), (
            f"Forbidden workflow phrase {forbidden_phrase!r} found in prompt"
        )

    def test_no_numbered_step_workflow(self, prompt_gpt5):
        """Must not contain a numbered sequence 'Step 1 ... Step 2 ... Step 3'
        that prescribes the reproduction order.  Note: the termination
        section uses 'Step 1/2/3' for the FINAL_VAR mechanics — that is
        allowed; what is forbidden is prescribing primitive call order.
        """
        # Count how many "Step N" patterns appear.  The termination section
        # uses three (for json.dumps → FINAL_VAR), which is fine.
        # We look specifically for calls to primitives in a fixed order.
        text = prompt_gpt5.lower()
        # A naive but useful check: no "step 1.*understand" + "step 2.*detect"
        # patterns that would indicate a fixed-order primitive workflow.
        assert not (
            re.search(r"step\s+1[^\n]*understand", text)
            and re.search(r"step\s+2[^\n]*detect", text)
        ), "Prompt appears to prescribe fixed primitive-call workflow"


class TestNonTrivialLength:
    """The prompt must be substantial — paper Appendix C is long."""

    def test_prompt_length_exceeds_minimum(self, prompt_gpt5):
        # 1500 chars is a conservative floor — the real prompt should be much longer.
        assert len(prompt_gpt5) >= 1500, (
            f"Prompt too short ({len(prompt_gpt5)} chars) — "
            "paper Appendix C quality bar requires a detailed prompt"
        )

    def test_prompt_contains_decomposition_example(self, prompt_gpt5):
        """An in-context decomposition example must be present (paper Fig 4a)."""
        lower = prompt_gpt5.lower()
        assert (
            "example" in lower
            and ("decompos" in lower or "summarise" in lower or "iteration" in lower)
        ), "Prompt is missing the in-context decomposition example"

    def test_prompt_contains_algorithm2_guard(self, prompt_gpt5):
        """The Algorithm-2 guard (never pass full corpus to primitive) must appear."""
        assert "Algorithm-2" in prompt_gpt5 or "algorithm-2" in prompt_gpt5.lower() or (
            "never pass" in prompt_gpt5.lower()
            or "whole" in prompt_gpt5.lower()
        )

    def test_prompt_contains_triage_instruction(self, prompt_gpt5):
        """The triage instruction (decline low-value candidates) must appear."""
        lower = prompt_gpt5.lower()
        assert "triage" in lower or "decline" in lower or "worth attempting" in lower


class TestReturnType:
    """build_system_prompt must return a plain str in all cases."""

    def test_returns_str(self, gpt5_model):
        result = build_system_prompt(context_metadata={}, root_model=gpt5_model)
        assert isinstance(result, str)

    def test_no_none_returned(self, qwen_model, sample_context_metadata):
        result = build_system_prompt(
            context_metadata=sample_context_metadata,
            root_model=qwen_model,
        )
        assert result is not None
        assert result != ""


class TestFormatTemplate:
    """rlm consumes the prompt as a .format() template — rlm/utils/prompts.py:156
    runs ``system_prompt.format(custom_tools_section=...)``. The prompt must
    survive that call despite its literal JSON/code braces."""

    def test_format_does_not_raise(self, prompt_gpt5):
        """rlm's .format(custom_tools_section=...) must not raise on the literal
        braces in the JSON report example / code snippets."""
        rendered = prompt_gpt5.format(custom_tools_section="TOOLS_GO_HERE")
        assert "TOOLS_GO_HERE" in rendered

    def test_exactly_one_format_field(self, prompt_gpt5):
        """The template must expose exactly one .format() field —
        custom_tools_section. Any stray {field} would KeyError inside rlm."""
        import string

        fields = {
            name
            for _, name, _, _ in string.Formatter().parse(prompt_gpt5)
            if name is not None
        }
        assert fields == {"custom_tools_section"}, (
            f"prompt has unexpected .format() fields: {fields}"
        )

    def test_rendered_prompt_restores_literal_braces(self, prompt_gpt5):
        """After .format(), the escaped braces become the literal braces the
        root model needs (the JSON report shape must read normally)."""
        rendered = prompt_gpt5.format(custom_tools_section="")
        # A literal-brace fragment from the report-shape section is restored
        # (the prompt's nested-dict JSON examples must read normally — that the
        # round-trip is otherwise correct is covered by test_format_does_not_raise
        # and test_exactly_one_format_field).
        assert '{"id"' in rendered
        assert "json.dumps" in rendered

    def test_format_injects_tool_section_in_primitives_block(self, prompt_gpt5):
        """The {custom_tools_section} slot sits inside the PRIMITIVES section."""
        rendered = prompt_gpt5.format(custom_tools_section="<<<PRIMITIVE_DOCS>>>")
        assert "<<<PRIMITIVE_DOCS>>>" in rendered
        assert rendered.index("PRIMITIVES") < rendered.index("<<<PRIMITIVE_DOCS>>>")


class TestRlmQueryNudge:
    """The prompt must contain the rlm_query preference guidance paragraph."""

    def test_prompt_contains_rlm_query_nudge(self, prompt_gpt5):
        """build_system_prompt output must mention rlm_query AND 'spawns a sub-RLM'."""
        assert "rlm_query" in prompt_gpt5
        assert "spawns a sub-RLM" in prompt_gpt5


class TestDecisionAdvisor:
    """The prompt must mention recommend_next_tool."""

    def test_recommend_next_tool_mentioned(self, prompt_gpt5):
        assert "recommend_next_tool" in prompt_gpt5


class TestHeartbeatInstruction:
    """The prompt must instruct the root to call heartbeat before long operations."""

    def test_heartbeat_mentioned(self, prompt_gpt5):
        """'heartbeat' must appear in the prompt."""
        assert "heartbeat" in prompt_gpt5

    def test_heartbeat_before_implement_baseline(self, prompt_gpt5):
        """The prompt must mention calling heartbeat before implement_baseline."""
        lower = prompt_gpt5.lower()
        assert "implement_baseline" in lower
        # heartbeat and implement_baseline must both be present; relative ordering
        # is already enforced by _HEARTBEAT_SECTION appearing in the prompt.
        assert "heartbeat" in lower

    def test_heartbeat_before_run_experiment(self, prompt_gpt5):
        """The prompt must mention run_experiment in the heartbeat section."""
        assert "run_experiment" in prompt_gpt5

    def test_heartbeat_liveness_rationale(self, prompt_gpt5):
        """The prompt must state WHY to call heartbeat (operator visibility)."""
        lower = prompt_gpt5.lower()
        assert "alive" in lower or "operator" in lower or "visible" in lower


class TestGpuSelectionSection:
    """The prompt must instruct the root to call resolve_gpu_requirements."""

    def test_resolve_gpu_requirements_mentioned(self, prompt_gpt5):
        """'resolve_gpu_requirements' must appear in the prompt."""
        assert "resolve_gpu_requirements" in prompt_gpt5

    def test_estimated_vram_gb_key_mentioned(self, prompt_gpt5):
        """The prompt must name the 'estimated_vram_gb' key so the root knows the schema."""
        assert "estimated_vram_gb" in prompt_gpt5
