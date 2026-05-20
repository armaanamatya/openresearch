"""Algorithm 1 root loop.

Implements the RLM root loop from arXiv 2512.24601 §2. The loop:

    state = InitREPL(prompt=P)
    state = AddFunction(state, sub_RLM_M)
    hist = [Metadata(state)]
    for iter in range(MAX_ROOT_ITERATIONS):       # paper Appendix A: 20
        code = LLM_M(hist)                         # ≤ 4096 output tokens
        state, stdout = REPL(state, code)
        hist = hist || code || Metadata(stdout)    # Metadata, NOT raw stdout
        name = parse_final_var_tag(code or stdout)
        if name is not None and state.has_variable(name):
            return state.read_variable(name)
    raise RootIterationCapExceeded

Phase 2 (#59) implementation. See `docs/rlm-pivot-mapping.md` §5 for the
paper-anchored design and §6.1 for the sync/async bridge decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Paper Appendix A: MRCRv2 training used max 20 RLM iterations, 4096 output
# tokens per turn. These are the root-loop budgets (separate from sub-call
# budgets in sub_call.py and from the run-wide RunBudget).
DEFAULT_MAX_ROOT_ITERATIONS = 20
DEFAULT_MAX_OUTPUT_TOKENS_PER_TURN = 4096

# FINAL_VAR(name) — the correct termination path (reads value from REPL).
# FINAL(text) — the autoregressive path; per paper Appendix B it is brittle
# (models sometimes emit their plan as the "final answer"). Safeguard in
# parse_final_tag() rejects FINAL(text) when the trajectory looks like a
# plan rather than an answer.
FINAL_VAR_TAG_RE = re.compile(r"FINAL_VAR\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
FINAL_TEXT_TAG_RE = re.compile(r"FINAL\((.+?)\)", re.DOTALL)


class RootIterationCapExceeded(RuntimeError):
    """Raised when the root loop hits MAX_ROOT_ITERATIONS without terminating."""


@dataclass
class RootHistoryEntry:
    """One root-loop iteration recorded in the history fed to the next turn."""

    iteration: int
    code: str
    stdout_metadata: dict[str, Any]


class RootLoop:
    """Algorithm 1 loop. One instance per RLM run."""

    def __init__(
        self,
        *,
        repl_host: Any,         # ReplHost — typed as Any to avoid Phase 2 import cycle
        llm_client: Any,        # LlmClient Protocol
        system_prompt: str,
        max_iterations: int = DEFAULT_MAX_ROOT_ITERATIONS,
        max_output_tokens_per_turn: int = DEFAULT_MAX_OUTPUT_TOKENS_PER_TURN,
    ) -> None:
        raise NotImplementedError("Phase 2 (#59) — wire ReplHost + LlmClient + system prompt")

    def run(self) -> Any:
        """Drive the root loop until FINAL_VAR(name) terminates or the cap raises."""
        # Phase 2 contract:
        #   1. Build the initial history: [Metadata(REPL variables)]
        #   2. For each iteration (up to max_iterations):
        #      a. Call LLM_M(hist) — emit `repl_iteration` SSE event with the code
        #      b. Run repl_host.exec(code) — capture stdout metadata
        #      c. Emit `variable_update` events for any new bindings
        #      d. Append code + Metadata(stdout) to hist (NEVER raw stdout)
        #      e. Parse FINAL_VAR tag; if found and variable exists, return its value
        #      f. Checkpoint REPL state + history after each iteration
        #   3. If loop completes without termination, raise RootIterationCapExceeded.
        raise NotImplementedError("Phase 2 (#59) — Algorithm 1 implementation")


def parse_final_tag(text: str) -> tuple[str, str | None]:
    """Parse a FINAL_VAR or FINAL tag from model output.

    Returns (kind, value) where:
      - kind == "var" and value is the REPL variable name to read out, or
      - kind == "text" and value is the autoregressive answer (with safeguard), or
      - kind == "none" and value is None.

    Phase 2 (#59) implements the safeguard logic from paper Appendix B
    (e.g. reject FINAL(text) when the model has not yet written substantive
    code, to avoid the "model returned its plan as the final answer" failure).
    """
    raise NotImplementedError("Phase 2 (#59) — FINAL_VAR/FINAL parser with safeguards")
