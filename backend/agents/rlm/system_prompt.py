"""Root-model system prompt for the RLM loop.

Adapted from paper Appendix C. Per brief correction #3, this prompt is
deliberately long and carries in-context decomposition examples (Figure
4a: examples improve overall performance and first-decomposition quality
"even if the example is unrelated to the actual task"). The previous
~2000-token cap from the brief body is dropped.

The prompt describes:
  - The RLM operating model (paper §2 properties 1, 2, 3)
  - REPL variable names with types + lengths (NEVER contents)
  - Primitive function signatures with one-line descriptions
  - The two termination tags: FINAL_VAR(name) and FINAL(text)
  - Per-model addenda (e.g. Qwen anti-over-subcalling line)

The prompt does NOT prescribe a workflow ("first call understand_section,
then call detect_environment, ..."). The root figures out workflow from
REPL exploration. Brief §13 FM#6 — system prompt bloat — applies to
*workflow* instructions, not to in-context decomposition examples or
primitive descriptions.

Phase 2 (#59) implementation. See `docs/rlm-pivot-mapping.md` §5 for the
paper-anchored content and §6.4 for the per-model selection logic.
"""

from __future__ import annotations

from typing import Any


def build_system_prompt(
    *,
    repl_variables: dict[str, Any],     # name -> {type, length, ...} metadata
    primitive_signatures: list[str],     # one signature line per primitive
    root_model: str = "default",         # used to select per-model addenda
) -> str:
    """Compose the root system prompt for one run.

    Phase 2 (#59) contract:
      1. Start with paper Appendix C's core RLM operating principles.
      2. Add ≥1 in-context decomposition example (paper Fig 4a recommends
         examples even when unrelated to the task).
      3. Describe each REPL variable by name + type + length (NEVER value).
      4. Describe each primitive by signature + one-line behavior note.
      5. Document FINAL_VAR(name) (preferred) and FINAL(text) (fallback)
         termination tags with the safeguard note from Appendix B.
      6. Inject per-model addenda (e.g. for `root_model.startswith("qwen")`
         add the anti-over-subcalling line: "Be very careful about using
         llm_query as it incurs high runtime costs. Always batch...").
      7. NO workflow prescription (no "first do X, then Y" instructions).
    """
    raise NotImplementedError("Phase 2 (#59) — compose root system prompt per paper Appendix C")
