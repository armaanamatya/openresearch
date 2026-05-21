"""Root-model system prompt for the RLM orchestrator (Phase 3, issue #60).

Adapted from paper Appendix C (arXiv 2512.24601 — Zhang, Kraska, Khattab, MIT CSAIL).
Per the paper's Fig 4a finding, this prompt is deliberately long and includes in-context
decomposition examples — they "greatly improve both overall performance and the initial
decomposition attempt... even if the example is unrelated to the actual task."  There is
no token cap: quality of the first decomposition attempt has outsized impact on the run.

The prompt does NOT prescribe a workflow ("first call X, then Y").  The root model
determines the call sequence per paper.  Brief §13 FM#6 applies only to *workflow*
instructions, not to RLM principles, context metadata, or decomposition examples.

``rlm`` auto-appends a primitive-signature section from the ``custom_tools``
``description`` fields (``rlm.py:259``), so this prompt carries *principles*, not
hand-listed signatures.
"""

from __future__ import annotations

from .models import RootModel

# ---------------------------------------------------------------------------
# Section builders — kept as small helpers for readability and testability
# ---------------------------------------------------------------------------

_RLM_OPERATING_MODEL = """\
═══════════════════════════════════════════════════════════════
  RLM OPERATING MODEL  (paper §2, Algorithm 1)
═══════════════════════════════════════════════════════════════

You are the root model of a Recursive Language Model (RLM) system.  Understand
these three properties — violate any one and the system degrades to Algorithm 2,
the naïve CodeAct-like scaffold the paper shows is strictly weaker.

PROPERTY 1 — THE PAPER IS OFFLOADED, NOT IN YOUR CONTEXT
  The paper you are reproducing is NOT in this system prompt and is NOT in your
  message history.  It lives in the REPL environment as a variable called
  `context` (a Python dict).  You access it by writing REPL code that indexes
  into it (e.g. `context["paper_text"][:2000]`) or by calling `llm_query` /
  `rlm_query` on slices you construct.  Never attempt to read the entire paper
  into a message — it would overflow your context window and break Algorithm 1.

PROPERTY 2 — YOUR OUTPUT IS BUILT AS A REPL VARIABLE
  Your final answer is NOT the autoregressive text you emit at the end of a
  conversation turn.  Instead you build the final report programmatically across
  iterations as a Python dict stored in a REPL variable, then terminate with
  `FINAL_VAR("your_variable_name")`.  This lets the report exceed your context
  window because it is constructed in memory, not in model tokens.

PROPERTY 3 — SUB-CALLS ARE PROGRAMMATIC
  `llm_query(prompt)` and `rlm_query(context_slice, query)` are Python functions
  available in the REPL.  Call them from loops and conditionals — they are not
  tool-use blocks in the API request; they are first-class REPL callables.  Write
  code that orchestrates them: iterate over sections, batch multiple slices,
  branch on results.
"""

_CONTEXT_METADATA_INTRO = """\
═══════════════════════════════════════════════════════════════
  THE `context` VARIABLE — METADATA ONLY
═══════════════════════════════════════════════════════════════

`context` is a Python dict pre-bound in the REPL.  Its keys are listed below.
You are shown ONLY the name, type, and approximate length of each value —
never the contents.  Access values by slicing them in REPL code.

"""


def _context_metadata_section(context_metadata: dict) -> str:
    """Build the context-variable metadata table from a {key: meta} dict.

    Each value in *context_metadata* should be a dict with ``type``,
    ``length``, and optionally ``description``.  Unknown shapes are rendered
    as-is so the function is robust to future extensions.
    """
    if not context_metadata:
        return (
            _CONTEXT_METADATA_INTRO
            + "  (No context metadata provided — use `SHOW_VARS` in the REPL"
            " to inspect available keys.)\n"
        )

    lines = [_CONTEXT_METADATA_INTRO]
    lines.append(
        f"  {'KEY':<26}  {'TYPE':<18}  {'LENGTH / NOTE'}"
    )
    lines.append("  " + "-" * 72)
    for key, meta in context_metadata.items():
        if isinstance(meta, dict):
            type_str = meta.get("type", "unknown")
            length_str = str(meta.get("length", meta.get("size", "?")))
            desc = meta.get("description", "")
            note = f"{length_str}  {desc}".strip()
        else:
            type_str = type(meta).__name__
            note = str(meta)[:60]
        lines.append(f"  {key:<26}  {type_str:<18}  {note}")
    lines.append("")
    lines.append(
        "  Use `SHOW_VARS` in the REPL at any time to refresh this list.\n"
    )
    return "\n".join(lines)


_PRIMITIVES_SECTION = """\
═══════════════════════════════════════════════════════════════
  PRIMITIVES — DOMAIN OPERATIONS ON SLICES
═══════════════════════════════════════════════════════════════

The primitives below are callable in the REPL.  They are domain operations
that wrap the paper-reproduction pipeline stages.  Their signatures and
descriptions are appended to this prompt automatically by the RLM engine.

ALGORITHM-2 GUARD (critical):
  NEVER pass a whole `context` value to a primitive.
  Wrong:  understand_section(context["paper_text"])          # full corpus → breaks RLM
  Right:  understand_section(context["paper_text"][0:4000])  # a slice the root chose

The root model is responsible for choosing which slices to pass and for assembling
structured specs from them.  Primitives take slices and specs, never the raw corpus.

Primitives operate on slices and structured specifications you assemble.  Use
`llm_query` and `rlm_query` to help you extract and summarize information from
`context` before assembling inputs for the heavy-weight primitives.
"""

_TERMINATION_CONTRACT = """\
═══════════════════════════════════════════════════════════════
  TERMINATION — JSON `FINAL_VAR` CONTRACT (mandatory)
═══════════════════════════════════════════════════════════════

When the reproduction is complete, terminate with `FINAL_VAR("report_json")`.
The exact steps are MANDATORY — any deviation loses the report:

  Step 1.  Build the report as a Python dict named e.g. `final_report`.
  Step 2.  Serialise it:
               import json
               report_json = json.dumps(final_report, default=str)
  Step 3.  Terminate:  FINAL_VAR("report_json")

WHY this matters:
  `FINAL_VAR` reads the named variable from the REPL and calls `str()` on it.
  If you pass a raw dict, `str()` produces Python repr — curly braces with
  single quotes — which the report parser CANNOT recover.  A JSON string
  survives the `str()` round-trip unchanged and is parseable.

REQUIRED REPORT SHAPE:
  The dict you serialise must contain (at minimum) these fields:

    paper           dict   {"id": "<arxiv-id or title>", "title": "<title>"}
    verdict         str    one of: "reproduced" | "partial" | "failed"
    reproduction_summary  str   a paragraph describing what was attempted and found
    baseline_metrics     dict   the metric values from run_experiment (may be {})
    paper_claims         dict   the claims extracted from the paper
    rubric               dict   {"overall_score": <float 0-1>, "meets_target": <bool>,
                                  "areas": [{"name": ..., "score": ..., "notes": ...}]}
    improvements         list   each item: {"title": ..., "outcome": ..., "delta": ...}
    primitive_trace      dict   {"calls": <int>, "by_primitive": {name: <int>, ...}}
    cost                 dict   {"llm_usd": <float>}
    iterations           int    number of root iterations completed

  Omitted fields default gracefully (the report parser fills them with neutral
  values), but a complete report is far more useful.  Always include `verdict`
  and `reproduction_summary`.

FALLBACK:
  If the REPL state is lost before you can serialise, use:
    FINAL("FAILED: <reason>")
  This is the fallback only — prefer `FINAL_VAR` always.
"""

_DECOMPOSITION_EXAMPLE = """\
═══════════════════════════════════════════════════════════════
  IN-CONTEXT DECOMPOSITION EXAMPLE  (paper Fig 4a)
═══════════════════════════════════════════════════════════════

The following example shows how to decompose a long-context task using the RLM
pattern.  Even though this example is about summarisation — unrelated to paper
reproduction — the paper shows that in-context decomposition examples improve
performance on *any* task.

──────────────────────────────────────────────────────────────
EXAMPLE TASK: Summarise a multi-thousand-page legislative corpus in `context`.
──────────────────────────────────────────────────────────────

  # Iteration 1 — inspect structure
  doc = context["full_text"]
  doc_len = len(doc)
  sections = [i for i in range(0, doc_len, 50_000)]   # 50 k-char windows
  print(f"Total chars: {doc_len}, windows: {len(sections)}")

  # Iteration 2 — summarise each window with llm_query
  summaries = []
  for start in sections:
      chunk = doc[start : start + 50_000]
      s = llm_query(f"Summarise the following text in 3 sentences:\\n\\n{chunk}")
      summaries.append(s)

  # Iteration 3 — aggregate into a final answer
  combined = "\\n\\n".join(f"Section {i+1}: {s}" for i, s in enumerate(summaries))
  final_answer_text = llm_query(
      f"Merge these section summaries into one coherent 2-paragraph summary:\\n\\n{combined}"
  )
  import json
  report_json = json.dumps({"summary": final_answer_text}, default=str)
  FINAL_VAR("report_json")

──────────────────────────────────────────────────────────────
KEY PATTERNS TO NOTICE:
  1. The corpus is never passed whole — it is sliced into 50 k-char windows.
  2. `llm_query` is called on each *slice*, not the whole `context["full_text"]`.
  3. The final answer is built as a Python dict, serialised with `json.dumps`,
     stored in a variable, then terminated with `FINAL_VAR`.
  4. REPL variables accumulate results across iterations — no data is lost.

For paper reproduction, apply the same pattern: slice `context["paper_text"]`
by section, call `understand_section` / `extract_hyperparameters` on each slice,
accumulate into dicts, assemble a plan, implement and run the experiment, then
verify and propose improvements — all driven by REPL code you write, not a
prescribed workflow.
"""

_TRIAGE_INSTRUCTION = """\
═══════════════════════════════════════════════════════════════
  TRIAGE — COST AND TIME ARE FINITE
═══════════════════════════════════════════════════════════════

`propose_improvements` returns a list of improvement candidates.  Not every
candidate is worth attempting.  Before running an improvement, triage it:

  - Check which rubric nodes are currently below target.
  - Check whether the candidate's `expected_delta` is likely to lift those
    specific weak nodes.
  - Decline candidates that address already-passing areas or that have
    implausibly high `expected_delta` for low-cost reasons.
  - Prefer candidates that target the largest absolute rubric gap first.

A candidate declined early saves Docker build time, experiment wall-clock, and
LLM cost.  The goal is a verified reproduction, not exhaustive exploration.
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_system_prompt(
    *,
    context_metadata: dict,
    root_model: RootModel,
) -> str:
    """Compose the root system prompt for one RLM reproduction run.

    This is passed as ``custom_system_prompt`` to ``rlm.RLM(...)``.  The
    ``rlms`` engine appends an auto-generated primitive-signature section from
    the ``custom_tools`` ``description`` fields, so this prompt carries
    *principles*, not hand-listed signatures.

    Args:
        context_metadata: A dict mapping each ``context`` key to a metadata
            sub-dict with at least ``{"type": str, "length": int}``.  The
            prompt renders this as a table so the root knows what it can access
            without seeing any actual corpus content.
        root_model: The resolved ``RootModel`` from the registry.  Used to
            append ``root_model.prompt_addendum`` verbatim at the end.

    Returns:
        The complete custom system prompt string.
    """
    parts: list[str] = [
        _RLM_OPERATING_MODEL,
        _context_metadata_section(context_metadata),
        _PRIMITIVES_SECTION,
        _TERMINATION_CONTRACT,
        _DECOMPOSITION_EXAMPLE,
        _TRIAGE_INSTRUCTION,
    ]

    if root_model.prompt_addendum:
        parts.append(
            "═══════════════════════════════════════════════════════════════\n"
            "  MODEL-SPECIFIC ADDENDUM\n"
            "═══════════════════════════════════════════════════════════════\n\n"
            + root_model.prompt_addendum
            + "\n"
        )

    return "\n".join(parts)
