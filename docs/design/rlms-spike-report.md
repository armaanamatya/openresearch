# RLM Engine Spike — Fork RESOLVED

**Date:** 2026-05-21 · **Resolves:** the #64 architecture fork (drift D1) ·
**Method:** installed and probed every candidate engine in `.venv`.

## Verdict

**The `rlm` library wins. Hand-building (issue #59 / PR #65 skeleton) is
retired. `dspy.RLM` is evaluated and not adopted.** The brief §3 was correct
all along — this spike confirms it empirically.

## Evidence

### Option B — `rlm` library (brief §3) — ADOPT

- `pip install rlms` → package `rlms 0.1.1`, **import name `rlm`**. Imports clean.
- `rlm.RLM.__init__` signature matches brief §3's table **exactly**:
  - `environment: Literal['local','docker','modal','prime','daytona','e2b']` —
    **has `docker`.** ReproLab runs Docker builds + experiments; this is decisive.
  - `custom_tools: dict` — the domain-primitive injection point.
  - `custom_sub_tools`, `max_depth=1` (set 2 per brief correction #1),
    `max_iterations=30`, `max_budget`, `max_timeout`, `max_tokens`, `max_errors`.
  - `other_backends` — cheaper sub-call model (the GPT-5 / GPT-5-mini pattern).
  - `on_subcall_start/complete`, `on_iteration_start/complete` — **live
    callbacks**, map directly onto ReproLab's SSE UI.
  - `custom_system_prompt`, `logger: RLMLogger`, `persistent`, `compaction`.
  - `.completion()` entry point; `rlm.environments` exposes `get_environment`,
    `LocalREPL`, `RESERVED_TOOL_NAMES`, `validate_custom_tools`.
- It is the paper's first author's reference implementation — faithful by
  construction.

### Option C — `dspy.RLM` — NOT adopted (kept as reference, issue #66)

- `dspy 3.2.1` installs; `dspy.RLM` is real, signature as documented
  (`signature, max_iterations, max_llm_calls, tools, sub_lm, interpreter`).
- **Disqualifier:** its default interpreter is a Deno/Pyodide **WASM sandbox**.
  ReproLab primitives must run Docker image builds and GPU experiments — these
  cannot run inside Pyodide. `rlm`'s `environment='docker'` is the right model.
- Also: no live `on_*` callbacks (only post-hoc `trajectory`); no
  `custom_system_prompt` (DSPy drives the prompt via its signature
  abstraction). Both are friction against ReproLab's SSE UI and
  reproduction-domain prompt.
- Verdict: a clean reference implementation of the same paper, but the wrong
  execution model for this system. Issue #66 closed as evaluated-not-adopted.

### Option A — hand-build — RETIRED

Two real, installable libraries implement Algorithm 1 faithfully. Hand-writing
a REPL host + root loop + `sub_LLM`/`sub_RLM` (issue #59's old deliverables,
PR #65's `repl_host.py` / `root_loop.py` / `sub_call.py` stubs) is redundant
work and a faithfulness risk. Retired.

## Consequences

- Brief §3/§5/§11 stand — confirmed accurate. Banners flipped CONTESTED → RESOLVED.
- Issue #59 rewritten: Phase 2 is **domain-primitive extraction**, not
  foundation hand-building. `rlm` provides the foundation.
- PR #65: the `backend/agents/rlm/` hand-build skeleton (`repl_host.py`,
  `root_loop.py`, `sub_call.py`) is superseded — do not merge as the
  foundation. Its `primitives.py` / `system_prompt.py` survive as concepts;
  `docs/rlm-pivot-mapping.md` §1–§2 (stage→primitive table + signatures) survive.
- Next step is the brief's real Phase 1: a minimal `RLM(custom_tools=…)`
  `.completion()` run on a mock paper (needs an LLM key) — implementation, not
  a blocker for this decision.
