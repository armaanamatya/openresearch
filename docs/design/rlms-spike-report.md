# RLM Engine Spike — Fork RESOLVED

**Date:** 2026-05-21 · **Resolves:** the #64 architecture fork (drift D1) ·
**Method:** installed and probed every candidate engine in `.venv`.

## Verdict

**The `rlm` library wins. Hand-building (issue #59 / PR #65 skeleton) is
retired. `dspy.RLM` is evaluated and not adopted.** The brief §3 is
substantially accurate — this spike confirms it empirically, with one callback
claim corrected below (see "Verified vs. assumed").

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
  - `on_subcall_start` / `on_subcall_complete` — **invoked** at
    `rlm/core/rlm.py:739, 805`; usable for per-subcall SSE events.
  - `on_iteration_start` / `on_iteration_complete` — in the `rlm.RLM`
    signature but **never invoked in rlm 0.1.1** (no call site — grep-verified;
    see "Verified vs. assumed"). The SSE bridge must NOT depend on these.
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

### Verified vs. assumed

The fork resolution rested partly on `inspect.signature` probing — confirming
constructor parameters EXIST, not that they are USED. A follow-up pass
(`tools/rlms_spike.py`, source reads of rlm 0.1.1 — 2026-05-21) classifies the
key claims by evidence strength.

**VERIFIED — empirically exercised, or confirmed at the call site in source:**

- `environment='docker'` — present in the `EnvironmentType` literal (signature).
- `custom_tools` — injected into the REPL and callable there (mock spike: the
  mock primitives ran inside `exec`).
- `.completion()` — runs the Algorithm-1 loop end to end (mock spike).
- `custom_system_prompt` — constructor arg; overrides the default prompt (source).
- `on_subcall_start` / `on_subcall_complete` — invoked at `rlm/core/rlm.py:739`
  and `:805` (inside `_subcall`).
- Termination — the model emits a `FINAL_VAR(name)` / `FINAL(text)` tag, parsed
  by `rlm/utils/parsing.py::find_final_answer`; the mock spike terminated via
  `FINAL_VAR`. rlm has **no reserved `answer` variable** — the reserved REPL
  variable is `context` (the offloaded prompt). Where the brief or an issue
  says "termination via the `answer` variable," read `FINAL_VAR`.
- Depth-2 recursion — at `max_depth=2`, a `rlm_query()` from the root spawns a
  genuine child RLM (`on_subcall_*` fire at depth 1); a `rlm_query()` at the
  depth cap degrades to a plain LM call (no child RLM, no depth-2 `on_subcall_*`).
  Verified by `tools/rlms_spike.py --depth2` (brief paper-accuracy correction #1).

**ASSUMED but FALSE — corrected:**

- `on_iteration_start` / `on_iteration_complete` — declared, documented, and
  stored (`rlm/core/rlm.py:75-76, 107-108, 150-151`) but **never invoked** — no
  call site anywhere in rlm 0.1.1 (grep-verified). They are dead parameters.
  Per-iteration observability comes instead from the `logger`: `rlm.RLM` calls
  `logger.log(iteration)` once per iteration (`rlm/core/rlm.py:367-368`). An SSE
  bridge should pass a custom `RLMLogger` subclass whose `.log()` override emits
  a per-iteration event (calling `super().log()` to keep trajectory capture);
  `RLMLogger(log_dir=...)` also appends each iteration to a JSONL file
  synchronously during the run.

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
