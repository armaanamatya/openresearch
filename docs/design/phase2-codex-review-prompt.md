# Codex Deep-Review Prompt — RLM Pivot Phase 2 (PR #68)

> Paste everything below the line into Codex, with the `openresearch` repo available
> and the branch `feat/rlm-phase2-foundation` checked out. It is a self-contained
> brief — Codex needs no other context.

---

You are a principal software engineer performing an **independent, adversarial code
review** of a completed feature branch. Be thorough, skeptical, and concrete. Your
review is the last gate before this is merged.

## Why you, specifically

This feature — the *plan*, the *implementation*, and **every code review done so far**
— was produced end-to-end by Claude AI agents (a multi-agent "subagent-driven"
pipeline: a planner, 15 fresh per-task implementers, a spec reviewer and a code-quality
reviewer per task, and a final whole-implementation reviewer). It passed all of those.

You are a **different model**. Your highest-value contribution is **not** to repeat
those reviews — it is to find what a single-model pipeline *structurally could not
see*: shared assumptions every agent inherited from the same planning docs and never
independently verified; "verified against source" claims that actually traced one
reading of one file; correctness arguments that are internally consistent but built on
a wrong premise; whole categories of concern (security, concurrency-under-load,
operability) that the pipeline may have under-weighted because the plan did.

Treat the existing reviews as **untrusted**. Verify every load-bearing claim yourself,
from primary sources (the actual `rlm` library source, the actual stage-agent code,
the actual schemas) — not from the design docs.

## What the repository is

**OpenResearch / ReproLab** is an agent pipeline that reproduces research papers
end-to-end (ingest paper → understand claims → build environment → implement + run a
baseline → gate the result → explore improvements → emit a benchmark report). It is
mid-pivot from a 14-stage `PipelineStage` state machine to an **RLM (Recursive Language
Model) orchestrator** built on the `rlms` PyPI library (`pip install rlms`, import
`rlm`; arXiv 2512.24601).

An RLM offloads its prompt as a `context` variable inside a Python REPL; the "root"
model writes Python in that REPL, slices `context`, and calls tools/sub-LMs — instead
of receiving the whole prompt in its own context window. ReproLab's plan: the paper is
offloaded as `context`; the root model drives reproduction by calling **primitives**.

## What Phase 2 (issue #59) delivers — the thing you are reviewing

Phase 2 builds the **primitive layer**. It extracts each surviving stage-agent's core
logic into a callable "primitive", wraps each primitive so a call emits a
`primitive_call` event + a `cost_ledger.jsonl` row, and assembles the primitives into
the `custom_tools` dict that `rlm.RLM(...)` consumes. Nine primitives:
`understand_section`, `extract_hyperparameters`, `detect_environment`,
`build_environment`, `plan_reproduction`, `implement_baseline`, `run_experiment`,
`verify_against_rubric`, `propose_improvements`.

## Exact scope to review

PR **#68** (`feat/rlm-phase2-foundation` → `main`). Its diff against `main` is large
because it **stacks on PR #65** (Phase 1, branch `rlm-pivot`). **Review only the Phase 2
delta** — the 35 commits past `rlm-pivot`:

```
git diff 63adb10..HEAD            # 63adb10 = rlm-pivot HEAD; the Phase 2 delta
git log  63adb10..HEAD --oneline  # 35 commits
```

Concretely, the Phase 2 code surface is:
- `backend/agents/rlm/context.py` — `RunContext` dataclass
- `backend/agents/rlm/binding.py` — `wrap_primitive`, `build_custom_tools`, summarizers
- `backend/agents/rlm/primitives.py` — the 9 primitives, `PRIMITIVE_REGISTRY`,
  `PRIMITIVE_DESCRIPTIONS`, `_extract_json`, `_clamp01`
- `backend/agents/rlm/__init__.py` — package exports
- `backend/agents/dashboard_emitter.py` — **only** the added `primitive_call` method
- `backend/requirements.txt` — 2 added dependency lines
- `tests/rlm/` — 31 tests (conftest + 14 test files)

Reference docs (read for intent, **do not treat as ground truth** — they were written
by the same pipeline):
- `docs/design/rlm-pivot-brief.md` — canonical architecture; §7 lists the environment,
  §8 lists the RLM-fidelity invariants
- `docs/design/phase2-implementation-plan.md` — the 15-task spec
- `docs/design/phase2-execution-brief.md` — the curated per-subagent context
- `docs/design/phase2-plan-audit.md` — a resolved audit of the plan
- `docs/design/rlms-spike-report.md` — the `rlm`-library spike findings

The `rlm` library itself is installed at `.venv/lib/python3.14/site-packages/rlm/` —
**read it** to verify every claim about its API.

## Review checklist

### A. Per-primitive correctness
For each of the 9 primitives: does it faithfully wrap its underlying stage-agent
helper/prompt (the design rule is "wrap, not rewrite")? Is the signature
`(<args>, *, ctx: RunContext)`? Is the return shape what `PRIMITIVE_DESCRIPTIONS` and
downstream consumers expect? Are error paths sane?

### B. Cross-primitive data-flow contracts (highest priority)
The root model chains primitives. Trace every hand-off and confirm the producer's
output is a valid input to the consumer — by reading both sides, not the docstring:
- `understand_section` → `detect_environment(method_spec)` and
  `implement_baseline(plan["paper_claim_map"])`
- `detect_environment` → `build_environment(env_spec)` and
  `implement_baseline(plan["environment_spec"])`
- `build_environment` returns `{ok, image_tag, ...}` → `run_experiment(code_path, env_id)`
  consumes `image_tag` as `env_id`
- `implement_baseline` returns a code-dir path + writes `code/commands.json` →
  `run_experiment` reads `commands.json` from that path
- `run_experiment` returns `{success, metrics, logs}` → `verify_against_rubric` and
  `propose_improvements`
Any key-name or shape mismatch is a Critical finding. Look hard at paths: where does
`implement_baseline` write `commands.json`, where does the wrapped code-writer write
the code, where does `run_experiment` look — do all three agree without relying on an
unenforced invariant?

### C. The `rlm` 0.1.1 library contract — verify from installed source
- `build_custom_tools` emits `{name: {"tool": callable, "description": str}}`. Does
  `rlm` 0.1.1 actually consume exactly that shape? Read `rlm/environments/`.
- The root REPL is constructed with `environment="local"`. The design claims `rlm`'s
  `DockerREPL` does **not** inject `custom_tools` into the in-container globals, so
  `local` is mandatory. **Verify this in `rlm/environments/docker_repl.py` and
  `local_repl.py`.** If it is wrong, the whole approach is wrong.
- Termination is via a `FINAL_VAR(name)` / `FINAL(text)` tag. Confirm against
  `rlm/utils/parsing.py`.
- The integration test monkeypatches `rlm.core.rlm.get_client`. Is that the correct
  and stable seam? Could the test pass while the real wiring is broken?

### D. The wrapper (`binding.py`)
`wrap_primitive` must emit `primitive_call` `start` then `ok`/`error`, append a
`cost_ledger.jsonl` row on **both** success and failure paths, and re-raise on error.
Verify. Check the `kwargs.pop("ctx", None)` guard. Check `_summarize` / `_result_summary`
are side-effect-free and never leak full argument/return *values* into the event log.

### E. Concurrency / async
`build_environment`, `implement_baseline`, `run_experiment` bridge async work via
`ThreadPoolExecutor(max_workers=1)` + `asyncio.run` in a worker thread. Is this correct?
**Phase 3 will call these primitives from inside `rlm`'s REPL, which itself may run
under an event loop** — does the bridge hold up then, or only in the synchronous test
harness? Any shared-state hazard if two primitives run concurrently? Is a per-iteration
`ThreadPoolExecutor` the right cost?

### F. Security / threat model — scrutinize hard; the pipeline may have under-weighted this
- The root REPL is `environment="local"` — `rlm` executes the model's Python via
  `exec` **on the host process**. `rlm`'s own README says `local` is "minimal security
  ... should not be used for production." What is ReproLab's threat model here? Is the
  paper (offloaded as `context`) ever attacker-influenced?
- `run_experiment` runs arbitrary commands (read from `commands.json`) inside a Docker
  container. `build_environment` builds Docker images from a model-/LLM-influenced
  Dockerfile. `implement_baseline` invokes a code-writing agent. Trace what is
  attacker-controllable and what isolation actually exists.
- `_execute_in_sandbox` hard-codes `timeout=3600`. Resource bounds — memory, network,
  CPU — for the experiment container: present? enforced?

### G. Fail-soft & honesty logic
- `build_environment` claims to be fully fail-soft except `SandboxRuntimeError`. Is
  that actually true for *every* internal failure (import error, `llm_client` error,
  `write_text`, a malformed `env_spec`)?
- `verify_against_rubric` caps area scores at 0.35 on a degraded run and `_clamp01`s
  LLM floats. Is the cap logic correct? Can a malformed LLM response crash it?
- `propose_improvements` drops malformed items with a blanket `except Exception`. Too
  broad — could it swallow a real bug? Is the silent drop observable anywhere?

### H. `_extract_json`
Reused by 3 primitives. Does the `JSONDecoder.raw_decode` scan-from-each-`{` actually
handle code fences, prose, braces-in-strings, and multiple objects correctly? Find an
input that breaks it. Is raising `ValueError` on no-JSON the right contract?

### I. RLM fidelity (brief §8 invariants)
- **The paper must never enter the root model's own context** — it is offloaded as
  `context`. Does anything in Phase 2 violate that?
- **Primitives take slices/specs, not the whole corpus.** Check every primitive
  signature: could a primitive be handed the entire paper and feed it to an LLM,
  re-creating the context-window problem RLM exists to avoid?
- The "Algorithm-2 guard": primitives must not take `project_id`/`runs_root` as call
  arguments (those come from `ctx`). Confirm.

### J. Schemas
Pydantic models (`PaperClaimMap`, `EnvironmentSpec`, `ReproductionContract`,
`RubricVerification`, `ImprovementHypothesis`). Check the `core_contribution`
defaulting trick, the `extra="ignore"` reliance, and `RubricVerification.from_areas`
usage. Does `.model_dump()` always yield a JSON-serializable dict the REPL can hold?

### K. Test quality — do the tests prove behavior, or mock the thing under test?
31 tests. Scrutinize: the integration test uses a `ScriptedLM` — is the LLM correctly
mocked while the *real* `rlm` REPL + `custom_tools` injection is exercised, or is the
system-under-test itself stubbed? Which assertions are schema-trivially true? Where are
the coverage gaps *between* tasks (each task was tested in isolation by a separate
agent)? Run the suite and judge it as a whole.

### L. The known deferral
`run_experiment` returns `metrics={}` (real metric extraction deferred to Phase 5 /
#62). Consequence: `verify_against_rubric`'s `degraded` backstop fires on *every*
Phase-2 run, capping all scores ≤0.35 — so the verify→improve loop cannot produce a
meaningful score yet. Is this deferral handled honestly and documented, or does it
quietly make a chunk of Phase 2 vacuous? Is shipping it in this state acceptable?

### M. AI-pipeline blind spots — the meta-review
Step back. The plan, the code, and the reviews all came from one model family reading
one set of docs. Name the assumptions that were never independently checked. Where does
the implementation "verify against source" but actually only mirror a doc claim? What
would a human reviewer with production-operations scars flag that this pipeline did not?

## How to verify empirically

```bash
# from the repo root, branch feat/rlm-phase2-foundation
pip install -r backend/requirements.txt
.venv/bin/python -m pytest tests/rlm/ -v          # expect 31 passed
.venv/bin/python -m pytest tests/ -q              # full suite — confirm no NEW failures
.venv/bin/python -c "import backend.agents.rlm; print(sorted(backend.agents.rlm.__all__))"
# read the rlm library to verify the API claims:
ls .venv/lib/python3.14/site-packages/rlm/environments/
```

Re-run anything the existing reviews "confirmed" — do not take a pass count on faith.

## Output format

### Verdict
One of: **Ship** / **Ship with fixes** / **Do not ship** — plus one paragraph of
reasoning.

### Findings
Grouped by severity. For each: `file:line`, what is wrong, *why it matters*, and a
concrete fix.
- **Critical** — bugs, broken cross-primitive contracts, security holes, missing #59 scope
- **Important** — should fix before merge
- **Minor** — nice to have

### Cross-primitive data-flow assessment
Explicitly: do the 9 primitives chain correctly end-to-end? List every hand-off you
traced and whether it holds.

### What the AI pipeline missed
The deliverable section unique to your review: assumptions never independently
verified, under-weighted concern categories, premises that should have been questioned.

### `rlm` library contract — confirmed or refuted
State, from your own reading of the installed `rlm` source, whether each claimed API
fact (custom_tools injection, `environment="local"` requirement, `FINAL_VAR`
termination, the `get_client` seam) is correct.

## Ground rules

- Verify load-bearing claims from **primary source** — the installed `rlm` package, the
  actual stage-agent modules, the actual schemas — never from the design docs alone.
- Be specific: `file:line`, not "improve error handling".
- Be adversarial but fair. Categorize by *actual* severity; do not inflate nits to
  Critical, and do not soften a real bug.
- If the implementation is genuinely sound, say so plainly — a clean review is a valid
  outcome. But you are explicitly being asked to look for what a same-model pipeline
  could not see; look hard before concluding it is clean.
