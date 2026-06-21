# Investigation Plan — Optimizing gpt-chat-latest as the RLM Root (system prompt + guardrailing + orchestration)

> **Status:** IMPLEMENTED 2026-06-19 (flag-gated, default-OFF). Authored 2026-06-19.
> **Owner hint:** Opus designs the prompt/guardrail surface + reviews; delegate mechanical
> edits + the CPU validation harness to Sonnet. Most of this is **CPU-only and cheap** — the
> orchestration defects are all *pre-GPU*. Only the final A/B validation needs a GPU run.
>
> ### What landed (this session, all default-OFF + model-agnostic)
> - **G1 `OPENRESEARCH_ARG_CONTRACTS`** — argument pre-validation in `binding.wrap_primitive`
>   (`backend/agents/rlm/arg_contracts.py`): a declarative per-primitive table blocks placeholder/
>   sentinel arg values before the primitive runs, returning a crisp `failure_class="arg_contract"`
>   repair dict. Closes the non-blocking `paper_grounding_failed` gap (defects #1/#2).
> - **G2 `OPENRESEARCH_STUB_METRICS_GUARD`** — route-agnostic stub detection in `run_experiment`
>   (`backend/agents/rlm/stub_detection.py`): a `success=True` result with only placeholder metric
>   keys (no real-metric key) → repairable `fabrication_suspected` + re-drive directive. Complements
>   the VRAM antifab verdict (which only fires on gpu-training-CLAIMING metrics — the exact gap the
>   2026-06-19 monolithic stub slipped through). Defect #4.
> - **P1-P3** — the shared `azure-foundry` `prompt_addendum` (`models.py`) now carries argument
>   grounding (null-not-guess + exact types), full-paper persistence + honest-failure, and
>   run_experiment result-quality (stub → re-drive). Brace-free (escape round-trip verified). Defects
>   #1/#3/#4/#5 reinforcement.
> - **G3** (degenerate detector) — verified already implements the June-2026 best-practice pattern
>   (signature counter + reset-on-state-change + feed-back-then-escalate); **no change** (threshold
>   stays default-3 pending WS4-B data; semantics A/B-gated).
> - **A1** (base-prompt audit) — **no genuine contradiction / over-strict contract** found; **no
>   change** (the prompt already self-guards brevity-vs-correctness + has graceful escapes).
> - **WS4** — Tier-A CI guard tests (`tests/rlm/test_{arg_contracts,stub_detection,guard_integration}.py`)
>   + Tier-B operator-run A/B harness `scripts/rlm_root_ab.py` (pure parser unit-tested in
>   `tests/test_rlm_root_ab.py`).
> - **Full suite green: 3600 passed.** Enable on the gpt-chat deployment with
>   `OPENRESEARCH_ARG_CONTRACTS=1 OPENRESEARCH_STUB_METRICS_GUARD=1`.
>
> ### Key finding (supersedes WS-executor speculation)
> The non-Claude executor ALREADY runs on the **OpenAI Agents SDK** (`openai-agents`,
> `backend/agents/runtime/openai_runtime.py`) with full Read/Write/Edit/Bash tool parity to
> `claude-agent-sdk`. So gpt-chat-latest stubbing is **model coding-ability, not a harness limit** —
> no SDK swap is warranted, and Sonnet/gpt-5 remain the recommended validated executor. The guardrails
> raise the floor (a stub is non-shippable → re-driven, or the run fails honestly) for every model.
>
> ### Remaining (operator, GPU/creds)
> Run `scripts/rlm_root_ab.py --paper <id>` (needs `AZURE_FOUNDRY_*`) for the Tier-B before/after
> rates, then the spec's ≥3-paired-SDAR-run A/B before flipping any guard default ON.

---

## 1. Goal

Make **gpt-chat-latest** (Azure Foundry, OAuth-free) — and reasoning-class *chat* roots
generally — an **optimal RLM root / orchestrator**, via (a) a strict, model-tuned system prompt,
(b) best-practice **guardrailing**, and (c) best-practice **orchestration** discipline.

**Scope boundary (read this first):** gpt-chat-latest already *functions* as the root — in run
`sdar_gcp_gptchat_v5_20260619` (2026-06-19) it drove the REPL loop cleanly, reached
`run_experiment`, and self-corrected several errors WITHOUT operator help. This investigation is
about making it **correct + optimal**, not making it work from scratch. It is **distinct from the
EXECUTOR-capability problem**: as the executor (`implement_baseline`) gpt-chat-latest writes 0-GPU
stubs (see memory `[[foundry-gptchat-root-not-executor]]`), which is fixed by a *validated*
executor (Sonnet/gpt-5), NOT by root prompt work. **A better orchestrator partially compensates**
(it can recognize a stub result and re-drive `implement_baseline` with stronger guidance instead
of finalizing) — but the executor model remains the ceiling on *implementation* quality. Pursue
this plan AND the validated-executor lever together.

---

## 2. Why this is high-leverage
- The root drives EVERYTHING (decomposition, primitive sequencing, repair-vs-finalize decisions,
  steering of the executor). Small root-prompt gains compound across the whole run.
- gpt-chat-latest is OAuth-free, capable, and has huge rate limits (2.5M TPM) — the most
  attractive root on the current infra if hardened.
- The defects are **pre-GPU**, so iteration is cheap (CPU-only root loops, seconds-to-minutes,
  ~$0). This is a fast research loop, unlike GPU reproduction runs.

---

## 3. Current state (what already exists — don't rebuild)
- **Foundry root path:** `backend/agents/runtime/foundry_endpoint.py` (canonical creds resolver),
  `models.py` `azure-foundry` entry (`rlm_backend="openai"`, env-driven base_url+deployment),
  `run.py::_build_llm_client` (foundry primitive client). Reasoning-model params are handled
  everywhere: `OpenAILlmClient._is_reasoning_model` covers `gpt-chat*` → `max_completion_tokens`,
  no temperature; the rlms root loop + the Agents SDK executor both OMIT null params. **No param
  work remains** (the prior "chat refuses the loop" belief was a misdiagnosed `max_tokens` 400).
- **Root system prompt:** `backend/agents/rlm/system_prompt.py::build_system_prompt` — composes
  `_RLM_OPERATING_MODEL`, context-metadata table, `_CHAT_STEERING_SECTION`, `_PRIMITIVES_SECTION`
  (the `[[OPENRESEARCH_CUSTOM_TOOLS_SECTION]]` placeholder where rlms injects primitive signatures),
  `_TERMINATION_CONTRACT`, `_ITERATION_DISCIPLINE`, `_TURN_EFFICIENCY`, decomposition example,
  heartbeat, GPU-selection, optional hints, and a per-`RootModel.prompt_addendum` "MODEL-SPECIFIC
  ADDENDUM". **The whole prompt is brace-escaped then `.format(custom_tools_section=...)` by rlms —
  any new text must be brace-free (or it gets `{{`-escaped).**
- **Guardrail addendum (added 2026-06-19):** the `azure-foundry` `prompt_addendum` in `models.py`
  carries anti-refusal posture + ```repl discipline + the `FINAL_VAR("var")` call contract. This is
  the seed to extend.
- **Existing guardrails:** `forced_iteration.py` (refuses `FINAL_VAR` below target / before any
  experiment) + the degenerate-loop detector (`root_progress.py`, `OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD`);
  `evidence_gate.py` (vetoes unsubstantiated result leaves); operator steering via
  `runs/<id>/user_messages.jsonl` → `check_user_messages()`.

---

## 4. Observed root/orchestration defects (from run sdar_gcp_gptchat_v5_20260619)
Each is a concrete target. All were caught by downstream guards, but a *better root* would not
produce them:
1. **Placeholder names.** Passed `method_spec`/`paper_claim_map` name fields = literal `'unknown'`
   → `paper_grounding_failed` ("2 name(s) … not found in paper text: ['unknown','unknown']"). The
   root should extract REAL paper tokens during understand/plan and never emit placeholders.
2. **Wrong argument type.** Passed `compute_scope` as a prose string ("GPU multi-device run …")
   → `compute_scope_invalid` ("must be a dict or null; got 'str'"). The root doesn't know the
   primitive's argument schema precisely.
3. **Termination misuse.** Emitted `FINAL_VAR = report` (assignment) instead of `FINAL_VAR("report")`
   (call) — does not terminate. (The 2026-06-19 addendum starts addressing this; verify it sticks.)
4. **No stub recognition.** As orchestrator it accepted the executor's `success=True` stub result
   (0 GPU, placeholder metrics) and moved toward finalizing, rather than recognizing the weak
   result and re-driving `implement_baseline` with corrective `repair_context`.
5. **Degeneration.** Repeated `FINAL_VAR` on weak work → degenerate-loop abort. A better root
   converts "blocked from finalizing" into "do the missing work", not "retry FINAL_VAR".

---

## 5. Workstreams

### WS1 — Strict, model-tuned system prompt
Design a reasoning-chat-root prompt section (extend the `azure-foundry` `prompt_addendum`, or add a
dedicated reasoning-chat addendum keyed off `_is_reasoning_model`). Cover, concretely:
- **Primitive ARGUMENT CONTRACTS** — the exact types + shapes the root must pass (e.g.
  `compute_scope` is a dict|null; `method_spec`/`paper_claim_map` names must be REAL tokens that
  appear verbatim in `context["paper_text"]`, never `'unknown'`/placeholders). The rlms-injected
  signature section gives names, not the type/grounding nuances — this fills that gap.
- **Grounding-first** — extract and cache the paper's real names (method, components, models,
  envs, baselines, metrics) via understand/extract BEFORE any spec-building primitive.
- **Termination contract** — reinforce the call form; add the failure mode (`FINAL_VAR = x`).
- **Orchestration discipline** — the canonical loop (understand → plan → implement → run → verify
  → repair) and the repair-vs-finalize decision: a weak/low/None rubric or a suspicious experiment
  result means DO MORE WORK (propose_improvements + implement_baseline with repair_context + run),
  never retry FINAL_VAR.
- **Result-quality recognition** — how to read a `run_experiment` result and tell a real result
  from a stub (0-GPU on a GPU paper, placeholder metric keys, empty `cells.json`) and react.
- Keep it brace-free and concise (token cost rides every iteration). A/B strict vs minimal.

### WS2 — Guardrailing best practices
- **Argument-level pre-validation (highest value).** Add a binding-layer check (`binding.py`
  `wrap_primitive`, or per-primitive) that rejects placeholder/typed-wrong args BEFORE the deep
  primitive guard, returning a crisp, actionable repair message (e.g. "compute_scope must be a
  dict; you passed a string — pass {...} or None"). Faster, clearer self-correction than the
  current grounding-guard round-trip. Mirror `_validate_dockerfile_shape`'s pattern.
- **Stub-result guardrail (root-facing).** When `run_experiment` returns `success=True` on a
  GPU-required paper but used ~0 GPU / emitted placeholder metric keys, surface a loud
  `run_warning` + repair directive so the root re-implements instead of finalizing. (Complements
  the executor-side `_ENGINEERING_STANDARDS_BLOCK` self-verify rail added 2026-06-19.)
- **Anti-degeneration tuning.** Evaluate `OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD` and the
  experimental `OPENRESEARCH_OAUTH_AUTODRIVE` for chat roots — should a degenerate signal AUTO-DRIVE
  the missing stage (`recommend_next_tool` → implement) rather than abort? (See the 2026-06-17
  oauth-root reliability plan.)
- Keep anti-refusal (already in the addendum) and confirm it holds under adversarial framing.

### WS3 — Orchestration best practices (research component the user explicitly wants)
- Web-research the LATEST best practices for reasoning-model agents (OpenAI gpt-5 / o-series /
  chatgpt-latest): system-prompt design, tool-use discipline, structured-output reliability,
  planning + self-reflection, avoiding placeholder/hallucinated arguments, when reasoning models
  need explicit step contracts vs. when they self-organize. Use the deep-research skill or
  context7/WebSearch. Distill into the prompt + guardrails above.
- Cross-check against the RLM paper's own root-prompt guidance (arXiv 2512.24601 App C) already
  encoded in `system_prompt.py`.

### WS4 — Empirical methodology (CPU-only, cheap)
- Build a CPU-only root-loop harness: run gpt-chat-latest as root on a SMALL paper (or a trivial
  fixture) with the **executor stubbed/mocked**, so the loop exercises understand → plan →
  implement-call → (mocked run) → verify → terminate WITHOUT a GPU. The orchestration defects
  (placeholders, arg types, FINAL_VAR form, repair-vs-finalize, degeneration) all reproduce here.
- **Metrics to track (before/after each prompt+guardrail variant):** grounding-failure rate,
  arg-type-error rate, iterations-to-first-valid-implement_baseline, termination-contract
  compliance, stub-recognition rate, degeneration rate, total iterations to terminal.
- A/B variants deterministically; pick the winner; THEN validate once on a real GPU SDAR run
  (paired with a validated executor) to confirm end-to-end.

---

## 6. Deliverables
1. An optimized reasoning-chat-root system prompt / addendum (brace-safe, concise).
2. Guardrail enhancements (arg pre-validation + stub-result directive + anti-degeneration tuning).
3. A CPU-only root-loop validation harness + a before/after metrics table.
4. CHANGELOG + memory update; if defaults change, follow the A/B-before-flip rule.

## 7. Key files / pointers
- Prompt: `backend/agents/rlm/system_prompt.py`; addendum: `backend/agents/rlm/models.py`
  (`azure-foundry` `prompt_addendum`).
- Guardrails: `backend/agents/rlm/forced_iteration.py`, `root_progress.py`,
  `backend/agents/rlm/binding.py` (arg pre-validation site), `backend/agents/rlm/primitives.py`
  (the grounding / `compute_scope` guards + `_validate_dockerfile_shape` pattern to mirror),
  `backend/agents/rlm/evidence_gate.py`.
- Foundry path: `foundry_endpoint.py`, `models.py` (`azure-foundry`), `run.py::_build_llm_client`,
  `services/context/workspace/tools/openai_client.py` (`_is_reasoning_model`).
- Executor side (related, not this plan): `backend/agents/baseline_implementation.py`
  (`_ENGINEERING_STANDARDS_BLOCK`, `_NO_STUB_BLOCK`).
- Evidence: run `runs/sdar_gcp_gptchat_v5_20260619/` (boot disk on the GCP VM `sdar-a100-8g`;
  flip to CPU machine type to read without GPU billing — see
  `docs/local/2026-06-19-kimi-sdar-run-handoff.md`).
- Prior art: `docs/superpowers/plans/2026-06-17-oauth-root-reliability-and-harness-backstop.md`
  (degenerate detector + auto-drive), `docs/superpowers/specs/2026-05-28-rlm-stability-remediation-design.md`.
- Memory: `[[foundry-gptchat-root-not-executor]]`, `[[project_per_role_model_selection]]`.

## 8. Risks / open questions
- Optimizing the root cannot fix executor implementation quality — pair with executor=Sonnet/gpt-5.
- Prompt length is a per-iteration token cost; keep additions high-signal + A/B for regressions on
  the already-working roots (gpt-5/claude/grok) — the addendum is shared by all foundry deployments.
- Reasoning chat models can be non-deterministic; use the CPU harness with enough trials per variant
  to see real rate differences, not single-run noise.
