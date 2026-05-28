# SDAR local-qwen run log — 2026-05-28

Paper: 2605.15155 (SDAR — Self-Distilled Agentic RL)
Root model: local-qwen (qwen3.5:9b via Ollama + LiteLLM proxy :4000)
Sandbox: RunPod COMMUNITY

---

## Attempt 1 — prj_09047604e591d969 (warm-restart of prior stuck session)

**Started:** 2026-05-28 00:19 UTC (original start, carried over from killed session)
**Re-launched:** 2026-05-28 01:10 UTC
**Killed:** 2026-05-28 01:18 UTC (user request — stale wall-clock, fresh project needed)

### Findings

#### [SERVER-SIDE BUG] rubric generation fails — unparseable JSON from qwen3.5:9b

```
generate_rubric_tree: unparseable JSON on attempt 1
generate_rubric_tree: unparseable JSON on attempt 2
generate_rubric_tree: unparseable JSON on attempt 3
generate_rubric_tree: all 3 attempts failed — last: unparseable JSON on attempt 3
run_pipeline_rlm: rubric generation failed — run proceeds rubric-less
```

**Root cause (confirmed via code inspection):** `generate_rubric_tree` calls `OpenAILlmClient.complete()` which hits Ollama at `:11434` (OpenAI-compat endpoint). The client never passes `num_ctx` to Ollama — Ollama loads the model with its built-in default context (typically 2048 tokens for a cold load). The rubric-gen user message is `paper_title + paper_text[:48000]` ≈ 12,000 tokens of input. 12k >> 2048 → Ollama silently truncates the prompt → model generates garbage → JSON parse fails. 

`_extract_json` in primitives.py IS robust to markdown fences (raw_decode scan); it raises ValueError("truncated JSON object") on EOF-at-brace, which `_extract_json_object` in rubric_gen catches and returns None → "unparseable JSON" log.

The `num_ctx` values in `litellm_config.yaml` only affect sub-agent calls through the LiteLLM proxy, NOT root-model calls to Ollama directly.

**Fix applied (Attempt 2 post-tick-1):**
- `backend/services/context/workspace/tools/openai_client.py`: added `extra_body: dict | None` param, forwarded to `chat.completions.create(extra_body=...)`.
- `backend/agents/rlm/run.py` `_build_llm_client`: detects Ollama endpoint (`:11434` in base_url) and passes `extra_body={"options": {"num_ctx": 32768}}`. This forces Ollama to allocate 32k context for all root-model LLM calls including rubric gen.

**Will take effect on Attempt 3** (attempt 2 already past rubric gen, running rubric-less).

#### [UI NOTE] paperTitle shows "paper_text" instead of actual title

`demo_status.json` → `paperTitle: "paper_text"` and `paper.title: "paper_text"`. The lab UI will show a generic placeholder title. Not a blocker but looks broken.

#### [OPS NOTE] warm-restart reuses startedAt from prior killed session

When the CLI detects `prior code/ present, no final_report.json` it resumes rather than archiving. Wall-clock budget (`--max-wall-clock 5400`) counts from original `startedAt`, so 55 min of budget was already consumed on re-launch.

**Fix:** Pass a flag or delete the old run dir before relaunching. Alternatively: always generate a new project ID for a fresh CLI invocation.

---

## Attempt 2 — archived + relaunched fresh

**Started:** 2026-05-28T01:17:51 UTC (fresh startedAt, prior artifacts archived to attempts/)
**pid:** 64448
**Changes from attempt 1:** archived prior attempt, full 90-min wall-clock budget, num_ctx bumped to 16384 in LiteLLM (but had no effect on rubric gen — see root cause above)

### Tick 1 (01:21 UTC)
- **Rubric gen: FAILED again** — same 3 unparseable JSON errors. Confirmed root cause above.
- **RLM loop: RUNNING** — iteration 1 completed in 62s. Model exploring paper context (check_user_messages → SHOW_VARS → paper_text[:3000] → paper_metadata). 5 events in dashboard_events.jsonl.
- **Cost:** $0.00 (local model, no API charges)
- **Screenshot:** `scripts/localqwen/screens/sdar2-tick-1.png`
- **Action taken:** Fixed `OpenAILlmClient` + `_build_llm_client` to pass `num_ctx=32768` to Ollama for root-model calls. Will take effect on Attempt 3.

### Tick 2 (01:28 UTC)
- **iter_count=7** — strong progress, 65 events. Model methodically chunking SDAR paper with `understand_section` (abstract 0-5k, method 5-15k, results 15-25k). Iteration 7 timing: 55s.
- **No RunPod pod yet** — model still in paper-reading phase, expected.
- **Cost:** $0.00
- **Decision:** NOT killing — run is healthy and making real progress rubric-less.
- **Screenshot:** `scripts/localqwen/screens/sdar2-tick-2.png`

### Tick 3 (01:31 UTC)
- **iter_count=9**, 92 events. Major milestone: `detect_environment` + `plan_reproduction` both completed successfully.
- **Environment:** Dockerfile using `python:3.11-slim` + git (CPU-only image — may need GPU fix before run_experiment).
- **Plan:** GRPO + gated OPSD auxiliary loss. Datasets: ALFWorld, WebShop, Search-QA. Expected: +9.4%/+7.0%/+10.2% over GRPO.
- **Warning:** `compute_scope_invalid` — model passed compute_scope as string not dict; plan_reproduction dropped it (non-fatal).
- **Next expected:** `implement_baseline` → Sonnet sub-agent via LiteLLM proxy → RunPod pod.
- **Cost:** $0.00
- **Screenshot:** `scripts/localqwen/screens/sdar2-tick-3.png`

### Tick 4 (01:34 UTC)
- **iter_count=10**, 112 events.
- **`implement_baseline` IN FLIGHT** — `worker_report_started` at 01:32:54. Qwen sub-agent writing SDAR code via LiteLLM proxy. This is the major milestone.
- **Prior failure (iter 9→10):** Model called `implement_baseline(plan, repair_context=...)` — `repair_context` is not a valid kwarg. Model self-corrected next iter and called correctly.
- **[SERVER NOTE] hallucinated kwarg:** System prompt should clarify `implement_baseline(plan)` signature explicitly to prevent wasted iterations. Not a crash — auto-recovered.
- **Cost:** $0.00 (sub-agent call in progress, no RunPod pod yet)
- **Screenshot:** `scripts/localqwen/screens/sdar2-tick-4.png`

### Tick 5 (01:38 UTC) — CRITICAL: implement_baseline sub-agent producing no files
- **iter_count=11**, 116 events. `implement_baseline` completed with error: "artifacts incomplete: missing runnable source file". `Files: []` — zero files written.
- **Root cause investigation:**
  - `code/` dir contains: `CLAUDE.md` (context-mode routing rules — injected from Desktop parent), `commands.json` (`["python train.py"]`), `paper.pdf`. No `.py` files.
  - LiteLLM proxy shows only **1** `POST /v1/messages?beta=true` call for the entire sub-agent session — should be 10-20+ rounds for a real implementation. Sub-agent barely ran.
  - **Hypothesis A:** qwen3.5:9b is not correctly handling multi-turn tool-use via the Anthropic Messages beta endpoint. The model may be generating prose instead of tool_use blocks, causing the sub-agent to terminate early.
  - **Hypothesis B:** The context-mode CLAUDE.md injected into code/ is confusing the sub-agent with tool redirections.
- **Model retrying at iter=11** — another `implement_baseline(plan)` call in flight.
- **Cost:** $0.00
- **Screenshot:** `scripts/localqwen/screens/sdar2-tick-5.png`

### Tick 6 (01:42 UTC) — CONFIRMED: qwen3.5:9b incompatible with claude-agent-sdk tool-use
- **iter_count=13**, 163 events. Still zero .py files. Model looping on implement_baseline.
- **New error this tick:** `AttributeError: 'str' object has no attribute 'get'` — model passed `simple_plan` with `reproduction_contract` as a plain string; primitives.py expects a dict. Non-root cause.
- **ROOT CAUSE CONFIRMED:** LiteLLM `/v1/messages?beta=true` call count = **1 total** across all implement_baseline attempts. The claude-agent-sdk sends a message, qwen3.5:9b responds with prose/markdown (not a `tool_use` block), SDK sees "no tool calls = task done", exits after 1 round-trip. Zero files written every time.
- **This is a structural incompatibility**: qwen3.5:9b via Ollama → LiteLLM does not produce Anthropic-format `tool_use` JSON blocks. The claude-agent-sdk requires these for the write/edit/bash tool calls that create the implementation files.
- **Required fix for attempt 3:** Sub-agents must use a model that natively speaks Anthropic tool-use. Options: (a) `claude-oauth` (subscription, free), (b) `ANTHROPIC_API_KEY` with credits. Root model can remain `local-qwen` (qwen3.5:9b via Ollama at :11434).
- **Screenshot:** `scripts/localqwen/screens/sdar2-tick-6.png`

### Attempt 2 conclusion
Run killed after iter 13 — stuck in implement_baseline loop with no path forward using qwen3.5:9b as sub-agent. Attempt 3 will use local-qwen root + claude-oauth sub-agents.
