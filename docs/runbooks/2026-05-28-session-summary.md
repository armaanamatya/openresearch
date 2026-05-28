# Session summary — 2026-05-28: local-qwen SDAR reproduction attempt

## Goal
Run SDAR (arXiv 2605.15155) entirely on local hardware to eliminate cloud LLM costs. Architecture: qwen3.5:9b on Ollama as the root reasoning model; same model re-used for sub-agents (implement_baseline, etc.) via a LiteLLM proxy that translates Anthropic Messages API → Ollama OpenAI-compat.

---

## Infrastructure set up

| Component | Config |
|-----------|--------|
| Root model | qwen3.5:9b via Ollama at :11434 (OpenAI-compat) |
| Sub-agent proxy | LiteLLM container at :4000, translates `/v1/messages` → Ollama `/api/chat` |
| Sandbox | RunPod COMMUNITY (~$0.34/hr RTX 4090) |
| GPU image | `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` |

**Why qwen3.5:9b and not qwen3-coder:30b** (decided in prior session): the 30B MoE model needs 17.3 GB GPU; RTX 5060 Ti has 16 GB. 9B dense model fits at 8.8 GB. Freed ~36 GB disk by deleting the 30B model.

**Why `think: false` in LiteLLM config**: qwen3.5:9b is a thinking model — it emits `<think>…</think>` blocks before text. The `rlm` library's `AnthropicClient` reads `response.content[0].text` without filtering by content type, so a thinking-first response crashes it. `think: false` tells Ollama to suppress the thinking block, giving a clean `[text]` content item.

---

## Bugs found and fixed

### 1. Rubric generation — always "unparseable JSON" (new this session)

**Symptom:** Every run logged:
```
generate_rubric_tree: unparseable JSON on attempt 1/2/3
run_pipeline_rlm: rubric generation failed — run proceeds rubric-less
```

**Root cause (traced through code):** `generate_rubric_tree` calls `OpenAILlmClient.complete()`, which hits Ollama at `:11434` directly — *not* through LiteLLM. The `num_ctx` values in `litellm_config.yaml` only affect sub-agent proxy calls. For root-model Ollama calls, Ollama loads with its internal default context (~2048 tokens). The rubric-gen prompt sends up to 48 000 chars of paper text ≈ 12 000 tokens — far exceeding 2048. Ollama silently truncates the input; the model generates garbage; `_extract_json` raises `ValueError("truncated JSON object")`.

**Fix (two files):**
- `backend/services/context/workspace/tools/openai_client.py`: added `extra_body: dict | None` parameter, forwarded to `chat.completions.create(extra_body=...)`.
- `backend/agents/rlm/run.py` `_build_llm_client`: when constructing `OpenAILlmClient` for an Ollama endpoint (`:11434` in base_url), inject `extra_body={"options": {"num_ctx": 32768}}`. This forces Ollama to allocate a 32k-token context for all root-model calls including rubric gen.

**Status:** Fix committed. Will take effect on attempt 3.

### 2. REPL history context overflow (observed, not fixed)

By iter 13, `history` variable in REPL is 68 572 bytes. qwen3.5:9b's effective context window is finite; as REPL history grows the model increasingly hallucinates wrong primitive signatures (`repair_context` kwarg that doesn't exist, `reproduction_contract` as a string when a dict is expected). This is a degradation pattern, not a hard crash.

---

## Run attempts

### Attempt 1 — killed at launch
Reused project `prj_09047604e591d969` from a prior stuck session. `startedAt` was already 55 minutes old, consuming most of the `--max-wall-clock 5400` budget. Rubric gen failed (same unparseable JSON). Killed immediately; relaunched fresh.

### Attempt 2 — ran 25 min, killed (iter 13)

**Timeline:**

| Iter | What happened |
|------|---------------|
| 1–8 | Model reads paper in chunks via `understand_section`, builds `paper_claim_map` |
| 9 | `detect_environment` ✓ → CPU Dockerfile. `plan_reproduction` ✓ → GRPO + gated OPSD plan |
| 10 | `build_environment` ✓ → Docker image built. `implement_baseline` ✗ — TypeError: unexpected kwarg `repair_context` (model hallucinated signature) |
| 11 | `implement_baseline(plan)` ✗ — "artifacts incomplete: missing runnable source file" (0 files written) |
| 12–13 | More `implement_baseline` retries, same result. Model also tries `propose_improvements` (returns empty without a prior run result) and passes malformed plan → AttributeError |

**Root cause of implement_baseline failure (confirmed):**

The claude-agent-sdk sub-agent makes requests to `POST /v1/messages?beta=true`. LiteLLM translates this to Ollama and returns qwen3.5:9b's response. LiteLLM proxy logs showed **1 total `/v1/messages?beta=true` call** across all three implement_baseline attempts combined. A working sub-agent session would involve 10–20+ rounds (read task → write file → verify → write next file).

After 1 round-trip, the sub-agent terminates with zero files. Cause: qwen3.5:9b responds to the tool-use prompt with **prose** (e.g., "I'll implement the SDAR baseline…" followed by a markdown code block), not a structured Anthropic `tool_use` content block. The claude-agent-sdk sees "no tool calls in response" → considers task complete → exits. Nothing gets written to disk.

This is a fundamental incompatibility. The claude-agent-sdk requires models that natively emit `tool_use` JSON content blocks in the Anthropic response format. Ollama + LiteLLM does not coerce qwen3.5:9b's prose output into that format.

---

## What worked

- Full paper ingestion (arXiv fetch, parse, index) ✓
- Root model reasoning loop (13 iterations, ~60–160s each) ✓
- `understand_section` primitive (paper chunking) ✓
- `detect_environment` + `plan_reproduction` ✓
- `build_environment` (Docker image built successfully) ✓
- LiteLLM proxy routing for sub-agent calls (Anthropic → Ollama, correct endpoint) ✓
- RunPod credentials, SSH key, cloud type all verified ✓

---

## What doesn't work

- **Rubric generation** — fixed (num_ctx injection), untested in attempt 3
- **implement_baseline sub-agent** — qwen3.5:9b cannot produce Anthropic tool_use blocks; sub-agent always writes 0 files

---

## Recommended fix for attempt 3

**Use claude-oauth for sub-agents; keep local-qwen as root.**

The CLAUDE.md project spec describes this as the "cheapest local-dev cost model": root model on OpenAI/local (~$1/run), sub-agents on Claude OAuth subscription ($0 extra). No API key credits consumed.

Change needed in `.env`:
```
ANTHROPIC_API_KEY=        # clear — OAuth doesn't need a key
ANTHROPIC_BASE_URL=       # clear — let SDK route to real Anthropic via OAuth
```

The `local-qwen` root model still calls Ollama at :11434 directly (unaffected by ANTHROPIC_* env vars). Sub-agents will fall through to claude-oauth. The LiteLLM proxy and `think: false` config can remain in place for any future attempt to fix Qwen tool-use — but are not in the critical path.

Alternative (harder): patch LiteLLM to force `tool_use` block output from Ollama via grammar-constrained decoding or a response-format wrapper. Not recommended for near-term runs.
