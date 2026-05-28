# Local Models & Cost-Reduction Options

Status: 2026-05-27, branch `localqwen`.
Author of options table: design notes, not yet implemented across the board.

## Goal

Reduce per-run cost / token-spend for ReproLab while keeping rubric quality close to today's Sonnet-4.6 baseline, and without consuming the user's Claude Code subscription quota for sub-agent traffic.

## Surfaces

A run touches **two independent LLM auth surfaces** (see top-level `CLAUDE.md` §"RLM auth"):

1. **Root model** — the `rlm` library's chat backend; raw HTTP to an OpenAI- or Anthropic-shaped endpoint. Selected via `--model` / `REPROLAB_RLM_ROOT_MODEL`. Already supports OpenAI-compatible custom `base_url` (Featherless, OpenRouter, vLLM-as-OpenAI).
2. **Sub-agents** — `implement_baseline`, the code-writing agent, `understand_section` etc. Run inside `claude-agent-sdk` (`backend/agents/runtime/claude_runtime.py`). Anthropic Messages API only; honors `ANTHROPIC_BASE_URL` so a translating proxy works.

## Levers already in the repo (audit before adding more)

| Lever | Code | Status |
|---|---|---|
| Per-primitive content-addressed cache | `backend/agents/rlm/primitive_cache.py` | Live for `understand_section`, `extract_hyperparameters`, `detect_environment`, `plan_reproduction`, `verify_against_rubric`, `implement_baseline`. Verify `REPROLAB_PRIMITIVE_CACHE` is unset/enabled. |
| Patch-mode `implement_baseline` | `primitives.py` (PR-ι.2) | Active on retry — diffs prior `train.py` instead of full rewrite. Big lever already on. |
| Haiku for sub-calls | `models.py` `sub_backend_kwargs={"model_name":"claude-haiku-4-5-20251001"}` | Set for `claude`/`claude-oauth` roots. The `agent_model` (feeding `implement_baseline`) still defaults to **Sonnet 4.6** — the highest-spend slot. |
| Anthropic prompt caching | `_oauth_backend_patch.py::apply_anthropic_caching_patch`, plus `cache_control` in `baseline_implementation.py`, `rdr/agent.py`, `claude_runtime.py` | Partially wired. The API-key path is patched; the OAuth path delegates to the SDK. ~50% input savings claimed; worth measuring. |
| RDR mode (`--mode rdr`) | `backend/agents/rdr/` | Deterministic Python controller, fewer LLM round-trips than RLM. Use as a baseline before paying for RLM exploration. |

## External options ranked by $/quality (no subscription burn)

### 1. GLM Coding Plan — $18/mo flat (Anthropic-compatible drop-in)

Z.AI exposes an Anthropic Messages-shaped endpoint at `https://api.z.ai/api/anthropic`. `claude-agent-sdk` honors `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN`, so the sub-agent path swaps with no code changes. GLM-4.6/4.7 land around 90% of Sonnet 4.6 on code benchmarks.

- **Catch:** Lite-tier rate limit (~400 prompts/5h). A multi-iteration paper reproduction can hit it.
- **Plumbing:** env vars only. No model-registry change.
- **Best for:** day-to-day exploration, single-paper-at-a-time runs.

### 2. DeepSeek V4-Flash / V4-Pro via OpenRouter — 5–10× cheaper than Sonnet

V4-Flash on SWE-Bench Verified is within 0.6 pt of Sonnet 4.6; V4-Pro within 0.2 pt of Opus 4.6. Prices (May 2026, post-promotion): V4-Flash $0.14/M input cache-miss, $0.0028/M cache-hit, $0.28/M output.

- **Root model:** add a `ROOT_MODELS["deepseek-v4-flash"]` entry using the existing `openrouter` backend (~10 lines).
- **Sub-agent:** still needs a proxy (LiteLLM) translating Anthropic → DeepSeek's OpenAI-compatible API. Same engineering as the local-Qwen path.

### 3. Local Qwen3-Coder-30B-A3B via Ollama on the dev box — $0/run on tokens

Hardware target: RTX 5060 Ti 16 GB. The 30B MoE (3B active params, Q4_K_M ~13 GB VRAM) runs comfortably on 16 GB. Ollama serves OpenAI-compatible chat completions on `http://localhost:11434/v1`.

- **Root model:** OpenAI-compatible with `base_url=http://localhost:11434/v1`. Trivial — add a `ROOT_MODELS["local-qwen"]` entry. **This branch ships this.**
- **Sub-agent:** route the `claude-agent-sdk` traffic through a local LiteLLM proxy that translates Anthropic Messages → Ollama's OpenAI API. **This branch ships this via a LiteLLM `config.yaml`.**
- **Catch:** A 30B local model will fail rubric leaves Sonnet passes — SDAR-class papers especially. Worth running for cost-free exploration, dev iteration, and any paper that doesn't actually need Sonnet code synthesis.

### 4. Featherless Qwen3-Coder root + OAuth Sonnet sub-agents (status quo, already plumbed)

`--model qwen3-coder-featherless` cuts root spend ~95% vs GPT-5/Sonnet root. Sub-agents stay on the Claude Code subscription. Lowest engineering cost, but still burns subscription quota.

### 5. Aggressive prompt-cache audit on the current Anthropic API path

If staying on Anthropic API for sub-agents, make sure the long static blocks (system prompt + tool defs + paper excerpt slices) carry `cache_control: {"type":"ephemeral"}`. Anthropic gives 90% off cached inputs; the patch is in `_oauth_backend_patch.py` for the API-key path only — the SDK manages caching for the OAuth path. **Action:** measure `cache_creation_input_tokens` vs `cache_read_input_tokens` on a real run to see if caching is actually hot.

## Honest recommendation

| Goal | Pick |
|---|---|
| Lowest cost, willing to accept quality cliff | **Local Qwen** (this branch) |
| Lowest cost while keeping ~Sonnet quality | **GLM Coding Plan** ($18/mo) |
| Best $/quality on the API metering model | **DeepSeek V4-Flash via OpenRouter + LiteLLM proxy for sub-agents** |
| Zero engineering, "just lower the bill" | Audit Anthropic prompt cache + force `agent_model=haiku` on non-baseline primitives |

A 30B local model **will not match Sonnet on SDAR-class rubrics**. Treat the local-Qwen path as: cost-free iteration loop while debugging, plus a quality-floor measurement so the next pick (GLM / DeepSeek) has a baseline to beat.

## What this branch (`localqwen`) does

1. Adds `RootModel.sub_api_key_env` so a single registry entry can mix providers across root and sub-call (required because root → Ollama, sub → Anthropic-shaped proxy).
2. Registers `ROOT_MODELS["local-qwen"]`: root via Ollama OpenAI-compat, sub-call via `anthropic` backend pointed at a local LiteLLM proxy.
3. Ships `scripts/localqwen/litellm_config.yaml` — translates Anthropic model names (`claude-sonnet-4-6`, `claude-haiku-4-5-*`) to local Ollama Qwen variants.
4. `.env.example` additions for `OLLAMA_BASE_URL`, `ANTHROPIC_BASE_URL` (proxy), and dummy keys.
5. Runbook: `docs/runbooks/2026-05-27-localqwen-setup.md`.

It does **not** implement a new `OllamaAgentRuntime` — that's a ~500–1500 LOC project deferred until the proxy approach is validated end-to-end on a real paper.
