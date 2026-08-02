<!-- doc-meta: status=current; authored=2026-08-01 -->
# Remote/GCP-VM LLM auth — API keys ONLY, never OAuth

> # ⛔ NEVER USE OAuth — operator directive (2026-08-01)
> **Do NOT use `--model claude-oauth`. Do NOT use `CLAUDE_CODE_OAUTH_TOKEN`. Do NOT use
> Keychain `claude login`.** Not for the root, not for sub-agents, not for the grader, not
> ever — local or remote. OAuth is FORBIDDEN. Use **API keys only.** If `CLAUDE_CODE_OAUTH_TOKEN`
> is present in `.env`, treat it as DEAD and consider removing it so nothing auto-falls-back to it.

**Read this BEFORE provisioning any GPU VM.** The #1 way a paid GCP/Azure run dies is LLM auth.
This doc exists so the 2026-08-01 debugging cascade never repeats — pick a sanctioned API-key
surface, run the preflight.

## The ONLY sanctioned surfaces (API keys, never OAuth)

| Surface | Model | Auth (API key, NOT OAuth) | Status | Use |
|---|---|---|---|---|
| **`sonnet-foundry` / `opus-foundry`** | real Claude 5 / 4.8 via Foundry `…/anthropic` | `AZURE_FOUNDRY_API_KEY` | **LIVE + configured** | **RECOMMENDED today. Needs the thinking patch (below) — already landed.** |
| `--model claude` | real Claude (`claude-sonnet-4-6`) | `ANTHROPIC_API_KEY` | **empty — provide a funded key** | the standard non-OAuth Claude path once a key is funded |
| `grok` / `azure-foundry` | grok-4.3 (`AZURE_FOUNDRY_DEPLOYMENT`) | `AZURE_FOUNDRY_API_KEY` | LIVE | NOT ML-paper-validated → infra-`failed` verdicts as executor |
| `--model azure` | gpt-4o | `AZURE_OPENAI_*` | **INCOMPLETE** — `AZURE_OPENAI_ENDPOINT`/`_DEPLOYMENT` empty | unusable until endpoint+deployment are set |
| `--model gpt-5` | gpt-5 | `OPENAI_API_KEY` | **DEAD** — rotated `sk-svcacct-` (2026-07-21 leak) | do not use |
| `qwen3-coder-featherless` | qwen | `FEATHERLESS_API_KEY` | LIVE, cheap | not paper-validated |
| ~~`claude-oauth`~~ / ~~`CLAUDE_CODE_OAUTH_TOKEN`~~ / ~~Keychain `claude login`~~ | — | OAuth | ⛔ **FORBIDDEN** | **NEVER — operator directive** |

**Bottom line:** for a remote GPU run today, use **`--model sonnet-foundry`** (real Claude via
Foundry API key, thinking-patched). If you want the standard `--model claude` path, **fund an
`ANTHROPIC_API_KEY`**. Never OAuth.

```
--model sonnet-foundry --models executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry
# stage AZURE_FOUNDRY_ENDPOINT/_API_KEY/_DEPLOYMENT into the VM .env (NOT any OAuth token)
```

## The Foundry-Claude extended-thinking gotcha (found 2026-08-01, now patched)
Foundry's `claude-sonnet-5` / `claude-opus-4-8` deployments **default to extended thinking** on
complex prompts. This breaks the harness in a cascade — each symptom looks different:
1. **Root crashes at iteration 0:** `AttributeError: 'ThinkingBlock' object has no attribute 'text'`
   — the rlm client read `response.content[0].text`, but `content[0]` is a `ThinkingBlock`.
2. **Rubric generation returns empty/unparseable JSON** → `run proceeds rubric-less` → **no score.**
   The thinking block eats the `max_tokens` budget, truncating the JSON.

**Fix (landed 2026-08-01):** `backend/agents/rlm/_anthropic_thinking_patch.py` — `extract_text()`
skips thinking blocks, and a global `anthropic` SDK patch injects `thinking={"type":"disabled"}`
for `sonnet-5`/`opus-4-8` at the request boundary (covers every in-process client). Applied at
import in `run.py`; byte-identical for every other model. Empirically verified: default →
`blocks=['thinking','text']`, invalid JSON; `thinking:disabled` → `blocks=['text']`, valid JSON.
Tests: `tests/rlm/test_anthropic_thinking_patch.py`. **Caveat:** the `claude-agent-sdk` executor
is a subprocess (separate path) — smoke-validate a full `sonnet-foundry` run before fanning out.

## OFF-RIP PREFLIGHT — run before any remote GPU spend (no VM cost)
```bash
# 1. Sanctioned API-key surface present? (Foundry is the live one)
grep -E '^AZURE_FOUNDRY_(API_KEY|ENDPOINT)=.\+' .env && echo "Foundry key present"
grep -c '^ANTHROPIC_API_KEY=.\+' .env   # 1 = funded key present (the --model claude path)
# 2. NEVER OAuth — confirm nothing steers you to it:
echo "Do NOT stage CLAUDE_CODE_OAUTH_TOKEN to the VM; do NOT pass --model claude-oauth."
# 3. Azure OpenAI complete? (needs BOTH non-empty to be usable)
grep -E '^AZURE_OPENAI_(ENDPOINT|DEPLOYMENT)=.\+' .env || echo "Azure OpenAI INCOMPLETE — do not use --model azure"
# 4. GCP ready (see 2026-07-22-gcp-vm-e2e-run-procedure.md)
gcloud config list account project && gcloud auth application-default print-access-token >/dev/null && echo ADC OK
```

## Full credential inventory (2026-08-01, .env — names only)
- **Foundry (sanctioned):** `AZURE_FOUNDRY_ENDPOINT`/`_API_KEY`/`_DEPLOYMENT`(=`grok-4.3`) ✅ —
  grok (OpenAI-compat) + real Claude via `…/anthropic` with `claude-sonnet-5`.
- **Anthropic API (sanctioned, needs funding):** `ANTHROPIC_API_KEY` ❌ empty.
- **Azure OpenAI:** `AZURE_OPENAI_KEY1`/`KEY2` ✅ but `AZURE_OPENAI_ENDPOINT`/`_DEPLOYMENT` ❌ empty → **unusable**.
- **OpenAI:** `OPENAI_API_KEY` = dead rotated `sk-svcacct-` ❌.
- **Other:** `FEATHERLESS_API_KEY` ✅ (qwen, cheap), `TAVILY_API_KEY`/`APIFY_API_TOKEN` (ingestion).
- **⛔ OAuth (FORBIDDEN, do not use):** `CLAUDE_CODE_OAUTH_TOKEN` is present in `.env` but is
  **off-limits** per the operator directive — never stage it, never `--model claude-oauth`.
- **GCP:** `OPENRESEARCH_GCP_*` ✅ (project `deepinvent-ext-ut`, single-VM path).
- **Azure/AKS compute:** `az` not installed, no AKS `kubectl` context, `AZURE_AKS_*` empty → **AKS not usable here.**

See also: `2026-07-22-gcp-vm-e2e-run-procedure.md`, `2026-08-01-feature-ablation-gcp-runbook.md`,
`backend/agents/rlm/CLAUDE.md` (§"Anthropic on Azure Foundry"). Rule: **API keys only, never OAuth.**
