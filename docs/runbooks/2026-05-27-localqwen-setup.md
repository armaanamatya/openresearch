# Local Qwen setup — `--model local-qwen`

Branch: `localqwen`. Date: 2026-05-27.

Run ReproLab end-to-end with zero LLM-token spend by routing both surfaces to a local Qwen3-Coder via Ollama.

- **Root model**: Ollama OpenAI-compatible endpoint, direct.
- **Sub-agents** (`implement_baseline` + every `claude-agent-sdk` call): translated by a local LiteLLM proxy from Anthropic Messages API → Ollama.

> Quality is **not** Sonnet-equivalent. SDAR-class rubrics will fail leaves Sonnet passes. See `docs/design/2026-05-27-local-models-cost-reduction.md` for the trade-off table.

## Prerequisites

- Windows 11 + RTX 5060 Ti 16 GB (or any 16 GB+ NVIDIA card).
- Ollama installed (already present at `%LocalAppData%\Programs\Ollama\ollama.exe`).
- Either Python 3.10+ in your project venv **or** Docker Desktop (one of them — pick LiteLLM's host or container mode).

## 1. Pull the models

```powershell
ollama pull qwen3-coder:30b      # ~18 GB on disk, ~13 GB VRAM when loaded
ollama list                      # confirm present
```

The shipped `litellm_config.yaml` routes **both** the sonnet slot (`implement_baseline`) and the haiku slot (cheap primitives) to `qwen3-coder:30b`. Single-model setup — simplest, only one pull, only one set of weights in VRAM. The haiku-class primitives pay full 30B compute, but you avoid juggling two models.

If you want a dedicated cheap model later (faster haiku-class calls, lower VRAM when both are warm), pull e.g. `ollama pull qwen2.5-coder:7b` and edit the `claude-haiku-4-5-20251001` mapping in `scripts/localqwen/litellm_config.yaml` to point at it.

Note: the Ollama CLI accepts **one** model per `pull` invocation, not a space-separated arg list.

Ollama auto-loads on demand. First call to each model warms the weights into VRAM — expect a 10-30s cold-start.

If you have a custom Modelfile that raises `num_ctx` above 32K, set `REPROLAB_OLLAMA_CONTEXT_LIMIT` to match (otherwise the rlm orchestrator believes it has more headroom than Ollama actually serves).

## 2. Start the LiteLLM proxy

### Option A — pip (simpler)

```powershell
pip install 'litellm[proxy]'
litellm --config scripts/localqwen/litellm_config.yaml --port 4000
```

When running LiteLLM on the host, edit the YAML and change every `api_base: http://host.docker.internal:11434` to `api_base: http://localhost:11434`.

### Option B — Docker (no Python install needed)

```powershell
cd scripts/localqwen
docker compose up -d
docker compose logs -f litellm     # watch first few requests
```

`host.docker.internal` resolves to the Windows-host Ollama from inside the container — no extra wiring.

### Smoke test the proxy

```powershell
curl -s http://localhost:4000/v1/messages `
  -H "x-api-key: sk-localqwen-proxy" `
  -H "anthropic-version: 2023-06-01" `
  -H "content-type: application/json" `
  -d '{"model":"claude-sonnet-4-6","max_tokens":64,"messages":[{"role":"user","content":"say pong"}]}'
```

Expect a JSON response with `"content":[{"type":"text","text":"pong"}]` (or similar). If you get a connection error, the proxy isn't up. If you get a 400 with `model not found`, the YAML mapping is wrong.

## 3. Configure ReproLab

Add to `.env`:

```dotenv
REPROLAB_RLM_ROOT_MODEL=local-qwen

# Root → Ollama
OLLAMA_API_KEY=ollama                          # dummy bearer
# REPROLAB_OLLAMA_BASE_URL=http://localhost:11434/v1   # default — uncomment to override
# REPROLAB_OLLAMA_ROOT_MODEL=qwen3-coder:30b           # default
# REPROLAB_OLLAMA_CONTEXT_LIMIT=32768                  # default — must match Modelfile num_ctx

# Sub-agents → local proxy
ANTHROPIC_BASE_URL=http://localhost:4000
ANTHROPIC_API_KEY=sk-localqwen-proxy           # any non-empty value
```

**Important**: `ANTHROPIC_BASE_URL` is a process-global override. While it points at the proxy, **every** `claude-agent-sdk` call in this process goes to the proxy — including any code path that previously used the Anthropic API directly. Unset / comment when switching back to `--model claude-oauth` or `--model claude`.

## 4. Run a paper

```powershell
python -m backend.cli reproduce 2605.15155 --mode rlm --sandbox local
```

(`SDAR` is the canonical test paper per `CLAUDE.md`. It will likely fail rubric leaves Sonnet passes — that's the expected quality cliff. Use a smaller / less rigorous paper for first-pass functional testing.)

## 5. Verify it's actually local

While the run is going, check:

```powershell
# Ollama is being hit by both root and sub-agents:
ollama ps                         # should show qwen3-coder:30b and qwen2.5-coder:7b loaded
docker compose logs litellm | Select-String "POST /v1/messages" | Select-Object -Last 5
```

If you see traffic on Ollama but **not** on the proxy, the sub-agent path is misconfigured (probably `ANTHROPIC_BASE_URL` isn't being read — restart the backend process after editing `.env`).

If you see proxy traffic but Ollama isn't loading models, check the YAML `api_base` paths.

## Cost expectation

- Token spend: **$0**.
- Electricity / GPU heat: real, but negligible vs API spend.
- Wall-clock: ~2–5× slower than Sonnet API per primitive call on a 5060 Ti 16 GB. Total run time goes up; total dollars go down to zero.
- Rate limits: none — only VRAM contention if you push beyond 16 GB.

## Reverting

```dotenv
# In .env, comment all four of:
#   REPROLAB_RLM_ROOT_MODEL=local-qwen
#   ANTHROPIC_BASE_URL=...
#   OLLAMA_API_KEY=...
#   ANTHROPIC_API_KEY=sk-localqwen-proxy   (or restore your real Anthropic key)
```

Then restart any running `uvicorn` / CLI process so the changed env is re-read. The LiteLLM proxy can stay running idle — it costs nothing if no traffic arrives.

## Known limits / what to test before declaring done

1. **Tool-use fidelity**: `implement_baseline` uses Read/Write/Bash tool calls through the SDK. LiteLLM should pass tool calls through, but Qwen's tool-call format vs Anthropic's differs subtly. If you see "no tool call emitted" errors, this is the first place to look. Solution: switch to a Qwen variant fine-tuned for Anthropic-style tool use, or accept the brittleness.
2. **MCP servers**: if `APIFY_API_TOKEN` is set, the sub-agent runtime registers an MCP server with the SDK. The proxy should be transparent here, but I haven't tested it. If MCP calls fail, unset `APIFY_API_TOKEN` for `local-qwen` runs.
3. **Prompt caching**: Anthropic-style `cache_control` blocks are no-ops against Ollama. LiteLLM strips them via `drop_params: true`. No correctness issue, but the ~50% Anthropic prompt-cache savings don't translate — the local model re-processes the system prompt every iteration.
4. **Long-context behavior**: Qwen3-Coder native ctx is 256K but Ollama defaults to 32K (`num_ctx`). The root model orchestrator can easily exceed this on a deep run. The registered `MODEL_CONTEXT_LIMITS` value is honored by rlm's compaction, but verify by watching for `context length exceeded` 400s.
