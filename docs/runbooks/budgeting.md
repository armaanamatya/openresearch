# Budgeting — Claude Code subscription, API spend, and GPU cost

How to keep reproduction runs cheap. Covers the three independent cost surfaces:
**(1)** Claude Code OAuth subscription (rate-limited, not per-token), **(2)** Anthropic/OpenAI/Featherless API tokens (per-token, hard $), **(3)** RunPod GPU pods (per-hour, hard $).

See `CLAUDE.md` for the full auth-surface explanation. This doc is the "how do I spend less" reference.

## The two LLM auth surfaces

| Surface | What it runs | How it's billed | Knob |
|---|---|---|---|
| Root model | `rlm._completion_turn` — every REPL turn | OpenAI / Anthropic API key, **or** Claude Code subscription via `--model claude-oauth` | `REPROLAB_RLM_ROOT_MODEL` / `--model` |
| Sub-agents | `implement_baseline` and other Sonnet calls via `claude-agent-sdk` | `ANTHROPIC_API_KEY` if set and funded, **else** Claude Code subscription (OAuth) | Presence/absence of funded `ANTHROPIC_API_KEY` |

The subscription is consumed by **whichever surface routes to OAuth**. `--model claude-oauth` routes *both* surfaces to it — that's the most expensive subscription configuration.

## Levers, ranked by impact on subscription usage

### 1. Don't put the root on OAuth
`--model claude-oauth` makes every root turn (dozens per run) bill against the subscription. Move the root off:

```bash
export REPROLAB_RLM_ROOT_MODEL=gpt-5                  # OPENAI_API_KEY, ~$1/run
# or
export REPROLAB_RLM_ROOT_MODEL=qwen3-coder-featherless # FEATHERLESS_API_KEY, cheapest
```

Leave `ANTHROPIC_API_KEY` empty so sub-agents still use the subscription (free per-message, subject to rate limits).

### 2. Move sub-agents off the subscription entirely
Set a **funded** `ANTHROPIC_API_KEY`. `claude-agent-sdk` prefers the API key over OAuth, so subscription usage drops to zero.

**Pitfall (2026-05-22):** an unfunded key fails with 400 *"credit balance too low"* and does **not** fall back to OAuth. Working `claude --print "ping"` proves only the subscription works — the API key needs its own credits. If unsure, leave `ANTHROPIC_API_KEY=` empty.

### 3. Cap sub-agent fan-out per run
`implement_baseline` is the expensive call. Bound the run:

```bash
--max-wall-clock 1800   # 30 min
--max-usd 2             # hard cost cap
--max-pod-seconds 1800  # bounds GPU spend too
```

Prefer `--mode rdr` over `--mode rlm` during dev — RDR dispatches scoped coding agents per rubric cluster instead of the RLM hybrid's adaptive repair loop, so fewer total Sonnet calls per run.

### 4. Skip LLM entirely when iterating on plumbing
When you're testing prompts, primitives, SSE, or UI — not the model itself — don't burn sub-agent calls:

```bash
python -m backend.cli ingest paper.pdf   # zero LLM calls, exercises parser only
```

For RLM loop changes, a tiny synthetic rubric + a 2-page paper costs an order of magnitude less than SDAR.

## GPU cost (orthogonal to LLM cost)

Confirm RunPod settings — defaults are already cheap since 2026-05-22:

| Var | Default | Effect |
|---|---|---|
| `REPROLAB_RUNPOD_CLOUD_TYPE` | `COMMUNITY` | ~$0.34/hr on RTX 4090 (vs `SECURE` ~$0.69/hr) |
| `REPROLAB_RUNPOD_IMAGE` | `...cuda11.8.0-runtime-ubuntu22.04` | `runtime` ~4 GB vs `devel` ~18 GB; saves 5–10 min provision + $0.50–1.50/run |
| `REPROLAB_MAX_RUN_GPU_USD` | `10.0` | Hard per-run GPU spend cap (float; `0` disables) |
| `REPROLAB_MAX_GPU_USD_PER_HOUR` | `10.0` | Per-SKU $/hr cap for dynamic GPU selection |
| `REPROLAB_FORCE_SINGLE_GPU` | `true` | Hard-caps count=1 — set false only when paper genuinely needs multi-GPU |

## Recommended configurations

### Cheapest local dev (zero Anthropic API spend, subscription only)
```bash
REPROLAB_RLM_ROOT_MODEL=gpt-5
ANTHROPIC_API_KEY=                          # empty — sub-agents use OAuth
OPENAI_API_KEY=sk-...                       # ~$1/run for root
REPROLAB_RUNPOD_CLOUD_TYPE=COMMUNITY        # ~$0.34/run for GPU
```
Total: ~$1.34/run + subscription rate-limit consumption.

### Zero subscription usage (pure API spend, no rate-limit risk)
```bash
REPROLAB_RLM_ROOT_MODEL=gpt-5
ANTHROPIC_API_KEY=sk-ant-...                # funded — sub-agents bill per-token
OPENAI_API_KEY=sk-...
REPROLAB_RUNPOD_CLOUD_TYPE=COMMUNITY
```
Total: ~$1 (root) + ~$2–8 (Sonnet sub-agents, paper-dependent) + ~$0.34 (GPU).

### Absolute cheapest (subscription for everything)
```bash
REPROLAB_RLM_ROOT_MODEL=claude-oauth
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
REPROLAB_RUNPOD_CLOUD_TYPE=COMMUNITY
```
Total: ~$0.34/run GPU only, **but** heavy subscription burn — you'll hit rate limits faster.

## Sandbox gotcha

`REPROLAB_FORCE_SANDBOX` overrides `--sandbox` flags. The pydantic default is `"docker"`, so commenting the line out is **not** the same as disabling — every run silently pins to Docker. To honor per-run `--sandbox runpod` you must set it explicitly empty:

```bash
REPROLAB_FORCE_SANDBOX=
```

## Where this is enforced in code

- Run-level USD cap: `RunBudget.check_run_gpu_usd` (`backend/services/runtime/`)
- Per-primitive deadlines: `RunContext` (`backend/services/runtime/interface.py`)
- Wall-clock watchdog: hard-exits wedged runs at `--max-wall-clock`
- Cost ledger: `runs/<id>/cost_ledger.jsonl` — append-only per-primitive USD
