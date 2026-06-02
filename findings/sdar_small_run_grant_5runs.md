# Mini-Grant Proposal — Five Scoped-Faithful Reproductions of SDAR

| | |
|---|---|
| **Project** | OpenResearch / ReproLab — autonomous end-to-end reproduction of ML papers (ingest → derive rubric → implement → train on real GPUs → score against the paper's own claims) |
| **Principal investigator** | A. Amatya |
| **Target** | *Self-Distilled Agentic Reinforcement Learning* (SDAR), arXiv 2605.15155v1 — **Tier-2 "smallest-two" scope** (Qwen3-1.7B + Qwen2.5-3B, GRPO + SDAR, full 150 steps, one 24–48 GB GPU) |
| **Scope of this ask** | **5 scoped-faithful runs**, costed across **all supported LLM back-ends** |
| **Amount requested** | **≈ $1,000** (A100-80, metered Sonnet 4.6) · operable floor **~$60–120** (RTX 4090 + OAuth) |
| **Period** | 2 months · **Prices web-verified 2026-06-02** (Anthropic, OpenAI, Moonshot) |

## Proposal

This mini-grant funds **five full reproductions of SDAR**, one of the hardest recent AI papers to reproduce. Each run starts from the published paper and rebuilds the work on its own: it downloads the real Qwen language models and the real task data (ALFWorld, WebShop, Search-QA), re-implements the paper's exact training method and settings, trains the **two smallest models** for the paper's full schedule (150 steps) on a single GPU, and grades the result against a detailed checklist that both reads the generated code and confirms the model actually trained on real data. We ask for five runs because that is the smallest number that tells a reliable reproduction apart from a lucky one: it covers the runs that stumble and have to retry many times (the hardest run so far needed **11 attempts**), it stays inside firm spending limits, and it lets us replace our current cost estimate with real, measured numbers.

Each run has two separate costs: the **AI models** that read the paper and write the code, and the **GPU time** that actually trains the models. The main finding is that the AI-model cost is tiny no matter which provider we use, while GPU time is what really drives the total. Five complete runs cost **$0 on a flat-rate subscription, no more than $24 on our recommended setup (Claude Sonnet 4.6), and at most $79 even on the priciest option** (Table 1) — so the choice of model is about quality, not cost. GPU time is the real lever: five runs cost **$60–120 on an RTX 4090, ~$90–175 on an A6000, or $340–680 on an A100-80** (Table 2). We request **≈ $1,000** — enough for five runs on an A100-80 with the recommended model plus a 20% cushion for retries — and the work can be done for as little as **~$60–120** by using a cheaper GPU and the free subscription option.

### Table 1 — LLM cost across all back-ends (5 runs, Tier-2 smallest-two; ~4 M tok/run)
| Coding / root model | $/run | **5 runs** | Notes |
|---|---:|---:|---|
| `claude-oauth` (subscription) | $0.00 | **$0** | per-message-free; rate-limited |
| Qwen3-coder Featherless (flat) | $0.00 | **$0** | flat ~$10–25/mo subscription |
| DeepSeek-V3.x *(indicative)* | $0.36 | **$2** | navigation/root only |
| Kimi K2.5 | $1.01 | **$5** | $0.60/$3.00, $0.10 cache-hit (verified) |
| Claude Haiku 4.5 *(nav only)* | $1.59 | **$8** | under-reproduces on coding |
| GPT-5 *(default root)* | $2.27 | **$11** | $1.25/$10 (verified) |
| Claude Sonnet 4.6 — batch (−50 %) | $2.38 | **$12** | async ≤24 h |
| **Claude Sonnet 4.6 — standard** | **$4.76** | **$24** | **recommended coding/grading** |
| Claude Opus 4.8 — batch (−50 %) | $3.96 | **$20** | |
| **Claude Opus 4.8 — standard** | **$7.93** | **$40** | hardest-paper verification |
| Claude Opus 4.8 — +35 % tokenizer | $10.71 | **$54** | new-tokenizer worst case |
| Claude Opus 4.8 — fast mode (2×) | $15.86 | **$79** | upper bound, all back-ends |

### Table 2 — GPU cost (5 runs, 36–72 GPU-hr/run; RunPod COMMUNITY) & program total
| GPU SKU | $/hr | $/run | **5 runs** |
|---|---:|---:|---:|
| RTX 4090 (24 GB) | $0.34 | $12–24 | **$60–120** |
| RTX A6000 (48 GB) | $0.49 | $18–35 | **$90–175** |
| A100-80 | $1.89 | $68–136 | **$340–680** |

| Program total (5 runs) | LLM | GPU | +20 % contingency | **Total** |
|---|---:|---:|---:|---:|
| **Recommended** (Sonnet 4.6 + A100-80) | $24 | $340–680 | ~$73–141 | **~$440–845 → request $1,000** |
| **Floor** (OAuth + RTX 4090) | $0 | $60–120 | — | **~$60–120** |

## Orchestration regimes — dynamic workflows & `/ultrathink` (Claude Opus 4.8)

Two Claude 4.8 features can be enabled to raise reproduction *quality*; both touch **only the LLM line, never GPU**.

- **Dynamic workflows** — an **Opus 4.8** feature (the "ultracode" effort mode). An Opus-4.8 **orchestrator** writes a script that spawns parallel subagents (capped **16 concurrent / 1,000 total per run**). Only the thin orchestrator runs on Opus 4.8; the token-heavy fan-out (coding, verification, lookups) routes to **Sonnet 4.6 / Haiku 4.5** — so cost is a **blend dominated by the cheap fan-out**, close to the Sonnet line.
- **`/ultrathink`** — a **31,999-token thinking budget per turn**. Thinking is billed as **output tokens** (the full amount, not the summary shown), so it fattens the small, expensive output slice. (The keyword is Claude-Code-CLI-only; the API equivalent is `budget_tokens`, and the cost math is identical.)

Blended rates from the measured 3/84/13 split: **Sonnet 4.6 = $1.19/M, Opus 4.8 = $1.98/M**. "Blended" = ~10% orchestrator on Opus 4.8 + ~90% fan-out on Sonnet/Haiku.

| Regime | tok/run | Sonnet | **Blended** | Opus | 5 runs (blended) |
|---|---:|---:|---:|---:|---:|
| Baseline (single RLM loop) | ~4 M | $4.76 | — | $7.93 | $24 |
| + `/ultrathink` (reasoning turns) | ~4.3–5.2 M | $9–23 | — | $15–38 | $45–115 |
| Dynamic workflow, modest (~3×) | ~12 M | $14 | **~$15** | $24 | ~$75 |
| Dynamic workflow, aggressive (~8×) | ~32 M | $38 | **~$41** | $63 | ~$205 |
| Workflow (8×) + `/ultrathink` | ~36 M | $50–70 | **~$55–75** | $85–115 | ~$275–375 |
| *GPU line (5 runs, Tier-2 A100-80)* | | | | | **$340–680** |

**Read-out.** Requiring Opus 4.8 for the orchestrator adds only a small premium over the Sonnet line, because Opus drives the *thin* reasoning layer while the token-heavy fan-out stays on Sonnet/Haiku. Even the heaviest regime (~$275–375 for 5 runs) lands **at or below the GPU line and well under the $1,000 ask**; on the Max subscription (OAuth) it is **$0 marginal**, just consuming more of the rate limit. Caveats: ultracode targets long runs (>30 min, million-token budgets) — it *intends* to spend more LLM tokens for quality — and the 16-concurrent / 1,000-total subagent cap is a natural per-run ceiling.

### Sources
- **Anthropic** (Sonnet 4.6 $3/$15·$0.30; Opus 4.8 $5/$25·$0.50; Haiku 4.5 $1/$5; +35 % tokenizer) — web-verified 2026-06-02, [platform.claude.com/docs/en/about-claude/pricing](https://platform.claude.com/docs/en/about-claude/pricing).
- **OpenAI** GPT-5 $1.25/$10 — [openai.com/api/pricing](https://openai.com/api/pricing/), [pricepertoken](https://pricepertoken.com/pricing-page/model/openai-gpt-5).
- **Moonshot** Kimi K2.5 $0.60/$3.00 ($0.10 cache-hit) — [tokenmix.ai](https://tokenmix.ai/blog/kimi-k2-api-pricing), [devtk.ai](https://devtk.ai/en/models/kimi-k2-5/).
- **GPU** RunPod COMMUNITY catalog — in-repo `gpu_catalog.py`, cross-checked vs [runpod.io/pricing](https://www.runpod.io/pricing).
- **Token profile** measured anchor: `findings/sdar_reproduction_budget.md` §0 (24 Sonnet sessions, ~216 k tok/session, 84/3/13 split).
- **Extended thinking / `/ultrathink`** (thinking billed as output tokens; 31,999-token budget) — [extended-thinking docs](https://platform.claude.com/docs/en/build-with-claude/extended-thinking), [ultrathink ref](https://claudelog.com/faqs/what-is-ultrathink/).
- **Dynamic workflows** (Opus 4.8 "ultracode"; 16 concurrent / 1,000 total cap; orchestrator on Opus, fan-out on Sonnet/Haiku) — [Claude Code sub-agents](https://code.claude.com/docs/en/sub-agents), [MarkTechPost launch](https://www.marktechpost.com/2026/05/28/anthropic-ships-claude-opus-4-8-alongside-dynamic-workflows-and-cheaper-fast-mode-with-workflows-capped-at-1000-subagents/).
