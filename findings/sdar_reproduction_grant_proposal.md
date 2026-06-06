# Grant Proposal — Autonomous Reproduction of Machine-Learning Research at Scale

| | |
|---|---|
| **Project** | OpenResearch / OpenResearch — an autonomous agent that reproduces ML papers end-to-end (ingest PDF → derive rubric → implement → train on GPU → score against the paper's own claims), with no human in the coding loop. |
| **Principal investigator** | A. Amatya |
| **Representative hard target** | *Self-Distilled Agentic Reinforcement Learning* (SDAR), arXiv 2605.15155v1 |
| **Calibration target (light)** | *Adam: A Method for Stochastic Optimization* (Kingma & Ba, ICLR 2015) — already reproduced end-to-end, rubric 0.741 |
| **Amount requested** | **≈ $45,000** (full-faithful program) · operable floor ~$5–15/run |
| **Period** | 12 months |
| **Prepared** | 2026-06-02 · LLM prices **web-verified** against the Anthropic pricing page on this date |
| **Technical appendix** | `findings/sdar_reproduction_budget.md` (token-budget methodology §0; GPU tiers §1–5) |

---

## Project summary / abstract

Reproducing a research paper is the gold standard of scientific verification and the most labor-intensive. This program funds an autonomous agent that performs the full reproduction loop — reading the paper, deriving a grading rubric, writing and repairing the training code, running it on real GPUs against real weights and data, and scoring the result — and asks a single fundable question: **what does one reproduction cost, and how does that cost scale with paper difficulty?**

We answer it empirically. Costs split into two independent surfaces: **LLM tokens** (the agent's reasoning and code generation) and **GPU compute** (the actual model training). Using *measured* token telemetry from real runs, we show the LLM surface is small and predictable — **under $5 per hard-paper run on Anthropic Claude Sonnet 4.6, and $0 on the subscription path used in development** — while GPU compute is the dominant, scope-driven term ($5 at mechanism level to ~$30k for a full faithful reproduction). The result is a program fundable at a known ceiling (~$45k) and operable at a near-zero marginal floor, with a model-routing playbook that lets investigators dial cost against fidelity.

---

## 1. Specific aims

**Aim 1 — Faithful autonomous reproduction of a hard agentic-RL paper.** Reproduce SDAR with real Qwen weights and real ALFWorld/WebShop/Search-QA episodes, enforcing the exact mechanism (SDAR loss `g_t = σ(β·Δ_t)`, stop-gradient gate, λ=0.1, β=10), scored by a leaf-level rubric that inspects the code *and* verifies real training.

**Aim 2 — Characterize the compute envelope** across paper difficulty (Adam → SDAR) and across LLM back-ends (Claude Sonnet 4.6 / Opus 4.8 and alternatives), so the program is fundable and operable at known bounds. *This proposal funds Aim 2 directly and bounds Aims 1 and 3.*

**Aim 3 — A cost-reduction playbook** (model routing, subscription-vs-metered, GPU-tier scoping) that keeps day-to-day iteration near zero marginal cost while preserving a fundable, fully-metered line for reproducibility.

---

## 2. Background & significance

Autonomous reproduction stresses two cost surfaces, billed separately:

| Surface | Role | Cost driver |
|---|---|---|
| **LLM tokens** | Root orchestrator writes Python each iteration; **Sonnet-class sub-agents write and repair the reproduction code** (dominant token sink); cheap models handle navigation | tokens processed × model rate |
| **GPU compute** | Builds the environment and **trains the actual models** (Qwen 1.7B–7B for SDAR) | GPU-hours × SKU rate |

SDAR is the canonical *hard* test — 3 Qwen scales × 3 agentic environments × 6 trained methods, GRPO RL with sigmoid-gated self-distillation, and a rubric a surrogate cannot pass. Adam is the *light* calibration point — one optimizer, MNIST/CIFAR, a single coding session. Bracketing the two bounds the whole program.

**Significance.** If the LLM line is small and the GPU line scales predictably with reproduction scope, autonomous reproduction becomes fundable per-paper at the same order as a graduate student running the experiments by hand — but at machine speed, around the clock, with a verifiable audit trail. That changes reproduction from a rare, heroic effort into routine infrastructure.

---

## 3. Research design & methods (cost-relevant summary)

The agent runs as a long-lived process invoking 12 domain primitives. The token-relevant routing:

- **`implement_baseline` / `run_experiment`** — full Claude-Agent-SDK coding sessions on **Sonnet 4.6** (the dominant token sink).
- **`understand_section` / `extract_hyperparameters` / `llm_query` / `rlm_query`** — navigation, routed to **cheap models** (Haiku 4.5 / gpt-5-mini).
- **`verify_against_rubric` / `propose_improvements`** — quality-critical grading; kept on **Sonnet-class or Opus**, never downgraded.

Reproduction is scoped in three tiers (full GPU detail in appendix §4):
- **Tier 1 — Lean / mechanism-level:** short training under the $10 GPU cap; clears code + real-weights-load rubric leaves → partial score.
- **Tier 2 — Scoped faithful ("smallest-two"):** Qwen3-1.7B + Qwen2.5-3B, GRPO+SDAR, full 150 steps on one 24–48 GB GPU. **Recommended iteration target.**
- **Tier 3 — Full faithful:** all 45–54 cells × 150 steps on 8× H100-class — the grant ceiling.

---

## 4. Budget I — LLM tokens (Claude, web-verified pricing)

### 4.1 Measured token envelope, by paper-difficulty range
Built bottom-up and **anchored on real, measured Sonnet 4.6 sub-agent sessions** (24 sessions across six committed runs: mean **~216 k tokens/session**, split **84 % cache-read / 3 % output / 13 % cache-create**). We budget across the full difficulty spectrum — Adam's root is measured on a completed run, the **Medium class is measured directly** (PPO runs), and the Hard class is scaled by paper weight. Full method in appendix §0.

| Paper-difficulty range | **Tokens / run** | output (3 %) | cache-read (84 %) | cache-create (13 %) | basis |
|---|---:|---:|---:|---:|---|
| **Light** (Adam-class) | **~0.6 M** (0.35–1.2 M) | ~18 k | ~0.50 M | ~78 k | root measured |
| **Medium** (PPO-class) | **~1.5 M** (1–2 M) | ~45 k | ~1.26 M | ~0.20 M | **sub-agent measured** |
| **Hard** (SDAR-class, smallest-two) | **~4 M** (2.5–8 M) | ~120 k | ~3.36 M | ~0.52 M | scaled |

The envelope is **~84 % cache-read** — the paper + system prompt re-sent each SDK turn. A single "try" is one full run at these figures; the proposal budgets **≥10 tries per paper-difficulty range** (reproduction is iterative — the hard run alone logged 11 attempts). Pathological within-run death-spirals add a further ×3–5 and are covered by contingency (§6, §9).

### 4.2 Verified Claude rates (Anthropic pricing page, 2026-06-02)
| Model | Base input | 5m cache-write | 1h cache-write | **Cache-read** | Output |
|---|---:|---:|---:|---:|---:|
| **Claude Sonnet 4.6** | $3 / MTok | $3.75 | $6 | **$0.30** | $15 / MTok |
| **Claude Opus 4.8** | $5 / MTok | $6.25 | $10 | **$0.50** | $25 / MTok |
| Claude Haiku 4.5 | $1 / MTok | $1.25 | $2 | $0.10 | $5 / MTok |

Discounts on the page: **prompt caching up to −90 %** (already in our cache-read-heavy profile), **batch −50 %**, US-only inference +10 %, **Opus fast mode = 2×**. **Tokenizer note:** Opus 4.7+ uses a new tokenizer that can consume **up to 35 % more tokens for the same text**; our measured profile is in Sonnet (old-tokenizer) tokens, so the Opus "same-work" cost is bracketed upward below.

### 4.3 Cost of ≥10 tries, per model × per paper-difficulty range
Each cell is **$/run · cost of 10 tries**, computed on the measured token profile (§4.1) with the web-verified rates (§4.2). Claude rows are verified 2026-06-02; non-Claude rows are indicative. Cheap models are fine for navigation but under-reproduce on coding/grading — keep quality-critical leaves on **Sonnet 4.6 / Opus 4.8**.

| Back-end (coding model) | **Light — 10 tries** | **Medium — 10 tries** | **Hard — 10 tries** |
|---|---:|---:|---:|
| `claude-oauth` (subscription) | $0.00 · **$0** | $0.00 · **$0** | $0.00 · **$0** |
| Qwen3-coder Featherless (flat) | $0.00 · **$0** | $0.00 · **$0** | $0.00 · **$0** |
| DeepSeek-V3.x *(indic.)* | $0.05 · $1 | $0.14 · $1 | $0.36 · $4 |
| Kimi K2.5 *(indic.)* | $0.18 · $2 | $0.44 · $4 | $1.18 · $12 |
| Claude Haiku 4.5 *(nav only)* | $0.24 · $2 | $0.59 · $6 | $1.59 · $16 |
| GPT-5 *(indic. root)* | $0.34 · $3 | $0.85 · $9 | $2.27 · $23 |
| Claude Sonnet 4.6 — batch (−50 %) | $0.36 · $4 | $0.89 · $9 | $2.38 · $24 |
| **Claude Sonnet 4.6 — standard** | **$0.71 · $7** | **$1.78 · $18** | **$4.76 · $48** |
| Claude Opus 4.8 — batch (−50 %) | $0.59 · $6 | $1.49 · $15 | $3.96 · $40 |
| **Claude Opus 4.8 — standard** | **$1.19 · $12** | **$2.97 · $30** | **$7.93 · $79** |
| Claude Opus 4.8 — +35 % tokenizer | $1.61 · $16 | $4.01 · $40 | $10.71 · $107 |
| Claude Opus 4.8 — fast mode (2×) | $2.38 · $24 | $5.95 · $59 | $15.86 · $159 |

**Read-out.** **10 tries on the hardest paper costs $48 on Sonnet 4.6 and $79 on Opus 4.8** (≤$107 even with the Opus tokenizer penalty). Covering **10 tries across *all three* difficulty ranges** is **$73 on Sonnet 4.6 / $121 on Opus 4.8** ($163 with tokenizer, $242 on Opus fast). Opus 4.8 runs **~1.7× Sonnet 4.6** (≈ 2.2× with the tokenizer) — so model choice is a **fidelity decision, not a budget one**; both are a rounding error against GPU (§5). **Recommended:** Sonnet 4.6 for coding/grading; reserve Opus 4.8 (optionally batch −50 %) for the hardest reproductions and final verification passes.

---

## 5. Budget II — GPU compute (per run, by tier)

From the system's RunPod COMMUNITY catalog (RTX 4090 $0.34/hr · A100-80 $1.89 · H100-80 $4.39; H800 ≈ H100-class). +20 % idle factor; ×1.5 Tier-3 contingency for failed-cell waste. Derivation in appendix §4.

| Tier | Scope | GPU-hr (modeled) | **GPU $/run** |
|---|---|---|---:|
| **1 — Lean** | short training under $10 cap | ≤29 hr @4090 / ~5 hr @A100-80 | **$3–10** |
| **2 — Scoped faithful** | smallest-two, 150 steps | 36–72 GPU-hr | **$68–136** (A100-80) / $158–316 (H100) |
| **3 — Full faithful** | 45–54 cells × 150 steps × 8 GPU | 3,000–8,000 GPU-hr | **$20,000–$52,000** (~$30k point) |

Adam's GPU need is trivial (single small GPU, minutes) — its run cost is essentially the LLM line in §4.

---

## 6. Budget summary & justification

| Line item | Configuration | Request |
|---|---|---:|
| **LLM — orchestration + coding** | GPT-5 root + **Sonnet 4.6** sub-agents (metered, cached); **≥10 tries × each of Light/Medium/Hard** = ~$73, scaled to the full ~60-run program incl. retries + occasional Opus 4.8 verification | **$2,500** |
| **GPU — Tier 1+2 iteration** | RTX 4090 / A100-80, ~40 scoped dev runs | **$5,000** |
| **GPU — Tier 3 full faithful** | 8× H100, 45–54 cells × 150 steps, ×1.5 contingency | **$30,000** |
| **Contingency (20 % overall)** | death-spiral retries (11 observed), OOM SKU escalation | **$7,500** |
| **TOTAL (full-faithful program)** | | **≈ $45,000** |
| *Single representative run (Tier 2, fundable)* | smallest-two, A100-80, metered LLM | **~$70–140 + ~$5 LLM** |
| *Single dev run (Tier 1, current practice)* | $10 GPU cap + OAuth sub-agents | **~$5–15** |

**Justification.** The LLM line is small and now *web-verified*: a hard-paper run processes ~4 M tokens — 84 % cheap cache-read — costing **$4.76 on Sonnet 4.6** or **$7.93 on Opus 4.8**, and **$0** on the OAuth subscription path used in development. **Budgeting ≥10 tries across all three difficulty ranges is just $73 (Sonnet 4.6) / $121 (Opus 4.8).** Even with full death-spiral retries the LLM line stays under $40/run; the $2,500 LLM budget therefore covers the entire program — dozens of tries per paper across difficulty ranges — with ample headroom for an all-Opus 4.8 verification pass. The dominant cost is GPU training, scaling transparently as `GPU$ = N_cells × hrs/cell × N_gpus × $/hr × contingency`. The Tier-3 bracket is wide because the paper reports no wall-clock time, so per-cell hours are modeled (2–4× swing; agentic RL is rollout-bound). The program is therefore fundable at a ~$45k ceiling and operable at a ~$5–15 floor.

---

## 7. Timeline & milestones

| Phase | Work | Tier | Budget draw |
|---|---|---|---|
| **M1 — Calibration (mo. 1–2)** | Reproduce Adam + 1–2 light papers; capture real metered Claude SDK `usage` to convert the SDAR token estimate into a measurement | 1 | <$50 |
| **M2 — Scoped SDAR (mo. 3–6)** | Smallest-two SDAR, full 150 steps; clear code + real-train rubric leaves | 2 | ~$150/run × ~10 |
| **M3 — Full faithful SDAR (mo. 7–10)** | Full cell matrix on 8× H100 | 3 | ~$30k |
| **M4 — Playbook (mo. 11–12)** | Publish model-routing + tier-scoping cost-reduction guide (Aim 3) | — | — |

---

## 8. Broader impacts

A per-paper reproduction cost on the order of a student-day of compute — with a machine-checkable rubric and full audit trail — lowers the barrier to independent verification of published ML results. The model-routing playbook (Aim 3) is directly transferable to any LLM-agent workload, and the measured token methodology (appendix §0) gives other groups a template for honest, telemetry-grounded compute budgeting.

---

## 9. Risk & cost-reduction

**Levers (built into the system).** OAuth sub-agents → Sonnet coding at **$0** (cuts the LLM line ~95 %); navigation already on Haiku/gpt-5-mini; `--model qwen3-coder-featherless` flat-rate root; batch processing −50 % on Claude; RunPod COMMUNITY vs SECURE −45 %; `cuda-runtime` image saves $0.50–1.50 + 5–10 min/run; `--max-usd` / `OPENRESEARCH_MAX_RUN_GPU_USD` hard-cap each tier; smallest-two scope keeps iteration in Tier 1/2.

**Risks & mitigations.** (a) *Per-cell GPU hours unknown* (paper reports none) — widest factor; mitigated by Tier-2 scoping and hard $ caps before any Tier-3 commit. (b) *Death-spiral retries* (11 attempts observed on the hard run) — covered by the 20 % contingency and ×1.5 Tier-3 factor. (c) *SDAR token model partly extrapolated* (no clean completed SDAR run yet; sub-agent anchor from PPO-class runs) — M1 captures real SDAR metered `usage` to convert estimate → measurement.

---

## Sources
- **Claude pricing (web-verified 2026-06-02):** [platform.claude.com/docs/en/about-claude/pricing](https://platform.claude.com/docs/en/about-claude/pricing) — Sonnet 4.6 $3/$15 (cache-read $0.30); Opus 4.8 $5/$25 (cache-read $0.50); tokenizer note (Opus 4.7+ up to +35 % tokens); fast mode 2×; batch −50 %; [anthropic.com/news/claude-opus-4-8](https://www.anthropic.com/news/claude-opus-4-8).
- **Token budget & methodology:** `findings/sdar_reproduction_budget.md` §0 (measured Sonnet sub-agent anchor: 24 sessions, ~216 k tok/session, 84/3/13 split).
- **GPU pricing:** in-repo `backend/services/runtime/gpu_catalog.py`, cross-checked vs [runpod.io/pricing](https://www.runpod.io/pricing).
- **Alternative back-ends:** indicative rates (GPT-5/Kimi/DeepSeek) — verify at M1.
- **Paper compute footprint:** arXiv 2605.15155v1 §Implementation Details (8× H800, 150 steps).
</content>
