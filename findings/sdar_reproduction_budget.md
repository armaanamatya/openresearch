# Grant-Style Budget — Full Reproduction of SDAR through ReproLab

**Paper:** *Self-Distilled Agentic Reinforcement Learning* (SDAR), arXiv **2605.15155v1**, 14 May 2026.
**System:** OpenResearch / ReproLab (RLM orchestrator).
**Prepared:** 2026-05-29. **Prices:** current as of May 2026 (sources at end).
**Classification of the paper:** *hard*. It stresses every dimension at once — 3 Qwen scales (1.7B / 3B / 7B) × 3 agentic environments (ALFWorld, WebShop, Search-QA) × 6 trained methods (GRPO, OPSD, Skill-SD, GRPO+OPSD, RLSD, **SDAR**) + untrained Vanilla, GRPO RL with sigmoid-gated OPSD, and a leaf-level rubric that *reads the code* **and** *checks that real Qwen weights + real ALFWorld episodes actually load and train*.

> **Paper's own compute footprint (extracted from §Implementation Details):** "We train the Qwen2.5-Instruct and Qwen3-Instruct series using SDAR for 150 steps on **8× H800 GPUs**." ALFWorld: batch 16 tasks × 8 rollouts/prompt, max prompt 2,048 tok. Search-QA: batch 128 tasks, max prompt 4,096 tok. WebShop: 1,000 tasks. **The paper reports no wall-clock time** — so per-cell GPU hours below are *modeled* and bracketed wide.

---

## 1. Cost anatomy of one ReproLab run

A run has two independent cost surfaces (see `CLAUDE.md` → "RLM auth — two surfaces, billed separately"):

| Surface | What it is | Billing |
|---|---|---|
| **Root model** (RLM orchestrator, `rlm` lib) | Writes Python each iteration, drives the 12 primitives | Per-token API, **or** $0 on `claude-oauth` / flat Featherless |
| **Sub-agents** (`implement_baseline`, Sonnet coding agent) | Writes the actual reproduction code | Per-token API, **or** $0 on `claude-oauth` subscription |
| **GPU sandbox** (RunPod) | Builds env + runs/trains the experiment | Per-second wall-clock at the SKU rate |

### Observed (not modeled) — LLM half
Aggregating the **metered** cost ledgers on disk (`runs/prj_*/cost_ledger.jsonl`):

| Run | Rows | Output tok | **$ (metered)** |
|---|---|---|---|
| prj_c9befbf837180b39 | 7 | 51,565 | **$1.94** |
| prj_cdeb54fe2cd74ec3 | 5 | 42,005 | **$1.46** |
| prj_58783d551f293208 | 5 | 30,940 | **$1.17** |
| prj_01a6d176008af0b3 | 3 | 27,283 | **$1.13** |
| prj_126b2082c6242d80 | 3 | 23,392 | **$1.09** |
| prj_a984fa8d22973f7e | 1 | 11,161 | **$0.35** |

**Observed root-model LLM cost ≈ $0.35–$1.94/run (median ~$1.2)**, with sub-agents on OAuth (=$0). This matches `CLAUDE.md`'s "~$1/run via `--model gpt-5`". These runs did **not** complete heavy GPU training, so they ground the *LLM* half only; the GPU half below is modeled.

### Primitive-call shape (from real SDAR run `prj_09047604`, summed over attempts)
`understand_section ×25 · extract_hyperparameters ×7 · detect/build/plan_env ×5 each · implement_baseline ×10 · run_experiment ×3 · verify_against_rubric ×1 · propose_improvements ×2`. This is the per-run work profile used to model token volume below.

### Retry reality
The canonical SDAR run `prj_09047604e591d969` has **11 attempt directories** (the 2026-05-28 "SDAR death-spiral"). Contingency multipliers below are grounded in this, not guessed.

---

## 2. Model-role price sheet (sensitivity on the root model)

Current per-million-token rates (input / output):

| Model | $/M in | $/M out | Role fit | Marginal $ for root (≈3 M in, 90% cached / 250 k out) |
|---|---|---|---|---|
| **Kimi K2.5** | 0.60 | 3.00 | cheapest capable API root | **~$1.0–2.0** |
| **Kimi K2.6** | 0.95 | 4.00 | flagship agentic/coding | ~$1.5–2.5 |
| **GPT-5** | 1.25 | 10.00 | default root (observed) | **~$1.2–3.0** |
| GPT-5.4 | 2.50 | 15.00 | stronger planner | ~$4–6 |
| **Claude Sonnet 4.6** | 3.00 | 15.00 | **coding sub-agents** | (driver — see §3) |
| **Claude Opus 4.7** | 5.00 | 25.00 | hardest-paper root/verify | ~$8–12 |
| Claude Haiku 4.5 | 1.00 | 5.00 | cheap verification cross-check | ~$1–2 |
| **claude-oauth** (Sonnet/Opus on CLI sub) | — | — | root **and/or** sub-agents | **$0 marginal** (subscription rate-limited) |
| **Qwen3-coder (Featherless)** | flat sub | flat sub | cheapest root | **~$0 marginal** (flat ~$10–25/mo) |

Prompt caching (90% off cached input on every vendor) and batch/flex (50% off) are assumed where applicable.

---

## 3. Two LLM-cost regimes

Because a grant cannot cleanly fund a personal subscription, the **fundable** number is metered API; the **dev-path** number is what you actually pay "how we're doing it."

| Regime | Root | Sub-agents (coding) | LLM $/healthy run |
|---|---|---|---|
| **A. Fundable (all metered API)** | GPT-5 ($1.25/$10) | Sonnet 4.6 metered (~10 implement calls, ~12 M in @90% cache + ~0.7 M out) | **~$20–40** (Sonnet-dominated) |
| **B. Dev path (observed)** | GPT-5 metered | Sonnet via **OAuth** ($0) | **~$1–2** |
| **C. Zero-marginal** | `claude-oauth` or Featherless flat | OAuth ($0) | **~$0** (+ flat subscription) |

> The grant should budget **Regime A** for the LLM line; Regimes B/C are the cost-reduction story.

---

## 4. GPU budget — three tiers (the real grant lever)

RunPod COMMUNITY rates from the system's own catalog (`backend/services/runtime/gpu_catalog.py`): RTX 4090 (24 GB) **$0.34/hr**, A6000 (48 GB) $0.49, L40S (48 GB) $0.86, A100-40 **$1.19**, A100-80 **$1.89**, **H100-80 $4.39**, H200 (141 GB, SECURE) $7.99. The paper's H800 ≈ H100-class.

**Pod-lifecycle assumption:** RunPod bills wall-clock; pods are *owned and swept* (`pod_sweeper.delete_pod`, `_owned_pod_ids`). A long run that **holds one pod across multiple `run_experiment` calls** pays *idle GPU* while the LLM writes code between experiments. We add a **+20% idle factor** to active GPU-hours. (Per-experiment spin-up instead recurs 5–10 min provisioning each call — comparable order; +20% covers either.)

### Tier 1 — Lean / mechanism-level (the system's default guardrails)
ReproLab ships `REPROLAB_MAX_RUN_GPU_USD=10.0` and `REPROLAB_MAX_GPU_USD_PER_HOUR=10.0`. Under the $10 cap you get ≈ **29 GPU-hr on RTX 4090** or **~5 hr on A100-80** — enough to load **real Qwen3-1.7B weights + real ALFWorld episodes** and run a **short (~10–30 step) training** that demonstrates the SDAR loss `g_t = σ(β·Δ_t)`, stop-gradient gate, λ=0.1, β=10.

- **Rubric reality:** this **clears the code-inspection leaves and the "real weights/data load + trains" leaves** → a *partial, non-zero* score. It does **not** reproduce paper-level metrics (150 steps × all cells). A pure-code surrogate with no GPU scores *lower* — so the $10 floor is the minimum to score meaningfully.
- **GPU:** **$3–10.** LLM: $1–2 (Regime B) / $20–40 (Regime A). **Total: ~$5–50/run.**

### Tier 2 — Scoped faithful ("smallest-two" — recommended iteration target)
Per `CLAUDE.md`: pin to **Qwen3-1.7B + Qwen2.5-3B**, key methods **GRPO + SDAR**, on a single 24–48 GB GPU, full **150 steps**.

| Cell | Steps | GPU-hr (modeled, 1 GPU) |
|---|---|---|
| 1.7B × {GRPO, SDAR} | 150 | 2 × (4–8) = 8–16 |
| 3B × {GRPO, SDAR} | 150 | 2 × (8–16) = 16–32 |
| ALFWorld smoke + Search-QA in-domain eval | — | ~6–12 |

Active **~30–60 GPU-hr** → +20% idle ≈ **36–72 GPU-hr**.
- On **A100-80 ($1.89):** **$68–136.** On **H100 ($4.39):** **$158–316.**
- LLM (Regime A): ~$30–60 (more implement/repair cycles). **Total ~$100–375/run.**

### Tier 3 — Full faithful reproduction (the grant ceiling)
Transparent formula:

```
GPU$ = N_cells × hrs_per_cell × N_gpus × $/hr × contingency
```

| Factor | Value | Basis |
|---|---|---|
| **N_cells** (trained) | **45–54** | 3 scales × 3 envs × 6 trained methods (Vanilla untrained, excluded) |
| **hrs_per_cell** (wall, 8 GPU) | **4–24** | modeled by scale (1.7B 4–8 / 3B 8–14 / 7B 12–24); **paper reports none — widest-bracket factor** |
| **N_gpus** | **8** | paper spec (H800 ≈ H100) |
| **$/hr** | **$4.39** | H100-80 COMMUNITY |
| **contingency** | **×1.5** | partial-GPU waste on failed cells (11 attempts observed on the hard run) |

- GPU-hours: ~45–54 cells × avg ~12.5 wall-hr × 8 = **~4,500–5,400 GPU-hr** (range **3,000–8,000** across the hrs/cell bracket).
- Base GPU $: **3,000–8,000 × $4.39 = $13,000–$35,000.**
- **× 1.5 contingency → GPU $20,000–$52,000.**
- LLM (Regime A, ~45 sub-runs × $25): **~$1,000–1,500** — *rounding error at this tier.*
- **Tier-3 total: ≈ $21,000–$53,000** (point estimate **~$30k**). Sub-agents on OAuth (Regime C) removes ~$1k.

> The hrs/cell factor can swing 2–4× because agentic multi-turn RL is **rollout-bound** (env interaction, not just forward passes). Treat Tier 3 as an order-of-magnitude grant ceiling, not a quote.

---

## 5. Recommended grant line items

| Line item | Config | Budget |
|---|---|---|
| **LLM — orchestration + coding** | GPT-5 root + Sonnet 4.6 sub-agents (metered, cached), all tiers, ~60 runs incl. retries | **$2,500** |
| **GPU — Tier 1+2 iteration** | RTX 4090 / A100-80, ~40 scoped runs during development | **$5,000** |
| **GPU — Tier 3 full faithful** | 8× H100, 45–54 cells × 150 steps, ×1.5 contingency | **$30,000** |
| **Contingency (overall, 20%)** | death-spiral retries (11 observed), OOM escalations (up to ×2 SKU ladder) | **$7,500** |
| **TOTAL (full-faithful program)** | | **≈ $45,000** |
| *Single representative run (Tier 2, fundable)* | smallest-two, A100-80, metered LLM | **~$200–375** |
| *Single dev run (Tier 1, how we're doing it)* | $10 GPU cap + OAuth sub-agents | **~$5–15** |

### Cost-reduction levers (built into the system)
- **OAuth sub-agents** (`claude login`) → Sonnet coding at **$0** (Regime C). Cuts LLM by ~95%.
- **`--model qwen3-coder-featherless`** flat root → ~$0 marginal root.
- **RunPod COMMUNITY** (default) vs SECURE saves ~45%; **`cuda-runtime` image** (default) saves $0.50–1.50 + 5–10 min/run vs `devel`.
- **`--max-usd` / `REPROLAB_MAX_RUN_GPU_USD`** hard caps per run — set these to fence each tier.
- **Smallest-two scope** (`REPROLAB_BASELINE_EXTRA_GUIDANCE`) keeps iteration in Tier 1/2.

---

## Sources
- RunPod / GPU pricing: [runpod.io/pricing](https://www.runpod.io/pricing), [spheron 2026](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/), [getdeploying RTX 4090](https://getdeploying.com/gpus/nvidia-rtx-4090) — cross-checked against in-repo `gpu_catalog.py`.
- OpenAI: [openai.com/api/pricing](https://openai.com/api/pricing/), [devtk.ai 2026 guide](https://devtk.ai/en/blog/openai-api-pricing-guide-2026/).
- Anthropic: [platform.claude.com/docs/pricing](https://platform.claude.com/docs/en/about-claude/pricing), [cloudzero](https://www.cloudzero.com/blog/claude-api-pricing/).
- Kimi: [tokenmix.ai](https://tokenmix.ai/blog/kimi-k2-api-pricing), [pricepertoken K2.6](https://pricepertoken.com/pricing-page/model/moonshotai-kimi-k2.6).
- Paper compute footprint: arXiv 2605.15155v1 §Implementation Details (8× H800, 150 steps).
- In-repo grounding: `cost_ledger.jsonl` (6 metered runs $0.35–$1.94), `prj_09047604` (11 attempts), `generated_rubric.json` (real-weights/ALFWorld/train leaves), `backend/services/runtime/gpu_catalog.py`, `backend/cli.py` budget flags.
