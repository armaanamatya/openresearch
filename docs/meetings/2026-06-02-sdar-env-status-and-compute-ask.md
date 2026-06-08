# SDAR Reproduction — Environment Status & Compute Request

**Date:** 2026-06-02
**Paper:** SDAR — Self-Distilled Agentic Reinforcement Learning (arXiv 2605.15155)
**Box:** `sun.cs.txstate.edu` — 8× NVIDIA RTX A5000 (24 GB each), 128 CPU cores, 503 GB RAM (on-prem)
**Run under analysis:** `prj_09047604e591d969` (live, started 2026-06-02 14:10), plus completed run 2026-06-01

---

## TL;DR

| Environment | Status | Root cause |
|---|---|---|
| **Search-QA** | ✅ **Working** — reward climbs 0.086 → 0.444, SDAR gate active | **Dense reward** (token-F1 partial credit) gives a usable learning signal |
| **ALFWorld** | ❌ **Not learning** — reward stuck at 0.0000 for every step | **Sparse 0/1 reward + cold-start 7B** → no successes → no gradient. *Plus* env reloaded every episode (slow). **This is a methodology/software problem, not a hardware problem.** |
| **WebShop** | ⚪ **Deliberately excluded** (not a crash) | Heavy server to provision; de-scoped on purpose and declared a rubric gap so it doesn't count against the score |

**The single binding hardware constraint is the 24 GB VRAM ceiling on the A5000s.** It forces LoRA-only training, batch size 1, tiny rollout groups, and 64-token actions — and it forces every 7B cell to be split across 2 GPUs, halving throughput.

**Bottom line for the ask:** the dollar cost is small (~$0 cash today on the Claude subscription; ~$200–400 in rented GPU time for a *full* paper sweep). The real value of bigger GPUs is **faster iteration and the ability to run the 7B + multi-turn envs at paper scale.** But do **not** buy hardware to "fix ALFWorld" — that's a code fix.

---

## 1. What the runs actually show

The training matrix is run one cell per `(model × baseline × environment × seed)`, each cell a separate process. From the live run right now:

```
search_qa SDAR : step 90,  reward 0.086 → 0.444, gate≈0.89, gap≈2.14, opsd_loss decaying   ✅ real SDAR
search_qa GRPO : step 50,  reward → ~0.30,        gate/gap/opsd = 0 (correct GRPO baseline)  ✅
alfworld  SDAR : step 20,  reward 0.0000 every step, grpo_loss 0.0000                        ❌
alfworld  GRPO : step 20,  reward 0.0000 every step, grpo_loss 0.0000                        ❌
```

### Why Search-QA learns and ALFWorld doesn't — same RL code, different reward shape

- **Search-QA reward = token-F1** — a continuous 0–1 *partial-credit* score. Almost every rollout earns *some* reward, so the GRPO advantage `(rᵢ − mean) / std` has non-zero variance → gradient flows → reward climbs. **This is the textbook "dense reward trains" case.**
- **ALFWorld reward = `float(won)`** — sparse binary success. The cold-start Qwen2.5-7B-Instruct (no SFT warm-start, 8 turns, 64-token actions, only 4 rollouts per step) **never solves a single task** in the first 80 episodes. All rewards = 0 → advantage = 0 → `grpo_loss = 0.0000` exactly → **no learning signal at all.**

Important nuance: **the SDAR algorithm itself is working in ALFWorld.** The self-distillation machinery (`gate`, teacher-student `gap`, `opsd_loss`) is live and evolving — gate climbs 0.0 → 0.32, gap grows. The problem is purely that the *task* reward never fires, so GRPO has nothing to optimize. SDAR's auxiliary loss can't substitute for a real reward.

### Second ALFWorld problem: the environment is rebuilt every episode

The ALFWorld log shows `Initializing AlfredTWEnv...` **82 times** (once per rollout episode), each reload re-scanning the full TextWorld game corpus. Result:

| | Search-QA | ALFWorld |
|---|---|---|
| Seconds per training step | **~30 s** | **~145–400 s** (5–13× slower) |
| Est. time per cell | ~1.3 hr | **~3.5–6.5 hr** |

So ALFWorld cells dominate wall-clock *and* produce nothing. Fixing the env to construct once and reset in place would cut a large chunk of that per-step time.

### WebShop — excluded on purpose, not broken

The scope spec for this run explicitly states WebShop is **out of scope** and must be "declared a scope gap so its rubric leaves are excluded (numerator AND denominator)." The env code exists and is real, but the WebShop server (a multi-GB product catalog + search-index Flask app) is heavy to stand up, so the team de-scoped it to spend compute on the two envs we can run honestly. **This is a deliberate, principled exclusion — the rubric is not penalized for it.** It is *not* a failure of the same kind as ALFWorld.

### Progress note (worth saying out loud in the meeting)

A week ago ALFWorld and WebShop were both `data_unavailable` — the simulators weren't even installed (completed run 2026-06-01 ran only Search-QA on the small models). This week we got the **real ALFWorld simulator running** (24 train tasks / 8 eval tasks loading), which is what *uncovered* the deeper sparse-reward issue. We moved the blocker from "infrastructure" to "RL methodology."

---

## 2. The binding hardware constraint: 24 GB VRAM

The A5000's 24 GB is the ceiling that shapes every training choice we're forced to make:

| Forced compromise | Why | Paper-faithful alternative (needs more VRAM) |
|---|---|---|
| **LoRA adapters only** (0.53% of params trainable) | Full 7B fine-tune won't fit | Full-parameter fine-tuning |
| **Batch size = 1, group size G = 4–8** | Each extra rollout multiplies the KV-cache / activation memory | Bigger G → **directly helps sparse-reward envs** (more chances at a success) |
| **max_new_tokens = 64** | Long generations blow the activation budget | Longer agent actions / reasoning traces |
| **Gradient checkpointing on** | Trades compute for memory | Off (faster) |
| **7B cell sharded across 2 GPUs** | A 7B RL cell + rollout buffers > 24 GB on one card | One 7B cell per card, with headroom |

That last row is the throughput killer. Confirmed live: each 7B cell holds ~9 GB on one A5000 **and** ~9 GB on a second (device_map="auto" tensor-sharding). So **8 GPUs run only 4 cells at a time.** The small models (1.7B/3B) fit one card and would run 8-wide — but the 7B reproduction halves our effective fleet.

**On an 80 GB card (A100/H100), a 7B RL cell fits on one GPU with room to raise G** — which removes the sharding tax (2× throughput) *and* gives the bigger rollout groups that sparse-reward envs like ALFWorld actually need.

---

## 3. Software fix vs hardware fix (the honest split)

To avoid spending hardware money on a code problem:

**ALFWorld zero-reward — fix in software, ~no new hardware needed:**
1. **SFT warm-start** the policy on a few hundred ALFWorld expert trajectories before RL, so it can occasionally succeed and produce a gradient. (Standard practice; the paper does warm-start.)
2. **Reward shaping / partial credit** for sub-goal progress, instead of pure 0/1 terminal success.
3. **Construct the env once per cell** and reset in place (kills the 82× reload, ~2–4× faster steps).
4. **Larger group size G** so at least one of N rollouts succeeds early — *this one wants more VRAM.*

**What genuinely wants bigger hardware:**
- Running the **full 3-model × 3-env × 5-baseline × multi-seed matrix** at a reasonable wall-clock.
- **Bigger G and longer generations** (helps ALFWorld/WebShop learning *and* fidelity to the paper).
- Optional **full fine-tuning** for a true apples-to-apples reproduction of the paper's headline numbers.

---

## 4. Compute request (tiered)

| Tier | Ask | Unblocks | Rough cost |
|---|---|---|---|
| **0 — keep current box** | 8× A5000 (have it) | Search-QA + small-model dev/iteration | $0 (on-prem) |
| **1 — burst rental (recommended)** | **4–8× A100 80 GB** or **H100 80 GB**, cloud, on demand | 7B on one card; 2× throughput; bigger G → ALFWorld/WebShop become learnable; full matrix in **hours not days** | **~$200–400 per full sweep** (see §5) |
| **2 — owned upgrade** | A dedicated 4–8× 80 GB node | Same as Tier 1, no per-hour metering, full-param fine-tune headroom | Capital — only if this becomes a sustained program |

**Recommendation:** Tier 1. The highest-leverage single change is **VRAM per card: 24 GB → 80 GB.** It removes the sharding tax, enables the bigger rollout budgets that sparse-reward envs need, and turns a 3-day sweep into a same-day sweep. Cloud burst keeps it cheap and avoids a capital request until the program justifies it.

---

## 5. Cost estimation

### LLM cost (the agent that writes the reproduction code)
- Measured: **~$3.2–3.6 of API-equivalent per run**, entirely the code-writing sub-agent.
- We run on the **Claude subscription (OAuth)**, so **actual cash cost ≈ $0** — the dollar figure is the equivalent if billed per-token. **Negligible either way.**

### GPU cost — current box
- On-prem A5000s: electricity only (~2.5 kW for the node). A multi-day sweep is **~$20 of power.** The real cost is **opportunity** — the box is tied up for ~3 days per full seed sweep.

### GPU cost — full paper sweep on rented cloud
Assumptions: full matrix ≈ **90–135 cells** (3 models × 3 envs × 5 baselines × 2–3 seeds), avg ~3 hr/cell.

| Hardware | $/GPU-hr (community/spot) | Concurrency on 8 GPUs | Wall-clock, full sweep | $ per full sweep |
|---|---|---|---|---|
| A5000 24 GB (current) | ~$0.35 (rental-equiv) | 4 cells (7B sharded) | **~2–3 days** | ~$190 |
| **A100 80 GB** | ~$1.5–2.0 | 8 cells, faster | **~12–18 hr** | **~$250–350** |
| **H100 80 GB** | ~$2.5–3.5 | 8 cells, fastest | **~6–10 hr** | **~$200–350** |

**Key insight:** the dollar cost barely changes across tiers (~$200–350 either way) because faster cards finish proportionally sooner — **but the wall-clock collapses from days to hours.** For an iterative research program where we'll run the sweep many times while debugging, **that wall-clock is the entire value proposition.** Budget **~$1–2k of cloud GPU credit** to cover the full reproduction plus the 5–10 debug iterations it will realistically take.

---

## 6. One-paragraph version for the boss

We can faithfully reproduce one of the paper's three environments today (Search-QA — it has a dense reward, so RL learns and our SDAR numbers climb correctly). The second (ALFWorld) now runs on real data but doesn't learn yet, because its reward is sparse 0/1 and our 7B model — constrained by 24 GB GPUs to tiny rollout budgets and no warm-start — never succeeds, so there's no gradient; the fix is mostly software (warm-start + reward shaping + caching the env), not hardware. The third (WebShop) we deliberately excluded for now. The hard hardware limit is the 24 GB VRAM on our A5000s: it forces us to split each 7B job across two GPUs (halving our throughput) and caps the rollout budget that sparse-reward tasks need. Renting **4–8× 80 GB A100/H100 GPUs in short cloud bursts (~$200–400 per full sweep, ~$1–2k including debug iterations)** would let us run the complete model × environment × baseline matrix in hours instead of days and give the bigger rollout budgets that make ALFWorld and WebShop learnable. The LLM/agent cost is effectively zero (we run on the Claude subscription).
