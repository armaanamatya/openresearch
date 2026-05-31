# SDAR Full Reproduction — Resource Map (2026-05-31)

Map the 27-leaf rubric to the resources each component needs **before** launching, so we
size compute/time/cost deliberately instead of launch-and-steer. Estimates grounded in
observed runs (~5 s/optimizer-step Search-QA, ~7–9 GB/model bf16, ~30–42 min per matrix
`run_experiment`, agentic ALFWorld rollouts slower).

## Components × resources

| Component | Rubric leaves | Resource | Feasibility |
|---|---|---|---|
| **SDAR core** (gate, OPSD single-sample KL, λ/β, loss) | Method ×4 (~0.55 of area) | code; Search-QA training | ✅ proven |
| **5 baselines** (GRPO, OPSD, Skill-SD, GRPO+OPSD, RLSD) | Method 0.15 + Result + Artifact | code (env-agnostic algorithms) | ✅ code |
| **Gating** (Gap/Entropy/Soft-OR) | Method 0.10 | code | ✅ code |
| **Skill retrieval** (UCB/KM/Full/Random) + SkillBank | Experiment 0.30 + Data 0.15 | code + small bank | ✅ code |
| **Search-QA** (NQ+HotpotQA in-domain; TriviaQA/PopQA/2Wiki/MuSiQue/Bamboogle OOD) | Data 0.50 | **all 7 datasets cached**; token-F1 reward | ✅ ready |
| **ALFWorld** (GiGPO split, 6 categories) | Data 0.25 + Experiment 0.20 + Result | **alfworld+textworld installed**; agentic multi-turn; success-rate reward; GiGPO data may need fetch | ⚠️ feasible, slow |
| **WebShop** (1000 train / 128 val) | Data 0.20 + Experiment + Result | **not installed** (web simulator + Java) | ❌ high-risk → best-effort/gap |
| **3 models** (1.7B, 3B, 7B) | Experiment 0.25 | 1.7B+3B cached; **7B = ~15 GB download** (internet ✅), ~14 GB bf16 fits a 24 GB card | ✅ download |
| **Config file + repo structure + README** | Artifact 0.75 | scaffolding | ✅ trivial |
| **Figures 10–14 + Table 1** | Eval 0.44 + Artifact 0.25 | plotting from gate dynamics | ✅ ready |
| **Result-match absolute numbers** (84.4% ALFWorld, +7% Search-QA, …) | Result ×5 (whole area) | **8×H800 scale** to hit magnitudes | ❌ trend-level only on 1 box |

## Compute / time / cost estimate (4 GPUs)

| Phase | GPU-hours | Notes |
|---|---|---|
| Search-QA core (3 models × 5 baselines × 150 steps) | ~3 | fast single-turn; ~45 min wall on 4 GPUs |
| Sweeps (β/λ/gating/retrieval, short) | ~0.5 | a few steps each |
| ALFWorld (3 models × ~2 baselines, agentic) | ~3–6 | slow rollouts; dominant cost |
| WebShop (if it installs) | ~3–5 | else gap |
| 7B (download + ~2× 3B compute) | ~2–3 | one-time ~10-min download |
| **Total** | **~9–15 GPU-h** | **~3–5 h wall on 4 GPUs**, + RLM iteration overhead (implement_baseline ~5–13 min × ~30–50 iters) |

- **LLM cost: $0** (claude-oauth subscription). **Tokens/run:** ~0.7–1 M cache_read + ~100 k output for the root — *to be logged* (see below) for calibration.
- **Iteration budget:** needs **~40–50 RLM iterations** (the default 20 cap cut the last run off mid-improvement) + **~5–6 h wall-clock** + the robust fail-safe-persistence design so partial coverage is never lost.

## Verdict / recommended scope
- **Definitely reproducible (~4–6 GPU-h):** SDAR core + all 5 baselines + gating + skill-retrieval + Search-QA (all 7 datasets) + 3 models + config/repo/figures. Targets Method, Data (Search-QA), Eval, Artifact areas.
- **Feasible but slow:** ALFWorld (installed) — adds Data/Experiment/Result coverage.
- **Best-effort / documented gap:** WebShop (install risk).
- **Residual cap:** the 5 absolute-number Result-match leaves need 8×H800 — trend-level credit only.
- **Realistic ceiling on one box: ~0.75–0.85.**

## Generality (not Search-QA-specific)
The SDAR algorithm (gate, OPSD, λ/β, baselines, retrieval, gating) is **environment-agnostic**;
only the **reward** differs per env: Search-QA = max-alias token-F1 on the Answer span; ALFWorld
= task success rate; WebShop = task score. The guidance must define reward as a per-env adapter,
not a Search-QA token-F1 baked into the core loop.

## Cost/token estimator + logging (the tuning loop)
Existing infra: `backend/services/pricing/{estimator,token_accumulator,calibration,catalog}.py`,
`cost_ledger.jsonl` → `tokens_total.json`, and `recompute_calibration()` (runs → per-primitive
priors → estimator). Gaps + plan:
1. **Root cache tokens not logged** (rlm `ModelUsageSummary` has no cache fields). → client-side
   capture landed (`5d0895f`); next: a decoupled `drain_root_usage()` sink + a `run.py` hook that
   ledgers each root turn (cache-only, to avoid double-counting rlm's input/output).
2. **`estimated_usd` is null per row** → populate from the catalog at ledger-write time.
3. **Calibration not auto-run** → call `recompute_calibration(runs_root)` at run end so each run's
   (now-complete) tokens tune the priors → the upfront estimator improves over time.
