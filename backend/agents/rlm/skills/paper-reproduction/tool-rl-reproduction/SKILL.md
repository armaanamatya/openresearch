---
name: tool-rl-reproduction
description: Use when reproducing the Tool-RL paper (arXiv 2604.02869, "Multi-Turn Reinforcement Learning for Tool-Calling Agents with Iterative Reward Calibration", ARAC — NO code released) — MT-GRPO with turn+trajectory advantage, a GTPO hybrid estimator, and the Iterative Reward Calibration (IRC) loop, trained on Tau-Bench airline and evaluated on held-out Tau2-Bench with Qwen3.5-4B / Qwen3-30B-A3B. Names the estimators, the git-pinned frameworks, the held-out pass-rate eval, and the fabrication traps a no-code paper invites.
category: paper-reproduction
tags: [tool-RL, tool-calling, function calling, MT-GRPO, GTPO, IRC, iterative reward calibration, Tau-Bench, tau2-bench, airline, agent, multi-turn, reinforcement learning, RL, GRPO, verl, Qwen, reward, pass rate]
---

# Reproducing Tool-RL (Multi-Turn RL for Tool-Calling Agents + IRC)

**Paper:** arXiv 2604.02869 (ARAC) · **NO code released** — highest fabrication risk in the test set.
**Not UCPO** (that is a separate paper, 2605.00365). First published RL-training results on the
Tau-Bench airline benchmark: stacks **MT-GRPO + a GTPO-hybrid advantage** and introduces **Iterative
Reward Calibration (IRC)**. **Headline:** Qwen3.5-4B 63.8→**66.7%** (+2.9pp, beating GPT-4.1's 49.4%
at ~50× smaller); Qwen3-30B-A3B MoE 58.0→**69.5%** (+11.5pp, approaching Claude Sonnet 4.5's 70.0%).

## When to use this skill
- The paper is arXiv 2604.02869 / mentions MT-GRPO, GTPO, IRC, or RL on Tau-Bench / tau2-bench airline.
- You are writing the advantage estimator, the IRC loop, the env wiring, or the eval.
- Because there is no author code, the temptation to stub is high — read the fabrication traps first.

Run-spec: `configs/tool_rl_2604_run_spec.json` (**scratch mode**, `USE_AUTHOR_REPO=0`; GPU cap
`$300`; `opus-foundry` root, `sonnet-foundry` executor). The `baseline_extra_guidance` there is the
load-bearing steering — read it.

## The method — implement exactly this on top of real frameworks
- **MT-GRPO (turn + trajectory):** `A_{i,k} = Σ_{l=k}^{K−1} A^I_{i,l} + A^O_i`, where
  `A^I_{i,l} = (r_{i,l} − μ_{r_l}) / (σ_{r_l} + ε)` is group-normalized **per turn index l** across the
  G rollouts, and `A^O_i = (o_i − μ_o) / (σ_o + ε)` is the group-normalized trajectory outcome.
- **GTPO hybrid (discounted + dampened):** `raw_{i,k} = Σ γ^{l−k} r_{i,l} + γ^{K−k} o_i`, then
  `A^hybrid = GN_k(raw) + λ·A^O_i`; defaults **γ=0.9, λ=0.3**. Purpose: eliminate the advantage-sign
  **misalignment** naïve dense MT-GRPO suffers.
- **IRC (Algorithm 1):** collect a rollout buffer → classify every turn into tiers
  `{gold, soft, read, state, error, duplicate, message}` → compute each tier's **point-biserial
  correlation ρ_c** with binary episode success → set `r_c = α·ρ_c if |ρ_c| > δ else 0` → recalibrate
  weights `new_w = clip(prior + η·ρ_c, −1.5, 1.5)` → retrain; loop until zero advantage mismatches and
  `Corr(mean r_i, o_i) > η`. Compute Table 3/4 diagnostics from **your own rollouts** — do not hardcode
  the paper's numbers.
- **Golden-action match:** order/type-tolerant deep-equal (sort dict-lists, coerce numeric strings,
  drop empties); soft-match scores `0.5 + 0.5·|args∩args*|/|args*|`.

**Build on the real frameworks — do NOT hand-roll:** the RL trainer is **verl** (its GRPO + multi-turn
tool-calling rollout machinery); the env is **tau-bench / tau2-bench airline**. Implement only the
paper's method (MT-GRPO/GTPO/IRC) on top. Real weights via `AutoModelForCausalLM.from_pretrained`.

## Environment & libraries (git-pinned; author says PyPI-verified)
```
torch==2.4.0   transformers==4.46.3   accelerate==1.1.1   numpy==1.26.4
litellm==1.51.0   tenacity==8.5.0
verl==0.8.0                      # best-effort: the estimator registry hookup is optional;
                                 # degrade to pure-numpy if verl's registry API is unavailable
tau-bench   @ 59a200c6…          # v1, pip install git+…  (train env)
tau2-bench  @ 1901a301…          # v2, pip install git+…  (eval env)
```
- **tau2 needs python ≥3.12,<3.14** → base image `python:3.12-slim`.
- **tau2's hatchling build does NOT package its `data/` dir.** Vendor the airline assets
  (`db.json, tasks.json, policy.md, split_tasks.json` — JSON/markdown only) into `code/tau2_data/` and
  set `TAU2_DATA_DIR` **before** importing `tau2.*`.

## Datasets & tools
- **Train:** Tau-Bench (v1) airline — real `MockAirlineDomainEnv`, 50 tasks, real tool classes with
  DB-mutation semantics, DB-state-hash reward.
- **Eval:** Tau2-Bench (v2) airline — the **test split = 20 tasks held out** from the v1 pool. Train
  (Tau1) and test (Tau2) task sets are **non-overlapping** (a graded fidelity leaf).
- **Tool taxonomy:** read-only = `get_user_details, get_reservation_details, search_direct_flight,
  search_onestop_flight, list_all_airports, calculate, think`; state-changing = `book_reservation,
  cancel_reservation, update_reservation_*, send_certificate, transfer_to_human_agents`.

## Training procedure
- Paper compute: verl + **Megatron-Core** on 8× H20 (96 GB); batch 8, **N=4 rollouts/prompt**, max
  10K prompt / 45K response tokens, **max 40 turns**, temp 0.9, MT-GRPO, Adam, `low_var_kl`.
- **8-version ablation** (Qwen3.5-4B): V1 GRPO-sparse → V3 MT-GRPO-sparse (lr 2e-6, KL 0.05, 60 steps)
  → V5 dense (read 0.3, state 0.1, 116 steps) → V6 IRC (read 0.0, state −0.1, lr 5e-7, KL 0.2, 180
  steps) → V8 IRC-final. (A parallel grid for the 30B MoE, up to 480 steps.)
- **Start with the smallest meaningful slice:** Qwen3.5-4B + MT-GRPO + **sparse** reward, ~60 steps.
  Confirm a **genuine nonzero non-stubbed reward** and a held-out pass-rate trend **before** adding IRC
  or scaling to the 30B MoE. The run-spec's harness slice is
  `["smoke", "qwen3_5_4b_V3_mtgrpo_sparse", "qwen3_5_4b_V6_irc"]`.
- Model ids: `Qwen/Qwen3.5-4B-Instruct` (fallbacks `Qwen/Qwen3-4B-Instruct-2507`,
  `Qwen/Qwen2.5-3B-Instruct`); 30B `Qwen/Qwen3-30B-A3B-Instruct-2507`.

## Evaluation & the fabrication traps
- **Held-out metric `pass_rate`** = mean per-episode reward, where
  `reward = float(db_pass and communicate_pass)`: `db_pass` compares the episode's final DB hash to
  the hash from **replaying the task's reference action trajectory on a fresh env** (tau2's own
  `EnvironmentEvaluator`); `communicate_pass` = all required `communicate_info` substrings appear. Also
  report **Pass^4** (all 4 trials pass) and avg reward. Sidecar via
  `record_eval(metric_name="pass_rate", held_out=True)`.
- **TRAP — hardcoded-literal scanner:** write `reward = float(db_pass and communicate_pass)`, **not**
  `1.0 if … else 0.0` — the AST fabrication scanner flags a bare `1.0`/`0.0` literal as a hardcoded
  metric even when it is data-driven.
- **Targets:** 4B base 63.8 / mt_grpo 64.6 / irc 66.7; 30B base 58.0 / mt_grpo 68.0 / irc 69.5. **V5
  dense 4B = 57.3 (−6.5pp vs base)** — the paper's dense-degradation is itself a graded "must
  reproduce" leaf; don't smooth it away.

## Failure modes specific to this paper
- **No author code = fabricate-by-accident risk.** The paper *and* both methods it stacks (MT-GRPO
  2505.11821, GTPO 2511.14846) have no public code. Mitigate by building on verl + tau-bench/tau2 and
  grading on measured on-disk evidence only — a stubbed reward / random-init model / hardcoded metric
  fails the fidelity checks.
- **Uncapped LLM-user-sim cost.** tau-bench's user simulator calls an LLM **every turn** — a cost axis
  *outside* the GPU cap. Wire the sim to the **already-configured Anthropic/Azure-Foundry** provider
  (NOT external DeepSeek/OpenAI keys), keep turn counts modest, and **degrade loudly to a deterministic
  offline user-sim** if creds aren't forwarded into the GPU sandbox.
- **RL-config guards false-fail legit values.** This paper's ablations use lr down to 3e-7 and a
  **negative** `state_w`; `loss>0 / variance>0 / lr∈[1e-7,1]` are supervised assumptions. `alpha=0.0`
  and negative coefficients are legitimate — don't let a range heuristic block them.
- **GKE code-staging (the P0 that actually killed the live run) + driver durability + cost-ledger
  blindness** → all covered in the `gcp-gke-reproduction` skill; enable `OPENRESEARCH_GKE_SYNTH_CELL`.

## Reproduction checklist
1. Build on verl (GRPO + tool-calling rollouts) + tau-bench/tau2 airline; implement only MT-GRPO/GTPO/IRC.
2. `python:3.12-slim` base; vendor `tau2_data/` + set `TAU2_DATA_DIR` before importing tau2.
3. Real Qwen3.5-4B weights via `from_pretrained`; smallest slice first (MT-GRPO sparse, ~60 steps).
4. User-sim on Foundry, not DeepSeek/OpenAI; degrade to offline sim if creds missing.
5. Held-out `pass_rate` on the 20 non-overlapping Tau2 tasks; `reward = float(db_pass and communicate_pass)`.
6. Reproduce the V5 dense degradation, not just the IRC gain.

## Sources (repo-native grounding — not vendored)
`configs/tool_rl_2604_run_spec.json`, `docs/runbooks/2026-07-07-tool-rl-gke-reproduction-handoff.md`,
`runs/prj_c912f5df415f410c/{parsed_full_text.txt,generated_rubric.json,code/}` (the faithful impl:
`advantage_estimators.py`, `irc.py`, `tau2_airline_eval.py`). For the GKE/GPU path see the
`gcp-gke-reproduction` skill.
