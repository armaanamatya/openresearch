# SDAR Reproduction — Search-QA Scope

## What was reproduced

Self-Distilled Agentic Reinforcement Learning (arXiv 2605.15155) on the
Search-QA closed-book QA proxy (NQ-Open + HotpotQA).  Two model variants:

- **Qwen/Qwen3-1.7B** (base model)
- **Qwen/Qwen2.5-3B-Instruct**

Three baselines:

| Baseline   | Loss                                        |
|------------|---------------------------------------------|
| SDAR       | L_GRPO + 0.1 * L_OPSD  (sigmoid gate)      |
| GRPO+OPSD  | L_GRPO + 0.1 * L_OPSD  (gate=1, ungated)   |
| GRPO       | L_GRPO only                                 |

SDAR invariants:  BETA=10, LAMBDA=0.1, gate=sigmoid(beta*delta_t).detach(),
                  loss = grpo_loss + LAMBDA * opsd_loss.

## What was omitted and why

- ALFWorld / WebShop: Search-QA only per operator scope
- Qwen2.5-7B: out of scope (VRAM budget)
- Retrieval-augmented evaluation: closed-book QA used as surrogate
- SkillBank / skill retrieval: separate SkillRL contribution, out of scope
- Additional OOD eval sets (TriviaQA, PopQA, 2Wiki, MuSiQue, Bamboogle):
  out of scope for the quick run; NQ + HotpotQA used as in-domain data

## How to read metrics.json

- `per_env.searchqa.sdar.mean_final_reward` — mean token-F1 over last 20 steps
  averaged across both model variants for SDAR
- `per_env.searchqa.sdar_minus_grpo` — SDAR – GRPO improvement
- `per_model.<model_key>.*` — per-variant SDAR/GRPO/GRPO+OPSD final token-F1
- `comparison.<model_key>.delta` — SDAR – GRPO per model
- `gate.sdar_mean_g_t` — mean gate value over final 20 steps (should be ∈ (0.4, 0.95))
- `stability.*_cross_seed_std` — std of final rewards across models as a proxy
