# SDAR Reproduction
## Paper: arXiv 2605.15155 — Self-Distilled Agentic Reinforcement Learning

## What was reproduced
- **SDAR algorithm**: sigmoid gate g_t = sigmoid(β·Δ_t).detach() with β=10,
  distillation coefficient λ=0.1, teacher = same model + retrieved skills,
  student = same model without skills. SDAR inference uses EMPTY skill context.
- **5 baselines**: GRPO (clipped surrogate), OPSD (standalone self-distillation),
  Skill-SD (skill-conditioned), GRPO+OPSD (naive ungated sum — reproduces
  instability), RLSD (RL + self-distillation).
- **3 gating strategies**: Gap (default), Entropy, Soft-OR.
- **4 retrieval strategies**: KM (keyword matching, default), UCB (Eq 1),
  Full, Random.
- **3 environments**: Search-QA (NQ+HotpotQA train; TriviaQA/PopQA/2Wiki/
  MuSiQue/Bamboogle OOD eval), ALFWorld (6 task categories), WebShop.
- **3 model scales**: Qwen3-1.7B, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct.
- **Ablations**: β sweep [0,1,5,10,20], λ sweep [0,0.01,0.05,0.1,0.5,1.0],
  gate-type comparison, retrieval strategy comparison.

## What was omitted and why
- **Scale**: Paper used 8×H800 (80 GB each); this run used 1×RTX A5000 (25.4 GB).
  Batch sizes reduced from paper: Search-QA 128→16, ALFWorld 16→4, WebShop 16→8.
  Real model weights and real datasets were preserved throughout.
- **SkillBank**: Attempted ZJU-REAL/SkillBank from HuggingFace Hub. If unavailable,
  used a built-in representative bank constructed from paper domain knowledge
  (see sdar/skills.py for provenance details).
- **WebShop full simulator**: If the full WebShop simulator could not be installed,
  used the lightweight items_human_ins.json catalog for task construction.

## How to read metrics.json
- `sdar.alfworld.success_rate`: SDAR success rate on ALFWorld (eval without skills)
- `sdar.webshop.score`: SDAR score on WebShop (eval without skills)
- `grpo.alfworld.success_rate`: GRPO baseline success rate on ALFWorld
- `grpo.webshop.score`: GRPO baseline score on WebShop
- `comparisons.sdar_minus_grpo_alfworld_success`: SDAR - GRPO on ALFWorld (≥0 = SDAR wins)
- `per_model[model][env][algo].metric`: primary metric for each (model, env, algo) cell
- `baselines_vs_sdar[model][env]`: all 5 baselines vs SDAR comparison table
- `retrieval_comparison[model][env]`: 4 retrieval strategies vs GRPO baseline
- `gate_dynamics[model][env]`: training-time gate statistics (active ratio, mean, etc.)
- `ablations`: β sweep, λ sweep, gate-type comparison results
