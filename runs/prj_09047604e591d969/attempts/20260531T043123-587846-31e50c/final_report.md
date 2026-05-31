# PARTIAL REPRODUCTION

**Paper:** Self-Distilled Agentic Reinforcement Learning (SDAR)

## Rubric Score

**Overall score:** 0.476  (✘ below target)

_20/22 rubric leaves graded · PaperBench bundle rubric_

| Area | Score | Notes |
|---|---|---|
| Method and code fidelity to the paper | 0.645 |  |
| Data and preprocessing fidelity | 0.070 |  |
| Experiment execution and reproducibility | 0.180 |  |
| Evaluation protocol and metric correctness | 0.900 |  |
| Result match versus the paper's reported targets | 0.020 |  |
| Artifact completeness and provenance | 0.235 |  |

### Weakest rubric leaves

| Score | Justification |
|---|---|
| 0.00 | Reproduction is explicitly Search-QA only with ALFWorld marked out of scope, so no ALFWorld success-rate comparison is provided. |
| 0.00 | No evidence of UCB, KM, Full, or Random skill retrieval implementations; SDAR runs with empty skill_context. |
| 0.00 | ALFWorld is explicitly stated as out of scope per operator. |
| 0.00 | WebShop is out of scope; no evidence of WebShop training setup. |
| 0.00 | No SkillBank from SkillRL referenced or loaded. |

## Reproduction Summary

Reproduced SDAR (Self-Distilled Agentic Reinforcement Learning) on Search-QA with Qwen3-1.7B and Qwen2.5-3B-Instruct. Implemented GRPO + sigmoid-gated OPSD with beta=10, lambda=0.1, stop-gradient on gate. Ran 50 steps per model. Baseline rubric score: 0.266. After adding GRPO/OPSD baseline entry points, improved score: 0.438 (target: 0.600). ALFWorld and WebShop were out of scope (Search-QA only); Qwen2.5-7B was skipped for budget. Core SDAR invariants (sigmoid gate, stop-gradient, lambda=0.1, beta=10) all passed.

## Scope

**Requested:** Qwen3-1.7B + Qwen2.5-3B on Search-QA (smallest-two scope)

**Ran:**
- qwen3_1_7b on Search-QA
- qwen2_5_3b on Search-QA

**Gaps:** _(items requested but not reproduced; datasets marked "unobtainable" were excluded from the rubric score, not penalised)_
- qwen2_5_7b: out-of-scope per operator (budget)
- alfworld: out-of-scope per operator (Search-QA only)
- webshop: out-of-scope per operator (Search-QA only)
- grpo_baseline_run: GRPO baseline implemented but not separately executed (time budget)

## Baseline Metrics vs. Paper Claims

| Metric | Reproduced | Paper Claim |
|---|---|---|
| L_total = L_GRPO + lambda*L_OPSD | — | lambda=0.1, beta=10 |
| SDAR > GRPO on Search-QA | — | reward_ckpt2=0.159 |
| SDAR sigmoid gate | — | g_t = sigmoid(beta * delta_t).detach() |
| accuracy | 0.1171875 | — |
| config | {'beta': 10.0, 'lambda_opsd': 0.1, 'eps_clip': 0.2, 'lr': 1e-05, 'steps': 50, 'batch_q': 4, 'G': 4, 'max_new_tokens': 96, 'seed': 42, 'device': 'cuda:0', 'gating': 'gap', 'methods_run': ['sdar', 'grpo']} | — |
| gating_ablation | {} | — |
| loss | -0.00343049954099115 | — |
| opsd_collapse | {} | — |
| per_model | {'qwen3_1_7b': {'loss': -0.0025200577219948173, 'reward_checkpoint1': 0.056578573770821095, 'reward_checkpoint2': 0.15107211340218782, 'accuracy': 0.109375, 'eval_f1': 0.14296874999999998, 'zero_shot_f1': 0.03697916666666667}, 'qwen2_5_3b': {'loss': -0.004340941359987483, 'reward_checkpoint1': 0.2318340688943863, 'reward_checkpoint2': 0.16711676605045794, 'accuracy': 0.125, 'eval_f1': 0.14024395743145746, 'zero_shot_f1': 0.12706956785904153}} | — |
| reward_checkpoint1 | 0.1442063213326037 | — |
| reward_checkpoint2 | 0.15909443972632287 | — |
| scope | {'models_run': ['qwen3_1_7b', 'qwen2_5_3b'], 'models_skipped': ['qwen2_5_7b'], 'environments_skipped': ['alfworld', 'webshop'], 'baselines_executed': ['grpo_qwen3_1_7b', 'grpo_qwen2_5_3b'], 'gaps': [{'item': 'qwen2_5_7b', 'reason': 'out-of-scope per operator (budget)'}, {'item': 'alfworld', 'reason': 'out-of-scope per operator (Search-QA only)'}, {'item': 'webshop', 'reason': 'out-of-scope per operator (Search-QA only)'}]} | — |
| sdar_vs_grpo | {'qwen3_1_7b': {'sdar_eval_f1': 0.14296874999999998, 'sdar_accuracy': 0.109375, 'sdar_reward_ckpt2': 0.15107211340218782, 'grpo_eval_f1': 0.04295012537127475, 'grpo_accuracy': 0.0, 'grpo_reward_ckpt2': 0.07427516914904117, 'sdar_minus_grpo_f1': 0.10001862462872523}, 'qwen2_5_3b': {'sdar_eval_f1': 0.14024395743145746, 'sdar_accuracy': 0.125, 'sdar_reward_ckpt2': 0.16711676605045794, 'grpo_eval_f1': 0.19047506313131313, 'grpo_accuracy': 0.140625, 'grpo_reward_ckpt2': 0.20283234342932702, 'sdar_minus_grpo_f1': -0.05023110569985567}} | — |
| status | completed | — |
| wall_time_seconds | 1211.1756291389465 | — |

## Improvement Candidates

**1. Candidate 1** — promoted (0.17118644067796607)
**2. Candidate 2** — declined
**3. Candidate 3** — declined

## Cost

| Category | USD |
|---|---|
| Primitive-internal LLM | $0.000000 |
| **Total LLM** | **$0.000000** |

**Iterations:** 17

## Token Usage

| Metric | Value |
|---|---|
| Total LLM calls | 20 |
| Input tokens | 39 |
| Output tokens | 46,338 |
| Cache creation (input) | 85,762 |
| Cache read (input, prompt-cache-billed) | 1,836,674 |

### Per-primitive token usage

| Primitive | Calls | Input | Output |
|---|---|---|---|
| baseline-implementation | 1 | 27 | 43,198 |
| plan_reproduction | 1 | 6 | 1,891 |
| propose_improvements | 1 | 6 | 1,249 |

---
_Generated by ReproLab RLM orchestrator (Issue #60)._
