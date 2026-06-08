# PARTIAL REPRODUCTION

**Paper:** SDAR: Self-Distilled Agentic Reinforcement Learning (`2605.15155`)

## Rubric Score

**Overall score:** 0.363  (✘ below target)

_25/26 rubric leaves graded · PaperBench bundle rubric_

| Area | Score | Notes |
|---|---|---|
| Method and code fidelity to the paper | 0.627 |  |
| Data and preprocessing fidelity | 0.090 |  |
| Experiment execution and reproducibility | 0.370 |  |
| Evaluation protocol and metric correctness | 0.185 |  |
| Result match versus the paper's reported targets | 0.015 |  |
| Artifact completeness and provenance | 0.330 |  |

### Weakest rubric leaves

| Score | Justification |
|---|---|
| 0.00 | ALFWorld is explicitly out of scope and no ALFWorld results are produced. |
| 0.00 | WebShop is not implemented or evaluated in this reproduction. |
| 0.00 | ALFWorld is out of scope; no Qwen3-1.7B ALFWorld results are produced. |
| 0.00 | ALFWorld is explicitly declared out of scope ('ALFWorld, WebShop... declared as scope gaps'); no evidence of GiGPO training data or the six task categories. |
| 0.00 | WebShop is explicitly declared out of scope; no training or validation task configuration is shown. |

## Reproduction Summary

Partial reproduction of SDAR (arXiv 2605.15155). Algorithm: L_total=L_GRPO+0.1*L_OPSD, g_t=sigmoid(10*Delta_t), Delta_t=log pi_teacher - log pi_student (stop-gradient on teacher+gate). Qwen3-1.7B/Search-QA: SDAR=0.0543, GRPO=0.0567, GRPO+OPSD=0.0521. Qwen2.5-3B/Search-QA: SDAR=N/A, GRPO=0.1450. Rubric: 0.363/0.600. Scope limited to Search-QA on 1.7B/3B models. ALFWorld/WebShop unavailable (simulator/server required). 7B models capacity-skipped (>24GB). Skill-SD/RLSD/OPSD baselines not implemented.

## Scope

**Requested:** SDAR on Qwen2.5/Qwen3 across ALFWorld, WebShop, Search-QA

**Ran:**
- qwen3_1_7b/search_qa/sdar
- qwen3_1_7b/search_qa/grpo
- qwen3_1_7b/search_qa/grpo_opsd
- qwen2_5_3b/search_qa/grpo

**Gaps:** _(items requested but not reproduced; datasets marked "unobtainable" were excluded from the rubric score, not penalised)_
- ALFWorld: data_unavailable — alfworld simulator not installed
- WebShop: data_unavailable — WebShop server not available
- qwen2_5_7b: capacity_skipped — exceeds 24GB GPU VRAM
- Skill-SD baseline: not implemented
- RLSD baseline: not implemented
- OPSD standalone baseline: not implemented

## Baseline Metrics vs. Paper Claims

| Metric | Reproduced | Paper Claim |
|---|---|---|
| claims | — | [{'method': 'SDAR', 'dataset': 'ALFWorld', 'metric': 'task_success_rate', 'expected_result': '+9.4% over GRPO'}, {'method': 'SDAR', 'dataset': 'WebShop', 'metric': 'accuracy', 'expected_result': '+10.2% over GRPO'}, {'method': 'SDAR', 'dataset': 'Search-QA', 'metric': 'F1_or_accuracy', 'expected_result': '+7.0% over GRPO'}, {'method': 'GRPO', 'dataset': 'ALFWorld', 'metric': 'task_success_rate', 'expected_result': 'baseline'}, {'method': 'GRPO+OPSD', 'dataset': 'ALFWorld', 'metric': 'task_success_rate', 'expected_result': 'unstable'}] |
| core_contribution | — | SDAR (Self-Distilled Agentic Reinforcement Learning) treats On-Policy Self-Distillation (OPSD) as a gated auxiliary objective with RL (GRPO) as the primary optimization backbone. It maps detached token-level signals into a sigmoid gate g_t = sigma(beta * Delta_t), strengthening distillation on teacher-endorsed positive-gap tokens and softly attenuating negative teacher rejections. lambda=0.1, beta=10. |
| datasets | — | ['ALFWorld', 'WebShop', 'Search-QA'] |
| key_params | — | {'lambda': 0.1, 'beta': 10, 'gate': 'g_t=sigmoid(10*Delta_t)', 'stop_gradient': True} |
| per_model | {'qwen3_1_7b': {'search_qa': {'sdar': {'status': 'ok', 'metric': 0.054303844589571715, 'reward_mean': 0.054303844589571715, 'gate_mean': 0.4175413267686963, 'gate_active': 0.3173767340835184, 'zero_shot_f1': 0.06956104579015578, 'steps_run': 150, 'wall_time_s': 878.1680161952972, 'cell_id': 'qwen3_1_7b__sdar__search_qa__s42'}, 'grpo': {'status': 'ok', 'metric': 0.05668243855822004, 'reward_mean': 0.05668243855822004, 'gate_mean': 0.0, 'gate_active': 0.0, 'zero_shot_f1': 0.04914822620302072, 'steps_run': 150, 'wall_time_s': 824.5147993564606, 'cell_id': 'qwen3_1_7b__grpo__search_qa__s42'}, 'grpo_opsd': {'status': 'ok', 'metric': 0.05211247086247086, 'reward_mean': 0.05211247086247086, 'gate_mean': 0.9999999988824129, 'gate_active': 0.9999999988824129, 'zero_shot_f1': 0.06921722950684481, 'steps_run': 150, 'wall_time_s': 539.2502822875977, 'cell_id': 'qwen3_1_7b__grpo_opsd__search_qa__s42'}}}, 'qwen2_5_3b': {'search_qa': {'grpo': {'status': 'ok', 'metric': 0.14498335431348147, 'reward_mean': 0.14498335431348147, 'gate_mean': 0.0, 'gate_active': 0.0, 'zero_shot_f1': 0.05529471222451245, 'steps_run': 150, 'wall_time_s': 835.9527711868286, 'cell_id': 'qwen2_5_3b__grpo__search_qa__s42'}}}} | — |
| scope | {'models_run': ['qwen2_5_3b', 'qwen3_1_7b'], 'models_skipped': [], 'environments_skipped': [], 'gaps': []} | — |
| status | complete | — |

## Improvement Candidates

**1. Candidate 1** — promoted (0.146)
**2. Candidate 2** — failed (-0.323)
**3. Candidate 3** — promoted (0.04)
**4. Candidate 4** — promoted (0.04)
**5. Candidate 5** — failed (-0.008)
**6. Candidate 6** — failed
**7. Candidate 7** — promoted

## Cost

| Category | USD |
|---|---|
| Primitive-internal LLM | $3.554047 |
| **Total LLM** | **$3.554047** |

**Iterations:** 10

---
_Generated by ReproLab RLM orchestrator (Issue #60)._
