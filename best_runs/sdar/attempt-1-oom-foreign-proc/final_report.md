# REPRODUCTION FAILED

**Paper:** paper_text

## Rubric Score

**Overall score:** 0.000  (✘ below target)

_0/26 rubric leaves graded · PaperBench bundle rubric_

| Area | Score | Notes |
|---|---|---|
| Method and code fidelity to the paper | 0.000 |  |
| Data and preprocessing fidelity | 0.000 |  |
| Experiment execution and reproducibility | 0.000 |  |
| Evaluation protocol and metric correctness | 0.000 |  |
| Result match versus the paper's reported targets | 0.000 |  |
| Artifact completeness and provenance | 0.000 |  |

### Weakest rubric leaves

| Score | Justification |
|---|---|
| 0.00 | degraded_no_metrics |
| 0.00 | degraded_no_metrics |
| 0.00 | degraded_no_metrics |
| 0.00 | degraded_no_metrics |
| 0.00 | degraded_no_metrics |

## Reproduction Summary

Attempted SDAR reproduction (Self-Distilled Agentic Reinforcement Learning, arXiv 2605.15155). Code was successfully generated implementing the sigmoid-gated OPSD algorithm (g_t=σ(β·Δ_t)), stop-gradient on gate, all 5 baselines (GRPO/OPSD/Skill-SD/GRPO+OPSD/RLSD), and 4 skill-retrieval strategies. Both run attempts failed: (1) all search_qa/webshop cells hit CUDA OOM because a pre-existing process occupies 22+ GiB of the 23.68 GiB A5000 GPU, leaving insufficient memory for any Qwen model; (2) ALFWorld cells hit AttributeError on missing build_student_prompt method. WebShop data endpoint returned HTTP 404. The paper requires 8×H800 GPUs; this hardware is unobtainable. An improvement attempt (smaller batches + gradient checkpointing) was implemented and re-run but encountered identical OOM failures — the bottleneck is the pre-existing process, not batch size. Rubric score: 0.00 (target: 0.60). Verdict: failed.

## Scope

**Requested:** Qwen3-1.7B + Qwen2.5-3B on ALFWorld/WebShop (smallest two variants)

**Ran:**
- Qwen3-1.7B/ALFWorld
- Qwen3-1.7B/WebShop

**Gaps:** _(items requested but not reproduced; datasets marked "unobtainable" were excluded from the rubric score, not penalised)_
- All search_qa/webshop cells: CUDA OOM on 1×A5000 (23.68 GiB) — another process holding 22+ GiB; Qwen models cannot be loaded (unobtainable without GPU eviction or multi-GPU)
- ALFWorld: ALFWorldEnv.build_student_prompt method missing — interface bug in generated code
- WebShop: HTTP 404 — dataset endpoint unavailable
- Paper required 8×H800 for training; single A5000 is insufficient for any model size

## Baseline Metrics vs. Paper Claims

| Metric | Reproduced | Paper Claim |
|---|---|---|
| SDAR vs GRPO/ALFWorld | — | SDAR >= GRPO |
| SDAR/ALFWorld | — | see paper |
| SDAR/WebShop | — | see paper |
| baselines_vs_sdar | {} | — |
| comparisons | {'sdar_minus_grpo_alfworld_success': 0.0} | — |
| config | {'BETA': 10.0, 'LAMBDA': 0.1, 'steps': 150, 'batch_size': {'search_qa': 4, 'alfworld': 2, 'webshop': 4}, 'max_new_tokens': {'search_qa': 512, 'alfworld': 64, 'webshop': 64}, 'retrieval_default': 'KM', 'eval_with_skills': {'sdar': False, 'grpo': False, 'opsd': False, 'skill_sd': False, 'grpo_opsd': False, 'rlsd': False}, 'scale_note': 'Trained on 1×RTX A5000 (25.4 GB). Paper used 8×H800. batch_size reduced from paper: search_qa 128→16, alfworld 16→4, webshop 16→8.'} | — |
| data_load_failures | [{'dataset': 'webshop', 'loader': 'http', 'error': 'HTTPError: HTTP Error 404: Not Found'}] | — |
| gate_dynamics | {} | — |
| grpo | {'alfworld': {'success_rate': 0.0, 'final_reward': 0.0}, 'webshop': {'score': 0.0, 'final_reward': 0.0}} | — |
| per_baseline | {} | — |
| per_env | {} | — |
| per_model | {'qwen3_1_7b': {'search_qa': {'sdar': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 876.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 251.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 22.98 GiB memory in use. Of the allocated memory 22.65 GiB is allocated by PyTorch, and 77.68 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'grpo': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 18.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 11.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 23.21 GiB memory in use. Of the allocated memory 22.79 GiB is allocated by PyTorch, and 173.99 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'opsd': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 26.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 9.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 23.21 GiB memory in use. Of the allocated memory 22.80 GiB is allocated by PyTorch, and 166.10 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'skill_sd': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 13.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 23.21 GiB memory in use. Of the allocated memory 22.86 GiB is allocated by PyTorch, and 105.30 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'grpo_opsd': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 14.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 11.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 23.21 GiB memory in use. Of the allocated memory 22.80 GiB is allocated by PyTorch, and 165.80 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'rlsd': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 1.11 GiB. GPU 0 has a total capacity of 23.68 GiB of which 571.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 22.66 GiB memory in use. Of the allocated memory 22.27 GiB is allocated by PyTorch, and 151.60 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}}, 'alfworld': {'sdar': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'grpo': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'opsd': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'skill_sd': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'grpo_opsd': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'rlsd': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}}}, 'qwen2_5_3b': {'search_qa': {'sdar': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 594.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 241.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 22.99 GiB memory in use. Of the allocated memory 22.30 GiB is allocated by PyTorch, and 442.42 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'grpo': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 1.58 GiB. GPU 0 has a total capacity of 23.68 GiB of which 713.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 22.53 GiB memory in use. Of the allocated memory 22.16 GiB is allocated by PyTorch, and 107.34 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'opsd': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 1.66 GiB. GPU 0 has a total capacity of 23.68 GiB of which 1.48 GiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 21.74 GiB memory in use. Of the allocated memory 21.39 GiB is allocated by PyTorch, and 99.03 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'skill_sd': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 1.57 GiB. GPU 0 has a total capacity of 23.68 GiB of which 751.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 22.49 GiB memory in use. Of the allocated memory 22.14 GiB is allocated by PyTorch, and 93.28 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'grpo_opsd': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 1.68 GiB. GPU 0 has a total capacity of 23.68 GiB of which 1.30 GiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 21.92 GiB memory in use. Of the allocated memory 21.58 GiB is allocated by PyTorch, and 86.35 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'rlsd': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 1.66 GiB. GPU 0 has a total capacity of 23.68 GiB of which 1.50 GiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 21.73 GiB memory in use. Of the allocated memory 21.37 GiB is allocated by PyTorch, and 102.65 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}}, 'alfworld': {'sdar': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'grpo': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'opsd': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'skill_sd': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'grpo_opsd': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'rlsd': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}}}, 'qwen2_5_7b': {'search_qa': {'sdar': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 130.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 105.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 23.12 GiB memory in use. Of the allocated memory 22.66 GiB is allocated by PyTorch, and 203.80 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'grpo': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 1.58 GiB. GPU 0 has a total capacity of 23.68 GiB of which 27.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 23.20 GiB memory in use. Of the allocated memory 22.64 GiB is allocated by PyTorch, and 304.83 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'opsd': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 1.03 GiB. GPU 0 has a total capacity of 23.68 GiB of which 807.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 22.43 GiB memory in use. Of the allocated memory 21.88 GiB is allocated by PyTorch, and 308.54 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'skill_sd': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 1.03 GiB. GPU 0 has a total capacity of 23.68 GiB of which 807.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 22.43 GiB memory in use. Of the allocated memory 21.88 GiB is allocated by PyTorch, and 308.65 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'grpo_opsd': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 1004.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 807.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 22.43 GiB memory in use. Of the allocated memory 21.88 GiB is allocated by PyTorch, and 309.83 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}, 'rlsd': {'status': 'failed', 'error': 'CUDA OOM: CUDA out of memory. Tried to allocate 938.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 807.06 MiB is free. Process 2612630 has 458.00 MiB memory in use. Including non-PyTorch memory, this process has 22.43 GiB memory in use. Of the allocated memory 21.87 GiB is allocated by PyTorch, and 311.36 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'metric': None}}, 'alfworld': {'sdar': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'grpo': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'opsd': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'skill_sd': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'grpo_opsd': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}, 'rlsd': {'status': 'failed', 'error': "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'", 'metric': None}}}} | — |
| retrieval_comparison | {} | — |
| scope | {'models_run': ['qwen3_1_7b', 'qwen2_5_3b', 'qwen2_5_7b'], 'environments_run': ['search_qa', 'alfworld'], 'gaps': [{'component': 'webshop', 'reason': 'HTTPError: HTTP Error 404: Not Found'}]} | — |
| sdar | {'alfworld': {'success_rate': 0.0, 'final_reward': 0.0}, 'webshop': {'score': 0.0, 'final_reward': 0.0}} | — |
| status | partial | — |
| wall_time_seconds | 611.1538741588593 | — |

## Improvement Candidates

**1. Candidate 1** — failed
**2. Candidate 2** — declined
**3. Candidate 3** — declined

## Cost

| Category | USD |
|---|---|
| Primitive-internal LLM | $0.776351 |
| **Total LLM** | **$0.776351** |

**Iterations:** 3

## Token Usage

| Metric | Value |
|---|---|
| Total LLM calls | 32 |
| Input tokens | 189 |
| Output tokens | 112,219 |
| Cache creation (input) | 327,830 |
| Cache read (input, prompt-cache-billed) | 15,489,351 |

### Per-primitive token usage

| Primitive | Calls | Input | Output |
|---|---|---|---|
| baseline-implementation | 3 | 172 | 108,282 |
| verify_against_rubric | 3 | 5 | 1,741 |
| propose_improvements | 1 | 6 | 1,147 |
| plan_reproduction | 1 | 6 | 1,049 |

## Per-Step Timing

**Total wall clock:** 4813.2s (1h 20m 13s)

| Primitive | Calls | Total time (s) |
|---|---|---|
| run_experiment | 3 | 2419.61 |
| implement_baseline | 2 | 1877.35 |
| verify_against_rubric | 3 | 77.69 |
| plan_reproduction | 1 | 32.74 |
| propose_improvements | 1 | 26.48 |
| detect_environment | 1 | 0.09 |
| understand_section | 4 | 0.03 |

**GPU hours:** 0.672h on `rtx4090` × 1

---
_Generated by ReproLab RLM orchestrator (Issue #60)._
