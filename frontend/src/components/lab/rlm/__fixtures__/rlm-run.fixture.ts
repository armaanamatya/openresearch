/**
 * Hand-authored RLM run fixture — "Attention is all you need" (Transformer).
 * Exported as RlmDashboardEvent[]. This is the single test contract for every
 * downstream task (reducer, components, e2e).
 *
 * Scenario coverage (spec §10):
 *  - Full trunk: comprehension (i1-i3) → environment (i4-i6, 1 failed build) → baseline (i7-i9)
 *  - Rubric baseline ~0.22 with ≥1 fail area
 *  - Round 1: 6 candidates (c1-c6), all six outcome types represented:
 *      c1=promoted, c2=marginal, c3=failed, c4=promoted (out-of-order pair),
 *      c5=declined, c6=declined  →  2 declined in round 1 ✓
 *  - OUT-OF-ORDER: c4 candidate_outcome("running") emitted BEFORE c4 candidate_proposed
 *  - Round 2: 3 candidates (c7-c9), NO promotions (running/marginal/skipped)
 *    includes a sub_rlm_spawned + sub_rlm_complete pair
 *  - Final rubric ~0.53, run_complete (status: completed)
 *  - No parent_id on any candidate_proposed (reducer must infer)
 */

import type { RlmDashboardEvent } from "../../../../lib/events/rlm-events";

// Monotonically increasing ISO-8601 timestamps.
// Base: 2026-05-21T14:00:00.000Z, increment 8 s per event.
let _ts = new Date("2026-05-21T14:00:00.000Z").getTime();
function ts(): string {
  const result = new Date(_ts).toISOString();
  _ts += 8000;
  return result;
}

export const rlmRunFixture: RlmDashboardEvent[] = [
  // ─── Phase 1: Comprehension (iterations 1-3) ─────────────────────────────

  // i1 — understand_section start
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "understand_section",
    status: "start",
    args_summary: { section: "string[21]", depth: "int" },
    result_summary: null,
    iteration: 1,
    rubric_delta: null,
  },
  // i1 — repl_iteration
  {
    event: "repl_iteration",
    timestamp: ts(),
    iteration: 1,
    response:
      "Beginning paper comprehension. The Transformer architecture eliminates recurrence entirely, relying solely on attention mechanisms. Key claims: BLEU 28.4 (EN-DE), 41.0 (EN-FR) on WMT. I will start by extracting the abstract and introduction.",
    code_blocks: [
      {
        code: "paper = load_paper('attention_is_all_you_need.pdf')\nsections = extract_sections(paper)",
        stdout_meta: { length: 128, prefix: "Loaded 15 sections\nExtracted", has_traceback: false },
        stderr_meta: { length: 0, prefix: "", has_traceback: false },
        vars: {
          paper: { type: "PaperDoc", size: 1 },
          paper_text: { type: "str", size: 48230 },
          sections: { type: "list[Section]", size: 15 },
        },
        sub_calls: 0,
      },
    ],
    sub_calls: 0,
    timing: 3.21,
  },
  // i1 — understand_section ok
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "understand_section",
    status: "ok",
    args_summary: { section: "string[21]", depth: "int" },
    result_summary: "dict[6]",
    iteration: 1,
    rubric_delta: null,
  },

  // i2 — understand_section (model architecture) + extract_hyperparameters
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "understand_section",
    status: "start",
    args_summary: { section: "string[19]", depth: "int" },
    result_summary: null,
    iteration: 2,
    rubric_delta: null,
  },
  {
    event: "repl_iteration",
    timestamp: ts(),
    iteration: 2,
    response:
      "Analysing the model architecture section. Multi-head attention uses h=8 heads, d_model=512, d_ff=2048. Positional encoding uses sine/cosine. Dropout 0.1. Now extracting hyperparameters systematically.",
    code_blocks: [
      {
        code: "arch = understand_section(sections['model_architecture'])\nhyperparams = extract_hyperparameters(arch)",
        stdout_meta: { length: 96, prefix: "d_model=512  h=8", has_traceback: false },
        stderr_meta: { length: 0, prefix: "", has_traceback: false },
        vars: {
          paper: { type: "PaperDoc", size: 1 },
          sections: { type: "list[Section]", size: 15 },
          arch: { type: "dict[str,Any]", size: 12 },
          hyperparams: { type: "dict[str,Any]", size: 18 },
        },
        sub_calls: 1,
      },
    ],
    sub_calls: 1,
    timing: 4.87,
  },
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "understand_section",
    status: "ok",
    args_summary: { section: "string[19]", depth: "int" },
    result_summary: "dict[12]",
    iteration: 2,
    rubric_delta: null,
  },
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "extract_hyperparameters",
    status: "start",
    args_summary: { source: "dict[12]", fields: "list[18]" },
    result_summary: null,
    iteration: 2,
    rubric_delta: null,
  },
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "extract_hyperparameters",
    status: "ok",
    args_summary: { source: "dict[12]", fields: "list[18]" },
    result_summary: "dict[18]",
    iteration: 2,
    rubric_delta: null,
  },

  // i3 — understand_section (training details)
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "understand_section",
    status: "start",
    args_summary: { section: "string[16]", depth: "int" },
    result_summary: null,
    iteration: 3,
    rubric_delta: null,
  },
  {
    event: "repl_iteration",
    timestamp: ts(),
    iteration: 3,
    response:
      "Training details: Adam optimiser (β1=0.9, β2=0.98, ε=1e-9), warmup_steps=4000, label smoothing ε_ls=0.1. Trained 300k steps on 8×P100. Dataset: WMT2014 EN-DE (4.5M pairs) and EN-FR (36M pairs). I have enough to proceed to environment detection.",
    code_blocks: [
      {
        code: "training = understand_section(sections['training'])\nhyperparams.update(extract_hyperparameters(training))",
        stdout_meta: { length: 112, prefix: "warmup_steps=4000", has_traceback: false },
        stderr_meta: { length: 0, prefix: "", has_traceback: false },
        vars: {
          paper: { type: "PaperDoc", size: 1 },
          sections: { type: "list[Section]", size: 15 },
          arch: { type: "dict[str,Any]", size: 12 },
          hyperparams: { type: "dict[str,Any]", size: 24 },
          training: { type: "dict[str,Any]", size: 9 },
        },
        sub_calls: 1,
      },
    ],
    sub_calls: 1,
    timing: 5.14,
  },
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "understand_section",
    status: "ok",
    args_summary: { section: "string[16]", depth: "int" },
    result_summary: "dict[9]",
    iteration: 3,
    rubric_delta: null,
  },

  // ─── Phase 2: Environment (iterations 4-6) ───────────────────────────────

  // i4 — detect_environment
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "detect_environment",
    status: "start",
    args_summary: { paper_meta: "dict[24]" },
    result_summary: null,
    iteration: 4,
    rubric_delta: null,
  },
  {
    event: "repl_iteration",
    timestamp: ts(),
    iteration: 4,
    response:
      "Detecting the required environment. The paper uses PyTorch 1.x-compatible code; no official repo but fairseq/the annotated transformer are reference implementations. CUDA required for GPU training. Generating Dockerfile.",
    code_blocks: [
      {
        code: "env_spec = detect_environment(hyperparams)\nprint(env_spec['framework'], env_spec['cuda'])",
        stdout_meta: { length: 40, prefix: "pytorch 2.3.0", has_traceback: false },
        stderr_meta: { length: 0, prefix: "", has_traceback: false },
        vars: {
          paper: { type: "PaperDoc", size: 1 },
          sections: { type: "list[Section]", size: 15 },
          arch: { type: "dict[str,Any]", size: 12 },
          hyperparams: { type: "dict[str,Any]", size: 24 },
          training: { type: "dict[str,Any]", size: 9 },
          env_spec: { type: "dict[str,Any]", size: 7 },
        },
        sub_calls: 0,
      },
    ],
    sub_calls: 0,
    timing: 2.98,
  },
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "detect_environment",
    status: "ok",
    args_summary: { paper_meta: "dict[24]" },
    result_summary: "dict[7]",
    iteration: 4,
    rubric_delta: null,
  },

  // i5 — build_environment attempt 1 → error (torchtext version conflict)
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "build_environment",
    status: "start",
    args_summary: { env_spec: "dict[7]", dockerfile: "string[842]" },
    result_summary: null,
    iteration: 5,
    rubric_delta: null,
  },
  {
    event: "repl_iteration",
    timestamp: ts(),
    iteration: 5,
    response:
      "First Docker build attempt. Expecting a base pytorch image with CUDA 12.1. Installing torchtext, sacrebleu, subword-nmt. Build failed: torchtext version conflict with torch 2.3.0. Will pin torchtext==0.18.0 and retry.",
    code_blocks: [
      {
        code: "build_result = build_environment(env_spec, attempt=1)\nprint(build_result['error'])",
        stdout_meta: {
          length: 184,
          prefix: "ERROR: torchtext 0.19.0 requires torch>=2.4.0",
          has_traceback: false,
        },
        stderr_meta: { length: 64, prefix: "pip ERROR: Could not find", has_traceback: false },
        vars: {
          paper: { type: "PaperDoc", size: 1 },
          sections: { type: "list[Section]", size: 15 },
          arch: { type: "dict[str,Any]", size: 12 },
          hyperparams: { type: "dict[str,Any]", size: 24 },
          training: { type: "dict[str,Any]", size: 9 },
          env_spec: { type: "dict[str,Any]", size: 7 },
          build_result: { type: "dict[str,Any]", size: 3 },
        },
        sub_calls: 0,
      },
    ],
    sub_calls: 0,
    timing: 18.4,
  },
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "build_environment",
    status: "error",
    args_summary: { env_spec: "dict[7]", dockerfile: "string[842]" },
    result_summary: null,
    iteration: 5,
    rubric_delta: null,
  },

  // i6 — build_environment attempt 2 → ok
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "build_environment",
    status: "start",
    args_summary: { env_spec: "dict[7]", dockerfile: "string[891]" },
    result_summary: null,
    iteration: 6,
    rubric_delta: null,
  },
  {
    event: "repl_iteration",
    timestamp: ts(),
    iteration: 6,
    response:
      "Retrying build with pinned torchtext==0.18.0 and torch==2.3.0. Build succeeded. Environment verified: Python 3.11, PyTorch 2.3.0+cu121, torchtext 0.18.0, sacrebleu 2.3.1.",
    code_blocks: [
      {
        code: "env_spec['packages']['torchtext'] = '0.18.0'\nbuild_result2 = build_environment(env_spec, attempt=2)\nprint(build_result2['status'])",
        stdout_meta: { length: 24, prefix: "BUILD_SUCCESS", has_traceback: false },
        stderr_meta: { length: 0, prefix: "", has_traceback: false },
        vars: {
          paper: { type: "PaperDoc", size: 1 },
          sections: { type: "list[Section]", size: 15 },
          arch: { type: "dict[str,Any]", size: 12 },
          hyperparams: { type: "dict[str,Any]", size: 24 },
          training: { type: "dict[str,Any]", size: 9 },
          env_spec: { type: "dict[str,Any]", size: 7 },
          build_result: { type: "dict[str,Any]", size: 3 },
          build_result2: { type: "dict[str,Any]", size: 4 },
        },
        sub_calls: 0,
      },
    ],
    sub_calls: 0,
    timing: 22.1,
  },
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "build_environment",
    status: "ok",
    args_summary: { env_spec: "dict[7]", dockerfile: "string[891]" },
    result_summary: "dict[4]",
    iteration: 6,
    rubric_delta: null,
  },

  // ─── Phase 3: Baseline (iterations 7-9) ──────────────────────────────────

  // i7 — implement_baseline
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "implement_baseline",
    status: "start",
    args_summary: { arch: "dict[12]", hyperparams: "dict[24]" },
    result_summary: null,
    iteration: 7,
    rubric_delta: null,
  },
  {
    event: "repl_iteration",
    timestamp: ts(),
    iteration: 7,
    response:
      "Implementing the Transformer baseline. Writing the encoder-decoder stack with multi-head self-attention, encoder-decoder attention, and position-wise FFN. Using the annotated Transformer as a reference for correctness.",
    code_blocks: [
      {
        code: "baseline_code = implement_baseline(arch, hyperparams)\nprint(len(baseline_code['files']), 'files written')",
        stdout_meta: { length: 24, prefix: "11 files written", has_traceback: false },
        stderr_meta: { length: 0, prefix: "", has_traceback: false },
        vars: {
          paper: { type: "PaperDoc", size: 1 },
          sections: { type: "list[Section]", size: 15 },
          arch: { type: "dict[str,Any]", size: 12 },
          hyperparams: { type: "dict[str,Any]", size: 24 },
          training: { type: "dict[str,Any]", size: 9 },
          env_spec: { type: "dict[str,Any]", size: 7 },
          build_result2: { type: "dict[str,Any]", size: 4 },
          baseline_code: { type: "dict[str,list]", size: 11 },
        },
        sub_calls: 0,
      },
    ],
    sub_calls: 0,
    timing: 12.3,
  },
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "implement_baseline",
    status: "ok",
    args_summary: { arch: "dict[12]", hyperparams: "dict[24]" },
    result_summary: "dict[11]",
    iteration: 7,
    rubric_delta: null,
  },

  // i8 — run_experiment
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "run_experiment",
    status: "start",
    args_summary: { baseline_code: "dict[11]", epochs: "int", batch_size: "int" },
    result_summary: null,
    iteration: 8,
    rubric_delta: null,
  },
  {
    event: "repl_iteration",
    timestamp: ts(),
    iteration: 8,
    response:
      "Running the baseline experiment. Training for a reduced 5k steps on WMT14 EN-DE toy subset to evaluate reproducibility. BLEU scores will be compared against the paper's claims (28.4 big, 25.8 base).",
    code_blocks: [
      {
        code: "exp_result = run_experiment(baseline_code, steps=5000, dataset='wmt14_ende_toy')\nprint('BLEU:', exp_result['bleu'])",
        stdout_meta: { length: 32, prefix: "BLEU: 18.3", has_traceback: false },
        stderr_meta: { length: 0, prefix: "", has_traceback: false },
        vars: {
          paper: { type: "PaperDoc", size: 1 },
          sections: { type: "list[Section]", size: 15 },
          arch: { type: "dict[str,Any]", size: 12 },
          hyperparams: { type: "dict[str,Any]", size: 24 },
          training: { type: "dict[str,Any]", size: 9 },
          env_spec: { type: "dict[str,Any]", size: 7 },
          build_result2: { type: "dict[str,Any]", size: 4 },
          baseline_code: { type: "dict[str,list]", size: 11 },
          exp_result: { type: "dict[str,Any]", size: 6 },
        },
        sub_calls: 0,
      },
    ],
    sub_calls: 0,
    timing: 94.7,
  },
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "run_experiment",
    status: "ok",
    args_summary: { baseline_code: "dict[11]", epochs: "int", batch_size: "int" },
    result_summary: "dict[6]",
    iteration: 8,
    rubric_delta: null,
  },

  // i9 — verify_against_rubric → rubric baseline ~0.22 (2 fail areas)
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "verify_against_rubric",
    status: "start",
    args_summary: { exp_result: "dict[6]", rubric_spec: "dict[8]" },
    result_summary: null,
    iteration: 9,
    rubric_delta: null,
  },
  {
    event: "repl_iteration",
    timestamp: ts(),
    iteration: 9,
    response:
      "Verifying baseline against rubric. BLEU 18.3 vs target 25.8 (base model). Architecture correct (attention heads, d_model, d_ff). Training procedure partial — warmup schedule present but label smoothing off. Overall 22% of rubric met.",
    code_blocks: [
      {
        code: "rubric_result = verify_against_rubric(exp_result, rubric_spec)\nprint('score:', rubric_result['score'])",
        stdout_meta: { length: 16, prefix: "score: 0.22", has_traceback: false },
        stderr_meta: { length: 0, prefix: "", has_traceback: false },
        vars: {
          paper: { type: "PaperDoc", size: 1 },
          sections: { type: "list[Section]", size: 15 },
          arch: { type: "dict[str,Any]", size: 12 },
          hyperparams: { type: "dict[str,Any]", size: 24 },
          training: { type: "dict[str,Any]", size: 9 },
          env_spec: { type: "dict[str,Any]", size: 7 },
          build_result2: { type: "dict[str,Any]", size: 4 },
          baseline_code: { type: "dict[str,list]", size: 11 },
          exp_result: { type: "dict[str,Any]", size: 6 },
          baseline_metrics: { type: "dict[str,float]", size: 4 },
        },
        sub_calls: 0,
      },
    ],
    sub_calls: 0,
    timing: 3.2,
  },
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "verify_against_rubric",
    status: "ok",
    args_summary: { exp_result: "dict[6]", rubric_spec: "dict[8]" },
    result_summary: "dict[4]",
    iteration: 9,
    rubric_delta: null,
  },
  // Rubric baseline — 2 fail areas
  {
    event: "rubric_score",
    timestamp: ts(),
    iteration: 9,
    score: 0.22,
    target: 0.70,
    areas: [
      { area: "architecture_correctness", score: 0.75, weight: 0.25, status: "pass" },
      { area: "training_procedure", score: 0.30, weight: 0.25, status: "partial" },
      { area: "bleu_score_ende", score: 0.10, weight: 0.30, status: "fail" },
      { area: "bleu_score_enfr", score: 0.00, weight: 0.20, status: "fail" },
    ],
  },

  // ─── Round 1: propose_improvements (iteration 10) ────────────────────────
  // 6 candidates: c1=promoted, c2=marginal, c3=failed, c4=promoted (out-of-order),
  //               c5=declined, c6=declined  →  2 declined in round 1

  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "propose_improvements",
    status: "start",
    args_summary: { rubric_result: "dict[4]", history: "list[9]" },
    result_summary: null,
    iteration: 10,
    rubric_delta: null,
  },
  {
    event: "repl_iteration",
    timestamp: ts(),
    iteration: 10,
    response:
      "Proposing improvements to address the weak BLEU and incomplete training procedure. Key gaps: (1) no LR warmup schedule in code, (2) no attention dropout, (3) pre-norm placement unverified, (4) label smoothing missing, (5) beam search not implemented, (6) weight tying not applied.",
    code_blocks: [
      {
        code: "improvements = propose_improvements(rubric_result, history, n=6)\nfor imp in improvements:\n    print(imp['id'], imp['title'])",
        stdout_meta: {
          length: 192,
          prefix: "c1 learning-rate warmup\nc2 attention dropout",
          has_traceback: false,
        },
        stderr_meta: { length: 0, prefix: "", has_traceback: false },
        vars: {
          paper: { type: "PaperDoc", size: 1 },
          sections: { type: "list[Section]", size: 15 },
          arch: { type: "dict[str,Any]", size: 12 },
          hyperparams: { type: "dict[str,Any]", size: 24 },
          training: { type: "dict[str,Any]", size: 9 },
          env_spec: { type: "dict[str,Any]", size: 7 },
          build_result2: { type: "dict[str,Any]", size: 4 },
          baseline_code: { type: "dict[str,list]", size: 11 },
          exp_result: { type: "dict[str,Any]", size: 6 },
          baseline_metrics: { type: "dict[str,float]", size: 4 },
          improvements: { type: "list[dict]", size: 6 },
        },
        sub_calls: 0,
      },
    ],
    sub_calls: 0,
    timing: 4.1,
  },
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "propose_improvements",
    status: "ok",
    args_summary: { rubric_result: "dict[4]", history: "list[9]" },
    result_summary: "list[6]",
    iteration: 10,
    rubric_delta: null,
  },

  // ── OUT-OF-ORDER PAIR ──────────────────────────────────────────────────────
  // c4's outcome("running") arrives BEFORE c4's candidate_proposed.
  // The reducer must tolerate this; tests pin the behavior (spec §5.3).
  {
    event: "candidate_outcome",
    timestamp: ts(),
    iteration: 10,
    candidate_id: "c4",
    outcome: "running",
    rubric_delta: null,
  },

  // Round 1 candidate_proposed ×6 — no parent_id on any (reducer must infer)
  {
    event: "candidate_proposed",
    timestamp: ts(),
    iteration: 10,
    round: 1,
    candidate: {
      id: "c1",
      title: "learning-rate warmup",
      category: "optimizer",
      description:
        "Implement the Noam learning-rate schedule with warmup_steps=4000 as described in §5.3. The current flat LR is the primary cause of the low BLEU.",
      reasoning:
        "The paper explicitly requires the warmup schedule for stable training; without it the model diverges early.",
    },
  },
  {
    event: "candidate_proposed",
    timestamp: ts(),
    iteration: 10,
    round: 1,
    candidate: {
      id: "c2",
      title: "attention dropout",
      category: "regularisation",
      description:
        "Apply dropout (p=0.1) to the attention weights inside each multi-head attention layer before the softmax-weighted sum.",
      reasoning:
        "The paper specifies attention dropout; omitting it causes overfitting and degrades generalisation.",
    },
  },
  {
    event: "candidate_proposed",
    timestamp: ts(),
    iteration: 10,
    round: 1,
    candidate: {
      id: "c3",
      title: "pre-norm placement",
      category: "architecture",
      description:
        "Move LayerNorm before (rather than after) each sub-layer (Pre-LN variant). Verify this matches the paper's formulation exactly.",
      reasoning:
        "Placement of LayerNorm differs between implementations and can account for ~0.4 BLEU at scale.",
    },
  },
  // c4 proposed — its "running" outcome was already emitted above (out-of-order)
  {
    event: "candidate_proposed",
    timestamp: ts(),
    iteration: 10,
    round: 1,
    candidate: {
      id: "c4",
      title: "label-smoothing sweep",
      category: "training",
      description:
        "Enable label smoothing (ε_ls=0.1) in the cross-entropy loss as specified in §5.4 of the paper.",
      reasoning:
        "Label smoothing is an explicit training detail that improves perplexity and BLEU; missing it contributes to the gap.",
    },
  },
  {
    event: "candidate_proposed",
    timestamp: ts(),
    iteration: 10,
    round: 1,
    candidate: {
      id: "c5",
      title: "beam search decoder",
      category: "decoding",
      description:
        "Replace greedy decoding with beam search (beam_size=4, length_penalty=0.6) for inference BLEU measurement.",
      reasoning:
        "All reported BLEU numbers in the paper use beam search; greedy decoding underestimates by 1-2 BLEU.",
    },
  },
  {
    event: "candidate_proposed",
    timestamp: ts(),
    iteration: 10,
    round: 1,
    candidate: {
      id: "c6",
      title: "output weight tying",
      category: "architecture",
      description:
        "Tie the weights between the embedding layer and the pre-softmax linear projection as described in the paper.",
      reasoning:
        "Weight tying reduces parameters and typically improves perplexity; the paper reports it in the model-variant ablation.",
    },
  },

  // Round 1 outcomes — interleaved with rubric_score events as candidates resolve

  // i11 — root evaluates c1 (warmup applied) and c2 (attention dropout) outcomes
  {
    event: "repl_iteration",
    timestamp: ts(),
    iteration: 11,
    response:
      "Evaluating round-1 results for c1 (learning-rate warmup) and c2 (attention dropout). The warmup schedule dramatically stabilised training — BLEU on the toy subset jumped from 18.3 to ~20.1 after only 5k steps, a clear signal. Attention dropout (c2) shows a marginal +0.2 BLEU; worth keeping but not the bottleneck. Marking c1=promoted, c2=marginal.",
    code_blocks: [
      {
        code: "c1_result = evaluate_candidate('c1', baseline_code, hyperparams)\nc2_result = evaluate_candidate('c2', baseline_code, hyperparams)\nprint('c1 BLEU:', c1_result['bleu'], '| c2 BLEU:', c2_result['bleu'])",
        stdout_meta: { length: 48, prefix: "c1 BLEU: 20.1 | c2 BLEU: 18.5", has_traceback: false },
        stderr_meta: { length: 0, prefix: "", has_traceback: false },
        vars: {
          paper: { type: "PaperDoc", size: 1 },
          sections: { type: "list[Section]", size: 15 },
          arch: { type: "dict[str,Any]", size: 12 },
          hyperparams: { type: "dict[str,Any]", size: 24 },
          training: { type: "dict[str,Any]", size: 9 },
          env_spec: { type: "dict[str,Any]", size: 7 },
          build_result2: { type: "dict[str,Any]", size: 4 },
          baseline_code: { type: "dict[str,list]", size: 11 },
          exp_result: { type: "dict[str,Any]", size: 6 },
          baseline_metrics: { type: "dict[str,float]", size: 4 },
          improvements: { type: "list[dict]", size: 6 },
          applied_candidates: { type: "dict[str,dict]", size: 2 },
        },
        sub_calls: 0,
      },
    ],
    sub_calls: 0,
    timing: 11.2,
  },
  {
    event: "candidate_outcome",
    timestamp: ts(),
    iteration: 11,
    candidate_id: "c1",
    outcome: "promoted",
    rubric_delta: 0.09,
  },
  {
    event: "rubric_score",
    timestamp: ts(),
    iteration: 11,
    score: 0.31,
    target: 0.70,
    areas: [
      { area: "architecture_correctness", score: 0.75, weight: 0.25, status: "pass" },
      { area: "training_procedure", score: 0.50, weight: 0.25, status: "partial" },
      { area: "bleu_score_ende", score: 0.15, weight: 0.30, status: "fail" },
      { area: "bleu_score_enfr", score: 0.00, weight: 0.20, status: "fail" },
    ],
  },
  // c2 — marginal (attention dropout helps but benefit is small at 5k steps)
  {
    event: "candidate_outcome",
    timestamp: ts(),
    iteration: 11,
    candidate_id: "c2",
    outcome: "marginal",
    rubric_delta: 0.02,
  },
  {
    event: "rubric_score",
    timestamp: ts(),
    iteration: 11,
    score: 0.33,
    target: 0.70,
    areas: [
      { area: "architecture_correctness", score: 0.76, weight: 0.25, status: "pass" },
      { area: "training_procedure", score: 0.52, weight: 0.25, status: "partial" },
      { area: "bleu_score_ende", score: 0.16, weight: 0.30, status: "fail" },
      { area: "bleu_score_enfr", score: 0.00, weight: 0.20, status: "fail" },
    ],
  },
  // i12 — root evaluates c3/c4/c5/c6 outcomes
  {
    event: "repl_iteration",
    timestamp: ts(),
    iteration: 12,
    response:
      "Evaluating remaining round-1 candidates. Pre-norm placement (c3) regressed BLEU by 0.2 — the paper uses post-LN; reverting. Label smoothing (c4) added +0.4 BLEU and reduced perplexity measurably — promoting. Beam search (c5) and weight tying (c6) are impractical at 5k steps on the toy subset: declining both for this run; they remain candidates for a full 300k-step reproduction.",
    code_blocks: [
      {
        code: "c3_result = evaluate_candidate('c3', baseline_code, hyperparams)\nc4_result = evaluate_candidate('c4', baseline_code, hyperparams)\nround1_results = {'c1': c1_result, 'c2': c2_result, 'c3': c3_result, 'c4': c4_result}\nprint('round1 deltas:', {k: v['bleu_delta'] for k,v in round1_results.items()})",
        stdout_meta: {
          length: 72,
          prefix: "round1 deltas: {'c1': +1.8, 'c2': +0.2, 'c3': -0.2, 'c4': +0.4}",
          has_traceback: false,
        },
        stderr_meta: { length: 0, prefix: "", has_traceback: false },
        vars: {
          paper: { type: "PaperDoc", size: 1 },
          sections: { type: "list[Section]", size: 15 },
          arch: { type: "dict[str,Any]", size: 12 },
          hyperparams: { type: "dict[str,Any]", size: 24 },
          training: { type: "dict[str,Any]", size: 9 },
          env_spec: { type: "dict[str,Any]", size: 7 },
          build_result2: { type: "dict[str,Any]", size: 4 },
          baseline_code: { type: "dict[str,list]", size: 11 },
          exp_result: { type: "dict[str,Any]", size: 6 },
          baseline_metrics: { type: "dict[str,float]", size: 4 },
          improvements: { type: "list[dict]", size: 6 },
          applied_candidates: { type: "dict[str,dict]", size: 4 },
          round1_results: { type: "dict[str,dict]", size: 4 },
        },
        sub_calls: 0,
      },
    ],
    sub_calls: 0,
    timing: 9.8,
  },
  // c3 — failed (pre-norm regresses quality)
  {
    event: "candidate_outcome",
    timestamp: ts(),
    iteration: 12,
    candidate_id: "c3",
    outcome: "failed",
    rubric_delta: -0.02,
  },
  // c4 — final outcome (its "running" status was set out-of-order before its proposed)
  {
    event: "candidate_outcome",
    timestamp: ts(),
    iteration: 12,
    candidate_id: "c4",
    outcome: "promoted",
    rubric_delta: 0.05,
  },
  {
    event: "rubric_score",
    timestamp: ts(),
    iteration: 12,
    score: 0.42,
    target: 0.70,
    areas: [
      { area: "architecture_correctness", score: 0.82, weight: 0.25, status: "pass" },
      { area: "training_procedure", score: 0.70, weight: 0.25, status: "pass" },
      { area: "bleu_score_ende", score: 0.22, weight: 0.30, status: "fail" },
      { area: "bleu_score_enfr", score: 0.00, weight: 0.20, status: "fail" },
    ],
  },
  // c5 — declined (requires >300k steps to see benefit; deferred)
  {
    event: "candidate_outcome",
    timestamp: ts(),
    iteration: 12,
    candidate_id: "c5",
    outcome: "declined",
    rubric_delta: null,
  },
  // c6 — declined (weight tying needs full vocab training; not viable at 5k steps)
  // 2nd declined in round 1 → satisfies the ≥2 declined requirement
  {
    event: "candidate_outcome",
    timestamp: ts(),
    iteration: 12,
    candidate_id: "c6",
    outcome: "declined",
    rubric_delta: null,
  },

  // ─── Round 2: branches off the most-recent promoted node (c4) ───────────
  // Round 2 promotes NOTHING — all outcomes are running / marginal / skipped.
  // Round-2 candidates still parent via §5.3 branch (a) on c4 (the last promoted
  // node from round 1). Branch (b) — the previous fan's parent — would only fire
  // for a hypothetical round 3 where round 2 had produced no promoted node.

  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "propose_improvements",
    status: "start",
    args_summary: { rubric_result: "dict[4]", history: "list[12]" },
    result_summary: null,
    iteration: 13,
    rubric_delta: null,
  },
  {
    event: "repl_iteration",
    timestamp: ts(),
    iteration: 13,
    response:
      "Building on warmup, attention-dropout, and label-smoothing improvements. Proposing second-round refinements: (1) beam-size sweep, (2) BPE vocabulary tuning, (3) mixed-precision training.",
    code_blocks: [
      {
        code: "round2 = propose_improvements(rubric_result, history, n=3)\nfor imp in round2:\n    print(imp['id'], imp['title'])",
        stdout_meta: {
          length: 96,
          prefix: "c7 beam-size sweep\nc8 bpe-vocab-tuning",
          has_traceback: false,
        },
        stderr_meta: { length: 0, prefix: "", has_traceback: false },
        vars: {
          paper: { type: "PaperDoc", size: 1 },
          sections: { type: "list[Section]", size: 15 },
          arch: { type: "dict[str,Any]", size: 12 },
          hyperparams: { type: "dict[str,Any]", size: 24 },
          training: { type: "dict[str,Any]", size: 9 },
          env_spec: { type: "dict[str,Any]", size: 7 },
          build_result2: { type: "dict[str,Any]", size: 4 },
          baseline_code: { type: "dict[str,list]", size: 11 },
          exp_result: { type: "dict[str,Any]", size: 6 },
          baseline_metrics: { type: "dict[str,float]", size: 4 },
          improvements: { type: "list[dict]", size: 6 },
          round2_improvements: { type: "list[dict]", size: 3 },
        },
        sub_calls: 0,
      },
    ],
    sub_calls: 0,
    timing: 3.9,
  },
  {
    event: "primitive_call",
    timestamp: ts(),
    primitive: "propose_improvements",
    status: "ok",
    args_summary: { rubric_result: "dict[4]", history: "list[12]" },
    result_summary: "list[3]",
    iteration: 13,
    rubric_delta: null,
  },

  // Sub-RLM spawned to evaluate the beam-size sweep in a sub-run
  {
    event: "sub_rlm_spawned",
    timestamp: ts(),
    depth: 1,
    model: "claude-sonnet-4-6",
    prompt_preview:
      "Evaluate beam sizes [2, 4, 6] on WMT14 EN-DE toy subset. Baseline BLEU 18.3 (greedy). Report BLEU per beam size and select the best.",
  },

  // Round 2 candidate_proposed ×3 — no parent_id
  {
    event: "candidate_proposed",
    timestamp: ts(),
    iteration: 13,
    round: 2,
    candidate: {
      id: "c7",
      title: "beam-size sweep",
      category: "decoding",
      description:
        "Sweep beam sizes 2, 4, and 6 at inference time to find the optimal beam size for WMT14 EN-DE.",
      reasoning:
        "The paper uses beam=4 with length_penalty=0.6; a sweep confirms this is optimal and adds 0.5-1 BLEU over greedy.",
    },
  },
  {
    event: "candidate_proposed",
    timestamp: ts(),
    iteration: 13,
    round: 2,
    candidate: {
      id: "c8",
      title: "bpe-vocab-tuning",
      category: "data",
      description:
        "Tune the BPE vocabulary size from 32k to 37k shared tokens as used in the paper's EN-FR model.",
      reasoning:
        "The paper uses a larger shared BPE vocabulary for EN-FR; worth testing on EN-DE to close the remaining BLEU gap.",
    },
  },
  {
    event: "candidate_proposed",
    timestamp: ts(),
    iteration: 13,
    round: 2,
    candidate: {
      id: "c9",
      title: "mixed-precision training",
      category: "efficiency",
      description:
        "Enable fp16 mixed-precision training via torch.cuda.amp to match the paper's training throughput on a single GPU.",
      reasoning:
        "Mixed precision can double throughput and allow larger batch sizes, moving the reproduction closer to the reported steps/sec.",
    },
  },

  // Sub-RLM completes
  {
    event: "sub_rlm_complete",
    timestamp: ts(),
    depth: 1,
    model: "claude-sonnet-4-6",
    duration_ms: 28400,
    error: null,
  },

  // i14 — root evaluates round-2 outcomes; wraps up
  {
    event: "repl_iteration",
    timestamp: ts(),
    iteration: 14,
    response:
      "Evaluating round-2 results. Beam-size sweep (c7): sub-RLM reported beam=4 optimal at BLEU 19.8 (+1.5 over greedy baseline), but the experiment is still running at the time of this iteration — marking as running. BPE vocab tuning (c8): marginal +0.1 BLEU, not worth the tokeniser retrain overhead for this short run. Mixed-precision (c9): deferred — throughput improvement only, skipping for quality-focused run. No promotions in round 2; final rubric reflects cumulative gains from round 1.",
    code_blocks: [
      {
        code: "round2_results = collect_round2_outcomes()\nbleu_per_beam = round2_results.get('c7', {}).get('bleu_per_beam', {})\nprint('beam sweep:', bleu_per_beam)\nprint('c8 delta:', round2_results.get('c8', {}).get('bleu_delta', 0))",
        stdout_meta: {
          length: 80,
          prefix: "beam sweep: {2: 19.2, 4: 19.8, 6: 19.7}",
          has_traceback: false,
        },
        stderr_meta: { length: 0, prefix: "", has_traceback: false },
        vars: {
          paper: { type: "PaperDoc", size: 1 },
          sections: { type: "list[Section]", size: 15 },
          arch: { type: "dict[str,Any]", size: 12 },
          hyperparams: { type: "dict[str,Any]", size: 24 },
          training: { type: "dict[str,Any]", size: 9 },
          env_spec: { type: "dict[str,Any]", size: 7 },
          build_result2: { type: "dict[str,Any]", size: 4 },
          baseline_code: { type: "dict[str,list]", size: 11 },
          exp_result: { type: "dict[str,Any]", size: 6 },
          baseline_metrics: { type: "dict[str,float]", size: 4 },
          improvements: { type: "list[dict]", size: 6 },
          applied_candidates: { type: "dict[str,dict]", size: 4 },
          round1_results: { type: "dict[str,dict]", size: 4 },
          round2_improvements: { type: "list[dict]", size: 3 },
          round2_results: { type: "dict[str,dict]", size: 3 },
          bleu_per_beam: { type: "dict[int,float]", size: 3 },
        },
        sub_calls: 1,
      },
    ],
    sub_calls: 1,
    timing: 7.6,
  },

  // Round 2 outcomes — NO promoted (c7=running, c8=marginal, c9=skipped)
  {
    event: "candidate_outcome",
    timestamp: ts(),
    iteration: 14,
    candidate_id: "c7",
    outcome: "running",
    rubric_delta: null,
  },
  {
    event: "candidate_outcome",
    timestamp: ts(),
    iteration: 14,
    candidate_id: "c8",
    outcome: "marginal",
    rubric_delta: 0.01,
  },
  // c9 — skipped (root decided mixed-precision is an efficiency improvement, not a
  //        quality one, and deferred it for a later run focused on throughput)
  {
    event: "candidate_outcome",
    timestamp: ts(),
    iteration: 14,
    candidate_id: "c9",
    outcome: "skipped",
    rubric_delta: null,
  },

  // ─── Final rubric score and run_complete ──────────────────────────────────

  {
    event: "rubric_score",
    timestamp: ts(),
    iteration: 14,
    score: 0.53,
    target: 0.70,
    areas: [
      { area: "architecture_correctness", score: 0.90, weight: 0.25, status: "pass" },
      { area: "training_procedure", score: 0.82, weight: 0.25, status: "pass" },
      { area: "bleu_score_ende", score: 0.42, weight: 0.30, status: "partial" },
      { area: "bleu_score_enfr", score: 0.10, weight: 0.20, status: "fail" },
    ],
  },
  {
    event: "run_complete",
    timestamp: ts(),
    status: "completed",
    iterations: 14,
    rubric_score: 0.53,
    cost_usd: 1.84,
    final_report_path: "runs/transformer-repro-001/final_report.md",
  },
];
