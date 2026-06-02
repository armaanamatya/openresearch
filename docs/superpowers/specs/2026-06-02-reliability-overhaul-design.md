# OpenResearch Reliability Overhaul — Infrastructure, Architecture & Research Survey

**Date:** 2026-06-02  
**Branch:** `feat/gepa-integration`  
**Status:** Research synthesis — implementation planning phase  
**Author:** Brainstormed from 6 live runs, 2 research sweeps (80+ papers), advisor review

---

## 1. Problem Statement

We have run the MixUp paper (arXiv:1710.09412) six times and have never received a passing rubric score, despite the fact that training itself succeeds. Run #4 produced:

- ERM test accuracy: **93.4%** (`test_err=0.0661`)
- MixUp test accuracy: **94.5%** (`test_err=0.0548`)
- Paper claim: MixUp beats ERM — `qualitative_pass=True` ✓

Yet the final report said `verdict=failed, overall_score=0.0`.

This is not a training problem. It is an **infrastructure-level result-capture and scoring problem**. Every failure in our run history has been at the boundary between "training succeeded" and "the rubric saw the result."

### Failure taxonomy (all 6 runs)

| Run | Root cause | Training reached? |
|-----|-----------|-------------------|
| #1 | `RUNPOD_CAPACITY_EXHAUSTED` × 3 — A6000 COMMUNITY pool empty, escalation maxed at 2 | Never |
| #2 | Backend process died mid sub-RLM (~04:00 UTC), orphaned | Never |
| #3 | Same project dir, wall-clock expiry, no experiment spawned | Never |
| #4 | Three sequential experiments (CIFAR-10 ERM + MixUp + CIFAR-100) totalling ~6000s exceeded 7200s wall-clock; CIFAR-100 timed out at epoch 17/200 | CIFAR-10 yes, CIFAR-100 no |
| #5 | `DYNAMIC_GPU_MAX_ESCALATIONS=2` exhausted after rtx4090→a5000→a6000; A6000 COMMUNITY also exhausted | Never |
| #6 | `DYNAMIC_GPU_MAX_ESCALATIONS=4`; training reached A6000; root passed `results={}` to `verify_against_rubric` | Training yes; score=0 (BUG-NEW-050) |

**The throughline:** the scoring system is binary and brittle. One empty dict, one overrun, one empty GPU pool and the run scores 0.0 despite correct training.

---

## 2. Known Bugs (current)

### BUG-NEW-046 — `paper_grounding_failed` false positives on dict-repr dataset strings
**Severity:** Low  
**Fixed in this branch:** `backend/agents/paper_grounding.py` — extract `name` field from dict-repr strings before grounding check.  
The root model sometimes returns `datasets` as a list of dict objects (`{"name": "CIFAR-10", ...}`). `PaperClaimMap` stringified these to `str(dict)` which the grounding validator correctly rejected as not found in paper text. Advisory only; run continued.

### BUG-NEW-047 — `compute_scope_invalid` — root concludes CPU-only from `detect_environment` output
**Severity:** Low  
**Fixed in this branch:** `backend/agents/environment_detective.py` — expanded `compatibility_notes` to explicitly state this is the LOCAL dev machine spec, not the RunPod execution environment.  
`detect_environment` always runs locally (CPU). The root read "Generated for pytorch==2.2.0 on CPU." and passed a CPU-only string for `compute_scope`.

### BUG-NEW-048 — GEPA produces 0 candidates; 60s timeout consumed by seed evaluation
**Severity:** Medium  
**Fixed in this branch:** `backend/config.py` — `gepa_timeout_plan_s` 60→180, `gepa_timeout_baseline_s` 30→90, `gepa_timeout_improve_s` 60→180.  
With `claude-oauth` reflection LM, seed evaluation takes ~2 min (subscription latency). Prior 60s timeout was exhausted before any proposals generated. GEPA silently fell back to seed prompt on every run.

### BUG-NEW-049 — `RUNPOD_TRANSIENT_500` escalates GPU tier same as `CAPACITY_EXHAUSTED`
**Severity:** Medium  
**Open:** The escalation handler in `primitives.py` treats API hiccups (transient 500) identically to genuine pool exhaustion (capacity exhausted). A momentary API flap on an available RTX 4090 COMMUNITY node escalates to A5000 ($0.34/hr → $0.36/hr), burning an escalation slot. Fix: add a same-tier retry with longer backoff before consuming an escalation slot when `TRANSIENT_500` is the only failure mode. Code: `primitives.py` ~line 3434.

### BUG-NEW-050 — Root passes `results={}` to `verify_against_rubric`; all leaves score 0.0
**Severity:** HIGH — directly caused false `verdict=failed` on Run #6  
**Partially fixed in this branch:** `verify_against_rubric` now falls back to the most recent successful `experiment_runs.jsonl` entry when `results` is empty. Cache key now includes `experiment_runs.jsonl` hash to prevent stale cached scores.  
**Root fix still needed:** the root model prompt must explicitly instruct the root to pass the full `run_experiment` return value, not a reconstructed or empty dict.

### BUG-NEW-051 — Multi-experiment `train.py` exceeds wall-clock budget
**Severity:** HIGH — caused Run #4 to complete training but score 0  
**Workaround:** `REPROLAB_BASELINE_EXTRA_GUIDANCE` explicitly limits to CIFAR-10 only.  
**Root fix:** see Proposal 2 (multi-fidelity scope planning) — estimate runtime before writing `train.py` and drop experiments that won't fit within `remaining_wall_clock_s * 0.7`.

---

## 3. Research Survey

### 3.1 Automated ML Paper Reproduction

#### PaperBench (OpenAI, April 2025) — arXiv:2504.01848
First rigorous benchmark for end-to-end paper reproduction: 20 ICML 2024 papers, 8,316 atomic grading tasks. Claude 3.5 Sonnet achieves 21.0%; human PhD baseline (48 hrs, best of 3) = 41.4%.

**Key insight:** PaperBench awards **partial credit per atomic leaf**. A run that reproduces 3 of 5 experiments scores ~60%, not 0%. This is the scoring model we are missing. Our system scores each run all-or-nothing.

**Application:** The anytime evidence architecture (Proposal 1) directly mirrors PaperBench's partial-credit model. Each rubric leaf that can be evidenced by a completed experiment should be scored independently as that experiment finishes.

---

#### PaperRepro: Tacit Knowledge Recovery (Microsoft/Stanford, 2025) — arXiv:2603.01801
Formalizes paper reproduction as recovering three types of tacit knowledge:
- **Relational:** Citation graph analysis to find reusable patterns from referenced papers
- **Somatic:** Execution-feedback refinement — runtime signals (stderr, missing deps, shape errors) guide debugging
- **Collective:** Graph-level induction across clusters of papers with similar implementations

Performance: 10.04% gap vs. official implementations; 24.68% improvement over strongest baseline on ReproduceBench.

**Application:** Our root model currently receives a flat `run_experiment` failure string. Structuring the error as typed categories (missing dependency, wrong shape, OOM, timeout) and feeding it back with somatic examples from prior runs would significantly reduce the "code was generated but ran wrong" failure class. See Proposal 4.

---

#### SciReplicate-Bench: Algorithmic Reproduction (2024) — arXiv:2504.00255
100 tasks from 36 NLP papers. Two-agent system: Paper Agent (algorithmic comprehension) + Code Agent (dependency resolution). Even the best LLM achieves only 39% execution accuracy.

**Key failure modes:**
- Algorithmic comprehension gaps (multi-step math, notation)
- Missing implementation context (undocumented hyperparameters)
- Dependency mismatch (wrong library versions)
- Validation mismatch (metrics not aligned to paper's definition)

**Application:** Our `understand_section` / `extract_hyperparameters` / `detect_environment` pipeline maps to the Paper Agent. Separating these concerns more explicitly — giving the Paper Agent a structured output schema that forces it to enumerate all unknowns before code generation — would reduce downstream surprises in `implement_baseline`.

---

#### AutoReproduce: Paper Lineage Algorithm (2025) — arXiv:2505.20662
Traces citation relationships to recover implicit implementation details. Papers omit standard techniques from prior work; the lineage algorithm finds those referenced papers and extracts the missing pieces. Uses multi-agent framework + sampling-based unit testing. Outperforms baselines on PaperBench and ReproduceBench.

**Application:** When `implement_baseline` hits unclear implementation details (e.g., "we use standard data augmentation"), the root should query the paper's references for the specific technique. Our current system relies on the root model's training data for these gaps — explicitly chaining to cited papers would close them more reliably.

---

#### Executable Knowledge Graphs (xKG, 2024) — arXiv:2510.17795
Hierarchical, multi-relational graph over arXiv papers + GitHub repos. Captures conceptual relations and runnable code snippets. Up to 10.9% improvement on agent replication tasks. Works best for papers that refine existing techniques; less effective for methodologically novel papers.

**Application:** A domain-specific xKG for common ML training patterns (ResNets, RL loops, GRPO training) cached locally would let `implement_baseline` retrieve verified boilerplate rather than regenerating it. Lower priority given our SDAR focus (novel paper), but valuable for common architectures.

---

#### Prompt-Free Collaborative Agents (2024) — arXiv:2512.02812
Two-agent loop without explicit user prompts:
- **Verification Agent:** Checks outputs against step-by-step requirements
- **Refinement Agent:** Revises outputs based on identified issues

Agents infer necessary actions from context alone. This is architecturally similar to what `verify_against_rubric` + `propose_improvements` already does, but more tightly coupled — the verification agent actively checks intermediate outputs before `run_experiment` fires.

**Application:** A pre-execution verification pass — "does this `train.py` look like it can produce the metrics the rubric asks for?" — would catch the most common failure modes (wrong output path, metrics.json never written, wrong metric name) before burning 2000s of GPU time.

---

### 3.2 Checkpointing and Failure Recovery

#### Just-In-Time Checkpointing (Microsoft/EuroSys 2024) — ACM SIGOPS 2024
Checkpoint on-demand at failure time, not periodically. Each GPU replays only one minibatch post-failure. Near-zero steady-state overhead; reduces recovery time from minutes to seconds.

**Application:** RunPod emits no pre-timeout signal on COMMUNITY pods, but the `--max-wall-clock` watchdog in our system does know when time is about to expire. At `remaining_s() ≤ 300` (5 min before hard deadline), `run_experiment` should: (1) signal the training subprocess to write a checkpoint immediately, (2) write partial metrics to `experiment_runs.jsonl`, (3) call `verify_against_rubric` with whatever is available. This converts a "timeout → score 0" into "timeout → partial score."

---

#### Universal Checkpointing (UniChkpt, USENIX ATC 2025) — arXiv:2406.18820
Decouples checkpoint format from parallelism strategy. Checkpoints saved under one GPU configuration can be loaded under a different one. Critical for elastic capacity recovery: if RunPod A6000 capacity is exhausted mid-training, the job can resume on L40S without full restart.

**Application:** When GPU escalation occurs mid-training (not just pre-training), UniChkpt-style elastic resume would preserve training progress. Currently, escalation restarts the experiment from scratch. With UniChkpt patterns, the generated `train.py` should always include epoch-level checkpointing + a `--resume` flag. The implementer guidance should explicitly require this.

---

#### FFTrainer: Fast Failover (2024) — arXiv:2512.03644
98% recovery time reduction using surplus network capacity for state transfer instead of disk I/O. 68% GPU utilization recovery post-failure.

**Application:** For SDAR's Qwen models (multi-GB weights), checkpoint save/load latency is a real concern. FFTrainer's network-based state transfer is directly applicable if we ever run multi-GPU or multi-pod experiments. Lower priority for single-experiment CIFAR/MixUp scale.

---

### 3.3 Multi-Fidelity and Adaptive Experimentation

#### HyperBand / ASHA (2018-2020)
Multi-fidelity HPO: run cheap short trials first, keep top performers, kill the rest. ASHA (Asynchronous Successive Halving) extends this to asynchronous settings.

**Application to scope planning:** Before generating the full `train.py`, run a 5-epoch probe (60-120s on GPU) to:
1. Verify the GPU environment is correctly set up (CUDA accessible, deps installed)
2. Estimate per-epoch training time and extrapolate to full-run duration
3. If estimated total > `remaining_s * 0.7`, drop the lowest-value experiments from the scope

This "probe before committing" pattern directly prevents BUG-NEW-051 (wall-clock overrun from multi-experiment scope). See Proposal 2.

---

#### Ax: Adaptive Experimentation Platform (Meta, 2025) — AutoML 2025
Bayesian optimization for iterative exploration. Expressive API for multi-objective optimization, constraints, noisy observations. Outperforms Optuna, SMAC3, HEBO, SyneTune.

**Application:** For multi-paper reproduction campaigns (e.g., leaderboard runs across N papers), Ax could adaptively allocate GPU compute — spending more on papers that show early convergence toward rubric targets and less on clear failures. Lower priority than single-run reliability fixes.

---

### 3.4 Agentic Reliability Patterns

#### LLM Agent Watchdogs / Process Supervision
Our `SUB_RLM_STALL read-idle 120s` mechanism is a watchdog — it kills stalled sub-RLM children and returns a sentinel. This is the right pattern. Relevant production practices:

- **Circuit breakers for LLM API calls:** If 3 consecutive sub-RLM calls fail within 5 minutes, temporarily stop spawning new ones and report degraded mode to the root. Prevents exponential retry burns.
- **Heartbeat checks over long-running experiments:** `run_experiment` on SDAR takes 45+ minutes. A periodic heartbeat ping (every 60s) to the RunPod pod that records last-seen timestamp would detect silent hangs before the wall-clock hard limit.
- **Process group tracking:** When `run_experiment` spawns a RunPod job, tracking the job ID in a persistent file allows recovery if the local process crashes (resume from known job ID rather than spawning a new one).

---

#### SWE-EVO: Long-Horizon Software Evolution (Dec 2024) — arXiv:2512.18470
Captures iterative codebase changes. Agents like OpenHands, SWE-agent are capped at 100 iterations/calls per task. Key finding: iteration budget is the primary limiting factor, not model capability.

**Application:** Our root model has no explicit iteration budget display. The root should be told "you have used N of M iterations" as part of `check_user_messages()` output. This lets it plan whether to attempt CIFAR-100 given remaining budget.

---

#### FeedbackEval: Structured Error Feedback (2024) — arXiv:2504.06939
Structured reasoning + dynamic example selection improve iterative code repair. LLMs struggle with *diverse* feedback comprehension — they do better when feedback is categorized (missing file, wrong import, shape mismatch, OOM) rather than raw stderr.

**Application:** When `run_experiment` fails, our error feedback to the root should categorize the failure before passing it back. The `failure_class` field already does this at a coarse level (e.g., `dockerfile_invalid`, `contract_violation`). Adding sub-categories for Python exceptions (ImportError, CUDA OOM, AssertionError from rubric guard, FileNotFoundError for metrics.json) and including the first 5 lines of each error class as a structured block rather than raw 2000-char truncated stderr would significantly improve repair quality.

---

### 3.5 LLM Code Repair Loops

#### SWE-Bench and OpenHands (2024-2025)
2,294 real-world GitHub issue repair tasks. LLM agents generate fixes without explicit test cases. Best-performing systems (Claude 3.5 Sonnet in OpenHands) achieve ~50% resolution rate.

Key pattern: **edit-execute-observe loops.** Each iteration: (1) read error, (2) make minimal targeted edit, (3) re-run, (4) observe new output. The loop terminates when the error disappears or a new class of error emerges.

**Application:** Our `propose_improvements` primitive already implements this pattern at the experiment level. The gap is that it receives coarse feedback ("experiment failed") rather than line-level diagnostics. Returning the specific assertion that failed in `assert_metrics_schema` (rubric guard) rather than just "metrics schema invalid" would enable surgical repairs.

---

#### TextGrad: Automatic Differentiation for Text (2024)
LLM generates natural language "textual gradients" on outputs. Improved GSM8K: 72.9% → 81.1%; LeetCode: +20% solve rate.

**Application:** We already have GEPA for prompt optimization. TextGrad is complementary — where GEPA does population-based search over prompt candidates, TextGrad does gradient-based refinement of a single candidate. Could be used to refine the `implement_baseline` system prompt based on specific failure patterns from `experiment_runs.jsonl` (e.g., "the generated `train.py` never writes `metrics.json` at the expected path"). Lower priority given GEPA coverage.

---

### 3.6 GPU Scheduling and Elastic Compute

#### Agent.xpu: Heterogeneous SoC Scheduling (2025) — arXiv:2506.24045
Fine-grained preemption and replica splicing for LLM workloads on NPU/GPU heterogeneous SoCs. 4.6× lower latency for reactive tasks; 1.6-6.8× higher throughput for proactive tasks.

**Application to RunPod:** The prefill-decode disaggregation concept maps to our system: preprocessing (rubric generation, ingestion, plan_reproduction) is CPU-bound "prefill" work; GPU training is "decode" work. Explicitly scheduling these on separate resources — running rubric generation on local CPU while the RunPod pod is already warming — reduces the serial gap between "run starts" and "first experiment fires."

---

#### MQFQ-Sticky: Fair Queueing for Serverless GPU Functions (2024) — arXiv:2507.08954
Fair queue discipline for serverless GPU allocation — prevents long-running jobs from starving short ones.

**Application:** When running multiple paper reproduction jobs against the same RunPod account, fair queueing prevents a slow SDAR run (45+ min) from blocking a fast CIFAR run (34 min). Relevant when the leaderboard sprint involves concurrent runs.

---

### 3.7 Prompt Optimization

#### OPRO: Optimization by PROmpting (2023) — arXiv:2309.03409
Black-box prompt optimization: LLM generates and iteratively refines instruction candidates. Outperforms human prompts by up to 8% on GSM8K, 50% on Big-Bench Hard.

**Application:** GEPA (our existing system) is a genetic-Pareto variant of this pattern. The BUG-NEW-048 fix (raising GEPA timeouts to 180/90/180s) should allow GEPA to actually generate candidates on runs that currently produce 0 proposals due to the 60s timeout. OPRO is the closest published reference for GEPA's approach.

---

#### DSPy: Demonstration-based Prompt Optimization (2024)
Compiles high-level programs into optimized prompts using few-shot demonstrations. Particularly strong when training examples are available.

**Application:** Our `gepa_examples.jsonl` file (persisted per run) is the demonstration set for GEPA. DSPy's pattern of using structured examples with explicit inputs/outputs maps well to our `(prompt_candidate, rubric_score)` pairs. If GEPA's population-based approach proves insufficient, DSPy-style compilation could provide a complementary signal.

---

### 3.8 Speculative Execution Patterns

#### Speculative Decoding for LLM Inference (SpecPipe, MagicDec, FlowSpec — 2024)
Draft model generates candidate tokens in parallel; verifier accepts/rejects. For sequences beyond critical length, KV cache becomes memory bottleneck.

**Application to our system (limited):** These are inference-time speedups for the LLM serving layer. Not directly applicable to RunPod GPU scheduling or result capture. However, the *conceptual pattern* — generate candidates speculatively while verifying prior candidates in parallel — maps to Proposal 2 (parallel experiment execution: start CIFAR-100 speculatively while CIFAR-10 is verifying).

---

### 3.9 Partial Credit and Incremental Scoring

#### MLE-bench (OpenAI, 2024)
ML engineering agent benchmark. Awards partial scores on a per-task basis. Unlike pass/fail benchmarks, MLE-bench scores proportionally to how close the agent gets to the target metric.

**Key data point:** The top agent on MLE-bench achieves ~23% of available points. This is only possible because of partial credit — an all-or-nothing scorer would show 0% for all but the easiest tasks.

**Application:** Our rubric already has leaf-level structure. The gap is that `verify_against_rubric` only scores meaningfully when called with complete results at `FINAL_VAR` time. Scoring incrementally — one leaf cluster per completed experiment — converts our current binary outcome into a partial-credit outcome aligned with how PaperBench and MLE-bench operate.

---

## 4. Proposed Architecture Changes

### Proposal 1: Anytime Evidence Architecture *(Priority 1 — fixes the core gap)*

**Problem it solves:** BUG-NEW-050 (empty results), BUG-NEW-051 (wall-clock overrun), and the general "training succeeded but score = 0.0" class of failures.

**Core idea:** Decouple result capture from `FINAL_VAR`. Every call to `run_experiment` that succeeds writes a scoreable artifact immediately. The rubric is scored incrementally as evidence arrives. `final_report.json` is a live snapshot of best-evidence-so-far, written after every successful `run_experiment`. Wall-clock timeout yields the best partial score achieved, never 0.0.

**Mechanism:**

```
run_experiment completes (success=True)
  → write entry to experiment_runs.jsonl  [already happens]
  → call _score_incremental_rubric(entry.metrics, rubric)  [new]
  → write best-score-so-far to final_report.json  [new]
  → emit SSE event rubric_score_incremental {score, leaf_count}  [new]

wall-clock watchdog fires
  → check final_report.json for best score
  → if score > 0: emit run_complete with that score
  → never emit score=0.0 when any experiment succeeded
```

**What changes:**
1. `primitives.py::run_experiment` — after `success=True`, call `_try_incremental_score(ctx)` which reads `experiment_runs.jsonl`, runs a lightweight rubric check, and updates `final_report.json` atomically.
2. `run.py::_watchdog` — at timeout, emit the score from `final_report.json` (which may be partial) rather than `score=0.0`.
3. `sse_bridge.py` — add `rubric_score_incremental` event type to the schema.

**Research backing:** PaperBench partial-credit model; MLE-bench proportional scoring; JIT checkpointing "emit best state on failure signal" principle.

**Tradeoff:** Medium implementation complexity. Requires rubric to be structurally decomposable by experiment (it already is — generated rubric has leaf-level structure). Does not help if *no* experiments complete; that case needs Proposal 2.

---

### Proposal 2: Multi-Fidelity Scope Planning *(Priority 2 — prevents wall-clock overruns)*

**Problem it solves:** BUG-NEW-051 (multi-experiment wall-clock overrun).

**Core idea:** Before writing the full `train.py`, write and run a 5-epoch probe (60-120s GPU time) that:
1. Verifies the GPU environment works (CUDA accessible, deps installed, data downloadable)
2. Measures per-epoch wall time
3. Extrapolates to full-run duration per experiment
4. Drops experiments whose estimated total duration would exceed `remaining_wall_clock_s * 0.70`

The probe is a new primitive or a capability added to `run_experiment` (`mode="probe"`). The root is taught to call it after `implement_baseline` generates the initial `train.py`.

**Mechanism:**

```
implement_baseline → generates train.py with all experiments
run_experiment(mode="probe", epochs=5)
  → returns {per_epoch_s: float, env_ok: bool, estimated_full_duration_s: dict[experiment_name, float]}
root plans: if sum(estimated_full_duration) > remaining_s * 0.7:
  → drop lowest-value experiments (instructed by system prompt)
  → call implement_baseline again with reduced scope
run_experiment(mode="full") on remaining experiments
```

**Research backing:** HyperBand (multi-fidelity: cheap probes before expensive runs); Ax adaptive experiments (allocate budget proportional to expected value); ASHA (asynchronous early stopping).

**Tradeoff:** Adds ~60-120s probe overhead per run. Prevents 2000s wasted GPU time on experiments that won't finish. The probe also doubles as an environment health check — catches missing deps, wrong CUDA version, data download failures before the full training run.

---

### Proposal 3: GPU Pre-warming + Capacity Pre-flight *(Priority 3 — eliminates cold-start and capacity surprise)*

**Problem it solves:** Runs #1, #5 — GPU capacity exhausted only discovered at `run_experiment` time, after 30+ minutes of planning work.

**Core idea:**
1. **Capacity pre-flight:** At `build_environment` time (not `run_experiment`), probe RunPod API for available capacity on the target GPU tier. If capacity is 0, immediately escalate the GPU plan and emit a `run_warning`.
2. **Pod pre-warming:** Spin up a minimal "warm standby" pod at `build_environment` time with a keep-alive. By the time `run_experiment` fires (15-30 min later), the pod is already booted and deps are installing. Cold-start penalty (~3-5 min) disappears.

**Mechanism:**

```
build_environment:
  → check RunPod capacity for gpu_plan.short_name via RunPod API
  → if capacity == 0: escalate now, emit run_warning gpu_capacity_preempted
  → else: spin up warm_standby pod with task="sleep 1800"
  → store warm_pod_id in gpu_plan.json

run_experiment:
  → if warm_pod_id exists and pod is still alive: reuse it (send training job)
  → else: cold-start as before
```

**RunPod API support:** RunPod serverless supports job pre-queuing (job enters IN_QUEUE state; pod spins up when capacity becomes available). The gap is that currently we don't pre-queue — we wait until `run_experiment` fires. Pre-queuing at `build_environment` time shifts the capacity wait to overlap with the LLM orchestration time.

**Tradeoff:** Pre-warming costs ~$0.01-0.05 per run (15-min warm standby on RTX 4090 = 15/60 × $0.34 = $0.085). Eliminates the "capacity exhausted surprise" and the 3-5 min cold start. The warm pod is killed if it's not used within 30 min.

---

### Proposal 4: Structured Error Feedback + Tacit Knowledge Recovery *(Priority 4 — improves code quality)*

**Problem it solves:** The "code was wrong" class of failures — `contract_violation`, `dockerfile_invalid`, `metrics not found`.

**Core idea:** When `run_experiment` fails, the error returned to the root is currently a flat string. Structure it:

```python
ExperimentFailureFeedback(
    failure_class="metrics_not_found",          # typed, not free-text
    location="code/outputs/{run_id}/metrics.json",  # exact path
    hint="train.py wrote metrics to code/metrics.json — move write to code/outputs/{run_id}/",
    somatic_example={                           # from experiment_runs history
        "paper": "1710.09412",
        "fixed_in_iteration": 3,
        "fix": "changed outdir to ctx.output_dir"
    }
)
```

The root receives a structured dict, not a truncated stderr blob. It can act on each field independently.

**Tacit knowledge recovery (from AutoReproduce):** When `understand_section` or `extract_hyperparameters` returns an "unknown" or leaves a field blank, the root should query the paper's top-3 cited papers for that field. A new `query_citations(field, paper_refs)` tool that fetches and parses the referenced papers for a specific concept would close the "the paper says `standard data augmentation` but doesn't define it" gap.

**Research backing:** PaperRepro somatic recovery; FeedbackEval structured categories; AutoReproduce paper lineage.

**Tradeoff:** Higher implementation complexity. Yields the largest quality improvement *after* the infra fixes (Proposals 1-3) are in place.

---

### Proposal 5: Parallel Experiment Execution *(Priority 5 — for cost-tolerant runs)*

**Problem it solves:** Wall-clock is the sum of sequential experiments. Proposal 2 (probe) prevents overruns; Proposal 5 eliminates the sum entirely.

**Core idea:** Spawn each experiment on its own RunPod pod in parallel. Wall-clock = max(t₁, t₂, ...), not sum. Cancel pods whose experiments finish last or whose results aren't needed (early exit when rubric already at target score).

**Mechanism:**
```
implement_baseline → generates train.py for N experiments
for each experiment e_i:
    pod_i = run_experiment.async(e_i)   # fires in parallel
wait_any(pods)  # as each finishes, score its rubric leaves
if rubric_score >= target: cancel remaining pods
```

**Research backing:** Speculative execution (SpecPipe conceptual pattern); Ax parallelism; SWE-EVO parallel agent dispatch.

**Tradeoff:** 2-N× GPU cost. Only viable with COMMUNITY pods ($0.34/hr × 2 = $0.68 vs sequential $0.34). For SDAR (45 min × 3 models × 3 environments = 6h sequential), parallelism is transformative: 6h → 45 min at 9× cost. For MixUp (2 experiments × 34 min = 68 min sequential), parallelism: 34 min at 2× cost. The cost tradeoff is favorable for papers with expensive experiment matrices.

---

## 5. Implementation Roadmap

Priority order is defined by "most likely to convert a 0.0 score into a non-zero score with minimal new code."

### Phase 1: Zero-to-Score (target: first passing run)

| Item | Code location | Effort | Impact |
|------|--------------|--------|--------|
| Fix BUG-NEW-050 root cause: system prompt explicitly requires passing full `run_experiment` output to `verify_against_rubric` | `backend/agents/rlm/system_prompt.py` | 1 day | HIGH |
| Anytime incremental scoring: `_try_incremental_score` after each `run_experiment(success=True)` | `backend/agents/rlm/primitives.py` | 2 days | HIGH |
| Probe mode for `run_experiment`: 5-epoch env-check + duration extrapolation | `backend/agents/rlm/primitives.py` + root system prompt | 2 days | HIGH |
| Root told to call probe + plan scope before full training | `backend/agents/rlm/system_prompt.py` | 1 day | HIGH |
| Fix BUG-NEW-049: same-tier retry on TRANSIENT_500 before escalating | `backend/agents/rlm/primitives.py` ~line 3434 | 0.5 days | Medium |
| Watchdog emits partial score from `final_report.json` on timeout | `backend/agents/rlm/run.py` | 1 day | Medium |

### Phase 2: Score Improvement (target: rubric > 50%)

| Item | Code location | Effort | Impact |
|------|--------------|--------|--------|
| Structured error feedback on `run_experiment` failure | `backend/agents/rlm/primitives.py` + run context | 2 days | High |
| GPU capacity pre-flight at `build_environment` time | `backend/agents/rlm/primitives.py` + RunPod client | 2 days | Medium |
| Pod pre-warming via pre-queued RunPod job | `backend/services/runtime/runpod_backend.py` | 3 days | Medium |
| `query_citations` tool for tacit knowledge recovery | `backend/agents/rlm/primitives.py` (new primitive) | 3 days | Medium |

### Phase 3: Scale and Quality (target: leaderboard competitive)

| Item | Code location | Effort | Impact |
|------|--------------|--------|--------|
| Parallel experiment execution (async pod per experiment) | `backend/agents/rlm/primitives.py` + SDK | 4 days | High at scale |
| Epoch-level checkpointing + UniChkpt-style elastic resume | Generated `train.py` + implementer guidance | 2 days | Medium |
| Paper lineage citation graph for tacit knowledge | New service `backend/services/knowledge/` | 5 days | Medium |
| xKG domain cache for common architectures | New service | 5 days | Low-Medium |

---

## 6. Success Criteria

A run is passing when:
1. `final_report.json::overall_score > 0.0` on the first attempt (no restarts needed)
2. The rubric score reflects the actual training outcome — if CIFAR-10 MixUp beats ERM, the relevant leaves score `True`
3. Wall-clock timeout does not produce `score=0.0` — it produces "best partial score achieved"
4. GPU capacity exhaustion does not abort the run — it escalates gracefully with pre-flight awareness

The minimum viable fix to reach criterion 1 and 2: **BUG-NEW-050 root cause fix** (system prompt) + **BUG-NEW-048 fix** (already committed: GEPA timeouts raised). These two changes alone should convert Run #7 from `score=0.0` to `score>0`.

---

## 6b. Additional Research Findings (Sweep 2 + 3)

### 6b.1 Anytime Inference — Conditional Monotonicity

**"Towards Anytime Classification in Early-Exit Architectures by Enforcing Conditional Monotonicity"** (NeurIPS 2023) — arXiv:2306.02652

The critical problem in existing anytime systems: early-exit networks can produce *worse* predictions at deeper exits than at shallower ones. This paper solves it with a **Product-of-Experts post-hoc modification** that enforces monotonic confidence across exit points.

**Application to our incremental rubric scorer:** When scoring rubric leaves incrementally as experiments complete, we must ensure that a run's score never *decreases* as more evidence arrives — each additional successful experiment can only add evidence, never invalidate prior leaves. The monotonicity constraint maps directly: implement `_try_incremental_score` so it only ever updates `final_report.json` with a score `≥ prior_score`.

---

### 6b.2 Streaming Evaluation — Prequential Pattern

**StreamBench** — Evaluates agents with continuous model updates as new data arrives. **Prequential evaluation** (test-then-train per sample) prioritizes recent predictions.

**Application:** After each `run_experiment(success=True)`, the incremental scorer evaluates only the rubric leaves that the just-completed experiment can evidence. It does not re-evaluate previously scored leaves. This is the prequential pattern — score each sub-task once, at the moment its evidence arrives, rather than re-grading the full rubric each time.

---

### 6b.3 LLM Agent Reliability — Meltdown Behavior and Circuit Breakers

**"Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents"** (arXiv:2603.29231)

Identifies **meltdown behavior**: catastrophic cascades where agents transition from incorrect-but-coherent to incoherent looping as task duration increases. Consistent with our `SUB_RLM_STALL read-idle 120s` observations.

**Circuit breaker pattern** (3-state):
- **Closed:** Count failures in sliding window (e.g., 10-request window)
- **Open:** Fast-fail with error sentinel; cooldown timer
- **Half-Open:** Graduated probing (1 test call before fully reopening)
- Trip condition: error rate > 50% over 10 requests OR P99 latency > 3× baseline

**Application:** Wrap `rlm_query` / sub-RLM spawning with a circuit breaker. If 3 consecutive sub-RLM calls stall within 10 minutes, enter Open state and return a degraded-mode sentinel to the root ("rlm_query_circuit_open — use primitive tools directly"). This prevents the exponential retry burns we see when `claude-oauth` subscription hits rate limits.

---

### 6b.4 AutoReproduce — Implementation Details and Performance

**AutoReproduce** (arXiv:2505.20662) achieves **89.74% of executable runs** vs. official implementations — dramatically above baselines. Three-stage pipeline:
1. Literature review (parse paper + identify knowledge gaps)
2. **Paper lineage mining** (traverse citation graph to fill gaps)
3. Code development

Open-source: https://github.com/AI9Stars/AutoReproduce

**Concrete gaps it closes:** "We use standard data augmentation" → query the cited papers → find the exact transforms used. "We follow the training protocol of [X]" → fetch paper X → extract batch size, LR schedule, warmup epochs.

**Application:** Our `understand_section` + `extract_hyperparameters` pipeline should add a third step: for any field that returns null/unknown, call `query_citations(field, paper.references)` to search the top-5 cited papers for that field. This is a new primitive or a post-processing step in `extract_hyperparameters`.

---

### 6b.5 SWE-Agent ACI — Structured Output as the Game-Changer

**SWE-Agent** (NeurIPS 2024, arXiv:2405.15793): Agent-Computer Interfaces with fixed, uniform output formats achieve **3–5× improvement** vs. non-interactive LMs; **12.5% on SWE-bench, 87.7% on HumanEvalFix**.

The critical insight: **structured output templates remove ambiguity**. The ACI defines exact schemas for search, view, edit, and context operations. Agents succeed not because they are smarter, but because they receive and emit well-structured information.

**Application — structured experiment failure feedback:**
```json
{
  "failure_class": "metrics_not_found",
  "exit_code": 1,
  "stderr_first_500": "FileNotFoundError: code/metrics.json",
  "expected_path": "code/outputs/{run_id}/metrics.json",
  "stdout_last_10_lines": ["Epoch 100/100: acc=0.9452...", "Saving checkpoint..."],
  "gpu_memory_peak_gb": 6.2,
  "wall_time_s": 2047,
  "suggestion": "Change metrics output path from code/ to code/outputs/{run_id}/"
}
```

Compare to current: a 2000-char truncated stderr blob. The structured version lets the root address `failure_class` and `suggestion` directly, reducing repair iterations from 3+ to 1.

---

### 6b.6 LLMParser — Automated ML Log Parsing at 96% Accuracy

**LLMParser** (ICSE 2024, arXiv:2404.18001): Few-shot LLM-based log parsing extracts structured templates from free-form logs. Achieves **96% accuracy** (vs. Drain/Logram at 85-90%). Tested on Flan-T5, LLaMA-7B, ChatGLM-6B.

**PARSE Schema Optimization** (arXiv:2510.08623): Two-stage: ARCHITECT (iteratively refine extraction schema from failure analysis) + SCOPE (three-stage validation: missing attrs, value grounding, rule compliance). **64.7% accuracy improvement, 92% error reduction** within first retry.

**Application:** `run_experiment` currently captures `exec.log` as a free-form string and passes a truncated excerpt to the root. Instead, pipe `exec.log` through an LLMParser-style extractor that produces:
```json
{"epoch": 100, "train_loss": 0.124, "val_acc": 0.9452, "lr": 1e-4, "gpu_mem_gb": 6.2, "duration_s": 2047}
```
The root and `verify_against_rubric` receive structured metrics instead of log text. This makes `verify_against_rubric` rubric-grading more reliable (grader sees structured data, not stdout noise) and enables early stopping based on real-time metric signals.

---

### 6b.7 BOHB — Multi-Fidelity Implementation Details

**BOHB** (Robust and Efficient Hyperparameter Optimization, OpenReview) combines HyperBand's successive halving with Bayesian Optimization's TPE sampling. Key parameters:
- `min_budget` / `max_budget`: cheapest vs. most expensive fidelity level (e.g., 1 epoch vs. 200 epochs)
- `eta`: halving factor (typically 3 — each round keeps top 1/3)
- `n_workers`: parallel evaluations

**A-BOHB** (Asynchronous BOHB): uses Gaussian process with fidelity dimension, enables fully async evaluation without synchronization barriers.

**BOAH Tool Suite** (arXiv:1908.06756): Python implementation of BOHB + analysis tools. Open-source.

**Application:** For the multi-fidelity probe in Proposal 2, BOHB provides the extrapolation math: given 5-epoch loss curve [L₁, L₂, L₃, L₄, L₅], fit a power law L(t) = a·t^(-b) and predict L(200). If predicted L(200) > baseline + ε, this experiment will fail the rubric — skip it. The 5-epoch probe cost is ~60s on RTX4090; it prevents a 2000s wasted run.

---

### 6b.8 AdaRubric — Task-Adaptive Per-Dimension Partial Credit

**AdaRubric** (arXiv:2603.21362): Dynamically generates task-specific evaluation dimensions rather than applying fixed criteria. Uses a "DimensionAwareFilter" to prevent quality masking — a high score on one dimension doesn't hide a low score on another. Achieved **6.8–8.5% task success improvements** on downstream models (WebArena, ToolBench, AgentBench).

**Application:** Our `verify_against_rubric` uses a fixed rubric generated at run start. AdaRubric suggests that for each experiment type (training, evaluation, comparison), dynamically generate the relevant evaluation dimensions at scoring time. For the MixUp paper, the dimensions are: (1) correct MixUp formula, (2) correct Beta(1,1) sampling, (3) ERM baseline trained, (4) MixUp beats ERM on test error, (5) reported numbers match paper within tolerance. Each dimension scored independently → 4/5 correct gives 80%, not 0%.

---

### 6b.9 Cost-Aware Orchestration — BOA Constrictor Pattern

**BOA Constrictor** (arXiv:2602.01404): Budget-constrained GPU scheduling that minimizes job completion time subject to cost constraints. Uses a **budget-optimal allocation** algorithm that dynamically resizes GPU assignments as new job information arrives.

**Application to our escalation ladder:** Instead of a fixed `DYNAMIC_GPU_MAX_ESCALATIONS` ceiling, implement a cost-aware escalation policy:
- Track `cumulative_run_gpu_usd` (already in `RunBudget`)
- On escalation request, compute: `new_gpu_cost_per_s * estimated_remaining_s`
- If projected total exceeds `REPROLAB_MAX_RUN_GPU_USD`, deny escalation
- If cost budget allows, escalate even beyond the current `max_escalations` ceiling

This converts a hard escalation count limit into a soft cost limit — semantically correct.

---

## 7. Appendix: Paper References

| Area | Paper | arXiv / URL |
|------|-------|-------------|
| Paper reproduction benchmark | PaperBench: Evaluating AI's Ability to Replicate AI Research (OpenAI 2025) | arXiv:2504.01848 |
| Paper reproduction benchmark | MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering (OpenAI 2024) | arXiv:2410.07095 |
| Tacit knowledge recovery | PaperRepro: Recovering Tacit Knowledge for Automated Paper Reproduction (2025) | arXiv:2603.01801 |
| Algorithmic reproduction | SciReplicate-Bench (2024) | arXiv:2504.00255 |
| Paper lineage | AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage (2025) | arXiv:2505.20662 |
| Knowledge graphs | Executable Knowledge Graphs for AI Research Replication (2024) | arXiv:2510.17795 |
| Collaborative agents | Prompt-Free Collaborative Agents for Paper Reproduction (2024) | arXiv:2512.02812 |
| JIT checkpointing | Just-In-Time Checkpointing (Microsoft/EuroSys 2024) | ACM SIGOPS 2024 |
| Universal checkpointing | Universal Checkpointing (USENIX ATC 2025) | arXiv:2406.18820 |
| Fast failover | FFTrainer: Fast Failover for LLM Training (2024) | arXiv:2512.03644 |
| Multi-fidelity HPO | HyperBand: A Novel Bandit-Based Approach to Hyperparameter Optimization (2018) | JMLR 2018 |
| Adaptive experiments | Ax: A Platform for Adaptive Experimentation (Meta 2025) | AutoML 2025 |
| GPU scheduling | Agent.xpu: Efficient Scheduling of Agentic LLM Workloads (2025) | arXiv:2506.24045 |
| Serverless GPU fairness | MQFQ-Sticky: Fair Queueing for Serverless GPU Functions (2024) | arXiv:2507.08954 |
| LLM code repair | SWE-EVO: Long-Horizon Software Evolution Benchmark (2024) | arXiv:2512.18470 |
| Structured feedback | FeedbackEval: Feedback-Driven Code Repair Benchmark (2024) | arXiv:2504.06939 |
| Prompt optimization | OPRO: Large Language Models as Optimizers (2023) | arXiv:2309.03409 |
| Text gradients | TextGrad: Automatic Differentiation via Text (2024) | — |
| Speculative decoding | SpecPipe: Pipeline Parallelism + Speculative Decoding (2024) | arXiv:2504.04104 |
| Speculative decoding | MagicDec: Breaking Latency-Throughput Tradeoff (2024) | arXiv:2408.11049 |
| Anytime inference | Anytime Classification via Conditional Monotonicity (NeurIPS 2023) | arXiv:2306.02652 |
| Streaming evaluation | StreamBench: Benchmarking Continuous Agent Improvement | arXiv:2406.08747 |
| Agent reliability | ReliabilityBench: LLM Agent Reliability Under Production Stress (2025) | arXiv:2601.06112 |
| Agent reliability | Beyond pass@1: Reliability Science for Long-Horizon Agents (2025) | arXiv:2603.29231 |
| Circuit breakers | Weak-to-Strong Monitoring of LLM Agents (2025) | arXiv:2508.19461 |
| Structured error feedback | SWE-Agent: Agent-Computer Interfaces (NeurIPS 2024) | arXiv:2405.15793 |
| Structured error feedback | SWE-Adept: Deep Codebase Analysis with LLM Agents (2025) | arXiv:2603.01327 |
| Log parsing | LLMParser: Exploratory Study on LLMs for Log Parsing (ICSE 2024) | arXiv:2404.18001 |
| Log parsing | PARSE: Schema Optimization for Reliable Entity Extraction (2024) | arXiv:2510.08623 |
| Partial credit rubrics | AdaRubric: Task-Adaptive Rubrics for LLM Agent Evaluation (2024) | arXiv:2603.21362 |
| Multi-fidelity HPO | BOHB: Robust and Efficient Hyperparameter Optimization | OpenReview |
| Multi-fidelity HPO | BOAH: Multi-Fidelity BO + Analysis Tool Suite (2019) | arXiv:1908.06756 |
| Cost-aware scheduling | BOA Constrictor: Budget-Optimal GPU Allocation (2024) | arXiv:2602.01404 |
| Speculative experiments | PEARL: Parallel Speculative Decoding with Adaptive Draft (ICLR 2025) | arXiv:2408.11850 |
| Partial recovery | CPR: Understanding DNN Training Partial Recovery (2020) | arXiv:2011.02999 |
