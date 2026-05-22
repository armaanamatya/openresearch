# Run Log

Every paper-reproduction run and its outcome. **Leaf score** = the authoritative
PaperBench rubric score (flatten tree → batched LLM grading → weighted roll-up).

## Session 2026-05-22 — debug-and-harden

| # | Paper | Source | Verdict | Leaf score | Leaves | Run dir |
|---|-------|--------|---------|-----------:|-------:|---------|
| 1 | Sequential Neural Score Estimation | PaperBench bundle | partial | **0.366** | 92/92 | `pb_sequential-neural-score-estimation_1779449793` |
| 2 | Mechanistic Understanding (DPO/toxicity) — pre-fix | PaperBench bundle | failed | **0.055** | 96/96 | `pb_mechanistic-understanding_1779453277` |
| 2b | Mechanistic Understanding — re-run, run_experiment fix | PaperBench bundle | failed | **0.079** | 96/96 | `pb_mechanistic-understanding_1779457326` |
| 3 | GoRL — diffusion+evolutionary policies for online RL | arXiv 2512.02581 | failed | **0.000** | — | `prj_11d7a6f3bb7e2ab7` |

**Run 1.** Core SNPSE method reproduced well (core-method 0.8, eval-metrics 1.0).
`baseline_metrics` populated with measured C2ST — Two Moons 0.97, Gaussian Linear
0.997 (I1 verified live: experiment wrote `metrics.json`, `_execute_in_sandbox`
read it back). Score capped by **breadth** — 2/9 benchmark tasks, no baseline
methods. `plan_reproduction` fail-softed (the `evaluation_plan` list/str bug,
since fixed), so the run had no reproduction contract scoping the full job.
All four #62 DC#4 artifacts present (`final_report.{json,md}`,
`repl_state.pickle` @ iter 12 / 54 vars, `iterations/` × 12).

**Run 2 (pre-fix).** `run_experiment` failed in 6 s — root-caused to two
compounding bugs: `_execute_in_sandbox` logged stdout only (the
`ModuleNotFoundError` traceback was on stderr → `logs=""`, undiagnosable, and
the repair loop got an empty `repair_context`), and the experiment ran an image
`detect_environment` built before any code existed (missing `transformers`).
A third issue surfaced under the fix: the sandbox ran `network_disabled` so HF
model downloads failed. Honest 0.0 — the pipeline completed and reported the
failure. Fixed (Fix A/B/C in `primitives.py`); the re-run (2b) carries the fix.

**Run 2b (post-fix).** `run_experiment` SUCCEEDED — the live verification of
Fix A/B/C: the image rebuilt with the right deps (Fix B), `train.py` reached
the network for the gpt2 download (Fix C), and the full experiment log —
stdout and stderr — was captured (Fix A). Real `baseline_metrics` (DPO loss,
SVD subspace preservation), 3 improvement paths, 16 iterations, all DC#4
artifacts. The leaf score is low (0.079, verdict `failed`) — the run launched
before I3, and the code agent built a degenerate baseline (synthetic toxicity
data scoring 0.0). The pipeline works end-to-end and reports the weak
reproduction honestly; the run_experiment fix itself is verified.

**Run 3 (first attempt).** Looped — the root spent all 21 iterations on
`understand_section` and never reached `detect_environment` / `run_experiment`.
Root cause: the I3 `_PAPER_GROUNDING` prompt section anchored the root on
paper-reading. I3 reverted (`c22feb7`).

**Run 3 (re-run, I3 reverted).** Progressed through the full pipeline —
detect_environment → build_environment → plan_reproduction → implement_baseline
→ run_experiment — confirming the I3 revert fixed the loop. `run_experiment`
timed out at the 3600 s per-command cap: GoRL's diffusion + evolutionary-RL
training on DMControl is genuinely compute-heavy (research vetting noted ~6 h
for a full run). The run then **crashed**: the Featherless plan caps
Qwen3-Coder context at 49 152 tokens, and the RLM root's accumulated prompt
grew to 56 291 — `context_length_exceeded` (HTTP 400). Verdict `failed`, leaf
0.0. The I3 revert worked (the pipeline ran end-to-end); the 49k context cap is
a separate hard limit of the Featherless Qwen3-Coder root — strong validation
of the redesign's "Claude root" decision (Claude's window is 200k+).

## Prior session 2026-05-21/22 (per handoff record)

| Paper | Source | Leaf score | Leaves |
|-------|--------|-----------:|-------:|
| sequential-neural-score-estimation | PaperBench bundle | 0.404 | 92/92 |
| mechanistic-understanding | PaperBench bundle | 0.286 | 96/96 |
| ftrl | PaperBench bundle | 0.000 | 178/178 |
| Recursive Language Models (arXiv 2512.24601) | arXiv, self-rubric | 0.325 | 21/21 |
| Minimal Circuits for IOI (arXiv 2510.25013) | arXiv, self-rubric | 0.363 | 23/23 |

`ftrl` scored 0.0 — the root reproduced "FTRL-Proximal" (online learning), an
acronym collision, not the paper's RL-finetuning method (issue I3).
