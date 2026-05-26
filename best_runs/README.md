# Best runs

Two end-to-end reproductions from the OpenResearch agent. PDF in, scored reproduction out, no human in the loop on the coding side.

| Paper | Verdict | Rubric | Iter | Wall |
|---|---|---:|---:|---:|
| Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014) | reproduced | 0.741 | 19 | 16m |
| Auto-Encoding Variational Bayes (Kingma & Welling, 2013) | partial | 0.646 | 3 | 30m |

Each subdirectory carries the final report (`final_report.json` + `.md`), the auto-derived rubric, the leaf-by-leaf grading, the environment spec, the generated training code, and the telemetry sidecars (token counts, per-primitive timing, cost ledger, every `run_experiment` result).

---

## Adam — what the agent extracted, what it re-derived

| Paper claim | Expected | Reproduced |
|---|---|---|
| CIFAR-10 CNN at 45 epochs: Adam + SGD+Nesterov ≪ AdaGrad | ordering | Adam 0.536, SGD+N 0.473, AdaGrad 0.983 |
| Bias correction stabilizes training as β₂ → 1 (VAE softplus) | bc < no-bc early | 10 ep: bc −119.87 vs no-bc −97.25 |
| MNIST logreg training NLL: Adam < SGD+N < AdaGrad | ordering | 0.231 / 0.251 / 0.354 |

All three quantitative claims the agent pulled out of the paper were independently re-derived from agent-authored `train.py`.

## VAE — close on direction, short on absolute targets

| Paper claim | Expected | Reproduced |
|---|---|---|
| MNIST AEVB test ELBO, Nz=20 | ≈ −98 (Table 2) | −123.19 |
| ELBO improves monotonically with Nz | Nz=3 ≪ Nz=10 < Nz=20 | −156.03 / −126.63 / −123.19 |
| AEVB > Wake-Sleep > MCEM (Ntr=1000) | sign | −200.01 / −200.56 / −204.93 |
| Frey Face latent traversal (Figure 2) | qualitative | skipped — mirror returned HTTP 403 |

Directional claims hold. The absolute ELBO gap reflects a smaller training budget than the paper, not a bug — the agent ran four model variants in ~30 minutes of wall clock against a 24 GB consumer GPU. Frey Face was *not* silently substituted with a synthetic dataset; the loader failure is recorded in `vae/final_report.json` under `baseline_metrics.data_load_failures`.

---

## Rubric by area

A PaperBench-style grader runs after the reproduction and scores 24 leaf criteria across six areas. Failed-or-skipped leaves are excluded from the roll-up rather than being scored as zero.

| Area | Adam | VAE |
|---|---:|---:|
| Method + code fidelity | 0.888 | 0.745 |
| Data + preprocessing fidelity | 0.617 | 0.500 |
| Experiment execution | 0.750 | 0.465 |
| Evaluation protocol | 0.617 | 0.415 |
| Result match vs. paper | 0.800 | 0.215 |
| Artifact completeness | 0.250 | 0.265 |

The pattern across both runs is the same: methodology and execution land high, artifact provenance lands low. Fixing one primitive — `emit_model_card` — would lift the last column on every future paper.

---

## How the reproduction actually happens

The orchestrator is a Recursive Language Model (RLM, arXiv 2512.24601). The paper is loaded into a Python REPL as a variable; the root model never carries the corpus in its context window. When it needs paper content it writes Python that slices the variable. That is what makes 100K-token papers tractable.

The root model can call twelve domain primitives:

`understand_section`, `extract_hyperparameters`, `detect_environment`, `build_environment`, `plan_reproduction`, `implement_baseline`, `run_experiment`, `verify_against_rubric`, `propose_improvements`, `record_candidate_outcome`, `check_user_messages`, `respond_to_user`.

Iteration count, primitive order, and termination are decisions made by the root model. The orchestrator's job is to enforce guard-rails (rubric-guard schema assertions on the generated `train.py`, a forced-iteration policy that refuses early `FINAL_VAR` when the score is sub-target, a wall-clock watchdog, the SSE sanitizer that ensures REPL locals never leak to the UI).

The same orchestrator runs under Claude Code OAuth, Anthropic API, OpenAI, Azure OpenAI, or Featherless — picked by one CLI flag. The recorded runs in this directory used OAuth, so per-token cost is zero on the LLM side and the marginal cost is GPU time only.

---

## Files

```
adam/
  final_report.json          canonical artifact: verdict, baseline_metrics, paper_claims, cost, iterations
  final_report.md            human-readable version with telemetry rendered inline
  generated_rubric.json      24 leaves auto-derived from the paper text
  rubric_evaluation.json     per-leaf score + justification from the grader
  environment_spec.json      packages, framework, GPU/CPU plan
  Dockerfile                 built image
  tokens_total.json          token counts by model + by primitive
  timing.json                wall clock + per-primitive duration + GPU hours
  cost_ledger.jsonl          append-only USD per primitive
  experiment_runs.jsonl      every run_experiment: success, metrics, logs, failure-class, fix
  demo_status.json           run lifecycle snapshot
  code/                      the agent-written train.py + supporting files

vae/
  same shape (VAE pre-dates the cost_ledger / tokens / timing sidecars)
```
