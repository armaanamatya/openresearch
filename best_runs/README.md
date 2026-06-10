# Best runs

End-to-end reproductions from the OpenResearch agent — clean single runs (All-CNN, Adam, VAE) and one multi-attempt campaign on the hard SDAR stress paper. PDF in, scored reproduction out, no human in the loop on the coding side.

| Paper | Verdict | Rubric | Iter | Wall |
|---|---|---:|---:|---:|
| [Striving for Simplicity: The All Convolutional Net (Springenberg et al., 2014)](allcnn/) | reproduced | 0.696 | 1 | 8.5h |
| Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014) | reproduced | 0.741 | 19 | 16m |
| Auto-Encoding Variational Bayes (Kingma & Welling, 2013) | partial | 0.646 | 3 | 30m |
| [SDAR: Self-Distilled Agentic RL (2605.15155)](sdar/) — 4-attempt campaign | partial | 0.363 | 10 | 197m |

Each subdirectory carries the final report (`final_report.json` + `.md`), the auto-derived rubric, the leaf-by-leaf grading, the environment spec, the generated training code, and the telemetry sidecars (token counts, per-primitive timing, cost ledger, every `run_experiment` result).

The **SDAR campaign** (`sdar/`) is shaped differently: it packages four back-to-back attempts from 2026-05-31→06-01 — three infrastructure failures (GPU contention, a launch crash, a dead dataset endpoint) and one scored `partial` — each with its own run logs and token-usage logs. See [`sdar/README.md`](sdar/README.md) for the per-attempt breakdown and the campaign token table.

---

## All-CNN — GPU-scale CIFAR grid with an in-run repair loop

A 14-cell training grid (models A/B/C × base/strided/convpool/all-conv variants,
350 epochs each on CIFAR-10, plus All-CNN-C on CIFAR-10+aug and CIFAR-100):
the first grid dead-trained half its cells on a shared learning rate; the
harness's dead-training divergence report drove an in-run repair (per-cell lr
probe) whose second grid resurrected `c_base` and `c_strided` to paper level.

| Measured (final grid, test error %) | base | strided | convpool | all-conv |
|---|---:|---:|---:|---:|
| Model A | **12.86** | 64.82 | 71.33 | 90.0 |
| Model B | **10.77** | **13.27** | 90.0 | 90.0 |
| Model C | **9.99** | **10.52** | 90.0 | 90.0 |

Bold = paper-grade (the paper's CIFAR-10 table spans ~9–16% for these
configurations; 90% = chance, an honest dead cell — recorded, not hidden). The
per-cell results incl. each cell's probed learning rate are in
[`allcnn/cells_results.json`](allcnn/cells_results.json); the leaf-by-leaf
grading (`allcnn/rubric_evaluation.json`) shows exactly which claims earned the
0.696 and which leaves (the all-conv/convpool families, the Section-4
ReLU-masking figure, ImageNet) hold the remaining headroom.

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

## Token usage

Both runs used Claude Code OAuth (Claude Sonnet via subscription), so per-token API spend is zero. The numbers are a *workload* signal — what an equivalent Anthropic-API or Azure-OpenAI deployment would meter.

### How to read the columns

- **Input** — fresh tokens the model reads on a call, excluding anything served from cache.
- **Output** — tokens the model generates. Priced highest on every vendor.
- **Cache write** — first-time-seen prompt content the provider caches for cheap replay.
- **Cache read** — prompt content served from cache. ~10× cheaper than fresh input on Anthropic and Azure.
- **Total** — sum of all four. Everything the meter touched.

### Adam — 19 iterations, ~16 min wall clock

| | Tokens |
|---|---:|
| Input (fresh) | 12 |
| Output (generated) | 2,075 |
| Cache write | 13,629 |
| Cache read | 72,978 |
| **Total** | **88,694** |
| Root calls | 20 |

Per-primitive output (only the primitives that emit at the root — the rest dispatch to sub-agents or are pure I/O):

| Primitive | Calls | Output |
|---|---:|---:|
| `plan_reproduction` | 1 | 1,104 |
| `propose_improvements` | 1 | 971 |
| `understand_section` | 4 | 0 (sub-agent meter) |
| `implement_baseline` | 1 | 0 (Sonnet sub-agent) |
| `verify_against_rubric` | 1 | 0 (grader sub-agent) |
| `run_experiment` | 1 | 0 (Docker, no LLM) |
| `heartbeat` | 6 | 0 |
| 5 other primitives | 1 ea. | 0 |

### VAE — 3 iterations, ~30 min wall clock

| | Tokens |
|---|---:|
| Input (fresh) | 32 |
| Output (generated) | 4,867 |
| Cache write | 63,858 |
| Cache read | 217,338 |
| **Total** | **286,095** |
| Root calls | 34 |

Per-primitive output:

| Primitive | Calls | Output |
|---|---:|---:|
| `plan_reproduction` | 1 | 2,549 |
| `propose_improvements` | 1 | 1,186 |
| `verify_against_rubric` | 4 | 1,132 |
| `understand_section` | 4 | 0 (sub-agent meter) |
| `implement_baseline` | 3 | 0 (Sonnet sub-agent) |
| `run_experiment` | 3 | 0 (Docker, no LLM) |
| `record_candidate_outcome` | 3 | 0 |
| `heartbeat` | 10 | 0 |
| 5 other primitives | 1 ea. | 0 |

### Side-by-side

| | Adam | VAE | Combined |
|---|---:|---:|---:|
| Root calls | 20 | 34 | 54 |
| Input (fresh) | 12 | 32 | 44 |
| Output (generated) | 2,075 | 4,867 | **6,942** |
| Cache write | 13,629 | 63,858 | 77,487 |
| Cache read | 72,978 | 217,338 | 290,316 |
| **Total tokens** | **88,694** | **286,095** | **374,789** |

### What this tells you

- **~98% of traffic is cache I/O.** The root generated only 6,942 tokens across both reproductions; everything else is the RLM scratchpad being replayed against a warm prompt cache. That replay pattern is what makes 19-iteration runs affordable.
- **VAE used ~3.2× Adam's tokens** despite running 6× fewer iterations. The driver is scratchpad length per turn — VAE's `plan_reproduction` emitted 2,549 output tokens (vs Adam's 1,104), and each subsequent iteration replays the cumulative state.
- **Three primitives carry all root-side generation:** `plan_reproduction`, `propose_improvements`, and (on multi-iteration runs) `verify_against_rubric`. Everything else dispatches to a sub-agent on its own meter, or is pure file I/O.
- **Sub-agent tokens are not in these totals.** `implement_baseline` invokes a Sonnet sub-agent through `claude-agent-sdk`; that traffic rolls up under the local subscription, not under the root run's `tokens_total.json`. A production billing view would add the sub-agent ledger on top.

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
  same shape; tokens_total.json + cost_ledger.jsonl included, timing.json absent
  (VAE pre-dates the full sidecar emit, but token telemetry was backfilled).
```
