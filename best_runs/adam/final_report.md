# REPRODUCED

**Paper:** paper_text

## Rubric Score

**Overall score:** 0.831  (✔ meets target)

_20/20 rubric leaves graded · PaperBench bundle rubric_

| Area | Score | Notes |
|---|---|---|
| Method and code fidelity to the paper | 0.985 |  |
| Data and preprocessing fidelity | 0.880 |  |
| Experiment execution and reproducibility | 0.881 |  |
| Evaluation protocol and metric correctness | 0.772 |  |
| Result match versus the paper's reported targets | 0.613 |  |
| Artifact completeness and provenance | 0.775 |  |

### Weakest rubric leaves

| Score | Justification |
|---|---|
| 0.40 | The IMDB BoW experiment ran all three optimizers (adam, adagrad, sgd_nesterov) with a figure produced, but final metrics show near-identical accuracy (~0.82-0.83) and SGD-Nesterov actually has the lowest NLL, so the c… |
| 0.55 | The bias-correction study ran all three configs and final ELBO shows bias-corrected Adam (100.44) slightly better than RMSProp (101.36) and no-bias (101.33), reproducing the equal-or-better claim, but the early-epoch … |
| 0.60 | The CIFAR-10 CNN was run for Adam, SGD-Nesterov, and AdaGrad with loss/accuracy figures, and the accuracy ordering (Adam 0.695, SGD 0.745 both > AdaGrad 0.632) is consistent with the claim, though 45-epoch convergence… |
| 0.60 | CIFAR-10 loads and runs, but provenance states per-channel mean/std normalization as a 'ZCA approximation' rather than the paper's actual whitening, so it is only an approximation. |
| 0.60 | best_lr values per optimizer in mnist_lr indicate a learning-rate grid search, but a dense momentum grid and identical initialization across optimizers are not evidenced. |

## Reproduction Summary

Reproduced the paper 'paper_text' using the RLM framework. Implemented baseline, ran experiment, and scored against rubric. Final rubric score: 0.848 (target: 0.600). Run succeeded.

## Scope

**Requested:** full paper reproduction

**Ran:**
- /home/sww35/openresearch/runs/prj_6d41d2f09c026403/code

## Baseline Metrics vs. Paper Claims

| Metric | Reproduced | Paper Claim |
|---|---|---|
| cifar10_cnn | {'adam': {'final_accuracy': 0.6953, 'final_nll': 1.6653250820159913, 'initial_train_loss': 1.5849443960677632}, 'sgd_nesterov': {'final_accuracy': 0.7452, 'final_nll': 1.6023629070281982, 'initial_train_loss': 1.6409936140260428}, 'adagrad': {'final_accuracy': 0.6318, 'final_nll': 1.2345273859024049, 'initial_train_loss': 2.6038157662467274}} | — |
| data_load_failures | [] | — |
| imdb_lr | {'adam': {'final_accuracy': 0.82848, 'final_nll': 0.5539318472290039, 'initial_nll': 0.6904043555259705}, 'sgd_nesterov': {'final_accuracy': 0.82344, 'final_nll': 0.5092434591674805, 'initial_nll': 0.6904043555259705}, 'adagrad': {'final_accuracy': 0.83204, 'final_nll': 0.5162493334960937, 'initial_nll': 0.6904043555259705}} | — |
| mlp_mnist | {'adam': {'final_accuracy': 0.9739, 'final_nll': 0.08967941943109035}, 'adagrad': {'final_accuracy': 0.9733, 'final_nll': 0.092402837318182}, 'rmsprop': {'final_accuracy': 0.9707, 'final_nll': 0.10775472101569175}, 'sgd_nesterov': {'final_accuracy': 0.9713, 'final_nll': 0.09877280540466309}, 'adadelta': {'final_accuracy': 0.9731, 'final_nll': 0.09236933890283108}} | — |
| mnist_lr | {'adam': {'final_accuracy': 0.914, 'final_nll': 0.31004185650348665, 'initial_nll': 2.32299542427063, 'best_lr': 0.003}, 'sgd_nesterov': {'final_accuracy': 0.9063, 'final_nll': 0.33408871240615845, 'initial_nll': 2.32299542427063, 'best_lr': 0.01}, 'adagrad': {'final_accuracy': 0.8903, 'final_nll': 0.43803787903785707, 'initial_nll': 2.32299542427063, 'best_lr': 0.01}, 'adamax': {'final_accuracy': 0.9233, 'final_nll': 0.27588974616527556}} | — |
| per_dataset | {'mnist': {'final_accuracy': 0.914, 'final_nll': 0.31004185650348665, 'initial_nll': 2.32299542427063}, 'cifar10': {'final_accuracy': 0.6953, 'final_nll': 1.6653250820159913}} | — |
| provenance | {'paper': 'Kingma & Ba (2014) Adam: A Method for Stochastic Optimization arXiv:1412.6980', 'unresolved': "The paper_claim_map listed metric 'return'=2 which is not interpretable for supervised image classification. Faithfulness metric is NLL + accuracy.", 'assumptions': {'stepsize_decay': 'alpha_t = alpha / sqrt(t+1) applied in Sec 6.1 experiments', 'cifar10_whitening': 'per-channel mean/std normalization (standard ZCA approximation)', 'imdb_bow': '10k words, binary presence, 50% Bernoulli dropout on input', 'mlp': '2 x 1000 ReLU hidden layers, 50% dropout, weight_decay=1e-5', 'cnn': 'c64-c64-c128-1000, 5x5 conv + 3x3 maxpool stride 2 (3 stages)'}} | — |
| scope | {'models_run': ['adam', 'adagrad', 'rmsprop', 'sgd_nesterov', 'adadelta'], 'models_skipped': [], 'gaps': []} | — |
| status | completed | — |
| vae | {'adam_bias_corrected': {'final_elbo': 100.44331399281819, 'initial_elbo': 157.13462783813478, 'label': 'Adam β2=0.999, bias-corrected'}, 'adam_no_bias_correction': {'final_elbo': 101.33066977183024, 'initial_elbo': 142.58349756876626, 'label': 'Adam β2=0.999, NO bias correction'}, 'rmsprop': {'final_elbo': 101.36477029164632, 'initial_elbo': 158.16506591796875, 'label': 'RMSProp ρ=0.999'}} | — |
| wall_time_seconds | 3899.9701511859894 | — |

## Improvement Candidates

**1. Candidate 1** — declined
**2. Candidate 2** — declined
**3. Candidate 3** — declined

## Cost

| Category | USD |
|---|---|
| Primitive-internal LLM | $0.150887 |
| **Total LLM** | **$0.150887** |

**Iterations:** 1

## Token Usage

| Metric | Value |
|---|---|
| Total LLM calls | 23 |
| Input tokens | 50 |
| Output tokens | 68,076 |
| Cache creation (input) | 174,662 |
| Cache read (input, prompt-cache-billed) | 3,721,469 |

### Per-primitive token usage

| Primitive | Calls | Input | Output |
|---|---|---|---|
| baseline-implementation | 2 | 48 | 63,093 |
| plan_reproduction | 1 | 2 | 4,983 |

---
_Generated by ReproLab RLM orchestrator (Issue #60)._
