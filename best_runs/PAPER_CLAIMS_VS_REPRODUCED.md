# Paper Claims vs. Reproduced Metrics

The agent extracts every quantitative claim from the paper, then re-derives the numbers from a freshly-written baseline. This file shows the side-by-side. **All numbers below were computed by the agent autonomously** — no human typed an `--expected-result`.

---

## Adam (1412.6980 — Kingma & Ba, 2014)

| Paper Claim | Expected | Agent Reproduced | ✓ |
|---|---|---|:-:|
| Adam + SGD+Nesterov converge **far below** AdaGrad on CIFAR-10 CNN (45 epochs) | Adam, SGD+N « AdaGrad | Adam = 0.5358, SGD+N = 0.4726, AdaGrad = 0.9832 | ✓ |
| Bias correction stabilizes training, especially at β₂ → 1 (VAE softplus, 500 hidden) | bias-corrected < uncorrected at low epochs | 10 ep: bc = −119.87 vs nobias = −97.25; 100 ep: −133.02 vs −132.89 | ✓ |
| Adam < SGD+Nesterov < AdaGrad on MNIST logreg (training NLL) | ordering | Adam = 0.231, SGD+N = 0.251, AdaGrad = 0.354 | ✓ |

Headline: every quantitative claim the agent extracted from the Adam paper was independently re-derived from agent-authored code in the same training regime. Final rubric: **0.741 (reproduced)**.

---

## VAE / Auto-Encoding Variational Bayes (1312.6114 — Kingma & Welling, 2013)

| Paper Claim | Expected | Agent Reproduced | Notes |
|---|---|---|---|
| MNIST AEVB test ELBO at Nz=20 | ~ −98 (Table 2 / Fig 3) | **−123.19** | Lower-bound looser than paper; agent ran fewer epochs to fit local-GPU budget. |
| MNIST AEVB Nz=10 test ELBO | (qualitatively close to Nz=20) | −126.63 | Monotone ordering vs Nz=20 preserved. |
| MNIST AEVB Nz=3 test ELBO | (worst of the three) | −156.03 | Ordering preserved. |
| AEVB log p(x) > Wake-Sleep at same Nz | AEVB beats WS | AEVB = −200.01, WS = −200.56 at Ntr=1000 | ✓ (small margin, sign matches paper) |
| AEVB beats Wake-Sleep across MCEM too | AEVB > WS > MCEM | AEVB −200.01 > WS −200.56 > MCEM −204.93 | ✓ |
| Frey Face latent-space generation (Fig 2) | qualitative | **SKIPPED** — original mirror (cs.nyu.edu) returns HTTP 403 | Recorded as `data_load_failures`. The agent did NOT silently substitute a synthetic dataset — it logged the failure and reduced scope. |

Headline: agent reproduced the **directional claims** (AEVB > WakeSleep > MCEM; ELBO improves monotonically with latent dim) but missed the absolute ELBO target because it ran 4 model variants in 1,793 s of wall clock (a fraction of the paper's training budget). Final rubric: **0.646 (partial)**.

---

## How to read these numbers

- **Verdict** is the agent's own judgment (`reproduced` / `partial` / `failed`). It only writes `reproduced` when every extracted claim's *direction* matches and the result-match rubric area exceeds 0.7.
- **Rubric score** is computed by an independent grader pass: 24 leaf criteria, six areas, each weighted by paper-specified importance. Failed-or-skipped leaves are excluded from the roll-up (PR-κ), so the score reflects what was actually evaluable rather than punishing data-acquisition failures the agent couldn't prevent.
- **Data-load failures are first-class** — see `vae/final_report.json::baseline_metrics.data_load_failures`. The agent refuses to fabricate datasets.
