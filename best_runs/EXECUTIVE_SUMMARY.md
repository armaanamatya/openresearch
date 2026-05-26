# OpenResearch — Executive Summary

**One sentence:** an autonomous AI agent that ingests a research paper PDF and produces a working, scored reproduction of it — no human-in-the-loop coding.

## Headline results

| Paper | Verdict | Rubric Score | Iterations | Wall Clock | Output Tokens |
|---|---|---:|---:|---:|---:|
| **Adam: A Method for Stochastic Optimization** (Kingma & Ba, 2014) | **reproduced** | **0.741 / 1.0** | 19 | 16.3 min | 2,075 |
| **Auto-Encoding Variational Bayes / VAE** (Kingma & Welling, 2013) | partial | 0.646 / 1.0 | 3 | — | — |

Both runs used the agent's **subscription-only auth path** (Claude Code OAuth + local GPU) — zero per-token API spend on the recorded artifacts. Production deployments can flip to per-token Anthropic / Azure / OpenAI billing transparently.

## Why this matters for MSFT

1. **Replicates frontier-model agentic behavior end-to-end** — pdf in, code + metrics out. The orchestrator is the Recursive Language Model (RLM, arXiv 2512.24601), with paper content offloaded as a REPL variable so it never bloats the model's context window. This is a worked, observable example of context-engineered agents at >100K-token paper sizes.
2. **Scored, not just demoed** — every run is graded against an auto-derived PaperBench-style rubric (24 leaves per paper, six areas) so quality is measurable rather than asserted.
3. **First-class telemetry** — token usage / wall clock / per-primitive timing / cost ledger are emitted as sidecar JSON on every run. This is the surface a leaderboard, billing dashboard, or A/B compare needs (see `adam/tokens_total.json`, `adam/timing.json`, `adam/cost_ledger.jsonl`).
4. **Auth-surface flexibility** — same orchestrator runs under Claude Code OAuth, Anthropic API, OpenAI, or Azure OpenAI with one CLI flag. Field deployments don't need to repick a vendor.

## Rubric breakdown (where the agent is strong vs. where it's weak)

| Rubric Area | Adam | VAE |
|---|---:|---:|
| Method + code fidelity to the paper | 0.888 | 0.745 |
| Data + preprocessing fidelity | 0.617 | 0.500 |
| Experiment execution + reproducibility | 0.750 | 0.465 |
| Evaluation protocol + metric correctness | 0.617 | 0.415 |
| Result match vs. paper's reported targets | 0.800 | 0.215 |
| Artifact completeness + provenance | 0.250 | 0.265 |

**Read:** the agent reliably writes correct *methodology* and gets close to paper-reported *results* on well-behaved papers (Adam). Its weakest dimensions are artifact provenance (release a model card, dump configs, etc.) and dataset acquisition on papers whose loaders have rotted (VAE's Frey Face mirror returned HTTP 403; the agent recorded the data-load failure and proceeded with MNIST only).

## What's in this directory

- `adam/` — full reproduction of Adam (1412.6980): final report, generated code, env spec, telemetry sidecars, leaf-level rubric scoring.
- `vae/` — full reproduction of VAE (1312.6114): same shape.
- `METHODOLOGY.md` — how the agent does this (12 domain primitives, RLM root, sub-agent workers, sandbox).
- `PAPER_CLAIMS_VS_REPRODUCED.md` — side-by-side of what each paper *claims* vs. what the agent actually re-derived.

## Concrete next bets if MSFT engages

- Wire the same orchestrator to **Azure OpenAI** as root + sub-agent (already supported behind `--model azure`; needs deployment-grade testing on real customer GPUs).
- **Scale axis:** dynamic-GPU resolver (spec `docs/superpowers/specs/2026-05-23-dynamic-gpu-selection-design.md`) picks the right RunPod SKU per paper. Identical hook would land on Azure ML compute.
- **Quality axis:** the lowest-scoring rubric area is *artifact completeness* — fix one primitive (`emit_model_card`) and every paper's score moves up ~10 pts simultaneously. High leverage.
