# SDAR Search-QA 3B — FINAL results (150-step faithful reproduction)

**Paper:** SDAR — *Self-Distilled Agentic Reinforcement Learning* (arXiv 2605.15155)
**Model:** Qwen2.5-3B-Instruct · **Env:** Search-QA (Search-R1 test suite, dense wiki-18 retrieval)
**Trainer:** authors' `verl.trainer.main_sdar` (GRPO + OPSD gate, vLLM 0.11.0, FSDP, Ray)
**Hardware:** GCP `a2-ultragpu-4g`, 4×A100-80GB · **Steps:** 150 · **Wall:** ~15.5h train + ~2.3h final validation
**Run id:** wandb `offline-run-20260702_224258-3bv4bm9z` · **Completed:** 2026-07-03 ~15:26 UTC

This is the **faithful path** — the SDAR authors' own trainer run directly, *not* the
OpenResearch harness's from-scratch reimplementation (that attempt is in `../campaign/`
and flat-zeroed; see the root-cause note there). It is the evidence that the reproduction
**works** when run through the real verl+vLLM framework.

---

## Headline: held-out validation @ step 150

`val/success_rate` (overall) = **0.456** — ~45.6% Search-QA success for Qwen2.5-3B after 150 SDAR steps.

| Dataset | test_score | success_rate | avg tool calls |
|---|---:|---:|---:|
| TriviaQA        | 0.562 | 0.605 | 1.55 |
| PopQA           | 0.406 | 0.465 | 1.61 |
| NQ              | 0.402 | 0.418 | 1.55 |
| 2WikiMultiHopQA | 0.381 | 0.399 | 2.28 |
| HotpotQA        | 0.379 | 0.403 | 1.99 |
| Bamboogle       | 0.325 | 0.669 | 2.10 |
| Musique         | 0.137 | 0.147 | 2.46 |
| **macro-avg**   | **0.370** | **0.438** | 1.93 |

Ordering is exactly the expected difficulty curve for a 3B agent: single-hop factoid
(TriviaQA/PopQA/NQ) strongest, hard multi-hop (Musique) weakest. Tool-call counts rise
with hop-count (1.5 single-hop → 2.5 Musique), i.e. the agent is genuinely doing
multi-step retrieval, not one-shotting.

## Training signal (the reproduction is real, not a fake-zero)

- **Reward** climbed off zero and up: `0.28 → 0.44 → 0.50 → ~0.48–0.53` plateau (full curve in `metrics/reward_trajectory.txt`).
- **SDAR/OPSD gate active** the whole run: `sdar/gate_mean ~0.47–0.48`, `teacher_gap_mean ~-0.03…-0.06`, `sdar/loss ~0.01–0.02` — the sigmoid-gated self-distillation is doing real work, not inert.
- **Healthy RL internals:** `actor/pg_clipfrac ~0.002`, `ppo_kl ~0.000–0.001`, `grad_norm ~0.5–0.6`, `kl_coef 0.001` — stable GRPO, no collapse.
- **Dense retrieval working:** E5/faiss over wiki-18 on port 8000 served every rollout (real Wikipedia docs returned; see `retrieval_server.log.gz`).

## Note on the paper's Table 1

This arXiv id is future-dated; the exact per-dataset Table-1 cells are **not** quoted here
to avoid fabrication. The reproduced band (macro success ~0.44, single-hop ~0.42–0.61,
multi-hop tailing to ~0.15 on Musique) is squarely in the plausible range for Search-R1-style
agentic QA at the 3B scale. To close the loop, drop the paper PDF in and diff the exact cells.

## Files
- `run_search_3b.log` — full training log (all step metrics, reward, gate, val block)
- `search3b_driver.log` — launcher/driver (verl preflight, retriever startup, lifecycle)
- `retrieval_server.log.gz` — E5/faiss dense retrieval access log (gzip)
- `metrics/EXTRACTED_METRICS.txt` — clean val + final-step metrics
- `metrics/reward_trajectory.txt` — per-step `episode/reward/mean`
- `config/run_search_3b.sh`, `config/run_search3b_proof.sh` — exact commands
- `final_bundle.tar.gz` — complete archive incl. the wandb offline run (`wandb sync`-able)
- Durable copy: `gs://deepinvent-ext-ut-sdar-runs/final_bundles/search_3b_150step_20260703.tar.gz`
