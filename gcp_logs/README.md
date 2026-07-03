# GCP VM logs — SDAR reproduction (2026-07-02 → 07-03)

Logs captured from the GCP VM running the SDAR (arXiv 2605.15155, *Self-Distilled
Agentic Reinforcement Learning*) reproduction effort.

- **VM:** `sdar-2model-a` · `us-central1-a` · `a2-ultragpu-4g` (4×A100-80GB)
- **Captured:** 2026-07-03 ~18:11 UTC (**final** — the authors' Search-QA run
  **completed all 150 steps + held-out validation**; see
  `sdar_authors_run/FINAL_RESULTS.md`)
- **Secret-scanned:** clean (no API keys / tokens / credentials).

There are **two distinct run types** here, and the difference is the whole story:

## 1. `sdar_authors_run/` — the faithful reproduction (verl + vLLM), WORKING

Runs the SDAR authors' actual trainer (`verl.trainer.main_sdar`, GRPO+OPSD,
vLLM 0.11.0) directly on the 4×A100 VM, via the authors' `run_search_3b.sh`.

**FINAL RESULT (150/150 steps + validation):** overall `val/success_rate` = **0.456**
(macro test_score 0.370). Per-dataset: TriviaQA 0.562 · PopQA 0.406 · NQ 0.402 ·
2Wiki 0.381 · HotpotQA 0.379 · Bamboogle 0.325 · Musique 0.137. Full breakdown +
training-signal analysis in **`sdar_authors_run/FINAL_RESULTS.md`**.

| File | What it is |
|---|---|
| `FINAL_RESULTS.md` | **The result** — final validation table, reward trajectory, SDAR-gate/RL-internals analysis, Table-1 framing. |
| `run_search_3b.log` | The training log — Qwen2.5-3B, Search-QA, **150/150 steps done**. Reward climbs 0.28 → ~0.50, SDAR gate active — a real, learning reproduction. |
| `metrics/` | `EXTRACTED_METRICS.txt` (clean val + final-step) + `reward_trajectory.txt` (per-step reward). |
| `config/` | Exact run commands (`run_search_3b.sh`, `run_search3b_proof.sh`). |
| `final_bundle.tar.gz` | Complete archive incl. the wandb offline run (`wandb sync`-able). Durable copy in `gs://deepinvent-ext-ut-sdar-runs/final_bundles/`. |
| `search3b_driver.log` | Launcher/driver: verl preflight, retriever startup, run lifecycle. |
| `retrieval_server.log` | The E5/faiss dense retrieval server (wiki-18, port 8000) serving the Search-QA env. |
| `debug-internal.log` | wandb offline-run internal log. |
| `webshop.log`, `search.log`, `alfworld.log`, `base.log`, `cpu_stage.log`, `stage_chain.log` | Env-staging logs (conda envs, data staging). `webshop.log` shows the Google-Drive corpus download that rate-limit-failed (the WebShop blocker). |

**Key result:** reward moves *off zero and climbs* — the faithful path produces a
genuine SDAR learning signal.

## 2. `campaign/` — the autonomous harness campaign (flat-zero, diagnosed)

The OpenResearch harness's own from-scratch reproduction attempt
(`prj_09047604e591d969`) — a campaign that reimplemented SDAR into single-cell
trainers. It produced **flat-zero reward** and `EXHAUSTED` on the LLM budget.
These logs are the evidence for the root-cause diagnosis (bm25-not-dense
retrieval, `max_turns=6` on ALFWorld, and the fact that SDAR needs the verl
framework — none of which is fixable by more budget).

| File / dir | What it is |
|---|---|
| `dashboard_events.jsonl` | Full SSE event stream for the campaign run. |
| `experiment_runs.jsonl` | Every `run_experiment` result (per-cell metrics, success flags). |
| `final_report.json` / `.md` | The campaign's final report (verdict `failed`, score 0.0). |
| `rubric_evaluation.json`, `rubric_tree.json`, `generated_rubric.json` | The auto-generated rubric + per-leaf scoring. |
| `run_config.json`, `environment_spec.json` | Run configuration + detected environment. |
| `cost_ledger.jsonl`, `tokens_total.json`, `demo_status.json` | Spend/token ledgers + status snapshot. |
| `ledger/campaign.json`, `ledger/attempts.jsonl` | Campaign state machine + per-attempt assessments (terminal: `EXHAUSTED budget_floor:llm_usd`). |
| `ledger/attempt_{1,2,3}.log`, `ledger/directives/*.json` | Per-attempt driver logs + steering directives. |
| `ledger/understanding.json`, `ledger/seed_staging.json` | Understanding-phase output + champion seed staging. |

**Key result:** the harness correctly *refused to ship a fake success*
(score 0.0, honest `failed`) — the flat-zero was a correctness/architecture
problem, not a budget one.
