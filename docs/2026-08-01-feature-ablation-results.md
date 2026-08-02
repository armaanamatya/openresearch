<!-- doc-meta: status=live; started=2026-08-01 -->
# Feature-ablation results — per-feature reproduction scores

Live results of the per-feature ablation on GCP single-VM (L4), **`sonnet-foundry`** (real
Claude via Foundry API key — **NOT OAuth**, thinking-patched `_anthropic_thinking_patch.py`).
Each arm = the fixed honest baseline (`configs/ablation/baseline_run_spec.json`) plus ONE
feature (`configs/ablation/arms.json`); `all_on` = every feature. Scored by the auto-generated
PaperBench rubric (`rubric.overall_score`). Method + launch: `docs/runbooks/2026-08-01-feature-ablation-gcp-runbook.md`.

## Scoreboard (paper: ResNet arXiv 1512.03385, seed 1, L4)

| Feature (arm) | What it adds | Rubric score | Verdict | Δ vs baseline | Date/time (UTC) | Status |
|---|---|---:|---|---:|---|---|
| **baseline** | fixed honest infra, no test features | _pending_ | — | — | started 2026-08-01 | 🟡 running |
| bes | best-of-N candidate pool | _pending_ | — | — | — | ⏳ queued |
| champion | champion-artifact + evidence-fingerprint rails | _pending_ | — | — | — | ⏳ queued |
| recipes | cross-run positive recipes | _pending_ | — | — | — | ⏳ queued |
| expmem | cross-run experience memory | _pending_ | — | — | — | ⏳ queued |
| lessons | cross-run negative lessons | _pending_ | — | — | — | ⏳ queued |
| audit | evidence-audit deterministic critic | _pending_ | — | — | — | ⏳ queued |
| leafgate | per-leaf evidence gate (anti-fabrication) | _pending_ | — | — | — | ⏳ queued |
| **all_on** | all 7 features combined | _pending_ | — | — | — | ⏳ queued |

> Scores populate as each ~1.5 h run completes. The baseline is validated running end-to-end
> (root → rubric-gen → executor producing code → GPU training → grade) on `sonnet-foundry`.

## Method (so the numbers are trustworthy)
- **Auth: `sonnet-foundry` (Foundry API key), never OAuth** (operator directive). Root +
  executor + grader + verifier all `sonnet-foundry`.
- **Baseline infra ON in every arm** (reliability + anti-fabrication + grader-fidelity +
  feasibility scope), so a score reflects the feature, not a broken run. Only the one ablated
  feature differs per arm.
- **Δ vs baseline** is the per-feature contribution. `all_on` vs Σ(individual Δ) tells you if
  features compose or are redundant.
- Fidelity caveat: a full per-feature verdict needs ≥3 seeds through the grader-σ gate
  (`asha_authority_gate`); this scoreboard is the 1-seed screen.

## Run log
- **2026-08-01** — GCP pipeline validated on `sonnet-foundry`: root crash fixed, rubric-gen
  fixed (thinking-disable patch), executor generates code. Two prior Foundry-Sonnet blockers
  found + fixed (see `docs/runbooks/2026-08-01-remote-run-llm-auth.md`).
- **2026-08-02 — baseline (`base_rn`) confirmed TRAINING ON GPU** (L4 at 22% util / 476 MiB;
  `train_cell.py` cell `plain20__cifar10__s42` running, torch loaded, CUDA engaged). Full chain
  validated end-to-end on `sonnet-foundry` + torch: rubric → CUDA-correct codegen → real GPU
  training. Score pending run completion (~1–2 h for the ResNet cell matrix). SSH to the VM is
  intermittently 255-rate-limited; the log stdout is buffered (monitor via disk artifacts +
  `nvidia-smi`, not the log tail).
- **2026-08-01 (3rd blocker)** — first baseline finished in ~15 min with a **null score**:
  `train_cell.py` failed `ModuleNotFoundError: No module named 'torch'` (`cell_execution_error`).
  **The VM bootstrap installed only `backend/requirements.txt` (orchestrator deps), not the ML
  training deps.** Fix: after `backend/requirements.txt`, ALSO
  `uv pip install torch torchvision numpy --index-url https://download.pytorch.org/whl/cu121`
  into the run venv on EVERY arm VM. Baseline re-launched with torch (real GPU training this
  time). This is now in the fan-out cron + the runbook — do NOT fan out without it.
