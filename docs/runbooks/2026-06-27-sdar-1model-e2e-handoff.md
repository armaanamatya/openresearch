# SDAR Reproduction — Operator Handoff (1-model, end-to-end, guardrailed)

> **Status:** ready to run · last verified 2026-06-27. Run this from a **clean
> session** after `git pull` on the branch carrying the perfect-logic fixes.

## 0. What this run is

Reproduce the **SDAR** paper (arXiv 2605.15155, *Self-Distilled Agentic RL*)
**end-to-end** for **one model** across **all three experiments**, on a
guardrailed GCP 4×A100-80GB VM:

```
Qwen3-1.7B  ×  {ALFWorld, WebShop, Search-QA}  ×  {SDAR, GRPO}  =  6 cells
```

This is the **cheapest robust full-pipeline validation**. Once it proves out,
scale to the 2nd model (Qwen2.5-3B) by widening one flag (§7).

**Honest expectation:** a faithful 1-model run scores **~0.75–0.85** on the
rubric (not 1.0). The rubric grades the paper's *full breadth* (5 baselines, 3
gating modes, SkillBank, the 7B, exact hardware); a literal 1.0 ≈ reproducing
the entire paper. The goal here is **every environment genuinely earning reward
and the SDAR-vs-GRPO lift demonstrated**, with honest omissions declared.

---

## 1. What was perfected before this run (so it scores honestly)

| Fix | Why it matters |
|---|---|
| **WebShop in-process** (`webshop_env.py`) | Removed the `:3000` server bug that zeroed WebShop; runs the authors' in-process env via BM25 |
| **ALFWorld games-check** (`env_cache.py`) | A gameless download is now an honest *exclusion*, not a counted 0.0 |
| **paper_hints constants** | β=5.0 / sdar_coef=0.01 (authors' released values), WebShop+Search marked in-process |
| **gate_beta invariant** | The gate invariant no longer caps the score at 0.5 when the code uses the authors' `gate_beta` name |
| **SDAR-vs-GRPO lift backstop** (`cell_matrix.py`) | `baselines_vs_sdar` is derived deterministically — the headline can never silently score 0 |
| **Code-breadth + honest-omit guidance** | Adapt the authors' *complete* codebase (method-fidelity reads the code); declare untrained baselines + the 7B in `metrics.json['omitted']` |
| **Held-out eval guidance (FIX 6)** | Accuracy is a real held-out measurement, never `reward×100` |
| **Run-spec wiring** | The guardrailed launcher forwards the perfect-logic flags + 1-model scope + repo-first |
| **Cache + error/idle fast-shutdown guardrails** | The VM can never sit idle or burn after a crash |

All flag-gated and additive; the full test suite stays green.

---

## 2. Prerequisites (verify once, before provisioning)

```bash
cd /home/abheekp/openresearch
git status            # clean working tree on the perfect-logic branch
git log --oneline -1  # note the commit you are running

# GCP: project + account + zone (the warm cache lives in us-central1-c)
gcloud config list 2>/dev/null | grep -E 'project|account'   # deepinvent-ext-ut / abheek@deepinvent.ai
gcloud compute instances list --project deepinvent-ext-ut    # confirm NOTHING is RUNNING ($0)
gcloud compute disks list --project deepinvent-ext-ut --filter="name~sdar"   # sdar-ultra (1TB, us-central1-c) present
```

- **Quota:** `A100_80GB=4`, `A2_CPUS=48` → exactly one `a2-ultragpu-4g`. (To run 2 VMs later, bump `A2_CPUS`.)
- **Warm cache:** `sdar-ultra` (1 TB, us-central1-c) + machine image `sdar-mi-20260620` — carry HF weights / datasets / the wiki-18 FAISS index, so no 132 GB re-download.

---

## 3. Provision the guardrailed VM (us-central1-c, 4×A100-80GB, warm cache)

The runner polls for on-demand capacity (free while TERMINATED), provisions with
**all four guardrail layers**, attaches the warm cache, prepares the env, and
launches the run fully detached.

```bash
cd /home/abheekp/openresearch
setsid nohup env \
  OPENRESEARCH_GCP_ZONE=us-central1-c \
  OPENRESEARCH_GCP_GPU_MACHINE_TYPE=a2-ultragpu-4g \
  OPENRESEARCH_GCP_INSTANCE=sdar-1model \
  SDAR_VRAM_GB=80 \
  OPENRESEARCH_SDAR_USE_CACHE_DISK=1 \
  OPENRESEARCH_SDAR_CACHE_DISK=sdar-ultra \
  OPENRESEARCH_SDAR_CACHE_DISK_ZONE=us-central1-c \
  OPENRESEARCH_SDAR_USE_MI=1 \
  OPENRESEARCH_SDAR_ROOT=claude-oauth \
  OPENRESEARCH_SDAR_OUTER_WALL_S=144000 \
  OPENRESEARCH_GCP_MAX_RUN_DURATION=158400s \
  NO_AUTOSTOP=1 \
  bash scripts/sdar_gcp_optimal_run.sh > runs/_sdar_1model.log 2>&1 < /dev/null &

# watch the controller
tail -f runs/_sdar_1model.log
```

**Knob notes:**
- `OPENRESEARCH_SDAR_ROOT=claude-oauth` — keyless root + the **lifecycle driver** (default on) makes it reliable. If a funded `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` is present, `gpt-5` / `claude` are more reliable; set it accordingly.
- `OUTER_WALL_S=144000` (40 h) + `MAX_RUN_DURATION=158400s` (44 h) size the VM lifetime for 6× 1.7B cells at 150 steps. The **smoke (§4) confirms** the real per-cell time; shrink these if it's faster.
- The 1-model scope (`{"models": ["Qwen3-1.7B"]}`) is the launcher default — override with `OPENRESEARCH_SDAR_SCOPE_SPEC`.
- All perfect-logic flags (guards, repo-first, cache root) are baked into the run-spec; `EVAL_PROVENANCE_GUARD` defaults **OFF** for run 1 (see §6 troubleshooting).

> **If the warm cache lacks the WebShop corpus or wiki-18 index** (it is a former
> boot disk — verify on first SSH), populate it once with the authors' setup:
> `bash scripts/sdar_authors_repro.sh base alfworld webshop search` (CPU-only
> downloads; no GPU needed) before the harness run, or let the harness
> self-provision (slower).

---

## 4. The smoke = the first cell (fail-fast, then auto-continues)

The first training cell **is** the smoke. It costs ~$30–50 and ~30–60 min and
tells you the real per-round rate. **Watch these three signals on the VM:**

```bash
# SSH in (the runner prints the exact ssh command; or:)
gcloud compute ssh abheekp@sdar-1model --zone us-central1-c --project deepinvent-ext-ut

cd /home/abheekp/openresearch
RUN=$(ls -dt runs/2605.15155* runs/prj_* 2>/dev/null | head -1)   # the active run dir

tail -f "$RUN/code/.exec_live.log"          # 1) live training stdout (per-round logs)
tail -f "$RUN/dashboard_events.jsonl"        # 2) structured events (rubric_score, run_warning, …)
cat  "$RUN/code/.exec_heartbeat.json"        # 3) liveness heartbeat (proves it's progressing)
```

**Pass criteria for the smoke (before letting the grid continue):**
- WebShop/ALFWorld/Search each **construct and roll real episodes** — look for `env_health.jsonl` with `served > 0`, not an `env_setup_failed` exclusion.
- **Reward is non-zero and moving** (mean reward in the per-round logs climbs off 0).
- **No `eval_provenance` false-veto** in `dashboard_events.jsonl` run-warnings (it's OFF by default for run 1; if you turned it on, confirm cells aren't vetoed).
- The measured **per-round seconds** → multiply by 150 × cells to project the full cost; if it blows the budget, stop here (`down`, §5) and reconsider scope.

If healthy, the run **auto-continues** into the full 6-cell grid — no action needed.

---

## 5. Guardrails (the VM can never idle or burn)

Four independent layers, all already armed by the provisioning command:

1. **GCP `max-run-duration` → STOP** (44 h hard ceiling) — control-plane enforced, survives any process/kernel death. The ultimate backstop.
2. **Idle watchdog** (systemd timer) — no run process *and* GPU idle → `shutdown`. Two-grace: **300 s** when the run is known-dead (sentinel present), **3600 s** otherwise.
3. **Error/exit fast-shutdown** — a trap around the run command: on *any* exit (crash, OOM, error, success) the VM stops in **seconds**, not after the idle grace. (Set `OPENRESEARCH_SDAR_FASTCRASH_STAY_UP=1` only if you want to *hold* the VM on a <600 s crash for debugging.)
4. **Watcher** — pulls the report + stops the VM on terminal state; `NO_AUTOSTOP=1` lets the watcher own the stop (so the report is pulled before shutdown).

**Manual controls** (from your laptop):
```bash
bash scripts/sdar_gcp_optimal_run.sh down      # stop the VM now (halts GPU billing; disk persists)
bash scripts/sdar_gcp_optimal_run.sh inspect   # pull the latest report from the VM
```

---

## 6. Logging / debugging reference

| What | Where |
|---|---|
| Controller (provision→watch) | `runs/_sdar_1model.log` (local) |
| Live training stdout | `runs/<id>/code/.exec_live.log` (VM) — `tail -f` |
| Structured events | `runs/<id>/dashboard_events.jsonl` — `rubric_score`, `run_warning`, `primitive_call`, `experiment_progress` |
| Heartbeat / liveness | `runs/<id>/code/.exec_heartbeat.json` |
| Per-cell metrics | `runs/<id>/code/outputs/<run>/<cell>/metrics.json` |
| Aggregated metrics + lift | `runs/<id>/code/metrics.json` (`per_model`, `baselines_vs_sdar`) |
| Env health (did it earn?) | `runs/<id>/code/outputs/*/env_health.jsonl` (`served`, `unavailable`) |
| Provenance / held-out eval | `runs/<id>/code/**/provenance.json`, `eval_provenance.json` |
| Final report | `runs/<id>/final_report.{json,md}` (+ GCS if `OPENRESEARCH_SDAR_REPORT_GCS` set) |

**Run-warnings worth grepping for** (`grep run_warning dashboard_events.jsonl`):
`env_setup_failed` (an env didn't come up), `fabrication_suspected` (a guard vetoed a fake result), `no_learning_signal` (flat curves → inconclusive), `root_degenerate_refusal_loop` (root stuck — the lifecycle driver should recover), `cells_manifest_dropped`.

---

## 7. Cost, time, and scaling

- **1-model run (6 cells, 1.7B, 150 steps):** ~$300–800 / ~1–2 days, pinned by the smoke. The 1.7B fits one 80 GB card, so up to 4 cells run concurrently → it fits the VM lifetime.
- **Cost levers (no quality loss, §engineered):** the authors' vLLM rollouts are faster than the harness baseline; early-stop converged cells; right-size GPUs. The smoke reveals the real rate.
- **Scale to 2 models** (Qwen3-1.7B + Qwen2.5-3B, 12 cells): re-run §3 with
  `OPENRESEARCH_SDAR_SCOPE_SPEC='{"models": ["Qwen3-1.7B", "Qwen2.5-3B-Instruct"]}'`.

---

## 8. Success criteria

- Every env **earns**: `served > 0` and reward climbs off 0 for ALFWorld, WebShop, Search-QA.
- The **SDAR-vs-GRPO lift** is present in `code/metrics.json['baselines_vs_sdar']` (sdar_lift > 0 directionally).
- The **final report scores ~0.75–0.85** with no fabrication veto.
- The 7B (and untrained baselines) appear under `metrics.json['omitted']` — **honest omission**, not failure.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `eval_provenance` vetoes cells | It's OFF by default for run 1. Only enable (`OPENRESEARCH_EVAL_PROVENANCE_GUARD=1`) once the smoke shows the agent writes `eval_provenance.json` sidecars. |
| WebShop excluded (`env_setup_failed`) | The corpus/package isn't staged. SSH in, run `scripts/sdar_authors_repro.sh webshop` to build the WebShop data + index onto the cache, then re-run. |
| Search-QA falls back to BM25 (not E5) | `SEARCH_QA_INDEX_DIR` isn't pointing at the cached wiki-18 FAISS index — set it (or confirm the cache has it). |
| A cell OOMs | The harness shrink-retries (batch 0.5→0.25 + grad-ckpt); if all cells OOM it STOPs honestly. |
| 1.7B cell exceeds the VM ceiling | Raise `OPENRESEARCH_GCP_MAX_RUN_DURATION` + `OPENRESEARCH_SDAR_OUTER_WALL_S`; or reduce steps for a first pass. |
| Root model churns / degenerates | The lifecycle driver + degenerate detector recover keyless `claude-oauth`; or switch to a funded `gpt-5`/`claude` root. |
| VM left running | It can't — but to be sure: `bash scripts/sdar_gcp_optimal_run.sh down`. |

---

*Generated 2026-06-27. The perfect-logic fixes, run-spec wiring, and guardrails
referenced here are on the current branch; commit SHA is in `git log`.*
