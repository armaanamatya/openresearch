---
name: sdar-reproduction
description: Use when reproducing SDAR (arXiv 2605.15155, "Self-Distilled Agentic Reinforcement Learning", repo ZJU-REAL/SDAR) — GRPO with a sigmoid-gated on-policy self-distillation term (OPSD/gap/soft_or), trained as a multi-turn agent on ALFWorld, WebShop, and Search-QA with Qwen3/Qwen2.5 policies via verl + vLLM. Names the loss objective and its stop-gradient invariant, the pinned toolchain and dataset staging, held-out evaluation, and the specific fidelity traps that fail an otherwise-faithful run.
category: paper-reproduction
tags: [SDAR, GRPO, OPSD, self-distillation, RLSD, agentic, reinforcement learning, RL, reasoning, search, QA, ALFWorld, WebShop, Search-QA, Qwen, verl, vLLM, TRL, reward, policy, success rate, held-out eval, multi-turn]
---

# Reproducing SDAR (Self-Distilled Agentic Reinforcement Learning)

**Paper:** arXiv 2605.15155 · **Official repo:** `github.com/ZJU-REAL/SDAR` · this repo's canonical baseline.

SDAR adds a **gated on-policy self-distillation** term to GRPO. One set of weights is both the
"student" (plain prompt) and the "teacher" (privileged/skill-augmented prompt); a per-token sigmoid
gate decides how hard to distill teacher→student on top of the RL objective. Trained as a multi-turn
agent on three environments. **Headline claim (≈, within 10% rel.):** SDAR beats vanilla GRPO on all
three — ALFWorld ~70→82% success, Search-QA ~38→46% EM, WebShop ~60→70 score.

## When to use this skill
- The paper is arXiv 2605.15155 / "SDAR" / mentions OPSD, gated self-distillation, or GRPO on
  ALFWorld+WebShop+Search-QA.
- You are about to write the loss, the agentic rollout, or the eval for an SDAR reproduction.
- A run scored ~0 or a guard blocked it and you need to tell a genuine faithfulness failure from a
  harness artifact.

Run-spec: `configs/autonomous_reproduction_run_spec.json` (execute mode, `USE_AUTHOR_REPO=1` — the
repo is seeded; execute-mode fail-closes without a resolvable repo URL). Prefer **adapting the
authors' verl trainer** over hand-rolling — a from-scratch trainer is where fidelity dies.

## The method — implement exactly this
Total objective (`sdar_loss.py`): `L(θ) = L_GRPO(θ) + λ_sdar · L_SDAR(θ)`.

- **GRPO** (no value function / no GAE): group-relative advantage over `G` rollouts of the SAME
  prompt, `A_i = (r_i − mean(r_1..G)) / (std + ε)`, clipped surrogate with `eps_clip = 0.2`.
- **Gate:** `g_t = sigmoid(β · Δ_t)`, where `Δ_t = stop_grad(log π_teacher(y_t | s⁺) − log π_student(y_t | s))`.
- **RED-LINE INVARIANT (a graded rubric leaf inspects this):** the gate AND the teacher term are
  **stop-gradient**; the gradient flows **only** through the student log-prob. Getting this wrong is
  the single most common fidelity failure.
- **Gating strategies** (`GATING_STRATEGIES = ("none","entropy","gap","soft_or")`): `entropy` = OPSD
  (`g_t = sigmoid(β·h_t)`), `gap` = the proposed SDAR gate, `soft_or` = RLSD.
- **Constants:** released scripts use **λ=0.01, β=5.0**; the paper text says **λ=0.1, β=10** — a
  documented code-vs-paper discrepancy; either is accepted, but state which you used in provenance.

## Environment & libraries (pinned)
Authoritative list: `backend/requirements-sdar.txt`. Authors train with **verl + Ray + vLLM**.
```
transformers>=4.51.0,<4.58   # Qwen3 needs >=4.51
accelerate>=1.7   datasets>=2.19,<4   sentence-transformers>=3   # E5 retriever
rank-bm25   faiss-cpu>=1.8   # wiki-18 E5 index   scikit-learn   tqdm
alfworld>=0.4.2   textworld>=1.6
```
**Build gotchas** (each was a real debug cycle — see the `gcp-gke-reproduction` skill's CPU-staging
section): accept conda ToS (miniconda ≥25); `apt-get install build-essential` (flash-attn needs
`g++`); `export MAX_JOBS=4` or the flash-attn nvcc build OOMs; `pip install tensordict` (+ verl deps)
for Search-QA; `apt-get install default-jre-headless` for WebShop's Lucene index.

## Datasets (retrieval topology DEPENDS ON MODE — read this before flagging a `:8000` server)
- **Adapt mode (harness-written envs `search_qa_env.py`/`webshop_env.py`/`alfworld_env.py`):
  all retrieval is IN-PROCESS — no HTTP servers.** The bullets below describe this mode.
- **Execute mode (authors' verl repo) DOES launch the authors' own retrieval server** — declared
  as a Search-QA `services[]` entry in `configs/sdar_execute_cells_phase1.json`
  (`retrieval_launch.sh`, readiness probe `http://127.0.0.1:8000/retrieve`). This is **correct,
  not a violation**: the authors' trainer queries that server, and the successful 2026-07-03
  Search-QA-3B run (0.456) used it. Do not "fix" an execute-mode cell by removing its server.
- **ALFWorld:** `pip install alfworld`, set `ALFWORLD_DATA` **before** `alfworld-download`; GiGPO
  train split, 6 task categories (Pick/Look/Clean/Heat/Cool/Pick2).
- **Search-QA:** NQ/HotpotQA/TriviaQA/PopQA/2Wiki/MuSiQue/Bamboogle via HF `datasets`; an **E5
  retriever over the wiki-18 corpus** (Search-R1). The wiki-18 E5 FAISS index is **~70 GB,
  unauthenticated → HF-rate-limited but resumable**; point `SEARCH_QA_INDEX_DIR` at the cached index.
  In adapt mode retrieval runs in-process (no `:8000` server); in execute mode the authors'
  retrieval server on `:8000` is expected (see above).
- **WebShop:** in-process gym `WebAgentTextEnv`/`SimServer` over the pickled product corpus + Lucene
  index — no HTTP server in adapt mode; 128 validation tasks; needs its own py3.10
  (`OPENRESEARCH_WEBSHOP_PYTHON`).

## Training procedure
- Paper compute: 8× H800, **150 steps**, `env.rollout.n=8` (group size G=8), `lr=1e-6`,
  `eps_clip=0.2`, **seed=0** (single seed — released scripts don't multi-seed).
- **Sync policy weights into vLLM after every optimizer step.** Stale vLLM weights make rollouts
  off-policy and collapse the GRPO advantage; using slow HF `model.generate` is what forced prior
  runs to undertrain to ~20 steps.
- **Never `save_pretrained` / `torch.save` per cell** — a multi-cell grid of 15–30 GB checkpoints
  fills the disk. Each cell writes only `metrics.json` + `provenance.json`. (The harness has no
  mid-training checkpoint/resume: an OOM or spot-preempt restarts from step 0.)
- Cost-bounded harness slice: smallest-model-first; per-model steps/group scaled down; grid truncated
  by `OPENRESEARCH_SDAR_MAX_CELLS`. Stage assets on a cheap CPU VM first (`scripts/sdar_cpu_stage.sh`)
  so the paid GPU only trains.

## Evaluation & provenance (the two ways a fake sails through)
Metrics: ALFWorld success %, Search-QA exact-match %, WebShop score (0–100) + accuracy %. Eval must
be on tasks **disjoint from the training rollouts**, scored against gold, `n_eval ≥ 8`, reported as a
**fraction in [0,1]** — use the harness `record_eval(..., held_out=True, train_ids=...)` sidecar.

- **TRAP 1 — `accuracy == reward × 100`:** computing `accuracy = mean_success × 100` on *training*
  rollouts (no held-out eval) turns a true 0.008 into a reported 0.83 that passes the `rate>1` check
  (0.83 ∈ [0,1]). `OPENRESEARCH_EVAL_PROVENANCE_GUARD` vetoes it — never scale success by 100, never
  treat `success ≡ reward`, never eval on training ids.
- **TRAP 2 — all-0.0 / sparse-reward wall:** small models on sparse terminal reward earn ~0 in a
  bounded budget; `L_GRPO` collapses to 0 and the gate never activates. This is honest-inconclusive,
  not success — `ZERO_METRICS_GUARD` + `NO_LEARNING_SIGNAL_GATE` force `inconclusive`, not a fake 0.

Required metrics keys: `per_model`, `baselines_vs_sdar`, `omitted`; every baseline (GRPO, OPSD,
Skill-SD, GRPO+OPSD, RLSD) **and** `sdar` must appear in `per_model` OR in `omitted` with a reason.

## Fidelity traps that fail a faithful-looking run
- **Surrogate TinyLM:** an `nn.Linear`/`nn.Embedding` toy head caps the rubric at ~0.13. Use real
  weights via `AutoModelForCausalLM.from_pretrained("Qwen/…")`; a `class TinyLM` / `# surrogate`
  marker is a hard fidelity violation.
- **Qwen3 model-id:** `Qwen/Qwen3-1.7B-Instruct` **401s (does not exist)** — Qwen3 drops the
  `-Instruct` suffix (`Qwen/Qwen3-1.7B`). Only **Qwen2.5** keeps `-Instruct`. A wrong id passes
  preflight then 401s at runtime (hollow success).
- **ALFWorld turn horizon:** an ALFRED task is 15–50 primitives; `max_turns=6` guarantees `won=False`.
  Use `max_turns ≥ 30`.
- **OOM:** fix via smaller micro-batch / lower `rollout.gpu_memory_utilization` / optimizer
  offload / gradient checkpointing — **never** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`:
  vLLM's `CuMemAllocator` raises `AssertionError: Expandable segments are not compatible with
  memory pool` at model load, so the run dies before training. This is why
  `gpu_cell_runner.py` exempts command cells (execute mode) from its
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` default — the authors' launcher owns its
  own CUDA memory config.
- **WebShop fake-zero:** a dead env (0 episodes served) must become a verified `env_setup_failed`
  exclusion (`ENV_LIVENESS_GATE` → `env_health.jsonl`), never a real-looking 0.
- **Preflight false-block ≠ fake impl:** SDAR's real code lives across `train_cell.py` /
  `sdar_loss.py` / `build_cells.py`; a single-file preflight scan false-blocked a *faithful* impl for
  3 cycles. `PREFLIGHT_UNION_SCOPE` + `IMPL_ABANDON_GUARD` fix it — enable them in the run-spec.

## Reproduction checklist
1. Seed/adapt the authors' verl trainer (execute mode); do not hand-roll GRPO.
2. Loss = GRPO + λ·SDAR with the **stop-gradient gate**; pick λ/β and record which.
3. Real Qwen weights (correct id: Qwen3 no `-Instruct`); `max_turns ≥ 30` for ALFWorld.
4. Adapt mode: in-process ALFWorld/WebShop/Search-QA; execute mode: authors' retrieval server
   on `:8000` for Search-QA. wiki-18 E5 index cached; vLLM weight-sync every step.
5. Held-out disjoint eval via `record_eval(..., held_out=True)`; metrics as [0,1] fractions.
6. Emit `per_model`/`baselines_vs_sdar`/`omitted`; each baseline + `sdar` present or omitted-with-reason.

## Sources (repo-native grounding — not vendored)
`configs/papers/2605.15155.yaml`, `docs/runbooks/2026-05-23-sdar-baseline-handoff.md`,
`backend/agents/prompts/paper_hints.py`, `backend/requirements-sdar.txt`, the faithful impl under
`runs/prj_23f04429cd3beaf7/code/`. For the GKE/GPU execution path see the `gcp-gke-reproduction` skill.
