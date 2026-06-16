# TRL GRPOTrainer + vLLM RL Scaffold — Implementation Spec (for codex)

**Date:** 2026-05-30
**Branch/worktree:** `feat/rl-trl-grpo-engine` @ `/home/sww35/openresearch-rl-trl`
**Status:** Design locked (grilled with the user 2026-05-30); implementation pending
**Implementer:** codex (this spec is your full context — read it end to end before coding)

---

## 0. Mission

Build a **harness-owned, reusable RL-training scaffold** that replaces the fragile
hand-rolled-FSDP + `model.generate()` pattern for reinforcement-learning paper
reproductions. The scaffold owns the distributed-RL infra — **TRL `GRPOTrainer`
+ vLLM rollouts + the collective discipline** — so the agent only injects the
paper-specific reward and custom-loss term. **SDAR (arXiv 2605.15155)** and its
sigmoid-gated OPSD term are the first instantiation.

This is a **parallel track**. It must NOT disturb the existing hardened
`accelerate`+FSDP default path (it is **opt-in** behind a flag). The main line
will keep reproducing SDAR on hardened-FSDP; this scaffold is validated
separately and integrated once proven.

### Why this exists (the motivating failure)
Run `prj_09047604e591d969` (SDAR smallest-two, 4×A5000) crashed: the agent's
hand-written `train.py` ran a zero-shot `generate()` **only on rank 0**
(`if accelerator.is_main_process:`) before `accelerator.wait_for_everyone()`.
After `accelerator.prepare()` the model is FSDP-sharded, so `generate()` is a
**collective** — ranks 1-3 skipped it and blocked at the barrier → 600 s NCCL
timeout → C++ `terminate()` → SIGABRT (exit -6). Plus HF `generate()` rollouts
are painfully slow. A vetted scaffold that owns the generation/training split
makes this bug class **structurally impossible** and 10-50× faster rollouts.

---

## 1. Locked design decisions (from the grill)

| # | Decision | Choice | Implication |
|---|----------|--------|-------------|
| 1 | **Deliverable shape** | **Harness-owned scaffold** | A copyable module (mirrors the `rubric_guard.py` "paste verbatim into code/" pattern) the agent imports; NOT pure guidance, NOT a new primitive. |
| 2 | **Generality** | **General GRPO scaffold, OPSD-first** | A general `GRPOTrainer`+vLLM scaffold with a **pluggable custom-loss hook**; SDAR's OPSD is the first implementation. Reusable for other GRPO/PPO papers. |
| 3 | **vLLM placement** | **Separate-GPU server** | vLLM runs on dedicated leased GPU(s); FSDP trainer on the rest; concurrent gen+train. The scaffold OWNS a multi-process launch (vLLM server + `accelerate launch` trainer) and must coexist with / suppress the harness's generic FSDP rewriter. |
| 4 | **OPSD injection** (default) | **Subclass `GRPOTrainer.compute_loss`** | Expose a `custom_loss_term(...)` hook; total loss = `L_GRPO + λ·L_custom`. SDAR fills it with OPSD; the gate/β/λ constants appear **literally** (rubric reads them). |
| 5 | **Agent-flow** (default) | **Opt-in for RL papers** | A new guidance block tells the agent, *for RL papers when the flag is set*, to copy the scaffold + fill the hooks. Default OFF → hardened-FSDP path is unchanged. |
| 6 | **Validation** (default) | **Unit tests + tiny 1-GPU smoke ONLY** | Do NOT launch a full SDAR training run (GPU etiquette — shared box). The maintainer integration-tests SDAR on the main line. |

---

## 2. Hard constraints (non-negotiable)

- **GPU etiquette (CRITICAL):** the box (`sun.cs.txstate.edu`, 8×A5000, kernel
  5.4) is shared with users `zby22` and `bgu9`. **NEVER** touch their GPUs or
  processes. Only use GPUs reported free by
  `backend/services/runtime/local_gpu_allocator.py::free_devices(...)`. Your
  smoke test must run on **one** free GPU (or CPU) — never grab the box.
- **Do not disturb the FSDP default.** The scaffold path is opt-in
  (`OPENRESEARCH_RL_SCAFFOLD=1` or a per-run flag). With it off, behavior is
  byte-identical to today. All existing tests must stay green.
- **Coexist with the distributed-launch rewriter.** See §5 — the separate-server
  topology conflicts with `_resolve_distributed_launch`; you must add a clean
  opt-out so the scaffold owns its own launch.
- **Copyable, low-dep scaffold.** The scaffold's *orchestration* logic is pasted
  into `code/` (like `rubric_guard.py`); its heavy deps (`trl`, `vllm`) go in
  the agent's `requirements.txt` / per-run venv, NOT the harness `.venv`.
- **Paper constants literal.** SDAR rubric leaves read the source for
  `g_t = σ(β·Δ_t)`, `β=10`, `λ=0.1`, stop-gradient on the gate — these must
  appear verbatim in the emitted code.
- **Kernel-5.4 NCCL:** multi-proc launches need `NCCL_P2P_DISABLE=1
  NCCL_IB_DISABLE=1` (the harness sets this via `_nccl_env_prefix()`; preserve it).
- **torch 2.5.1** on this host → **FSDP1** (FSDP2 needs torch≥2.6). Per-run venvs
  may install newer torch if a paper needs it, but don't assume FSDP2.

---

## 3. Integration map (verified — file:line)

### 3.1 Agent code-gen → execution chain
- `backend/agents/rlm/primitives.py:1322` `implement_baseline(plan, *, ctx)` →
  spawns the code-writing agent; returns `{ok, code_path, files}`.
- `backend/agents/baseline_implementation.py:1956` `run_with_sdk(...)` — the
  agent writes `code/` + `commands.json`. **Guidance assembled in
  `_compute_constraint_guidance` (~line 1712), concatenation at lines 1794-1857.**
- `backend/agents/rlm/primitives.py:3451` `run_experiment(code_path, env_id, ...)`
  → `_resolve_distributed_launch(...)` (line ~3654) → `_execute_in_sandbox`
  (line 2444) reads `commands.json`, runs each command, reads back `metrics.json`.

### 3.2 The copyable-module pattern (your template)
- `backend/agents/rlm/rubric_guard.py` — zero-dep module pasted verbatim into
  `code/rubric_guard.py`. Guidance directive: `_RUBRIC_GUARD_BLOCK`
  (`baseline_implementation.py:1000-1033`). **Model your scaffold's copy-in block
  on this.** `class RubricGuardFailure(AssertionError)` (line 60),
  `assert_metrics_schema(metrics, required_keys, required_artifacts, artifact_dir)`.

### 3.3 Distributed-launch rewriter (the coexistence point)
- `backend/agents/rlm/primitives.py:2353` `_resolve_distributed_launch(commands,
  code_dir, ngpu, run_id)`. Markers `_DISTRIBUTED_MARKERS` (line ~2247) include
  `from accelerate`, `Accelerator(`, `torch.distributed`, `fully_shard`, etc.
- Rewrite (ngpu≥2 + marker): `python train.py …` →
  `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch --config_file
  _reprolab_fsdp.yaml --num_processes <ngpu> --num_machines 1
  --main_process_port <free> train.py …` (template ~line 2416).
- `_write_fsdp_accelerate_config` (line 2278), `_nccl_env_prefix` (line 2329).
- **Problem:** with separate-server vLLM, wrapping the WHOLE `train.py` in
  `accelerate launch --num_processes N` would spawn N copies that each try to
  start a vLLM server and fight over GPUs. **You need an opt-out** (see §5).
- Tests to mirror: `tests/agents/rlm/test_distributed_launch.py:34-167`.

### 3.4 Per-run venv + deps
- `scripts/batch_reproduce.py:306-367` — creates `runs/<id>/.venv`, shared
  `HF_HOME`/`PIP_CACHE_DIR` (`runs/.cache/...`), per-run torch/triton/xdg/mpl
  caches, then `python -m backend.cli reproduce … --project-id <id>`. The agent's
  `code/requirements.txt` is installed here. **`trl`+`vllm` go in that
  requirements.txt.** Heads-up: vLLM pins specific torch/CUDA versions — pin a
  known-compatible (trl, vllm, torch) triple and document it; this is a real
  fragility surface.

### 3.5 GPU leasing
- `backend/services/runtime/local_gpu_allocator.py`: `discover_gpus()` (~157),
  `free_devices(devices, free_mem_threshold_mb=1024, own_pids=())` (~243),
  `LocalGpuAllocator.acquire(lease_id, pid, count)` / `release(lease_id)`
  (file-locked). The run leases ALL its GPUs together; `CUDA_VISIBLE_DEVICES` is
  set to the leased UUIDs. **Your scaffold partitions those visible devices
  internally** (e.g. device 0 → vLLM server, 1..N → FSDP trainer).

---

## 4. SDAR / OPSD specifics (the first loss-hook instantiation)

From `docs/papers/2605.15155.yaml` `algorithm_invariants`:
```
gate_formula: g_t = sigmoid(beta * Delta_t),
  Delta_t = log π_student(y_t|x_t) − log π_teacher(y_t|x_t, c_priv)
stop_gradient_on_gate: true
loss: L = L_GRPO + lambda * L_OPSD
lambda: 0.1   beta: 10
divergence: reverse KL (mode-seeking)
gating_strategy: gap (Teacher-Student gap on the student-sampled token)
```
Rubric leaves (`runs/prj_09047604e591d969/generated_rubric.json`) grade the
literal presence of: the sigmoid gate on the **detached** T-S log-prob gap
(stop-grad so gradient flows only through the student log-prob); the composite
`L = L_GRPO + 0.1·L_OPSD`; GRPO with IS ratio + PPO clip ε + group-relative
advantages over G samples; and the five baselines (GRPO, OPSD, Skill-SD,
GRPO+OPSD, RLSD).

**Required metrics keys** (emit these in `metrics.json`, `per_model` shape):
`alfworld_success_rate_per_model`, `searchqa_em_per_model`,
`webshop_score_per_model`, `per_model`, `baselines_vs_sdar`, `omitted`.
Smallest-two scope = Qwen3-1.7B + Qwen2.5-3B, Search-QA only (declare ALFWorld /
WebShop / 7B in `omitted`). The teacher and student are the **same** Qwen weights
under different contexts (self-distillation) — load real weights, not a surrogate.

---

## 5. The launch-orchestration design you must build (separate-server)

The scaffold owns a 2-tier launch inside the sandbox:
1. **vLLM rollout server** on dedicated free GPU(s): `trl vllm-serve` (or the
   programmatic equivalent) bound to a partition of `CUDA_VISIBLE_DEVICES`.
2. **FSDP trainer** on the remaining GPUs via `accelerate launch` with a config
   the scaffold writes (NOT `_reprolab_fsdp.yaml` — its own, FSDP1, bf16,
   matched to the trainer GPU count), pointed at the vLLM server URL.

**Coexistence with `_resolve_distributed_launch` (pick the cleanest, justify in
the PR):**
- Preferred: have the scaffold emit its launch command already in non-`python
  train.py` form (e.g. the agent's `commands.json` runs `python rl_launch.py`,
  an orchestrator that starts vLLM then `accelerate launch`s the trainer), and
  add a **single guard** in `_resolve_distributed_launch` that **skips rewriting
  when a scaffold sentinel is present** (e.g. a `# reprolab:rl-scaffold-owns-launch`
  marker in the command or a `code/.reprolab_rl_scaffold` file, or
  `OPENRESEARCH_RL_SCAFFOLD=1`). Add a focused test for the skip.
- Keep the NCCL env prefix applied to the trainer launch (reuse
  `_nccl_env_prefix()`).

Weight sync: after each optimizer step, push updated policy weights into the
vLLM server (TRL's GRPO server mode handles this; verify the path works on
FSDP-sharded weights — gather then push).

---

## 6. Deliverables

1. **`backend/agents/rlm/rl_scaffold.py`** — the scaffold module (copyable into
   `code/`). Public API (sketch — refine):
   - `class GRPOScaffold` wrapping TRL `GRPOTrainer`, configured with model(s),
     reward fn, vLLM-server endpoint, FSDP/accelerate config, and a
     `custom_loss_term` hook.
   - `compute_loss` override: `loss = grpo_loss + lambda * custom_loss_term(...)`.
   - Metrics emission following `_EAGER_METRICS_BLOCK` (atomic incremental writes
     + terminal flush; `per_model` populated with measured eval metrics).
   - Calls `assert_metrics_schema` (rubric_guard) at the end.
2. **`rl_launch.py` template** (or a function the agent's thin `train.py` calls)
   — the vLLM-server + accelerate-trainer orchestrator (§5).
3. **Guidance block** `_RL_SCAFFOLD_BLOCK` in
   `backend/agents/baseline_implementation.py`, injected **only** for RL papers
   when opt-in is set, that tells the agent to: copy `rl_scaffold.py` verbatim,
   write a thin `train.py` that configures `GRPOScaffold` + defines the reward +
   the `custom_loss_term` (OPSD for SDAR, with the literal constants), and
   declare deps in `requirements.txt` (pinned trl/vllm/torch).
4. **Rewriter opt-out** in `_resolve_distributed_launch` (§5) + test.
5. **SDAR OPSD example** — a reference `custom_loss_term` implementing
   `g_t=σ(β·Δ_t)`, stop-grad, `λ=0.1`, reverse-KL, that the guidance points to
   (and that the rubric's literal-constant leaves would pass).
6. **Tests** (`tests/agents/rlm/test_rl_scaffold.py` + launch test): scaffold API;
   the OPSD loss math on tiny tensors (gate detached, λ applied, shapes); the
   rewriter skip-on-sentinel; metrics schema; opt-in OFF → no behavior change.
   Plus a **tiny 1-GPU (or CPU) smoke** gated behind an env flag so CI/devs can
   run a 1-step GRPO on a tiny model WITHOUT the full box.

---

## 7. Out of scope / future
- Multi-node and 7B+ scale → **verl / OpenRLHF** (Ray + vLLM + FSDP/Megatron).
  Note as the scaling path; do not build now.
- Colocated vLLM mode — the user chose separate-server; colocated can be a future
  `vllm_mode` option.
- Making the scaffold the default for RL papers — stays opt-in until the
  maintainer's main-line SDAR integration run validates it.

---

## 8. Definition of done
- `rl_scaffold.py` + `_RL_SCAFFOLD_BLOCK` + rewriter opt-out implemented.
- New tests pass; **full existing suite stays green** (`.venv/bin/python -m
  pytest tests/ -q` from the main checkout, or the worktree with the shared
  venv).
- A tiny 1-GPU/CPU smoke runs a 1-step GRPO on a tiny model (e.g.
  `HuggingFaceTB/SmolLM-135M` or similar) — proving the gen→loss→step→metrics
  path — **without** touching others' GPUs.
- A short `README`/PR note: the pinned (trl, vllm, torch) triple, how the opt-in
  flag works, and how the maintainer flips it on for a real SDAR run.
- Do NOT run a full SDAR training; hand back for main-line integration.
```
