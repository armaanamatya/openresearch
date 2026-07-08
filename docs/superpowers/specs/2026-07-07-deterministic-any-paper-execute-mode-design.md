# Deterministic any-paper execute-mode reproduction — design

> **Doc status:** Draft · spec tier · authored 2026-07-07. Grounded by two read-only recon
> passes over the live `feat/gke-gpu-path-reproduction-reliability` branch + the UCPO
> (arXiv 2605.00365) prove-now run. Policy: `docs/policies/documentation.md`.

## 1. Problem

When a paper ships an official repo, reproduce it by **running the authors' code** — for
**any** paper, dynamically — never a per-paper hack and never an LLM re-implementation.

Today (recon, 2026-07-07): repo-first + `execute` mode are **live and real** (clone → seed
`code/` verbatim → `gpu_cell_runner` runs `cell["command"]` → `verl_metrics_adapter` bridges
the reward). But there is **one structural hole**: even in execute mode the executor SDK
agent is still asked to hand-author the cell shim (`run_with_sdk`), so a wandering model can
re-implement instead of run — exactly what sank the UCPO run (it degenerated to reward 0.0
by rewrapping the repo). Two secondary gaps: execute mode is not the default when a repo
clones, and the base GPU image can't host framework stacks (verl/vLLM) — a runtime
`pip install -e verl` cascades torch to an incompatible build.

## 2. Goal / non-goals

**Goal:** a deterministic, framework-agnostic path that, from a cloned repo, produces a
runnable `code/cells.json` + `code/train_cell.py` that executes the **authors' own launch
entrypoint** (downscaled to available compute) and extracts the **authors' own reward** —
with **no LLM in the code-writing path**. Modular: adding a framework = adding one adapter.

**Non-goals (this spec):** replacing the LLM implement path for scratch/adapt modes;
private-repo auth; multi-node training; changing the cell-runner/capacity machinery;
the evidence/verdict layer (unchanged — reward extraction stays value-preserving).

## 3. Architecture

```
repo/ (cloned, execute mode)
  └─▶ execute_planner.plan(repo_dir, claim_map) ─────────────▶ ExecuteSpec
        │  deterministic detection FIRST, bounded-LLM extraction only to
        │  RESOLVE ambiguity (reads README/scripts → STRUCTURED facts; never writes code)
        ▼
   framework adapter registry  {verl, hf_trainer, accelerate, python, bash}
        │  each adapter: detect() · downscale() · reward_spec()
        ▼
   execute_cell_synth.synthesize(ExecuteSpec, code_dir) ─────▶ code/execute_spec.json
        │  deterministic (mirror of gke_cell_synth)              code/cells.json
        │                                                        code/train_cell.py (GENERIC shim)
        ▼
   framework→image selection  (verl → gke-cell-verl · else → gke-cell-base)
        ▼
   run_experiment cell route (EXISTING) → GKE pod → reward via the spec's adapter
```

The **generic `train_cell.py` shim is identical for every paper** — it reads
`execute_spec.json` and executes it. Only the spec differs. That is the modular core.

## 4. Contracts

```python
@dataclass(frozen=True)
class LaunchSpec:
    kind: str            # "module" | "script" | "shell"
    command: str         # "python -m ucpo.main_run <args>" | "bash scripts_c/run_ucpo_1.5b.sh"
    cwd: str             # relative to code/ ("" | "verl")
    overrides: dict      # framework knob → downscaled value (applied deterministically)

@dataclass(frozen=True)
class RewardSpec:
    kind: str            # "verl" | "hf_trainer" | "json_file" | "log_regex"
    keys: tuple[str, ...]# ordered reward-key candidates, first found wins
    log_glob: str        # "$OUTPUT_DIR/*.log"
    metrics_file: str | None  # for json_file: "all_results.json"

@dataclass(frozen=True)
class ExecuteSpec:
    framework: str       # verl | hf_trainer | accelerate | python | bash
    setup: tuple[str, ...]     # ["pip install -e verl --no-deps"]
    launch: LaunchSpec
    reward: RewardSpec
    image_key: str       # "verl" | "base"
    est_vram_gb: float
    confidence: float
    source: str          # "deterministic" | "hybrid" | "llm"
    reason: str
```

`ExecuteSpec` is persisted to `code/execute_spec.json` — the deterministic source of truth
the generic shim reads (never the root's untrusted plan).

## 5. Detection (deterministic-first, bounded-LLM to break ties)

1. **Framework** (`framework_detector.py`, pure): fingerprints — `verl` (imports `verl`,
   hydra `actor_rollout_ref`, bundled `verl/setup.py`); `hf_trainer`
   (`transformers.Trainer`/`TrainingArguments`); `accelerate` (`accelerate launch` in
   scripts); else `python`/`bash`. Returns `(framework, confidence)`.
2. **Setup**: parse README "Setup/Installation" fenced blocks; rewrite editable framework
   installs to `--no-deps` (the image owns the heavy stack — §6). Fallback root
   `requirements.txt` (framework deps excluded to avoid the torch cascade).
3. **Launch**: deterministically enumerate `scripts*/`, `examples/`, root `run*.sh`/
   `train*.sh` + README "Training/Usage/Run" fenced blocks; a launch script's real command
   (e.g. `run_ucpo_1.5b.sh` → `python3 -m ucpo.main_run …`) is extracted by parsing the
   script. **Bounded-LLM (flag-gated, fail-soft)**: only when >1 candidate or ambiguous, ONE
   `ctx.llm_client` call reads README + candidate scripts and returns *which* is canonical +
   its command as STRUCTURED output (mirrors `skill_selection`'s deterministic-recall →
   bounded-LLM-prune). It can never emit code — only pick/copy an existing command.
4. **Reward** (framework adapter): verl → `("critic/rewards/mean","val/acc/mean",…)`;
   hf_trainer → `json_file` `all_results.json` key `eval_*`; generic → the authors' own
   eval output. Value-preserving; fail-honest when absent.
5. **Downscale** (framework adapter): verl → `trainer.n_gpus_per_node=N`,
   `tensor_model_parallel_size=1`, proportional `train_batch_size`/`ppo_mini_batch_size`,
   `gpu_memory_utilization=0.6`, `enable_gradient_checkpointing=True`, `param_offload=True`,
   optional tiny-slice for a fast proof; hf_trainer → `--nproc_per_node=N`,
   `--per_device_train_batch_size`, `--max_steps`. Downscaling is the adapter's contract.

## 6. Framework → image

A framework needing a heavy CUDA stack maps to a validated pre-baked image, so the cell
never runtime-installs torch: `verl → gke-cell-verl:v1` (torch 2.6.0/cu124 + vLLM 0.8.5 +
tensordict 0.6.2 + math-verify; `docker/gke-cell-verl/`), everything else →
`gke-cell-base:v1`. Selection sets the Job image (`OPENRESEARCH_GCP_BASE_IMAGE` per-run or a
`config` framework→image map). The bundled framework is `pip install -e <dir> --no-deps` at
cell runtime (code-only).

## 7. Wiring + flags (all default-OFF; byte-identical off)

- **`OPENRESEARCH_EXECUTE_SYNTH`** (master) — in `implement_baseline`, execute mode, after the
  repo→`code/` seed: run `execute_planner` + `execute_cell_synth`. On a confident ExecuteSpec,
  **short-circuit `run_with_sdk`** (closes the "LLM authors the shim" hole). Low
  confidence / failure → fall through to today's LLM execute path (graceful degradation).
- **`OPENRESEARCH_EXECUTE_SYNTH_LLM`** — allow the bounded-LLM tie-break in detection (off →
  deterministic-only; a repo whose entrypoint isn't deterministically resolvable falls
  through to the LLM execute path).
- **`OPENRESEARCH_EXECUTE_DEFAULT`** — when `repo_spec.clone_succeeded` and a launch
  entrypoint is detected, default the reproduction mode to `execute` (so "use the repo" is
  automatic — the reason UCPO silently ran `adapt`). `assert_execute_mode_stamped` backstop
  unchanged.
- Framework→image map in `config.py` (verl→gke-cell-verl); off → today's single base image.

## 8. Evidence red line (unchanged)

Reward extraction is **always value-preserving** — the generic shim's reward bridge dispatches
only to value-preserving adapters (`verl_metrics_adapter` etc.), copies the authors' own
number, writes an `eval_provenance.json` sidecar, and fails honest (`{"status":"failed"}`)
when no reward is found — never a fabricated 0.0. The fitness signal stays the deterministic
evidence layer, never the LLM. All existing guards (zero-metrics, degenerate-training,
eval-provenance) remain the floor.

## 9. Rollout / testing

- Unit (pure, no network): `framework_detector` fingerprints; `execute_planner` on fixture
  repos (verl/HF); `scaling` downscale math; `execute_cell_synth` output shape.
- Wiring: execute-mode `implement_baseline` short-circuits `run_with_sdk` on a confident spec;
  falls through on low confidence; off-state byte-identical (no synth, LLM path unchanged).
- Reward: generic shim bridges a synthetic verl log → non-degenerate `metrics.json`.
- E2E (operator, GPU): **UCPO** (the prove-now control) + **SDAR** both on `gke-cell-verl`,
  `execution.ran=true` + a real non-zero reward, before any default flip.
- Increment: ship the **verl adapter first** (UCPO-validated), registry open for hf_trainer /
  accelerate as pure follow-on adapters — no core change.

## 10. Reference implementation

The hand-authored `scripts/ucpo_execute_cell/train_cell.py` (the prove-now control) is the
concrete verl-adapter reference: setup `pip install -e verl --no-deps`, launch
`python -m ucpo.main_run` with the downscaled hydra overrides, reward via
`verl_metrics_adapter(success_rate_key="critic/rewards/mean")`. `execute_cell_synth` emits
exactly this shape from an `ExecuteSpec`, parameterized — not UCPO-specific.
