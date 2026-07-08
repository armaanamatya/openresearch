#!/usr/bin/env python3
"""Deterministic execute-mode cell — run the UCPO authors' pipeline VERBATIM on 1xA100.

Hand-authored proof (Track A) that the author repo produces a real RLVR reward when
RUN, not re-implemented. It runs the authors' own entrypoint (`python -m ucpo.main_run`,
the command from scripts_c/run_ucpo_1.5b.sh) downscaled to a single A100-80 and a tiny
train slice for a fast NON-ZERO-reward proof, then bridges the authors' OWN reward into
the canonical metrics.json via the existing verl_metrics_adapter (value-preserving,
fail-honest — never a fabricated 0.0).

This file is the template for the Track-B deterministic execute-cell synthesizer.

Cell contract (from gke_cell_entrypoint.py / k8s_job_cell_runner):
  invoked as `python train_cell.py --cell-id=<id> --output-dir=<dir>`; also reads
  OPENRESEARCH_CELL_OUTPUT_DIR. Writes <output_dir>/metrics.json. Locates sibling
  modules via Path(__file__).parent (never cwd). Never raises out of main.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent

# Candidate reward keys verl logs, most-specific first. The authors' fn_score returns
# {"score":1,"acc":1} on a correct boxed answer; verl aggregates that into these console
# metrics. We take the FIRST key the adapter can actually find in the logs.
_REWARD_KEYS = (
    "critic/rewards/mean",
    "critic/score/mean",
    "reward/mean",
    "critic/rewards/mean/all",
)


def _resolve_output_dir(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)
    for key in ("OPENRESEARCH_CELL_OUTPUT_DIR", "OUTPUT_DIR", "OPENRESEARCH_ARTIFACT_DIR"):
        val = os.environ.get(key)
        if val:
            return Path(val)
    return CODE_DIR / "cell_output"


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> int:
    with open(log, "ab") as fh:
        fh.write(f"\n$ (cwd={cwd}) {' '.join(cmd)}\n".encode())
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT)
        return proc.wait()


def _write_metrics(out_dir: Path, payload: dict) -> None:
    tmp = out_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, out_dir / "metrics.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-id", default=os.environ.get("OPENRESEARCH_CELL_ID", "ucpo-execute"))
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    out_dir = _resolve_output_dir(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "train.log"

    def fail(reason: str, extra: dict | None = None) -> int:
        payload = {"status": "failed", "reason": reason}
        if extra:
            payload.update(extra)
        _write_metrics(out_dir, payload)
        print("CELL FAILED:", reason, flush=True)
        return 41

    env = os.environ.copy()
    env["OUTPUT_DIR"] = str(out_dir)
    env["OPENRESEARCH_CELL_OUTPUT_DIR"] = str(out_dir)
    env["MAX_JOBS"] = os.environ.get("MAX_JOBS", "4")          # cap native-ext compile OOM
    env["HYDRA_FULL_ERROR"] = "1"
    env.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")        # avoid flash-attn compile
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    slice_rows = int(os.environ.get("UCPO_SLICE_ROWS", "32"))
    train_bs = int(os.environ.get("UCPO_TRAIN_BS", "8"))

    # (1) Install the BUNDLED verl (code only) with --no-deps: the gke-cell-verl
    # image already owns the VALIDATED heavy stack (torch 2.6 / vLLM 0.8.5 /
    # tensordict 0.6.2 / math-verify). A full `pip install -e .` would re-resolve
    # verl's deps and CASCADE torch to a mismatched build — the exact failure that
    # crashed the raw-CUDA base. --no-build-isolation reuses the image's setuptools
    # + torch at build time (verl's setup.py imports torch).
    verl_dir = CODE_DIR / "verl"
    if not (verl_dir / "setup.py").exists() and not (verl_dir / "pyproject.toml").exists():
        return fail("bundled verl/ is not a valid pip project in code/")
    rc = _run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "--no-build-isolation"],
        cwd=verl_dir, env=env, log=log,
    )
    if rc != 0:
        return fail("verl editable install (--no-deps) failed (see train.log)")

    # (2) Smallest-slice: cap the 10k train set to a handful of rows for a fast proof.
    train_full = CODE_DIR / "dataset" / "train_data_10k.parquet"
    val_full = CODE_DIR / "dataset" / "valid_data.parquet"
    if not train_full.exists():
        return fail("dataset/train_data_10k.parquet missing in code/")
    slice_parquet = out_dir / "train_slice.parquet"
    try:
        import pandas as pd
        pd.read_parquet(train_full).head(slice_rows).to_parquet(slice_parquet)
    except Exception as exc:  # noqa: BLE001
        return fail(f"could not build train slice: {exc}")

    # (3) Run the authors' entrypoint VERBATIM (downscaled to 1xA100).
    model = os.environ.get("UCPO_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    argv = [
        sys.executable, "-m", "ucpo.main_run",
        "algorithm.adv_estimator=iq", "algorithm.tau=0.2", "algorithm.alpha=0.0",
        f"data.train_files={slice_parquet}", f"data.val_files={val_full}",
        f"data.train_batch_size={train_bs}",
        # Keep the authors' 3072 response budget: R1-Distill emits long CoT and a
        # 1024 cap truncates before the boxed answer -> math-verify scores ~0 ->
        # a FALSE degenerate signal. 3072 fits a 1.5B model on one A100-80.
        "data.max_prompt_length=1024",
        f"data.max_response_length={os.environ.get('UCPO_MAX_RESPONSE', '3072')}",
        "data.filter_overlong_prompts=True", "data.truncation=error",
        f"actor_rollout_ref.model.path={model}",
        "actor_rollout_ref.actor.optim.lr=5e-6",
        "actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={train_bs}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.use_kl_loss=True", "actor_rollout_ref.actor.kl_loss_coef=0",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "actor_rollout_ref.rollout.n=4",
        "actor_rollout_ref.rollout.n_low=4", "actor_rollout_ref.rollout.n_high=4",
        "actor_rollout_ref.rollout.n_update=0",
        "actor_rollout_ref.rollout.temperature=1",
        # match the authors' explicit scheduler settings (fidelity)
        "actor_rollout_ref.rollout.enable_temperature_scheduler=False",
        "actor_rollout_ref.rollout.enable_annealing=False",
        "actor_rollout_ref.rollout.max_steps=4",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0", "trainer.logger=[console]",
        "trainer.project_name=UCPO", "trainer.experiment_name=ucpo_execute_proof",
        "trainer.n_gpus_per_node=1", "trainer.nnodes=1",
        f"trainer.default_local_dir={out_dir}/checkpoint",
        "trainer.save_freq=-1", "trainer.test_freq=-1",
        "trainer.total_epochs=1",
    ]
    train_rc = _run(argv, cwd=CODE_DIR, env=env, log=log)

    # (4) Bridge the authors' OWN reward -> metrics.json (value-preserving, fail-honest).
    sys.path.insert(0, str(CODE_DIR))
    try:
        from verl_metrics_adapter import write_cell_metrics_from_verl
    except Exception as exc:  # noqa: BLE001
        return fail(f"verl_metrics_adapter import failed: {exc}", {"train_rc": train_rc})

    metrics: dict = {"status": "failed"}
    for key in _REWARD_KEYS:
        metrics = write_cell_metrics_from_verl(
            out_dir, model_key="default", env="math", baseline="ucpo_iq",
            log_glob=str(out_dir / "train.log"), success_rate_key=key,
        )
        if metrics.get("status") == "success":
            metrics["reward_key"] = key
            break

    if metrics.get("status") == "success" and "success_rate" in metrics:
        val = metrics["success_rate"]
        # headline + non-degenerate signals the harness guards read
        metrics["metric"] = val
        metrics["reward"] = val
        metrics["reward_mean"] = val
        metrics.setdefault("train_steps", max(1, slice_rows // train_bs))
        metrics["train_rc"] = train_rc
        _write_metrics(out_dir, metrics)
        print("CELL METRICS:", json.dumps(metrics), flush=True)
        return 0

    # No reward found — surface the log tail so the exact reward key can be fixed.
    tail = ""
    try:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
    except OSError:
        pass
    return fail(
        "ucpo.main_run produced no readable reward key (see train.log)",
        {"train_rc": train_rc, "tried_keys": list(_REWARD_KEYS), "log_tail": tail},
    )


if __name__ == "__main__":
    raise SystemExit(main())
