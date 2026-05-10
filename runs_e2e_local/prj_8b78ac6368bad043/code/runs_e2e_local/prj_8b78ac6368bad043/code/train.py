"""
Top-level entrypoint that drives the full PPO reproduction sweep
(Schulman et al. 2017 — Sections 6.1 & 6.4).

Reads `config.json`, runs all (env x seed) combinations sequentially,
and finally invokes evaluate.py to compute normalized scores.

Usage:
    python train.py --config config.json
    python train.py --config config.json --smoke   # ~2 min sanity run
    python train.py --config config.json --suite mujoco   # phase 1 only
    python train.py --config config.json --suite atari    # phase 2 only
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_subprocess(cmd, env=None):
    print("\n" + "=" * 80)
    print("$ " + " ".join(cmd))
    print("=" * 80, flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, env=env)
    dt = time.time() - t0
    print(f"[exit={rc} dt={dt:.1f}s] " + " ".join(cmd), flush=True)
    return rc


def build_mujoco_cmd(env_id, seed, hp, output_dir, total_timesteps=None):
    return [
        sys.executable, "ppo_continuous_action.py",
        "--env-id", env_id,
        "--seed", str(seed),
        "--total-timesteps", str(total_timesteps or hp["total_timesteps"]),
        "--learning-rate", str(hp["learning_rate"]),
        "--num-envs", str(hp["num_envs"]),
        "--num-steps", str(hp["num_steps"]),
        "--num-minibatches", str(hp["num_minibatches"]),
        "--update-epochs", str(hp["update_epochs"]),
        "--clip-coef", str(hp["clip_coef"]),
        "--gamma", str(hp["gamma"]),
        "--gae-lambda", str(hp["gae_lambda"]),
        "--ent-coef", str(hp["ent_coef"]),
        "--vf-coef", str(hp["vf_coef"]),
        "--max-grad-norm", str(hp["max_grad_norm"]),
        "--anneal-lr", str(hp.get("anneal_lr", True)),
        "--output-dir", output_dir,
        "--track", "False",
    ]


def build_atari_cmd(env_id, seed, hp, output_dir, total_timesteps=None):
    return [
        sys.executable, "ppo_atari.py",
        "--env-id", env_id,
        "--seed", str(seed),
        "--total-timesteps", str(total_timesteps or hp["total_timesteps"]),
        "--learning-rate", str(hp["learning_rate"]),
        "--num-envs", str(hp["num_envs"]),
        "--num-steps", str(hp["num_steps"]),
        "--num-minibatches", str(hp["num_minibatches"]),
        "--update-epochs", str(hp["update_epochs"]),
        "--clip-coef", str(hp["clip_coef"]),
        "--gamma", str(hp["gamma"]),
        "--gae-lambda", str(hp["gae_lambda"]),
        "--ent-coef", str(hp["ent_coef"]),
        "--vf-coef", str(hp["vf_coef"]),
        "--max-grad-norm", str(hp["max_grad_norm"]),
        "--anneal-lr", str(hp.get("anneal_lr", True)),
        "--output-dir", output_dir,
        "--track", "False",
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--smoke", action="store_true",
                   help="Run a 50k-step smoke test on one env per suite")
    p.add_argument("--suite", choices=["mujoco", "atari", "all"], default="all")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(args.results_dir, "run_log.txt")

    failures = []

    # Phase 1: MuJoCo
    if args.suite in ("mujoco", "all"):
        suite_cfg = cfg["suites"]["mujoco"]
        out_dir = os.path.join(args.results_dir, "mujoco")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        envs = suite_cfg["envs"][:1] if args.smoke else suite_cfg["envs"]
        seeds = [suite_cfg["seeds"][0]] if args.smoke else suite_cfg["seeds"]
        ts = 50_000 if args.smoke else suite_cfg["hyperparameters"]["total_timesteps"]

        for env_id in envs:
            for seed in seeds:
                cmd = build_mujoco_cmd(env_id, seed, suite_cfg["hyperparameters"], out_dir, ts)
                rc = run_subprocess(cmd, env=env)
                if rc != 0:
                    failures.append((env_id, seed, rc))

    # Phase 2: Atari (5-game subset by default)
    if args.suite in ("atari", "all"):
        suite_cfg = cfg["suites"]["atari"]
        out_dir = os.path.join(args.results_dir, "atari")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        envs = suite_cfg["envs"][:1] if args.smoke else suite_cfg["envs"]
        seeds = [suite_cfg["seeds"][0]] if args.smoke else suite_cfg["seeds"]
        ts = 50_000 if args.smoke else suite_cfg["hyperparameters"]["total_timesteps"]

        for env_id in envs:
            for seed in seeds:
                cmd = build_atari_cmd(env_id, seed, suite_cfg["hyperparameters"], out_dir, ts)
                rc = run_subprocess(cmd, env=env)
                if rc != 0:
                    failures.append((env_id, seed, rc))

    # Phase 3: aggregate & evaluate
    if not args.smoke:
        eval_cmd = [sys.executable, "evaluate.py", "--results-dir", args.results_dir]
        run_subprocess(eval_cmd, env=env)

    with open(log_path, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] suite={args.suite} smoke={args.smoke} "
                f"failures={len(failures)}\n")
        for env_id, seed, rc in failures:
            f.write(f"  FAILED: env={env_id} seed={seed} rc={rc}\n")

    if failures:
        print(f"\n[FAIL] {len(failures)} runs failed:", failures)
        sys.exit(1)
    print("\n[OK] All runs completed successfully.")


if __name__ == "__main__":
    main()
