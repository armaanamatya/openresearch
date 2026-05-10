"""
Aggregate per-run episode-return JSONs produced by ppo_continuous_action.py
and ppo_atari.py into the metrics required by the reproduction contract:

  MuJoCo (Section 6.1, Table 1):
    - episodic_returns_raw.json
    - normalized_scores.json
    - mujoco_summary_table.csv
    - avg_normalized_score_scalar.txt
  Atari (Section 6.4, Table 2):
    - episodic_returns_raw.json
    - atari_summary_table.csv
    - games_won_alltraining.json
    - games_won_last100.json
  Combined:
    - results/metrics.json

Per A007: per-env normalization ceiling = best mean-final-perf across PPO seeds.
Random-policy baselines are taken from the published openai/baselines benchmarks
(MuJoCo 1M-step random returns).
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


# Random-policy mean returns for MuJoCo -v4 (approximate; from common RL benchmarks).
# Used only for the (score - random) / (best - random) normalization (A007).
MUJOCO_RANDOM = {
    "HalfCheetah-v4": -280.0,
    "Hopper-v4": 18.0,
    "InvertedDoublePendulum-v4": 50.0,
    "InvertedPendulum-v4": 5.0,
    "Reacher-v4": -42.0,
    "Swimmer-v4": 1.0,
    "Walker2d-v4": 1.5,
    # -v1 fallbacks (for archival comparison)
    "HalfCheetah-v1": -280.0,
    "Hopper-v1": 18.0,
    "InvertedDoublePendulum-v1": 50.0,
    "InvertedPendulum-v1": 5.0,
    "Reacher-v1": -42.0,
    "Swimmer-v1": 1.0,
    "Walker2d-v1": 1.5,
}


def last_n_mean(returns, n=100):
    """Mean of last n episodic returns. returns is a list of [step, ret]."""
    if not returns:
        return float("nan")
    rets = [r for _, r in returns[-n:]]
    return float(np.mean(rets))


def all_mean(returns):
    if not returns:
        return float("nan")
    rets = [r for _, r in returns]
    return float(np.mean(rets))


def collect_runs(results_subdir):
    """Return dict[env_id] -> dict[seed] -> {'returns': [...], 'config': {...}}."""
    runs = defaultdict(dict)
    pattern = os.path.join(results_subdir, "returns__*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            payload = json.load(f)
        env_id = payload["env_id"]
        seed = payload["seed"]
        runs[env_id][seed] = payload
    return runs


def evaluate_mujoco(results_dir):
    sub = os.path.join(results_dir, "mujoco")
    if not os.path.isdir(sub):
        return None
    runs = collect_runs(sub)
    if not runs:
        return None

    raw = {env: {seed: rec["episodic_returns"] for seed, rec in seeds.items()}
           for env, seeds in runs.items()}
    with open(os.path.join(sub, "episodic_returns_raw.json"), "w") as f:
        json.dump(raw, f)

    # Per-env, per-seed mean of last 100 episodes
    per_env_per_seed = defaultdict(dict)
    for env, seeds in runs.items():
        for seed, rec in seeds.items():
            per_env_per_seed[env][seed] = last_n_mean(rec["episodic_returns"], n=100)

    # Per-env normalization ceiling: best mean across seeds for PPO Clip (A007).
    # In a full ablation sweep, this should be max across all algorithm variants.
    norm = {}
    for env, seed_scores in per_env_per_seed.items():
        random_score = MUJOCO_RANDOM.get(env, 0.0)
        scores = [v for v in seed_scores.values() if not np.isnan(v)]
        if not scores:
            continue
        best = max(scores)
        denom = best - random_score
        if denom <= 0:
            denom = 1.0
        norm[env] = {
            "random_score": random_score,
            "best_score": best,
            "per_seed_normalized": {
                str(s): (v - random_score) / denom for s, v in seed_scores.items()
            },
        }

    all_normalized = []
    for env, info in norm.items():
        all_normalized.extend(info["per_seed_normalized"].values())
    avg_normalized = float(np.mean(all_normalized)) if all_normalized else float("nan")

    with open(os.path.join(sub, "normalized_scores.json"), "w") as f:
        json.dump({"per_env": norm, "avg_normalized_score": avg_normalized}, f, indent=2)

    with open(os.path.join(sub, "avg_normalized_score_scalar.txt"), "w") as f:
        f.write(f"{avg_normalized:.6f}\n")

    csv_path = os.path.join(sub, "mujoco_summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_id", "seed", "mean_last100_return", "normalized_score"])
        for env, seed_scores in per_env_per_seed.items():
            for seed, score in seed_scores.items():
                ns = norm.get(env, {}).get("per_seed_normalized", {}).get(str(seed), float("nan"))
                w.writerow([env, seed, f"{score:.4f}", f"{ns:.6f}"])

    return {
        "n_envs": len(runs),
        "n_runs": sum(len(v) for v in runs.values()),
        "avg_normalized_score": avg_normalized,
        "per_env": {e: {"mean_last100": per_env_per_seed[e]} for e in runs},
    }


def evaluate_atari(results_dir):
    sub = os.path.join(results_dir, "atari")
    if not os.path.isdir(sub):
        return None
    runs = collect_runs(sub)
    if not runs:
        return None

    raw = {env: {seed: rec["episodic_returns"] for seed, rec in seeds.items()}
           for env, seeds in runs.items()}
    with open(os.path.join(sub, "episodic_returns_raw.json"), "w") as f:
        json.dump(raw, f)

    # Two metrics: mean over all training, and mean over last 100 episodes
    metric_a = {}  # all-training avg
    metric_b = {}  # last-100 avg
    for env, seeds in runs.items():
        a_scores = {seed: all_mean(rec["episodic_returns"]) for seed, rec in seeds.items()}
        b_scores = {seed: last_n_mean(rec["episodic_returns"], n=100) for seed, rec in seeds.items()}
        metric_a[env] = {
            "per_seed": a_scores,
            "mean_across_seeds": float(np.mean(list(a_scores.values()))),
        }
        metric_b[env] = {
            "per_seed": b_scores,
            "mean_across_seeds": float(np.mean(list(b_scores.values()))),
        }

    # Games-won placeholders — populated only when an A2C/ACER baseline is supplied.
    games_won_a = {"PPO": 0, "A2C": 0, "ACER": 0, "tie": 0,
                   "note": "Head-to-head requires A2C/ACER baselines, not run by this script."}
    games_won_b = {"PPO": 0, "A2C": 0, "ACER": 0, "tie": 0,
                   "note": "Head-to-head requires A2C/ACER baselines, not run by this script."}
    with open(os.path.join(sub, "games_won_alltraining.json"), "w") as f:
        json.dump(games_won_a, f, indent=2)
    with open(os.path.join(sub, "games_won_last100.json"), "w") as f:
        json.dump(games_won_b, f, indent=2)

    csv_path = os.path.join(sub, "atari_summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_id", "seed", "mean_all_training", "mean_last100"])
        for env in runs:
            for seed in runs[env]:
                w.writerow([
                    env, seed,
                    f"{metric_a[env]['per_seed'][seed]:.4f}",
                    f"{metric_b[env]['per_seed'][seed]:.4f}",
                ])

    return {
        "n_games": len(runs),
        "n_runs": sum(len(v) for v in runs.values()),
        "metric_all_training": {e: metric_a[e]["mean_across_seeds"] for e in runs},
        "metric_last_100": {e: metric_b[e]["mean_across_seeds"] for e in runs},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    summary = {
        "mujoco": evaluate_mujoco(args.results_dir),
        "atari": evaluate_atari(args.results_dir),
    }
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
