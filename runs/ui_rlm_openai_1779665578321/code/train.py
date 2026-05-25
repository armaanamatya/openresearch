"""
PPO (Proximal Policy Optimization) on CartPole-v1.

Implements the full PPO algorithm with:
  - Actor-Critic network with shared trunk
  - Generalized Advantage Estimation (GAE)
  - Clipped surrogate objective
  - Value function loss + entropy bonus
  - 500,000 total timesteps training
  - 100-episode final evaluation

Assumptions applied: A001 (PPO as RL algorithm), ENV001, ENV002, ENV003.

RUNTIME COMPUTE DETECTION: adapts CPU ↔ GPU automatically at startup.
EAGER METRICS EMISSION: writes metrics.json after each evaluation phase.
RUBRIC GUARD: validates schema at the end.
"""

from __future__ import annotations

import json
import os
import sys
import time
import tempfile
import random
import math
import traceback
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# OUTPUT DIR setup
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "artifacts"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CACHE dirs (so nothing writes under the read-only /code mount)
# ─────────────────────────────────────────────────────────────
for _env_var, _sub in [
    ("MPLCONFIGDIR", ".matplotlib"),
    ("TMPDIR", "tmp"),
]:
    _path = os.path.join(OUTPUT_DIR, _sub)
    os.makedirs(_path, exist_ok=True)
    os.environ.setdefault(_env_var, _path)

# ─────────────────────────────────────────────────────────────
# RUNTIME COMPUTE DETECTION
# ─────────────────────────────────────────────────────────────
HAS_GPU = torch.cuda.is_available()
DEVICE   = torch.device("cuda" if HAS_GPU else "cpu")
print(f"[INFO] Device: {DEVICE}  |  HAS_GPU={HAS_GPU}", flush=True)

# ─────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
# Full scale on GPU, smoke-safe on CPU (CartPole is CPU-fast, so we
# always run the full schedule — 500 K steps takes ~30–60 s on CPU).
TOTAL_TIMESTEPS   = 500_000
N_STEPS           = 2048      # steps per rollout buffer
N_EPOCHS          = 10        # PPO update epochs per buffer
MINIBATCH_SIZE    = 64        # matches paper
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
CLIP_EPSILON      = 0.2
ENTROPY_COEFF     = 0.01
VF_COEFF          = 0.5
MAX_GRAD_NORM     = 0.5
LEARNING_RATE     = 3e-4      # matches paper
EVAL_EPISODES     = 100       # matches paper requirement
SEED              = 42

N_UPDATES = TOTAL_TIMESTEPS // N_STEPS  # ≈ 244


def write_metrics(d: Dict[str, Any]) -> None:
    """Atomically write metrics.json under OUTPUT_DIR."""
    path = os.path.join(OUTPUT_DIR, "metrics.json")
    tmp  = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────
# ACTOR-CRITIC NETWORK
# ─────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """Shared-trunk Actor-Critic for discrete action spaces.

    Architecture:
      trunk:  Linear(obs_dim → 64) → Tanh → Linear(64 → 64) → Tanh
      actor:  Linear(64 → n_actions)     → Categorical logits
      critic: Linear(64 → 1)             → state value
    """

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor_head  = nn.Linear(64, n_actions)
        self.critic_head = nn.Linear(64, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        features = self.trunk(obs)
        logits   = self.actor_head(features)
        values   = self.critic_head(features).squeeze(-1)
        return logits, values

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        logits, values = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, values

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, values = self(obs)
        return values


# ─────────────────────────────────────────────────────────────
# ROLLOUT BUFFER
# ─────────────────────────────────────────────────────────────
class RolloutBuffer:
    """Stores one rollout (N_STEPS transitions) for PPO updates."""

    def __init__(self, n_steps: int, obs_dim: int, device: torch.device) -> None:
        self.n_steps   = n_steps
        self.obs_dim   = obs_dim
        self.device    = device
        self.obs        = torch.zeros(n_steps, obs_dim,  device=device)
        self.actions    = torch.zeros(n_steps,           device=device, dtype=torch.long)
        self.log_probs  = torch.zeros(n_steps,           device=device)
        self.rewards    = torch.zeros(n_steps,           device=device)
        self.dones      = torch.zeros(n_steps,           device=device)
        self.values     = torch.zeros(n_steps,           device=device)
        self.advantages = torch.zeros(n_steps,           device=device)
        self.returns    = torch.zeros(n_steps,           device=device)
        self.ptr        = 0

    def add(self, obs, action, log_prob, reward, done, value) -> None:
        i = self.ptr
        self.obs[i]       = obs
        self.actions[i]   = action
        self.log_probs[i] = log_prob
        self.rewards[i]   = reward
        self.dones[i]     = done
        self.values[i]    = value
        self.ptr += 1

    def compute_gae(self, last_value: float, last_done: float) -> None:
        """Compute GAE advantages and lambda-returns in-place."""
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1].item()
            next_done  = last_done   if t == self.n_steps - 1 else self.dones[t + 1].item()
            delta = (self.rewards[t].item()
                     + GAMMA * next_value * (1.0 - next_done)
                     - self.values[t].item())
            gae   = delta + GAMMA * GAE_LAMBDA * (1.0 - next_done) * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values
        self.ptr = 0  # reset for next rollout

    def get_batches(self, minibatch_size: int):
        """Yield shuffled mini-batches."""
        idx = torch.randperm(self.n_steps, device=self.device)
        for start in range(0, self.n_steps, minibatch_size):
            mb_idx = idx[start : start + minibatch_size]
            yield (
                self.obs[mb_idx],
                self.actions[mb_idx],
                self.log_probs[mb_idx],
                self.advantages[mb_idx],
                self.returns[mb_idx],
            )


# ─────────────────────────────────────────────────────────────
# PPO UPDATE (CLIPPED SURROGATE OBJECTIVE)
# ─────────────────────────────────────────────────────────────
def ppo_update(
    agent: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
) -> Dict[str, float]:
    """
    One complete PPO update: N_EPOCHS × mini-batches.

    The surrogate objective (Eq. 7 in Schulman et al., 2017):
        L_CLIP = E[min(r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t)]
    Combined with value function loss and entropy bonus:
        L = -L_CLIP + c1 * L_VF - c2 * S[π_θ]
    """
    # Normalize advantages
    adv = buffer.advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    buffer.advantages = adv

    stats: Dict[str, List[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []
    }

    for _ in range(N_EPOCHS):
        for obs_b, act_b, old_lp_b, adv_b, ret_b in buffer.get_batches(MINIBATCH_SIZE):
            _, new_lp, entropy, new_val = agent.get_action_and_value(obs_b, act_b)

            # Probability ratio  r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
            log_ratio = new_lp - old_lp_b
            ratio     = log_ratio.exp()

            # PPO clipped surrogate objective
            surrogate1 = ratio * adv_b
            surrogate2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * adv_b
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            # Value function loss (unclipped for simplicity)
            value_loss  = ((new_val - ret_b) ** 2).mean()

            # Entropy bonus
            entropy_loss = entropy.mean()

            # Combined loss
            loss = (policy_loss
                    + VF_COEFF   * value_loss
                    - ENTROPY_COEFF * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy_loss.item())
            stats["approx_kl"].append(approx_kl)

    return {k: float(np.mean(v)) for k, v in stats.items()}


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate(agent: ActorCritic, n_episodes: int = EVAL_EPISODES, seed: int = 100) -> float:
    """Evaluate agent for n_episodes; return mean episode reward."""
    env_eval = gym.make("CartPole-v1")
    agent.eval()
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env_eval.reset(seed=seed + ep)
        total  = 0.0
        done   = False
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                logits, _ = agent(obs_t)
                action = Categorical(logits=logits).sample().item()
            obs, reward, terminated, truncated, _ = env_eval.step(action)
            done   = terminated or truncated
            total += reward
        rewards.append(total)
    env_eval.close()
    agent.train()
    return float(np.mean(rewards))


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train() -> Dict[str, Any]:
    """Main PPO training loop on CartPole-v1."""
    t0 = time.time()
    set_seed(SEED)

    env = gym.make("CartPole-v1")
    obs_dim   = env.observation_space.shape[0]   # 4
    n_actions = env.action_space.n               # 2

    agent    = ActorCritic(obs_dim, n_actions).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    buffer   = RolloutBuffer(N_STEPS, obs_dim, DEVICE)

    metrics: Dict[str, Any] = {
        "status": "training",
        "total_timesteps": TOTAL_TIMESTEPS,
        "target_reward": 475.0,
    }
    write_metrics(metrics)  # emit initial status

    # Logging
    ep_rewards_all: List[float] = []  # episode rewards during training
    eval_checkpoints: Dict[str, Any] = {}  # {update_idx: mean_eval_reward}
    training_curve_steps: List[int]   = []
    training_curve_rewards: List[float] = []

    obs_np, _ = env.reset(seed=SEED)
    obs_t     = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
    ep_reward = 0.0
    total_steps = 0

    print(f"[INFO] Training PPO on CartPole-v1 | "
          f"total_timesteps={TOTAL_TIMESTEPS} | "
          f"n_updates={N_UPDATES} | device={DEVICE}", flush=True)

    for update in range(1, N_UPDATES + 1):
        # ── Collect rollout ──────────────────────────────────
        buffer.ptr = 0
        for step in range(N_STEPS):
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(obs_t.unsqueeze(0))
            action_np = action.item()
            obs_next, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            buffer.add(
                obs_t,
                action.squeeze(),
                log_prob.squeeze(),
                torch.tensor(reward, device=DEVICE),
                torch.tensor(float(done), device=DEVICE),
                value.squeeze(),
            )

            ep_reward += reward
            total_steps += 1

            if done:
                ep_rewards_all.append(ep_reward)
                ep_reward = 0.0
                obs_next, _ = env.reset()

            obs_np = obs_next
            obs_t  = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)

        # Bootstrap last value
        with torch.no_grad():
            last_value = agent.get_value(obs_t.unsqueeze(0)).item()
        buffer.compute_gae(last_value, float(terminated or truncated))

        # ── PPO update ───────────────────────────────────────
        update_stats = ppo_update(agent, optimizer, buffer)

        # ── Logging ──────────────────────────────────────────
        if update % 10 == 0 or update == N_UPDATES:
            recent = np.mean(ep_rewards_all[-50:]) if ep_rewards_all else 0.0
            elapsed = time.time() - t0
            print(f"[UPDATE {update:4d}/{N_UPDATES}] "
                  f"steps={total_steps:7d} | "
                  f"ep_reward(50)={recent:6.1f} | "
                  f"policy_loss={update_stats['policy_loss']:+.4f} | "
                  f"value_loss={update_stats['value_loss']:.4f} | "
                  f"elapsed={elapsed:.1f}s", flush=True)
            training_curve_steps.append(total_steps)
            training_curve_rewards.append(float(recent))

        # ── Eager eval + metrics every 50 updates ────────────
        if update % 50 == 0 or update == N_UPDATES:
            eval_reward = evaluate(agent, n_episodes=20, seed=200 + update)
            eval_checkpoints[str(total_steps)] = eval_reward
            metrics.update({
                "status": "training",
                "current_update": update,
                "total_steps": total_steps,
                "eval_reward_recent": eval_reward,
            })
            write_metrics(metrics)
            print(f"[EVAL  {update:4d}/{N_UPDATES}] "
                  f"mean_reward(20ep)={eval_reward:.2f}", flush=True)

            # Save checkpoint
            ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", f"ckpt_upd{update}.pt")
            torch.save(agent.state_dict(), ckpt_path)

    env.close()

    # ─────────────────────────────────────────────────────────
    # FINAL 100-EPISODE EVALUATION  (paper requirement)
    # ─────────────────────────────────────────────────────────
    print("[INFO] Final 100-episode evaluation …", flush=True)
    final_mean_reward = evaluate(agent, n_episodes=EVAL_EPISODES, seed=999)
    print(f"[INFO] Mean reward over {EVAL_EPISODES} episodes: {final_mean_reward:.2f}", flush=True)

    wall_time = time.time() - t0

    # Save final model
    model_path = os.path.join(OUTPUT_DIR, "model.pt")
    torch.save(agent.state_dict(), model_path)
    print(f"[INFO] Model saved to {model_path}", flush=True)

    # ─────────────────────────────────────────────────────────
    # TRAINING CURVES JSON
    # ─────────────────────────────────────────────────────────
    training_curves = {
        "ppo_cartpole": {
            "step":   training_curve_steps,
            "reward": training_curve_rewards,
        }
    }
    tc_path = os.path.join(OUTPUT_DIR, "training_curves.json")
    with open(tc_path, "w") as f:
        json.dump(training_curves, f, indent=2)
    print(f"[INFO] Training curves saved to {tc_path}", flush=True)

    # ─────────────────────────────────────────────────────────
    # PLOT
    # ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: training reward curve
    axes[0].plot(training_curve_steps, training_curve_rewards, color="steelblue", linewidth=2)
    axes[0].axhline(475.0, color="red", linestyle="--", label="Target 475.0")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Mean Episode Reward (last 50 eps)")
    axes[0].set_title("PPO Training on CartPole-v1")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 2: eval checkpoints
    eval_steps  = sorted(eval_checkpoints, key=lambda x: int(x))
    eval_vals   = [eval_checkpoints[s] for s in eval_steps]
    axes[1].plot([int(s) for s in eval_steps], eval_vals, "o-", color="darkorange", linewidth=2)
    axes[1].axhline(475.0, color="red", linestyle="--", label="Target 475.0")
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Mean Eval Reward (20 eps)")
    axes[1].set_title("PPO Eval Checkpoints")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig_ppo_training.png")
    plt.savefig(fig_path, dpi=100)
    plt.close()
    print(f"[INFO] Figure saved to {fig_path}", flush=True)

    # ─────────────────────────────────────────────────────────
    # FINAL METRICS
    # ─────────────────────────────────────────────────────────
    metrics = {
        # Top-level headline metric the rubric checks
        "reward":                   final_mean_reward,
        "mean_reward_100ep":        final_mean_reward,
        "eval_episodes":            EVAL_EPISODES,
        "total_timesteps":          total_steps,
        "target_reward":            475.0,
        "achieved_target":          final_mean_reward >= 475.0,
        # Training diagnostics
        "training_episodes":        len(ep_rewards_all),
        "final_training_reward50":  float(np.mean(ep_rewards_all[-50:])) if ep_rewards_all else 0.0,
        # Meta
        "wall_time_seconds":        round(wall_time, 1),
        "device":                   str(DEVICE),
        "seed":                     SEED,
        "status":                   "complete",
    }
    write_metrics(metrics)
    print(f"[METRICS] Final metrics: {json.dumps(metrics, indent=2)}", flush=True)
    return metrics


# ─────────────────────────────────────────────────────────────
# CONFIG USED JSON  (written once at startup)
# ─────────────────────────────────────────────────────────────
def write_config_used() -> None:
    config = {
        "algorithm":        "PPO",
        "environment":      "CartPole-v1",
        "total_timesteps":  TOTAL_TIMESTEPS,
        "n_steps":          N_STEPS,
        "n_epochs":         N_EPOCHS,
        "minibatch_size":   MINIBATCH_SIZE,
        "gamma":            GAMMA,
        "gae_lambda":       GAE_LAMBDA,
        "clip_epsilon":     CLIP_EPSILON,
        "entropy_coeff":    ENTROPY_COEFF,
        "vf_coeff":         VF_COEFF,
        "max_grad_norm":    MAX_GRAD_NORM,
        "learning_rate":    LEARNING_RATE,
        "optimizer":        "Adam",
        "optimizer_eps":    1e-5,
        "seed":             SEED,
        "eval_episodes":    EVAL_EPISODES,
        "network": {
            "hidden_dim":   64,
            "activation":   "Tanh",
            "init":         "orthogonal",
        },
        "framework":        f"pytorch=={torch.__version__}",
        "device":           str(DEVICE),
        "python":           sys.version,
    }
    path = os.path.join(OUTPUT_DIR, "config_used.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Config written to {path}", flush=True)


# ─────────────────────────────────────────────────────────────
# README
# ─────────────────────────────────────────────────────────────
README_CONTENT = """\
# PPO on CartPole-v1 — Reproduction

## What was reproduced

A full Proximal Policy Optimization (PPO) agent trained on CartPole-v1 from OpenAI Gym /
Gymnasium. The implementation follows Schulman et al. (2017) with:

- Clipped surrogate objective (ε = 0.2)
- Generalized Advantage Estimation (GAE, λ = 0.95)
- Shared Actor-Critic network (64→64 Tanh trunk)
- Adam optimizer (lr = 3e-4, ε = 1e-5)
- 500,000 total training timesteps
- Final evaluation over 100 episodes (target mean reward ≥ 475.0)

## What was omitted and why

- Parallel environments (vectorized envs): CartPole is fast enough single-threaded
  within the wall-clock budget.
- Learning rate annealing: not specified in the paper claim map.
- Full paper architecture beyond CartPole: paper claim map only mentions CartPole-v1.

## How to read metrics.json

- `reward` / `mean_reward_100ep`: mean episode reward over the final 100-episode
  evaluation. Primary rubric metric; target ≥ 475.0.
- `achieved_target`: bool — True when the agent hit the 475.0 threshold.
- `total_timesteps`: actual steps simulated (≈ 500 000).
- `training_curves.json`: per-update `step` and `reward` arrays for plot reconstruction.
- `fig_ppo_training.png`: training curve + eval checkpoint panels.
"""


def write_readme() -> None:
    path = os.path.join(OUTPUT_DIR, "README.md")
    with open(path, "w") as f:
        f.write(README_CONTENT)
    print(f"[INFO] README written to {path}", flush=True)


# ─────────────────────────────────────────────────────────────
# RUBRIC GUARD
# ─────────────────────────────────────────────────────────────
def run_rubric_guard(metrics: Dict[str, Any]) -> None:
    try:
        from rubric_guard import assert_metrics_schema
        assert_metrics_schema(
            metrics,
            required_keys=[
                "reward",
                "mean_reward_100ep",
                "eval_episodes",
                "total_timesteps",
                "achieved_target",
                "wall_time_seconds",
            ],
            required_artifacts=[
                "README.md",
                "training_curves.json",
                "config_used.json",
                "fig_ppo_training.png",
                "metrics.json",
            ],
            artifact_dir=OUTPUT_DIR,
        )
        print("[RUBRIC GUARD] All checks passed ✓", flush=True)
    except ImportError:
        print("[RUBRIC GUARD] rubric_guard not found — skipping guard", flush=True)
    except Exception as e:
        print(f"[RUBRIC GUARD] FAILURE: {e}", flush=True)
        raise


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[INFO] Starting PPO reproduction | OUTPUT_DIR={OUTPUT_DIR}", flush=True)
    write_config_used()
    write_readme()

    try:
        final_metrics = train()
    except Exception as exc:
        err_metrics = {
            "status":  "error",
            "error":   str(exc),
            "traceback": traceback.format_exc(),
        }
        write_metrics(err_metrics)
        print(f"[ERROR] Training failed: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    run_rubric_guard(final_metrics)
    print("[INFO] Done.", flush=True)
