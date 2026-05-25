"""
PPO (Proximal Policy Optimization) implementation for CartPole-v1.

Implements the algorithm from:
  "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
  https://arxiv.org/abs/1707.06347

Key algorithmic features:
  - Clipped surrogate objective (PPO-Clip variant)
  - Generalized Advantage Estimation (GAE, lambda=0.95)
  - Shared Actor-Critic network with two separate heads
  - Multiple epochs of minibatch SGD per rollout (n_epochs=10)
  - Entropy bonus for exploration
  - Value-function loss coefficient
  - Gradient norm clipping

Assumption IDs applied:
  - ENV001: PyTorch 2.2.0 (inferred from paper date)
  - ENV002: Python 3.11
  - ENV003: CPU-only (no GPU required for CartPole)
"""

import os
import sys
import json
import time
import math
import random
import logging
import argparse
from collections import deque
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Runtime compute detection  (Sandbox Execution Contract §RUNTIME)
# ──────────────────────────────────────────────────────────────────────────────

HAS_GPU: bool = torch.cuda.is_available()
DEVICE: torch.device = torch.device("cuda" if HAS_GPU else "cpu")
log.info(f"Compute: {'GPU (' + torch.cuda.get_device_name(0) + ')' if HAS_GPU else 'CPU'}")

# Output directory — use $OUTPUT_DIR when running in the sandbox
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.json") -> dict:
    """Load hyperparameters from config.json, falling back to defaults."""
    # Resolve config relative to this script
    script_dir = Path(os.path.abspath(__file__)).parent
    cfg_file = script_dir / config_path
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = json.load(f)
        log.info(f"Loaded config from {cfg_file}")
    else:
        log.warning(f"Config not found at {cfg_file}; using built-in defaults.")
        cfg = {}

    # Merge with defaults — paper-specified values take precedence when present
    defaults = {
        # --- Paper-specified ---
        "total_timesteps": 500_000,      # Paper: 500,000 timesteps
        "learning_rate": 3e-4,           # Paper: Adam lr = 3e-4
        "batch_size": 64,                # Paper: mini-batch size = 64
        # --- Standard PPO (Schulman et al. 2017 defaults) ---
        "n_steps": 2048,                 # rollout buffer length per env
        "n_epochs": 10,                  # optimisation epochs per rollout
        "gamma": 0.99,                   # discount factor
        "gae_lambda": 0.95,              # GAE lambda
        "clip_coef": 0.2,               # PPO clip epsilon
        "ent_coef": 0.0,                # entropy bonus coefficient
        "vf_coef": 0.5,                 # value loss coefficient
        "max_grad_norm": 0.5,           # gradient norm clipping
        "hidden_size": 64,               # hidden layer width
        "seed": 42,
        # --- Evaluation (paper: 100 evaluation episodes) ---
        "eval_episodes": 100,
        "eval_interval": 10_000,        # eval every N timesteps
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    # Scale-down on CPU to keep wall-time reasonable while keeping
    # the architecture and data identical (Sandbox Contract rule 1-4).
    # We always run the FULL 500K on CPU too — CartPole is lightweight enough.
    if not HAS_GPU:
        log.info("CPU detected — running full 500K timesteps (CartPole is CPU-friendly).")

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Actor-Critic network
# ──────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared-backbone Actor-Critic network used by PPO.

    Architecture:
      Input → Linear(hidden) → Tanh → Linear(hidden) → Tanh
           ├─► Policy head: Linear(hidden → n_actions)   [logits → Categorical]
           └─► Value  head: Linear(hidden → 1)           [state value V(s)]

    This mirrors the architecture used in the original PPO paper's reference
    implementation (OpenAI Baselines) for MuJoCo and Atari.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        # Policy head (actor)
        self.policy_head = nn.Linear(hidden_size, action_dim)
        # Value head (critic)
        self.value_head = nn.Linear(hidden_size, 1)

        # Orthogonal initialisation — standard for RL
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Policy head: smaller gain for stable initial distribution
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        # Value head
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action (or evaluate a given one) and return:
          action, log_prob, entropy, value
        """
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


# ──────────────────────────────────────────────────────────────────────────────
# PPO training
# ──────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Fixed-size buffer that stores one rollout for PPO updates."""

    def __init__(self, n_steps: int, obs_dim: int, device: torch.device):
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.device = device
        self._reset()

    def _reset(self):
        self.observations = torch.zeros((self.n_steps, self.obs_dim), device=self.device)
        self.actions = torch.zeros(self.n_steps, dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros(self.n_steps, device=self.device)
        self.rewards = torch.zeros(self.n_steps, device=self.device)
        self.dones = torch.zeros(self.n_steps, device=self.device)
        self.values = torch.zeros(self.n_steps, device=self.device)
        self.advantages = torch.zeros(self.n_steps, device=self.device)
        self.returns = torch.zeros(self.n_steps, device=self.device)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ):
        self.observations[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float,
        gae_lambda: float,
    ):
        """Compute GAE advantages and discounted returns in-place.

        Storage convention: dones[t] = True if the transition AT step t ended
        the episode (i.e. env.step returned terminated|truncated=True at t).
        This means obs stored at step t+1 (or the bootstrap obs) is the first
        observation of a new episode when dones[t]=True.

        Therefore next_non_terminal[t] = 1 - dones[t], NOT 1 - dones[t+1].
        Using dones[t+1] would invert the mask: it treats the step that ended
        the episode as non-terminal and the first step of the new episode as
        terminal — both wrong.
        """
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                # BUG FIX: was `self.dones[t + 1]` — corrected to `self.dones[t]`
                # so that episode-boundary masking is applied at the right step.
                next_non_terminal = 1.0 - self.dones[t].item()
                next_value = self.values[t + 1].item()

            delta = (
                self.rewards[t].item()
                + gamma * next_value * next_non_terminal
                - self.values[t].item()
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """Yield shuffled mini-batches of (obs, actions, log_probs, advantages, returns)."""
        indices = torch.randperm(self.n_steps, device=self.device)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                self.observations[idx],
                self.actions[idx],
                self.log_probs[idx],
                self.advantages[idx],
                self.returns[idx],
            )


def evaluate(env_fn, model: ActorCritic, n_episodes: int, device: torch.device) -> float:
    """
    Run n_episodes deterministically (greedy actions) and return mean reward.

    The paper evaluates mean episode reward over 100 episodes.
    Each episode uses a fresh environment instance — no shared state between episodes.
    """
    rewards = []
    for _ in range(n_episodes):
        # Create one env instance per episode so we have a clean, independent reset
        env = env_fn()
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_t)
            action = logits.argmax(dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        env.close()
        rewards.append(ep_reward)
    return float(np.mean(rewards))


def train(cfg: dict) -> dict:
    """
    Main PPO training loop.

    Returns a flat dict of metrics suitable for metrics.json.
    """
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed_all(seed)

    # ── Environment ───────────────────────────────────────────────────────────
    # CartPole-v1: observation shape (4,), actions {0=left, 1=right}, max 500 steps
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n            # 2
    env_fn = lambda: gym.make("CartPole-v1")   # factory for evaluation

    log.info(f"Environment: CartPole-v1 | obs_dim={obs_dim} | action_dim={action_dim}")

    # ── Model & optimiser ─────────────────────────────────────────────────────
    model = ActorCritic(obs_dim, action_dim, hidden_size=cfg["hidden_size"]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], eps=1e-5)

    log.info(
        f"Model params: {sum(p.numel() for p in model.parameters()):,} | "
        f"Device: {DEVICE} | LR: {cfg['learning_rate']}"
    )

    # ── Rollout buffer ────────────────────────────────────────────────────────
    buffer = RolloutBuffer(cfg["n_steps"], obs_dim, DEVICE)

    # ── Tracking ──────────────────────────────────────────────────────────────
    episode_rewards: deque = deque(maxlen=100)
    episode_reward = 0.0
    eval_results = []          # [(timestep, mean_reward)]
    best_eval_reward = -float("inf")
    last_eval_ts = 0

    # ── Initial reset ─────────────────────────────────────────────────────────
    obs, _ = env.reset()
    done = False

    t_start = time.time()
    global_step = 0
    update_count = 0

    log.info(
        f"Starting training: {cfg['total_timesteps']:,} timesteps | "
        f"n_steps={cfg['n_steps']} | batch_size={cfg['batch_size']} | "
        f"n_epochs={cfg['n_epochs']} | clip={cfg['clip_coef']}"
    )

    # ── Main collection + update loop ─────────────────────────────────────────
    while global_step < cfg["total_timesteps"]:

        # ── Collect rollout ────────────────────────────────────────────────
        buffer._reset()
        model.eval()
        for step in range(cfg["n_steps"]):
            if global_step >= cfg["total_timesteps"]:
                break

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs_t)

            action_item = action.item()
            next_obs, reward, terminated, truncated, _ = env.step(action_item)
            done = terminated or truncated
            episode_reward += reward

            buffer.add(obs, action_item, log_prob.item(), reward, done, value.item())

            obs = next_obs
            global_step += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs, _ = env.reset()
                done = False

        # ── Compute GAE returns ────────────────────────────────────────────
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, last_value = model(obs_t)
            last_value = last_value.item()

        buffer.compute_returns_and_advantages(last_value, done, cfg["gamma"], cfg["gae_lambda"])

        # Normalise advantages (zero-mean, unit-variance)
        adv = buffer.advantages
        buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ── PPO update (n_epochs passes over the rollout) ──────────────────
        model.train()
        policy_losses, value_losses, entropy_losses = [], [], []

        for _ in range(cfg["n_epochs"]):
            for obs_b, act_b, old_log_b, adv_b, ret_b in buffer.get_batches(cfg["batch_size"]):

                # Evaluate current policy on the collected actions
                new_log_probs, entropy, new_values = model.get_action_and_value(obs_b, act_b)[1:]

                # Probability ratio r_t(θ)
                log_ratio = new_log_probs - old_log_b
                ratio = torch.exp(log_ratio)

                # PPO clipped surrogate objective (Equation 7 from the paper)
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(ratio, 1 - cfg["clip_coef"], 1 + cfg["clip_coef"])
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value function loss (MSE)
                value_loss = 0.5 * ((new_values - ret_b) ** 2).mean()

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + cfg["vf_coef"] * value_loss
                    + cfg["ent_coef"] * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())

        update_count += 1

        # ── Periodic evaluation ────────────────────────────────────────────
        if global_step - last_eval_ts >= cfg["eval_interval"] or global_step >= cfg["total_timesteps"]:
            mean_eval = evaluate(env_fn, model, cfg["eval_episodes"], DEVICE)
            eval_results.append((global_step, mean_eval))
            last_eval_ts = global_step

            if mean_eval > best_eval_reward:
                best_eval_reward = mean_eval
                # Save best model weights
                best_ckpt = OUTPUT_DIR / "best_model.pt"
                torch.save(model.state_dict(), best_ckpt)

            log.info(
                f"[{global_step:>7,}/{cfg['total_timesteps']:,}] "
                f"eval_reward={mean_eval:.1f}  best={best_eval_reward:.1f}  "
                f"train_ep_mean={np.mean(episode_rewards):.1f} (n={len(episode_rewards)})  "
                f"updates={update_count}  "
                f"elapsed={time.time()-t_start:.0f}s"
            )

    # ── Final evaluation (paper: 100 episodes after training) ─────────────────
    log.info("Training complete — running final 100-episode evaluation …")
    final_mean_reward = evaluate(env_fn, model, cfg["eval_episodes"], DEVICE)
    log.info(f"Final mean reward (100 eps): {final_mean_reward:.2f}  (target ≥ 475.0)")

    # Save final model
    final_ckpt = OUTPUT_DIR / "final_model.pt"
    torch.save(model.state_dict(), final_ckpt)
    log.info(f"Saved final checkpoint → {final_ckpt}")

    wall_time = time.time() - t_start

    # Build metrics dict
    metrics = {
        "mean_reward_100ep": final_mean_reward,
        "best_eval_reward": best_eval_reward,
        "total_timesteps": global_step,
        "n_updates": update_count,
        "wall_time_seconds": round(wall_time, 1),
        "target_reward": 475.0,
        "target_met": bool(final_mean_reward >= 475.0),
        "device": str(DEVICE),
        "seed": seed,
    }

    # Eval trajectory for analysis
    metrics["eval_trajectory"] = {str(ts): round(r, 2) for ts, r in eval_results}

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PPO on CartPole-v1")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--output-dir", default=None, help="Override OUTPUT_DIR")
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)

    # Persist resolved config for reproducibility
    cfg_out = OUTPUT_DIR / "resolved_config.json"
    with open(cfg_out, "w") as f:
        json.dump(cfg, f, indent=2)
    log.info(f"Resolved config → {cfg_out}")

    # ── Run training ──────────────────────────────────────────────────────────
    metrics = train(cfg)

    # ── Write metrics.json to OUTPUT_DIR (Sandbox Execution Contract §5) ──────
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"metrics.json → {metrics_path}")

    # Also write to code root for the orchestrator fallback
    code_root_metrics = Path(os.path.abspath(__file__)).parent / "metrics.json"
    try:
        with open(code_root_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"metrics.json (code root) → {code_root_metrics}")
    except OSError:
        # code/ is read-only in sandbox — this is expected and safe to skip
        log.debug("Could not write metrics.json to code root (read-only mount); skipping.")

    # Summary
    log.info("=" * 60)
    log.info(f"  mean_reward_100ep : {metrics['mean_reward_100ep']:.2f}")
    log.info(f"  target (≥475.0)   : {'✓ PASS' if metrics['target_met'] else '✗ FAIL'}")
    log.info(f"  wall_time         : {metrics['wall_time_seconds']:.0f}s")
    log.info("=" * 60)

    return 0 if metrics["target_met"] else 1


if __name__ == "__main__":
    sys.exit(main())
