# -*- coding: utf-8 -*-
"""
PPO (Proximal Policy Optimization) baseline on CartPole-v1.

Reproduces the training recipe from the paper:
  - Adam optimizer, lr = 3e-4
  - Batch size = 64 (mini-batch size for PPO update steps)
  - 500,000 total timesteps
  - Evaluated over 100 episodes after training

Assumptions applied:
  A001 - CartPole-v1 from Gymnasium (bundled, no download)
  A002 - PPO inferred from rubric leaf "proximal policy optimization"
  A003 - GAE lambda=0.95 (Schulman 2017 default)
  A004 - n_steps=2048 rollout buffer; batch_size=64 is the mini-batch SGD size
  A005 - MLP with 2 hidden layers of 64 units each
  ENV001 - PyTorch 2.2.0
  ENV002 - Python 3.11
  ENV003 - CPU-only (CartPole is lightweight)
"""

import os
import sys
import json
import time
import math
import random
import numpy as np

# Force UTF-8 stdout/stderr so Unicode in log messages works on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Runtime compute detection (MUST detect at runtime, not hard-coded)
# ---------------------------------------------------------------------------
HAS_GPU = torch.cuda.is_available()
device = torch.device("cuda" if HAS_GPU else "cpu")
print(f"[train.py] Compute: {'GPU (' + torch.cuda.get_device_name(0) + ')' if HAS_GPU else 'CPU'}")
print(f"[train.py] PyTorch version: {torch.__version__}")

# ---------------------------------------------------------------------------
# Output directory (sandbox contract: ALWAYS write outputs to $OUTPUT_DIR)
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[train.py] OUTPUT_DIR = {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
with open(_cfg_path) as f:
    cfg = json.load(f)

# Core hyperparameters (from paper / config.json)
TOTAL_TIMESTEPS  = int(cfg["total_timesteps"])    # 500 000
N_STEPS          = int(cfg["n_steps"])            # 2048 – rollout buffer size
BATCH_SIZE       = int(cfg["batch_size"])         # 64   – PPO mini-batch size
N_EPOCHS         = int(cfg["n_epochs"])           # 10   – gradient epochs per rollout
LR               = float(cfg["learning_rate"])    # 3e-4
GAMMA            = float(cfg["gamma"])            # 0.99
GAE_LAMBDA       = float(cfg["gae_lambda"])       # 0.95
CLIP_EPS         = float(cfg["clip_epsilon"])     # 0.2
ENT_COEF         = float(cfg["entropy_coef"])     # 0.01
VAL_COEF         = float(cfg["value_coef"])       # 0.5
MAX_GRAD_NORM    = float(cfg["max_grad_norm"])    # 0.5
EVAL_EPISODES    = int(cfg["eval_episodes"])      # 100
EVAL_FREQ        = int(cfg["eval_freq"])          # 10 000
HIDDEN_SIZE      = int(cfg["hidden_size"])        # 64
N_HIDDEN         = int(cfg["n_hidden_layers"])    # 2
SEED             = int(cfg["seed"])               # 42

# Scale-down on CPU: CartPole is cheap enough that full budget is fine on CPU
# (small MLP on a 4-dim env). No reduction needed.
print(f"[train.py] TOTAL_TIMESTEPS={TOTAL_TIMESTEPS}, N_STEPS={N_STEPS}, "
      f"BATCH_SIZE={BATCH_SIZE}, N_EPOCHS={N_EPOCHS}, LR={LR}")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if HAS_GPU:
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Actor-Critic network (shared backbone, separate heads)
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """
    MLP actor-critic for discrete action spaces.
    Shared trunk → policy head (logits) + value head (scalar).
    Architecture: 2 hidden layers of 64 units, tanh activations (standard PPO).
    """
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_size: int = 64, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_size), nn.Tanh()]
            in_dim = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_size, act_dim)
        self.value_head  = nn.Linear(hidden_size, 1)
        # Orthogonal init (standard PPO initialisation)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight,  gain=1.0)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def get_action(self, obs: torch.Tensor):
        logits, value = self(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, values = self(obs)
        dist     = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values, entropy


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------
class RolloutBuffer:
    """Stores a single rollout of N_STEPS transitions."""
    def __init__(self, n_steps: int, obs_dim: int):
        self.n_steps  = n_steps
        self.obs_dim  = obs_dim
        self.clear()

    def clear(self):
        self.obs       = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions   = np.zeros(self.n_steps, dtype=np.int64)
        self.rewards   = np.zeros(self.n_steps, dtype=np.float32)
        self.dones     = np.zeros(self.n_steps, dtype=np.float32)
        self.values    = np.zeros(self.n_steps, dtype=np.float32)
        self.log_probs = np.zeros(self.n_steps, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, done, value, log_prob):
        i = self.ptr
        self.obs[i]       = obs
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.dones[i]     = done
        self.values[i]    = value
        self.log_probs[i] = log_prob
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: float,
                                       gamma: float, gae_lambda: float):
        """GAE (Generalized Advantage Estimation), Schulman et al. 2016."""
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        last_gae   = 0.0
        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            if t == self.n_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            delta    = (self.rewards[t]
                        + gamma * next_value * next_non_terminal
                        - self.values[t])
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return returns, advantages

    def get_tensors(self, returns, advantages):
        obs_t  = torch.FloatTensor(self.obs).to(device)
        acts_t = torch.LongTensor(self.actions).to(device)
        lp_t   = torch.FloatTensor(self.log_probs).to(device)
        ret_t  = torch.FloatTensor(returns).to(device)
        adv_t  = torch.FloatTensor(advantages).to(device)
        # Normalise advantages over the full rollout (cleanRL / SB3 standard)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return obs_t, acts_t, lp_t, ret_t, adv_t


# ---------------------------------------------------------------------------
# PPO surrogate objective update
# ---------------------------------------------------------------------------
def ppo_update(model: ActorCritic,
               optimizer: optim.Optimizer,
               buffer: RolloutBuffer,
               returns: np.ndarray,
               advantages: np.ndarray,
               n_epochs: int,
               batch_size: int,
               clip_eps: float,
               ent_coef: float,
               val_coef: float,
               max_grad_norm: float):
    """
    Implements the PPO clipped surrogate objective (Schulman et al. 2017).
    The training loop alternates between sampling data and optimising the
    surrogate objective over n_epochs gradient steps with mini-batches of
    size batch_size.
    """
    obs_t, acts_t, old_lp_t, ret_t, adv_t = buffer.get_tensors(returns, advantages)
    n = len(obs_t)
    total_loss_sum = 0.0
    total_pg_loss  = 0.0
    total_vf_loss  = 0.0
    total_ent      = 0.0
    n_updates      = 0

    for _ in range(n_epochs):
        # Shuffle → mini-batches (implements the "sampling + optimisation" loop)
        indices = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx      = indices[start:start + batch_size]
            mb_obs   = obs_t[idx]
            mb_acts  = acts_t[idx]
            mb_old_lp = old_lp_t[idx]
            mb_ret   = ret_t[idx]
            mb_adv   = adv_t[idx]

            # Current policy evaluation
            new_lp, values, entropy = model.evaluate_actions(mb_obs, mb_acts)

            # Clipped ratio — PPO surrogate objective
            ratio  = torch.exp(new_lp - mb_old_lp)
            surr1  = ratio * mb_adv
            surr2  = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            pg_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            vf_loss  = 0.5 * (values - mb_ret).pow(2).mean()

            # Entropy bonus (encourages exploration)
            ent_loss = -entropy.mean()

            loss = pg_loss + val_coef * vf_loss + ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss_sum += loss.item()
            total_pg_loss  += pg_loss.item()
            total_vf_loss  += vf_loss.item()
            total_ent      += (-ent_loss).item()
            n_updates      += 1

    return {
        "loss":    total_loss_sum / max(n_updates, 1),
        "pg_loss": total_pg_loss  / max(n_updates, 1),
        "vf_loss": total_vf_loss  / max(n_updates, 1),
        "entropy": total_ent      / max(n_updates, 1),
    }


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate(model: ActorCritic, env_id: str, n_episodes: int, seed: int) -> float:
    """Run `n_episodes` greedy (deterministic) episodes, return mean total reward."""
    eval_env = gym.make(env_id)
    episode_rewards = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        total_reward = 0.0
        done = truncated = False
        while not (done or truncated):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(obs_t)
            action = logits.argmax(dim=-1).item()   # greedy
            obs, reward, done, truncated, _ = eval_env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
    eval_env.close()
    return float(np.mean(episode_rewards))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train():
    t_start = time.time()

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]   # 4
    act_dim = env.action_space.n               # 2

    model     = ActorCritic(obs_dim, act_dim, HIDDEN_SIZE, N_HIDDEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    buffer    = RolloutBuffer(N_STEPS, obs_dim)

    # Logging
    episode_rewards_log = []
    current_ep_reward   = 0.0
    eval_log            = []          # [(timestep, mean_eval_reward), ...]

    obs, _ = env.reset(seed=SEED)
    global_step = 0
    n_updates   = 0

    pbar = tqdm(total=TOTAL_TIMESTEPS, desc="PPO CartPole-v1", unit="step",
                dynamic_ncols=True)

    while global_step < TOTAL_TIMESTEPS:
        # ---- Collect rollout (data collection phase) ----
        buffer.clear()
        for step in range(N_STEPS):
            if global_step >= TOTAL_TIMESTEPS:
                break
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value = model.get_action(obs_t)
            action_np = action.item()

            next_obs, reward, done, truncated, _ = env.step(action_np)
            current_ep_reward += reward
            global_step += 1

            terminal = done or truncated
            buffer.add(obs, action_np, reward, float(terminal),
                       value.item(), log_prob.item())

            if terminal:
                episode_rewards_log.append(current_ep_reward)
                current_ep_reward = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

            pbar.update(1)

            # Periodic evaluation
            if global_step % EVAL_FREQ == 0:
                mean_r = evaluate(model, "CartPole-v1", 10, SEED + 1000)
                eval_log.append((global_step, mean_r))
                pbar.set_postfix({"eval10": f"{mean_r:.1f}"})

        # Bootstrap value for GAE
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            _, last_value = model(obs_t)
        last_value = last_value.item()

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, GAMMA, GAE_LAMBDA)

        # ---- Optimise surrogate objective (PPO update phase) ----
        ppo_update(
            model, optimizer, buffer,
            returns, advantages,
            N_EPOCHS, BATCH_SIZE, CLIP_EPS,
            ENT_COEF, VAL_COEF, MAX_GRAD_NORM,
        )
        n_updates += 1

    pbar.close()
    env.close()

    # ---- Final evaluation over EVAL_EPISODES (=100) episodes ----
    print(f"\n[train.py] Final evaluation: {EVAL_EPISODES} episodes ...")
    mean_reward_100ep = evaluate(model, "CartPole-v1", EVAL_EPISODES, SEED + 9999)
    wall_time = time.time() - t_start

    print(f"[train.py] Mean reward ({EVAL_EPISODES} ep): {mean_reward_100ep:.2f}")
    print(f"[train.py] Target:                           475.0")
    print(f"[train.py] {'PASS' if mean_reward_100ep >= 475.0 else 'FAIL'} - "
          f"{'met' if mean_reward_100ep >= 475.0 else 'did NOT meet'} target")
    print(f"[train.py] Wall time: {wall_time:.1f}s")

    # ---- Save model weights ----
    model_path = os.path.join(OUTPUT_DIR, "ppo_cartpole.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[train.py] Model saved -> {model_path}")

    # ---- Save episode rewards log ----
    rewards_log_path = os.path.join(OUTPUT_DIR, "episode_rewards.json")
    with open(rewards_log_path, "w") as f:
        json.dump({"episode_rewards": episode_rewards_log,
                   "eval_checkpoints": eval_log}, f)

    # ---- Plot training curve ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        window = min(50, max(1, len(episode_rewards_log) // 20))
        if episode_rewards_log:
            smoothed = np.convolve(
                episode_rewards_log,
                np.ones(window) / window,
                mode="valid"
            )
            ax.plot(smoothed, label=f"Rolling mean (w={window})", color="steelblue")
        ax.axhline(475, ls="--", color="crimson", label="Target (475)")
        if eval_log:
            steps_e, rewards_e = zip(*eval_log)
            ax2 = ax.twinx()
            ax2.plot(range(len(rewards_e)), rewards_e, "o--",
                     color="darkorange", label="Eval (10ep)", markersize=4)
            ax2.set_ylabel("Eval reward")
            ax2.legend(loc="upper left")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("PPO on CartPole-v1 - Training Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "rewards_plot.png")
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"[train.py] Plot saved -> {plot_path}")
    except Exception as e:
        print(f"[train.py] Warning: plot failed ({e})")

    # ---- Write metrics.json (must be $OUTPUT_DIR/metrics.json per sandbox contract) ----
    metrics = {
        "mean_reward":              mean_reward_100ep,
        "mean_reward_100ep":        mean_reward_100ep,
        "total_timesteps":          global_step,
        "n_episodes_training":      len(episode_rewards_log),
        "mean_training_reward_last100": (
            float(np.mean(episode_rewards_log[-100:]))
            if len(episode_rewards_log) >= 100
            else (float(np.mean(episode_rewards_log)) if episode_rewards_log else 0.0)
        ),
        "n_ppo_updates":            n_updates,
        "wall_time_seconds":        wall_time,
        "target_met":               1 if mean_reward_100ep >= 475.0 else 0,
        "eval_episodes":            EVAL_EPISODES,
        "device":                   str(device),
    }

    # Primary write: $OUTPUT_DIR/metrics.json
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train.py] metrics.json -> {metrics_path}")

    # Secondary write: code root (for local runs where OUTPUT_DIR == code dir)
    code_root = os.path.dirname(os.path.abspath(__file__))
    local_metrics_path = os.path.join(code_root, "metrics.json")
    if os.path.abspath(local_metrics_path) != os.path.abspath(metrics_path):
        try:
            with open(local_metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"[train.py] metrics.json (code root) -> {local_metrics_path}")
        except OSError as err:
            print(f"[train.py] Note: code root metrics.json not written ({err}); "
                  "$OUTPUT_DIR copy is authoritative")

    print("\n[train.py] Training complete.")
    return metrics


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
