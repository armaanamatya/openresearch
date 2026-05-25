"""
PPO (Proximal Policy Optimization) on CartPole-v1
--------------------------------------------------
Implements the clipped surrogate objective from:
  Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347

Target: mean episode reward >= 475.0 over 100 evaluation episodes.

Assumptions applied:
  A001 / ENV001 - PyTorch 2.2.0 (CPU-only inference inferred from paper hardware clues)
  A002 / ENV002 - Python 3.11
  A003 / ENV003 - CPU-only training (no GPU required)

Sandbox contract:
  - All outputs (metrics, logs, checkpoints) go to $OUTPUT_DIR.
  - metrics.json is written at $OUTPUT_DIR/metrics.json AND ./metrics.json
    so both the sandbox read-back and any local run find it.
"""

import os
import json
import time
import math
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

# ---------------------------------------------------------------------------
# Sandbox / output directory setup
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)

# Redirect matplotlib cache if set by the harness
MPLCONFIGDIR = os.environ.get("MPLCONFIGDIR", os.path.join(OUTPUT_DIR, ".matplotlib"))
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = MPLCONFIGDIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_path = os.path.join(OUTPUT_DIR, "train.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime compute detection (ALWAYS auto-detect — never hard-code)
# ---------------------------------------------------------------------------
HAS_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
log.info(f"Compute: {'GPU (' + torch.cuda.get_device_name(0) + ')' if HAS_GPU else 'CPU'}")

# ---------------------------------------------------------------------------
# Hyperparameters (from config.json if present, else defaults)
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        CFG = json.load(f)
else:
    CFG = {}

# Core PPO hyperparameters — scale down steps on CPU to allow a timely run
# while keeping everything else (model, env, metric) identical.
TOTAL_TIMESTEPS   = CFG.get("total_timesteps",   500_000 if HAS_GPU else 200_000)
N_STEPS           = CFG.get("n_steps",            2048)        # rollout length per update
N_ENVS            = CFG.get("n_envs",             1)           # parallel envs (single here)
LEARNING_RATE     = CFG.get("learning_rate",      3e-4)
GAMMA             = CFG.get("gamma",              0.99)        # discount factor
GAE_LAMBDA        = CFG.get("gae_lambda",         0.95)        # GAE λ
CLIP_EPSILON      = CFG.get("clip_epsilon",       0.2)         # PPO clip ratio
ENTROPY_COEF      = CFG.get("entropy_coef",       0.01)
VALUE_COEF        = CFG.get("value_coef",         0.5)
MAX_GRAD_NORM     = CFG.get("max_grad_norm",      0.5)
N_EPOCHS          = CFG.get("n_epochs",           10)          # gradient epochs per rollout
MINIBATCH_SIZE    = CFG.get("minibatch_size",     64)
EVAL_EPISODES     = CFG.get("eval_episodes",      100)         # per rubric: 100 ep evaluation
LOG_INTERVAL      = CFG.get("log_interval",       10)          # update steps between logs
NUM_SEEDS         = CFG.get("num_seeds",          3)           # multiple runs → reproducibility
ENV_ID            = CFG.get("env_id",             "CartPole-v1")


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Shared-trunk MLP policy + value head for discrete action spaces."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head  = nn.Linear(hidden_dim, act_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Orthogonal init (recommended for PPO)
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.constant_(self.actor_head.bias,  0.0)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        feat  = self.shared(obs)
        logits = self.actor_head(feat)
        value  = self.critic_head(feat).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor):
        """Sample action and return (action, log_prob, value)."""
        logits, value = self.forward(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate stored actions. Returns (log_probs, values, entropy)."""
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), values, dist.entropy()


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int):
        self.n_steps = n_steps
        self.obs      = torch.zeros(n_steps, obs_dim)
        self.actions  = torch.zeros(n_steps, dtype=torch.long)
        self.log_probs = torch.zeros(n_steps)
        self.values   = torch.zeros(n_steps)
        self.rewards  = torch.zeros(n_steps)
        self.dones    = torch.zeros(n_steps)
        self.ptr = 0

    def add(self, obs, action, log_prob, value, reward, done):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.ptr += 1

    def compute_gae(self, last_value: torch.Tensor, last_done: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        advantages = torch.zeros(self.n_steps)
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value.item()
            else:
                next_non_terminal = 1.0 - self.dones[t + 1].item()
                next_value = self.values[t + 1].item()
            delta = (self.rewards[t].item()
                     + GAMMA * next_value * next_non_terminal
                     - self.values[t].item())
            gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * gae
            advantages[t] = gae
        returns = advantages + self.values
        return advantages, returns

    def reset(self):
        self.ptr = 0

    def get_batches(self, advantages, returns) -> List[dict]:
        """Return shuffled minibatches."""
        idx = torch.randperm(self.n_steps)
        batches = []
        for start in range(0, self.n_steps, MINIBATCH_SIZE):
            mb_idx = idx[start: start + MINIBATCH_SIZE]
            batches.append({
                "obs":       self.obs[mb_idx].to(DEVICE),
                "actions":   self.actions[mb_idx].to(DEVICE),
                "old_lp":    self.log_probs[mb_idx].to(DEVICE),
                "adv":       advantages[mb_idx].to(DEVICE),
                "returns":   returns[mb_idx].to(DEVICE),
            })
        return batches


# ---------------------------------------------------------------------------
# PPO update step — stochastic gradient ascent on the surrogate objective
# ---------------------------------------------------------------------------
def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    advantages: torch.Tensor,
    returns: torch.Tensor,
) -> dict:
    """
    Optimise the clipped surrogate objective for N_EPOCHS passes.
    Returns dict of mean losses for logging.
    """
    # Normalize advantages
    adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
    advantages_norm = (advantages - adv_mean) / adv_std

    pg_losses, v_losses, ent_losses, total_losses = [], [], [], []
    clip_fracs = []

    for _ in range(N_EPOCHS):
        for batch in buffer.get_batches(advantages_norm, returns):
            new_lp, values, entropy = model.evaluate(batch["obs"], batch["actions"])

            # Clipped surrogate loss (stochastic gradient *ascent* on reward)
            ratio      = (new_lp - batch["old_lp"]).exp()
            pg_loss1   = -batch["adv"] * ratio
            pg_loss2   = -batch["adv"] * ratio.clamp(1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
            pg_loss    = torch.max(pg_loss1, pg_loss2).mean()

            # Value function loss (clipped)
            v_loss = nn.functional.mse_loss(values, batch["returns"])

            # Entropy bonus
            ent_loss = -entropy.mean()

            loss = pg_loss + VALUE_COEF * v_loss + ENTROPY_COEF * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > CLIP_EPSILON).float().mean()
            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            ent_losses.append(ent_loss.item())
            total_losses.append(loss.item())
            clip_fracs.append(clip_frac.item())

    return {
        "pg_loss":   np.mean(pg_losses),
        "v_loss":    np.mean(v_losses),
        "ent_loss":  np.mean(ent_losses),
        "total_loss": np.mean(total_losses),
        "clip_frac": np.mean(clip_fracs),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: ActorCritic, n_episodes: int = EVAL_EPISODES) -> float:
    """Run n_episodes with the greedy policy. Return mean episode reward."""
    env = gym.make(ENV_ID)
    ep_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, _ = model.forward(obs_t)
                action = logits.argmax(dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        ep_rewards.append(total_reward)
    env.close()
    return float(np.mean(ep_rewards))


# ---------------------------------------------------------------------------
# Single training run for one seed
# ---------------------------------------------------------------------------
def train_single_seed(seed: int) -> dict:
    """Full PPO training loop for CartPole-v1. Returns metrics dict."""
    log.info(f"=== Seed {seed} | device={DEVICE} | timesteps={TOTAL_TIMESTEPS} ===")
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(ENV_ID)
    env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model     = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)
    buffer    = RolloutBuffer(N_STEPS, obs_dim)

    obs, _   = env.reset(seed=seed)
    obs_t    = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
    done     = False

    global_step = 0
    update_step = 0
    ep_rewards: List[float] = []
    ep_reward_cur = 0.0
    reward_history: List[Tuple[int, float]] = []   # (timestep, mean_reward)

    t_start = time.time()

    while global_step < TOTAL_TIMESTEPS:
        # ---- collect rollout ----
        buffer.reset()
        for _ in range(N_STEPS):
            with torch.no_grad():
                action, log_prob, value = model.act(obs_t.unsqueeze(0))
            action_int = action.item()

            next_obs, reward, terminated, truncated, _ = env.step(action_int)
            ep_reward_cur += float(reward)
            done = terminated or truncated

            buffer.add(
                obs_t.cpu(),
                action.cpu(),
                log_prob.cpu(),
                value.cpu(),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(float(done), dtype=torch.float32),
            )
            global_step += 1

            if done:
                ep_rewards.append(ep_reward_cur)
                ep_reward_cur = 0.0
                obs, _ = env.reset()
                done = False
            else:
                obs = next_obs

            obs_t = torch.tensor(next_obs if not done else obs,
                                 dtype=torch.float32).to(DEVICE)

            if global_step >= TOTAL_TIMESTEPS:
                break

        # ---- compute GAE ----
        with torch.no_grad():
            _, last_value = model.forward(obs_t.unsqueeze(0))
        advantages, returns = buffer.compute_gae(last_value.cpu(), float(done))

        # ---- PPO update ----
        losses = ppo_update(model, optimizer, buffer, advantages, returns)
        update_step += 1

        # ---- logging ----
        if ep_rewards and (update_step % LOG_INTERVAL == 0):
            recent_mean = float(np.mean(ep_rewards[-20:]))
            reward_history.append((global_step, recent_mean))
            log.info(
                f"seed={seed} step={global_step:7d}  "
                f"mean_rew(last20)={recent_mean:7.2f}  "
                f"pg={losses['pg_loss']:.4f}  v={losses['v_loss']:.4f}  "
                f"ent={losses['ent_loss']:.4f}  clip={losses['clip_frac']:.3f}"
            )

    env.close()
    wall_time = time.time() - t_start

    # ---- final evaluation ----
    log.info(f"Evaluating seed={seed} over {EVAL_EPISODES} episodes …")
    mean_eval_reward = evaluate(model, n_episodes=EVAL_EPISODES)
    log.info(f"Seed {seed}: mean_eval_reward = {mean_eval_reward:.2f}  ({wall_time:.1f}s)")

    # ---- save checkpoint ----
    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", f"ppo_cartpole_seed{seed}.pt")
    torch.save(model.state_dict(), ckpt_path)
    log.info(f"Checkpoint saved: {ckpt_path}")

    return {
        "seed": seed,
        "mean_eval_reward": mean_eval_reward,
        "total_timesteps_run": global_step,
        "wall_time_seconds": wall_time,
        "reward_history": reward_history,
    }


# ---------------------------------------------------------------------------
# Main — run multiple seeds (reproducibility), aggregate, write metrics.json
# ---------------------------------------------------------------------------
def main():
    global TOTAL_TIMESTEPS, EVAL_EPISODES  # declared before any use in this scope

    parser = argparse.ArgumentParser(description="PPO on CartPole-v1")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(NUM_SEEDS)),
                        help="Random seeds to run (default: 0 1 2)")
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--eval-episodes",   type=int, default=EVAL_EPISODES)
    args = parser.parse_args()

    TOTAL_TIMESTEPS = args.total_timesteps
    EVAL_EPISODES   = args.eval_episodes

    log.info(f"PPO CartPole-v1 | seeds={args.seeds} | timesteps={TOTAL_TIMESTEPS} "
             f"| eval_ep={EVAL_EPISODES} | device={DEVICE}")

    all_results = []
    overall_start = time.time()

    for seed in args.seeds:
        result = train_single_seed(seed)
        all_results.append(result)

    # Aggregate
    eval_rewards = [r["mean_eval_reward"] for r in all_results]
    mean_reward  = float(np.mean(eval_rewards))
    std_reward   = float(np.std(eval_rewards))
    best_reward  = float(np.max(eval_rewards))
    wall_total   = time.time() - overall_start

    log.info(f"\n{'='*60}")
    log.info(f"FINAL RESULTS over {len(args.seeds)} seeds:")
    log.info(f"  mean_reward = {mean_reward:.2f} ± {std_reward:.2f}")
    log.info(f"  best_reward = {best_reward:.2f}")
    log.info(f"  target      = 475.0")
    log.info(f"  wall_time   = {wall_total:.1f}s")
    log.info(f"{'='*60}")

    # Build metrics dict (flat, rubric-compatible)
    metrics = {
        "mean_reward":              mean_reward,
        "std_reward":               std_reward,
        "best_seed_reward":         best_reward,
        "eval_episodes":            EVAL_EPISODES,
        "total_timesteps":          TOTAL_TIMESTEPS,
        "num_seeds":                len(args.seeds),
        "target_mean_reward":       475.0,
        "target_met":               float(mean_reward >= 475.0),
        "wall_time_seconds":        wall_total,
        "per_seed_rewards":         {f"seed_{r['seed']}": r["mean_eval_reward"]
                                     for r in all_results},
    }

    # Write to OUTPUT_DIR (primary — required by sandbox contract)
    out_metrics = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"metrics.json written to: {out_metrics}")

    # Best-effort write to the code root for local runs / inspection.
    # On RunPod the code dir is mounted READ-ONLY so this write may fail —
    # that is expected and must NOT crash the process after metrics are
    # already safely written to OUTPUT_DIR above.
    local_metrics = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "metrics.json"
    )
    if local_metrics != out_metrics:
        try:
            with open(local_metrics, "w") as f:
                json.dump(metrics, f, indent=2)
            log.info(f"metrics.json also written to: {local_metrics}")
        except OSError as _e:
            log.info(
                f"Could not write metrics.json to code root "
                f"({_e}) — this is expected when /code is read-only (RunPod). "
                f"The authoritative copy is at: {out_metrics}"
            )

    # Save full run detail
    run_detail_path = os.path.join(OUTPUT_DIR, "run_detail.json")
    with open(run_detail_path, "w") as f:
        # reward_history contains tuples — convert for JSON
        serialisable = []
        for r in all_results:
            r2 = dict(r)
            r2["reward_history"] = [list(x) for x in r["reward_history"]]
            serialisable.append(r2)
        json.dump(serialisable, f, indent=2)

    log.info("Done.")
    return metrics


if __name__ == "__main__":
    main()
