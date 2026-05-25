"""
PPO (Proximal Policy Optimization) for CartPole-v1
===================================================
Implements the paper's algorithm with all assumption decisions from the
assumption ledger (A001–A010).

Assumptions applied:
  A001 – Adam epsilon = 1e-5
  A002 – Orthogonal weight initialization
  A003 – Linear LR decay schedule
  A004 – Per-minibatch advantage normalization
  A005 – Clipped value-function loss
  A006 – Gradient clipping max_grad_norm = 0.5
  A007 – GAE lambda = 0.95
  A008 – Entropy bonus coefficient = 0.01
  A009 – Seed = 42 (not specified in paper)
  A010 – num_envs = 4 (not specified in paper)

Sandbox contract:
  - Reads $OUTPUT_DIR for the writable output surface.
  - Writes metrics.json to $OUTPUT_DIR/metrics.json.
  - Detects GPU at runtime; scales down on CPU.
"""

import os
import json
import math
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    """Orthogonal initialisation (A002)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        # Small std for actor head so initial actions are near-uniform
        self.actor_head = layer_init(nn.Linear(64, action_dim), std=0.01)
        # Unit std for critic head
        self.critic_head = layer_init(nn.Linear(64, 1), std=1.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.shared(x))

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        features = self.shared(x)
        logits = self.actor_head(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic_head(features)
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict, output_dir: str) -> dict:
    # ---- compute detection (always-on per contract) ----
    HAS_GPU = torch.cuda.is_available()
    device = torch.device("cuda" if HAS_GPU else "cpu")
    print(f"[INFO] Device: {device}  (GPU={HAS_GPU})")

    # Scale down on CPU to keep runtime reasonable; still use the REAL env
    total_timesteps = config["total_timesteps"]
    if not HAS_GPU:
        cpu_cap = 200_000
        if total_timesteps > cpu_cap:
            print(
                f"[INFO] CPU-only sandbox detected — capping timesteps "
                f"{total_timesteps} → {cpu_cap} (A003/scale-down rule)."
            )
            total_timesteps = cpu_cap

    # ---- hyperparameters ----
    lr            = float(config["learning_rate"])
    num_envs      = int(config["num_envs"])
    num_steps     = int(config["num_steps"])
    minibatch_sz  = int(config["minibatch_size"])
    update_epochs = int(config["update_epochs"])
    gamma         = float(config["gamma"])
    gae_lambda    = float(config["gae_lambda"])      # A007
    clip_coef     = float(config["clip_coef"])
    clip_vloss    = bool(config["clip_vloss"])        # A005
    ent_coef      = float(config["ent_coef"])         # A008
    vf_coef       = float(config["vf_coef"])
    max_grad_norm = float(config["max_grad_norm"])    # A006
    adam_eps      = float(config["adam_eps"])         # A001
    seed          = int(config.get("seed", 42))
    num_eval_eps  = int(config.get("num_eval_episodes", 100))
    env_id        = config["env_id"]

    batch_size  = num_envs * num_steps
    num_updates = total_timesteps // batch_size

    print(
        f"[CONFIG] env={env_id} timesteps={total_timesteps} "
        f"batch={batch_size} updates={num_updates} lr={lr} eps={adam_eps}"
    )

    # ---- reproducibility ----
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed_all(seed)

    # ---- vectorised environments ----
    def _make_env(seed_offset: int = 0):
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + seed_offset)
        return env

    envs = gym.vector.SyncVectorEnv(
        [lambda i=i: _make_env(i) for i in range(num_envs)]
    )

    obs_dim    = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n

    # ---- model + optimiser ----
    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=adam_eps)  # A001

    # ---- rollout buffers ----
    obs_buf     = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((num_steps, num_envs), device=device)
    logprobs_buf= torch.zeros((num_steps, num_envs), device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), device=device)
    dones_buf   = torch.zeros((num_steps, num_envs), device=device)
    values_buf  = torch.zeros((num_steps, num_envs), device=device)

    # ---- tracking ----
    global_step       = 0
    start_time        = time.time()
    episodic_returns: list[float] = []

    obs_np, _ = envs.reset(seed=seed)
    obs  = torch.tensor(obs_np, dtype=torch.float32, device=device)
    done = torch.zeros(num_envs, device=device)

    # ---- main PPO loop ----
    for update in range(1, num_updates + 1):
        # A003: Linear LR decay
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * lr

        # ---- data collection ----
        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step]  = obs
            dones_buf[step] = done

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(obs)
                values_buf[step] = value.flatten()

            actions_buf[step]  = action
            logprobs_buf[step] = logprob

            obs_np, rew_np, term_np, trunc_np, infos = envs.step(
                action.cpu().numpy()
            )
            done_np = term_np | trunc_np

            rewards_buf[step] = torch.tensor(rew_np, dtype=torch.float32, device=device)
            obs  = torch.tensor(obs_np, dtype=torch.float32, device=device)
            done = torch.tensor(done_np, dtype=torch.float32, device=device)

            # Collect completed episode returns
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        episodic_returns.append(float(info["episode"]["r"]))

        # ---- compute GAE advantages ----
        with torch.no_grad():
            next_value = model.get_value(obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues      = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues      = values_buf[t + 1]
                delta = (
                    rewards_buf[t]
                    + gamma * nextvalues * nextnonterminal
                    - values_buf[t]
                )
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buf

        # ---- flatten batch ----
        b_obs       = obs_buf.reshape(-1, obs_dim)
        b_logprobs  = logprobs_buf.reshape(-1)
        b_actions   = actions_buf.reshape(-1).long()
        b_advantages= advantages.reshape(-1)
        b_returns   = returns.reshape(-1)
        b_values    = values_buf.reshape(-1)

        # ---- PPO update ----
        b_inds = np.arange(batch_size)
        for _epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_sz):
                end    = start + minibatch_sz
                mb_idx = b_inds[start:end]

                _, new_logprob, entropy, new_value = model.get_action_and_value(
                    b_obs[mb_idx], b_actions[mb_idx]
                )
                logratio = new_logprob - b_logprobs[mb_idx]
                ratio    = logratio.exp()

                # A004: Per-minibatch advantage normalisation
                mb_adv = b_advantages[mb_idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy (surrogate) loss — PPO-clip objective
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss — A005: clipped
                new_value = new_value.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (new_value - b_returns[mb_idx]) ** 2
                    v_clipped = b_values[mb_idx] + torch.clamp(
                        new_value - b_values[mb_idx], -clip_coef, clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_idx]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()

                # Entropy bonus — A008
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                # A006: gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        if update % 20 == 0:
            recent = (
                float(np.mean(episodic_returns[-20:]))
                if len(episodic_returns) >= 20
                else (float(np.mean(episodic_returns)) if episodic_returns else float("nan"))
            )
            elapsed = time.time() - start_time
            print(
                f"[{update:4d}/{num_updates}] step={global_step:7d} "
                f"mean_reward(last20)={recent:7.2f}  elapsed={elapsed:.0f}s"
            )

    envs.close()

    # ---- evaluation: 100 episodes (real CartPole-v1) ----
    print(f"\n[EVAL] Running {num_eval_eps} evaluation episodes …")
    eval_env = gym.make(env_id)
    eval_returns: list[float] = []
    for ep in range(num_eval_eps):
        obs_e, _ = eval_env.reset(seed=seed + 10_000 + ep)
        ep_ret   = 0.0
        done_e   = False
        while not done_e:
            obs_t = torch.tensor(obs_e, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_e, _, _, _ = model.get_action_and_value(obs_t)
            obs_e, rew_e, term_e, trunc_e, _ = eval_env.step(action_e.item())
            ep_ret += float(rew_e)
            done_e = term_e or trunc_e
        eval_returns.append(ep_ret)
    eval_env.close()

    mean_eval_reward = float(np.mean(eval_returns))
    wall_time        = time.time() - start_time
    training_tail    = (
        float(np.mean(episodic_returns[-20:]))
        if len(episodic_returns) >= 20
        else (float(np.mean(episodic_returns)) if episodic_returns else 0.0)
    )

    print(f"[EVAL] Mean reward over {num_eval_eps} episodes: {mean_eval_reward:.2f}")
    print(f"[INFO] Wall time: {wall_time:.1f}s | Training tail mean: {training_tail:.2f}")

    # ---- save model ----
    model_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved → {model_path}")

    # ---- metrics (flat JSON, key 'reward' matches rubric) ----
    metrics = {
        "reward":                     mean_eval_reward,
        "mean_eval_reward_100ep":     mean_eval_reward,
        "total_timesteps_run":        global_step,
        "num_eval_episodes":          num_eval_eps,
        "training_tail_mean_reward":  training_tail,
        "wall_time_seconds":          wall_time,
        "target_reward":              475.0,
        "target_met":                 mean_eval_reward >= 475.0,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] metrics.json written → {metrics_path}")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PPO baseline for CartPole-v1")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: $OUTPUT_DIR or ./outputs)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json (default: next to train.py)")
    args = parser.parse_args()

    # Resolve output directory: $OUTPUT_DIR > --output-dir > ./outputs
    output_dir = (
        os.environ.get("OUTPUT_DIR")
        or args.output_dir
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Set up matplotlib cache inside output_dir (avoids read-only mount errors)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, ".matplotlib"))

    print(f"[INFO] output_dir = {output_dir}")

    # Load config
    config_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.json"
    )
    with open(config_path) as f:
        config = json.load(f)

    metrics = train(config, output_dir)
    print(f"\n[DONE] {metrics}")


if __name__ == "__main__":
    main()
