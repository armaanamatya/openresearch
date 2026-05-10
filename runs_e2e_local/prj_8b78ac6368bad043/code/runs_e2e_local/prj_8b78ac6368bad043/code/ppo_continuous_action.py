"""
PPO for continuous control (MuJoCo) — adapted from CleanRL v1.0.0
(https://github.com/vwxyzjn/cleanrl/blob/v1.0.0/cleanrl/ppo_continuous_action.py)

Reproduces Schulman et al. 2017 (arXiv:1707.06347) Section 6.1 / Table 1.

Assumptions applied from ledger:
  A003: N=1 actor for MuJoCo (Table 3 omits N) → --num-envs 1
  A007: per-env best mean across PPO seeds is normalization ceiling
  A008: V(s_T) bootstrap uses stopped gradient (default torch behavior)
  ENV001: Python 3.10
  ENV002: PyTorch 2.1.0 (replaces TF1)
  ENV003: gymnasium 0.29 + mujoco 3.1.1 with -v4 envs (paper used -v1)
  ENV004: Adam eps=1e-5 (NOT PyTorch default 1e-8)
  ENV005: CleanRL v1.0.0 codebase
  ENV007: MUJOCO_GL=egl headless rendering
  ENV009: max_grad_norm=0.5 (absent from paper, present in ref impls)
  ENV010: Orthogonal init — hidden=sqrt(2), policy_mean=0.01, value=1.0
"""
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

try:
    import tyro
except ImportError:
    tyro = None

import gymnasium as gym


@dataclass
class Args:
    exp_name: str = "ppo_continuous_action"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ppo-paper-repro"
    wandb_entity: Optional[str] = None
    capture_video: bool = False

    # Algorithm specific
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_envs: int = 1                # A003: N=1 for MuJoCo
    num_steps: int = 2048            # T=2048 (Table 3)
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32        # minibatch size = (1*2048)/32 = 64 (Table 3)
    update_epochs: int = 10          # K=10 (Table 3)
    norm_adv: bool = True            # per-minibatch advantage normalization
    clip_coef: float = 0.2           # epsilon=0.2 (Table 3)
    clip_vloss: bool = True
    ent_coef: float = 0.0            # MuJoCo: no entropy bonus (Section 6.1)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # ENV009
    target_kl: Optional[float] = None

    # Bookkeeping
    output_dir: str = "results/mujoco"

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """ENV010: orthogonal init with custom scales."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """MLP actor-critic with separate policy/value networks (Section 6.1).

    2 hidden layers, 64 units, tanh; Gaussian head with state-independent
    learned log_std.
    """

    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # ENV010 value head std=1.0
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),  # ENV010 policy head std=0.01
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


def parse_args() -> Args:
    if tyro is not None:
        return tyro.cli(Args)
    # Fallback: argparse-style
    import argparse

    p = argparse.ArgumentParser()
    defaults = Args()
    for f_name, f_def in defaults.__dataclass_fields__.items():
        default = getattr(defaults, f_name)
        if isinstance(default, bool):
            p.add_argument(f"--{f_name.replace('_','-')}", type=lambda v: v.lower() in ("true", "1", "yes"), default=default)
        elif default is None:
            p.add_argument(f"--{f_name.replace('_','-')}", type=float, default=None)
        else:
            p.add_argument(f"--{f_name.replace('_','-')}", type=type(default), default=default)
    ns = p.parse_args()
    return Args(**{k: getattr(ns, k) for k in defaults.__dataclass_fields__})


def main():
    args = parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard", run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in asdict(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "continuous action space required"

    agent = Agent(envs).to(device)
    # ENV004: Adam eps=1e-5 (CRITICAL — NOT PyTorch default 1e-8)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    episodic_returns = []  # list of (global_step, return)

    for iteration in range(1, args.num_iterations + 1):
        # LR annealing (CleanRL default; for MuJoCo this is often disabled in paper but
        # CleanRL benchmark uses it. We follow CleanRL.)
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np.astype(np.float32)).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_r = float(info["episode"]["r"])
                        episodic_returns.append((global_step, ep_r))
                        writer.add_scalar("charts/episodic_return", ep_r, global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # GAE — A008: bootstrap value uses no_grad (stopped gradient)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO update
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss — clipped surrogate
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                # ENV009: global gradient clipping
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        print(f"[{args.env_id} seed={args.seed}] step={global_step} sps={sps}")

    # Persist episodic returns
    out_payload = {
        "env_id": args.env_id,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "global_step_final": global_step,
        "episodic_returns": episodic_returns,  # list of [step, return]
        "config": asdict(args),
    }
    out_path = os.path.join(args.output_dir, f"returns__{args.env_id}__seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out_payload, f)
    print(f"[OK] wrote {out_path} ({len(episodic_returns)} episodes)")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
