"""Baseline Implementation Agent — generates runnable code for paper reproduction.

Provides:
  - ``run_offline()`` — generates PPO CartPole-v1 implementation (no LLM)
  - ``run_with_sdk()`` — full LLM-powered code generation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.agents.runtime.base import AgentRuntime, ProviderName
from backend.agents.schemas import (
    BaselineResult,
    EnvironmentSpec,
    PaperClaimMap,
    ReproductionContract,
)
from backend.utils.io import read_json

logger = logging.getLogger(__name__)


def _copy_source_pdf_to_code_root(runs_root: Path, project_id: str, code_dir: Path) -> None:
    source_pdf = Path(runs_root) / project_id / "raw_paper.pdf"
    target_pdf = code_dir / "paper.pdf"
    if target_pdf.exists() or not source_pdf.exists():
        return
    target_pdf.write_bytes(source_pdf.read_bytes())


# ---------------------------------------------------------------------------
# PPO CartPole-v1 implementation template
# ---------------------------------------------------------------------------

PPO_TRAIN_PY = '''\
"""PPO CartPole-v1 Baseline — ReproLab generated implementation.

This implements Proximal Policy Optimization (Schulman et al., 2017) on
CartPole-v1 with all assumption decisions applied from the assumption ledger.
"""

import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Config (all assumptions applied)
# ---------------------------------------------------------------------------

CONFIG = {
    # Environment
    "env_id": "CartPole-v1",
    "num_envs": 4,
    "total_timesteps": 500_000,
    # PPO hyperparameters
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,          # A007
    "clip_range": 0.2,
    "entropy_coef": 0.01,        # A008
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,        # A006
    # Training
    "n_steps": 128,
    "n_epochs": 4,
    "batch_size": 64,            # A004: per-minibatch normalization
    "adam_epsilon": 1e-5,        # A001
    # Schedule
    "lr_schedule": "linear",     # A003
    # Evaluation
    "eval_episodes": 100,
    "eval_frequency": 50_000,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Network (A002: orthogonal init)
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization per A002."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # A002: value head std=1.0
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),  # A002: policy head std=0.01
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)


# ---------------------------------------------------------------------------
# PPO Training Loop
# ---------------------------------------------------------------------------

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return thunk


def linear_schedule(initial_lr, total_steps):
    """A003: Linear LR decay."""
    def schedule(step):
        frac = 1.0 - step / total_steps
        return frac * initial_lr
    return schedule


def train():
    cfg = CONFIG
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg["env_id"], cfg["seed"] + i) for i in range(cfg["num_envs"])]
    )
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n

    agent = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(agent.parameters(), lr=cfg["learning_rate"], eps=cfg["adam_epsilon"])
    lr_fn = linear_schedule(cfg["learning_rate"], cfg["total_timesteps"])

    # Storage
    num_steps = cfg["n_steps"]
    obs_buf = torch.zeros((num_steps, cfg["num_envs"], obs_dim))
    actions_buf = torch.zeros((num_steps, cfg["num_envs"]), dtype=torch.long)
    logprobs_buf = torch.zeros((num_steps, cfg["num_envs"]))
    rewards_buf = torch.zeros((num_steps, cfg["num_envs"]))
    dones_buf = torch.zeros((num_steps, cfg["num_envs"]))
    values_buf = torch.zeros((num_steps, cfg["num_envs"]))

    # Training
    global_step = 0
    num_updates = cfg["total_timesteps"] // (num_steps * cfg["num_envs"])
    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    done = torch.zeros(cfg["num_envs"])

    episode_rewards = []
    metrics_history = []
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # LR schedule (A003)
        frac = 1.0 - (update - 1) / num_updates
        lr = lr_fn(global_step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Rollout
        for step in range(num_steps):
            global_step += cfg["num_envs"]
            obs_buf[step] = obs
            dones_buf[step] = done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value.flatten()

            next_obs, reward, terminated, truncated, info = envs.step(action.numpy())
            done = torch.tensor(terminated | truncated, dtype=torch.float32)
            rewards_buf[step] = torch.tensor(reward, dtype=torch.float32)
            obs = torch.tensor(next_obs, dtype=torch.float32)

            # Track episode rewards
            if "final_info" in info:
                for ep_info in info["final_info"]:
                    if ep_info is not None and "episode" in ep_info:
                        episode_rewards.append(ep_info["episode"]["r"])

        # GAE (A007: lambda=0.95)
        with torch.no_grad():
            next_value = agent.get_value(obs).flatten()
        advantages = torch.zeros_like(rewards_buf)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_buf[t + 1]
                nextvalues = values_buf[t + 1]
            delta = rewards_buf[t] + cfg["gamma"] * nextvalues * nextnonterminal - values_buf[t]
            advantages[t] = lastgaelam = delta + cfg["gamma"] * cfg["gae_lambda"] * nextnonterminal * lastgaelam
        returns = advantages + values_buf

        # Flatten
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # PPO update
        batch_size = num_steps * cfg["num_envs"]
        minibatch_size = cfg["batch_size"]
        b_inds = np.arange(batch_size)

        for epoch in range(cfg["n_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                # A004: per-minibatch advantage normalization
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (clipped)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg["clip_range"], 1 + cfg["clip_range"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (A005: clipped)
                newvalue = newvalue.flatten()
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -cfg["clip_range"], cfg["clip_range"]
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy loss (A008)
                entropy_loss = entropy.mean()

                loss = pg_loss - cfg["entropy_coef"] * entropy_loss + cfg["value_loss_coef"] * v_loss

                optimizer.zero_grad()
                loss.backward()
                # A006: gradient clipping
                nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"])
                optimizer.step()

        # Logging
        if global_step % cfg["eval_frequency"] < num_steps * cfg["num_envs"]:
            if episode_rewards:
                recent = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                mean_reward = np.mean(recent)
                metrics_history.append({
                    "step": global_step,
                    "mean_reward": float(mean_reward),
                    "episodes": len(episode_rewards),
                    "lr": lr,
                })
                print(f"Step {global_step:>7d} | Mean Reward: {mean_reward:.1f} | Episodes: {len(episode_rewards)}")

    # Final evaluation
    eval_rewards = []
    eval_env = gym.make(cfg["env_id"])
    for _ in range(cfg["eval_episodes"]):
        obs_eval, _ = eval_env.reset(seed=cfg["seed"])
        total_reward = 0
        done_eval = False
        while not done_eval:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(torch.tensor(obs_eval, dtype=torch.float32).unsqueeze(0))
            obs_eval, reward, terminated, truncated, _ = eval_env.step(action.item())
            total_reward += reward
            done_eval = terminated or truncated
        eval_rewards.append(total_reward)
    eval_env.close()
    envs.close()

    mean_eval_reward = float(np.mean(eval_rewards))
    elapsed = time.time() - start_time

    # Write metrics
    output_dir = Path(os.environ.get("OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "mean_reward": mean_eval_reward,
        "eval_episodes": cfg["eval_episodes"],
        "total_timesteps": cfg["total_timesteps"],
        "elapsed_seconds": elapsed,
        "final_lr": lr,
        "history": metrics_history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Write config
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    print(f"\\nFinal evaluation: mean_reward={mean_eval_reward:.1f} over {cfg[\'eval_episodes\']} episodes")
    print(f"Training time: {elapsed:.1f}s")
    return mean_eval_reward


if __name__ == "__main__":
    train()
'''

PPO_CONFIG_JSON = json.dumps({
    "env_id": "CartPole-v1",
    "total_timesteps": 500000,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "n_steps": 128,
    "n_epochs": 4,
    "batch_size": 64,
    "adam_epsilon": 1e-5,
    "num_envs": 4,
    "seed": 42,
    "assumptions_applied": ["A001", "A002", "A003", "A004", "A005", "A006", "A007", "A008"],
}, indent=2)


def run_offline(
    project_id: str,
    runs_root: Path,
    paper_claim_map: PaperClaimMap,
    environment_spec: EnvironmentSpec,
    reproduction_contract: ReproductionContract | None = None,
    artifact_index: dict[str, Any] | None = None,
) -> BaselineResult:
    """Generate PPO CartPole-v1 implementation (deterministic, no LLM).

    For the hackathon demo, this generates a complete PPO implementation
    with all 8 assumption decisions from the PRD applied.
    """
    code_dir = Path(runs_root) / project_id / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    _copy_source_pdf_to_code_root(Path(runs_root), project_id, code_dir)

    # Write implementation files
    (code_dir / "train.py").write_text(PPO_TRAIN_PY, encoding="utf-8")
    (code_dir / "config.json").write_text(PPO_CONFIG_JSON, encoding="utf-8")

    # Write Dockerfile
    (code_dir / "Dockerfile").write_text(environment_spec.dockerfile, encoding="utf-8")

    # Track commands
    commands_log = [
        "python train.py",
    ]
    (code_dir / "commands.log").write_text("\n".join(commands_log), encoding="utf-8")

    result = BaselineResult(
        mode="implement_from_paper",
        code_path=str(code_dir),
        dockerfile_path=str(code_dir / "Dockerfile"),
        diff_summary=(
            "Generated PPO CartPole-v1 implementation from paper. "
            "Applied all 8 assumption decisions: A001 (Adam epsilon=1e-5), "
            "A002 (orthogonal init), A003 (linear LR decay), "
            "A004 (per-minibatch advantage norm), A005 (value loss clipping), "
            "A006 (grad clip 0.5), A007 (GAE lambda=0.95), A008 (entropy=0.01)."
        ),
        commands_to_run=commands_log,
        assumptions_applied=["A001", "A002", "A003", "A004", "A005", "A006", "A007", "A008"],
    )

    # Write result
    out_path = Path(runs_root) / project_id / "baseline_result.json"
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Baseline implementation written to %s", code_dir)

    return result


async def run_with_sdk(
    project_id: str,
    runs_root: Path,
    paper_claim_map: PaperClaimMap,
    environment_spec: EnvironmentSpec,
    reproduction_contract: ReproductionContract | None = None,
    artifact_index: dict[str, Any] | None = None,
    *,
    model: str | None = None,
    provider: ProviderName | str | None = None,
    runtime: AgentRuntime | None = None,
) -> BaselineResult:
    """Full LLM-powered baseline implementation via the configured agent runtime."""
    from backend.agents.runtime.invoke import collect_agent_text

    project_dir = Path(runs_root) / project_id
    code_dir = project_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    _copy_source_pdf_to_code_root(Path(runs_root), project_id, code_dir)

    context = {
        "paper_claim_map": paper_claim_map.model_dump(),
        "environment_spec": environment_spec.model_dump(),
        "reproduction_contract": reproduction_contract.model_dump() if reproduction_contract else {},
        "artifact_index": artifact_index or {},
    }

    prompt = (
        f"Implement the baseline for project {project_id}.\n"
        f"Write code to {code_dir}\n"
        f"Context:\n```json\n{json.dumps(context, indent=2)}\n```"
    )

    await collect_agent_text(
        "baseline-implementation",
        prompt,
        project_dir=code_dir,
        model=model,
        provider=provider,
        runtime=runtime,
    )

    # Read result from disk or parse from output
    result_path = project_dir / "baseline_result.json"
    if result_path.exists():
        return BaselineResult(**read_json(result_path))

    return BaselineResult(
        mode="implement_from_paper",
        code_path=str(code_dir),
        commands_to_run=["python train.py"],
    )
