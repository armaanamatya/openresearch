"""Baseline Implementation Agent — generates runnable code for paper reproduction.

Provides:
  - ``run_offline()`` — generates PPO CartPole-v1 implementation (no LLM)
  - ``run_with_sdk()`` — full LLM-powered code generation
"""

from __future__ import annotations

import json
import logging
import re
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

# Repo root — used by _load_paper_override to locate docs/papers/<id>.yaml
_REPO_ROOT = Path(__file__).parent.parent.parent

# Regex matching bare arXiv IDs in project_id strings (e.g. "2605.15155",
# "arXiv_2605.15155", "pb_2605_15155_...").  Only the canonical NNNN.NNNNN
# and NNNNN.NNNNN formats are recognised.  Uses a non-digit lookbehind so it
# matches even when preceded by an underscore (e.g. "arXiv_2605.15155").
_ARXIV_ID_RE = re.compile(r"(?<!\d)(\d{4,5}\.\d{4,5})(?!\d)")


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


_NO_STUB_BLOCK = (
    "\n\nNO STUB / NO SURROGATE — hard rule:\n"
    "Your `train.py` MUST be a fully-fledged reproduction. NEVER substitute:\n"
    "  - the paper's model with a `TinyLM`, hand-rolled mini-transformer, or random-init MLP\n"
    "  - the paper's dataset with synthetic / mock / Gaussian / 'ALFWorld-like' data\n"
    "  - the paper's training loop with a no-op that emits zero-everything metrics\n"
    "Even a smoke run loads the REAL pretrained weights named in the paper "
    "(via `transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')` "
    "or the equivalent for other frameworks) and the REAL dataset (HuggingFace Datasets, "
    "the paper's GitHub release, etc.). If the paper's full dataset is too large, use the "
    "paper's *own* released eval split or a public subset — NOT a synthesised stand-in.\n"
    "Scale-down is allowed ONLY along these axes, in this order:\n"
    "  1. shorter training (e.g. 5 steps instead of 150)\n"
    "  2. smaller batch / sequence length\n"
    "  3. fewer eval examples (but real ones, not synthetic)\n"
    "  4. smaller model variant FROM THE SAME FAMILY if the paper offers one "
    "(e.g. Qwen2.5-0.5B if Qwen2.5-3B won't fit — never a random tiny transformer)\n"
    "If even (1)–(4) cannot make the run fit, FAIL the experiment with a clear "
    "`metrics.json = {\"error\": \"compute_infeasible\", \"required_vram_gb\": N}` so the rubric "
    "scorer records honest zero on result-match instead of getting fake numbers from a surrogate.\n"
    "An adapted-down real reproduction scores higher than a complete synthetic surrogate — "
    "the rubric's leaf scorer reads your code AND inspects whether you loaded the paper's "
    "actual model + data.\n"
)

_POD_SETUP_BLOCK = (
    "\n\nRUNPOD SANDBOX — pod env setup (when sandbox=runpod):\n"
    "On RunPod the pod boots from a GENERIC pytorch image (typically "
    "runpod/pytorch:*-py3.10-cuda*-ubuntu22.04). Your Dockerfile is NOT used "
    "to build the pod — it's documentation only.\n"
    "\n"
    "DEPENDENCY INSTALLATION IS HANDLED FOR YOU. The backend will automatically\n"
    "run `python -m pip install --no-cache-dir -r requirements.txt` BEFORE your\n"
    "commands.json entries fire. You do NOT need to repeat this in commands.json.\n"
    "Just list requirements.txt with the deps you need (transformers, accelerate,\n"
    "alfworld, etc., pinned versions).\n"
    "\n"
    "commands.json on runpod should contain ONLY the experiment commands —\n"
    "typically a 1-2 entry list ending in `python train.py`. Example:\n"
    "  [\"alfworld-download 2>&1 || true\", \"python train.py\"]\n"
    "(The `|| true` on alfworld-download tolerates the case where the data\n"
    "is already present from a prior attempt.)\n"
    "\n"
    "Special-case packages that need CUDA dev headers (bitsandbytes, flash-attn, "
    "deepspeed, apex): the default RunPod image is cuda-devel, so dev headers "
    "ARE available. Prefer pre-built wheels from pypi where they exist.\n"
)


# Lane γ: per_model metrics block
_PER_MODEL_METRICS_BLOCK = (
    "\n\nPER-MODEL METRICS — when the paper tests multiple model variants:\n"
    "If the paper specifies more than one model variant (e.g. Qwen2.5-0.5B and\n"
    "Qwen2.5-3B, or BERT-base and BERT-large), your `metrics.json` MUST include\n"
    "a `per_model` dict in addition to any flat top-level metrics. Shape:\n"
    "  {\n"
    "    \"per_model\": {\n"
    "      \"<model_short_name>\": {\n"
    "        \"<metric_name>\": <number>,\n"
    "        ...one entry per metric measured for this model variant...\n"
    "      },\n"
    "      ...one entry per model variant actually run...\n"
    "    },\n"
    "    \"wall_time_seconds\": <number>,\n"
    "    \"scope\": {\n"
    "      \"models_run\": [\"<short_name>\", ...],\n"
    "      \"models_skipped\": [\"<short_name>\", ...]\n"
    "    }\n"
    "  }\n"
    "Model short names MUST be Python-identifier-safe (use underscores, not dots\n"
    "or slashes — e.g. `qwen2_5_3b` not `Qwen/Qwen2.5-3B-Instruct`). You MAY\n"
    "also include flat top-level metrics (e.g. averaged or best-of) for backward\n"
    "compatibility — `per_model` does not replace them, only adds richer detail.\n"
    "If only one model variant is evaluated, omit `per_model` entirely; the flat\n"
    "format is sufficient. Never fabricate `per_model` entries for variants you\n"
    "did not actually run — use `scope.models_skipped` instead.\n"
)

_RUNTIME_DETECTION_BLOCK = (
    "\n\nRUNTIME COMPUTE DETECTION — always-on:\n"
    "Your code MUST detect available compute at runtime and adapt accordingly. "
    "Do NOT hard-code an assumption about GPU availability. The same `train.py` "
    "should work whether the sandbox is CPU-only docker or a GPU-bearing runpod:\n"
    "  - At startup: `import torch; HAS_GPU = torch.cuda.is_available()` "
    "(or the framework equivalent — `jax.devices('gpu')`, `tf.config.list_physical_devices('GPU')`, etc.)\n"
    "  - `device = 'cuda' if HAS_GPU else 'cpu'` and pass through to every model/tensor\n"
    "  - Scale-down on CPU: reduce STEPS and BATCH (per the NO STUB rules above) — "
    "do NOT downgrade model or data identity\n"
    "  - Scale-up on GPU: full batch + epoch count + real datasets — match the paper\n"
    "  - `commands.json` should run ONE entrypoint that branches internally on `HAS_GPU`. "
    "Do NOT write two separate scripts; write one adaptive script.\n"
    "  - For evaluation papers without training: load the real evaluation model and "
    "the real benchmark data. If a remote API is unreachable, FAIL with an explicit "
    "`metrics.json={\"error\":\"api_unreachable\"}` rather than substituting mock outputs.\n"
    "\n"
    "DEVICE-PLACEMENT ORDERING — REQUIRED (the 2026-05-24 Dropout crash):\n"
    "  - `model.to(device)` MUST happen BEFORE `Optimizer(model.parameters(), ...)`. "
    "If you flip this order, the optimizer's internal state tensors (e.g. Adam's "
    "self.m, self.v allocated via zeros_like(p)) end up on CPU while the model's "
    "gradients are on GPU, and optimizer.step() raises "
    "`RuntimeError: tensors on cuda:0 and cpu`.\n"
    "  - Do NOT put `model.to(device)` inside a function that takes `optimizer` as "
    "a parameter — by the time that function runs, the optimizer was already built "
    "from CPU refs and the .to() call is too late.\n"
    "  - Custom optimizer state (m, v, exp_avg, etc.) MUST be allocated with "
    "`torch.zeros_like(p)` AFTER `model.to(device)`, or explicitly `torch.zeros(p.shape, device=p.device)`. "
    "Never use `torch.zeros(p.shape)` without `device=` — that defaults to CPU.\n"
    "  - When in doubt, use the canonical PyTorch order:\n"
    "        model = MyModel().to(device)\n"
    "        optimizer = Optimizer(model.parameters(), ...)\n"
    "        # only NOW pass model + optimizer to your train loop\n"
    "\n"
    "LEARNING-RATE SANITY (the 2026-05-25 Dropout regression):\n"
    "  - For SGD / Momentum / Adam-class optimizers, sane training LRs live\n"
    "    in [1e-4, 1e-1]. NEVER use lr > 1.0 — that immediately produces\n"
    "    train_loss = NaN for every architecture. If the paper mentions a\n"
    "    'scaling factor of 10' or similar, that is NOT the base learning\n"
    "    rate; encode it as `lr = base_lr * scale` rather than `lr = 10`.\n"
    "  - Pre-flight HARD-BLOCKS any optimizer / config with lr outside\n"
    "    [1e-7, 1.0]. The block message includes the file:line of the bad\n"
    "    literal.\n"
    "\n"
    "TRAINING-LOOP HEALTH GUARDS (always-on):\n"
    "  - At the end of EVERY epoch, check `train_loss`. If it is NaN or\n"
    "    Inf, raise RuntimeError('train_loss=NaN at epoch=N, lr=X — abort').\n"
    "    Letting NaN training churn for 500 epochs wastes a pod and produces\n"
    "    no useful artifact.\n"
    "  - Print epoch metrics EVERY epoch (not every 5, not every 25). With\n"
    "    the watchdog's 25-min kill threshold the silence between sparse\n"
    "    prints CAN still trip the kill even for working pods. Per-epoch\n"
    "    prints are also what the live UI uses to render the training-curve\n"
    "    sparkline.\n"
    "  - Flush stdout explicitly after each print (print(..., flush=True)\n"
    "    or sys.stdout.flush()) so the host can see the line within the\n"
    "    rsync window. Python's default line-buffering is OK only if\n"
    "    stdout is a terminal; over SSH it can buffer up to 4 KB.\n"
    "\n"
    "DATASET LOADING — canonical paths only (the 2026-05-25 Adam regression):\n"
    "  - HuggingFace `load_dataset` REQUIRES `owner/name` format. Bare\n"
    "    short names like `load_dataset('imdb')` are deprecated and crash\n"
    "    with `HfUriError: Repository id must be 'namespace/name'`.\n"
    "    Pre-flight HARD-BLOCKS deprecated short names. Use the canonical\n"
    "    owner-prefixed forms:\n"
    "        imdb               → `load_dataset('stanfordnlp/imdb')`\n"
    "        glue               → `load_dataset('nyu-mll/glue', 'sst2')`\n"
    "        squad / squad_v2   → `load_dataset('rajpurkar/squad')`\n"
    "        snli               → `load_dataset('stanfordnlp/snli')`\n"
    "        ag_news            → `load_dataset('fancyzhx/ag_news')`\n"
    "        yelp_polarity      → `load_dataset('fancyzhx/yelp_polarity')`\n"
    "        wikitext           → `load_dataset('Salesforce/wikitext')`\n"
    "    Full registry: backend/agents/rlm/dataset_aliases.py.\n"
    "  - For VISION datasets (MNIST, CIFAR-10/100, SVHN, Fashion-MNIST,\n"
    "    STL10), DO NOT use HuggingFace at all. Use torchvision directly\n"
    "    — it caches to disk, pins URLs, and never breaks on HF schema\n"
    "    drift:\n"
    "        torchvision.datasets.MNIST(root='/artifacts/datasets',\n"
    "                                   train=True, download=True,\n"
    "                                   transform=transform)\n"
    "        torchvision.datasets.CIFAR10(root='/artifacts/datasets', ...)\n"
    "  - When a paper names a dataset (e.g. 'IMDb' / 'CIFAR-10'), look up\n"
    "    the canonical loader in the registry, NOT the bare name in the\n"
    "    paper. The paper is years old; the dataset's hosted location has\n"
    "    moved.\n"
    "\n"
    "DATASET-LOAD FAILURE = SOFT FAILURE — never cancel the whole run:\n"
    "  - When a single dataset fails to load (HF URI broken, torchvision\n"
    "    URL 404, S3 mirror timeout, etc.) catch the exception, do NOT let\n"
    "    it bubble up out of train.py. The other experiments in the run\n"
    "    still have value; a single missing dataset is a SCOPE REDUCTION,\n"
    "    not a run-fail.\n"
    "  - Emit the gap explicitly in metrics.json:\n"
    "        metrics = {\n"
    "          \"experiments\": {\n"
    "             \"mnist_baseline\":  {\"status\": \"ok\",                ...},\n"
    "             \"imdb_baseline\":   {\"status\": \"data_unavailable\",\n"
    "                                 \"reason\": \"HF URI rejected: ...\"},\n"
    "             \"cifar_baseline\":  {\"status\": \"ok\",                ...}\n"
    "          },\n"
    "          \"data_load_failures\": [\n"
    "             {\"dataset\": \"imdb\", \"loader\": \"hf\",\n"
    "              \"error\": \"<exception class + first 200 chars of msg>\"}\n"
    "          ],\n"
    "          ... (the other top-level metrics)\n"
    "        }\n"
    "    Also append the gap to the final_report's `scope.gaps` list. The\n"
    "    rubric grader is data-unavailable-aware: a leaf that depends on a\n"
    "    missing dataset is downweighted, not scored 0. So scope reduction\n"
    "    is HONEST partial reproduction, not failure.\n"
    "  - In the agent's REPL after the run: call propose_improvements only\n"
    "    on the experiments that DID run. The unavailable ones get listed\n"
    "    under scope.gaps with an actionable hint (\"swap to the canonical\n"
    "    `owner/name` HF id\", \"point to a torrent mirror\", \"vendor a\n"
    "    subsample of the dataset into runs/<id>/data/\").\n"
    "  - The ONLY case where a run-level abort is correct is when ZERO\n"
    "    experiments could load any data. In that case raise a clear\n"
    "    `RuntimeError(\"all-experiments-data-unavailable: <list of\n"
    "    datasets>\")` so the harness's failure_classifier tags the run\n"
    "    appropriately.\n"
)


# GPU VRAM estimates (approx, in GB) — keyed by RunPod GPU type string.
# Refresh quarterly when SKU lineup changes. Conservative — assumes
# the SDC version (i.e. single-card capacity).
_GPU_VRAM_ESTIMATE_GB: dict[str, int] = {
    "NVIDIA GeForce RTX 4090": 24,
    "NVIDIA RTX 4090": 24,
    "NVIDIA RTX A6000": 48,
    "NVIDIA A6000": 48,
    "NVIDIA L40S": 48,
    "NVIDIA L40": 48,
    "NVIDIA A40": 48,
    "NVIDIA A100-SXM4-40GB": 40,
    "NVIDIA A100 40GB": 40,
    "NVIDIA A100-SXM4-80GB": 80,
    "NVIDIA A100 80GB": 80,
    "NVIDIA H100": 80,
    "NVIDIA H100 SXM": 80,
    "NVIDIA H200": 141,
}


def _hardware_specs_block(sandbox_mode: object) -> str:
    """Static hardware brief — what GPU + image + disk the agent will actually
    run against. Saves the agent from having to discover via probes and
    prevents OOM-by-batch-size-guessing.

    Only emits when the run targets RunPod. Pulls live values from the
    same env vars that drive the backend (REPROLAB_RUNPOD_*); if those
    differ from what's docked in code, this prompt reflects the truth.
    """
    if "runpod" not in str(sandbox_mode).lower():
        return ""
    import os as _os
    gpu_type = _os.environ.get("REPROLAB_RUNPOD_GPU_TYPE", "").strip()
    gpu_count = _os.environ.get("REPROLAB_RUNPOD_GPU_COUNT", "1").strip()
    cloud_type = _os.environ.get("REPROLAB_RUNPOD_CLOUD_TYPE", "SECURE").strip()
    image = _os.environ.get("REPROLAB_RUNPOD_IMAGE", "").strip()
    container_disk = _os.environ.get("REPROLAB_RUNPOD_CONTAINER_DISK_GB", "50").strip()
    volume_gb = _os.environ.get("REPROLAB_RUNPOD_VOLUME_GB", "20").strip()
    volume_mount = _os.environ.get("REPROLAB_RUNPOD_VOLUME_MOUNT_PATH", "/workspace").strip()
    vram_override = _os.environ.get("REPROLAB_VRAM_OVERRIDE_GB", "").strip()
    if not gpu_type:
        return ""  # no spec to share — quietly skip
    if vram_override:
        vram_gb: int | None = int(vram_override)
    else:
        vram_gb = _GPU_VRAM_ESTIMATE_GB.get(gpu_type)
    vram_line = f"  - VRAM: {vram_gb} GB" if vram_gb else "  - VRAM: unknown (assume ≤24 GB to be safe)"
    return (
        "\n\nSANDBOX HARDWARE BRIEF — your actual runtime:\n"
        f"  - GPU: {gpu_type} × {gpu_count} ({cloud_type} tier)\n"
        f"{vram_line}\n"
        f"  - Base image: {image}\n"
        f"    (torch + torchvision + torchaudio + CUDA libs are PRE-INSTALLED — "
        f"do NOT list them in requirements.txt)\n"
        f"  - Container disk: {container_disk} GB (ephemeral, wiped on pod destroy)\n"
        f"  - Persistent volume: {volume_gb} GB at {volume_mount} (survives pod replacement)\n"
        "Pick batch_size / model_size / sequence_length so the activation memory\n"
        "fits in VRAM with headroom — empirically aim for ≤80% peak. If the paper\n"
        "used a bigger GPU (e.g. 8× H100, 80GB), declare a scope reduction in\n"
        "plan_reproduction (e.g. epochs ÷4, batch ÷2) and the verification rubric\n"
        "will adjust accordingly. NEVER use mocks/surrogates to fit a real model\n"
        "into smaller VRAM — reduce scope, do not substitute.\n"
    )


_ARTIFACT_COMPLETENESS_BLOCK = (
    "\n\nARTIFACT COMPLETENESS — always emit these alongside metrics.json:\n"
    "  $OUTPUT_DIR/\n"
    "    metrics.json          (flat dict + nested per_model — already required)\n"
    "    config_used.json      (every hyperparameter actually used: lr, batch, epochs,\n"
    "                           seed, architecture, framework versions, device)\n"
    "    README.md             (3-section: 'What was reproduced', 'What was omitted and\n"
    "                           why', 'How to read metrics.json' — under 400 words)\n"
    "    training_curves.json  (per-step / per-epoch arrays: {'baseline': {'step':[...],\n"
    "                           'acc':[...]}, 'bn': {...}}). Lets the rubric grader verify\n"
    "                           the convergence-speed-style claims, not just final numbers.\n"
    "    fig_*.png             (matplotlib curves matching the paper's figure layout —\n"
    "                           use matplotlib.use('Agg') so it works in headless docker)\n"
    "Without these the 'Artifact completeness and provenance' rubric area scores 0,\n"
    "and 'Evaluation protocol and metric correctness' loses partial credit because the\n"
    "grader can't verify intermediate-step claims (e.g. 'BN reaches baseline's final\n"
    "acc 60% faster').  All five are cheap — make them all unconditional.\n"
)


_RUBRIC_GUARD_BLOCK = (
    "\n\nSELF-VALIDATING RUBRIC GUARD — always-on:\n"
    "At the END of train.py (after writing metrics.json) call:\n"
    "  from rubric_guard import assert_metrics_schema\n"
    "  assert_metrics_schema(\n"
    "      metrics,\n"
    "      required_keys=[<dotted keys the paper rubric inspects>],\n"
    "      required_artifacts=[<filenames or globs the rubric grader reads>],\n"
    "      artifact_dir=os.environ.get('OUTPUT_DIR', '/artifacts'),\n"
    "  )\n"
    "The guard raises `RubricGuardFailure` (an AssertionError subclass) with a\n"
    "structured JSON detail if any required key is missing from `metrics` OR any\n"
    "required artifact does not exist under `artifact_dir`. That exception text\n"
    "becomes the next iteration's `repair_context` — the more precise the error,\n"
    "the faster you can repair the gap.\n"
    "\n"
    "Required keys / artifacts come from the paper rubric. Derive them from\n"
    "context['paper_targets'] / docs/papers/<arxiv_id>.yaml when present, else\n"
    "pick the minimal set the rubric's leaf descriptions explicitly name. Common\n"
    "always-on artifacts: README.md, training_curves.json, config_used.json,\n"
    "fig_*.png. Common always-on keys: every leaf metric the paper headlines\n"
    "(e.g. 'mnist_baseline_final_acc', 'per_model').\n"
    "\n"
    "ALSO: emit `rubric_guard.py` as a TOP-LEVEL FILE under code/ alongside\n"
    "train.py. Paste the module's source verbatim from\n"
    "`backend/agents/rlm/rubric_guard.py` — it has zero non-stdlib deps so the\n"
    "copy-and-paste route always works under any sandbox. Do NOT add a project\n"
    "import path hack; one file at code/rubric_guard.py is enough for\n"
    "`from rubric_guard import assert_metrics_schema` to resolve.\n"
    "\n"
    "The guard is unconditional — even when the run is a smoke-test, schema\n"
    "completeness must hold; a smoke-test that writes 1 sample is fine, a\n"
    "smoke-test that writes 0 keys is not.\n"
)


_EAGER_METRICS_BLOCK = (
    "\n\nEAGER METRICS EMISSION — always-on:\n"
    "Write `metrics.json` AS YOU GO, not just at the end.  Whenever a sub-experiment "
    "completes (e.g. 'MNIST no-BN final test acc = 0.81'), update `metrics.json` "
    "immediately with that value plus a partial-success marker.  A mid-script kill "
    "(sandbox timeout, OOM, container crash) then still leaves measurable results "
    "for the rubric — the failing path is partial coverage, not zero coverage.\n"
    "Pattern:\n"
    "  def write_metrics(d):\n"
    "      import json, os, tempfile\n"
    "      path = os.path.join(os.environ.get('OUTPUT_DIR', '/artifacts'), 'metrics.json')\n"
    "      tmp = path + '.tmp'\n"
    "      with open(tmp, 'w') as f: json.dump(d, f, indent=2)\n"
    "      os.replace(tmp, path)  # atomic — no half-written file on kill\n"
    "  ...\n"
    "  metrics = {}\n"
    "  metrics['mnist_no_bn_final_acc'] = ...\n"
    "  write_metrics(metrics)         # flush after no-BN finishes\n"
    "  metrics['mnist_bn_final_acc'] = ...\n"
    "  write_metrics(metrics)         # flush after BN finishes\n"
    "  ...\n"
    "Always write atomically (tempfile + os.replace) so a kill mid-write cannot corrupt the file.\n"
)

_DATASET_SETUP_BLOCK = (
    "\n\nDATASET SETUP — required patterns by environment family:\n"
    "Download and verify datasets BEFORE training. Use the canonical tool for each env:\n"
    "\n"
    "ALFWorld:\n"
    "  python -m pip install alfworld          # MUST come first — alfworld-download\n"
    "                                           #   does not exist until the package is installed\n"
    "  alfworld-download                        # downloads ALFWorld env data\n"
    "  assert os.path.exists('/workspace/data/alfworld'), 'ALFWorld data missing'\n"
    "  Data dir: /workspace/data/alfworld (NOT ~/alfworld or ./data)\n"
    "\n"
    "HuggingFace datasets (NQ, HotpotQA, TriviaQA, PopQA, 2WikiMultiHop, MuSiQue …):\n"
    "  from datasets import load_dataset\n"
    "  ds = load_dataset('hotpot_qa', 'distractor', cache_dir='/workspace/data/hf')\n"
    "  assert len(ds) > 0, 'HotpotQA load failed'\n"
    "  Set HF_HOME=/workspace/data/hf and HF_DATASETS_CACHE=/workspace/data/hf/datasets\n"
    "  so repeated runs reuse the cache without re-downloading.\n"
    "\n"
    "WebShop:\n"
    "  python -m pip install webshop-text-env  # or the upstream package from\n"
    "                                           #   https://github.com/princeton-nlp/WebShop\n"
    "  import webshop_text_env; env = webshop_text_env.WebShopEnv()\n"
    "  assert env is not None, 'WebShop env init failed'\n"
    "  Data dir: /workspace/data/webshop\n"
    "\n"
    "General rules:\n"
    "  - The pod filesystem is /workspace-rooted. Always default data dirs to\n"
    "    /workspace/data/<env>, NEVER to ~ or relative paths.\n"
    "  - Emit an explicit assert os.path.exists(...) after EVERY download step.\n"
    "    A missing dataset dir that passes silently will produce zero/NaN metrics.\n"
    "  - Install the package BEFORE invoking any CLI tool it provides — e.g.\n"
    "    `pip install alfworld` must precede `alfworld-download`.\n"
    "  - Export HF_HOME and HF_DATASETS_CACHE in commands.json so the train script\n"
    "    inherits them.\n"
)


# Area-specific repair guidance — keys match the canonical PaperBench area
# names emitted by score_reproduction.  Used by _prior_rubric_feedback_block.
_AREA_REPAIR_HINTS: dict[str, str] = {
    "Data and preprocessing fidelity":
        "use the FULL paper datasets (e.g. all 60K MNIST train samples — no subsampling); "
        "match the paper's exact preprocessing (normalisation constants, augmentation, splits).",
    "Experiment execution and reproducibility":
        "run EVERY model variant and ablation the paper compares; cover all paper-defined steps/epochs; "
        "do not silently skip experiments.  Declare anything genuinely infeasible in "
        "metrics.json['omitted'] with a one-line reason.",
    "Evaluation protocol and metric correctness":
        "report metrics in the paper's exact format — e.g. validation accuracy at fixed checkpoint steps, "
        "test error rate (%) not just loss, mean over the paper's seed count.  Use the metrics.json "
        "key names the rubric checklist above expects.",
    "Result match versus the paper's reported targets":
        "run paper-faithful epoch / step counts so the numbers actually approach the paper's reported "
        "values.  This area is scored from the actual numbers in metrics.json against the paper's targets.",
    "Artifact completeness and provenance":
        "emit figures (matplotlib .png), a README.md describing the run, config_used.json with every "
        "hyperparameter, and per-epoch / per-step training curves so the run is independently verifiable.",
    "Method and code fidelity to the paper":
        "double-check algorithmic invariants in the paper's pseudocode: layer ordering, normalisation "
        "placement, activation function, initialisation scheme, regularisation constants, momentum schedule.",
}


def _prior_rubric_feedback_block(project_dir: Path) -> str:
    """Surface the latest rubric_score event as targeted repair guidance.

    Read the most-recent ``rubric_score`` event from ``dashboard_events.jsonl``
    (written by ``verify_against_rubric``) and produce a prompt section that:

    * tells the agent its previous overall score and the target
    * lists the lowest-scoring areas in priority order
    * pairs each area with a concrete, area-specific repair hint

    Returns ``""`` when no prior rubric event exists (first iteration of the
    run) or when the prior score already meets the target.  This is the
    closed-loop fix that turns the rubric-checklist+verify pair from open-loop
    "trust the root model" into a deterministic prompt-side feedback signal.
    """
    events_file = project_dir / "dashboard_events.jsonl"
    if not events_file.exists():
        return ""

    latest: dict | None = None
    try:
        with events_file.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("event") == "rubric_score":
                    latest = d
    except OSError:
        return ""

    if latest is None:
        return ""

    try:
        score = float(latest.get("score") or 0.0)
        target = float(latest.get("target") or 0.6)
    except (TypeError, ValueError):
        return ""

    if score >= target:
        return ""  # already passing — no repair guidance needed

    areas = latest.get("areas") or []
    if not areas:
        return ""

    # Sort by (status==fail first, lowest score first).  Limit to the worst 5
    # so the prompt stays focused — addressing the top 5 weak areas typically
    # closes most of the score gap.
    def _sort_key(a: dict) -> tuple:
        return (a.get("status") != "fail", float(a.get("score", 1.0)))

    weak = sorted(areas, key=_sort_key)[:5]

    lines = [
        "",
        "PRIOR RUBRIC RESULT — targeted repair guidance:",
        f"  Last iteration scored {score:.3f} / target {target:.2f}.  The lowest-scoring",
        "  areas are listed below.  Concentrate this iteration's changes on these.",
        "",
    ]
    for a in weak:
        name = str(a.get("area", "?"))[:64]
        ascore = float(a.get("score", 0.0))
        weight = float(a.get("weight", 0.0))
        status = a.get("status", "?")
        lines.append(f"    [{status:7s}] {name:<64} score={ascore:.2f}  weight={weight:.3f}")
        hint = _AREA_REPAIR_HINTS.get(name)
        if hint:
            # Wrap hint across two lines so it stays readable in the prompt.
            lines.append(f"               → {hint}")
        lines.append("")

    lines.append(
        "  Weight × (1 − score) gives the residual capacity each area carries.  "
        "Fixing a high-weight low-score area gives the biggest rubric jump.\n"
    )
    return "\n".join(lines)


def _rubric_checklist_block(project_dir: Path) -> str:
    """Return a prompt block listing the top-20 rubric leaves by weight.

    Reads ``runs/<project>/generated_rubric.json`` when present and walks
    ``sub_tasks`` recursively to collect leaf nodes (nodes with no further
    sub_tasks or with sub_tasks=[]).  Leaves are sorted by ``weight``
    descending; the top 20 are formatted as a checklist.

    Returns ``""`` when the rubric file does not exist — no crash, no append.
    """
    rubric_path = project_dir / "generated_rubric.json"
    if not rubric_path.exists():
        return ""

    try:
        rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    def _collect_leaves(node: dict) -> list[dict]:
        children = node.get("sub_tasks") or []
        if not children:
            return [node]
        leaves: list[dict] = []
        for child in children:
            if isinstance(child, dict):
                leaves.extend(_collect_leaves(child))
        return leaves

    # The rubric may be a dict with a top-level list or a bare list.
    if isinstance(rubric, dict):
        top_nodes = rubric.get("sub_tasks") or rubric.get("tasks") or [rubric]
    elif isinstance(rubric, list):
        top_nodes = rubric
    else:
        return ""

    all_leaves: list[dict] = []
    for node in top_nodes:
        if isinstance(node, dict):
            all_leaves.extend(_collect_leaves(node))

    if not all_leaves:
        return ""

    all_leaves.sort(key=lambda n: float(n.get("weight", 0) or 0), reverse=True)
    top = all_leaves[:20]

    lines = ["\n\nRUBRIC CHECKLIST — leaves you'll be scored on (top 20 by weight):"]
    for leaf in top:
        weight = float(leaf.get("weight", 0) or 0)
        req = str(leaf.get("requirements") or leaf.get("description") or "")
        if len(req) > 250:
            req = req[:247] + "..."
        lines.append(f"  [w={weight:.2f}] {req}")

    return "\n".join(lines)


def _load_paper_override(arxiv_id: str | None) -> str:
    """Return a prompt block loaded from ``docs/papers/<arxiv_id>.yaml``.

    The yaml schema is open-ended; the loader formats it as a markdown-style
    prompt block so the agent sees it in a readable format.

    Returns ``""`` when arxiv_id is None, the file doesn't exist, or parsing
    fails — no crash, no append.
    """
    if not arxiv_id:
        return ""

    yaml_path = _REPO_ROOT / "docs" / "papers" / f"{arxiv_id}.yaml"
    if not yaml_path.exists():
        return ""

    try:
        import yaml  # PyYAML — available in the repo venv
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    if not data:
        return ""

    # Format the yaml content as a readable markdown block.
    try:
        import yaml
        formatted = yaml.dump(data, default_flow_style=False, allow_unicode=True)
    except Exception:
        formatted = str(data)

    return (
        f"\n\nPAPER-SPECIFIC GUIDANCE (loaded from docs/papers/{arxiv_id}.yaml):\n"
        + formatted
    )


def _extract_arxiv_id(project_id: str) -> str | None:
    """Extract a bare arXiv ID (e.g. '2605.15155') from a project_id string.

    Handles project IDs like '2605.15155', 'arXiv_2605.15155_abc', and the
    hyphenated form produced by arXiv fetcher normalisation.  Returns None
    when no arXiv ID pattern is found.
    """
    m = _ARXIV_ID_RE.search(project_id)
    return m.group(1) if m else None


def _derive_arxiv_id_from_disk(project_dir: Path) -> str | None:
    """Recover the arXiv ID from on-disk artifacts written during ingest.

    Belt-and-suspenders fallback for callers of ``run_with_sdk`` that do not
    thread ``RunContext.arxiv_id`` through.  Resolution order:

    1. ``artifact_index.json`` → ``paper.arxiv_id`` (most authoritative)
    2. ``demo_status.json``    → ``sourceUrl`` (``arxiv.org/abs/<id>`` URL)
    3. ``demo_status.json``    → ``sourceLabel`` (e.g. ``arxiv_2605.15155.pdf``)
    4. ``None`` — no ID recoverable; caller proceeds without an override.

    Note: ``run_pipeline_rlm`` already reads these files and sets
    ``RunContext.arxiv_id``, which is passed as the ``arxiv_id`` kwarg.  This
    function is only reached when that path is absent (e.g. direct callers
    outside the RLM orchestrator, unit-test harnesses).
    """
    if project_dir is None:
        return None

    # 1. artifact_index.json → paper.arxiv_id
    ai_path = project_dir / "artifact_index.json"
    if ai_path.exists():
        try:
            data = json.loads(ai_path.read_text(encoding="utf-8", errors="replace"))
            aid = (data.get("paper") or {}).get("arxiv_id")
            if aid and _ARXIV_ID_RE.search(str(aid)):
                return str(aid).strip()
        except Exception:  # noqa: BLE001 — corrupt JSON, skip silently
            pass

    # 2 & 3. demo_status.json → sourceUrl or sourceLabel
    ds_path = project_dir / "demo_status.json"
    if ds_path.exists():
        try:
            data = json.loads(ds_path.read_text(encoding="utf-8", errors="replace"))
            # 2. sourceUrl: "https://arxiv.org/abs/2605.15155"
            url = data.get("sourceUrl", "") or ""
            m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4,5}\.\d{4,5})", url)
            if m:
                return m.group(1)
            # 3. sourceLabel: "arxiv_2604.01733.pdf" or similar
            label = data.get("sourceLabel", "") or ""
            m = _ARXIV_ID_RE.search(label)
            if m:
                return m.group(1)
        except Exception:  # noqa: BLE001 — corrupt JSON, skip silently
            pass

    return None


_BUDGET_AWARENESS_BLOCK_TEMPLATE = """
EXECUTION-BUDGET AWARENESS — pick training scale that fits

You have approximately {budget_s} seconds of wall-clock time left for the
sandbox container to run train.py + finish writing metrics.json. When the
budget is exhausted the container is killed and any in-flight epochs are
lost — metrics.json never gets written, the rubric scores 0, and your work
ends up looking like a stub.

Rules:
  - Sum the wall-clock time of every experiment you plan to run inside the
    container (data download + training + figure render) and target AT MOST
    half of the budget above. The other half is reserved for the sub-agent
    bootstrap, image build, container startup, and metrics emission.
  - When CPU is the runtime (torch.cuda.is_available() == False) and the
    budget is under ~1000 s, pick the smoke profile: every experiment uses a
    tight max_steps (e.g. 40-80 steps per optimizer), 1-3 epochs, and a
    minibatch that fits comfortably in 4 GiB RAM. The point is real numbers
    on a smaller scale — not paper-faithful epochs.
  - When CPU and the budget exceeds ~2000 s, scale closer to the paper's
    full epochs but cap individual datasets that would dominate (e.g. cap
    IMDB to 5000 reviews; cap CIFAR-10 to 10 K samples).
  - Always emit a partial metrics.json eagerly — write what you have after
    each experiment finishes so a mid-script kill still leaves measurable
    output for the rubric.

If the paper has multiple experiments and you cannot run them all within
the budget, run a subset and record the omitted experiments in
metrics.json["omitted"] with a one-line reason each. Honest partial
coverage beats a timed-out empty metrics.json.
"""


def _budget_awareness_block(remaining_s: float | None) -> str:
    """Return the budget-awareness prompt block, or an empty string when no budget is known."""
    if remaining_s is None or remaining_s <= 0:
        return ""
    # Round to nearest 10 s — the LLM does not need finer precision.
    rounded = int(remaining_s // 10) * 10
    return "\n" + _BUDGET_AWARENESS_BLOCK_TEMPLATE.format(budget_s=rounded) + "\n"


def _compute_constraint_guidance(
    sandbox_mode: object,
    gpu_mode: object,
    project_dir: Path | None = None,
    arxiv_id: str | None = None,
    remaining_s: float | None = None,
) -> str:
    """Return capability-aware guidance for the implement_baseline agent.

    Goal: the baseline agent writes ONE script that works on CPU OR GPU,
    detecting at runtime via torch.cuda.is_available() (or framework equiv)
    and adapting scale. Hard-coding either mode at build time is wrong —
    the same artifact must run on the local CPU sandbox AND on RunPod GPU.

    Policy overlay on top of the always-on runtime detection:
    - gpu_mode=off → user demands CPU-only; emphasize smoke-test mode is
      the only valid path; GPU branch is dead code (still write it for
      portability, but commands.json must trigger CPU path).
    - gpu_mode=max → user demands GPU; the CPU branch is a safety net
      (still write it so the artifact is portable to a CPU sandbox for
      smoke validation), but commands.json should target the GPU path.
    - gpu_mode in {auto, prefer, None} → no override; the runtime detection
      decides at execution time.

    Sandbox signals are advisory:
    - sandbox=runpod → GPU very likely available; agent should still write
      the detection-branch (some runpod pods are CPU-only).
    - sandbox=docker/local → GPU uncertain; the detection-branch is THE
      protection against assuming wrong.

    Returns the always-on detection block PLUS any policy overlay. The
    agent gets ONE coherent guidance section covering both modes.

    Auth-agnostic by construction (no provider branching).

    Prompt assembly order:
    1. _NO_STUB_BLOCK
    2. _RUNTIME_DETECTION_BLOCK
    3. _POD_SETUP_BLOCK (only when sandbox=runpod)
    4. _DATASET_SETUP_BLOCK (always-on)
    5. Rubric auto-checklist (when generated_rubric.json exists)
    6. Per-paper override (when docs/papers/<arxiv_id>.yaml exists)
    7. REPROLAB_BASELINE_EXTRA_GUIDANCE env-var block
    8. gpu_mode policy overlays (off / max)
    """
    mode_str = str(sandbox_mode).lower() if sandbox_mode else ""
    gpu_str = str(gpu_mode).lower() if gpu_mode else ""

    # 1. NO-STUB block comes FIRST so the agent reads the anti-surrogate hard rule
    # before the runtime-detection nuance.
    # 2. RUNTIME COMPUTE DETECTION — always-on.
    # 2.25. EXECUTION-BUDGET AWARENESS — only when remaining_s is provided
    # (i.e. the calling primitive knows the run-budget deadline). Without this,
    # the agent has previously picked epoch counts that overran the budget
    # without any wall-clock signal.
    # 2.5. PER-MODEL METRICS — multi-scale-paper output shape (Lane γ), follows
    # RUNTIME_DETECTION so the agent understands compute constraints first.
    # Budget block: governed by REPROLAB_BUDGET_AWARENESS_MODE.
    #   - "auto" (default): include only on cost-bearing sandboxes (runpod /
    #     brev) where every minute of overrun maps to real $.  Local docker /
    #     local-process sandboxes pay only with wall-clock; the user can
    #     extend --max-wall-clock if they want paper-faithful epochs.
    #   - "always": include regardless of sandbox.  Useful when the user
    #     wants the agent to scale down even on free local compute.
    #   - "never": skip regardless.  Useful when the user has a big budget
    #     and wants the agent to follow the paper's full epoch counts.
    _COST_BEARING_SANDBOXES = ("runpod", "brev")
    _is_cost_bearing = any(s in mode_str for s in _COST_BEARING_SANDBOXES)
    from backend.config import get_settings as _get_settings
    _budget_mode = (_get_settings().budget_awareness_mode or "auto").lower()
    if _budget_mode == "never":
        _inject_budget = False
    elif _budget_mode == "always":
        _inject_budget = True
    else:  # auto
        _inject_budget = _is_cost_bearing

    guidance = _NO_STUB_BLOCK + _RUNTIME_DETECTION_BLOCK + _EAGER_METRICS_BLOCK
    if _inject_budget:
        guidance += _budget_awareness_block(remaining_s)
    guidance += _PER_MODEL_METRICS_BLOCK

    # 3. RUNPOD POD SETUP — only when sandbox=runpod.
    if "runpod" in mode_str:
        guidance += _POD_SETUP_BLOCK
        # 3.5. Concrete hardware brief — GPU type, VRAM, image, disk. Lets
        # the agent size batches without probing or guessing.
        guidance += _hardware_specs_block(sandbox_mode)

    # 4. DATASET SETUP — always-on; tells the agent how to download real data.
    guidance += _DATASET_SETUP_BLOCK

    # 5. Rubric auto-checklist — when generated_rubric.json exists.
    if project_dir is not None:
        checklist = _rubric_checklist_block(project_dir)
        if checklist:
            guidance += checklist

    # 5.5. Prior-rubric feedback — when a previous iteration produced a
    # rubric_score event, surface the lowest-scoring areas so this iteration
    # concentrates its repair effort where it matters most.  Closes the
    # verify → repair loop that was previously open-loop.
    if project_dir is not None:
        feedback = _prior_rubric_feedback_block(project_dir)
        if feedback:
            guidance += feedback

    # 5.7. Artifact completeness — always-on. Low-weight rubric area but free
    # to nail. Asks for README, figures, config_used.json, per-step curves.
    guidance += _ARTIFACT_COMPLETENESS_BLOCK

    # 5.8. Self-validating rubric guard — always-on. The agent's own train.py
    # imports `rubric_guard.assert_metrics_schema` and calls it at the end of
    # training. A missing key / missing artifact raises RubricGuardFailure
    # whose text becomes the next iteration's repair_context — a loud, precise
    # failure signal before the grader runs.
    guidance += _RUBRIC_GUARD_BLOCK

    # 6. Per-paper YAML override — when docs/papers/<arxiv_id>.yaml exists.
    override = _load_paper_override(arxiv_id)
    if override:
        guidance += override

    # 7. Per-run extra guidance from REPROLAB_BASELINE_EXTRA_GUIDANCE env var.
    # Generic paper-agnostic hook so an operator can scope a specific run
    # without modifying source. Common uses:
    #   - "reproduce only the smallest 2 model variants the paper tests"
    #   - "use a 5% subset of the eval set for time-bounded iteration"
    #   - "skip the multi-seed sweep; one seed=42 is sufficient"
    # The guidance is appended verbatim, so the operator is responsible for
    # phrasing it so it doesn't contradict the NO STUB block above.
    import os as _os
    extra = _os.environ.get("REPROLAB_BASELINE_EXTRA_GUIDANCE", "").strip()
    if extra:
        guidance += (
            "\n\nOPERATOR GUIDANCE — per-run scope override:\n"
            "  " + extra.replace("\n", "\n  ") + "\n"
            "  This guidance does NOT override the NO STUB / NO SURROGATE rule. "
            "If you cannot satisfy the operator's scope AND keep the reproduction "
            "real (paper's actual model + data), fail honestly via "
            "metrics.json={\"error\":\"scope_conflict\",\"detail\":\"...\"}.\n"
        )

    # 8. Policy overlays — explicit gpu_mode=off forces CPU entrypoint;
    #    gpu_mode=max forces GPU entrypoint.
    if gpu_str in {"off", "none"}:
        guidance += (
            "\nPOLICY OVERLAY — --gpu-mode=off:\n"
            "  User explicitly disabled GPU. Even if torch.cuda.is_available() "
            "returns True at runtime, your commands.json MUST trigger only the "
            "CPU/smoke path. The GPU branch in your code is dead code for this "
            "run but still required for portability.\n"
        )
    elif gpu_str == "max":
        guidance += (
            "\nPOLICY OVERLAY — --gpu-mode=max:\n"
            "  User explicitly demands GPU. Sandbox MUST provide one. Your "
            "commands.json should trigger the full-scale (GPU) path. The "
            "CPU branch remains in the code as a safety net for portability + "
            "smoke validation, but is not the primary entrypoint here.\n"
        )
    # auto/prefer/None or sandbox-runpod: no overlay — runtime detection wins.

    return guidance


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
    repair_context: dict[str, Any] | None = None,
    sandbox_mode: object = None,
    gpu_mode: object = None,
    arxiv_id: str | None = None,
    remaining_s: float | None = None,
) -> BaselineResult:
    """Full LLM-powered baseline implementation via the configured agent runtime.

    When ``repair_context`` is set, switches the agent to fix-existing-code mode:
    the prompt instructs it to diagnose the failure and correct the code in place
    rather than rewriting from scratch.

    ``arxiv_id`` — when set (threaded from ``RunContext.arxiv_id`` by the RLM
    primitives layer), takes precedence over ``_extract_arxiv_id(project_id)``
    for the ``docs/papers/<id>.yaml`` override lookup.  This is the P0 fix:
    arXiv-sourced runs receive hashed project IDs (``prj_<digest>``) that the
    regex cannot parse, so the override was dead code on every real arXiv run.
    """
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

    # P0: prefer the explicit arxiv_id (threaded from RunContext.arxiv_id, which
    # was resolved from artifact_index.json / demo_status.json by run_pipeline_rlm)
    # over the fallback regex.  The regex is kept as a fallback for legacy
    # non-hashed project IDs that happen to embed an arXiv ID shaped string.
    # When neither the explicit kwarg nor the regex succeeds (hashed project_id
    # with no ctx.arxiv_id threaded through), also try reading from on-disk
    # artifacts so direct callers of run_with_sdk without a RunContext still get
    # the override (belt-and-suspenders; the primary path is ctx.arxiv_id → kwarg).
    _resolved_arxiv_id = arxiv_id or _extract_arxiv_id(project_id) or _derive_arxiv_id_from_disk(project_dir)
    sandbox_guidance = _compute_constraint_guidance(
        sandbox_mode, gpu_mode, project_dir=project_dir,
        arxiv_id=_resolved_arxiv_id, remaining_s=remaining_s,
    )

    if repair_context:
        prompt = (
            f"The baseline for project {project_id} was already implemented in "
            f"{code_dir}, but running the experiment FAILED. Diagnose the failure "
            f"from the error below and FIX the existing code in place — read the "
            f"current files, find the bug, and correct it. Do NOT rewrite the "
            f"project from scratch. The experiment MUST write its measured numeric "
            f"results as a flat JSON object (metric name → number) to a file named "
            f"metrics.json in the code root, because that file is how the "
            f"reproduction's metrics are read back.\n\n"
            f"Experiment failure:\n```json\n"
            f"{json.dumps(repair_context, indent=2, default=str)}\n```\n\n"
            f"Reproduction context:\n```json\n{json.dumps(context, indent=2)}\n```"
            f"{sandbox_guidance}"
        )
    else:
        prompt = (
            f"Implement the baseline for project {project_id}.\n"
            f"Write code to {code_dir}\n"
            f"The experiment MUST write its measured numeric results as a flat JSON "
            f"object (metric name → number) to a file named metrics.json in the code "
            f"root, because that file is how the reproduction's metrics are read back.\n"
            f"Context:\n```json\n{json.dumps(context, indent=2)}\n```"
            f"{sandbox_guidance}"
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
