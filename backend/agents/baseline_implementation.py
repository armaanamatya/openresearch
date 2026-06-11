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


# Harness-owned helper modules the agent's generated code imports by name. Copied
# verbatim into code/ (2026-05-31 OOM/GPU remediation, comp 2b; extended 2026-06-01
# for the agentic full-scope envs) so the copy-and-paste import route always
# resolves inside any sandbox — mirror of the rubric_guard.py "emit alongside
# train.py" pattern, but a real file copy rather than a prompt instruction:
#   * cell_scheduler.py  — pure-stdlib resume/placement helpers; gpu_cell_runner imports
#                          it BARE, so it must be copied alongside (else the flat-sandbox
#                          `from cell_scheduler import …` falls back to an in-repo
#                          `backend.*` import that doesn't resolve — the bug that zeroed
#                          an SDAR matrix at import on 2026-06-07).
#   * gpu_cell_runner.py — single-cell trainer references its env-var contract.
#   * sdar_env_base.py   — BaseEnv (single-turn) + AgenticEnv (multi-turn) bases.
#   * agentic_rollout.py — multi-turn episode → (sequence_ids, response_mask, reward).
#   * search_qa_env.py   — real retrieval QA env (dense E5 / BM25 / overlap).
#   * alfworld_env.py    — real ALFWorld TextWorld agentic env.
#   * webshop_env.py     — real WebShop agentic env.
# The first three are zero-non-stdlib-dep; the three agentic envs lazy-import their
# heavy deps (rank_bm25 / sentence-transformers / faiss / alfworld), so the COPY +
# bare ``import`` always work and the deps load only when an env is actually used.
_HARNESS_CODE_HELPERS: tuple[str, ...] = (
    "cell_scheduler.py",
    "gpu_cell_runner.py",
    "dead_training_guard.py",  # zero-dep dead-training early-stop detector (imported by gpu_cell_runner)
    "sdar_env_base.py",
    "agentic_rollout.py",
    "search_qa_env.py",
    "alfworld_env.py",
    "webshop_env.py",
    "provenance.py",  # D2: emit_provenance / emit_figure_sidecar — legibility for the grader
    "convergence_evidence.py",  # Module A: structured convergence/sweep evidence (rubric_guard consults it)
    "fair_comparison.py",  # Module B: identical-init snapshot + verifiable init fingerprint
)


def _copy_harness_helpers_to_code_root(code_dir: Path) -> None:
    """Copy the stdlib-only harness helpers into ``code_dir`` (idempotent, fail-soft).

    A failed copy must never abort code generation — the agent can still emit the
    file itself from the prompt as a fallback, so we log and continue.
    """
    import shutil

    src_dir = Path(__file__).parent / "rlm"
    for helper in _HARNESS_CODE_HELPERS:
        try:
            shutil.copy2(src_dir / helper, code_dir / helper)
        except OSError as exc:  # missing source / unwritable dest — non-fatal
            logger.warning("could not copy harness helper %s into code/: %s", helper, exc)


def refresh_harness_helpers(code_dir: str | Path) -> Path:
    """Refresh the vendored stdlib-only harness helpers in ``code_dir`` for $0.

    Public re-copy entry point for the cell-level resume path (Track B): a warm
    retry that skips codegen leaves STALE helper copies in ``code/`` (the agent's
    ``run_with_sdk`` only re-copies them as a side effect of a full codegen pass).
    Resume needs the *current* helper bytes — both so a bug-fix to e.g.
    ``alfworld_env.py`` actually re-runs the affected cells and so the
    fingerprint reflects what is on disk — without paying for a regeneration.

    This is a thin wrapper over :func:`_copy_harness_helpers_to_code_root`: same
    file list, same idempotent + fail-soft semantics, no SDK / LLM call.  The
    directory is created if absent.

    Args:
        code_dir: The run's ``code/`` directory to refresh.

    Returns:
        The resolved ``code_dir`` as a :class:`~pathlib.Path`.
    """
    code = Path(code_dir)
    code.mkdir(parents=True, exist_ok=True)
    _copy_harness_helpers_to_code_root(code)
    return code


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
    "  - the paper's dataset with synthetic / mock / Gaussian / 'paper-like' data\n"
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
_PER_MODEL_METRICS_BLOCK_BASE = (
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
    "\n\nCONVERGENCE / TRAINING-COST CLAIMS — record the TRAJECTORY, not only finals:\n"
    "Many papers' HEADLINE claim is about HOW a method reaches its result — faster\n"
    "convergence, lower training cost, better sample efficiency, fewer iterations to a\n"
    "target — NOT the final scalar (which frequently ties across methods once every\n"
    "method has converged). A `metrics.json` carrying only FINAL accuracy/loss\n"
    "STRUCTURALLY cannot evidence such a claim, and the grader will (correctly) be\n"
    "unable to confirm it — `final_acc(adam) ≈ final_acc(sgd)` looks like a NON-result\n"
    "even when the paper's real claim (`adam converges faster`) is fully reproduced.\n"
    "  - If the paper compares optimizers/methods/configs on convergence speed,\n"
    "    training cost, sample efficiency, or ANY per-epoch / per-step behavior, your\n"
    "    `metrics.json` MUST include the per-epoch (or per-step) TRAJECTORY for each\n"
    "    method alongside the finals, e.g.:\n"
    "      \"history\": {\n"
    "        \"<method>\": {\"epoch\": [0,1,2,...],\n"
    "                       \"train_loss\": [...], \"train_cost\": [...],\n"
    "                       \"val_metric\": [...]},\n"
    "        ...one entry per method compared, on a COMMON x-axis...\n"
    "      }\n"
    "  - Use IDENTICAL initialization (same seed/weights) and a COMMON x-axis (same\n"
    "    epochs/steps) across every method compared — that direct comparability IS the\n"
    "    claim; without it the curves cannot be read against each other.\n"
    "  - Train LONG ENOUGH for the claimed effect to appear: an advantage the paper\n"
    "    shows over the first K epochs must be run for at least K epochs, or the\n"
    "    ordering it demonstrates simply will not be present in your data.\n"
    "  - Keep the final scalars too — the trajectory is ADDITIVE, never a replacement.\n"
)


def _per_model_metrics_block(arxiv_id: str | None = None) -> str:
    """Build the per-model metrics prompt block.

    For multi-env papers (e.g. an RL paper with multiple environments),
    the paper YAML's ``datasets:`` block declares >1 environment.
    ``load_paper_invariants`` exposes that as ``inv.multi_env``. When
    set, we append a nested ``per_model[<model>].per_dataset[<env>]``
    requirement + name the exact environment IDs the rubric grader
    expects. Otherwise the base block is sufficient.
    """
    if not arxiv_id:
        return _PER_MODEL_METRICS_BLOCK_BASE
    try:
        from backend.agents.rlm.paper_invariants import load_paper_invariants
        inv = load_paper_invariants(arxiv_id)
    except Exception:  # noqa: BLE001 — never block on paper-hint loading
        return _PER_MODEL_METRICS_BLOCK_BASE
    if inv is None or not inv.multi_env:
        return _PER_MODEL_METRICS_BLOCK_BASE

    env_list = ", ".join(inv.multi_env)
    env_example = inv.multi_env[0]
    variants = ""
    if inv.models is not None and inv.models.variants_required:
        variants = (
            f"\n  Variants the rubric grader expects (each must appear in "
            f"`per_model` OR be honestly listed in `scope.models_skipped`): "
            f"{', '.join(inv.models.variants_required)}.\n"
        )
    return _PER_MODEL_METRICS_BLOCK_BASE + (
        "\n\nMULTI-ENV METRICS NESTING — this paper benchmarks across\n"
        f"  multiple environments ({env_list}). Your `per_model` dict MUST\n"
        f"  nest a `per_dataset` entry per environment so the rubric grader\n"
        f"  can verify each (model, env) cell independently:\n"
        f"    \"per_model\": {{\n"
        f"      \"<model_short_name>\": {{\n"
        f"        \"per_dataset\": {{\n"
        f"          \"{env_example}\": {{\"<metric>\": <number>, ...}},\n"
        + "".join(
            f"          \"{e}\": {{\"<metric>\": <number>, ...}},\n"
            for e in inv.multi_env[1:]
        ) +
        f"        }},\n"
        f"        ...flat per-model metrics still allowed alongside per_dataset...\n"
        f"      }},\n"
        f"    }}\n"
        f"  Use the EXACT environment keys above ({env_list}) — the rubric\n"
        f"  grader pattern-matches on these short names. Skipped environments\n"
        f"  belong in `scope.envs_skipped` with a one-line reason.\n"
        + variants
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
    "  - Do NOT wrap the backward/optimizer step in a try/except that catches a\n"
    "    CUDA OutOfMemoryError and `continue`s — that silently skips the update,\n"
    "    so the run exits 0 having learned NOTHING (all-zero metrics). If a step\n"
    "    OOMs the config is too big for VRAM: reduce batch_size / rollouts /\n"
    "    seq-len, enable gradient_checkpointing, or — for a model that does not fit\n"
    "    one card — SHARD it across the leased GPUs with Accelerate (see MULTI-GPU\n"
    "    SHARDING below). Let an unexpected OOM RAISE so it is repairable, not buried.\n"
    "\n"
    "MULTI-GPU SHARDING — the harness shards big models FOR you (always-on):\n"
    "  - When >1 GPU is leased AND your training script uses HuggingFace Accelerate,\n"
    "    the harness automatically re-launches it under `accelerate launch` with an\n"
    "    FSDP (full-shard) config — params, gradients AND optimizer state are split\n"
    "    across the cards. A model that OOMs one 24 GB card (e.g. a 3B/7B with full\n"
    "    Adam + a frozen teacher) fits comfortably sharded. You do NOT write the launch\n"
    "    command or any torchrun/FSDP boilerplate — just use the Accelerate API:\n"
    "        from accelerate import Accelerator\n"
    "        accelerator = Accelerator()         # 1 proc on 1 GPU, N procs (FSDP) on N\n"
    "        device = accelerator.device         # use THIS — never a hard-coded cuda:0/cuda:1\n"
    "        model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)\n"
    "        optimizer = torch.optim.AdamW(model.parameters(), lr=...)\n"
    "        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)\n"
    "        loss = ...; accelerator.backward(loss); optimizer.step()   # not loss.backward()\n"
    "  - Frozen teacher / reference model: load it the same way and pass it through\n"
    "    `accelerator.prepare(teacher)` in eval()/no_grad — FSDP shards it too (no\n"
    "    optimizer state → nearly free). `.generate()` runs through the sharded model\n"
    "    directly (fine for short generations) — but ALL ranks must call it together\n"
    "    (see COLLECTIVE DISCIPLINE next).\n"
    "  - COLLECTIVE DISCIPLINE — the #1 multi-GPU footgun, and it is FATAL. After\n"
    "    prepare(), the model is FSDP-SHARDED: every forward / backward / .generate()\n"
    "    / eval pass is a COLLECTIVE that ALL ranks must execute together in lockstep.\n"
    "    NEVER put a model call inside `if accelerator.is_main_process:` / `if rank==0:`\n"
    "    — rank 0 enters the all-gather, the others skip it and block at the next\n"
    "    barrier until the 600 s NCCL watchdog fires and the run SIGABRTs (exit -6).\n"
    "    (This killed a real 4-GPU run on 2026-05-30: a rank-0-only zero-shot\n"
    "    `generate()` before `wait_for_everyone()` deadlocked every rank.)\n"
    "        WRONG: if accelerator.is_main_process: acc = eval_model(model)  # generate on rank 0 → DEADLOCK\n"
    "               accelerator.wait_for_everyone()\n"
    "        RIGHT: acc = eval_model(model)                      # ALL ranks run the forward/generate\n"
    "               acc = accelerator.gather(torch.tensor([acc], device=accelerator.device)).mean().item()\n"
    "               if accelerator.is_main_process: write_metrics({'accuracy': acc})  # I/O ONLY on rank 0\n"
    "    Gate ONLY pure I/O behind is_main_process (print, logging, json.dump, file\n"
    "    writes, wandb.log, tqdm, checkpoint save). Anything that runs the model = ALL\n"
    "    ranks. Every `accelerator.wait_for_everyone()` must be reached by EVERY rank\n"
    "    at the same point — never preceded by a collective that only some ranks run.\n"
    "  - Setup ONCE, not per-rank: guard PURE-I/O one-time prep (dataset downloads,\n"
    "    file extraction) with `if accelerator.is_main_process:` then\n"
    "    `accelerator.wait_for_everyone()` — correct ONLY because downloads touch no\n"
    "    GPU collective (NEVER gate a model call this way — see COLLECTIVE DISCIPLINE).\n"
    "    The harness runs `pip install` before launch — do NOT install inside\n"
    "    train.py, and do NOT re-download per rank.\n"
    "  - The SAME script is correct on 1 GPU (single plain process), on N GPUs (FSDP-\n"
    "    sharded), and on CPU — never branch on device count or hard-pin `cuda:K`.\n"
    "  - Do NOT use `DistributedDataParallel`/`DataParallel` for a model that does not\n"
    "    fit one card: DDP REPLICATES the full model per GPU and STILL OOMs. Accelerate\n"
    "    (FSDP) SHARDS — that is what fixes the OOM.\n"
    "  - A small model that fits one card comfortably (e.g. a 1.7B) can just train on a\n"
    "    single card. If you train several such models sequentially, release each before\n"
    "    the next: `del model, teacher, optimizer; import gc; gc.collect(); torch.cuda.empty_cache()`.\n"
    "  - Print progress metrics on EVERY meaningful step: epoch end\n"
    "    (supervised), rollout end (GRPO / PPO / REINFORCE), policy update\n"
    "    end, or eval batch end. Whichever your training loop uses, emit a\n"
    "    flushed line per step — don't batch the LOGGING. For RL-rollout-based\n"
    "    training, KEEP EACH OPTIMIZER STEP FAST (target ≤2-3 min): the per-step\n"
    "    cost is tasks_per_batch × rollouts_per_prompt × max_episode_steps ×\n"
    "    max_new_tokens generated tokens — budget it small (e.g. 2 × 2 × 8 × 64),\n"
    "    and BATCH the GENERATION across all (task×rollout) sequences in ONE\n"
    "    model.generate() call per env turn (never a Python loop of single-sequence\n"
    "    generates — that is the #1 cause of a step taking 45-80 min and a run that\n"
    "    never completes step 1; prefer vLLM for rollout sampling if available).\n"
    "    Print and check wall-seconds/step; if >3 min, cut rollouts/episode_steps.\n"
    "    Call `heartbeat(\"rollout N/M\")` BEFORE each rollout so the host sees\n"
    "    forward progress even when stdout is quiet. With\n"
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
    "  - SOFT-FAIL ONLY GENUINE DATA-UNAVAILABILITY, NOT YOUR OWN CODE BUGS.\n"
    "    Soft-fail (record + continue) when the DATA cannot be obtained: HTTP\n"
    "    404/403, licence gate, S3/mirror timeout, dataset pulled from the Hub.\n"
    "    Do NOT soft-fail a CODE/CONFIG/API error — a wrong file path, an invalid\n"
    "    model identifier (e.g. a bad HF id), AttributeError/ImportError, a missing\n"
    "    config you were supposed to build, or wrong API usage. Those are YOUR bugs:\n"
    "    let them RAISE so the harness returns them as repair_context and you fix\n"
    "    them next iteration. Masking a code bug as a data_load_failure hides it from\n"
    "    the repair loop and silently drops scope you could have reproduced. Rule of\n"
    "    thumb: 'the data isn't there' → soft-fail; 'my code/id/path/config is wrong'\n"
    "    → raise and fix.\n"
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


# GPU VRAM estimates (approx, in GB) — keyed by canonical GPU model name.
# Used by every cloud-provider hardware brief resolver. Refresh quarterly
# when SKU lineup changes.  Multi-vendor (RunPod GPU strings + Azure VM
# SKUs that embed the GPU model + raw H100/A100/L40S/etc. names).
_GPU_VRAM_ESTIMATE_GB: dict[str, int] = {
    # RunPod-style GPU strings
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
    "NVIDIA H100 NVL": 94,
    "NVIDIA H200": 141,
    "NVIDIA T4": 16,
    "NVIDIA V100": 32,
}


# Azure ML VM SKU → (GPU model, GPU count, VRAM per GPU).  Sourced from
# Microsoft Learn `/azure/virtual-machines/sizes/gpu-accelerated` (verified
# 2026-05-25 via context7 query).  Refresh quarterly — Azure adds H200
# (ND_H200_v5) and NC*as_T4_v3 successors regularly.
_AZURE_VM_SKU_CATALOG: dict[str, tuple[str, int, int]] = {
    # T4 (older, cheap)
    "Standard_NC4as_T4_v3":     ("NVIDIA T4",        1, 16),
    "Standard_NC8as_T4_v3":     ("NVIDIA T4",        1, 16),
    "Standard_NC16as_T4_v3":    ("NVIDIA T4",        1, 16),
    "Standard_NC64as_T4_v3":    ("NVIDIA T4",        4, 16),
    # A10 (NVads A10 v5 — visualization+inference; 8 GB partitioned slices)
    "Standard_NV6ads_A10_v5":   ("NVIDIA A10 (1/6)", 1,  4),
    "Standard_NV12ads_A10_v5":  ("NVIDIA A10 (1/3)", 1,  8),
    "Standard_NV18ads_A10_v5":  ("NVIDIA A10 (1/2)", 1, 12),
    "Standard_NV36ads_A10_v5":  ("NVIDIA A10",       1, 24),
    "Standard_NV72ads_A10_v5":  ("NVIDIA A10",       2, 24),
    # A100 80GB (NCads_A100_v4 — single-VM, PCIe)
    "Standard_NC24ads_A100_v4": ("NVIDIA A100 80GB", 1, 80),
    "Standard_NC48ads_A100_v4": ("NVIDIA A100 80GB", 2, 80),
    "Standard_NC96ads_A100_v4": ("NVIDIA A100 80GB", 4, 80),
    # A100 80GB (NDm_A100_v4 — 8-GPU NVLink scale-out training)
    "Standard_ND96amsr_A100_v4":("NVIDIA A100 80GB", 8, 80),
    # H100 NVL 94 GB (NCads_H100_v5 — single-VM, PCIe)
    "Standard_NC40ads_H100_v5": ("NVIDIA H100 NVL",  1, 94),
    "Standard_NC80adis_H100_v5":("NVIDIA H100 NVL",  2, 94),
    # H100 SXM 80 GB (ND_H100_v5 — 8-GPU NVLink scale-out training)
    "Standard_ND96isr_H100_v5": ("NVIDIA H100 SXM",  8, 80),
    # H200 (ND_H200_v5)
    "Standard_ND96isr_H200_v5": ("NVIDIA H200",      8,141),
}


def _resolve_cloud_hardware(sandbox_mode: object) -> dict | None:
    """Resolve concrete hardware specs from whichever cloud the run targets.

    Multi-cloud — works for RunPod (OPENRESEARCH_RUNPOD_*), Azure ML
    (OPENRESEARCH_AZURE_*), and Brev (OPENRESEARCH_BREV_*).  Returns a normalised
    dict::

        {
          "cloud":          "RunPod" | "Azure ML" | "Brev",
          "gpu":            "NVIDIA L40S",
          "gpu_count":      1,
          "tier":           "SECURE" | "<region>" | "",
          "vram_gb":        48,
          "image":          "runpod/pytorch:..." | "mcr.microsoft.com/azureml/..." | "",
          "container_disk_gb": 50,
          "volume_gb":      20,
          "volume_mount":   "/workspace",
          "vram_known":     True,
        }

    or ``None`` when no provider-specific env is set (e.g. local docker).
    """
    import os as _os

    mode = str(sandbox_mode or "").lower()
    vram_override_str = _os.environ.get("OPENRESEARCH_VRAM_OVERRIDE_GB", "").strip()
    vram_override = int(vram_override_str) if vram_override_str.isdigit() else None

    # --- RunPod ---
    rp_gpu = _os.environ.get("OPENRESEARCH_RUNPOD_GPU_TYPE", "").strip()
    if "runpod" in mode and rp_gpu:
        vram_gb: int | None = vram_override or _GPU_VRAM_ESTIMATE_GB.get(rp_gpu)
        return {
            "cloud": "RunPod",
            "gpu": rp_gpu,
            "gpu_count": int(_os.environ.get("OPENRESEARCH_RUNPOD_GPU_COUNT", "1") or "1"),
            "tier": _os.environ.get("OPENRESEARCH_RUNPOD_CLOUD_TYPE", "SECURE").strip(),
            "vram_gb": vram_gb,
            "vram_known": vram_gb is not None,
            "image": _os.environ.get("OPENRESEARCH_RUNPOD_IMAGE", "").strip(),
            "container_disk_gb": int(_os.environ.get("OPENRESEARCH_RUNPOD_CONTAINER_DISK_GB", "50") or "50"),
            "volume_gb": int(_os.environ.get("OPENRESEARCH_RUNPOD_VOLUME_GB", "20") or "20"),
            "volume_mount": _os.environ.get("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH", "/workspace").strip(),
        }

    # --- Azure ML ---
    az_size = _os.environ.get("OPENRESEARCH_AZURE_VM_SIZE", "").strip()
    if ("azure" in mode or _os.environ.get("OPENRESEARCH_AZURE_REGION")) and az_size:
        sku = _AZURE_VM_SKU_CATALOG.get(az_size)
        if sku is not None:
            gpu_model, gpu_count, per_gpu_vram = sku
        else:
            gpu_model, gpu_count, per_gpu_vram = (az_size, 1, 0)
        vram_gb = vram_override or per_gpu_vram or None
        return {
            "cloud": "Azure ML",
            "gpu": gpu_model,
            "gpu_count": gpu_count,
            "tier": _os.environ.get("OPENRESEARCH_AZURE_REGION", "").strip(),
            "vram_gb": vram_gb,
            "vram_known": vram_gb is not None,
            "image": _os.environ.get(
                "OPENRESEARCH_AZURE_IMAGE",
                "mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:latest",
            ).strip(),
            "container_disk_gb": int(_os.environ.get("OPENRESEARCH_AZURE_DATA_DISK_GB", "100") or "100"),
            "volume_gb": int(_os.environ.get("OPENRESEARCH_AZURE_DATASTORE_GB", "0") or "0"),
            "volume_mount": _os.environ.get("OPENRESEARCH_AZURE_DATASTORE_MOUNT", "/mnt/azureml").strip(),
        }

    # --- Brev ---
    brev_gpu = _os.environ.get("OPENRESEARCH_BREV_GPU_TYPE", "").strip()
    if "brev" in mode and brev_gpu:
        vram_gb = vram_override or _GPU_VRAM_ESTIMATE_GB.get(brev_gpu)
        return {
            "cloud": "Brev",
            "gpu": brev_gpu,
            "gpu_count": int(_os.environ.get("OPENRESEARCH_BREV_GPU_COUNT", "1") or "1"),
            "tier": _os.environ.get("OPENRESEARCH_BREV_REGION", "").strip(),
            "vram_gb": vram_gb,
            "vram_known": vram_gb is not None,
            "image": _os.environ.get("OPENRESEARCH_BREV_IMAGE", "").strip(),
            "container_disk_gb": int(_os.environ.get("OPENRESEARCH_BREV_CONTAINER_DISK_GB", "50") or "50"),
            "volume_gb": 0,
            "volume_mount": "",
        }

    return None


def _hardware_specs_block(sandbox_mode: object) -> str:
    """Static hardware brief — what GPU + image + disk the agent will actually
    run against. Saves the agent from having to discover via probes and
    prevents OOM-by-batch-size-guessing.

    Multi-cloud — emits for RunPod, Azure ML, and Brev, dispatching via
    :func:`_resolve_cloud_hardware`. Returns "" when no cloud-provider
    env is set (e.g. local docker / local process), since there's no
    fixed hardware shape to brief.
    """
    spec = _resolve_cloud_hardware(sandbox_mode)
    if spec is None:
        return ""
    if not spec.get("gpu"):
        return ""
    vram_line = (
        f"  - VRAM: {spec['vram_gb']} GB per GPU"
        if spec.get("vram_known")
        else "  - VRAM: unknown (assume ≤24 GB to be safe)"
    )
    tier_part = f" ({spec['tier']})" if spec.get("tier") else ""
    # Per-cloud image guidance — pre-installed packages differ.
    if spec["cloud"] == "RunPod":
        image_note = (
            "    (torch + torchvision + torchaudio + CUDA libs are PRE-INSTALLED — "
            "do NOT list them in requirements.txt)"
        )
    elif spec["cloud"] == "Azure ML":
        image_note = (
            "    (Azure ML curated environment — PyTorch + CUDA pre-installed "
            "via mcr.microsoft.com/azureml/curated/acpt-pytorch-*; do NOT "
            "re-install torch.  Custom env spec via conda_dependencies.yml "
            "or `docker.dockerfile_path` in your job YAML)"
        )
    else:
        image_note = "    (verify pre-installed packages before adding to requirements.txt)"
    # Volume line is provider-specific.
    if spec.get("volume_gb"):
        volume_line = (
            f"  - Persistent volume: {spec['volume_gb']} GB at {spec['volume_mount']} "
            f"(survives compute replacement)"
        )
    else:
        volume_line = (
            f"  - Datastore mount: {spec.get('volume_mount') or '/mnt/data'} "
            "(Azure ML Datastore — backed by Blob/ADLS; no fixed quota)"
            if spec["cloud"] == "Azure ML"
            else "  - No persistent volume configured"
        )
    return (
        "\n\nSANDBOX HARDWARE BRIEF — your actual runtime:\n"
        f"  - Cloud: {spec['cloud']}\n"
        f"  - GPU: {spec['gpu']} × {spec['gpu_count']}{tier_part}\n"
        f"{vram_line}\n"
        f"  - Base image: {spec.get('image') or '<unset>'}\n"
        f"{image_note}\n"
        f"  - Container disk: {spec['container_disk_gb']} GB (ephemeral, wiped on compute destroy)\n"
        f"{volume_line}\n"
        "Pick batch_size / model_size / sequence_length so the activation memory\n"
        "fits in VRAM with headroom — empirically aim for ≤80% peak. If the paper\n"
        "used a bigger GPU (e.g. 8× H100 80GB), declare a scope reduction in\n"
        "plan_reproduction (epochs ÷4, batch ÷2, fewer experiments) and the\n"
        "scope-adjusted verification rubric will reweight accordingly. NEVER use\n"
        "mocks/surrogates to fit a real model into smaller VRAM — reduce scope,\n"
        "do not substitute.\n"
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
    "FAIL-SOFT the OPTIONAL / VISUALIZATION imports: wrap matplotlib / seaborn / wandb /\n"
    "tensorboard imports AND the figure-writing calls in try/except so a MISSING viz or\n"
    "logging library degrades to 'skip that figure', NEVER an ImportError that aborts the\n"
    "whole training. metrics.json + config_used.json are MANDATORY; figures are best-effort.\n"
    "(2026-06-01: an unguarded top-level `import matplotlib` aborted a full training run\n"
    "with zero metrics — a one-line try/except would have saved the whole run.)\n"
    "MULTI-FAMILY ISOLATION: when train.py runs several experiment families / models\n"
    "sequentially in ONE process, wrap EACH family in try/except and write metrics.json\n"
    "incrementally after each family with that family's status='complete' (top-level\n"
    "status set only at the very end) — one family's crash (a CUDA device-side assert\n"
    "poisons the entire process) must cost one family, never the measured results of\n"
    "the others. Prefer the cells.json + train_cell.py route outright: one process per\n"
    "cell IS the isolation. (2026-06-11: a monolithic Adam repair lost a fully-measured\n"
    "logreg family — 92.6% accuracy already on disk — to a device-side assert in the\n"
    "family that ran after it; the whole matrix scored 0.)\n"
)


_PROVENANCE_BLOCK = (
    "\n\nPROVENANCE MANIFEST (makes your run legible to the TEXT-ONLY grader):\n"
    "The rubric grader cannot SEE your figures and reads only a bounded slice of code, so a\n"
    "faithful run gets docked for details it actually has ('45-epoch not confirmed', 'batch\n"
    "size only an assumption', 'log-axis not verifiable'). Fix this by emitting a machine-\n"
    "readable manifest. The helper `provenance.py` is ALREADY in your code dir:\n"
    "  from provenance import emit_provenance, emit_figure_sidecar\n"
    "  emit_provenance(OUTPUT_DIR, experiments={\n"
    "    '<exp_id>': {'model_key':..., 'baseline':..., 'seed':..., 'epochs':..., 'steps':...,\n"
    "      'batch_size':..., 'per_optimizer': {'adam': {'lr':..., 'betas':...}, ...},\n"
    "      'hardware':..., 'framework_versions': {'torch': torch.__version__},\n"
    "      'convergence': {'iteration':[...], '<metric>':[...]}}})\n"
    "  # for EACH fig_*.png you save, also emit its sidecar so the grader knows the axes:\n"
    "  emit_figure_sidecar(png_path, shows='val accuracy vs epoch (log-x)',\n"
    "    axis={'x': {'label':'epoch','scale':'log'}, 'y': {'label':'accuracy','scale':'linear'}},\n"
    "    series={'adam': {'x':[...], 'y':[...]}})\n"
    "emit_provenance auto-summarizes long convergence arrays (no byte-budget risk). Wrap both\n"
    "calls in try/except (FAIL-SOFT, exactly like the figures). This is the single biggest\n"
    "lever on the 'Artifact completeness', 'Evaluation protocol', and 'Experiment execution'\n"
    "rubric areas.\n"
)


_SMOKE_BLOCK = (
    "\n\nEXECUTION SMOKE — honor OPENRESEARCH_SMOKE_STEPS (a FREE pre-run crash check):\n"
    "Before the full run the harness may launch your entry script with OPENRESEARCH_SMOKE_STEPS\n"
    "set (e.g. =1) and CUDA_LAUNCH_BLOCKING=1. When it is set, run a MINIMAL dry-run:\n"
    "construct EVERY model/experiment you would run for real (especially the riskiest — a\n"
    "VAE, a custom loss) and take that many optimizer steps on a TINY data slice (≈2\n"
    "batches), then sys.exit(0). Skip full epochs, heavy downloads, and figure/metrics\n"
    "writing. Pattern:\n"
    "  import os, sys\n"
    "  SMOKE = int(os.environ.get('OPENRESEARCH_SMOKE_STEPS', '0') or 0)\n"
    "  ...\n"
    "  for step, batch in enumerate(loader):\n"
    "      train_step(batch)\n"
    "      if SMOKE and step + 1 >= SMOKE: break\n"
    "  ...\n"
    "  if SMOKE: sys.exit(0)   # every experiment constructed + stepped without crashing\n"
    "This runs in SECONDS and catches runtime crashes — e.g. a VAE `CUDA error: device-side\n"
    "assert` from feeding non-[0,1] data to a Bernoulli/BCE loss (binarize or [0,1]-scale\n"
    "the VAE's input; do NOT reuse the classifier's mean/std Normalize) — at the REAL line\n"
    "BEFORE the expensive run, so the traceback becomes your repair_context. Ignoring the\n"
    "env var just times the smoke out and skips it, but you LOSE the free crash check.\n"
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
    "\n"
    "STRUCTURED EVIDENCE (convergence / sweep / time-series claims): when the\n"
    "paper's HEADLINE claim is about convergence SPEED, a parameter sweep, or a\n"
    "time-series (your extra guidance will say so and name the exact families),\n"
    "final scalars alone score ~0 on the evaluation-protocol leaves. In that case:\n"
    "  - ALSO emit `convergence_evidence.py` as a top-level file under code/\n"
    "    (paste its source verbatim from\n"
    "    `backend/agents/rlm/convergence_evidence.py` — zero non-stdlib deps), and\n"
    "  - pass `structured_evidence={...}` to `assert_metrics_schema`, matching the\n"
    "    families your guidance names, e.g.\n"
    "      assert_metrics_schema(metrics, required_keys=[...],\n"
    "          structured_evidence={'history_methods': ['adam','sgd_nesterov', ...],\n"
    "                               'sweeps': ['<sweep_name>'],\n"
    "                               'series': ['regret']})\n"
    "    where `history.<exp>.<method>` carries per-epoch curves on a COMMON x-axis\n"
    "    with IDENTICAL initialization across methods, every named sweep's results\n"
    "    live in metrics.json (not only logs), and every named series is an ARRAY\n"
    "    over t (never a lone scalar). A missing curve / sweep / series then raises\n"
    "    RubricGuardFailure with the exact gap so you repair it BEFORE finalizing.\n"
    "    (This enforcement is active only when OPENRESEARCH_FIDELITY_EVIDENCE is set; the\n"
    "    call is harmless otherwise.)\n"
)


# Env interface contract (2026-05-31 OOM/GPU remediation, comp 2c). Self-gating:
# only behaviourally relevant when the reproduction defines `*Env` classes.
_SDAR_ENV_ABC_BLOCK = (
    "\n\nINTERACTIVE ENVIRONMENT INTERFACE — when you define environment classes:\n"
    "If your reproduction implements interactive RL environments (classes whose\n"
    "names end in `Env`, e.g. ALFWorldEnv / WebShopEnv / SearchQAEnv), EACH ONE\n"
    "MUST subclass the harness-provided BaseEnv:\n"
    "  from sdar_env_base import BaseEnv\n"
    "  class ALFWorldEnv(BaseEnv):\n"
    "      def build_student_prompt(self, *args, **kwargs) -> str: ...\n"
    "      def build_teacher_prompt(self, *args, **kwargs) -> str: ...\n"
    "`sdar_env_base.py` is already copied into your code/ root (zero deps). BaseEnv\n"
    "is an ABC: an env missing build_student_prompt / build_teacher_prompt raises\n"
    "TypeError at CONSTRUCTION (cell start) instead of an AttributeError mid-grid —\n"
    "the exact bug that zeroed the 2026-05-31 run's 18 ALFWorld cells. A pre-flight\n"
    "AST check ALSO rejects any *Env defined without these methods and without the\n"
    "BaseEnv base, so subclass it even if your trainer calls only one of the two.\n"
)


# GPU memory discipline (2026-05-31 OOM/GPU remediation, comp 3c) — ALWAYS-ON.
# The ~20 GB fp32 full-vocab log_softmax blowup OOM'd even Qwen3-1.7B on a 24 GB
# card; this is the single highest-leverage fix for "even the smallest model OOMs".
_MEMORY_DISCIPLINE_BLOCK = (
    "\n\nGPU MEMORY DISCIPLINE — always-on (the 2026-05-31 ~20 GB blowup that OOM'd even Qwen3-1.7B):\n"
    "FORBIDDEN: materializing a full-vocab fp32 log-prob tensor. NEVER write\n"
    "  logp = F.log_softmax(logits.float(), dim=-1)   # [B, T, vocab] in fp32 ~= 20 GB, kept in the autograd graph\n"
    "Compute token log-probs WITHOUT the [B, T, vocab] materialization:\n"
    "  - F.cross_entropy(logits.view(-1, V), labels.view(-1), reduction='none')  (negate -> token logp), OR\n"
    "  - torch.gather on logits for the taken tokens + a CHUNKED logsumexp over the vocab dim.\n"
    "Always: bf16 autocast (do NOT upcast logits to fp32), model.config.use_cache=False,\n"
    "model.gradient_checkpointing_enable(), and per-device mini_batch <= 2 for models >= 3B.\n"
    "When the harness sets OPENRESEARCH_CELL_BATCH_SCALE (a float in (0,1]) multiply your\n"
    "per-device batch by it, and when it sets OPENRESEARCH_CELL_GRAD_CHECKPOINT=1 enable\n"
    "gradient checkpointing — these are the harness's per-cell OOM-shrink retries.\n"
)


# Single-cell trainer + matrix manifest contract (comp 3b) — injected only on the
# harness-owned cell path (local/docker GPU). Tells the agent to STOP writing a
# monolithic cuda:0 coordinator and instead emit train_cell.py + cells.json.
_CELL_CONTRACT_BLOCK = (
    "\n\nSINGLE-CELL TRAINER CONTRACT — the harness owns the training matrix:\n"
    "Do NOT write a monolithic coordinator that loops over models/baselines/envs on\n"
    "cuda:0 in one process — that stacks the whole matrix onto one card and OOMs (the\n"
    "2026-05-31 collapse). Instead write ONE script `train_cell.py` that trains exactly\n"
    "ONE cell, and a manifest `cells.json` enumerating the full matrix. The harness runs\n"
    "each cell as its own subprocess pinned to a single GPU (CUDA_VISIBLE_DEVICES=<one id>;\n"
    "the cell sees only cuda:0) and runs min(free_gpus, num_cells) cells in parallel.\n"
    "\n"
    "train_cell.py MUST:\n"
    "  - read its cell from env OPENRESEARCH_CELL_PARAMS (JSON of ONE cells.json entry) and\n"
    "    OPENRESEARCH_CELL_OUTPUT_DIR, plus argv --cell-id / --output-dir;\n"
    "  - train on cuda:0 only — NO torchrun, NO DDP/FSDP, NO device loop, NO 'cuda:1';\n"
    "  - honor OPENRESEARCH_CELL_BATCH_SCALE / OPENRESEARCH_CELL_GRAD_CHECKPOINT (see memory discipline);\n"
    "  - write metrics.json into the output dir as a FLAT leaf dict for THIS cell:\n"
    '      {"status": "ok", "metric": <float>, "steps_run": <int>, "reward_mean": <float>}\n'
    "    The harness nests it at per_model.<model_key>.<env>.<baseline> and aggregates the grid;\n"
    "    do NOT write the per_model nesting yourself in a cell — emit only this cell's leaf.\n"
    "\n"
    "cells.json (a top-level file in code/) enumerates EVERY cell — it is the ONLY place the\n"
    "baseline axis is declared (the harness scope is model x dataset x seed, with no baseline\n"
    "axis), so the matrix is invisible to the harness without it:\n"
    '  {"cells": [{"id": "qwen3_1_7b__sdar__search_qa__s42", "model_id": "Qwen/Qwen3-1.7B",\n'
    '     "model_key": "qwen3_1_7b", "baseline": "sdar", "env": "search_qa", "seed": 42,\n'
    '     "dataset_url": "https://...", "est_vram_gb": 14.0}, ...]}\n'
    "Give est_vram_gb your honest full-FT estimate per cell. Before launching the grid the\n"
    "harness auto-drops any cell whose est_vram_gb exceeds the per-GPU budget (-> scope.gaps)\n"
    "and HEAD-probes each dataset_url (a confirmed 404 -> scope.gaps), so a too-big model or a\n"
    "dead dataset becomes an honest rubric gap instead of an OOM/crash. You do NOT need a\n"
    "commands.json when you provide cells.json + train_cell.py — the harness runs the matrix.\n"
    "STICKY ACROSS ITERATIONS: this holds on EVERY iteration including repairs/improvements.\n"
    "When you refine the method, EDIT train_cell.py + keep cells.json — do NOT collapse the\n"
    "matrix back into a single monolithic train.py (that silently drops you onto the legacy\n"
    "one-process path and forfeits the per-GPU-per-cell OOM safety).\n"
)


def _gpu_budget_brief_block(num_gpus: int, per_gpu_vram_gb: float) -> str:
    """The 'you have N GPUs x M GB; per-cell budget = M GB' brief (comp 3a).

    Emits a concrete numeric budget only when VRAM is known (>0); otherwise a
    conservative note so the agent still scopes deliberately.
    """
    if per_gpu_vram_gb and per_gpu_vram_gb > 0:
        return (
            f"\n\nGPU BUDGET — the harness owns placement (one cell per GPU, never shared/sharded):\n"
            f"You have {num_gpus} GPU(s) x {per_gpu_vram_gb:.0f} GB. The per-cell budget is ONE GPU = "
            f"{per_gpu_vram_gb:.0f} GB.\n"
            f"A model that cannot FULL-fine-tune within {per_gpu_vram_gb:.0f} GB is OUT OF SCOPE: do not put\n"
            f"it in cells.json; record it in scope.models_skipped + scope.gaps. On a ~24 GB card that\n"
            f"means the smallest-two only (e.g. Qwen3-1.7B + Qwen2.5-3B) — NEVER the 7B (its optimizer\n"
            f"state alone exceeds 24 GB). Scope it yourself so the rubric grades only what you intended;\n"
            f"the harness's auto-drop is a backstop, not the plan.\n"
        )
    return (
        f"\n\nGPU BUDGET — the harness owns placement (one cell per GPU):\n"
        f"You have {num_gpus} GPU(s) (per-card VRAM unknown). Size each cell to fit ONE card; prefer the\n"
        f"smallest model variants the paper tests and record larger ones in scope.models_skipped +\n"
        f"scope.gaps rather than risking an OOM that zeros the whole matrix.\n"
    )


# ---------------------------------------------------------------------------
# RL Scaffold guidance block (opt-in: OPENRESEARCH_RL_SCAFFOLD=1)
# ---------------------------------------------------------------------------
_RL_SCAFFOLD_BLOCK = (
    "\n\nRL SCAFFOLD — harness-owned GRPO + vLLM training scaffold:\n"
    "The harness provides a copyable RL-training scaffold in\n"
    "``backend/agents/rlm/rl_scaffold.py``.  Use it instead of writing raw\n"
    "FSDP/generate() loops — it owns the distributed-RL infra so you only\n"
    "inject the paper-specific reward function and custom-loss term.\n"
    "\n"
    "STEP 1 — copy the scaffold verbatim:\n"
    "  Paste ``backend/agents/rlm/rl_scaffold.py`` as ``code/rl_scaffold.py``.\n"
    "  Zero non-stdlib deps at module top-level; trl/vllm are lazy-imported.\n"
    "\n"
    "STEP 2 — emit a thin train.py:\n"
    "  from rl_scaffold import GRPOScaffold, opsd_custom_loss_term, BETA, LAMBDA\n"
    "  scaffold = GRPOScaffold(\n"
    "      model_name=\"Qwen/Qwen3-1.7B\",  # paper's actual model\n"
    "      ref_model_name=\"Qwen/Qwen3-1.7B\",  # teacher = student (self-distill)\n"
    "      reward_fn=my_reward_fn,\n"
    "      custom_loss_term=opsd_custom_loss_term,  # SDAR OPSD; None = plain GRPO\n"
    "      vllm_server_host=os.environ.get('OPENRESEARCH_VLLM_HOST', 'localhost'),\n"
    "      vllm_server_port=int(os.environ.get('OPENRESEARCH_VLLM_PORT', '8000')),\n"
    "      num_trainer_gpus=int(os.environ.get('OPENRESEARCH_TRAINER_GPUS', '1')),\n"
    "      output_dir=os.path.join(os.environ.get('OUTPUT_DIR', '/artifacts'), 'rl_output'),\n"
    "      metrics_path=os.path.join(os.environ.get('OUTPUT_DIR', '/artifacts'), 'metrics.json'),\n"
    "      model_tag='qwen3_1.7b',\n"
    "  )\n"
    "  scaffold.train(dataset)\n"
    "  scaffold.finalize_metrics(\n"
    "      final_eval={...},\n"
    "      required_keys=['per_model', 'baselines_vs_sdar'],\n"
    "      omitted=['alfworld', 'webshop', 'qwen2.5_7b'],\n"
    "  )\n"
    "\n"
    "STEP 3 — emit rl_launch.py from ``rl_scaffold.RL_LAUNCH_TEMPLATE``:\n"
    "  This orchestrator starts the vLLM server on GPU 0, then runs\n"
    "  ``accelerate launch train.py`` on GPUs 1..N (FSDP1, bf16).\n"
    "  When <= 1 GPU is visible it runs train.py directly.\n"
    "\n"
    "STEP 4 — commands.json entry MUST begin with the sentinel comment:\n"
    "  # openresearch:rl-scaffold-owns-launch\n"
    "  python rl_launch.py\n"
    "  (This suppresses the harness's generic accelerate-launch rewriter,\n"
    "  which would conflict with the scaffold's 2-tier launch.)\n"
    "\n"
    "STEP 5 — pin deps in requirements.txt:\n"
    "  trl==0.16.1\n"
    "  vllm==0.7.3\n"
    "  torch==2.5.1+cu121\n"
    "  fastapi uvicorn pydantic requests\n"
    "\n"
    "SDAR OPSD constants (literal — rubric reads the source):\n"
    "  BETA = 10.0   # gate sharpness: g_t = sigmoid(BETA * delta_t)\n"
    "  LAMBDA = 0.1  # composite:     total = grpo_loss + LAMBDA * opsd_loss\n"
    "  Gate is DETACHED (stop-grad): g_t = sigmoid(...).detach()\n"
    "  Divergence: reverse-KL (mode-seeking).\n"
    "\n"
    "Required SDAR metrics keys (emit in metrics.json, per_model shape):\n"
    "  alfworld_success_rate_per_model, searchqa_em_per_model,\n"
    "  webshop_score_per_model, per_model, baselines_vs_sdar, omitted.\n"
    "  Smallest-two scope: Qwen3-1.7B + Qwen2.5-3B, Search-QA only.\n"
    "  Declare ALFWorld / WebShop / 7B in omitted[].\n"
)


# ---------------------------------------------------------------------------
# SDAR baseline-coverage guidance block (opt-in: OPENRESEARCH_SDAR_BASELINES=1)
# ---------------------------------------------------------------------------
# BES Phase 1 — Coverage Completion (spec
# docs/superpowers/specs/2026-06-07-bes-integration/phase-1-coverage-completion.md).
# A "baseline" is purely agent-side: the generated train.py maps a baseline
# STRING to two flags (opsd_enabled, gate_type) inside train_one_run(baseline=...),
# so emitting more baselines is GUIDANCE, not a harness change. The SDAR paper
# reports FIVE (GRPO, OPSD, Skill-SD, GRPO+OPSD, RLSD) but a typical run emits
# only three (GRPO, GRPO+OPSD, SDAR), leaving the heaviest Method leaf
# under-scored. This block instructs the agent to ALSO emit the three missing
# ones — all on the SAME Search-QA env that already runs. It deliberately does
# NOT touch ALFWorld / WebShop: activating an env that cannot yet learn turns
# excluded leaves into counted zeros (the sequencing trap). Search-QA only.
_SDAR_BASELINES_BLOCK = (
    "\n\nSDAR BASELINE COVERAGE — emit ALL FIVE paper baselines (Search-QA only):\n"
    "The SDAR paper reports five baselines; a typical run emits only three\n"
    "(grpo, grpo_opsd, sdar). Emit the three MISSING ones so the full set is\n"
    "{grpo, opsd, skill_sd, grpo_opsd, rlsd} (plus your headline `sdar`). A\n"
    "baseline is JUST A STRING your train.py maps to (opsd_enabled, gate_type)\n"
    "inside train_one_run(baseline=...): e.g.\n"
    "  opsd_enabled = baseline in ('sdar', 'grpo_opsd', 'opsd', 'skill_sd', 'rlsd')\n"
    "  gate_type    = 'sigmoid' if baseline == 'sdar' else 'ones'\n"
    "Add one cell PER missing baseline to code/cells.json carrying that exact\n"
    "`baseline` string; aggregate_cell_metrics then nests each result at\n"
    "per_model[model][env][baseline]. Keep EVERY new cell on the Search-QA env\n"
    "(env='searchqa' / 'search_qa') — do NOT add ALFWorld or WebShop cells here.\n"
    "\n"
    "RECIPE 1 — standalone OPSD (baseline='opsd') — NEAR-FREE, do this FIRST:\n"
    "  OPSD self-distillation loss ONLY, with NO GRPO RL term. Set the GRPO\n"
    "  weight to zero and keep the OPSD term: opsd_enabled=True, grpo_weight=0.0\n"
    "  (i.e. total_loss = 0 * grpo_loss + LAMBDA * opsd_loss, LAMBDA=0.1). Use\n"
    "  the OPSD gate (gate_type='ones'), NOT the sigmoid SDAR gate. The OPSD\n"
    "  machinery already exists — this is a flag flip. Validate the cell→leaf\n"
    "  plumbing with this one before the two below.\n"
    "\n"
    "RECIPE 2 — Skill-SD (baseline='skill_sd'):\n"
    "  Self-distillation WITH a POPULATED skill_context prompt slot. Your\n"
    "  build_prompt(question, skill_context) already accepts skill_context but\n"
    "  it is normally EMPTY (''). For this baseline, retrieve a few relevant\n"
    "  skills/exemplars (reuse your Search-QA retriever — e.g. the top-k E5\n"
    "  passages, or a small fixed skill bank) and feed them as skill_context so\n"
    "  the prompt actually contains them. opsd_enabled=True, gate_type='ones'.\n"
    "  The ONLY structural difference from standalone OPSD is the non-empty\n"
    "  skill_context — make sure build_prompt receives it.\n"
    "\n"
    "RECIPE 3 — RLSD (baseline='rlsd') — RL + self-distillation:\n"
    "  Combine the GRPO RL term WITH self-distillation, a distinct\n"
    "  (opsd_enabled, gate_type, schedule) combination from `sdar`: keep the\n"
    "  GRPO RL term ON (grpo_weight=1.0) AND opsd_enabled=True with\n"
    "  gate_type='ones' (constant gate, NOT the sigmoid SDAR gate) — this is\n"
    "  the 'RL + SD without the learned sigmoid gate' point in the ablation. If\n"
    "  you schedule the OPSD term, anneal it on a fixed schedule rather than the\n"
    "  token-level sigmoid gap gate.\n"
    "\n"
    "Distinctness check (the leaf scorer reads the source): the five must be\n"
    "MECHANICALLY different, not relabelled copies —\n"
    "  grpo       : opsd_enabled=False, gate_type='ones'\n"
    "  opsd       : opsd_enabled=True,  gate_type='ones', grpo_weight=0.0\n"
    "  skill_sd   : opsd_enabled=True,  gate_type='ones', skill_context POPULATED\n"
    "  grpo_opsd  : opsd_enabled=True,  gate_type='ones', grpo_weight=1.0\n"
    "  rlsd       : opsd_enabled=True,  gate_type='ones', grpo_weight=1.0, scheduled SD\n"
    "  sdar       : opsd_enabled=True,  gate_type='sigmoid'  (g_t = sigmoid(BETA*delta_t))\n"
    "\n"
    "PROVENANCE — cite the reference implementation:\n"
    "  Emit an explicit link to the SDAR reference repository\n"
    "  (https://github.com/ZJU-REAL/SDAR) in your run artifacts AND in the\n"
    "  report (e.g. a `provenance` / `reference_repo` field in metrics.json and\n"
    "  a 'Reference implementation: ZJU-REAL/SDAR' line in README.md).\n"
    "\n"
    "CURVES — write per-step curves, not just terminal scalars:\n"
    "  In addition to metrics.json, write `curves.json` in OUTPUT_DIR holding\n"
    "  PER-STEP series so the training dynamics are inspectable. At minimum log,\n"
    "  every step (or every few steps), the four series:\n"
    "    gate_mean  — mean of the SDAR gate g_t over the batch's tokens\n"
    "    gap        — mean teacher-student gap delta_t (the gate's input)\n"
    "    opsd_loss  — the OPSD self-distillation loss term that step\n"
    "    reward     — mean episode/sequence reward that step\n"
    "  Shape: curves.json = {\"step\": [...], \"gate_mean\": [...], \"gap\": [...],\n"
    "  \"opsd_loss\": [...], \"reward\": [...]} (lists aligned by index), written\n"
    "  with the same atomic write pattern as metrics.json.\n"
)


_EAGER_METRICS_BLOCK = (
    "\n\nEAGER METRICS EMISSION — always-on (a timeout must NOT lose completed work):\n"
    "Write the canonical `metrics.json` ATOMICALLY (tmp path + os.replace) after EACH "
    "experiment family/stage finishes — NEVER once at the very end.  As soon as a "
    "family completes (e.g. 'MNIST-MLP final test error = 1.69%'), populate its "
    "MEASURED results into `per_model` (and any per-family results) and flush.  A "
    "wall-clock / stall timeout, OOM, or crash mid-run then still leaves the finished "
    "families' real numbers on disk for the rubric — partial coverage, not zero "
    "coverage.  (The harness's finalize-on-timeout reads the latest on-disk "
    "metrics.json and scores the families already populated, so what is on disk when "
    "the kill lands is exactly what you keep.)\n"
    "Pattern:\n"
    "  def write_metrics(d):\n"
    "      import json, os, tempfile\n"
    "      path = os.path.join(os.environ.get('OUTPUT_DIR', '/artifacts'), 'metrics.json')\n"
    "      tmp = path + '.tmp'\n"
    "      with open(tmp, 'w') as f: json.dump(d, f, indent=2)\n"
    "      os.replace(tmp, path)  # atomic — no half-written file on kill\n"
    "  ...\n"
    "  metrics = {'status': 'running', 'per_model': {}}\n"
    "  metrics['per_model']['mnist_mlp'] = {'test_error': ...}\n"
    "  write_metrics(metrics)         # flush AFTER mnist-mlp finishes (status still non-terminal)\n"
    "  metrics['per_model']['logreg'] = {'final_nll': ...}\n"
    "  write_metrics(metrics)         # flush AFTER logreg finishes\n"
    "  ...\n"
    "The `status` stays non-terminal (`running`) until the very end, but the MEASURED "
    "results for completed families MUST already be on disk after each flush.\n"
    "Always write atomically (tempfile + os.replace) so a kill mid-write cannot corrupt the file.\n"
    "DO NOT MONOLITH (2026-06-08 Adam timeout that zeroed 4 finished families): never pack N "
    "INDEPENDENT experiment families into a single un-checkpointed `python train.py` whose "
    "metrics land only at the end.  If you have multiple independent configs/families, either "
    "(a) checkpoint after each one with the atomic write above, or — PREFERRED for a "
    "multi-config matrix — (b) emit `cells.json` + `train_cell.py` so the harness's "
    "one-GPU-per-cell runner bounds each config with its own timeout and writes per-cell "
    "metrics deterministically (the harness-reliable form of incremental metrics).\n"
    "CAP / STREAM ANY SWEEP: never launch an unbounded hyperparameter grid as the last stage. "
    "Cap the grid to a small set of configs (or stream it), SMALLEST-CONFIG-FIRST, so partial "
    "results land early and a timeout truncates the tail rather than discarding everything.\n"
    "TERMINAL FLUSH IS MANDATORY (2026-05-30): the incremental writes are for liveness + "
    "timeout-survival, but the metrics.json present AT RUN END is what the rubric grades when "
    "the run finishes cleanly. A final metrics.json with a NON-TERMINAL status "
    "(`running`/`pending`/`started`/…) OR with empty `per_model[<model>]` entries is treated as "
    "a FAILED run (failure_class=incomplete_metrics), NOT partial credit, on a CLEAN exit — it "
    "measured nothing the grader can read, so eval/result/execution score ~0. Before the script "
    "exits you MUST (a) set a TERMINAL status (e.g. `completed` / `ok`), and (b) populate "
    "`per_model[<model>]` with the MEASURED eval metric (e.g. accuracy) plus reward/loss for "
    "every model you actually ran. Never leave a `status:\"running\"` placeholder or an empty "
    "`{}` per-model entry as the final state.\n"
)

def _resolve_data_root() -> str:
    """Writable data root for the active sandbox.

    ``run.py`` points ``OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH`` at a writable shared dir for
    LOCAL sandboxes (where ``/workspace`` does not exist); RunPod/Docker keep
    ``/workspace`` (the real pod/container volume). Reading the env var here keeps the
    guidance the agent sees identical to where data actually lands at runtime.
    """
    import os
    return (os.environ.get("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH") or "/workspace").strip() or "/workspace"


def _dataset_setup_block(data_root: str = "/workspace") -> str:
    """DATASET SETUP guidance, rooted at the sandbox's writable ``data_root``.

    ``data_root`` is the writable volume-mount root (``/workspace`` on RunPod/Docker, a
    writable shared cache dir on local). NEVER hardcode ``/workspace`` here: on a local
    host it is unwritable and every dataset download dies at ``os.makedirs``.
    """
    hf_default = f"{data_root}/data/hf"
    return (
        "\n\nDATASET SETUP — required patterns by environment family:\n"
        "Download and verify datasets BEFORE training. Use the canonical tool for each env:\n"
        "\n"
        "EXAMPLES — apply ONLY when the listed environment/dataset is named verbatim in YOUR paper:\n"
        "\n"
        "ALFWorld:\n"
        "  python -m pip install alfworld          # MUST come first — alfworld-download\n"
        "                                           #   does not exist until the package is installed\n"
        "  alfworld-download                        # downloads ALFWorld env data\n"
        f"  assert os.path.exists('{data_root}/data/alfworld'), 'ALFWorld data missing'\n"
        f"  Data dir: {data_root}/data/alfworld (NOT ~/alfworld or ./data)\n"
        "  ENV INIT — use the get_environment FACTORY (modern alfworld removed the\n"
        "  direct AlfredTWEnv class; `alfworld.agents.environment.AlfredTWEnv` no longer\n"
        "  exists and raises AttributeError). The installed package may NOT ship\n"
        "  agents/config/base_config.yaml — do NOT assume that path exists. Resolve the\n"
        "  config robustly: search the package for any base_config.yaml; if none, fetch\n"
        "  the canonical one. Load it into a DICT (not a path) and pass it to the factory:\n"
        "      import os, glob, yaml, urllib.request\n"
        "      from alfworld.agents.environment import get_environment\n"
        "      os.environ['ALFWORLD_DATA'] = f'{data_root}/data/alfworld'\n"
        "      pkg = os.path.dirname(__import__('alfworld').__file__)\n"
        "      hits = glob.glob(os.path.join(pkg, '**', 'base_config.yaml'), recursive=True)\n"
        "      cfg_path = hits[0] if hits else f'{data_root}/data/alfworld/base_config.yaml'\n"
        "      if not os.path.exists(cfg_path):\n"
        "          urllib.request.urlretrieve('https://raw.githubusercontent.com/alfworld/alfworld/master/configs/base_config.yaml', cfg_path)\n"
        "      with open(cfg_path) as fh: config = yaml.safe_load(fh)\n"
        "      config['dataset']['data_path'] = f'{data_root}/data/alfworld/json_2.1.1/train'\n"
        "      # CRITICAL: cap the game count. AlfredTWEnv.init_env() scans EVERY game in\n"
        "      # the train set (~3500 .tw-pddl files) at init — with the default\n"
        "      # num_train_games=-1 that is 20-60 MIN of 100% CPU BEFORE a single rollout\n"
        "      # (the run looks hung at step 0). Cap it small for a budget run:\n"
        "      config['dataset']['num_train_games'] = 24\n"
        "      config['dataset']['num_eval_games'] = 8\n"
        "      env = get_environment('AlfredTWEnv')(config, train_eval='train').init_env(batch_size=1)\n"
        "  Verify env.reset() returns a real observation before counting ALFWorld as\n"
        "  loaded. This is a CODE/API fix — ALFWorld data being present means it is NOT a\n"
        "  data_load_failure; do not soft-skip it on an AttributeError or a missing\n"
        "  config — obtain the config and fix the API call.\n"
        "\n"
        "HuggingFace datasets (NQ, HotpotQA, TriviaQA, PopQA, 2WikiMultiHop, MuSiQue …):\n"
        "  from datasets import load_dataset\n"
        f"  ds = load_dataset('hotpotqa/hotpot_qa', 'distractor', split='validation', streaming=True, cache_dir=os.environ.get('HF_HOME', '{hf_default}'))\n"
        "  rows = list(itertools.islice(ds, 64))   # take ONLY your eval slice\n"
        "  DISK IS SHARED AND LIMITED — never trigger a full-source dataset download:\n"
        "    - Use streaming=True + itertools.islice(ds, N), OR a split slice like\n"
        "      split='validation[:64]'. Cap each env to 32-64 eval tasks.\n"
        "    - NEVER load the full `natural_questions` config — it downloads ~140 GB of\n"
        "      Wikipedia HTML and WILL exhaust the disk and crash the run. For the NQ\n"
        "      in-domain eval use the lightweight 'nq_open' (a few MB) instead, or\n"
        "      stream a small slice. Prefer the lightest variant of EVERY dataset.\n"
        "    - If a dataset only exists as a huge full download, record it in\n"
        "      data_load_failures and skip it (soft failure) rather than downloading it.\n"
        "  If HF_HOME is already exported in the environment, USE IT (do NOT override it);\n"
        f"  otherwise default HF_HOME={hf_default} and HF_DATASETS_CACHE={hf_default}/datasets\n"
        "  so repeated runs reuse the cache without re-downloading.\n"
        "\n"
        "WebShop (needs DATA, not just the package — TRY to download the lightweight\n"
        "catalog; if the download genuinely fails, treat it as a SOFT failure per the\n"
        "rule above: record it in data_load_failures + scope.gaps and CONTINUE with the\n"
        "other envs. Do NOT silently skip it (record the gap) and do NOT hard-assert /\n"
        "abort the run over it):\n"
        "  python -m pip install webshop-text-env  # or upstream princeton-nlp/WebShop\n"
        "  import os, urllib.request\n"
        f"  ws = os.path.join('{data_root}', 'data', 'webshop'); os.makedirs(ws, exist_ok=True)\n"
        "  f = os.path.join(ws, 'items_human_ins.json')\n"
        "  try:\n"
        "    if not os.path.exists(f) or os.path.getsize(f) <= 1000:\n"
        "      urllib.request.urlretrieve('https://raw.githubusercontent.com/princeton-nlp/WebShop/master/data/items_human_ins.json', f)\n"
        "    assert os.path.getsize(f) > 1000  # obtained — WebShop is in scope\n"
        "    webshop_ok = True\n"
        "  except Exception as e:\n"
        "    webshop_ok = False  # SOFT failure — keep ALFWorld/Search-QA, drop only WebShop\n"
        "    data_load_failures.append({'dataset': 'webshop', 'loader': 'http',\n"
        "      'error': f'{type(e).__name__}: {str(e)[:200]}'})\n"
        f"  Data dir: {data_root}/data/webshop  (the few-MB items_human_ins.json catalog)\n"
        "\n"
        "General rules:\n"
        f"  - The writable data root for THIS sandbox is {data_root}. Default ALL data dirs to\n"
        f"    {data_root}/data/<env>, NEVER to /workspace (RunPod-only), ~, or relative paths.\n"
        "  - Use the CANONICAL HuggingFace owner/name for every dataset (e.g. 'hotpotqa/hotpot_qa',\n"
        "    'mandarjoshi/trivia_qa') — the modern Hub REJECTS bare short names with HfUriError.\n"
        "  - MODEL IDS: use the EXACT HuggingFace id, which is NOT always the paper's display\n"
        "    name. Qwen3 base models have NO '-Instruct' suffix → 'Qwen/Qwen3-1.7B' (NOT\n"
        "    'Qwen/Qwen3-1.7B-Instruct', which 404s). Qwen2.5 instruct models DO → 'Qwen/Qwen2.5-3B-Instruct'.\n"
        "    If a scope/paper name like 'Qwen3-1.7B-Instruct' does not resolve, strip/adjust the\n"
        "    suffix to the real Hub id. An invalid model id is a CODE bug — fix it, do NOT soft-skip it.\n"
        "  - Assert os.path.exists(...) after a download you EXPECT to succeed, so a\n"
        "    silent path typo surfaces (a missing dir → zero/NaN metrics). But when a\n"
        "    dataset is genuinely unobtainable in this sandbox (404/403/timeout after a\n"
        "    real attempt), do NOT let the assert abort the run: catch it, append to\n"
        "    data_load_failures + scope.gaps, and continue — the grader excludes those\n"
        "    leaves from the score rather than zeroing them.\n"
        "  - Install the package BEFORE invoking any CLI tool it provides — e.g.\n"
        "    `pip install alfworld` must precede `alfworld-download`.\n"
        "  - Export HF_HOME and HF_DATASETS_CACHE in commands.json so the train script\n"
        "    inherits them.\n"
    )


# Back-compat module alias: the default (/workspace) rendering. Prefer
# _dataset_setup_block(_resolve_data_root()) at prompt-build time so the guidance
# tracks the active sandbox's writable root.
_DATASET_SETUP_BLOCK = _dataset_setup_block()


_OUTPUT_DISCIPLINE_BLOCK = (
    "\n\nOUTPUT DISCIPLINE — write FOCUSED, minimal code:\n"
    "  - Generate only the code needed to advance the rubric. Do NOT re-emit large\n"
    "    files you already wrote, paste duplicate / near-duplicate blocks, or add\n"
    "    verbose narration — terse, non-redundant code is faster, cheaper (output\n"
    "    tokens are never cached) and usually MORE correct.\n"
    "  - Comment only what is non-obvious (the paper's exact invariants); skip\n"
    "    boilerplate. Edit in place rather than rewriting a whole file when a small\n"
    "    change suffices.\n"
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

    # Completeness nudge (Layer 1, lite): the agent already SEES the leaves, but a
    # prior run still skipped a complex component (SFO, a second-order optimizer) by
    # choice and reported only training loss. A simplified-but-present implementation
    # earns partial credit; an OMITTED component scores 0. So push best-effort coverage
    # of EVERY leaf + the evaluation metrics the leaves ask for, not just training loss.
    lines.append(
        "\nCOVER EVERY LEAF: implement a best-effort version of each item above — do NOT "
        "skip one just because it's complex (a basic correct version of a hard component, "
        "e.g. a second-order optimizer, beats absence: partial credit vs 0). Report the "
        "TEST/validation metrics the leaves name (accuracy, ELBO, etc.), not only training "
        "loss. If you must reduce scope, reduce SIZE (fewer epochs/steps), never OMIT a "
        "whole experiment or baseline a leaf asks for."
    )
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


_MINIMIZE_COMPUTE_BLOCK = (
    "\n\nMINIMIZE-COMPUTE MODE — reproduce the CLAIM, not the recipe:\n"
    "The user enabled minimize-compute (--minimize-compute or the lab UI\n"
    "checkbox), which means the paper's training schedule is allowed to be\n"
    "substituted with a modern fast equivalent AS LONG AS the metric claim\n"
    "is still verifiable. The paper's reported number is what you reproduce.\n"
    "The way they got there is editable when their way is a historical\n"
    "artefact (slow optimizers from 2012, 3000-epoch schedules, etc.).\n"
    "\n"
    "Substitution rules (apply when the paper's recipe is in the LEFT column):\n"
    "  • SGD + linear-decay-from-10 × 3000 epochs   →   Adam @ lr=0.001 × 200-500 epochs\n"
    "  • SGD + momentum × 1000+ epochs              →   Adam @ lr=0.001 × 100-300 epochs\n"
    "  • LR schedule starting at lr > 1.0           →   Adam OR SGD with init_lr ≤ 0.1\n"
    "  • Cosine schedule over 1000 epochs           →   cosine over 100 epochs (same min_lr)\n"
    "  • 500+ epochs with no warmup                 →   100-200 epochs with brief warmup\n"
    "\n"
    "Architectures (model parameters) are NEVER substituted — only the\n"
    "training schedule is. Replacing the model's hidden size or depth is a\n"
    "claim violation, not a recipe substitution.\n"
    "\n"
    "Datasets are NEVER substituted (in scope or scale). If a dataset is\n"
    "unavailable, that's a DATASET-LOAD failure (handled by the soft-fail\n"
    "block above), not a minimize-compute substitution.\n"
    "\n"
    "Every substitution MUST appear in scope.declared_reductions:\n"
    "  scope = {\n"
    "    \"requested\": \"<paper's full scope>\",\n"
    "    \"ran\": [...],\n"
    "    \"gaps\": [],\n"
    "    \"declared_reductions\": [\n"
    "      {\n"
    "        \"axis\": \"training_schedule\",\n"
    "        \"paper\": \"SGD+linear-decay-from-10 × 3000 epochs\",\n"
    "        \"actual\": \"Adam@lr=0.001 × 300 epochs\",\n"
    "        \"rationale\": \"minimize_compute=True; paper's recipe was a 2012-era SGD schedule; modern Adam reaches the paper's reported test error in ~10% of compute\"\n"
    "      }\n"
    "    ]\n"
    "  }\n"
    "\n"
    "The scope-adjusted rubric reads `declared_reductions` and weights the\n"
    "training-schedule leaves at 0 (they're scope-reduced) while keeping\n"
    "the metric-match leaves at full weight. Net effect: minimize-compute\n"
    "runs that match the paper's reported test error score well on the\n"
    "scope-adjusted rubric; runs that diverge from the metric score poorly\n"
    "regardless of minimize-compute.\n"
    "\n"
    "When to NOT minimize compute (override the user):\n"
    "  - Paper's central claim IS about the training schedule (curriculum\n"
    "    learning, warmup ablation, schedule comparison study) — then the\n"
    "    schedule IS the claim. Run faithfully and surface this in\n"
    "    scope.declared_reductions=[] with `notes=\"paper claim is schedule-dependent\"`.\n"
    "  - Paper studies long-horizon convergence (overfitting onset, final\n"
    "    plateau test error after 3000 ep) — short runs can't validate.\n"
    "    Note in scope, but follow the paper's epoch count.\n"
)


def _metrics_shape_binding_block(metrics_shape: list[dict]) -> str:
    """Return a prompt block binding the agent to its declared metrics_shape.

    Only injected when plan_reproduction produced a non-empty metrics_shape.
    The table format makes the path → id mapping unambiguous; the contract
    sentence closes the loop with rubric_guard.
    """
    if not metrics_shape:
        return ""
    rows = []
    for mp in metrics_shape:
        metric_id = str(mp.get("metric_id", "")).strip()
        json_path = str(mp.get("json_path", "")).strip()
        if metric_id and json_path:
            rows.append(f"  {metric_id:<44} {json_path}")
    if not rows:
        return ""
    table = "\n".join(rows)
    return (
        "\n\nMETRICS CONTRACT — binding from plan_reproduction:\n"
        "Your reproduction contract DECLARES these metric paths in metrics.json:\n\n"
        "  metric_id                                    json_path\n"
        "  " + "-" * 42 + " " + "-" * 45 + "\n"
        + table + "\n\n"
        "Your train.py MUST emit metrics.json with EXACTLY these dotted paths. "
        "Any deviation will fail rubric_guard's contract check. Use these paths "
        "verbatim — do not re-shape them. The eagerly-written metrics.json "
        "(see EAGER METRICS EMISSION above) MUST also follow these paths.\n"
    )


def _data_recipes_binding_block(data_recipes: list[dict] | None) -> str:
    """Return a prompt block binding the agent to canonical dataset loaders.

    Only injected when plan_reproduction found matching recipes in the paper
    text. Parallel to _metrics_shape_binding_block (PR-θ): a contract-binding
    table followed by a hard instruction.

    Columns:  Dataset  |  Import  |  Loader
    Normalization stats and license notes are appended per-recipe so the agent
    sees them in context rather than as a separate lookup.
    """
    if not data_recipes:
        return ""

    rows: list[str] = []
    notes_lines: list[str] = []
    for r in data_recipes:
        name = str(r.get("canonical_name", "")).strip()
        imp = str(r.get("canonical_import", "")).strip()
        loader = str(r.get("canonical_loader", "")).strip()
        if not name:
            continue
        # Truncate import for table readability (full values in notes). Loaders are
        # shown fuller because truncating a URL mid-path turns a working recipe into
        # a footgun — the agent inlines whatever it sees.
        imp_short = (imp[:42] + "…") if len(imp) > 43 else imp
        loader_short = (loader[:200] + "…") if len(loader) > 201 else loader
        rows.append(f"  {name:<22} {imp_short:<44} {loader_short}")

        # Per-recipe supplementary notes.
        norm = r.get("normalization_stats")
        if norm:
            mean_s = str(norm.get("mean", ""))
            std_s = str(norm.get("std", ""))
            notes_lines.append(
                f"  {name}: apply transforms.Normalize(mean={mean_s}, std={std_s})"
            )
        lic = str(r.get("license_note", "") or "").strip()
        if lic:
            notes_lines.append(f"  {name} license: {lic}")
        note = str(r.get("notes", "") or "").strip()
        if note:
            notes_lines.append(f"  {name}: {note}")
        # PR-ν.1: surface fallback_mirrors so the agent has an actionable URL list
        # to try on 403/404/network errors — the canonical_loader already names the
        # primary, but without the alternates the agent can't recover gracefully.
        mirrors = r.get("fallback_mirrors") or ()
        if mirrors:
            mirror_list = ", ".join(str(m) for m in mirrors)
            notes_lines.append(f"  {name} fallback mirrors: {mirror_list}")

    if not rows:
        return ""

    header = (
        "  " + f"{'Dataset':<22} {'Import':<44} Loader\n"
        "  " + "-" * 22 + " " + "-" * 44 + " " + "-" * 40
    )
    table = "\n".join(rows)
    supp = ("\n\nNotes per dataset:\n" + "\n".join(notes_lines)) if notes_lines else ""

    # PR-ξ: prepend a hard import contract for STRICT-severity recipes whose
    # helper has already been written to _reprolab_curated.py. This makes the
    # contract structurally unavoidable rather than advisory-text-only.
    strict_contracts: list[str] = []
    for r in data_recipes:
        if (r.get("severity") == "strict"
                and r.get("helper_name")
                and r.get("helper_body")):
            hn = str(r["helper_name"])
            banned = [str(b) for b in (r.get("banned_literals") or ())]
            strict_contracts.append(
                f"  Dataset: {r.get('canonical_name', hn)}\n"
                f"    In train.py you MUST write:\n"
                f"        from _reprolab_curated import {hn}\n"
                f"    and CALL {hn}(...) to obtain the dataset. "
                f"Do NOT inline the loader body.\n"
                f"    The helper is already written at code_dir/_reprolab_curated.py.\n"
                + (
                    "    Banned literal patterns (will fail postflight if found in train.py):\n"
                    + "".join(f"        {b}\n" for b in banned)
                    if banned else ""
                )
            )

    strict_block = ""
    if strict_contracts:
        strict_block = (
            "\n\nCURATED LOADER CONTRACT (STRICT) — applies to these datasets:\n"
            + "\n".join(strict_contracts)
        )

    return (
        strict_block
        + "\n\nDATASET LOADING — use these canonical loaders verbatim:\n\n"
        + header + "\n"
        + table + "\n\n"
        "Use the import lines and loader expressions in the table EXACTLY. "
        "Do NOT use bare short names (e.g. load_dataset('imdb')) — the modern "
        "HuggingFace Hub requires owner/name format and rejects bare short names "
        "with HfUriError. For vision datasets (MNIST, CIFAR-*, STL-10, SVHN) "
        "use torchvision.datasets directly — faster, no HF schema drift risk.\n"
        "If a dataset has license_note: it cannot be auto-downloaded — declare "
        "it in scope.gaps[] instead of substituting a surrogate.\n"
        + supp + "\n"
    )


def _compute_constraint_guidance(
    sandbox_mode: object,
    gpu_mode: object,
    project_dir: Path | None = None,
    arxiv_id: str | None = None,
    remaining_s: float | None = None,
    minimize_compute: bool = False,
    metrics_shape: list[dict] | None = None,
    data_recipes: list[dict] | None = None,
    gpu_parallelism: str | None = None,
    gpu_visible_count: int | None = None,
    gpu_cell_budget: dict | None = None,
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
    7. OPENRESEARCH_BASELINE_EXTRA_GUIDANCE env-var block
    8. gpu_mode policy overlays (off / max)
    """
    mode_str = str(sandbox_mode).lower() if sandbox_mode else ""
    gpu_str = str(gpu_mode).lower() if gpu_mode else ""

    # Harness-owned cell path (2026-05-31 OOM/GPU remediation, comp 3): active on a
    # local/docker backend that exposes >=1 GPU. On this path the agent writes a
    # single-cell train_cell.py + cells.json and the harness runs the matrix one
    # GPU per cell — so the GPU-budget brief + cell contract are injected and the
    # multi-GPU torchrun parallelism guidance is SUPPRESSED (it would tell the
    # agent to shard one big job across cards, the opposite of one-cell-per-card).
    _cell_budget = gpu_cell_budget or {}
    _cell_num_gpus = int(_cell_budget.get("num_gpus", 0) or 0)
    _cell_per_gpu_gb = float(_cell_budget.get("per_gpu_vram_gb", 0.0) or 0.0)
    _cell_path_active = (
        str(_cell_budget.get("backend_kind", "")).lower() in ("local", "docker")
        and _cell_num_gpus >= 1
    )

    # 1. NO-STUB block comes FIRST so the agent reads the anti-surrogate hard rule
    # before the runtime-detection nuance.
    # 2. RUNTIME COMPUTE DETECTION — always-on.
    # 2.25. EXECUTION-BUDGET AWARENESS — only when remaining_s is provided
    # (i.e. the calling primitive knows the run-budget deadline). Without this,
    # the agent has previously picked epoch counts that overran the budget
    # without any wall-clock signal.
    # 2.5. PER-MODEL METRICS — multi-scale-paper output shape (Lane γ), follows
    # RUNTIME_DETECTION so the agent understands compute constraints first.
    # Budget block: governed by OPENRESEARCH_BUDGET_AWARENESS_MODE.
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
    # comp 3c: memory discipline is always-on (the fp32 full-vocab logprob blowup
    # OOMs regardless of backend). comp 3a/3b (budget brief + cell contract) ride
    # the harness-owned cell path only.
    guidance += _MEMORY_DISCIPLINE_BLOCK
    if _cell_path_active:
        guidance += _gpu_budget_brief_block(_cell_num_gpus, _cell_per_gpu_gb)
        guidance += _CELL_CONTRACT_BLOCK
    if _inject_budget:
        guidance += _budget_awareness_block(remaining_s)
    # Lane AA — per-model block adapts to multi-env papers
    # by nesting per_dataset under each model. Arxiv id drives the lookup.
    guidance += _per_model_metrics_block(arxiv_id=arxiv_id)
    # Per-paper negative lessons (MUSE-lite, OPENRESEARCH_NEGATIVE_LESSONS):
    # advisory failure memory mined from prior runs of this arxiv_id. Flag-gated
    # + fail-soft; returns "" when off / paper unknown / nothing promoted.
    if project_dir is not None and arxiv_id:
        try:
            from backend.agents.rlm.lesson_distiller import negative_lessons_block
            _neg = negative_lessons_block(Path(project_dir).parent, arxiv_id)
            if _neg:
                guidance += "\n\n" + _neg + "\n"
        except Exception:  # noqa: BLE001 — advisory memory must never break the prompt
            pass
    # Lane Q — minimize-compute substitution rules + scope.declared_reductions
    # contract. Only injected when the user opted in via the CLI flag or the
    # lab UI checkbox; strict reproduction stays the default.
    if minimize_compute:
        guidance += _MINIMIZE_COMPUTE_BLOCK

    # 3. RUNPOD POD SETUP — only when sandbox=runpod.
    if "runpod" in mode_str:
        guidance += _POD_SETUP_BLOCK
        # 3.5. Concrete hardware brief — GPU type, VRAM, image, disk. Lets
        # the agent size batches without probing or guessing.
        guidance += _hardware_specs_block(sandbox_mode)

    # 4. DATASET SETUP — always-on; tells the agent how to download real data,
    #    rooted at the sandbox's writable data root (/workspace only on RunPod/Docker;
    #    a writable shared cache on local — see run._ensure_local_data_root).
    guidance += _dataset_setup_block(_resolve_data_root())

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
    guidance += _OUTPUT_DISCIPLINE_BLOCK
    guidance += _ARTIFACT_COMPLETENESS_BLOCK
    guidance += _PROVENANCE_BLOCK
    # Layer 1: only ask the agent to write smoke-aware code when the execution smoke
    # is actually enabled — keeps the prompt lean otherwise (flag default-OFF).
    try:
        from backend.agents.rlm import execution_smoke as _exec_smoke
        if _exec_smoke.is_enabled():
            guidance += _SMOKE_BLOCK
    except Exception:  # noqa: BLE001 — guidance assembly must never fail the run
        pass

    # 5.8. Self-validating rubric guard — always-on. The agent's own train.py
    # imports `rubric_guard.assert_metrics_schema` and calls it at the end of
    # training. A missing key / missing artifact raises RubricGuardFailure
    # whose text becomes the next iteration's repair_context — a loud, precise
    # failure signal before the grader runs.
    guidance += _RUBRIC_GUARD_BLOCK

    # 5.82. Env interface contract — always-on, self-gating (only matters when
    # the agent defines *Env classes). Pairs with sdar_env_base.BaseEnv (copied
    # into code/) + the preflight_ast env-contract backstop.
    guidance += _SDAR_ENV_ABC_BLOCK

    # 5.85. RL scaffold guidance — opt-in (OPENRESEARCH_RL_SCAFFOLD=1).
    # Tells the agent to copy rl_scaffold.py, write a thin train.py with
    # GRPOScaffold + the OPSD custom-loss term, emit rl_launch.py, and pin
    # trl/vllm/torch in requirements.txt.
    # DEFAULT OFF → not injected → guidance byte-identical to today.
    import os as _os_scaffold
    if _os_scaffold.environ.get("OPENRESEARCH_RL_SCAFFOLD", "").strip().lower() in ("1", "true", "yes"):
        guidance += _RL_SCAFFOLD_BLOCK

    # 5.86. SDAR baseline-coverage guidance — opt-in (OPENRESEARCH_SDAR_BASELINES=1).
    # BES Phase 1 (Coverage Completion). Tells the agent to ALSO emit the three
    # missing SDAR baselines (standalone OPSD, Skill-SD, RLSD) so all five are
    # present, plus provenance link + per-step curves.json. Search-QA only — it
    # deliberately does NOT activate ALFWorld/WebShop env cells (the sequencing
    # trap: an env that can't learn turns excluded leaves into counted zeros).
    # DEFAULT OFF → not injected → guidance byte-identical to today.
    if _os_scaffold.environ.get("OPENRESEARCH_SDAR_BASELINES", "").strip().lower() in ("1", "true", "yes"):
        guidance += _SDAR_BASELINES_BLOCK

    # 5.9. θ: metrics_shape binding — when plan_reproduction declared a non-empty
    # metrics_shape, bind the agent to those exact paths. Injected after the
    # rubric-guard block so the agent sees the contract in the context of the
    # guard that enforces it.
    if metrics_shape:
        guidance += _metrics_shape_binding_block(metrics_shape)

    # 5.10. λ: data_recipes binding — when plan_reproduction found canonical
    # loader recipes for paper-mentioned datasets, bind the agent to use them
    # verbatim. Injected after the metrics-shape block so both contracts are
    # visible together. Empty list → no block (backward compat).
    if data_recipes:
        guidance += _data_recipes_binding_block(data_recipes)

    # 6. Per-paper YAML override — when docs/papers/<arxiv_id>.yaml exists.
    override = _load_paper_override(arxiv_id)
    if override:
        guidance += override

    # 6.5 Prior-attempt measured evidence (2026-06-10, flag-gated). Past
    # attempts' per-cell results ride into the prompt so the implementer keeps
    # configs that measurably worked instead of re-deriving them from scratch
    # (the All-CNN lr "repair" killed a cell whose working config sat in the
    # previous attempt's archive). Fail-soft + capped inside the module.
    if project_dir is not None:
        try:
            from backend.agents.rlm import prior_attempt_evidence as _pae
            if _pae.is_enabled():
                guidance += _pae.build_evidence_block(project_dir)
        except Exception:  # noqa: BLE001 — evidence is advisory, never fatal
            logger.debug("prior_attempt_evidence block skipped", exc_info=True)

    # 6.6 Best-attempt anti-regression block (2026-06-11, flag-gated): the best
    # prior attempt's score, the pointer to its seeded reference code, and the
    # leaf-level regression list (what the best earned that the latest lost).
    if project_dir is not None:
        try:
            from backend.agents.rlm.best_attempt import best_attempt_guidance_block
            guidance += best_attempt_guidance_block(project_dir)
        except Exception:  # noqa: BLE001 — advisory, never fatal
            logger.debug("best_attempt guidance block skipped", exc_info=True)

    # 7. Per-run extra guidance from REPROLAB_BASELINE_EXTRA_GUIDANCE env var.
    # Generic paper-agnostic hook so an operator can scope a specific run
    # without modifying source. Common uses:
    #   - "reproduce only the smallest 2 model variants the paper tests"
    #   - "use a 5% subset of the eval set for time-bounded iteration"
    #   - "skip the multi-seed sweep; one seed=42 is sufficient"
    # The guidance is appended verbatim, so the operator is responsible for
    # phrasing it so it doesn't contradict the NO STUB block above.
    import os as _os
    # Both spellings: the alias bridge mirrors REPROLAB_<->OPENRESEARCH_ at
    # IMPORT time only, but bes_rlm._angle_guidance mutates the env at RUNTIME
    # (per-candidate prompt angles) under the REPROLAB_ name — read both so
    # the candidate pool diversifies regardless of which prefix won the merge.
    extra = (
        _os.environ.get("OPENRESEARCH_BASELINE_EXTRA_GUIDANCE", "").strip()
        or _os.environ.get("REPROLAB_BASELINE_EXTRA_GUIDANCE", "").strip()
    )
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

    # 9. Parallelism policy — controls whether generated train.py uses
    #    DDP/FSDP/vLLM-TP (multi) or a single device (single/auto-single).
    # comp 3: on the harness-owned cell path the matrix is parallelized BY THE
    # HARNESS (one cell per GPU), so each cell is single-GPU and the torchrun /
    # DDP / FSDP guidance below is actively wrong — emit the cell framing instead.
    _par = (gpu_parallelism or "auto").lower()
    _n = gpu_visible_count
    if _cell_path_active:
        guidance += (
            "\nPARALLELISM POLICY — harness-owned cell matrix (one GPU per cell):\n"
            "  The harness runs your matrix as one subprocess PER CELL, each pinned to a single\n"
            "  GPU (CUDA_VISIBLE_DEVICES=<one id>), min(free_gpus, num_cells) in parallel. So\n"
            "  train_cell.py is SINGLE-GPU: train on cuda:0 only. Do NOT use torchrun / DDP /\n"
            "  FSDP / tensor-parallel / a device loop / 'cuda:1' — cross-card sharding fights the\n"
            "  per-cell pinning and re-creates the cuda:0 stacking that OOM'd the 2026-05-31 run.\n"
            "  Multi-GPU throughput comes from many cells running concurrently, not from sharding\n"
            "  one cell across cards.\n"
        )
    elif _par == "single" or (_n is not None and _n <= 1):
        guidance += (
            "\nPARALLELISM POLICY — single GPU:\n"
            "  Use a SINGLE GPU (or the CPU fallback when none is present). Do NOT "
            "  use DistributedDataParallel/FSDP/torchrun/tensor-parallel — this run "
            "  is scoped to one device. Keep the single-GPU/CPU path as the entrypoint.\n"
        )
    elif _par == "multi":
        _np = str(_n) if _n else "<NUM_GPUS>"
        guidance += (
            f"\nPARALLELISM POLICY — multi GPU"
            f"{f' ({_n} visible)' if _n else ''}:\n"
            "  Use ALL visible GPUs. CRITICAL: distributed training (DDP/FSDP) ONLY\n"
            "  shards when LAUNCHED with torchrun. A plain `python train.py` runs\n"
            "  single-process (WORLD_SIZE=1) — it DISABLES sharding, uses only GPU 0,\n"
            "  and OOMs models too large for one card. Therefore your commands.json\n"
            "  training entry MUST be (NOT `python train.py`):\n"
            f"      torchrun --standalone --nproc_per_node={_np} train.py\n"
            "  In train.py: read RANK / WORLD_SIZE / LOCAL_RANK from the environment,\n"
            "  call init_process_group('nccl'), and wrap the model with FSDP (for\n"
            "  models too large for one card) or DistributedDataParallel (data-\n"
            "  parallel). Enable gradient_checkpointing BEFORE the FSDP wrap. Detect\n"
            "  torch.cuda.device_count() and shard accordingly; keep a single-GPU\n"
            "  fallback for smoke. Use vLLM tensor-parallel for generation as needed.\n"
            "  RANK-GUARD ALL ONE-TIME SETUP (critical — torchrun runs the WHOLE\n"
            "  script in EVERY rank): pip installs, dataset/model downloads, env\n"
            "  bootstrapping (e.g. `pip install alfworld`, `alfworld-download`,\n"
            "  game-file loading) must run on LOCAL_RANK 0 ONLY, then `dist.barrier()`\n"
            "  so other ranks wait and reuse the shared cache. Running these in all\n"
            "  ranks concurrently deadlocks/thrashes on the pip lock + data dir and\n"
            "  hangs the run before training (the 2026-05-30 4-rank ALFWorld hang).\n"
        )
    else:  # auto
        guidance += (
            f"\nPARALLELISM POLICY — auto"
            f"{f' ({_n} GPU(s) visible)' if _n else ''}:\n"
            "  Detect torch.cuda.device_count() at runtime. If the paper's training or "
            "  evaluation genuinely benefits from parallelism (large model, long "
            "  training, many RL rollouts) AND more than one GPU is visible, scale "
            "  across them (torchrun+DDP for training, FSDP for oversized models, vLLM "
            "  tensor-parallel for generation). If the workload fits comfortably on one "
            "  GPU, a single GPU is correct — do NOT add parallelism the paper does not "
            "  need. Always keep a single-GPU/CPU fallback path.\n"
        )

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
    minimize_compute: bool = False,
    metrics_shape: list[dict] | None = None,
    data_recipes: list[dict] | None = None,
    gpu_parallelism: str | None = None,
    gpu_visible_count: int | None = None,
    gpu_cell_budget: dict | None = None,
    on_event=None,  # Callable[[], None] | None — SDK-stream liveness hook, forwarded to collect_agent_text
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
    _copy_harness_helpers_to_code_root(code_dir)

    # P0: prefer the explicit arxiv_id (threaded from RunContext.arxiv_id, which
    # was resolved from artifact_index.json / demo_status.json by run_pipeline_rlm)
    # over the fallback regex.  The regex is kept as a fallback for legacy
    # non-hashed project IDs that happen to embed an arXiv ID shaped string.
    # When neither the explicit kwarg nor the regex succeeds (hashed project_id
    # with no ctx.arxiv_id threaded through), also try reading from on-disk
    # artifacts so direct callers of run_with_sdk without a RunContext still get
    # the override (belt-and-suspenders; the primary path is ctx.arxiv_id → kwarg).
    _resolved_arxiv_id = arxiv_id or _extract_arxiv_id(project_id) or _derive_arxiv_id_from_disk(project_dir)

    # λ: extract data_recipes from the reproduction_contract when not explicitly
    # passed. plan_reproduction populates this via dataset_recipes.find_recipes_in_text.
    # Resolved early (before knowledge channel) so _effective_data_recipes is
    # available for both curated-helper generation and the guidance block.
    _effective_data_recipes: list[dict] | None = data_recipes
    if _effective_data_recipes is None and reproduction_contract is not None:
        _raw_dr = getattr(reproduction_contract, "data_recipes", None) or []
        _effective_data_recipes = [
            (r.model_dump() if hasattr(r, "model_dump") else dict(r))
            for r in _raw_dr
            if r is not None
        ] or None

    # PR-ξ γ: knowledge channel — write _reprolab_curated.py + manifest before
    # the sub-agent is invoked. Facts are derived from the resolved data_recipes
    # so the channel is independent of whether plan_reproduction succeeded.
    from backend.agents import baseline_knowledge as _bk
    _kc_facts = _bk.from_recipes(_effective_data_recipes or [])
    _kc_manifest = _bk.write_curated_artifacts(code_dir, _kc_facts)

    context = {
        "paper_claim_map": paper_claim_map.model_dump(),
        "environment_spec": environment_spec.model_dump(),
        "reproduction_contract": reproduction_contract.model_dump() if reproduction_contract else {},
        "artifact_index": artifact_index or {},
    }

    # θ: extract metrics_shape from the reproduction_contract when not explicitly
    # passed. This path handles calls where the contract is available but the
    # caller hasn't extracted the shape separately (e.g. implement_baseline in
    # primitives.py passes it through the kwarg when present).
    _effective_metrics_shape: list[dict] | None = metrics_shape
    if _effective_metrics_shape is None and reproduction_contract is not None:
        _raw = getattr(reproduction_contract, "metrics_shape", None) or []
        # Coerce MetricPath instances to plain dicts for the guidance block.
        _effective_metrics_shape = [
            (mp.model_dump() if hasattr(mp, "model_dump") else dict(mp))
            for mp in _raw
            if mp is not None
        ] or None

    sandbox_guidance = _compute_constraint_guidance(
        sandbox_mode, gpu_mode, project_dir=project_dir,
        arxiv_id=_resolved_arxiv_id, remaining_s=remaining_s,
        minimize_compute=minimize_compute,
        metrics_shape=_effective_metrics_shape or [],
        data_recipes=_effective_data_recipes or [],
        gpu_parallelism=gpu_parallelism,
        gpu_visible_count=gpu_visible_count,
        gpu_cell_budget=gpu_cell_budget,
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
        ledger_dir=project_dir,
        model=model,
        provider=provider,
        runtime=runtime,
        on_event=on_event,
    )

    # PR-ξ γ: post-emit knowledge-channel verification. After the sub-agent has
    # written train.py, check that curated constraints are satisfied. Strict
    # violations block execution and return a repairable result so patch-mode
    # can fix the exact import/use gap on the next iteration.
    _train_py = code_dir / "train.py"
    if _train_py.exists() and _kc_manifest.get("facts"):
        _kc_violations = _bk.verify_emitted_code(_train_py, _kc_manifest, code_dir)
        _kc_strict = [v for v in _kc_violations if v.severity == _bk.Severity.STRICT]
        if _kc_strict:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "baseline_implementation[%s]: %d strict knowledge-channel violation(s) — "
                "returning repairable result; patch-mode will fix on next iteration",
                project_id, len(_kc_strict),
            )
            return BaselineResult(
                mode="implement_from_paper",
                code_path=str(code_dir),
                commands_to_run=["python train.py"],
                diff_summary=(
                    f"knowledge_channel: {len(_kc_strict)} strict violation(s); "
                    f"repair required before execution"
                ),
                assumptions_applied=[
                    f"kc_violation:{v.kind}:{v.fact_id}" for v in _kc_strict
                ],
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


# ---------------------------------------------------------------------------
# PR-ι.2 — Patch-mode implement_baseline
# ---------------------------------------------------------------------------


def _extract_violations_from_repair_context(repair_context: dict) -> list[str]:
    """Extract structured violation strings from a repair_context dict.

    Looks for ``contract_violations``, ``preflight_violations``, and generic
    ``error`` / ``logs`` keys that name specific failures.  Returns a list of
    human-readable violation strings suitable for a diff-request prompt.
    """
    violations: list[str] = []

    # contract_violations: list[str] or list[dict]
    for key in ("contract_violations", "preflight_violations"):
        raw = repair_context.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    violations.append(item)
                elif isinstance(item, dict):
                    msg = item.get("message") or item.get("hint") or item.get("detail") or str(item)
                    loc = item.get("location") or item.get("file") or ""
                    if loc:
                        violations.append(f"{loc}: {msg}")
                    else:
                        violations.append(msg)

    # Fall back to the error / stderr strings if no structured violations.
    if not violations:
        err = repair_context.get("error") or ""
        if err:
            violations.append(str(err))
        logs = repair_context.get("logs") or repair_context.get("stderr") or ""
        if logs and logs != err:
            # Trim long logs — the agent only needs the tail.
            violations.append(str(logs)[-4000:])

    return violations


def _apply_unified_diff(original: str, diff_text: str) -> str:
    """Apply a unified diff to ``original`` and return the patched content.

    Uses Python's ``patch`` library if available (provides full context-line
    verification), otherwise falls back to a simple line-level apply that
    handles ``+``/``-`` markers.  Raises ``ValueError`` on apply failure so
    the caller can fall back to full rewrite.

    Only the hunk body is inspected (lines starting with ``+``, ``-``, or
    `` ``).  The ``@@`` range headers are used to locate the insertion
    context.
    """
    import re as _re

    lines_orig = original.splitlines(keepends=True)

    # Extract hunks: each hunk starts with @@ -start,count +start,count @@
    hunk_re = _re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@", _re.MULTILINE)
    diff_lines = diff_text.splitlines(keepends=True)

    # Find hunk boundaries.
    hunk_starts: list[int] = []
    for i, line in enumerate(diff_lines):
        if hunk_re.match(line):
            hunk_starts.append(i)

    if not hunk_starts:
        raise ValueError("no hunks found in diff — cannot apply")

    result = list(lines_orig)

    # Apply hunks in REVERSE order so earlier hunks don't shift line offsets.
    for hunk_start in reversed(hunk_starts):
        header = diff_lines[hunk_start]
        m = hunk_re.match(header)
        if not m:
            continue
        orig_start = int(m.group(1)) - 1  # convert to 0-based
        orig_count = int(m.group(2)) if m.group(2) is not None else 1

        # Collect hunk body lines.
        body: list[str] = []
        j = hunk_start + 1
        while j < len(diff_lines):
            if hunk_re.match(diff_lines[j]):
                break
            body.append(diff_lines[j])
            j += 1

        # Build the replacement lines from the hunk body.
        new_lines: list[str] = []
        removed = 0
        for bline in body:
            if not bline:
                continue
            marker = bline[0]
            content = bline[1:]
            if marker == " ":
                new_lines.append(content)
            elif marker == "+":
                new_lines.append(content)
            elif marker == "-":
                removed += 1
            elif marker == "\\":
                # "\ No newline at end of file" — skip
                continue
            else:
                # Unexpected marker; treat as context.
                new_lines.append(bline)

        end = orig_start + orig_count
        if orig_start > len(result) or end > len(result):
            raise ValueError(
                f"hunk range {orig_start+1}-{end} exceeds file length {len(result)}"
            )

        result[orig_start:end] = new_lines

    return "".join(result)


async def patch_mode_run_with_sdk(
    project_id: str,
    runs_root: Path,
    prior_train_py: str,
    violations: list[str],
    repair_context: dict[str, Any],
    *,
    model: str | None = None,
    provider: ProviderName | str | None = None,
    runtime: AgentRuntime | None = None,
    on_event=None,  # Callable[[], None] | None — SDK-stream liveness hook
) -> tuple[bool, str]:
    """Attempt a MINIMAL DIFF repair of ``prior_train_py`` for the given violations.

    Calls Sonnet with a patch-mode prompt: provides the full existing train.py
    and the violation list; asks for a unified diff (not a full rewrite).
    Applies the diff to the existing file.

    Returns ``(success, patched_content_or_error_message)``.  On failure
    (no valid diff in response, diff apply failure) returns ``(False, reason)``
    so the caller can fall back to a full rewrite.
    """

    project_dir = Path(runs_root) / project_id
    code_dir = project_dir / "code"

    violation_block = "\n".join(f"  - {v}" for v in violations)

    prompt = (
        f"The train.py for project {project_id} (located at {code_dir}/train.py) was "
        f"already written and failed post-run validation with these specific violations:\n\n"
        f"{violation_block}\n\n"
        f"Below is the COMPLETE EXISTING train.py. Emit a MINIMAL UNIFIED DIFF "
        f"(standard `diff -u` format, starting with `--- a/train.py` and `+++ b/train.py`) "
        f"that fixes ONLY those violations. "
        f"DO NOT rewrite the file. DO NOT change unrelated code. "
        f"DO NOT remove any functionality. "
        f"If a violation names a specific line or token, change ONLY that line. "
        f"Output the diff block as a code fence marked ```diff.\n\n"
        f"Repair context (the failed experiment result):\n"
        f"```json\n{json.dumps(repair_context, indent=2, default=str)[:2000]}\n```\n\n"
        f"--- EXISTING train.py ---\n"
        f"```python\n{prior_train_py}\n```"
    )

    # Collect the agent's response into a temp accumulator (not to code_dir).
    # We intercept the output text rather than letting the agent write files.
    _response_parts: list[str] = []

    async def _collect(agent_name: str, prompt_: str, **kwargs: Any) -> None:
        from backend.agents.runtime.invoke import collect_agent_text as _cat
        text = await _cat(agent_name, prompt_, **kwargs)
        if text:
            _response_parts.append(text)

    await _collect(
        "baseline-implementation",
        prompt,
        project_dir=code_dir,
        ledger_dir=project_dir,
        model=model,
        provider=provider,
        runtime=runtime,
        on_event=on_event,
    )

    response = "\n".join(_response_parts)

    # Extract the diff from the response (code fence or bare diff markers).
    import re as _re
    # Try ``` diff ... ``` fence first.
    fence_re = _re.compile(r"```diff\s*\n(.*?)```", _re.DOTALL | _re.IGNORECASE)
    m = fence_re.search(response)
    if not m:
        # Try ``` ... ``` without language tag (some models omit it).
        fence_re2 = _re.compile(r"```\s*\n(---.*?)```", _re.DOTALL)
        m = fence_re2.search(response)
    if not m:
        # Try bare diff (no fences).
        bare_re = _re.compile(r"(---\s+a/.*?\+\+\+\s+b/.*?(?=\Z|\n---|\Z))", _re.DOTALL)
        m = bare_re.search(response)

    if not m:
        return False, "no unified diff found in agent response"

    diff_text = m.group(1)
    if not diff_text.strip():
        return False, "extracted diff is empty"

    try:
        patched = _apply_unified_diff(prior_train_py, diff_text)
    except ValueError as exc:
        return False, f"diff apply failed: {exc}"

    return True, patched
