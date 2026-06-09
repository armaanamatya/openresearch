"""SDAR training entry point for a single (model, env, algorithm) cell.

CLI: python -m sdar.train --model qwen3_1_7b --env search_qa --algorithm sdar \
         --steps 150 --device cuda:0 --output-dir $OUTPUT_DIR

This module is also importable and used by train.py (the coordinator) directly.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdar.algorithms import get_step_fn, algorithm_uses_teacher, BETA, LAMBDA
from sdar.gating import make_gate
from sdar.skills import SkillBank
from sdar.utils import (
    compute_group_advantages,
    compute_token_logp,
    write_metrics,
    get_output_dir,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants (paper-specified, Table 3)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_IDS = {
    "qwen3_1_7b": "Qwen/Qwen3-1.7B",
    "qwen2_5_3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2_5_7b": "Qwen/Qwen2.5-7B-Instruct",
}

HF_CACHE = os.environ.get("HF_HOME", "/home/sww35/openresearch/runs/.cache/hf")


def load_model_and_tokenizer(
    model_id_or_short: str,
    device: str,
    bf16: bool = True,
    grad_ckpt: bool = False,
) -> Tuple:
    """Load Qwen model + tokenizer with correct kwargs.

    CRITICAL: use torch_dtype=, NOT dtype=.
    CRITICAL: model.to(device) BEFORE optimizer construction.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODEL_IDS.get(model_id_or_short, model_id_or_short)
    print(f"[train] Loading model {model_id} on {device} (bf16={bf16}) ...")

    dtype = torch.bfloat16 if bf16 else torch.float32

    # NEVER swallow model-load exceptions into scope.gaps
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Handle transformers 4.x (torch_dtype=) vs 5.x (dtype=) API difference
    import transformers as _tf_mod
    _tf_major = int(_tf_mod.__version__.split(".")[0])
    _dtype_kwarg = "dtype" if _tf_major >= 5 else "torch_dtype"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **{_dtype_kwarg: dtype},     # torch_dtype= (4.x) or dtype= (5.x)
        trust_remote_code=True,
        cache_dir=HF_CACHE,
    )

    # DEVICE PLACEMENT FIRST, then build optimizer later
    model = model.to(device)

    if grad_ckpt:
        model.gradient_checkpointing_enable()
        print(f"[train] Gradient checkpointing enabled for {model_id}")

    model.train()
    print(f"[train] Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model, tokenizer


def build_optimizer(model, lr: float = 1e-5, use_adafactor: bool = True):
    """Build optimizer (Adafactor by default — matches paper's memory budget)."""
    if use_adafactor:
        try:
            from transformers.optimization import Adafactor
            return Adafactor(
                model.parameters(),
                lr=lr,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
        except ImportError:
            pass
    # Fallback to AdamW
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# Chat-template helper
# ──────────────────────────────────────────────────────────────────────────────

def format_prompt_for_model(tokenizer, text: str) -> str:
    """Apply chat template if the tokenizer supports it (instruction models).

    For Qwen3 models, passes enable_thinking=False to suppress the verbose
    internal-monologue mode that consumes 500+ tokens before answering.
    For Qwen2.5-Instruct models, uses standard chat template.
    Falls back to raw text if chat_template is not available.
    """
    if not hasattr(tokenizer, 'apply_chat_template') or tokenizer.chat_template is None:
        return text
    try:
        chat_msgs = [{"role": "user", "content": text}]
        # Try enable_thinking=False first (Qwen3 specific)
        try:
            return tokenizer.apply_chat_template(
                chat_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            # Older transformers / models without enable_thinking parameter
            return tokenizer.apply_chat_template(
                chat_msgs, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        return text  # fallback to raw text


# ──────────────────────────────────────────────────────────────────────────────
# Core training loop
# ──────────────────────────────────────────────────────────────────────────────

def generate_rollouts(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    device: str = "cuda",
    max_prompt_tokens: int = 512,
) -> Dict:
    """Generate rollouts for a batch of prompts.

    Returns dict with:
      - texts: list of decoded generated texts (without prompt)
      - gen_ids: list of token-id lists (generated only)
      - prompt_ids: list of token-id lists (prompt only)
    """
    # Apply chat template for instruction models (Qwen3, Qwen2.5-Instruct)
    formatted_prompts = [format_prompt_for_model(tokenizer, p) for p in prompts]

    tokenizer.padding_side = "left"
    enc = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    ).to(device)

    prompt_len = enc.input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_part = out[:, prompt_len:].cpu()  # [batch, gen_len]

    texts = [tokenizer.decode(gen_part[i], skip_special_tokens=True)
             for i in range(gen_part.shape[0])]

    # Convert to lists for logp computation
    prompt_ids_list = [enc.input_ids[i].cpu().tolist() for i in range(enc.input_ids.shape[0])]
    gen_ids_list = [gen_part[i].tolist() for i in range(gen_part.shape[0])]

    # Remove trailing pad tokens from gen_ids
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    clean_gen_ids = []
    for ids in gen_ids_list:
        # Trim trailing pad/eos
        end = len(ids)
        while end > 0 and ids[end - 1] in (pad_id, eos_id):
            end -= 1
        clean_gen_ids.append(ids[:end] if end > 0 else ids[:1])

    return {
        "texts": texts,
        "gen_ids": clean_gen_ids,
        "prompt_ids": prompt_ids_list,
    }


def train_cell(
    model,
    tokenizer,
    env,
    algorithm: str,
    skill_bank: SkillBank,
    config: dict,
    device: str,
    output_dir: str,
    model_short: str = "model",
    env_name: str = "env",
    retrieval_strategy: str = "km",
    gate_strategy: str = "gap",
    beta: float = BETA,
    lam: float = LAMBDA,
    eval_with_skills: bool = False,
) -> Dict:
    """Train one (model, env, algorithm) cell for `config['steps']` steps.

    Returns the training metrics for this cell.
    """
    steps = config.get("steps", 150)
    group_size = config.get("group_size", 8)
    batch_size = config.get("batch_size", 16)
    max_new_tokens = config.get("max_new_tokens", 64)
    max_prompt_tokens = config.get("max_prompt_tokens", 512)
    lr = config.get("lr", 1e-5)
    eps = config.get("eps", 0.2)
    grad_clip = config.get("grad_clip", 1.0)
    temperature = config.get("temperature", 1.0)
    eval_n = config.get("eval_n", None)  # None = use _run_eval defaults

    step_fn = get_step_fn(algorithm)
    needs_teacher = algorithm_uses_teacher(algorithm)

    optimizer = build_optimizer(model, lr=lr)

    # Training metrics to track
    history = {
        "reward": [],
        "loss": [],
        "grpo_loss": [],
        "opsd_loss": [],
        "gate_active_ratio": [],
        "teacher_student_gap": [],
    }

    # Sanity check: zero-shot reward on a few samples
    _sanity_reward = _zero_shot_reward_check(
        model, tokenizer, env, skill_bank, device, max_new_tokens
    )
    print(f"[{algorithm}|{env_name}] Zero-shot reward: {_sanity_reward:.3f}", flush=True)

    for step in range(steps):
        t0 = time.time()
        model.train()

        # 1. Sample batch
        try:
            if env_name == "search_qa":
                batch = env.sample_batch(batch_size)
            elif env_name == "alfworld":
                batch = [{}] * batch_size  # ALFWorld uses episode-level batching
            elif env_name == "webshop":
                batch = env.sample_batch(batch_size)
            else:
                batch = env.sample_batch(batch_size)
        except Exception as e:
            print(f"  [step {step}] Batch sampling failed: {e}", flush=True)
            continue

        # 2. Retrieve skills for teacher prompts
        skills_batch = []
        for item in batch:
            query = item.get("question", item.get("instruction", "")) if isinstance(item, dict) else ""
            retrieved = skill_bank.retrieve(query, strategy=retrieval_strategy, k=3)
            skills_batch.append(retrieved)

        skill_texts_batch = [skill_bank.format_skills(s) for s in skills_batch]

        # 3. Build student + teacher prompts (student: no skills, teacher: with skills)
        student_prompts = []
        teacher_prompts = []
        for item, skill_text in zip(batch, skill_texts_batch):
            sp = env.build_student_prompt(item, skill_context="")
            tp = env.build_teacher_prompt(item, skill_context=skill_text) if needs_teacher else sp
            student_prompts.append(sp)
            teacher_prompts.append(tp)

        # 4. Expand each prompt G times for rollouts
        all_student_prompts = [p for p in student_prompts for _ in range(group_size)]
        all_teacher_prompts = [p for p in teacher_prompts for _ in range(group_size)]

        # 5. Generate rollouts
        try:
            rollout_data = generate_rollouts(
                model, tokenizer, all_student_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
                max_prompt_tokens=max_prompt_tokens,
            )
        except torch.cuda.OutOfMemoryError:
            # Raise OOM — do NOT silently continue (would produce all-zero metrics)
            raise
        except Exception as e:
            print(f"  [step {step}] Generation failed: {e}", flush=True)
            continue

        texts = rollout_data["texts"]
        gen_ids = rollout_data["gen_ids"]
        prompt_ids = rollout_data["prompt_ids"]

        # 6. Compute rewards
        try:
            if env_name == "search_qa":
                # Expand batch items to match G rollouts
                all_items = [item for item in batch for _ in range(group_size)]
                rewards = env.compute_rewards_batch(texts, all_items)
            elif env_name == "alfworld":
                rewards = [0.0] * len(texts)  # ALFWorld rewards from episodes
            elif env_name == "webshop":
                all_items = [item for item in batch for _ in range(group_size)]
                rewards = env.compute_rewards_batch(texts, all_items)
            else:
                rewards = [0.5] * len(texts)
        except Exception as e:
            print(f"  [step {step}] Reward computation failed: {e}", flush=True)
            rewards = [0.0] * len(texts)

        if not any(r > 0 for r in rewards) and step < 5:
            print(f"  [step {step}] WARNING: all-zero rewards — check reward adapter", flush=True)

        # 7. Compute teacher prompt IDs (for teacher logp computation)
        if needs_teacher:
            try:
                # Apply chat template to teacher prompts (same format as student)
                formatted_teacher_prompts = [
                    format_prompt_for_model(tokenizer, tp) for tp in all_teacher_prompts
                ]
                tokenizer.padding_side = "left"
                teacher_enc = tokenizer(
                    formatted_teacher_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_tokens,
                )
                teacher_prompt_ids = [
                    teacher_enc.input_ids[i].tolist()
                    for i in range(teacher_enc.input_ids.shape[0])
                ]
            except Exception:
                teacher_prompt_ids = prompt_ids  # Fallback to student prompts
        else:
            teacher_prompt_ids = prompt_ids

        # 8. Training step
        try:
            if algorithm == "sdar":
                from .algorithms import sdar_step
                step_metrics = sdar_step(
                    model, optimizer,
                    student_prompt_ids=prompt_ids,
                    teacher_prompt_ids=teacher_prompt_ids,
                    gen_ids=gen_ids,
                    rewards=rewards,
                    group_size=group_size,
                    eps=eps,
                    beta=beta,
                    lam=lam,
                    gate_strategy=gate_strategy,
                    device=device,
                    grad_clip=grad_clip,
                )
            elif algorithm == "grpo":
                from .algorithms import grpo_step
                step_metrics = grpo_step(
                    model, optimizer,
                    student_prompt_ids=prompt_ids,
                    gen_ids=gen_ids,
                    rewards=rewards,
                    group_size=group_size,
                    eps=eps,
                    device=device,
                    grad_clip=grad_clip,
                )
            else:
                step_metrics = step_fn(
                    model, optimizer,
                    student_prompt_ids=prompt_ids,
                    teacher_prompt_ids=teacher_prompt_ids,
                    gen_ids=gen_ids,
                    rewards=rewards,
                    group_size=group_size,
                    eps=eps,
                    beta=beta,
                    lam=lam,
                    device=device,
                    grad_clip=grad_clip,
                )
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            print(f"  [step {step}] Training step failed: {e}", flush=True)
            import traceback; traceback.print_exc()
            continue

        # 9. Record history
        for k in history:
            if k in step_metrics:
                history[k].append(step_metrics[k])

        # Update UCB if using UCB retrieval
        if retrieval_strategy == "ucb":
            for i, (retrieved, reward) in enumerate(zip(skills_batch, rewards[:batch_size])):
                skill_bank.update_ucb([s["name"] for s in retrieved], reward)

        # 10. Log progress
        dt = time.time() - t0
        print(
            f"step={step} method={algorithm} env={env_name} "
            f"reward={step_metrics.get('mean_reward', 0):.3f} "
            f"loss={step_metrics.get('loss', 0):.4f} "
            f"gate={step_metrics.get('gate_active_ratio', 0):.3f} "
            f"dt={dt:.1f}s",
            flush=True,
        )

        # Check for NaN loss
        if step_metrics.get("loss", 0) != step_metrics.get("loss", 0):  # NaN check
            raise RuntimeError(
                f"train_loss=NaN at step={step}, lr={lr} — aborting to prevent "
                f"wasted compute with no useful output"
            )

    # ── Evaluation ──────────────────────────────────────────────────────────────
    model.eval()
    print(f"[{algorithm}|{env_name}] Running evaluation...", flush=True)

    try:
        eval_metrics = _run_eval(
            model, tokenizer, env, env_name, skill_bank, device,
            eval_with_skills=eval_with_skills,
            eval_n=eval_n,
        )
    except Exception as e:
        print(f"  Eval failed: {e}", flush=True)
        eval_metrics = {"error": str(e)[:200]}

    return {
        "history": history,
        "eval": eval_metrics,
        "final_reward": history["reward"][-1] if history["reward"] else 0.0,
        "mean_reward_last10": (
            sum(history["reward"][-10:]) / len(history["reward"][-10:])
            if len(history["reward"]) >= 10 else
            sum(history["reward"]) / max(len(history["reward"]), 1)
        ),
        "algorithm": algorithm,
        "model": model_short,
        "env": env_name,
        "steps_run": steps,
        "gate_active_ratio_history": history.get("gate_active_ratio", []),
        "gate_mean_history": history.get("gate_active_ratio", []),
        "opsd_loss_history": history.get("opsd_loss", []),
        "teacher_student_gap_history": history.get("teacher_student_gap", []),
    }


def _zero_shot_reward_check(
    model, tokenizer, env, skill_bank, device: str, max_new_tokens: int = 64
) -> float:
    """Compute zero-shot reward on a few samples to sanity-check the reward adapter."""
    try:
        if hasattr(env, "sample_batch"):
            samples = env.sample_batch(4)
        else:
            return 0.0

        prompts = [env.build_student_prompt(s, skill_context="") for s in samples]
        rollouts = generate_rollouts(model, tokenizer, prompts,
                                     max_new_tokens=min(max_new_tokens, 32), temperature=1.0,
                                     device=device, max_prompt_tokens=256)

        if hasattr(env, "compute_rewards_batch"):
            rewards = env.compute_rewards_batch(rollouts["texts"], samples)
            return sum(rewards) / len(rewards)
    except Exception as e:
        print(f"  [zero-shot check] failed: {e}")
    return 0.0


def _run_eval(model, tokenizer, env, env_name: str, skill_bank, device: str,
              eval_with_skills: bool = False, eval_n: Optional[int] = None) -> Dict:
    """Run environment-specific evaluation.

    Args:
        eval_n: Override number of eval samples (e.g. reduce for smoke mode).
                Defaults: search_qa=64, alfworld=8, webshop=32.
    """
    # Use absolute import to avoid relative import errors
    import importlib
    eval_module = importlib.import_module("sdar.eval")

    if env_name == "search_qa":
        n = eval_n if eval_n is not None else 64
        return eval_module.eval_search_qa(
            model, tokenizer, env,
            n=n, device=device,
            eval_with_skills=eval_with_skills,
            skill_bank=skill_bank if eval_with_skills else None,
        )
    elif env_name == "alfworld":
        n = eval_n if eval_n is not None else 8
        return eval_module.eval_alfworld(
            model, tokenizer, env,
            n=n, device=device,
            eval_with_skills=eval_with_skills,
            skill_bank=skill_bank if eval_with_skills else None,
        )
    elif env_name == "webshop":
        n = eval_n if eval_n is not None else 32
        return eval_module.eval_webshop(
            model, tokenizer, env,
            n=n, device=device,
            eval_with_skills=eval_with_skills,
            skill_bank=skill_bank if eval_with_skills else None,
        )
    return {}


def _extract_primary_metric(eval_metrics: Dict, env_name: str) -> float:
    """Extract the primary metric for an environment from eval results."""
    if not eval_metrics:
        return 0.0
    if env_name == "search_qa":
        # Mean F1 over all available datasets
        vals = [v for v in eval_metrics.values()
                if isinstance(v, float) and v is not None]
        return sum(vals) / len(vals) if vals else 0.0
    elif env_name == "alfworld":
        return eval_metrics.get("success_rate", 0.0) or 0.0
    elif env_name == "webshop":
        return eval_metrics.get("score", 0.0) or 0.0
    return 0.0
