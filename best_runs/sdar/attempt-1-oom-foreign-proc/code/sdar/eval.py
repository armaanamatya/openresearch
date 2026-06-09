"""Evaluation utilities for SDAR.

eval_with_skills=False for SDAR/GRPO/OPSD/Skill-SD/GRPO+OPSD/RLSD (paper default).
eval_with_skills=True for starred variants (Skill-GRPO*, SDAR*).

During inference, SDAR uses student input only — NO skill context.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch


def _format_prompt(tokenizer, prompt: str) -> str:
    """Apply chat template if available (for Qwen3/Qwen2.5-Instruct instruction models)."""
    if not hasattr(tokenizer, 'apply_chat_template') or tokenizer.chat_template is None:
        return prompt
    try:
        chat_msgs = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                chat_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                chat_msgs, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        return prompt


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    device: str = "cuda",
    temperature: float = 0.0,
) -> str:
    """Generate a single answer (greedy by default for evaluation)."""
    # Apply chat template for instruction models
    formatted_prompt = _format_prompt(tokenizer, prompt)

    tokenizer.padding_side = "left"
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    gen_ids = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def eval_search_qa(
    model,
    tokenizer,
    env,
    n: int = 64,
    device: str = "cuda",
    eval_with_skills: bool = False,
    skill_bank=None,
) -> Dict:
    """Evaluate on Search-QA (in-domain + OOD datasets).

    eval_with_skills=False (paper default for SDAR): empty {skill_context} at inference.
    eval_with_skills=True (starred variants): use retrieved skills at inference.
    """
    def model_fn(prompt: str) -> str:
        return generate_answer(model, tokenizer, prompt, device=device)

    results = {}
    # Evaluate on available OOD datasets
    for name, data in env._eval_data.items():
        if not data:
            continue
        sample = data[:n]
        from .utils import extract_answer_span, max_alias_f1
        f1s = []
        for row in sample:
            skill_ctx = ""
            if eval_with_skills and skill_bank is not None:
                skills = skill_bank.retrieve(row["question"], strategy="km", k=3)
                skill_ctx = skill_bank.format_skills(skills)
            prompt = env.build_student_prompt(row, skill_context=skill_ctx)
            gen = model_fn(prompt)
            f1 = env.compute_reward(gen, row)
            f1s.append(f1)
        results[name] = sum(f1s) / len(f1s) if f1s else 0.0

    # In-domain eval
    if env._train_data:
        id_sample = env._train_data[:n]
        f1s = []
        for row in id_sample:
            prompt = env.build_student_prompt(row, skill_context="")
            gen = model_fn(prompt)
            f1 = env.compute_reward(gen, row)
            f1s.append(f1)
        results["in_domain_f1"] = sum(f1s) / len(f1s) if f1s else 0.0

    return results


def eval_alfworld(
    model,
    tokenizer,
    env,
    n: int = 16,
    device: str = "cuda",
    eval_with_skills: bool = False,
    skill_bank=None,
) -> Dict:
    """Evaluate on ALFWorld by running n episodes."""
    if not env.available:
        return {"success_rate": None, "error": env.load_error}

    successes = []
    category_successes = {}

    def action_fn(prompt: str) -> str:
        return generate_answer(model, tokenizer, prompt, max_new_tokens=64, device=device)

    for i in range(n):
        skill_ctx = ""
        if eval_with_skills and skill_bank is not None:
            skills = skill_bank.retrieve("household navigation", strategy="km", k=2)
            skill_ctx = skill_bank.format_skills(skills)

        success, nsteps = env.run_episode(action_fn, skill_context=skill_ctx,
                                          eval_with_skills=eval_with_skills)
        successes.append(success)

    success_rate = sum(successes) / len(successes) if successes else 0.0
    return {
        "success_rate": success_rate,
        "n_episodes": n,
        "eval_with_skills": eval_with_skills,
    }


def eval_webshop(
    model,
    tokenizer,
    env,
    n: int = 64,
    device: str = "cuda",
    eval_with_skills: bool = False,
    skill_bank=None,
) -> Dict:
    """Evaluate on WebShop."""
    if not env.available:
        return {"score": None, "error": env.load_error}

    def model_fn(prompt: str) -> str:
        return generate_answer(model, tokenizer, prompt, max_new_tokens=64, device=device)

    results = env.evaluate(model_fn, n=n)
    results["eval_with_skills"] = eval_with_skills
    return results
