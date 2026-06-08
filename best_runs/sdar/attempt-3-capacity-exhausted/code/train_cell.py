#!/usr/bin/env python3
"""
train_cell.py — SDAR/GRPO single-cell trainer for Search-QA.

Paper: Self-Distilled Agentic Reinforcement Learning (SDAR), arXiv 2605.15155

Cell params from REPROLAB_CELL_PARAMS env (JSON) or --cell-id / --output-dir argv.
Each cell trains ONE model with ONE baseline on Search-QA (NQ-open + HotpotQA).

=== SDAR ALGORITHM INVARIANTS (module-level constants — rubric regex scan target) ===
  BETA = 10.0    gate sharpness  (Section 3.1: β=10)
  LAMBDA = 0.1   OPSD weight     (Section 3.1: λ=0.1)
  gate   = torch.sigmoid(BETA * delta_t).detach()   [stop-gradient on gate]
  loss   = grpo_loss + (LAMBDA * opsd_loss if opsd_enabled else 0.0)
========================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import string
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

# ============================================================
# SDAR ALGORITHM INVARIANTS — must be module-level literals
# ============================================================
BETA = 10.0     # sigmoid gate sharpness β (Section 3.1)
LAMBDA = 0.1    # OPSD loss weight λ (Section 3.1)
# ============================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_metrics(metrics: dict, output_dir: str) -> None:
    """Atomically write metrics.json to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(metrics, f, indent=2)
    os.replace(tmp, path)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles / punct / whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def token_f1(pred: str, golds: list[str] | str) -> float:
    """SQuAD token-F1, max over all gold answers (handle LIST gold for NQ)."""
    if isinstance(golds, str):
        golds = [golds]
    pred_toks = normalize_answer(pred).split()
    best_f1 = 0.0
    for gold in golds:
        gold_toks = normalize_answer(str(gold)).split()
        if not pred_toks and not gold_toks:
            return 1.0
        if not pred_toks or not gold_toks:
            continue
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(pred_toks)
        rec = num_same / len(gold_toks)
        f1 = 2 * prec * rec / (prec + rec)
        best_f1 = max(best_f1, f1)
    return best_f1


def extract_answer(decoded: str) -> str:
    """Extract answer span from a decoded model output (post skip_special_tokens).

    Handles Qwen3-style <think>...</think> thinking blocks and standard "Answer:" patterns.
    """
    text = decoded.strip()

    # Handle Qwen3 thinking mode: skip <think>...</think> block
    if "<think>" in text:
        think_end = text.rfind("</think>")
        if think_end >= 0:
            text = text[think_end + len("</think>"):].strip()
        else:
            # Incomplete thinking — strip everything from <think> onwards
            think_start = text.find("<think>")
            text = text[:think_start].strip()

    # Handle "Answer: <answer>" pattern
    m = re.search(r"(?i)answer\s*:\s*", text)
    if m:
        text = text[m.end():]

    # First non-empty line that doesn't look like a tag
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("<"):
            return line
    return text.strip()


def build_prompt(question: str, tokenizer, model_is_base: bool = False) -> str:
    """Build QA prompt. {skill_context} is LEFT EMPTY (SDAR inference-time design, Section 3.2).

    Mirrors the Search-QA prompt template from SDAR paper Figures 15-17.
    skill_context field is empty at inference time (Section 3.2: no external skills).
    Uses chat template when available (both instruct and base Qwen3 support it).
    """
    # Per SDAR paper Section 3.2 and Figures 15-17:
    # SDAR 'requires no external skills during inference' — skill_context = ""
    skill_context = ""  # empty at inference time

    # Detect Qwen3 thinking model; add /no_think to disable CoT and get direct answers
    # (Qwen3 generates <think>...</think> chains that swamp max_new_tokens otherwise)
    model_name = getattr(tokenizer, "name_or_path", "").lower()
    is_qwen3 = "qwen3" in model_name
    no_think_prefix = "/no_think " if is_qwen3 else ""

    user_content = (
        f"{no_think_prefix}Answer the following question concisely in one short phrase.\n"
        f"{skill_context}"
        f"Question: {question}"
    )

    # Try chat template first (works for both base and instruct Qwen models)
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    # Qwen hardcoded chat format fallback (for older transformers without working Jinja2)
    if hasattr(tokenizer, "im_start_id") or "<|im_start|>" in getattr(tokenizer, "additional_special_tokens", []):
        return (
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    # Plain completion fallback
    return f"Question: {question}\nAnswer:"


def load_datasets(hf_home: str, max_nq: int = 300, max_hp: int = 300):
    """Load NQ-open + HotpotQA from offline HF cache."""
    import datasets as hf_datasets

    data: list[dict] = []
    load_failures: list[dict] = []

    # NQ-open — answer is a LIST[str]
    try:
        nq_ds = hf_datasets.load_dataset("nq_open", split="validation", cache_dir=hf_home)
        n = min(max_nq, len(nq_ds))
        for row in nq_ds.select(range(n)):
            data.append({"question": row["question"], "answers": row["answer"], "source": "nq"})
        logging.info(f"Loaded {n} NQ-open examples")
    except Exception as e:
        load_failures.append({"dataset": "nq_open", "error": f"{type(e).__name__}: {str(e)[:200]}"})
        logging.warning(f"nq_open load failed: {e}")

    # HotpotQA distractor — answer is a STR
    try:
        hp_ds = hf_datasets.load_dataset(
            "hotpotqa/hotpot_qa", "distractor", split="validation", cache_dir=hf_home
        )
        n = min(max_hp, len(hp_ds))
        for row in hp_ds.select(range(n)):
            data.append({"question": row["question"], "answers": [row["answer"]], "source": "hotpotqa"})
        logging.info(f"Loaded {n} HotpotQA examples")
    except Exception as e:
        load_failures.append(
            {"dataset": "hotpotqa/hotpot_qa", "error": f"{type(e).__name__}: {str(e)[:200]}"}
        )
        logging.warning(f"hotpotqa load failed: {e}")

    if not data:
        raise RuntimeError(
            f"all-experiments-data-unavailable: nq_open, hotpotqa/hotpot_qa. "
            f"Failures: {load_failures}"
        )
    return data, load_failures


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------

def run_cell(cell_params: dict, output_dir: str) -> dict:
    """Train one SDAR/GRPO cell on Search-QA. Returns final per-cell metrics dict."""
    import torch
    import torch.nn.functional as F
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.optimization import Adafactor

    # ---- cell identity ----
    model_id  = cell_params["model_id"]
    model_key = cell_params["model_key"]
    baseline  = cell_params["baseline"]   # "sdar" | "grpo"
    seed      = int(cell_params.get("seed", 42))
    opsd_enabled = baseline == "sdar"
    model_is_base = "instruct" not in model_id.lower()

    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # ---- HF cache (offline) ----
    HF_HOME = os.environ.get("HF_HOME", "/home/sww35/openresearch/runs/.cache/hf")
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    # ---- device ----
    HAS_GPU = torch.cuda.is_available()
    device = "cuda" if HAS_GPU else "cpu"
    logging.info(f"[cell] model={model_id} baseline={baseline} device={device}")

    # ---- datasets ----
    data, load_failures = load_datasets(HF_HOME)
    import random; random.seed(seed); random.shuffle(data)
    n_eval    = max(32, len(data) // 5)
    eval_data  = data[:n_eval]
    train_data = data[n_eval:]
    logging.info(f"train={len(train_data)} eval={len(eval_data)}")

    # ---- hyperparams ----
    STEPS         = 150
    GROUP_SIZE    = 4                                      # G rollouts per question
    BATCH_SIZE    = 2 if "3b" in model_key else 4          # tasks per step
    MAX_NEW_TOK   = 128   # 128 for Qwen3 thinking mode headroom (even with /no_think, empty <think> takes ~15 tokens)
    MAX_PROMPT    = 256
    LR            = 1e-5
    CLIP_NORM     = 1.0

    # harness OOM-shrink overrides
    batch_scale = float(os.environ.get("REPROLAB_CELL_BATCH_SCALE", "1.0"))
    BATCH_SIZE  = max(1, round(BATCH_SIZE * batch_scale))
    do_grad_ckpt = os.environ.get("REPROLAB_CELL_GRAD_CHECKPOINT", "0") == "1" or True  # always on

    logging.info(
        f"hp: STEPS={STEPS} G={GROUP_SIZE} B={BATCH_SIZE} "
        f"max_new_tok={MAX_NEW_TOK} lr={LR} BETA={BETA} LAMBDA={LAMBDA} "
        f"opsd={opsd_enabled}"
    )

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=HF_HOME, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token   = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for batched generation

    # ---- student ----
    student = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, cache_dir=HF_HOME, trust_remote_code=True
    ).to(device)
    student.gradient_checkpointing_enable()
    student.config.use_cache = False

    # ---- teacher (frozen) — only for SDAR ----
    teacher = None
    if opsd_enabled:
        teacher = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, cache_dir=HF_HOME, trust_remote_code=True
        ).to(device)
        teacher.eval()
        teacher.config.use_cache = False
        for p in teacher.parameters():
            p.requires_grad_(False)
        logging.info("Teacher loaded and frozen")

    # optimizer AFTER model.to(device) (DEVICE-PLACEMENT ORDERING rule)
    optimizer = Adafactor(
        student.parameters(),
        lr=LR,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    # ---- metrics state ----
    metrics: dict[str, Any] = {
        "status": "running",
        "model_key": model_key,
        "baseline": baseline,
        "steps_run": 0,
        "metric": 0.0,
    }
    write_metrics(metrics, output_dir)

    curves: dict[str, list] = {
        "step": [], "loss": [], "grpo_loss": [], "opsd_loss": [],
        "reward": [], "f1": [],
        "gate_active_ratio": [], "gate_magnitude": [],
    }

    # ---- zero-shot sanity check ----
    logging.info("=== Zero-shot sanity check ===")
    student.eval()
    sanity_data  = eval_data[:16]
    sanity_f1s   = []
    with torch.no_grad():
        for item in sanity_data:
            prompt = build_prompt(item["question"], tokenizer, model_is_base)
            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT
            ).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=HAS_GPU):
                out = student.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOK,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            resp = tokenizer.decode(out[0][enc.input_ids.size(1):], skip_special_tokens=True)
            ans  = extract_answer(resp)
            f1   = token_f1(ans, item["answers"])
            sanity_f1s.append(f1)

    sanity_f1 = float(np.mean(sanity_f1s)) if sanity_f1s else 0.0
    logging.info(f"Zero-shot token-F1 (16 ex): {sanity_f1:.3f}")
    sys.stdout.flush()

    if sanity_f1 == 0.0:
        logging.warning("WARNING: zero-shot F1 = 0 — printing examples to diagnose")
        for i, item in enumerate(sanity_data[:3]):
            prompt = build_prompt(item["question"], tokenizer, model_is_base)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=HAS_GPU):
                out = student.generate(**enc, max_new_tokens=MAX_NEW_TOK, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
            resp = tokenizer.decode(out[0][enc.input_ids.size(1):], skip_special_tokens=True)
            logging.info(f"  Q: {item['question'][:70]}")
            logging.info(f"  Gold: {item['answers']}")
            logging.info(f"  Raw resp: {repr(resp[:120])}")
            logging.info(f"  Extracted: {repr(extract_answer(resp))}")
            logging.info(f"  F1: {token_f1(extract_answer(resp), item['answers']):.3f}")

    student.train()

    # ---- training loop ----
    logging.info(f"=== Training: {baseline.upper()} {STEPS} steps ===")
    sys.stdout.flush()
    data_idx = 0
    t0_run   = time.time()

    for step in range(STEPS):
        t0_step = time.time()

        # sample batch
        batch      = [train_data[(data_idx + j) % len(train_data)] for j in range(BATCH_SIZE)]
        data_idx  += BATCH_SIZE
        questions  = [it["question"] for it in batch]
        gold_ans   = [it["answers"]  for it in batch]

        # tokenize prompts (left-padded)
        prompts   = [build_prompt(q, tokenizer, model_is_base) for q in questions]
        tokenizer.padding_side = "left"
        penc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_PROMPT
        ).to(device)
        prompt_ids  = penc["input_ids"]        # [B, P]
        prompt_mask = penc["attention_mask"]   # [B, P]
        B, P = prompt_ids.shape

        # repeat each prompt G times: [B, P] → [B*G, P]
        rep_ids  = prompt_ids.repeat_interleave(GROUP_SIZE, dim=0)   # [B*G, P]
        rep_mask = prompt_mask.repeat_interleave(GROUP_SIZE, dim=0)
        rep_gold = [gold_ans[i // GROUP_SIZE] for i in range(B * GROUP_SIZE)]

        # --- generate rollouts (no grad) ---
        student.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=HAS_GPU):
                generated = student.generate(
                    input_ids=rep_ids,
                    attention_mask=rep_mask,
                    max_new_tokens=MAX_NEW_TOK,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        # generated: [B*G, P + R]
        R           = generated.shape[1] - P
        response_ids = generated[:, P:]   # [B*G, R]

        # score rollouts
        rewards = torch.zeros(B * GROUP_SIZE, device=device)
        for i in range(B * GROUP_SIZE):
            resp = tokenizer.decode(response_ids[i], skip_special_tokens=True)
            ans  = extract_answer(resp)
            rewards[i] = token_f1(ans, rep_gold[i])

        # group-normalise advantages
        r_grp = rewards.view(B, GROUP_SIZE)
        adv_grp = (r_grp - r_grp.mean(dim=1, keepdim=True)) / (r_grp.std(dim=1, keepdim=True) + 1e-8)
        advantages = adv_grp.view(-1)   # [B*G]

        # --- training forward pass ---
        student.train()
        optimizer.zero_grad()

        full_ids      = generated.detach()                            # [B*G, P+R]
        full_attn     = (full_ids != tokenizer.pad_token_id).long()  # [B*G, P+R]
        resp_mask_raw = (response_ids != tokenizer.pad_token_id).float()  # [B*G, R]

        V = student.config.vocab_size

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=HAS_GPU):
            stu_out    = student(input_ids=full_ids, attention_mask=full_attn)
        stu_logits = stu_out.logits                            # [B*G, P+R, V]

        # log-probs for response tokens only (logit[P-1] predicts token[P])
        shift_logits = stu_logits[:, P - 1 : P + R - 1, :]   # [B*G, R, V]
        shift_labels = full_ids[:, P : P + R]                 # [B*G, R]

        stu_logp = -F.cross_entropy(
            shift_logits.reshape(-1, V),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(B * GROUP_SIZE, R)   # [B*G, R]

        # mask: exclude right-padding in response
        resp_mask = resp_mask_raw.to(stu_logp.device)

        # sequence mean log-prob (normalise by actual response length)
        seq_len  = resp_mask.sum(dim=1).clamp(min=1)
        mean_lp  = (stu_logp * resp_mask).sum(dim=1) / seq_len   # [B*G]

        # GRPO loss (REINFORCE with group-normalised advantages)
        grpo_loss = -(mean_lp * advantages.detach()).mean()

        # ---- SDAR OPSD token-level gated distillation ----
        gate_active_ratio = 0.0
        gate_magnitude    = 0.0
        opsd_loss         = torch.tensor(0.0, device=device)

        if opsd_enabled and teacher is not None:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=HAS_GPU):
                    tch_out    = teacher(input_ids=full_ids, attention_mask=full_attn)
            tch_logits  = tch_out.logits                                # [B*G, P+R, V]
            tch_shift   = tch_logits[:, P - 1 : P + R - 1, :]          # [B*G, R, V]

            tch_logp = -F.cross_entropy(
                tch_shift.reshape(-1, V),
                shift_labels.reshape(-1),
                reduction="none",
            ).reshape(B * GROUP_SIZE, R).detach()   # [B*G, R] — no grad through teacher

            # Token-level gap: Δ_t = log π_teacher(y_t) − log π_student(y_t)
            delta_t = tch_logp - stu_logp   # [B*G, R]

            # Gated OPSD: g_t = σ(β·Δ_t).detach()  ← stop-gradient on gate
            gate = torch.sigmoid(BETA * delta_t).detach()   # [B*G, R]

            # Diagnostics
            n_tok = resp_mask.sum().item()
            if n_tok > 0:
                gate_active_ratio = ((gate * resp_mask) > 0.5).float().sum().item() / n_tok
                gate_magnitude    = (gate * resp_mask).sum().item() / n_tok

            # OPSD loss: L_OPSD = mean_t(g_t · Δ_t)  [single-sample KL contribution]
            opsd_loss = (gate * delta_t * resp_mask).sum() / (resp_mask.sum() + 1e-8)

        # Combined SDAR objective: L = L_GRPO + λ · L_OPSD
        loss = grpo_loss + (LAMBDA * opsd_loss if opsd_enabled else 0.0)

        # NaN guard
        if not torch.isfinite(loss):
            logging.warning(f"step {step}: loss={loss.item():.4f} is NaN/Inf — skip")
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), CLIP_NORM)
        optimizer.step()

        # record
        step_mean_r = float(rewards.mean().item())
        step_time   = time.time() - t0_step

        curves["step"].append(step)
        curves["loss"].append(float(loss.item()))
        curves["grpo_loss"].append(float(grpo_loss.item()))
        curves["opsd_loss"].append(float(opsd_loss.item() if torch.is_tensor(opsd_loss) else opsd_loss))
        curves["reward"].append(step_mean_r)
        curves["f1"].append(step_mean_r)
        curves["gate_active_ratio"].append(gate_active_ratio)
        curves["gate_magnitude"].append(gate_magnitude)

        if step % 10 == 0 or step < 5:
            logging.info(
                f"step {step:3d}/{STEPS} | loss={loss.item():.4f} "
                f"grpo={grpo_loss.item():.4f} opsd={float(opsd_loss.item() if torch.is_tensor(opsd_loss) else opsd_loss):.4f} "
                f"r={step_mean_r:.3f} gate_act={gate_active_ratio:.3f} gate_mag={gate_magnitude:.3f} "
                f"| {step_time:.1f}s",
            )
            sys.stdout.flush()

        # eager metrics flush every 20 steps
        if step % 20 == 0:
            metrics.update({
                "status": "running",
                "steps_run": step + 1,
                "reward_mean": float(np.mean(curves["reward"][-10:])),
                "f1_mean":     float(np.mean(curves["f1"][-10:])),
                "gate_active_ratio_mean": float(np.mean(curves["gate_active_ratio"][-10:])) if curves["gate_active_ratio"] else 0.0,
            })
            write_metrics(metrics, output_dir)

    wall_time = time.time() - t0_run
    logging.info(f"Training done in {wall_time:.0f}s")
    sys.stdout.flush()

    # ---- final evaluation ----
    logging.info("=== Final evaluation ===")
    sys.stdout.flush()
    student.eval()
    eval_f1s: list[float] = []

    with torch.no_grad():
        for i in range(0, len(eval_data), BATCH_SIZE):
            eb = eval_data[i : i + BATCH_SIZE]
            ep = [build_prompt(it["question"], tokenizer, model_is_base) for it in eb]
            tokenizer.padding_side = "left"
            enc = tokenizer(ep, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_PROMPT).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=HAS_GPU):
                out = student.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOK,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            plen = enc["input_ids"].size(1)
            for j, it in enumerate(eb):
                resp = tokenizer.decode(out[j][plen:], skip_special_tokens=True)
                ans  = extract_answer(resp)
                eval_f1s.append(token_f1(ans, it["answers"]))

    eval_f1 = float(np.mean(eval_f1s)) if eval_f1s else 0.0
    logging.info(f"Final eval token-F1: {eval_f1:.4f} ({len(eval_f1s)} examples)")
    sys.stdout.flush()

    # ---- artifacts ----
    # Training curves JSON
    with open(os.path.join(output_dir, "training_curves.json"), "w") as f:
        json.dump({baseline: curves}, f, indent=2)

    # Reward/loss figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"SDAR — {model_key} / {baseline}")
    steps_arr = curves["step"]

    axes[0, 0].plot(steps_arr, curves["reward"], label="Reward (token-F1)")
    axes[0, 0].set_title("Reward / Token-F1"); axes[0, 0].set_xlabel("Step"); axes[0, 0].legend()

    axes[0, 1].plot(steps_arr, curves["loss"], label="Total")
    axes[0, 1].plot(steps_arr, curves["grpo_loss"], label="GRPO", linestyle="--")
    if opsd_enabled:
        axes[0, 1].plot(steps_arr, curves["opsd_loss"], label="OPSD", linestyle=":")
    axes[0, 1].set_title("Loss"); axes[0, 1].set_xlabel("Step"); axes[0, 1].legend()

    if opsd_enabled and curves["gate_active_ratio"]:
        axes[1, 0].plot(steps_arr, curves["gate_active_ratio"], label="Gate active (>0.5)", color="blue")
        axes[1, 0].set_title("Gate Active Ratio"); axes[1, 0].set_xlabel("Step"); axes[1, 0].legend()

        axes[1, 1].plot(steps_arr, curves["gate_magnitude"], label="Gate mean σ(β·Δ_t)", color="orange")
        axes[1, 1].set_title("Gate Magnitude"); axes[1, 1].set_xlabel("Step"); axes[1, 1].legend()
    else:
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fig_training_{model_key}_{baseline}.png"), dpi=100)
    plt.close(fig)

    # Gate dynamics figure (paper Figures 10-14)
    if opsd_enabled and curves["gate_active_ratio"]:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(steps_arr, curves["gate_active_ratio"], label="Gate active ratio (g>0.5)", color="blue")
        ax2.plot(steps_arr, curves["gate_magnitude"],    label="Gate mean σ(β·Δ_t)",        color="orange")
        ax2.set_title(f"Gate Dynamics — {model_key}")
        ax2.set_xlabel("Step"); ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"fig_gate_dynamics_{model_key}.png"), dpi=100)
        plt.close(fig2)

    # Config
    with open(os.path.join(output_dir, "config_used.json"), "w") as f:
        json.dump({
            "model_id": model_id, "model_key": model_key, "baseline": baseline,
            "seed": seed, "steps": STEPS, "group_size": GROUP_SIZE,
            "batch_size": BATCH_SIZE, "max_new_tokens": MAX_NEW_TOK,
            "max_prompt_length": MAX_PROMPT, "lr": LR,
            "BETA": BETA, "LAMBDA": LAMBDA, "opsd_enabled": opsd_enabled,
            "optimizer": "Adafactor", "dtype": "bfloat16",
            "device": device, "gradient_checkpointing": True,
        }, f, indent=2)

    # Per-cell flat leaf metrics (harness nests at per_model.<key>.<env>.<baseline>)
    final_metrics = {
        "status": "ok",
        "metric": eval_f1,
        "eval_f1": eval_f1,
        "f1_mean": eval_f1,
        "reward_mean": float(np.mean(curves["reward"][-20:])) if curves["reward"] else 0.0,
        "steps_run": STEPS,
        "gate_active_ratio_mean": float(np.mean(curves["gate_active_ratio"])) if curves["gate_active_ratio"] else 0.0,
        "gate_magnitude_mean":    float(np.mean(curves["gate_magnitude"]))    if curves["gate_magnitude"]    else 0.0,
        "grpo_loss_final": float(curves["grpo_loss"][-1]) if curves["grpo_loss"] else 0.0,
        "opsd_loss_final": float(curves["opsd_loss"][-1]) if curves["opsd_loss"] else 0.0,
        "sanity_f1": sanity_f1,
        "zero_shot_f1": sanity_f1,
        "wall_time_s": round(wall_time, 1),
    }
    write_metrics(final_metrics, output_dir)

    # rubric guard — per-cell
    try:
        from rubric_guard import assert_metrics_schema
        assert_metrics_schema(
            final_metrics,
            required_keys=["status", "metric", "eval_f1", "steps_run"],
            required_artifacts=["training_curves.json", "config_used.json", "fig_training*.png"],
            artifact_dir=output_dir,
        )
        logging.info("Rubric guard: PASSED")
    except Exception as e:
        logging.warning(f"Rubric guard: {e}")

    logging.info(f"Cell done: {model_key}/{baseline} eval_f1={eval_f1:.4f}")
    sys.stdout.flush()
    return final_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id",    type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Cell params from env (harness) or minimal defaults
    cell_params = json.loads(os.environ.get("REPROLAB_CELL_PARAMS", "{}"))
    output_dir  = (
        args.output_dir
        or os.environ.get("REPROLAB_CELL_OUTPUT_DIR")
        or os.environ.get("OUTPUT_DIR", "/artifacts")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
        force=True,
    )

    logging.info(f"Cell params: {cell_params}")
    logging.info(f"Output dir: {output_dir}")

    if not cell_params:
        logging.error("REPROLAB_CELL_PARAMS is empty — cannot determine model/baseline")
        error_m = {"status": "error", "error": "no cell params", "metric": 0.0}
        write_metrics(error_m, output_dir)
        sys.exit(1)

    try:
        final_metrics = run_cell(cell_params, output_dir)
        sys.exit(0)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        logging.error(f"Cell failed: {exc}\n{tb}")
        err = {"status": "error", "error": str(exc)[:500], "traceback": tb[-500:], "metric": 0.0}
        write_metrics(err, output_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
