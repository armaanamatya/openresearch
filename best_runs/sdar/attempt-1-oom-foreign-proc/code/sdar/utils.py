"""Shared utilities: metrics I/O, token-logprob computation, prompt building."""
from __future__ import annotations

import json
import os
import re
import string
import time
from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Metrics helpers (atomic write, eager-flush)
# ──────────────────────────────────────────────────────────────────────────────

def get_output_dir() -> str:
    return os.environ.get("OUTPUT_DIR", "/artifacts")


def write_metrics(d: dict, output_dir: Optional[str] = None) -> None:
    """Atomically write metrics.json to output_dir (or $OUTPUT_DIR)."""
    out = Path(output_dir or get_output_dir())
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, path)


def load_metrics(output_dir: Optional[str] = None) -> dict:
    """Load existing metrics.json if present."""
    path = Path(output_dir or get_output_dir()) / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def deep_set(d: dict, dotted_path: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted path, creating dicts as needed."""
    keys = dotted_path.split(".")
    node = d
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


# ──────────────────────────────────────────────────────────────────────────────
# Token log-probability computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_token_logp(
    model,
    prompt_ids_list: List[List[int]],
    gen_ids_list: List[List[int]],
    device: str,
    use_grad: bool = False,
    mini_batch: int = 4,
) -> List[torch.Tensor]:
    """Compute per-token log probs for each (prompt, gen) pair.

    Uses left-padding so that generated tokens always start at the same
    absolute position (max_prompt_len).  Returns a list of 1-D tensors,
    each of shape [gen_len], under the context of use_grad.

    Args:
        model: The language model.
        prompt_ids_list: List of prompt token-id lists (variable length).
        gen_ids_list: List of generated token-id lists (variable length).
        device: Target device string.
        use_grad: If True, compute with gradient tracking; else torch.no_grad().
        mini_batch: Process in chunks to avoid OOM.
    """
    all_results: List[torch.Tensor] = []
    n = len(prompt_ids_list)

    for start in range(0, n, mini_batch):
        end = min(start + mini_batch, n)
        p_chunk = prompt_ids_list[start:end]
        g_chunk = gen_ids_list[start:end]
        chunk_results = _compute_logp_chunk(model, p_chunk, g_chunk, device, use_grad)
        all_results.extend(chunk_results)

    return all_results


def _compute_logp_chunk(
    model,
    prompt_ids_list: List[List[int]],
    gen_ids_list: List[List[int]],
    device: str,
    use_grad: bool,
) -> List[torch.Tensor]:
    """Compute logprobs for one mini-batch chunk."""
    max_p = max(len(p) for p in prompt_ids_list)
    max_g = max(len(g) for g in gen_ids_list)
    bs = len(prompt_ids_list)
    total_len = max_p + max_g

    # Build padded tensor (left-pad prompts, then concatenate gen)
    input_ids = torch.zeros(bs, total_len, dtype=torch.long)
    attn_mask = torch.zeros(bs, total_len, dtype=torch.long)

    prompt_lens = []
    gen_lens = []
    for i, (p, g) in enumerate(zip(prompt_ids_list, gen_ids_list)):
        pl, gl = len(p), len(g)
        prompt_lens.append(pl)
        gen_lens.append(gl)
        # Left-pad prompt
        input_ids[i, max_p - pl:max_p] = torch.tensor(p, dtype=torch.long)
        # Gen tokens after prompt
        input_ids[i, max_p:max_p + gl] = torch.tensor(g, dtype=torch.long)
        attn_mask[i, max_p - pl:max_p + gl] = 1

    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    ctx = torch.enable_grad() if use_grad else torch.no_grad()
    with ctx:
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits  # [bs, total_len, vocab]

    # Free the outputs object early (keeps only logits alive)
    del outputs

    # log-softmax in float32 for stability; cast back for gradient flow
    # NOTE: logprobs is float32 = 2× size of bf16 logits. Free bf16 logits ASAP.
    logprobs = F.log_softmax(logits.float(), dim=-1)  # fp32 [bs, total_len, vocab]
    del logits  # Free bf16 logits now — logprobs holds the fp32 version

    if not use_grad:
        logprobs = logprobs.detach()

    # Gather per-token log-probs for each sequence.
    # torch.gather backward (gather-and-scatter) does NOT need to save logprobs,
    # so token_lp's grad_fn chain goes: model params → logprobs → token_lp.
    # logprobs IS saved by log_softmax backward, so it stays alive until backward().
    results = []
    for i in range(bs):
        gl = gen_lens[i]
        # logit at position max_p-1 predicts gen_ids[0], etc.
        gen_lp_i = logprobs[i, max_p - 1:max_p - 1 + gl, :]  # view [gl, vocab]
        gen_id_i = input_ids[i, max_p:max_p + gl]             # [gl] on device
        token_lp = gen_lp_i.gather(-1, gen_id_i.unsqueeze(-1)).squeeze(-1)  # [gl]
        results.append(token_lp)
        del gen_lp_i  # View freed; logprobs kept alive by log_softmax backward until .backward()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Search-QA reward: token-level F1 on extracted answer span
# ──────────────────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, strip articles, collapse whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score_tokens(pred: str, gold: str) -> float:
    """Compute token-F1 between two strings."""
    pred_tok = normalize_answer(pred).split()
    gold_tok = normalize_answer(gold).split()
    if not pred_tok or not gold_tok:
        return float(pred_tok == gold_tok)
    common = set(pred_tok) & set(gold_tok)
    num_same = sum(min(pred_tok.count(t), gold_tok.count(t)) for t in common)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tok)
    recall = num_same / len(gold_tok)
    return 2 * precision * recall / (precision + recall)


def max_alias_f1(pred: str, gold_aliases: List[str]) -> float:
    """Compute max token-F1 over all gold aliases."""
    return max(f1_score_tokens(pred, alias) for alias in gold_aliases)


def extract_answer_span(text: str) -> str:
    """Extract the answer from a generated text by looking for 'Answer:' marker."""
    # Strip the echoed prompt — look for the LAST occurrence of the marker
    lower = text.lower()
    idx = lower.rfind("answer:")
    if idx != -1:
        answer_part = text[idx + len("answer:"):].strip()
        # Take the first line
        answer_part = answer_part.split("\n")[0].strip()
        return answer_part
    # Fallback: return last non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Advantage computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_group_advantages(rewards: List[float], group_size: int) -> List[float]:
    """Group-relative advantages: A_i = (r_i - mean(group)) / (std(group) + eps).

    Args:
        rewards: flat list of length batch_size * group_size
        group_size: G (rollouts per prompt)

    Returns:
        advantages: flat list of same length
    """
    import numpy as np
    rewards_np = np.array(rewards, dtype=np.float32).reshape(-1, group_size)
    mean_r = rewards_np.mean(axis=1, keepdims=True)
    std_r = rewards_np.std(axis=1, keepdims=True)
    adv_g = (rewards_np - mean_r) / (std_r + 1e-8)
    return adv_g.reshape(-1).tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Timer
# ──────────────────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self._start = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start
