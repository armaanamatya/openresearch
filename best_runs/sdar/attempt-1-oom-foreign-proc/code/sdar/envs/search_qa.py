"""Search-QA environment adapter (Section 3, Search-R1 setup).

Datasets (paper Table 1):
  Train (in-domain):  NQ (nq_open) + HotpotQA (hotpotqa/hotpot_qa, distractor split)
  Eval (out-of-domain): TriviaQA, PopQA, 2WikiMultiHopQA, MuSiQue, Bamboogle

Reward: max-alias token-F1 on the extracted "Answer:" span.

Prompt template (Figure 16): Question + retrieved passages + {skill_context} slot.

Retriever: E5 (intfloat/e5-small-v2, cached) for top-k passage retrieval.
  Falls back to no retriever (no-passage prompt) if E5 unavailable.
"""
from __future__ import annotations

import itertools
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils import extract_answer_span, max_alias_f1

# ──────────────────────────────────────────────────────────────────────────────
# Prompt template (Figure 16 of SDAR paper)
# ──────────────────────────────────────────────────────────────────────────────

SEARCH_QA_PROMPT = """\
You are a helpful question-answering assistant. Answer the following question based on the retrieved passages.

{skill_context}

Retrieved Passages:
{retrieved_passages}

Question: {question}

Think step by step, then provide your final answer on a new line starting with "Answer:".
"""

SEARCH_QA_PROMPT_NO_PASSAGES = """\
You are a helpful question-answering assistant. Answer the following question.

{skill_context}

Question: {question}

Think step by step, then provide your final answer on a new line starting with "Answer:".
"""


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────

HF_CACHE = os.environ.get("HF_HOME", "/home/sww35/openresearch/runs/.cache/hf")
HF_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

# Dataset canonical IDs (owner/name format for modern HF Hub)
DATASET_IDS = {
    "nq": ("nq_open", None),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor"),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc"),
    "popqa": ("akariasai/PopQA", None),
    "2wikimultihop": ("voidful/2WikiMultiHopQA", None),
    "musique": ("dgslibisey/MuSiQue", None),
    "bamboogle": ("chiayewken/bamboogle", None),
}

TRAIN_DATASETS = ["nq", "hotpotqa"]
EVAL_OOD_DATASETS = ["triviaqa", "popqa", "2wikimultihop", "musique", "bamboogle"]


def _load_dataset_safe(name: str, n: int = 256) -> Tuple[List[Dict], Optional[str]]:
    """Load a dataset slice; return (rows, error_msg) — never raises."""
    # Map dataset name to preferred split
    SPLIT_MAP = {
        "nq": "train",
        "hotpotqa": "train",
        "triviaqa": "validation",
        "popqa": "test",         # PopQA has no validation split
        "2wikimultihop": "validation",
        "musique": "validation",
        "bamboogle": "test",
    }

    try:
        from datasets import load_dataset
        ds_id, config = DATASET_IDS[name]
        split_name = SPLIT_MAP.get(name, "train" if name in TRAIN_DATASETS else "validation")
        n_slice = n
        split_with_slice = f"{split_name}[:{n_slice}]"

        # Try sliced first, then streaming fallback
        for attempt_split in [split_with_slice, split_name]:
            try:
                if config:
                    ds = load_dataset(ds_id, config, split=attempt_split,
                                      cache_dir=HF_CACHE, streaming=False)
                else:
                    ds = load_dataset(ds_id, split=attempt_split,
                                      cache_dir=HF_CACHE, streaming=False)
                rows = list(ds)[:n_slice]
                return _normalize_rows(name, rows), None
            except Exception as e1:
                last_err = e1

        # Last resort: streaming
        try:
            if config:
                ds = load_dataset(ds_id, config, split=split_name,
                                  cache_dir=HF_CACHE, streaming=True)
            else:
                ds = load_dataset(ds_id, split=split_name,
                                  cache_dir=HF_CACHE, streaming=True)
            rows = list(itertools.islice(ds, n_slice))
            return _normalize_rows(name, rows), None
        except Exception as e2:
            return [], f"{type(e2).__name__}: {str(e2)[:300]}"

    except Exception as e:
        return [], f"{type(e).__name__}: {str(e)[:300]}"


def _normalize_rows(name: str, rows: List) -> List[Dict]:
    """Normalize dataset rows to uniform {question, answers} format."""
    normalized = []
    for row in rows:
        if isinstance(row, dict):
            r = dict(row)
        else:
            continue

        q = r.get("question", r.get("query", ""))
        if not q:
            continue

        # Extract answer as list of strings
        if name == "nq":
            ans = r.get("answer", [])
            if isinstance(ans, str):
                ans = [ans]
        elif name == "hotpotqa":
            ans = [r.get("answer", "")]
        elif name == "triviaqa":
            ans_data = r.get("answer", {})
            if isinstance(ans_data, dict):
                ans = ans_data.get("aliases", ans_data.get("value", []))
                if isinstance(ans, str):
                    ans = [ans]
            else:
                ans = [str(ans_data)]
        elif name == "popqa":
            ans = r.get("possible_answers", r.get("answer", []))
            if isinstance(ans, str):
                try:
                    ans = json.loads(ans)
                except Exception:
                    ans = [ans]
        elif name in ("2wikimultihop", "musique"):
            ans = [r.get("answer", "")]
        elif name == "bamboogle":
            ans = [r.get("answer", "")]
        else:
            ans = [r.get("answer", "")]

        # Ensure list of non-empty strings
        ans = [str(a) for a in (ans or []) if a]
        if not ans:
            ans = [""]

        normalized.append({"question": str(q), "answers": ans, "source": name})

    return normalized


# ──────────────────────────────────────────────────────────────────────────────
# E5 retriever (intfloat/e5-small-v2)
# ──────────────────────────────────────────────────────────────────────────────

class E5Retriever:
    """Dense retriever using E5-small-v2 (Wang et al. 2022)."""

    MODEL_ID = "intfloat/e5-small-v2"

    def __init__(self, hf_cache: Optional[str] = None):
        self._model = None
        self._cache = hf_cache or HF_CACHE
        self._passages: List[str] = []
        self._embeddings = None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self._model = SentenceTransformer(self.MODEL_ID, cache_folder=self._cache)
            return True
        except Exception as e:
            print(f"[E5Retriever] Failed to load E5 ({e}); using no-retriever mode")
            return False

    def index(self, passages: List[str]) -> None:
        """Build passage index."""
        if not self._load():
            self._passages = passages
            return
        import numpy as np
        self._passages = passages
        texts = [f"passage: {p}" for p in passages]
        self._embeddings = self._model.encode(texts, batch_size=64, show_progress_bar=False,
                                               convert_to_numpy=True)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve top-k passages for a query."""
        if not self._passages:
            return []
        if self._embeddings is None or self._model is None:
            return self._passages[:k]
        import numpy as np
        q_emb = self._model.encode([f"query: {query}"], convert_to_numpy=True)
        scores = (q_emb @ self._embeddings.T).squeeze()
        top_k = np.argsort(-scores)[:k]
        return [self._passages[i] for i in top_k]


# ──────────────────────────────────────────────────────────────────────────────
# Main environment class
# ──────────────────────────────────────────────────────────────────────────────

class SearchQAEnv:
    """Search-QA environment for SDAR training/evaluation.

    Wraps dataset loading, E5 retrieval, prompt construction,
    and reward computation (max-alias token-F1).
    """

    def __init__(
        self,
        split: str = "train",
        max_passages: int = 3,
        hf_cache: Optional[str] = None,
        seed: int = 0,
    ):
        self.split = split
        self.max_passages = max_passages
        self.seed = seed

        self._train_data: List[Dict] = []
        self._eval_data: Dict[str, List[Dict]] = {}
        self._load_failures: List[Dict] = []

        self.retriever = E5Retriever(hf_cache=hf_cache or HF_CACHE)
        self._loaded = False

    def load(self) -> List[Dict]:
        """Load datasets. Returns list of failures."""
        if self._loaded:
            return self._load_failures

        # Load training data (NQ + HotpotQA)
        for name in TRAIN_DATASETS:
            rows, err = _load_dataset_safe(name, n=256)
            if err:
                self._load_failures.append({
                    "dataset": name, "loader": "hf", "error": err
                })
                print(f"[SearchQA] Train dataset {name} failed: {err[:200]}")
            else:
                self._train_data.extend(rows)
                print(f"[SearchQA] Loaded {len(rows)} train rows from {name}")

        # Load OOD eval data
        for name in EVAL_OOD_DATASETS:
            rows, err = _load_dataset_safe(name, n=128)
            if err:
                self._load_failures.append({
                    "dataset": name, "loader": "hf", "error": err
                })
                print(f"[SearchQA] Eval dataset {name} failed: {err[:200]}")
            else:
                self._eval_data[name] = rows
                print(f"[SearchQA] Loaded {len(rows)} eval rows from {name}")

        # Build a simple passage store from questions (fallback for retriever)
        all_questions = [r["question"] for r in self._train_data[:500]]
        if all_questions:
            self.retriever.index(all_questions)

        self._loaded = True
        random.seed(self.seed)
        return self._load_failures

    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample a batch from training data."""
        if not self._train_data:
            raise RuntimeError("No training data loaded. Call load() first.")
        return random.choices(self._train_data, k=batch_size)

    def build_student_prompt(self, sample: Dict, skill_context: str = "") -> str:
        """Build student prompt (no passages if retriever unavailable)."""
        question = sample["question"]
        passages = self.retriever.retrieve(question, k=self.max_passages)

        skill_block = f"Relevant Skills:\n{skill_context}\n" if skill_context else ""

        if passages:
            passage_text = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
            return SEARCH_QA_PROMPT.format(
                skill_context=skill_block,
                retrieved_passages=passage_text,
                question=question,
            )
        else:
            return SEARCH_QA_PROMPT_NO_PASSAGES.format(
                skill_context=skill_block,
                question=question,
            )

    def build_teacher_prompt(self, sample: Dict, skill_context: str) -> str:
        """Build teacher prompt: same as student but with skill context populated.

        The teacher is the SAME model weights but receives the skill context
        in its prompt (Section 2.1 of SDAR paper).
        """
        return self.build_student_prompt(sample, skill_context=skill_context)

    def compute_reward(self, generated_text: str, sample: Dict) -> float:
        """Compute max-alias token-F1 reward on the extracted Answer span."""
        pred = extract_answer_span(generated_text)
        aliases = sample["answers"]
        return max_alias_f1(pred, aliases)

    def compute_rewards_batch(
        self, generated_texts: List[str], samples: List[Dict]
    ) -> List[float]:
        """Compute rewards for a batch of (text, sample) pairs."""
        return [self.compute_reward(t, s) for t, s in zip(generated_texts, samples)]

    def evaluate(
        self,
        model_fn,  # callable(prompt: str) -> str
        dataset_name: Optional[str] = None,
        n: int = 64,
    ) -> Dict[str, float]:
        """Evaluate model on OOD datasets or training split.

        Returns dict of {dataset_name: mean_f1, ...}
        """
        results = {}

        if dataset_name:
            datasets_to_eval = {dataset_name: self._eval_data.get(dataset_name, [])}
        else:
            datasets_to_eval = {**self._eval_data}
            if self._train_data:
                datasets_to_eval["train_in_domain"] = self._train_data[:n]

        for name, data in datasets_to_eval.items():
            if not data:
                results[name] = None
                continue
            sample = data[:n]
            f1s = []
            for row in sample:
                prompt = self.build_student_prompt(row, skill_context="")
                try:
                    generated = model_fn(prompt)
                    f1 = self.compute_reward(generated, row)
                    f1s.append(f1)
                except Exception as e:
                    f1s.append(0.0)
            results[name] = sum(f1s) / len(f1s) if f1s else 0.0

        return results
