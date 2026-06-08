"""
search_qa_env.py — Search-QA environment for SDAR.

Implements the prompt templates from SDAR paper Figures 15-17.
{skill_context} is LEFT EMPTY at inference time (Section 3.2: SDAR requires no
external skills during inference).

Subclasses BaseEnv as required by the harness's pre-flight AST check.
"""

from __future__ import annotations

from typing import Any

from sdar_env_base import BaseEnv


# Search-QA prompt template (SDAR paper Figure 17 / Appendix)
# {skill_context} placeholder present, empty at inference time.
STUDENT_PROMPT_TEMPLATE = """\
Answer the following question concisely in one short phrase.
{skill_context}
Question: {question}
Answer:"""

TEACHER_PROMPT_TEMPLATE = """\
Answer the following question accurately in one short phrase.
{skill_context}
Question: {question}
Answer:"""


class SearchQAEnv(BaseEnv):
    """Single-turn Search-QA environment for SDAR.

    Prompts mirror SDAR paper Figures 15-17 with the {skill_context}
    placeholder left empty at inference time (no external skills needed).
    """

    def build_student_prompt(
        self,
        question: str,
        skill_context: str = "",
        **kwargs: Any,
    ) -> str:
        """Build the prompt shown to the student policy.

        skill_context is empty string at inference time (SDAR Section 3.2).
        """
        return STUDENT_PROMPT_TEMPLATE.format(
            question=question,
            skill_context=skill_context,
        )

    def build_teacher_prompt(
        self,
        question: str,
        skill_context: str = "",
        **kwargs: Any,
    ) -> str:
        """Build the prompt shown to the frozen teacher for self-distillation."""
        return TEACHER_PROMPT_TEMPLATE.format(
            question=question,
            skill_context=skill_context,
        )

    def score(self, prediction: str, gold_answers: list[str] | str) -> float:
        """SQuAD token-F1 reward, max over gold aliases."""
        import re, string
        from collections import Counter

        def normalize(s: str) -> str:
            s = s.lower()
            s = re.sub(r"\b(a|an|the)\b", " ", s)
            s = "".join(c for c in s if c not in string.punctuation)
            return " ".join(s.split())

        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]

        pred_toks = normalize(prediction).split()
        best = 0.0
        for gold in gold_answers:
            gold_toks = normalize(str(gold)).split()
            if not pred_toks and not gold_toks:
                return 1.0
            if not pred_toks or not gold_toks:
                continue
            common = Counter(pred_toks) & Counter(gold_toks)
            n = sum(common.values())
            if n == 0:
                continue
            p = n / len(pred_toks)
            r = n / len(gold_toks)
            best = max(best, 2 * p * r / (p + r))
        return best
