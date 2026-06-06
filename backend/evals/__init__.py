"""OpenResearch Evaluation System.

Evaluation surfaces:
  * PaperBench scoring — leaf-scorer, rubric accounting, submission validation
  * EvalStore — persistent storage for scores, A/B tests, Elo ratings
  * Elo / A/B testing utilities

Usage:
    from backend.evals import EvalStore
    from backend.evals.sources import EVAL_SOURCES, format_all_citations
"""

from backend.evals.schemas import (
    ABTestResult,
    EloMatchResult,
    EloRating,
    EvalMode,
    EvalRun,
    HypothesisScore,
    InnovationScore,
    IntegrityFlag,
    IntegrityReport,
    ReproductionScore,
    ResearchMapScore,
)
from backend.evals.store import EvalStore

__all__ = [
    "ABTestResult",
    "EloMatchResult",
    "EloRating",
    "EvalMode",
    "EvalRun",
    "EvalStore",
    "HypothesisScore",
    "InnovationScore",
    "IntegrityFlag",
    "IntegrityReport",
    "ReproductionScore",
    "ResearchMapScore",
]
