"""PaperBench integration helpers.

This package intentionally does not reimplement PaperBench's judge prompts or
grading logic.  It provides local bundle loading, rubric accounting, submission
validation, and command construction for the upstream PaperBench judge.
"""

from backend.evals.paperbench.bundle import (
    PaperBenchBundle,
    PaperBenchBundleError,
    load_paperbench_bundle,
)
from backend.evals.paperbench.score import (
    PaperBenchJudgeCommand,
    RubricSummary,
    SIMPLE_JUDGE_COMPLETER_CONFIG,
    SIMPLE_JUDGE_MODEL,
    SIMPLE_JUDGE_REASONING_EFFORT,
    code_development_ceiling,
    mean_standard_error,
    summarize_rubric,
)
from backend.evals.paperbench.submission import (
    PaperBenchSubmissionManifest,
    SubmissionValidation,
    create_submission_manifest,
    validate_submission_tree,
)

__all__ = [
    "PaperBenchBundle",
    "PaperBenchBundleError",
    "PaperBenchJudgeCommand",
    "PaperBenchSubmissionManifest",
    "RubricSummary",
    "SIMPLE_JUDGE_COMPLETER_CONFIG",
    "SIMPLE_JUDGE_MODEL",
    "SIMPLE_JUDGE_REASONING_EFFORT",
    "SubmissionValidation",
    "code_development_ceiling",
    "create_submission_manifest",
    "load_paperbench_bundle",
    "mean_standard_error",
    "summarize_rubric",
    "validate_submission_tree",
]
