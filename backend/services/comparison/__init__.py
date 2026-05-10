"""Multi-paper comparative study services."""

from backend.services.comparison.model import (
    ComparableGroup,
    ComparisonReport,
    IncomparableRun,
    PaperRunSummary,
)
from backend.services.comparison.service import MultiPaperComparisonService

__all__ = [
    "ComparableGroup",
    "ComparisonReport",
    "IncomparableRun",
    "MultiPaperComparisonService",
    "PaperRunSummary",
]
