"""Cross-project research memory.

Stores reusable verified lessons: environment recipes, failure modes, baseline
outcomes, and improvement results that can inform later projects.
"""

from backend.services.context.memory.model import MemoryKind, MemoryRecord, MemorySearchResult
from backend.services.context.memory.service import CrossProjectMemoryService

__all__ = [
    "CrossProjectMemoryService",
    "MemoryKind",
    "MemoryRecord",
    "MemorySearchResult",
]
