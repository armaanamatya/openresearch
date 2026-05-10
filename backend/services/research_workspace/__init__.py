"""Phase 2 research workspace facade."""

from backend.services.research_workspace.model import (
    KnowledgeGraphStats,
    ResearchWorkspaceSummary,
)
from backend.services.research_workspace.service import ResearchWorkspaceService

__all__ = [
    "KnowledgeGraphStats",
    "ResearchWorkspaceService",
    "ResearchWorkspaceSummary",
]
