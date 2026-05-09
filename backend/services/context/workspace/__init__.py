"""Workspace + citation tracking + tool surface (#16)."""

from backend.services.context.workspace.aggregate import (
    InvalidWorkspaceTransition,
    WorkspaceAggregate,
    WorkspaceState,
)
from backend.services.context.workspace.events import (
    CitationAttached,
    ToolInvoked,
    VariableEnriched,
    VariableLoaded,
    WorkspaceClosed,
    WorkspaceCreated,
    WorkspaceReady,
)
from backend.services.context.workspace.model import Cited
from backend.services.context.workspace.projections import (
    WorkspaceProjection,
    WorkspaceView,
)
from backend.services.context.workspace.service import (
    BuildWorkspace,
    WorkspaceAppService,
    WorkspaceError,
)
from backend.services.context.workspace.tools.interface import WorkspaceTool
from backend.services.context.workspace.tools.lookup import LookupTool
from backend.services.context.workspace.tools.semantic_search import SemanticSearchTool

__all__ = [
    "BuildWorkspace",
    "CitationAttached",
    "Cited",
    "InvalidWorkspaceTransition",
    "LookupTool",
    "SemanticSearchTool",
    "ToolInvoked",
    "VariableEnriched",
    "VariableLoaded",
    "WorkspaceAggregate",
    "WorkspaceAppService",
    "WorkspaceClosed",
    "WorkspaceCreated",
    "WorkspaceError",
    "WorkspaceProjection",
    "WorkspaceReady",
    "WorkspaceState",
    "WorkspaceTool",
    "WorkspaceView",
]
