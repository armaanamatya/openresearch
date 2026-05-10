"""Git worktree isolation for improvement branches."""

from backend.services.worktrees.manager import (
    GitWorktreeError,
    GitWorktreeManager,
    WorktreeInfo,
    WorktreeSpec,
)

__all__ = [
    "GitWorktreeError",
    "GitWorktreeManager",
    "WorktreeInfo",
    "WorktreeSpec",
]
