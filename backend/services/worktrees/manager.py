"""Git worktree manager for isolated improvement paths."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class WorktreeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_id: str
    path_id: str
    branch_name: str
    worktree_path: Path
    base_ref: str = "HEAD"


class WorktreeInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    branch: str = ""
    head: str = ""
    detached: bool = False


class GitWorktreeError(RuntimeError):
    def __init__(self, message: str, *, command: tuple[str, ...], stderr: str = "") -> None:
        super().__init__(message)
        self.command = command
        self.stderr = stderr


class GitWorktreeManager:
    """Creates one git worktree per improvement path.

    The manager is deliberately thin. It does not clean or reset the repo; it
    only asks git to create/remove worktrees in explicit directories.
    """

    def __init__(self, *, worktrees_root: Path) -> None:
        self.worktrees_root = worktrees_root

    def spec_for(
        self,
        *,
        project_id: str,
        path_id: str,
        slug: str,
        base_ref: str = "HEAD",
    ) -> WorktreeSpec:
        safe_project = _safe_segment(project_id)
        safe_path = _safe_segment(path_id)
        safe_slug = _safe_segment(slug)[:40] or "path"
        branch = f"improvement/{safe_path}-{safe_slug}"
        path = self.worktrees_root / safe_project / safe_path
        return WorktreeSpec(
            project_id=project_id,
            path_id=path_id,
            branch_name=branch,
            worktree_path=path,
            base_ref=base_ref,
        )

    def create(self, *, repo_root: Path, spec: WorktreeSpec) -> WorktreeInfo:
        repo = repo_root.resolve()
        target = spec.worktree_path.resolve()
        if target.exists() and any(target.iterdir()):
            raise GitWorktreeError(
                f"Worktree target is not empty: {target}",
                command=(),
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        self._run_git(
            repo,
            "worktree",
            "add",
            "-B",
            spec.branch_name,
            str(target),
            spec.base_ref,
        )
        return self.get(repo_root=repo, path=target)

    def remove(self, *, repo_root: Path, path: Path, force: bool = False) -> None:
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(path.resolve()))
        self._run_git(repo_root.resolve(), *args)

    def list(self, *, repo_root: Path) -> tuple[WorktreeInfo, ...]:
        result = self._run_git(repo_root.resolve(), "worktree", "list", "--porcelain")
        return _parse_worktree_porcelain(result.stdout)

    def get(self, *, repo_root: Path, path: Path) -> WorktreeInfo:
        target = path.resolve()
        for info in self.list(repo_root=repo_root):
            if info.path.resolve() == target:
                return info
        raise GitWorktreeError(
            f"Worktree was not registered by git: {target}",
            command=("git", "worktree", "list", "--porcelain"),
        )

    def _run_git(self, repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        command = ("git", "-C", str(repo_root), *args)
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise GitWorktreeError(
                f"Git command failed with exit code {result.returncode}",
                command=command,
                stderr=result.stderr,
            )
        return result


def _safe_segment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return safe.lower() or "unnamed"


def _parse_worktree_porcelain(output: str) -> tuple[WorktreeInfo, ...]:
    infos: list[WorktreeInfo] = []
    current: dict[str, str | bool] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                infos.append(_info_from_block(current))
                current = {}
            continue
        if line.startswith("worktree "):
            if current:
                infos.append(_info_from_block(current))
                current = {}
            current["path"] = line.removeprefix("worktree ")
        elif line.startswith("HEAD "):
            current["head"] = line.removeprefix("HEAD ")
        elif line.startswith("branch "):
            current["branch"] = line.removeprefix("branch refs/heads/")
        elif line == "detached":
            current["detached"] = True
    if current:
        infos.append(_info_from_block(current))
    return tuple(infos)


def _info_from_block(block: dict[str, str | bool]) -> WorktreeInfo:
    return WorktreeInfo(
        path=Path(str(block.get("path", ""))),
        branch=str(block.get("branch", "")),
        head=str(block.get("head", "")),
        detached=bool(block.get("detached", False)),
    )


__all__ = [
    "GitWorktreeError",
    "GitWorktreeManager",
    "WorktreeInfo",
    "WorktreeSpec",
]
