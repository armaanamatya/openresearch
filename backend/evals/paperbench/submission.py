"""Validate and describe PaperBench submission trees."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_COMMITTED_BYTES = 1_000_000_000
UPSTREAM_ZIP_FILE_LIMIT_BYTES = 10_000_000
IGNORED_DIR_NAMES = {"venv", ".venv", "__pycache__", ".git"}


@dataclass(frozen=True)
class SubmissionValidation:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    total_bytes: int
    file_count: int
    committed_bytes: int | None


@dataclass(frozen=True)
class PaperBenchSubmissionManifest:
    paper_id: str
    submission_dir: Path
    reproduce_sh: Path
    validation: SubmissionValidation
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "submission_dir": str(self.submission_dir),
            "reproduce_sh": str(self.reproduce_sh),
            "validation": {
                "ok": self.validation.ok,
                "errors": list(self.validation.errors),
                "warnings": list(self.validation.warnings),
                "total_bytes": self.validation.total_bytes,
                "file_count": self.validation.file_count,
                "committed_bytes": self.validation.committed_bytes,
            },
            "metadata": self.metadata,
        }


def validate_submission_tree(
    submission_dir: str | Path,
    *,
    max_committed_bytes: int = MAX_COMMITTED_BYTES,
    upstream_zip_file_limit_bytes: int = UPSTREAM_ZIP_FILE_LIMIT_BYTES,
) -> SubmissionValidation:
    """Validate the contract PaperBench expects for direct submissions."""

    root = Path(submission_dir).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not root.exists():
        return SubmissionValidation(
            ok=False,
            errors=(f"Submission directory does not exist: {root}",),
            warnings=(),
            total_bytes=0,
            file_count=0,
            committed_bytes=None,
        )
    if not root.is_dir():
        errors.append(f"Submission path is not a directory: {root}")

    reproduce = root / "reproduce.sh"
    if not reproduce.is_file():
        errors.append("Missing required reproduce.sh at submission root")
    elif not os.access(reproduce, os.X_OK):
        warnings.append("reproduce.sh exists but is not executable")

    readme = root / "README.md"
    if not readme.is_file():
        warnings.append("README.md is recommended so the judge can orient to the repo")

    total_bytes = 0
    file_count = 0
    for path in _iter_submission_files(root):
        file_count += 1
        size = path.stat().st_size
        total_bytes += size
        if size > upstream_zip_file_limit_bytes and path.name not in {"agent.log", "inspect.log"}:
            warnings.append(
                f"{path.relative_to(root)} is {size} bytes; upstream excludes files over "
                f"{upstream_zip_file_limit_bytes} bytes from the submission zip"
            )

    committed_bytes = _committed_size_bytes(root)
    if committed_bytes is not None and committed_bytes > max_committed_bytes:
        errors.append(
            f"Committed files total {committed_bytes} bytes, over PaperBench limit "
            f"{max_committed_bytes} bytes"
        )

    return SubmissionValidation(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        total_bytes=total_bytes,
        file_count=file_count,
        committed_bytes=committed_bytes,
    )


def create_submission_manifest(
    paper_id: str,
    submission_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    write: bool = False,
) -> PaperBenchSubmissionManifest:
    """Create an in-memory manifest, optionally writing ``paperbench_manifest.json``."""

    root = Path(submission_dir).expanduser().resolve()
    validation = validate_submission_tree(root)
    manifest = PaperBenchSubmissionManifest(
        paper_id=paper_id,
        submission_dir=root,
        reproduce_sh=root / "reproduce.sh",
        validation=validation,
        metadata=dict(metadata or {}),
    )
    if write:
        (root / "paperbench_manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2),
            encoding="utf-8",
        )
    return manifest


def _iter_submission_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIR_NAMES for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return files


def _committed_size_bytes(root: Path) -> int | None:
    if not (root / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    total = 0
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        path = root / raw.decode("utf-8", errors="replace")
        if path.is_file():
            total += path.stat().st_size
    return total
