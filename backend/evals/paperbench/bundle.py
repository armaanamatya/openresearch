"""Load vendored PaperBench paper bundles.

Expected layout:

    third_party/paperbench/<paper_id>/
      config.yaml
      paper.md
      addendum.md
      rubric.json
      task_instructions.md        # optional; upstream also uses global instructions
      paper.pdf                   # optional
      blacklist.txt               # optional
      assets/                     # optional
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PaperBenchBundleError(ValueError):
    """Raised when a PaperBench bundle is missing required files or is invalid."""


@dataclass(frozen=True)
class PaperBenchBundle:
    """Resolved paths for a single PaperBench paper bundle."""

    paper_id: str
    root: Path
    paper_md_path: Path
    addendum_path: Path
    rubric_path: Path
    task_instructions_path: Path | None = None
    config_path: Path | None = None
    paper_pdf_path: Path | None = None
    blacklist_path: Path | None = None
    assets_dir: Path | None = None

    def read_paper_markdown(self) -> str:
        return self.paper_md_path.read_text(encoding="utf-8")

    def read_addendum(self) -> str:
        return self.addendum_path.read_text(encoding="utf-8")

    def read_task_instructions(self) -> str:
        if self.task_instructions_path is None:
            return ""
        return self.task_instructions_path.read_text(encoding="utf-8")

    def rubric(self) -> dict[str, Any]:
        data = json.loads(self.rubric_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise PaperBenchBundleError(
                f"rubric.json for {self.paper_id!r} must contain a JSON object"
            )
        return data

    def blacklist_entries(self) -> tuple[str, ...]:
        if self.blacklist_path is None:
            return ()
        return tuple(
            line.strip()
            for line in self.blacklist_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )

    def metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.paper_id}
        if self.config_path is None:
            return data
        for line in self.config_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip().strip("\"'")
            if key.strip():
                data[key.strip()] = value
        data.setdefault("id", self.paper_id)
        return data


def load_paperbench_bundle(root: str | Path, paper_id: str | None = None) -> PaperBenchBundle:
    """Load and validate a PaperBench bundle.

    ``root`` may be either ``third_party/paperbench`` plus ``paper_id`` or the
    concrete bundle directory itself.
    """

    base = Path(root).expanduser().resolve()
    bundle_dir = base / paper_id if paper_id else base
    if not bundle_dir.exists():
        raise PaperBenchBundleError(f"PaperBench bundle directory does not exist: {bundle_dir}")
    if not bundle_dir.is_dir():
        raise PaperBenchBundleError(f"PaperBench bundle path is not a directory: {bundle_dir}")

    resolved_paper_id = paper_id or bundle_dir.name
    config_path = _optional_file(bundle_dir / "config.yaml")
    if config_path is not None:
        resolved_paper_id = _paper_id_from_config(config_path) or resolved_paper_id

    paper_md_path = _required_file(bundle_dir / "paper.md")
    addendum_path = _required_file(bundle_dir / "addendum.md")
    rubric_path = _required_file(bundle_dir / "rubric.json")
    task_instructions_path = _first_existing_file(
        bundle_dir / "task_instructions.md",
        bundle_dir / "instructions.txt",
    )
    paper_pdf_path = _optional_file(bundle_dir / "paper.pdf")
    blacklist_path = _optional_file(bundle_dir / "blacklist.txt")
    assets_dir = bundle_dir / "assets"
    if not assets_dir.is_dir():
        assets_dir = None

    bundle = PaperBenchBundle(
        paper_id=resolved_paper_id,
        root=bundle_dir,
        paper_md_path=paper_md_path,
        addendum_path=addendum_path,
        rubric_path=rubric_path,
        task_instructions_path=task_instructions_path,
        config_path=config_path,
        paper_pdf_path=paper_pdf_path,
        blacklist_path=blacklist_path,
        assets_dir=assets_dir,
    )
    bundle.rubric()
    return bundle


def _required_file(path: Path) -> Path:
    if path.is_file():
        return path
    raise PaperBenchBundleError(f"Required PaperBench file is missing: {path}")


def _optional_file(path: Path) -> Path | None:
    return path if path.is_file() else None


def _first_existing_file(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _paper_id_from_config(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("id:"):
            return line.split(":", 1)[1].strip().strip("\"'") or None
    return None
