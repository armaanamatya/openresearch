"""Persistent Python REPL host for the RLM root loop.

Hosts the `globals` dict the root LLM writes code into, runs `exec(code,
namespace)` per iteration, captures stdout, and serializes state to
`runs/<project_id>/repl_state.pickle` for resume-safety.

Phase 2 (#59) implementation. See `docs/rlm-pivot-mapping.md` §6.2 for the
serialization strategy (large strings stored as file refs, non-picklable
entries stripped).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ReplOutput:
    """Result of one `exec` invocation: stdout + metadata for the root history."""

    stdout: str
    length: int
    prefix: str
    has_traceback: bool
    var_assignments: list[str]


class ReplHost:
    """A persistent Python REPL. One instance per RLM run."""

    def __init__(self, project_dir: Path) -> None:
        # Phase 2: populate `self._globals` with paper_text, paper_metadata,
        # supplementary_text, repo_files, prior_work_refs, rubric_spec, and
        # the primitive registry. Bootstrap llm_query / rlm_query / print.
        raise NotImplementedError("Phase 2 (#59) — initialize REPL globals")

    def exec(self, code: str) -> ReplOutput:
        """Execute one iteration's code in the persistent namespace."""
        # Phase 2: capture stdout via `contextlib.redirect_stdout`, parse
        # `ast.Assign` nodes from `code` to surface var_assignments, return
        # length + 200-char prefix as Metadata(stdout). NEVER return raw
        # stdout to the root history — only ReplOutput, which the caller
        # converts to metadata.
        raise NotImplementedError("Phase 2 (#59) — exec into persistent namespace")

    def has_variable(self, name: str) -> bool:
        """True if `name` is set in the REPL namespace."""
        raise NotImplementedError("Phase 2 (#59) — REPL namespace introspection")

    def read_variable(self, name: str) -> Any:
        """Return the value of a REPL variable (used for FINAL_VAR resolution)."""
        raise NotImplementedError("Phase 2 (#59) — REPL namespace read")

    def serialize(self, path: Path) -> None:
        """Pickle the REPL globals (minus non-picklable handles) to `path`."""
        raise NotImplementedError("Phase 2 (#59) — checkpoint serialization")

    @classmethod
    def resume(cls, project_dir: Path, path: Path) -> "ReplHost":
        """Restore a ReplHost from a `repl_state.pickle` checkpoint."""
        raise NotImplementedError("Phase 2 (#59) — checkpoint resume")
