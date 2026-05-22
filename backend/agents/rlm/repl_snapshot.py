"""REPL snapshot writer — issue #62 DC#4 artifacts, corpus-safe.

Writes two artifacts to the run directory after every RLM iteration:

1. ``iterations/iteration_NNNN.json`` — a per-iteration JSON snapshot containing
   code block metadata, safe variable values, and a corpus reference.  Values are
   the actual REPL-namespace values (not just type/size), which enables offline
   analysis of the run.  The corpus sentinel is redacted at write time so this
   file can be stored or shared without leaking paper text.

2. ``repl_state.pickle`` — the latest REPL-namespace state as a Python pickle,
   overwritten after every iteration so that a run can be resumed or inspected.
   Large or un-picklable values are replaced with structured tombstones instead
   of crashing the run.

Both artifacts enforce the corpus-safety invariant:
- ``context`` and all ``context*`` keys are excluded.
- ``_*`` keys (REPL internals) are excluded.
- The corpus is referenced by filename (``parsed_full_text.txt``) only.
- ``redact_corpus`` is run over the JSON text as a final egress guard.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
from pathlib import Path

from backend.agents.rlm.sse_bridge import redact_corpus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_CORPUS_REF = "parsed_full_text.txt"
_MAX_PICKLE_VALUE_BYTES = 256 * 1024  # 256 KiB


# ---------------------------------------------------------------------------
# ReplSnapshotWriter
# ---------------------------------------------------------------------------


class ReplSnapshotWriter:
    """Write corpus-safe iteration snapshots and a rolling REPL-state pickle.

    Args:
        project_dir: The run directory (``runs/<id>/``).  Both artifact paths
                     are relative to this directory.
        sentinels:   Optional corpus sentinels (first ``_SENTINEL_LEN`` chars of
                     each ``context_dict`` value).  Passed to
                     :func:`~backend.agents.rlm.sse_bridge.redact_corpus` for a
                     final egress guard on the JSON snapshot text.
    """

    def __init__(
        self,
        *,
        project_dir: Path,
        sentinels: list[str] | None = None,
    ) -> None:
        self._project_dir = project_dir
        self._iterations_dir = project_dir / "iterations"
        self._iterations_dir.mkdir(parents=True, exist_ok=True)
        self._sentinels: list[str] = sentinels or []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_variables(self, locals_: dict) -> dict:
        """Return a filtered copy of *locals_* with corpus and private keys removed.

        Excludes:
        - Keys starting with ``_`` (REPL internals / builtins).
        - The key ``"context"`` or any key starting with ``"context"`` — these
          hold the paper corpus and must never appear in any persisted artifact.
        """
        out: dict = {}
        for name, value in locals_.items():
            if name.startswith("_"):
                continue
            if name == "context" or name.startswith("context"):
                continue
            out[name] = value
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, iteration, index: int) -> None:
        """Snapshot one iteration — corpus-safe, exception-safe.

        Writes ``iterations/iteration_{index:04d}.json`` and overwrites
        ``repl_state.pickle``.  Any exception is caught, logged, and swallowed
        so snapshotting never crashes a run.

        Args:
            iteration: An ``RLMIteration`` from the ``rlms`` library.
            index:     The 1-based iteration counter (same value used in
                       :func:`~backend.agents.rlm.sse_bridge.sanitize_iteration`).
        """
        try:
            self._write(iteration, index)
        except Exception:  # noqa: BLE001
            logger.exception(
                "repl_snapshot: failed to write snapshot for iteration %d — "
                "snapshotting is non-fatal; run continues",
                index,
            )

    def _write(self, iteration, index: int) -> None:
        locals_ = (
            iteration.code_blocks[-1].result.locals
            if iteration.code_blocks
            else {}
        )
        safe_vars = self._safe_variables(locals_)

        # The per-iteration JSON records every iteration — including a
        # pure-reasoning iteration that executed no code (variables == {}).
        self._write_iteration_json(iteration, index, safe_vars)

        # repl_state.pickle is the *latest REPL state* snapshot. The RLM root
        # interleaves pure-reasoning iterations (no code blocks) between code
        # iterations; such an iteration leaves the persistent REPL namespace
        # unchanged, so it must NOT clobber the last good snapshot with {}.
        if iteration.code_blocks:
            self._write_repl_pickle(index, safe_vars)

    def _write_iteration_json(self, iteration, index: int, safe_vars: dict) -> None:
        """Write ``iterations/iteration_{index:04d}.json`` atomically."""
        snapshot = {
            "iteration": index,
            "timing": iteration.iteration_time,
            "code_blocks": [
                {
                    "code": b.code,
                    "stdout_chars": len(b.result.stdout or ""),
                    "stderr_chars": len(b.result.stderr or ""),
                }
                for b in iteration.code_blocks
            ],
            "variables": safe_vars,
            "corpus_ref": _CORPUS_REF,
        }

        blob = json.dumps(snapshot, default=str, indent=2)
        blob = redact_corpus(blob, self._sentinels)

        dest = self._iterations_dir / f"iteration_{index:04d}.json"
        fd, tmp_path = tempfile.mkstemp(dir=self._iterations_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(blob)
            os.replace(tmp_path, dest)
        except Exception:
            # Clean up temp file if replace failed
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _write_repl_pickle(self, index: int, safe_vars: dict) -> None:
        """Overwrite ``repl_state.pickle`` atomically with the latest REPL state."""
        pickle_vars: dict = {}
        for name, value in safe_vars.items():
            try:
                blob = pickle.dumps(value)
            except Exception:  # noqa: BLE001
                pickle_vars[name] = {"__unpicklable__": type(value).__name__}
                continue
            if len(blob) > _MAX_PICKLE_VALUE_BYTES:
                pickle_vars[name] = {
                    "__omitted_large__": len(blob),
                    "__type__": type(value).__name__,
                }
            else:
                pickle_vars[name] = value

        payload = {
            "schema_version": 1,
            "iteration": index,
            "variables": pickle_vars,
            "corpus_ref": _CORPUS_REF,
        }

        dest = self._project_dir / "repl_state.pickle"
        fd, tmp_path = tempfile.mkstemp(dir=self._project_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(payload, fh)
            os.replace(tmp_path, dest)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


__all__ = ["ReplSnapshotWriter"]
