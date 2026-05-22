"""SSE bridge: corpus sanitizer, RLM logger, event schema, and emission locking.

This module is the *single chokepoint* between the raw ``rlm`` library and any
data that leaves the process — whether via the SSE/dashboard stream or the
event-store checkpoint.  The critical invariant (Algorithm-2, §9.1 of the
design spec) is:

    No value from ``RLMIteration.code_blocks[*].result.locals`` may ever reach
    the stream or the checkpoint, especially the ``context`` key that holds the
    entire paper corpus.

All public functions and classes in this module enforce that invariant.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable

from rlm.core.types import RLMIteration
from rlm.logger.rlm_logger import RLMLogger

from backend.agents.dashboard_emitter import DashboardEmitter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESPONSE_MAX_CHARS: int = 4_000
_STDOUT_PREFIX_MAX_CHARS: int = 200
_PROMPT_PREVIEW_MAX_CHARS: int = 200
_SENTINEL_LEN: int = 200  # chars from each corpus value used as a leak sentinel


# ---------------------------------------------------------------------------
# M-REDACT corpus-leak guard — applied at every egress point
# ---------------------------------------------------------------------------


def redact_corpus(text: str, sentinels: list[str]) -> str:
    """Replace any corpus sentinel that appears in *text* with ``[REDACTED]``.

    *sentinels* are the first ``_SENTINEL_LEN`` characters of each corpus value
    from ``context_dict`` (computed once per run by the caller).  A sentinel
    appearing verbatim in streamed or persisted text means the Algorithm-2
    invariant has been violated — we redact rather than crash so the run
    continues and the leak is visible in the stream without exposing the data.

    Only non-empty sentinels of at least 16 chars are checked to avoid
    false-positive redactions on short common strings.

    Args:
        text:      The string to sanitise.
        sentinels: First ``_SENTINEL_LEN`` chars of each corpus value.

    Returns:
        The sanitised string with any sentinel occurrence replaced.
    """
    for sentinel in sentinels:
        if len(sentinel) >= 16 and sentinel in text:
            text = text.replace(sentinel, "[REDACTED]")
    return text


# ---------------------------------------------------------------------------
# 9.1 The corpus sanitizer — the single chokepoint
# ---------------------------------------------------------------------------


def sanitize_iteration(
    iteration: RLMIteration,
    index: int,
    sentinels: list[str] | None = None,
) -> dict:
    """Return a corpus-free projection of one ``RLMIteration``.

    This is the ONLY form of an ``RLMIteration`` that may be streamed (SSE),
    persisted (event store), or snapshotted.  It never returns:

    - Any value from ``result.locals`` (which contains the paper corpus under
      the ``context`` key and raw primitive inputs/outputs under other keys).
    - Any key whose name is ``context`` or starts with ``context``.
    - The raw ``iteration.prompt`` (full message history).
    - The raw ``iteration.final_answer``.

    The output shape matches §9.1 of the design spec:

    .. code-block:: python

        {
            "iteration": int,
            "response": str,           # bounded to ≤4 000 chars
            "code_blocks": [
                {
                    "code": str,
                    "stdout_meta": {"length": int, "prefix": str, "has_traceback": bool},
                    "stderr_meta": {"length": int, "prefix": str, "has_traceback": bool},
                    "vars": {name: {"type": str, "size": int}, ...},
                    "sub_calls": int,
                }
            ],
            "sub_calls": int,          # total rlm_calls across all blocks
            "timing": float | None,    # iteration_time
        }

    Args:
        iteration: The raw ``RLMIteration`` from the ``rlms`` library.
        index:     1-based iteration counter (supplied by ``ReproLabRLMLogger``).
        sentinels: Optional list of corpus sentinels (first ``_SENTINEL_LEN``
                   chars of each corpus value).  When provided, stdout/stderr
                   prefixes are run through :func:`redact_corpus` (M-REDACT /
                   audit A1-M2) to catch any Algorithm-2 violations at egress.

    Returns:
        A sanitized dict that is safe to stream, persist, and snapshot.
    """
    _sentinels: list[str] = sentinels or []
    clean_blocks: list[dict] = []
    total_sub_calls = 0

    for block in iteration.code_blocks:
        result = block.result

        stdout_meta = _stream_metadata(result.stdout, _sentinels)
        stderr_meta = _stream_metadata(result.stderr, _sentinels)
        vars_meta = _locals_metadata(result.locals)
        block_sub_calls = len(result.rlm_calls) if result.rlm_calls else 0
        total_sub_calls += block_sub_calls

        clean_blocks.append({
            "code": block.code,
            "stdout_meta": stdout_meta,
            "stderr_meta": stderr_meta,
            "vars": vars_meta,
            "sub_calls": block_sub_calls,
        })

    response = iteration.response or ""
    if len(response) > _RESPONSE_MAX_CHARS:
        response = response[:_RESPONSE_MAX_CHARS]

    return {
        "iteration": index,
        "response": response,
        "code_blocks": clean_blocks,
        "sub_calls": total_sub_calls,
        "timing": iteration.iteration_time,
    }


def _stream_metadata(text: str | None, sentinels: list[str] | None = None) -> dict:
    """Reduce stdout/stderr to safe metadata only (never the raw content).

    The prefix (≤200 chars) is passed through :func:`redact_corpus` when
    *sentinels* are provided (M-REDACT / audit A1-M2) — a primitive that
    echoes corpus content to stdout would otherwise leak the first 200 chars
    of that content into the SSE stream.

    Returns ``{"length": int, "prefix": str (≤200 chars), "has_traceback": bool}``.
    """
    if text is None:
        text = ""
    prefix = text[:_STDOUT_PREFIX_MAX_CHARS]
    if sentinels:
        prefix = redact_corpus(prefix, sentinels)
    has_traceback = "Traceback (most recent call last)" in text
    return {
        "length": len(text),
        "prefix": prefix,
        "has_traceback": has_traceback,
    }


def _locals_metadata(locals_: dict) -> dict:
    """Reduce REPL locals to a variable-shape manifest — never values.

    Excludes:
    - Keys that start with ``_`` (private REPL internals).
    - Any key that is ``"context"`` or starts with ``"context"`` — these hold
      the paper corpus and must NEVER appear in any output, even as a key.

    For each remaining key, emits ``{name: {"type": str, "size": int}}`` where
    ``size`` is ``len(str(value))`` — a rough byte count without the value.
    """
    out: dict[str, dict] = {}
    for name, value in locals_.items():
        if name.startswith("_"):
            continue
        if name == "context" or name.startswith("context"):
            # Hard exclusion: drop the key entirely; never reflect it.
            continue
        type_name = type(value).__name__
        try:
            size = len(str(value))
        except Exception:  # noqa: BLE001
            size = -1
        out[name] = {"type": type_name, "size": size}
    return out


# ---------------------------------------------------------------------------
# 9.2 ReproLabRLMLogger
# ---------------------------------------------------------------------------


class ReproLabRLMLogger(RLMLogger):
    """``RLMLogger`` subclass that sanitizes every iteration before emission.

    The base ``RLMLogger.log()`` method is intentionally NOT called — doing so
    would capture the raw ``RLMIteration.to_dict()`` (which includes the corpus
    in ``locals``) in the in-memory trajectory and, if ``log_dir`` were set, on
    disk.  We own a sanitized trajectory instead; ``log_dir=None`` ensures the
    base never opens a file.

    Args:
        emit:            A thread-safe callable produced by :func:`make_emit`.
                         Accepts a pre-built event dict and writes it to the
                         dashboard JSONL stream.
        checkpointer:    An :class:`~backend.agents.rlm.checkpoint.IterationCheckpointer`
                         whose ``record(clean)`` persists the sanitized dict to the
                         event store and snapshot file.
        sentinels:       Optional corpus sentinels (first ``_SENTINEL_LEN`` chars of
                         each ``context_dict`` value) threaded into
                         :func:`sanitize_iteration` for M-REDACT / A1-M2 stdout
                         prefix hardening.  Computed once at run-start; ``None``
                         disables the secondary redaction pass.
        snapshot_writer: Optional :class:`~backend.agents.rlm.repl_snapshot.ReplSnapshotWriter`
                         that writes per-iteration JSON snapshots and a rolling
                         ``repl_state.pickle`` to the run directory (issue #62 DC#4).
                         ``None`` disables snapshotting (back-compat default).
    """

    def __init__(
        self,
        *,
        emit: Callable[[dict], None],
        checkpointer: Any,
        sentinels: list[str] | None = None,
        snapshot_writer: Any = None,
    ) -> None:
        super().__init__(log_dir=None)
        self._emit = emit
        self._checkpointer = checkpointer
        self._sentinels: list[str] = sentinels or []
        self._snapshot_writer = snapshot_writer
        self._next_index: int = 0
        self._index_lock = threading.Lock()  # A1-M3: guard concurrent index increments

    def next_index(self) -> int:
        """Return the next 1-based iteration index and advance the counter (thread-safe)."""
        with self._index_lock:
            self._next_index += 1
            return self._next_index

    @property
    def iteration_count(self) -> int:
        """Total iterations logged so far.

        Overrides ``RLMLogger.iteration_count`` — the base's ``_iteration_count``
        is never incremented because :meth:`log` deliberately does not call
        ``super().log()`` (see the class docstring).
        """
        return self._next_index

    def log(self, iteration: RLMIteration) -> None:
        """Sanitize, emit, and checkpoint one iteration.

        Does NOT call ``super().log(iteration)`` — see class docstring.

        Args:
            iteration: The raw ``RLMIteration`` from ``rlms``.  Treated as
                       read-only; never stored or forwarded.
        """
        clean = sanitize_iteration(iteration, self.next_index(), self._sentinels)
        self._emit(_repl_iteration_event(clean))
        self._checkpointer.record(clean)
        if self._snapshot_writer is not None:
            self._snapshot_writer.write(iteration, clean["iteration"])


# ---------------------------------------------------------------------------
# 9.3 Thread-safe emit factory
# ---------------------------------------------------------------------------


def make_emit(dashboard: DashboardEmitter) -> Callable[[dict], None]:
    """Build a thread-safe ``emit`` closure backed by ``dashboard._emit``.

    ``DashboardEmitter._emit`` opens and writes the JSONL file without a lock.
    This closure owns a ``threading.Lock`` so that the worker thread (via
    ``ReproLabRLMLogger``) and the ``rlm`` callback thread (via
    ``on_subcall_start`` / ``on_subcall_complete``) never interleave writes.

    Args:
        dashboard: The ``DashboardEmitter`` for the current run.

    Returns:
        A callable that accepts a pre-built event dict and writes it atomically.
    """
    lock = threading.Lock()

    def _emit(event: dict) -> None:
        with lock:
            dashboard._emit(event)  # noqa: SLF001 — intentional, documented

    return _emit


# ---------------------------------------------------------------------------
# 9.4 Event builders
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repl_iteration_event(clean: dict) -> dict:
    """Build a ``repl_iteration`` dashboard event from a sanitized iteration.

    The ``clean`` dict is the output of :func:`sanitize_iteration`.  All fields
    are already corpus-free.
    """
    return {
        "event": "repl_iteration",
        "timestamp": _now_iso(),
        "iteration": clean["iteration"],
        "response": clean["response"],
        "code_blocks": clean["code_blocks"],
        "sub_calls": clean["sub_calls"],
        "timing": clean["timing"],
    }


def build_sub_rlm_spawned_event(depth: int, model: str, prompt_preview: str) -> dict:
    """Build a ``sub_rlm_spawned`` dashboard event.

    Args:
        depth:          Nesting depth of the sub-call (1 = first level child).
        model:          Model identifier string as reported by ``rlm``.
        prompt_preview: Raw prompt preview from ``rlm``; bounded to
                        ≤200 chars to prevent corpus leakage.
    """
    return {
        "event": "sub_rlm_spawned",
        "timestamp": _now_iso(),
        "depth": depth,
        "model": model,
        "prompt_preview": prompt_preview[:_PROMPT_PREVIEW_MAX_CHARS],
    }


def build_sub_rlm_complete_event(
    depth: int,
    model: str,
    duration: float,
    error: str | None,
) -> dict:
    """Build a ``sub_rlm_complete`` dashboard event.

    Args:
        depth:    Nesting depth of the sub-call.
        model:    Model identifier string.
        duration: Wall-clock duration in seconds as reported by ``rlm``.
        error:    Error message string, or ``None`` on success.
    """
    return {
        "event": "sub_rlm_complete",
        "timestamp": _now_iso(),
        "depth": depth,
        "model": model,
        "duration_ms": round(duration * 1000),
        "error": error,
    }


def build_run_complete_event(
    *,
    status: str,
    iterations: int,
    rubric_score: float | None,
    cost_usd: float | None,
    final_report_path: str | None,
) -> dict:
    """Build a ``run_complete`` dashboard event.

    Args:
        status:             Run outcome: ``"completed"``, ``"partial"``, or ``"failed"``.
        iterations:         Total number of RLM iterations executed.
        rubric_score:       Final rubric score (0–1), or ``None`` if unavailable.
        cost_usd:           Total cost in USD, or ``None`` if unavailable.
        final_report_path:  Path to the written ``final_report.json``, or ``None``.
    """
    return {
        "event": "run_complete",
        "timestamp": _now_iso(),
        "status": status,
        "iterations": iterations,
        "rubric_score": rubric_score,
        "cost_usd": cost_usd,
        "final_report_path": final_report_path,
    }


# ---------------------------------------------------------------------------
# on_subcall_* callback builders
# ---------------------------------------------------------------------------


def make_on_subcall_start(emit: Callable[[dict], None]) -> Callable[[int, str, str], None]:
    """Return an ``on_subcall_start`` callback wired to ``emit``.

    The returned callable matches the ``rlm`` signature:
    ``(depth: int, model: str, prompt_preview: str) -> None``.

    Args:
        emit: The thread-safe emit closure from :func:`make_emit`.
    """

    def _on_subcall_start(depth: int, model: str, prompt_preview: str) -> None:
        emit(build_sub_rlm_spawned_event(depth, model, prompt_preview))

    return _on_subcall_start


def make_on_subcall_complete(
    emit: Callable[[dict], None],
) -> Callable[[int, str, float, str | None], None]:
    """Return an ``on_subcall_complete`` callback wired to ``emit``.

    The returned callable matches the ``rlm`` signature:
    ``(depth: int, model: str, duration: float, error: str | None) -> None``.

    Args:
        emit: The thread-safe emit closure from :func:`make_emit`.
    """

    def _on_subcall_complete(
        depth: int,
        model: str,
        duration: float,
        error: str | None,
    ) -> None:
        emit(build_sub_rlm_complete_event(depth, model, duration, error))

    return _on_subcall_complete


__all__ = [
    "ReproLabRLMLogger",
    "build_run_complete_event",
    "build_sub_rlm_complete_event",
    "build_sub_rlm_spawned_event",
    "make_emit",
    "make_on_subcall_complete",
    "make_on_subcall_start",
    "redact_corpus",
    "sanitize_iteration",
]
