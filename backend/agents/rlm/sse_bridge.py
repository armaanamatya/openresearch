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

# Thresholds for rubric area status derivation in build_rubric_score_event.
# score >= RUBRIC_AREA_PASS_THRESHOLD    → "pass"
# score >= RUBRIC_AREA_PARTIAL_THRESHOLD → "partial"
# otherwise                              → "fail"
# These are UI affordances, not rubric gates; the rubric gate uses target_score.
RUBRIC_AREA_PASS_THRESHOLD: float = 0.7
RUBRIC_AREA_PARTIAL_THRESHOLD: float = 0.4


# ---------------------------------------------------------------------------
# RDR cluster event builders
# ---------------------------------------------------------------------------

_REQUIRED_CLUSTER_LEAF_KEYS = frozenset({"id", "weight", "requirements"})


def build_cluster_started(
    *,
    cluster_id: str,
    cluster_title: str,
    leaves: list[dict[str, Any]],
    iteration: int,
) -> dict[str, Any]:
    """Build the public ``cluster_started`` RDR SSE payload."""
    for leaf in leaves:
        missing = _REQUIRED_CLUSTER_LEAF_KEYS.difference(leaf)
        if missing:
            raise KeyError(f"cluster leaf missing required keys: {sorted(missing)}")
    return {
        "cluster_id": cluster_id,
        "cluster_title": cluster_title,
        "leaves": leaves,
        "iteration": iteration,
    }


def build_cluster_artifact_emitted(
    *,
    cluster_id: str,
    artifact_path: str,
    byte_size: int,
    language: str | None,
) -> dict[str, Any]:
    """Build the public ``cluster_artifact_emitted`` RDR SSE payload."""
    return {
        "cluster_id": cluster_id,
        "artifact_path": artifact_path,
        "byte_size": byte_size,
        "language": language,
    }


def build_cluster_scored(
    *,
    cluster_id: str,
    score: float,
    leaf_scores: dict[str, float],
    degraded: bool,
) -> dict[str, Any]:
    """Build the public ``cluster_scored`` RDR SSE payload."""
    return {
        "cluster_id": cluster_id,
        "score": score,
        "leaf_scores": leaf_scores,
        "degraded": degraded,
    }


def build_repair_dispatched(
    *,
    cluster_id: str,
    attempt: int,
    prior_score: float,
    failed_leaves: list[str],
) -> dict[str, Any]:
    """Build the public ``repair_dispatched`` RDR SSE payload."""
    return {
        "cluster_id": cluster_id,
        "attempt": attempt,
        "prior_score": prior_score,
        "failed_leaves": failed_leaves,
    }


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
    if _sentinels:
        response = redact_corpus(response, _sentinels)  # close the M-REDACT egress

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
        ctx: Any = None,
    ) -> None:
        super().__init__(log_dir=None)
        self._emit = emit
        self._checkpointer = checkpointer
        self._sentinels: list[str] = sentinels or []
        self._snapshot_writer = snapshot_writer
        self._next_index: int = 0
        self._index_lock = threading.Lock()  # A1-M3: guard concurrent index increments
        self._ctx = ctx  # RunContext — for current_iteration plumbing (optional)

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

        Updates ``ctx.current_iteration`` (when ``ctx`` was supplied) to the
        just-completed 1-based index AFTER emitting and checkpointing.
        Primitives running inside the *next* iteration therefore see the last
        completed iteration's index — a one-behind ("last-completed") semantic.
        This is intentional and documented: the index is a UI label, not a
        precise in-flight counter.

        Args:
            iteration: The raw ``RLMIteration`` from ``rlms``.  Treated as
                       read-only; never stored or forwarded.
        """
        index = self.next_index()
        clean = sanitize_iteration(iteration, index, self._sentinels)
        self._emit(_repl_iteration_event(clean))
        self._checkpointer.record(clean)
        if self._snapshot_writer is not None:
            self._snapshot_writer.write(iteration, clean["iteration"])
        # Update ctx.current_iteration after emit/checkpoint so any failure in
        # those steps does not leave ctx with a stale counter.
        if self._ctx is not None:
            self._ctx.current_iteration = index
            # F-06: reset the forced-iteration policy's per-turn trackers at the
            # real REPL turn boundary, not only on a FINAL_VAR refusal. Without
            # this, single failing run_experiment outcomes from DIFFERENT
            # iterations accumulate and falsely trip the two-experiment-per-turn
            # guard. Fail-soft — a reset (a list assignment) must never break
            # logging.
            _pol = getattr(self._ctx, "_forced_iteration_policy", None)
            if _pol is not None:
                try:
                    _pol.on_iteration_advance()
                except Exception:  # noqa: BLE001 — best-effort turn-boundary reset
                    pass


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


def build_candidate_proposed_event(
    *,
    iteration: int,
    round: int,
    candidate: dict,
    parent_id: str | None = None,
) -> dict:
    """Build a ``candidate_proposed`` dashboard event.

    Emitted once per hypothesis returned by a successful ``propose_improvements``
    call.  Field names match the wire contract in
    ``frontend/src/lib/events/rlm-events.ts`` exactly.

    Args:
        iteration:  1-based root-loop iteration index (from ``RunContext.current_iteration``).
        round:      1-based per-run count of ``propose_improvements`` calls (from
                    ``RunContext.propose_round``).
        candidate:  Dict with keys ``id``, ``title``, ``category``, ``description``,
                    ``reasoning`` — derived from ``ImprovementHypothesis`` fields.
        parent_id:  The node this candidate branches from.  Omitted from the event
                    dict when ``None`` (TS optional property means absent, not null).
    """
    _CANDIDATE_KEYS = {"id", "title", "category", "description", "reasoning"}
    candidate_payload: dict = {k: candidate[k] for k in _CANDIDATE_KEYS}
    # Include display_title when present (computed by _friendly_candidate_title in binding.py).
    if "display_title" in candidate:
        candidate_payload["display_title"] = candidate["display_title"]
    ev: dict = {
        "event": "candidate_proposed",
        "timestamp": _now_iso(),
        "iteration": iteration,
        "round": round,
        "candidate": candidate_payload,
    }
    if parent_id is not None:
        ev["parent_id"] = parent_id
    return ev


def build_candidate_outcome_event(
    *,
    iteration: int,
    candidate_id: str,
    outcome: str,
    rubric_delta: float | None,
) -> dict:
    """Build a ``candidate_outcome`` dashboard event.

    Emitted when the run-level orchestrator determines the outcome for a
    candidate (promoted, failed, etc.).  Field names match the wire contract in
    ``frontend/src/lib/events/rlm-events.ts`` exactly.

    Args:
        iteration:    Root-loop iteration when the outcome was determined.
        candidate_id: Matches ``candidate_proposed.candidate.id``.
        outcome:      One of ``"running"``, ``"promoted"``, ``"marginal"``,
                      ``"failed"``, ``"skipped"``, ``"declined"``.
        rubric_delta: Overall-score change this candidate produced, or ``None``.
    """
    return {
        "event": "candidate_outcome",
        "timestamp": _now_iso(),
        "iteration": iteration,
        "candidate_id": candidate_id,
        "outcome": outcome,
        "rubric_delta": rubric_delta,
    }


def build_rubric_score_event(
    *,
    iteration: int,
    score: float,
    target: float,
    areas: list[dict],
) -> dict:
    """Build a ``rubric_score`` dashboard event.

    Emitted after a successful ``verify_against_rubric`` call.  Each area's
    ``status`` is derived from its ``score`` using module-level thresholds
    (``RUBRIC_AREA_PASS_THRESHOLD``, ``RUBRIC_AREA_PARTIAL_THRESHOLD``) — it is
    a UI affordance, not a rubric gate decision.  Field names match the wire
    contract in ``frontend/src/lib/events/rlm-events.ts`` exactly.

    Args:
        iteration:  1-based root-loop iteration index.
        score:      Overall rubric score, 0–1 (from ``RubricVerification.overall_score``).
        target:     Rubric target, 0–1 (from ``RubricVerification.target_score``).
        areas:      List of area dicts with keys ``area``, ``score``, ``weight``;
                    ``status`` is derived and added here.
    """
    def _area_status(area_score: float) -> str:
        if area_score >= RUBRIC_AREA_PASS_THRESHOLD:
            return "pass"
        if area_score >= RUBRIC_AREA_PARTIAL_THRESHOLD:
            return "partial"
        return "fail"

    return {
        "event": "rubric_score",
        "timestamp": _now_iso(),
        "iteration": iteration,
        "score": score,
        "target": target,
        "areas": [
            {
                "area": a["area"],
                "score": a["score"],
                "weight": a["weight"],
                "status": _area_status(a["score"]),
            }
            for a in areas
        ],
    }


def build_iteration_heartbeat_event(
    *,
    iteration: int | None,
    counter: int,
    note: str,
) -> dict:
    """Build an ``iteration_heartbeat`` dashboard event.

    Emitted by the ``heartbeat()`` primitive directly (not via ``wrap_primitive``
    alone) to give the UI a dedicated, easily-filterable liveness signal.

    Args:
        iteration:  Current root-loop iteration index (1-based), or ``None`` when
                    called before the first iteration has been logged.
        counter:    Monotonic per-process counter incremented on every call.
        note:       Optional human-readable note from the root model, e.g.
                    ``"about to implement_baseline"``.
    """
    return {
        "event": "iteration_heartbeat",
        "timestamp": _now_iso(),
        "iteration": iteration,
        "counter": counter,
        "note": note,
    }


def build_run_warning_event(
    *,
    level: str = "warn",
    code: str,
    message: str,
) -> dict:
    """Build a ``run_warning`` dashboard event.

    Emitted by the stderr watchdog when a degraded condition is detected
    (e.g. the SDK aclose deadlock pattern).  Passes through the SSE egress
    unchanged — the egress sanitizer treats ``run_warning`` like any other
    dashboard event; its payload carries no corpus data.

    Args:
        level:   Severity string, typically ``"warn"`` or ``"error"``.
        code:    Machine-readable tag, e.g. ``"sdk_aclose_loop"``.
        message: Human-readable description surfaced in the UI chip.
    """
    return {
        "event": "run_warning",
        "timestamp": _now_iso(),
        "level": level,
        "code": code,
        "message": message,
    }


__all__ = [
    "RUBRIC_AREA_PARTIAL_THRESHOLD",
    "RUBRIC_AREA_PASS_THRESHOLD",
    "ReproLabRLMLogger",
    "build_cluster_artifact_emitted",
    "build_cluster_scored",
    "build_cluster_started",
    "build_candidate_outcome_event",
    "build_candidate_proposed_event",
    "build_iteration_heartbeat_event",
    "build_repair_dispatched",
    "build_rubric_score_event",
    "build_run_complete_event",
    "build_run_warning_event",
    "build_sub_rlm_complete_event",
    "build_sub_rlm_spawned_event",
    "make_emit",
    "make_on_subcall_complete",
    "make_on_subcall_start",
    "redact_corpus",
    "sanitize_iteration",
]
