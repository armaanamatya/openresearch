"""System-driven run watchdog — catches stalled runs the agent can't.

Every existing liveness signal in the RLM harness is AGENT-DRIVEN: the
agent calls ``heartbeat()``, the RLM root loop emits ``iteration_heartbeat``
between iterations, sub-agents return tool-use results.  When the agent
itself is wedged (claude-agent-sdk silently retrying a rate-limited LLM
call, a sub_rlm spawn that never returns, an NCCL deadlock inside the
pod, a HuggingFace download stuck on a dead mirror) every one of these
goes silent.  Adam v5 / Adam v6 / Dropout v5 / Dropout v6 all showed the
same symptom: container alive, process alive, billing accruing, ZERO
events for 30-50 minutes.

This module is the SYSTEM-DRIVEN inverse: an async polling task that
watches three independent signals and ORs them (min-staleness wins —
ANY fresh signal means the run is alive):

  1. ``exec.log`` mtime — written incrementally by ``_execute_in_sandbox``
     for both LocalDocker and RunPod (Lane B).
  2. ``.heartbeat`` mtime — written every 30 s by a daemon injected into
     the pod's bootstrap commands.  Detects pod-level wedges that don't
     produce ANY stdout/stderr (e.g. an NCCL deadlock that hangs torch).
  3. Dashboard SSE event silence — tail ``runs/<id>/dashboard_events.jsonl``,
     stat its mtime.  Catches agent-side hangs where the pod is idle
     waiting on the agent's next instruction.

Two thresholds with tiered actions:

  * ``warn_after_seconds`` (default 600 = 10 min): emit a ``run_warning``
    SSE event with the diagnostic payload.  Run continues.
  * ``kill_after_seconds`` (default 1500 = 25 min): invoke ``on_kill``.
    With Lane N probe-recover the callback may return ``RECOVERED`` to
    keep the pod warm; only after the recovery budget is exhausted (or
    a probe fails) does the pod actually get destroyed. The kill bound
    is generous because a train.py that prints every N epochs at slow
    epoch speed legitimately produces silent stretches of 10-15 min;
    killing a working pod is much worse than a few extra min of wait.

Design contract:

  * Fail-soft on every path.  A ``stat()`` failure must NEVER raise out
    of the loop — the watchdog must NEVER crash the run it's watching.
  * Pure async task — spawned by the caller via ``asyncio.create_task``.
    No background threads, no global state, no IPC.
  * Bootstrap-grace period: if NO signal has EVER been observed (file
    doesn't exist yet) the watchdog reports "ok" — we can't tell if the
    run is wedged in bootstrap or has just started.  Once at least one
    signal has been observed, staleness from that point onward IS
    actionable.
  * ``on_kill`` is invoked AT MOST ONCE — repeated kill verdicts in
    subsequent poll cycles are a no-op.
  * Disable entirely via ``OPENRESEARCH_WATCHDOG_DISABLED=true`` — the
    coroutine returns immediately, no polling.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


class KillVerdict(str, enum.Enum):
    """Return-value contract for the ``on_kill`` callback.

    Lane N — replaces the prior all-or-nothing destroy semantics with a
    two-tier policy:

    * ``DESTROY`` — pod is truly dead OR recovery budget exhausted; the
      callback has already torn down the pod (or run-level cleanup will).
      The watchdog exits its poll loop.
    * ``RECOVERED`` — probe found the pod alive on a fresh transport; the
      callback has soft-killed the wedged in-pod process and reset
      whatever signal it can. The watchdog CONTINUES polling — staleness
      will reset naturally once the agent writes the next exec.log line.

    ``None`` (or any unrecognised return) is treated as ``DESTROY`` for
    backward compatibility with the v1 callback signature.
    """

    DESTROY = "destroy"
    RECOVERED = "recovered"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


_DISABLE_ENV_VAR: str = "OPENRESEARCH_WATCHDOG_DISABLED"
_WARN_ENV_VAR: str = "OPENRESEARCH_WATCHDOG_WARN_SECONDS"
_KILL_ENV_VAR: str = "OPENRESEARCH_WATCHDOG_KILL_SECONDS"
_POLL_ENV_VAR: str = "OPENRESEARCH_WATCHDOG_POLL_INTERVAL_SECONDS"


# ---------------------------------------------------------------------------
# PR-ι.4 — Per-primitive idle-threshold gradient
# ---------------------------------------------------------------------------

# Baseline idle thresholds (in seconds) for each named primitive.
# Primitives not listed here fall back to _DEFAULT_IDLE_BASELINE_S.
#
# Rationale:
#   - implement_baseline: Sonnet code-writing agent; can legitimately take
#     up to 4h per the existing timeout in primitives.py. Match that cap.
#   - run_experiment: GPU training; legitimately takes hours. Set to 4h so
#     this file-mtime/SSE idle threshold is >= the new 60-min inner stall
#     window in local_process (Pillar 2, spec 2026-06-08) — the watchdog
#     must NEVER pre-empt the GPU/CPU-aware inner stall. The real in-process
#     hang signal is GPU/CPU util (see compute-liveness check below), not a
#     short file-mtime idle. Mirrors the implement_baseline 4h rationale.
#   - build_environment: Docker image build; up to 30 min is plausible on
#     a cold node with many pip deps.
#   - All other primitives (understand_section, extract_hyperparameters,
#     detect_environment, plan_reproduction, verify_against_rubric, etc.)
#     are LLM-driven with < 5 min expected latency; 30 min is generous.
PRIMITIVE_IDLE_BASELINE_S: dict[str, float] = {
    "implement_baseline": 14400.0,  # 4 h — matches existing timeout in primitives.py
    "run_experiment": 14400.0,      # 4 h — >= inner stall window; util is the real hang signal
    "build_environment": 1800.0,    # 30 min — Docker build on cold node
}
_DEFAULT_IDLE_BASELINE_S: float = 1800.0  # 30 min for any other primitive


def effective_idle_threshold(primitive_name: str | None, config: "WatchdogConfig") -> float:
    """Return the effective kill-after-seconds for the currently-active primitive.

    When ``primitive_name`` is None (unknown / no primitive active), the config's
    global ``kill_after_seconds`` is returned unchanged.  When a known primitive
    is active, its specific baseline replaces the global threshold if larger —
    we never make the threshold MORE aggressive, only more lenient.

    This prevents the watchdog from killing a legitimately slow ``run_experiment``
    (GPU training, hours) or ``implement_baseline`` (Sonnet code-writing, 4h cap)
    while still catching truly wedged non-experiment primitives within 30 min.
    """
    if primitive_name is None:
        return config.kill_after_seconds
    baseline = PRIMITIVE_IDLE_BASELINE_S.get(primitive_name, _DEFAULT_IDLE_BASELINE_S)
    # Never more aggressive than the operator-configured threshold; only more lenient.
    return max(baseline, config.kill_after_seconds)


def is_enabled() -> bool:
    """Return ``False`` when ``OPENRESEARCH_WATCHDOG_DISABLED=true``.  Case-insensitive."""
    return os.environ.get(_DISABLE_ENV_VAR, "").lower() not in {"true", "1", "yes", "on"}


@dataclass(slots=True)
class WatchdogConfig:
    """Tunable thresholds.  Env-var aware via :meth:`from_env`.

    Policy (2026-05-24): NO warn-then-kill cascade.  If a run is stale long
    enough to be unrecoverable (no exec.log growth, no heartbeat, no SSE
    events) it is BLOCKING the rubric — we kill immediately rather than
    wait through a courtesy warning window.  ``warn_after_seconds`` is kept
    for backwards-compat but no longer triggers a separate event by default;
    setting it lower than ``kill_after_seconds`` enables an optional warning
    breadcrumb.
    """

    # 10 min default: long enough to clear legitimate bootstrap (cuda-devel
    # pip install can take 7-10 min on a cold pod) but short enough that a
    # genuinely wedged run doesn't burn $ for 30+ min.  warn==kill collapses
    # the policy to "stale -> kill" (no warn-then-kill courtesy window) per
    # operator direction 2026-05-24.
    warn_after_seconds: float = 600.0
    kill_after_seconds: float = 1500.0
    poll_interval_seconds: float = 30.0
    heartbeat_filename: str = ".heartbeat"
    exec_log_filename: str = "exec.log"
    dashboard_events_filename: str = "dashboard_events.jsonl"

    # --- Compute-liveness (Pillar 2b, spec 2026-06-08) ---------------------
    # OPTIONAL, default-empty so ``from_env`` is unchanged and every existing
    # positional/keyword construction stays valid. The caller (primitives.py,
    # the ``_run_watchdog(...)`` site) sets these POST-construction via
    # ``dataclasses.replace(cfg, gpu_device_ids=..., heartbeat_json_path=...)``
    # once it knows the pinned cards + sidecar path for the run.
    #
    # When populated, the watchdog consults REAL compute before emitting a
    # kill verdict: a run that looks file-mtime/SSE stale but is still burning
    # GPU (>5% util on a pinned card) or advancing process-tree CPU time is
    # treated as alive. This is the signal that survives the dumb ``.heartbeat``
    # daemon being gone (an in-process hang sits at ~0% on both).
    #
    # ``gpu_device_ids`` — the pinned PHYSICAL GPU ids for this run (never the
    #   remapped ``cuda:0..N``; concurrent batch runs lease disjoint cards).
    # ``heartbeat_json_path`` — path to the Pillar-1 sidecar
    #   ``<code>/.exec_heartbeat.json`` (a JSON dict carrying ``"pid"``).
    gpu_device_ids: tuple[str, ...] = ()
    heartbeat_json_path: str | None = None

    @classmethod
    def from_env(cls) -> "WatchdogConfig":
        def _f(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if not raw:
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                logger.warning("watchdog: ignoring non-numeric %s=%r", name, raw)
                return default

        return cls(
            warn_after_seconds=_f(_WARN_ENV_VAR, 600.0),
            kill_after_seconds=_f(_KILL_ENV_VAR, 1500.0),
            poll_interval_seconds=_f(_POLL_ENV_VAR, 30.0),
        )


# ---------------------------------------------------------------------------
# Per-cycle report
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StalenessReport:
    """The result of one polling cycle.

    ``stale_seconds`` is the minimum staleness across all signals that have
    been observed at least once.  Signals whose files don't yet exist are
    excluded — they don't contribute "infinite" staleness.

    ``verdict``: ``"ok"`` / ``"warn"`` / ``"kill"``.
    """

    stale_seconds: Optional[float]
    freshest_signal: Optional[str]
    freshest_signal_age_seconds: Optional[float]
    verdict: str
    exec_log_age_seconds: Optional[float] = None
    heartbeat_age_seconds: Optional[float] = None
    sse_event_age_seconds: Optional[float] = None
    exec_log_path: Optional[str] = None
    heartbeat_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Signal collection (pure, fail-soft)
# ---------------------------------------------------------------------------


def _file_age_seconds(path: Path, now: float) -> Optional[float]:
    """Return ``now - path.stat().st_mtime`` or None if the file is unstattable.

    Fail-soft: missing file, permission error, or any OSError → None.
    """
    try:
        st = path.stat()
    except (OSError, FileNotFoundError):
        return None
    return max(0.0, now - st.st_mtime)


def _latest_meaningful_sse_event_age(
    sse_log_path: Path, now: float, max_lookback_lines: int = 200,
) -> Optional[float]:
    """Return age of the most recent NON-heartbeat dashboard event, or None.

    The dashboard_events.jsonl stream is written to constantly by ``heartbeat()``
    primitive calls and ``iteration_heartbeat`` events — both fire every 30 s
    regardless of whether real work is happening.  Counting them as activity
    makes the watchdog blind to silent hangs where the agent keeps heartbeating
    but produces zero forward progress (Adam v8 sat 25 min in implement_baseline
    while iteration_heartbeat kept the SSE log mtime fresh).

    Filtering them out — looking for the latest primitive_call other than
    ``heartbeat``, OR any terminal/state event — makes the SSE signal a true
    measure of forward progress.

    Fail-soft: file missing or unparseable lines return None.
    """
    if not sse_log_path.exists():
        return None
    try:
        with sse_log_path.open(encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return None
    if not lines:
        return None
    # Walk backwards to find the most recent non-heartbeat event.
    import json as _json
    for line in reversed(lines[-max_lookback_lines:]):
        try:
            d = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        ev_type = d.get("event") or d.get("type") or ""
        primitive = d.get("primitive") or ""
        # Skip pure liveness signals — they don't indicate forward progress.
        if ev_type == "iteration_heartbeat":
            continue
        if ev_type == "primitive_call" and primitive == "heartbeat":
            continue
        # 2026-05-25 circular-freshness fix (Lane S): the watchdog itself
        # writes `run_warning` events to this same JSONL on every stale
        # detection. Counting those as fresh signals creates a feedback
        # loop where the watchdog can never escalate to kill verdict —
        # Adam sat wedged for 100+ min with 9 stale_run warnings and 0
        # kills because every warning emission reset sse_event_age below
        # the kill threshold. Filter run_warning + worker_report_* so the
        # only signals that count are real forward progress.
        if ev_type in {"run_warning", "worker_report_started", "worker_report_failed"}:
            continue
        # Found a real event.  Parse its timestamp.
        ts_str = d.get("ts") or d.get("timestamp") or ""
        if not ts_str:
            continue
        try:
            from datetime import datetime as _dt
            dt = _dt.fromisoformat(ts_str.replace("Z", "+00:00"))
            return max(0.0, now - dt.timestamp())
        except (ValueError, AttributeError):
            continue
    return None


def collect_staleness(
    *,
    artifact_root: Path,
    project_dir: Path,
    config: WatchdogConfig,
    now: Optional[float] = None,
    active_primitive: Optional[str] = None,
) -> StalenessReport:
    """Snapshot the three liveness signals + classify.

    Pure function — no side effects, fail-soft on every I/O.  Caller decides
    what to do with the verdict.

    The SSE-event signal filters out pure-liveness events (iteration_heartbeat,
    primitive_call:heartbeat) so the watchdog can't be deceived by an agent
    that emits heartbeats while making zero forward progress.

    ``active_primitive`` — the name of the currently-active primitive (PR-ι.4).
    When provided, the kill threshold is looked up from
    :data:`PRIMITIVE_IDLE_BASELINE_S` to avoid falsely killing a legitimately
    slow ``run_experiment`` or ``implement_baseline`` call.
    """
    if now is None:
        now = time.time()

    exec_log_path = artifact_root / config.exec_log_filename
    heartbeat_path = artifact_root / config.heartbeat_filename
    sse_log_path = project_dir / config.dashboard_events_filename

    exec_age = _file_age_seconds(exec_log_path, now)
    hb_age = _file_age_seconds(heartbeat_path, now)
    sse_age = _latest_meaningful_sse_event_age(sse_log_path, now)

    # Min-staleness across signals that have been observed at least once.
    # Signals without a file yet are excluded (bootstrap-grace).
    observed: list[tuple[str, float]] = []
    if exec_age is not None:
        observed.append(("exec_log", exec_age))
    if hb_age is not None:
        observed.append(("heartbeat", hb_age))
    if sse_age is not None:
        observed.append(("sse_event", sse_age))

    if not observed:
        # Nothing observed yet — bootstrap grace.
        return StalenessReport(
            stale_seconds=None,
            freshest_signal=None,
            freshest_signal_age_seconds=None,
            verdict="ok",
            exec_log_age_seconds=exec_age,
            heartbeat_age_seconds=hb_age,
            sse_event_age_seconds=sse_age,
            exec_log_path=str(exec_log_path),
            heartbeat_path=str(heartbeat_path),
        )

    # Freshest signal = smallest age.
    freshest_signal, freshest_age = min(observed, key=lambda x: x[1])
    stale_seconds = freshest_age

    # PR-ι.4: use per-primitive kill threshold when a primitive is active.
    _kill_threshold = effective_idle_threshold(active_primitive, config)
    if stale_seconds >= _kill_threshold:
        verdict = "kill"
    elif stale_seconds >= config.warn_after_seconds:
        verdict = "warn"
    else:
        verdict = "ok"

    return StalenessReport(
        stale_seconds=stale_seconds,
        freshest_signal=freshest_signal,
        freshest_signal_age_seconds=freshest_age,
        verdict=verdict,
        exec_log_age_seconds=exec_age,
        heartbeat_age_seconds=hb_age,
        sse_event_age_seconds=sse_age,
        exec_log_path=str(exec_log_path),
        heartbeat_path=str(heartbeat_path),
    )


# ---------------------------------------------------------------------------
# Heartbeat-daemon shell injection
# ---------------------------------------------------------------------------


def heartbeat_daemon_command(artifact_dir_in_container: str = "/artifacts") -> str:
    """Return the shell command that should be prepended to bootstrap_commands.

    Backgrounds a tiny daemon inside the sandbox that writes a unix timestamp
    to ``<artifact_dir>/.heartbeat`` every 30 s.

    **Detachment contract** (this is the subtle part — the previous version
    used plain ``nohup ... &`` which blocked over SSH because the daemon
    inherited SSH's stdout/stderr file descriptors; the SSH channel kept
    the parent exec alive forever, hanging the entire for-loop of bootstrap
    commands):

      * ``( ... )`` — wrap in a subshell so the outer parent has nothing to
        wait on once the subshell forks.
      * ``setsid -f`` — fork into a NEW session detached from the controlling
        terminal.  Without this, SSH treats the daemon as a child process
        and waits for it.
      * ``< /dev/null > /dev/null 2>&1`` — close stdin AND redirect stdout
        + stderr.  SSH only closes the exec channel when ALL inherited FDs
        are closed by every descendant.  Plain ``> /dev/null`` only closes
        stdout/stderr; without ``< /dev/null`` the daemon still owns SSH's
        stdin and the channel stays open forever.
      * ``&`` — the final backgrounding.  After the subshell forks the
        detached daemon and exits, SSH sees an immediate clean exit.

    The daemon naturally dies when the pod is destroyed (no parent to
    cling to, no controlling terminal, no shared file descriptors).
    """
    return (
        f"( setsid -f bash -c "
        f"'while true; do date +%s > {artifact_dir_in_container}/.heartbeat; sleep 30; done' "
        f"< /dev/null > /dev/null 2>&1 & ) ; exit 0"
    )


# ---------------------------------------------------------------------------
# Async watchdog loop
# ---------------------------------------------------------------------------


def _detect_active_primitive(sse_log_path: Path, max_lookback_lines: int = 100) -> Optional[str]:
    """Return the name of the most recently started (but not yet completed) primitive.

    Scans the last ``max_lookback_lines`` lines of ``dashboard_events.jsonl``
    looking for a ``primitive_call`` event with ``phase="start"``.  Returns
    the primitive name, or ``None`` if not found or the file doesn't exist.

    Fail-soft: any I/O or parse error returns ``None``.
    """
    if not sse_log_path.exists():
        return None
    try:
        with sse_log_path.open(encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return None
    if not lines:
        return None

    import json as _json
    # Walk backwards to find the most recent primitive_call with phase=start.
    for line in reversed(lines[-max_lookback_lines:]):
        try:
            d = _json.loads(line.strip())
        except _json.JSONDecodeError:
            continue
        if d.get("event") != "primitive_call":
            continue
        if str(d.get("phase") or "").lower() == "start":
            primitive = d.get("primitive") or d.get("name") or ""
            if primitive:
                return str(primitive)
    return None


def _read_heartbeat_pid(heartbeat_json_path: Optional[str]) -> Optional[int]:
    """Return the ``pid`` from the Pillar-1 ``.exec_heartbeat.json`` sidecar.

    The sidecar is a JSON dict ``{last_output_at, last_line, lines, pid, command}``
    written by ``LocalProcessBackend.exec``. We only need ``pid`` to walk the
    process tree's CPU-time.

    Fail-soft: missing/unset path, missing file, unparseable JSON, or a
    non-integer ``pid`` all return ``None`` (no CPU signal this poll).
    """
    if not heartbeat_json_path:
        return None
    try:
        import json as _json

        with open(heartbeat_json_path, encoding="utf-8") as fh:
            data = _json.load(fh)
        pid = data.get("pid")
        if pid is None:
            return None
        return int(pid)
    except Exception:  # noqa: BLE001 — sidecar read must never crash the watchdog
        return None


def _compute_liveness(
    config: "WatchdogConfig", last_cpu_seconds: Optional[float],
) -> tuple[bool, Optional[float]]:
    """Consult REAL compute (GPU util + process-tree CPU delta). Fail-soft.

    Returns ``(is_computing, cpu_seconds_now)``:

      * ``is_computing`` is True when EITHER a pinned physical GPU is busy
        (>5% util, instantaneous) OR the process-tree CPU-time has INCREASED
        since the previous poll (``last_cpu_seconds``). On the first poll
        (``last_cpu_seconds is None``) a CPU reading establishes the baseline
        only — no "increase" can be claimed yet — so CPU alone never trips
        liveness on the very first sample.
      * ``cpu_seconds_now`` is the CPU reading to carry into the next poll
        (``None`` when unmeasurable), so the caller updates its running state.

    Lazy import of the ``local_process`` helpers avoids an import cycle
    (``primitives.py`` imports this module; ``local_process`` is a runtime
    backend). ANY exception in either compute check is swallowed and treated
    as "no compute signal" — the caller then falls back to file/SSE staleness.
    """
    try:
        from backend.services.runtime.local_process import (
            _gpu_busy_blocking,
            _proc_tree_cpu_seconds,
        )
    except Exception:  # noqa: BLE001 — import failure → no compute signal
        return (False, last_cpu_seconds)

    # 1. GPU busy on a pinned PHYSICAL card (instantaneous >5% util check).
    try:
        if config.gpu_device_ids and _gpu_busy_blocking(tuple(config.gpu_device_ids)):
            # Still take a CPU reading so the running baseline stays current.
            try:
                cpu_now = _proc_tree_cpu_seconds(_read_heartbeat_pid(config.heartbeat_json_path))
            except Exception:  # noqa: BLE001
                cpu_now = last_cpu_seconds
            return (True, cpu_now if cpu_now is not None else last_cpu_seconds)
    except Exception:  # noqa: BLE001 — GPU poll must never crash the watchdog
        pass

    # 2. CPU advancing: process-tree CPU-time delta since the previous poll.
    cpu_now: Optional[float] = last_cpu_seconds
    try:
        pid = _read_heartbeat_pid(config.heartbeat_json_path)
        reading = _proc_tree_cpu_seconds(pid)
        if reading is not None:
            cpu_now = reading
            if last_cpu_seconds is not None and reading > last_cpu_seconds:
                return (True, cpu_now)
    except Exception:  # noqa: BLE001 — CPU poll must never crash the watchdog
        cpu_now = last_cpu_seconds

    return (False, cpu_now)


WarnCallback = Callable[[StalenessReport], Awaitable[None]]
# Lane N: on_kill MAY return ``KillVerdict.RECOVERED`` to ask the watchdog
# to keep polling.  Returning ``None`` (or anything else) defaults to
# ``KillVerdict.DESTROY`` — v1 semantics preserved.
KillCallback = Callable[[StalenessReport], Awaitable[Optional[KillVerdict]]]


async def run_watchdog(
    *,
    artifact_root: Path,
    project_dir: Path,
    config: Optional[WatchdogConfig] = None,
    on_warn: Optional[WarnCallback] = None,
    on_kill: Optional[KillCallback] = None,
) -> None:
    """Poll signals until cancelled OR ``on_kill`` returns ``DESTROY``.

    Cancellation: the caller's ``finally`` block typically calls
    ``task.cancel()`` to stop the watchdog when the protected coroutine
    finishes.  This coroutine handles ``CancelledError`` cleanly.

    Disabled: returns immediately when ``OPENRESEARCH_WATCHDOG_DISABLED=true``.

    On-kill semantics (Lane N): the callback may decide to recover the
    sandbox instead of destroying it.  If it returns ``KillVerdict.RECOVERED``
    the watchdog continues polling — staleness will reset on the next
    exec.log write.  Returning ``DESTROY`` (or ``None``, or raising) makes
    the watchdog exit; the callback is expected to have torn down the
    pod (or set fail-soft state so run-level cleanup will).

    Fail-soft on warn-callback errors — if ``on_warn`` raises, log + carry
    on.  Kill-callback errors propagate (they indicate the operator's
    teardown logic is broken — we should NOT swallow that).
    """
    if not is_enabled():
        logger.info("watchdog: disabled via %s", _DISABLE_ENV_VAR)
        return

    cfg = config or WatchdogConfig.from_env()
    interval = cfg.poll_interval_seconds
    warn_emitted_at: float | None = None
    # Pillar 2b — running process-tree CPU-seconds baseline for the compute
    # delta check. None until the first measurable reading; carried across
    # polls so a kill verdict can be vetoed by genuine forward compute.
    last_cpu_seconds: float | None = None

    logger.info(
        "watchdog: armed (warn_after=%.0fs kill_after=%.0fs poll=%.0fs)",
        cfg.warn_after_seconds, cfg.kill_after_seconds, interval,
    )

    sse_log_path = project_dir / cfg.dashboard_events_filename

    try:
        while True:
            await asyncio.sleep(interval)
            try:
                # PR-ι.4: detect the currently-active primitive so collect_staleness
                # can apply a more lenient idle threshold for slow primitives
                # (run_experiment, implement_baseline) that legitimately take hours.
                _active = _detect_active_primitive(sse_log_path)
                report = collect_staleness(
                    artifact_root=artifact_root,
                    project_dir=project_dir,
                    config=cfg,
                    active_primitive=_active,
                )
            except Exception:  # noqa: BLE001 — collection MUST NOT crash the run
                logger.exception("watchdog: collect_staleness raised — skipping cycle")
                continue

            # Pillar 2b — consult REAL compute every poll so the running CPU
            # baseline stays current (a kill could otherwise be vetoed by a
            # stale baseline). Fully fail-soft inside _compute_liveness.
            is_computing, last_cpu_seconds = _compute_liveness(cfg, last_cpu_seconds)

            if report.verdict == "kill":
                # File-mtime/SSE staleness says "kill", but a quiet-but-computing
                # run (big matmul, kernel compile, CPU preprocess) is NOT wedged.
                # Treat it as alive and skip the kill; only a run that is silent
                # on ALL of {output, SSE, GPU, CPU} is genuinely hung.
                if is_computing:
                    logger.info(
                        "watchdog: KILL verdict suppressed (stale=%.0fs freshest=%s) — "
                        "real compute detected (gpu_busy or cpu advancing); run is alive",
                        report.stale_seconds or 0.0, report.freshest_signal or "?",
                    )
                    # Reset warn de-dup so the next genuinely-stale cycle reports cleanly.
                    warn_emitted_at = None
                    continue
                logger.warning(
                    "watchdog: KILL verdict (stale=%.0fs freshest=%s) — invoking on_kill",
                    report.stale_seconds or 0.0, report.freshest_signal or "?",
                )
                if on_kill is not None:
                    verdict = await on_kill(report)
                    if verdict == KillVerdict.RECOVERED:
                        logger.info(
                            "watchdog: on_kill returned RECOVERED — pod kept warm, "
                            "resuming poll loop",
                        )
                        # Reset warn de-dup so the next stale cycle reports cleanly.
                        warn_emitted_at = None
                        continue
                return

            if report.verdict == "warn":
                # De-dup: emit a warn every ``warn_after`` seconds at most,
                # so a long stall doesn't spam.
                now = time.time()
                if warn_emitted_at is None or (now - warn_emitted_at) >= cfg.warn_after_seconds:
                    warn_emitted_at = now
                    logger.warning(
                        "watchdog: WARN verdict (stale=%.0fs freshest=%s)",
                        report.stale_seconds or 0.0, report.freshest_signal or "?",
                    )
                    if on_warn is not None:
                        try:
                            await on_warn(report)
                        except Exception:  # noqa: BLE001 — observability MUST NOT block the run
                            logger.exception("watchdog: on_warn raised — continuing")

    except asyncio.CancelledError:
        logger.debug("watchdog: cancelled — exiting cleanly")
        raise


__all__ = [
    "KillVerdict",
    "PRIMITIVE_IDLE_BASELINE_S",
    "StalenessReport",
    "WatchdogConfig",
    "collect_staleness",
    "effective_idle_threshold",
    "heartbeat_daemon_command",
    "is_enabled",
    "run_watchdog",
]
