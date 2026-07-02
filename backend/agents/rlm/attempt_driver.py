"""attempt_driver.py — the AttemptDriver seam: LiveCliDriver (spec §7).

``AttemptDriver`` is the pluggable boundary between the campaign state
machine and however an attempt actually runs; ``AttemptRawResult`` is a
pointer set only (parsing/judgment belongs to ASSESS, U4). ``LiveCliDriver``
spawns ``backend.cli reproduce`` as a detached per-attempt child, force-
quarantining stale residue before every launch so the warm-retry heuristic is
structurally unreachable under a campaign (Codex F6), and asserts every
per-attempt env key it sets actually lands in the env sink (Codex F15).
"""

from __future__ import annotations

import abc
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.agents.rlm.run_spec_contract import run_spec_key_applies
from backend.services.runs.attempt_isolation import force_archive_incomplete

# Relative-to-run-dir seed marker path. Duplicated (NOT imported) from
# ``best_attempt.CAMPAIGN_SEED_MARKER`` — this module's import surface is
# deliberately limited to stdlib + attempt_isolation + run_spec_contract (see
# the campaign module map), so both modules independently agree on this
# literal string instead of one importing the other.
_SEED_MARKER_RELPATH = "campaign/seed_staging.json"

# Mirrors backend/services/events/live_runs.py::RunStatus's terminal subset
# (BUG-NEW-045) — duplicated here for the same import-boundary reason as
# _SEED_MARKER_RELPATH above; a future edit to that Literal should be
# reflected here too.
_TERMINAL_DEMO_STATUSES: frozenset[str] = frozenset(
    {"stopped", "completed", "failed", "killed", "interrupted"}
)

# Outer await_result() bound beyond the VM control-plane ceiling — a driver-
# level backstop in case the child wedges without ever reaching a terminal
# demo_status (spec §7 await_result).
_AWAIT_TIMEOUT_MARGIN_S = 3600.0


class DriverError(Exception):
    """The attempt driver cannot safely launch/await/abort an attempt."""


@dataclass(frozen=True)
class AttemptHandle:
    attempt_n: int
    project_id: str
    run_dir: str
    driver: str
    pid: int | None
    launched_at: float
    lease_ref: str | None
    # Absolute await-timeout deadline (epoch seconds), computed once at
    # launch() from the VM control-plane ceiling and persisted ON the handle
    # itself — NOT in driver instance memory. A resumed campaign process
    # constructs a FRESH LiveCliDriver with no memory of the original launch,
    # so the deadline must travel with the handle or a re-attach to a
    # healthy multi-hour run would compute a bogus deadline from the (by
    # then long-past) launched_at and abort it. None on a handle
    # reconstructed with no persisted deadline (see await_result).
    await_deadline: float | None = None


@dataclass(frozen=True)
class AttemptRawResult:
    run_dir: str
    report_path: str | None
    # "exited:<code>" | "signaled:<sig>" | "already_dead" | "completed" | "await_timeout"
    exit_condition: str


class AttemptDriver(abc.ABC):
    """Launch/await/abort one campaign attempt. Drivers stay dumb + swappable
    — all judgment lives in ASSESS, never here."""

    kind: str = "abstract"

    @abc.abstractmethod
    def launch(self, directives: Any) -> AttemptHandle: ...

    @abc.abstractmethod
    def await_result(self, handle: AttemptHandle) -> AttemptRawResult: ...

    @abc.abstractmethod
    def abort(self, handle: AttemptHandle, *, reason: str) -> None: ...


# ---------------------------------------------------------------------------
# Pure builders — unit-testable without spawning anything.
# ---------------------------------------------------------------------------


def build_reproduce_argv(directives: Any, *, python_exe: str) -> list[str]:
    """``[python_exe, -m backend.cli reproduce <paper_ref> --project-id <id>,
    (--run-spec <path>)?, *enforcement.cli_args, (--scope-spec <spec>)?]``.

    Exact order per spec §7: positional + --project-id first, then
    --run-spec (when set), then every enforcement ``cli_args`` pair verbatim,
    then --scope-spec (when set). ``directives`` is duck-typed — no import of
    ``campaign_directives`` (may not exist yet).
    """
    argv = [
        python_exe, "-m", "backend.cli", "reproduce", str(directives.paper_ref),
        "--project-id", str(directives.project_id),
    ]
    run_spec_path = getattr(directives, "run_spec_path", None)
    if run_spec_path:
        argv += ["--run-spec", str(run_spec_path)]
    enforcement: Mapping[str, Any] = dict(getattr(directives, "enforcement", None) or {})
    for flag, value in enforcement.get("cli_args", ()) or ():
        argv += [str(flag), str(value)]
    scope_spec = getattr(directives, "scope_spec", None)
    if scope_spec:
        argv += ["--scope-spec", str(scope_spec)]
    return argv


def build_attempt_env(directives: Any, base_env: Mapping[str, str]) -> dict[str, str]:
    """``base_env | enforcement["env"]`` plus the three per-attempt knobs.

    Every per-attempt key this function itself sets is asserted against
    :func:`run_spec_key_applies` before being written — defense against a
    renamed/typo'd knob silently doing nothing on the far side of the
    ``--run-spec`` env sink (Codex F15).
    """
    env: dict[str, str] = dict(base_env)
    enforcement: Mapping[str, Any] = dict(getattr(directives, "enforcement", None) or {})
    env.update({str(k): str(v) for k, v in (enforcement.get("env") or {}).items()})

    extra_guidance = getattr(directives, "extra_guidance", "") or ""
    per_attempt: dict[str, str] = {}
    if extra_guidance:
        per_attempt["OPENRESEARCH_BASELINE_EXTRA_GUIDANCE"] = str(extra_guidance)
    per_attempt["OPENRESEARCH_SEED_BEST_ATTEMPT"] = (
        "1" if getattr(directives, "seed_pointer", None) else "0"
    )
    per_attempt["OPENRESEARCH_TARGET_BEST_FLOOR"] = (
        "1" if getattr(directives, "target_floor", None) is not None else "0"
    )

    for key in per_attempt:
        if not run_spec_key_applies(key):
            raise DriverError(
                f"attempt_driver: per-attempt env key {key!r} would be "
                "silently ignored by run_spec_contract.run_spec_key_applies "
                "— refusing to launch with a knob that would silently do nothing"
            )

    env.update(per_attempt)
    return env


def _stage_seed_marker(run_dir: Path, directives: Any) -> None:
    """Write ``campaign/seed_staging.json`` whenever the campaign has
    EITHER a seed pointer OR a target floor for this attempt -- a
    fresh-lineage attempt (no seed_pointer) can still carry a campaign
    target_floor, and that floor must reach ``floored_target`` via the
    marker rather than falling through to the score-ranked ``attempts/``
    scan (Codex F5). Deleted only when BOTH are absent -- the genuine
    no-campaign-guidance-at-all case. Shared by every driver kind (spec
    §7): whichever driver's ``launch`` calls this, the marker lands under
    THIS CALL's ``run_dir`` (the attempt's own dir, width-child or not).
    """
    seed_pointer = getattr(directives, "seed_pointer", None)
    target_floor = getattr(directives, "target_floor", None)
    marker_path = run_dir / _SEED_MARKER_RELPATH
    if seed_pointer is None and target_floor is None:
        marker_path.unlink(missing_ok=True)
        return
    marker = {
        "attempt_n": directives.attempt_n,
        "source_code_dir": str(seed_pointer) if seed_pointer else None,
        "target_floor": target_floor,
        "lineage": getattr(directives, "seed_lineage", None),
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(marker), encoding="utf-8")
    tmp.replace(marker_path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a torn/absent status file reads as "not terminal yet"
        return None
    return data if isinstance(data, dict) else None


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# LiveCliDriver
# ---------------------------------------------------------------------------


class LiveCliDriver(AttemptDriver):
    """Spawns ``python -m backend.cli reproduce`` as a detached per-attempt
    child on the campaign's own project id.

    Force-quarantines stale residue before every launch (Codex F6) so the
    ``attempt_isolation`` warm-retry heuristic is never exercised under a
    campaign; stages/clears the campaign seed marker (Codex F5 delivery
    half); polls ``demo_status.json`` for a terminal state. Clock/sleep/popen
    are all injectable so tests never spawn a real child or sleep for real.
    """

    kind = "live"

    def __init__(
        self,
        *,
        runs_root: Path,
        repo_root: Path,
        python_exe: str | None = None,
        popen: Callable[..., subprocess.Popen] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_s: float = 5.0,
        abort_grace_s: float = 30.0,
    ) -> None:
        self._runs_root = Path(runs_root)
        self._repo_root = Path(repo_root)
        self._python_exe = python_exe or sys.executable
        self._popen = popen or subprocess.Popen
        self._clock = clock
        self._sleep = sleep
        self._poll_interval_s = poll_interval_s
        self._abort_grace_s = abort_grace_s

    # -- AttemptDriver -------------------------------------------------

    def launch(self, directives: Any) -> AttemptHandle:
        run_dir = self._runs_root / str(directives.project_id)
        (run_dir / "campaign").mkdir(parents=True, exist_ok=True)

        # Unconditional (Codex F6 review LOW-1): force_archive_incomplete
        # already no-ops safely via its own _has_attempt_residue, which
        # covers the FULL manifest (incl. cost_ledger/user_messages/
        # rlm_state sidecars) -- a narrower pre-check here could miss
        # residue the full manifest would catch (e.g. a lone
        # rlm_state/gpu_escalation_state.json with no code/ and none of the
        # three top-level markers).
        force_archive_incomplete(
            str(directives.project_id), self._runs_root,
            reason="campaign_pre_launch",
        )

        _stage_seed_marker(run_dir, directives)

        argv = build_reproduce_argv(directives, python_exe=self._python_exe)
        env = build_attempt_env(directives, os.environ)

        log_path = run_dir / "campaign" / f"attempt_{directives.attempt_n}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as logf:
            proc = self._popen(
                argv, env=env, cwd=str(self._repo_root),
                stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        launched_at = self._clock()
        enforcement: Mapping[str, Any] = dict(getattr(directives, "enforcement", None) or {})
        ceiling = enforcement.get("vm_ceiling_s") or 0.0
        run_dir_key = str(run_dir)

        return AttemptHandle(
            attempt_n=directives.attempt_n,
            project_id=str(directives.project_id),
            run_dir=run_dir_key,
            driver=self.kind,
            pid=getattr(proc, "pid", None),
            launched_at=launched_at,
            lease_ref=None,
            await_deadline=launched_at + float(ceiling) + _AWAIT_TIMEOUT_MARGIN_S,
        )

    def await_result(self, handle: AttemptHandle) -> AttemptRawResult:
        run_dir = Path(handle.run_dir)
        status_path = run_dir / "demo_status.json"
        report_path = run_dir / "final_report.json"
        # A deadline-less reconstructed handle (e.g. an older serialization)
        # gets a full margin measured from NOW — never from the possibly
        # long-past launched_at (see the AttemptHandle.await_deadline note).
        deadline = (
            handle.await_deadline
            if handle.await_deadline is not None
            else self._clock() + _AWAIT_TIMEOUT_MARGIN_S
        )

        while True:
            status = _read_json_dict(status_path)
            if status is not None and status.get("status") in _TERMINAL_DEMO_STATUSES:
                return self._result(handle, report_path, "completed")
            if handle.pid is not None and not _pid_alive(handle.pid):
                return self._result(handle, report_path, "already_dead")
            if self._clock() >= deadline:
                self.abort(handle, reason="await_timeout")
                return self._result(handle, report_path, "await_timeout")
            self._sleep(self._poll_interval_s)

    def abort(self, handle: AttemptHandle, *, reason: str) -> None:
        if handle.pid is not None:
            self._kill_group(handle.pid)
        self._patch_demo_status_killed(Path(handle.run_dir), reason)

    # -- internals -------------------------------------------------------

    @staticmethod
    def _result(handle: AttemptHandle, report_path: Path, exit_condition: str) -> AttemptRawResult:
        return AttemptRawResult(
            run_dir=handle.run_dir,
            report_path=str(report_path) if report_path.exists() else None,
            exit_condition=exit_condition,
        )

    def _kill_group(self, pid: int) -> None:
        """SIGTERM the process group, escalate to SIGKILL after abort_grace_s.

        Fail-soft: a group that has already exited raises
        ``ProcessLookupError``/``PermissionError`` on either signal — treated
        as "nothing left to signal", never propagated.
        """
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = self._clock() + self._abort_grace_s
        while _pid_alive(pid) and self._clock() < deadline:
            self._sleep(1.0)
        if _pid_alive(pid):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def _patch_demo_status_killed(self, run_dir: Path, reason: str) -> None:
        """Atomically flip demo_status.json to status=killed (mirrors
        cli.py::_mark_demo_status_killed / the kill_and_restart.sh
        BUG-NEW-041 workaround). Terminal: a status already in
        ``_TERMINAL_DEMO_STATUSES`` is never overwritten. Best-effort."""
        path = run_dir / "demo_status.json"
        try:
            existing = _read_json_dict(path) or {}
            if existing.get("status") in _TERMINAL_DEMO_STATUSES:
                return
            now_iso = datetime.now(timezone.utc).isoformat()
            merged = {
                **existing,
                "status": "killed",
                "killReason": reason,
                "error": reason,
                "updatedAt": now_iso,
                "completedAt": now_iso,
            }
            _atomic_write_json(path, merged)
        except Exception:  # noqa: BLE001 — best-effort status bookkeeping
            return


# ---------------------------------------------------------------------------
# UnifiedRunDriver
# ---------------------------------------------------------------------------


class UnifiedRunDriver(AttemptDriver):
    """``build_reproduction_run(...).run()`` under the opt-in unified
    composition-root path (spec §7). Same clean-context contract as
    ``LiveCliDriver`` -- force-quarantine + seed-marker staging, both via
    the shared module-level helpers above -- but the "child" here is an
    in-process ``ReproductionRun`` object, not a subprocess.

    ``.run()`` executes SYNCHRONOUSLY, so ``launch()`` only builds and
    stashes the run object (keyed by run_dir); ``await_result`` -- called
    right after, in the SAME process, by the campaign loop's
    PLAN -> LAUNCH -> AWAIT sequencing -- pops the stash and calls
    ``.run()``. A resumed campaign process constructs a FRESH driver with
    an empty stash, so ``await_result`` on an old handle correctly falls
    through to assess-from-disk semantics (spec §5/F7) instead of
    re-running anything.

    Module import surface stays stdlib + the two existing imports at every
    OTHER call site in this file; the unified_run/resilience/runtime
    imports below are local to ``launch`` so importing this module alone
    never pulls in the heavier compute-provider dependency chain.
    """

    kind = "unified"

    def __init__(
        self,
        *,
        runs_root: Path,
        build_run: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._runs_root = Path(runs_root)
        self._build_run = build_run
        self._clock = clock
        # Keyed by run_dir (str) -- the SAME key that travels on the handle
        # through the campaign loop's launch -> await_result round trip.
        self._pending: dict[str, Any] = {}

    def launch(self, directives: Any) -> AttemptHandle:
        run_dir = self._runs_root / str(directives.project_id)
        (run_dir / "campaign").mkdir(parents=True, exist_ok=True)

        force_archive_incomplete(
            str(directives.project_id), self._runs_root,
            reason="campaign_pre_launch",
        )
        _stage_seed_marker(run_dir, directives)

        from backend.agents.resilience.budget import RunBudget
        from backend.agents.rlm.unified_run import build_reproduction_run
        from backend.services.runtime.cloud_profile import VmSpec

        envelope = directives.envelope
        enforcement: Mapping[str, Any] = dict(getattr(directives, "enforcement", None) or {})
        wall_s = enforcement.get("effective_wall_s")
        vm_ceiling_s = enforcement.get("vm_ceiling_s")

        budget = RunBudget(
            max_usd=envelope.llm_usd,
            max_wall_clock_seconds=float(wall_s) if wall_s is not None else None,
            max_run_gpu_usd=envelope.gpu_usd,
            max_gpu_hours=envelope.gpu_hours,
        )
        # EnforcementPlan.vm_ceiling_s (post co-tightening) — NEVER
        # envelope.vm_ceiling_s — mirrors campaign_composition's own
        # _enforcement_mapping docstring precedent.
        vm_spec = VmSpec(max_run_duration_s=int(vm_ceiling_s) if vm_ceiling_s is not None else None)

        build_run = self._build_run or build_reproduction_run
        run_obj = build_run(
            paper_id=directives.paper_ref,
            state_dir=run_dir,
            budget=budget,
            vm_spec=vm_spec,
        )

        launched_at = self._clock()
        run_dir_key = str(run_dir)
        self._pending[run_dir_key] = run_obj

        return AttemptHandle(
            attempt_n=directives.attempt_n,
            project_id=str(directives.project_id),
            run_dir=run_dir_key,
            driver=self.kind,
            pid=None,
            launched_at=launched_at,
            lease_ref=None,
        )

    def await_result(self, handle: AttemptHandle) -> AttemptRawResult:
        run_obj = self._pending.pop(handle.run_dir, None)
        if run_obj is None:
            # Resumed handle: a fresh process's driver has no memory of the
            # original launch. ReproductionRun has no re-enter API (spec
            # §5/F7) -- assess-from-disk, never re-run.
            report_path = Path(handle.run_dir) / "final_report.json"
            return AttemptRawResult(
                run_dir=handle.run_dir,
                report_path=str(report_path) if report_path.exists() else None,
                exit_condition="already_dead",
            )

        try:
            outcome = run_obj.run()
        except Exception as exc:  # noqa: BLE001 — defensive: the real ReproductionRun.run() never raises (it
            # catches everything internally), but an injected/test build_run's object might.
            raise DriverError(f"unified_run_driver: run() raised: {exc}") from exc

        report = getattr(outcome, "report", None)
        report_path = getattr(report, "report_path", None) if report is not None else None
        return AttemptRawResult(
            run_dir=handle.run_dir,
            report_path=report_path,
            exit_condition="completed",
        )

    def abort(self, handle: AttemptHandle, *, reason: str) -> None:
        # Best-effort only: ReproductionRun has no public abort/teardown
        # today (spec §5/F7) -- the VM-side max_run_duration_s ceiling is
        # the real backstop. Still checked generically so a future/injected
        # run object exposing either is honoured.
        run_obj = self._pending.pop(handle.run_dir, None)
        if run_obj is None:
            return
        for method_name in ("abort", "teardown"):
            method = getattr(run_obj, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:  # noqa: BLE001 — best-effort abort
                    pass
                return


# ---------------------------------------------------------------------------
# PairedDriver
# ---------------------------------------------------------------------------


class PairedDriver(AttemptDriver):
    """Alternates between a live and a unified delegate by attempt_n parity
    (odd -> live, even -> unified; deterministic). ``handle.driver`` on any
    returned handle is the DELEGATE's own kind ("live"/"unified"), never
    "paired" -- ``launch``/``await_result``/``abort`` forward verbatim.

    Conducts the Phase-1f A/B (spec §7/§20 F8) -- it does not declare it
    satisfied. Paired campaigns spend GPU on both drivers, so construction
    demands an explicit ``operator_ack`` keyword (no default: omitting it
    is a ``TypeError`` at the call site) and additionally refuses a falsy
    value with ``DriverError`` -- no default flip can ever construct a
    working ``PairedDriver``; only an explicit CLI ``--driver paired``
    selection may.
    """

    kind = "paired"

    def __init__(self, *, live: AttemptDriver, unified: AttemptDriver, operator_ack: bool) -> None:
        if not operator_ack:
            raise DriverError(
                "PairedDriver requires an explicit operator_ack=True — paired campaigns "
                "spend GPU on both drivers and require explicit operator initiation "
                "(spec §7/§20 F8); no default flip constructs this driver."
            )
        self._live = live
        self._unified = unified

    def _delegate(self, attempt_n: int) -> AttemptDriver:
        return self._live if attempt_n % 2 == 1 else self._unified

    def launch(self, directives: Any) -> AttemptHandle:
        return self._delegate(directives.attempt_n).launch(directives)

    def await_result(self, handle: AttemptHandle) -> AttemptRawResult:
        return self._delegate(handle.attempt_n).await_result(handle)

    def abort(self, handle: AttemptHandle, *, reason: str) -> None:
        self._delegate(handle.attempt_n).abort(handle, reason=reason)


__all__ = [
    "AttemptDriver",
    "AttemptHandle",
    "AttemptRawResult",
    "DriverError",
    "LiveCliDriver",
    "PairedDriver",
    "UnifiedRunDriver",
    "build_attempt_env",
    "build_reproduce_argv",
]
