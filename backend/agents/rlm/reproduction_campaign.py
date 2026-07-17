"""Deterministic outer state machine + fail-CLOSED spend ledger for the
repeat-until-reproduced campaign (spec §5; Codex resolutions F1/F7, spec §20).

``CampaignLedger`` durably records intent BEFORE money moves: atomic+fsync
writes, a write-ahead intent row precedes every launch, and any I/O failure
halts the campaign rather than risk a double-spend or a silent skip. No real
stage logic lives here -- every stage is an injected ``CampaignStages``
callable; later units supply real implementations.

ONE exception to "no stage logic here": the doomed-run early kill (spec
§10.3, ``OPENRESEARCH_DOOMED_KILL``, default OFF). Cutting losses on a
diverging attempt is a MONEY decision about an attempt that is mid-flight,
so it belongs to whoever owns the AWAIT stage and the in-flight handle --
this machine. It is deliberately thin: all judgement lives in the pure
``doomed_run_comparator`` module (evidence reader + comparator + watch), and
this file only owns the impure half -- the poll thread, the process-group
kill, and the honest ledger/SSE record. When the flag is off, none of it is
constructed and every code path below is byte-identical to before.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.agents.rlm import doomed_run_comparator as doomed


class CampaignLedgerError(Exception):
    """A durable ledger write/read failed; the campaign halts (spec F1)."""


class CampaignInitError(Exception):
    """``validate_init`` refused the campaign before any money moved."""


def _check_keys(data: Any, allowed: set, label: str) -> None:
    if not isinstance(data, Mapping):
        raise CampaignLedgerError(f"{label}.from_dict: expected a mapping, got {type(data).__name__}")
    present = set(data)
    unknown = present - allowed
    if unknown:
        raise CampaignLedgerError(f"{label}.from_dict: unknown keys {sorted(unknown)}")
    missing = allowed - present
    if missing:
        raise CampaignLedgerError(f"{label}.from_dict: missing keys {sorted(missing)}")


@dataclass(frozen=True)
class InFlight:
    attempt_n: int
    driver: str
    run_dir: str
    pid: int | None
    lease_ref: str | None
    launched_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_n": self.attempt_n,
            "driver": self.driver,
            "run_dir": self.run_dir,
            "pid": self.pid,
            "lease_ref": self.lease_ref,
            "launched_at": self.launched_at,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> InFlight:
        _check_keys(data, {"attempt_n", "driver", "run_dir", "pid", "lease_ref", "launched_at"}, "InFlight")
        return InFlight(
            attempt_n=int(data["attempt_n"]),
            driver=str(data["driver"]),
            run_dir=str(data["run_dir"]),
            pid=int(data["pid"]) if data["pid"] is not None else None,
            lease_ref=data["lease_ref"],
            launched_at=float(data["launched_at"]),
        )


_CAMPAIGN_STATE_KEYS = {
    "project_id", "paper_ref", "state", "next_attempt_n", "mode", "driver",
    "budget", "spent", "scope_rung", "in_flight", "understanding_sha256",
    "rubric_sha256", "steering_cursor", "pending_approval", "warnings",
    "terminal", "created_at", "updated_at",
}


@dataclass
class CampaignState:
    project_id: str
    paper_ref: str
    state: str  # "init"|"understand"|"attempt_loop"|"paused"|"terminal"
    next_attempt_n: int
    mode: str  # "unattended"|"checkpoint"
    driver: str
    budget: dict
    spent: dict
    scope_rung: int
    in_flight: InFlight | None
    understanding_sha256: str | None
    rubric_sha256: str | None
    steering_cursor: int
    pending_approval: dict | None
    warnings: list
    terminal: dict | None
    created_at: float
    updated_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "paper_ref": self.paper_ref,
            "state": self.state,
            "next_attempt_n": self.next_attempt_n,
            "mode": self.mode,
            "driver": self.driver,
            "budget": dict(self.budget),
            "spent": dict(self.spent),
            "scope_rung": self.scope_rung,
            "in_flight": self.in_flight.to_dict() if self.in_flight is not None else None,
            "understanding_sha256": self.understanding_sha256,
            "rubric_sha256": self.rubric_sha256,
            "steering_cursor": self.steering_cursor,
            "pending_approval": dict(self.pending_approval) if self.pending_approval is not None else None,
            "warnings": list(self.warnings),
            "terminal": dict(self.terminal) if self.terminal is not None else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> CampaignState:
        _check_keys(data, set(_CAMPAIGN_STATE_KEYS), "CampaignState")
        in_flight_raw = data["in_flight"]
        return CampaignState(
            project_id=str(data["project_id"]),
            paper_ref=str(data["paper_ref"]),
            state=str(data["state"]),
            next_attempt_n=int(data["next_attempt_n"]),
            mode=str(data["mode"]),
            driver=str(data["driver"]),
            budget=dict(data["budget"]),
            spent=dict(data["spent"]),
            scope_rung=int(data["scope_rung"]),
            in_flight=InFlight.from_dict(in_flight_raw) if in_flight_raw is not None else None,
            understanding_sha256=data["understanding_sha256"],
            rubric_sha256=data["rubric_sha256"],
            steering_cursor=int(data["steering_cursor"]),
            pending_approval=dict(data["pending_approval"]) if data["pending_approval"] is not None else None,
            warnings=list(data["warnings"]),
            terminal=dict(data["terminal"]) if data["terminal"] is not None else None,
            created_at=float(data["created_at"]),
            updated_at=float(data["updated_at"]),
        )


def _zero_spend() -> dict:
    return {"llm_usd": 0.0, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 0.0}


class CampaignLedger:
    """Fail-closed spend ledger: ``campaign.json`` (atomic snapshot) +
    ``attempts.jsonl`` (append-only, newline-guarded). ANY ``OSError`` on any
    write or read (other than a tolerated corrupt final JSONL line) becomes a
    ``CampaignLedgerError`` -- the caller halts rather than launch on doubt.
    """

    def __init__(self, campaign_dir: Path) -> None:
        self.campaign_dir = Path(campaign_dir)
        self._state_path = self.campaign_dir / "campaign.json"
        self._attempts_path = self.campaign_dir / "attempts.jsonl"

    def load_state(self) -> CampaignState | None:
        if not self._state_path.exists():
            return None
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise CampaignLedgerError(f"cannot read campaign.json: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise CampaignLedgerError(f"campaign.json is corrupt: {exc}") from exc
        return CampaignState.from_dict(raw)

    def write_state(self, state: CampaignState) -> None:
        payload = json.dumps(state.to_dict(), sort_keys=True, indent=2, default=str)
        tmp_name: str | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(dir=str(self.campaign_dir), prefix=".campaign_json_", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, str(self._state_path))
            tmp_name = None
            dir_fd = os.open(str(self.campaign_dir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError as exc:
            raise CampaignLedgerError(f"cannot write campaign.json: {exc}") from exc
        finally:
            if tmp_name is not None and os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def append_row(self, row: Mapping[str, Any]) -> None:
        try:
            if self._attempts_path.exists() and self._attempts_path.stat().st_size > 0:
                with self._attempts_path.open("r+b") as fh:
                    data = fh.read()
                    if not data.endswith(b"\n"):
                        # A crash mid-append leaves a torn tail: a partial,
                        # never-fsynced-complete final line with no trailing
                        # "\n". Repairing it (instead of newline-PREFIXING the
                        # next row, which would entomb the fragment as a
                        # corrupt MID-FILE line that read_rows halts on
                        # forever) truncates back to just after the last
                        # complete ("\n"-terminated) row. This can only ever
                        # discard bytes written after the last successfully
                        # fsync'd row -- never a complete, already-fsynced one.
                        cut = data.rfind(b"\n") + 1  # 0 if no "\n" in file at all
                        fh.truncate(cut)
                        fh.flush()
                        os.fsync(fh.fileno())
            with self._attempts_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(row), sort_keys=True, default=str))
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise CampaignLedgerError(f"cannot append to attempts.jsonl: {exc}") from exc

    def read_rows(self) -> list[dict]:
        if not self._attempts_path.exists():
            return []
        try:
            text = self._attempts_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CampaignLedgerError(f"cannot read attempts.jsonl: {exc}") from exc
        lines = [ln for ln in text.split("\n") if ln.strip()]
        rows: list[dict] = []
        for idx, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if idx == len(lines) - 1:
                    break
                raise CampaignLedgerError(f"attempts.jsonl corrupt at line {idx + 1}: {exc}") from exc
        return rows

    @staticmethod
    def latest_by_status(rows: Sequence[Mapping[str, Any]], attempt_n: int) -> dict[str, dict]:
        latest: dict[str, dict] = {}
        for row in rows:
            if row.get("attempt_n") == attempt_n and isinstance(row.get("status"), str):
                latest[row["status"]] = dict(row)
        return latest


@dataclass(frozen=True)
class CampaignStages:
    validate_init: Callable[[], list]
    understand: Callable[[], dict]
    plan_attempt: Callable[[CampaignState, list], dict]
    launch: Callable[[dict], dict]
    await_result: Callable[[dict], dict]
    # ``planned`` (2nd arg) is dual-shaped: plan_attempt's dict on the live
    # path, or the durable "launched" ledger row on a resume reattach --
    # both carry attempt_n/directives_sha256/envelope; launch_payload only
    # exists on the live path.
    assess: Callable[[dict, dict], dict]
    assess_from_disk: Callable[[InFlight], dict]
    quarantine: Callable[[InFlight], None]
    distill: Callable[[dict], None]
    decide: Callable[[CampaignState, list], dict]
    write_reports: Callable[[CampaignState, list, dict], None]
    liveness_probe: Callable[[InFlight], bool]
    emit_event: Callable[[str, dict], None]
    # Optional steering poll (U9): may mutate state.mode/state.steering_cursor;
    # None (default) keeps every existing test/construction working unchanged.
    poll_steering: Callable[[CampaignState], None] | None = None


_PID_UNKNOWN_GRACE_S = 900.0  # bounded pid=None liveness-ambiguity grace window (controller finding I1)

# --- doomed-run early kill (spec §10.3, OPENRESEARCH_DOOMED_KILL) -----------
#
# Cadence knobs are deliberately NOT env-exposed: the operator-facing contract
# is the three documented ones the comparator already owns (DOOMED_MARGIN /
# DOOMED_POLLS / DOOMED_MIN_PROGRESS). The interval below only sets how often
# we LOOK; it cannot change WHETHER a kill is justified, because an observation
# only counts when the measured curve has advanced to a new step.
_DOOMED_POLL_INTERVAL_S = 30.0
#: How long the AWAIT stage waits for the watch thread to wind down once the
#: attempt has resolved (it is a daemon thread; this is politeness, not safety).
_DOOMED_JOIN_TIMEOUT_S = 10.0
#: SIGTERM -> grace -> SIGKILL, mirroring ``LiveCliDriver._kill_group``.
_DOOMED_TERM_GRACE_S = 20.0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _terminate_process_group(pid: int, *, grace_s: float = _DOOMED_TERM_GRACE_S) -> bool:
    """SIGTERM the attempt's process group, escalating to SIGKILL after
    ``grace_s``. Returns True IFF we actually signalled a LIVE process.

    That return value is load-bearing, not cosmetic: it is what keeps a run
    that finished on its own inside the last poll window from being
    mislabelled a kill. If the child is already gone we signalled nothing, we
    saved nothing, and its real result stands untouched.

    The child was spawned ``start_new_session=True`` (``LiveCliDriver``), so
    signalling the GROUP is what actually stops the GPU-burning grandchildren
    -- SIGTERM to the pid alone would leave the sandbox/executor subprocesses
    running, which is the whole failure this feature exists to prevent.
    """
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return False

    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.5)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    return True


def default_liveness_probe(in_flight: InFlight) -> bool:
    """Alive iff a live pid (host-scoped), ``demo_status.json`` says
    running/queued, or -- when the pid is not yet known and no parseable
    status exists -- we are still inside the bounded post-write-ahead grace
    window. A crash between the write-ahead ``pid=None`` intent write and
    the post-launch pid-stamped write must not read as instantly dead (the
    freshly-spawned child may not have written ``demo_status.json`` yet);
    after the grace window a still-unknown pid reads dead as before
    (controller finding I1).
    """
    status: dict[str, Any] | None = None
    status_path = Path(in_flight.run_dir) / "demo_status.json"
    if status_path.exists():
        try:
            parsed = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                status = parsed
        except (OSError, ValueError):
            status = None

    if in_flight.pid is not None:
        pid_host = str(status.get("pidHost") or "").strip() if status is not None else ""
        if not pid_host or pid_host == socket.gethostname():
            try:
                os.kill(in_flight.pid, 0)
                return True
            except PermissionError:
                return True
            except OSError:
                pass

    if status is not None:
        # A parseable demo_status always decides directly -- active statuses
        # alive, terminal statuses dead -- regardless of the grace window
        # below (a completed run must resolve immediately, not linger alive
        # for the rest of the grace period).
        return bool(status.get("status") in ("running", "queued"))

    if in_flight.pid is None and (time.time() - in_flight.launched_at) < _PID_UNKNOWN_GRACE_S:
        return True

    return False


class ReproductionCampaign:
    def __init__(
        self,
        *,
        run_dir: Path,
        project_id: str,
        paper_ref: str,
        budget: Mapping[str, Any],
        mode: str,
        driver: str,
        stages: CampaignStages,
        resume: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.project_id = project_id
        self.paper_ref = paper_ref
        self.budget = dict(budget)
        self.mode = mode
        self.driver = driver
        self.stages = stages
        self.resume = resume
        self.campaign_dir = self.run_dir / "campaign"
        self.ledger = CampaignLedger(self.campaign_dir)
        self._state: CampaignState | None = None
        # attempt_n -> the kill evidence packet, set by the watch thread and
        # consumed by the AWAIT caller once ``await_result`` has returned.
        # Empty forever when OPENRESEARCH_DOOMED_KILL is off.
        self._doomed_kills: dict[int, dict] = {}

    def _new_state(self, *, state: str) -> CampaignState:
        now = time.time()
        return CampaignState(
            project_id=self.project_id, paper_ref=self.paper_ref, state=state,
            next_attempt_n=1, mode=self.mode, driver=self.driver,
            budget=dict(self.budget), spent=_zero_spend(), scope_rung=0,
            in_flight=None, understanding_sha256=None, rubric_sha256=None,
            steering_cursor=0, pending_approval=None, warnings=[], terminal=None,
            created_at=now, updated_at=now,
        )

    def _safe_emit(self, state: CampaignState, event: str, payload: dict) -> None:
        # SSE emission is observability, never money/trust: a broken emitter
        # must not fail the campaign -- record a warning and continue.
        try:
            self.stages.emit_event(event, payload)
        except Exception as exc:  # noqa: BLE001 -- fail-soft emit (spec §12)
            state.warnings.append(f"emit_failed:{event}:{type(exc).__name__}")

    @staticmethod
    def _directives_sha256_for_attempt(rows: list, attempt_n: int) -> str | None:
        launched = CampaignLedger.latest_by_status(rows, attempt_n).get("launched")
        return launched.get("directives_sha256") if isinstance(launched, dict) else None

    # --- doomed-run early kill (spec §10.3) --------------------------------
    #
    # The AWAIT stage is the ONLY place in the campaign where money is being
    # spent and nobody is looking. `stages.await_result` blocks until the
    # child reaches a terminal state -- for a diverging SDAR-class attempt
    # that means burning the FULL wall-clock/GPU envelope before anyone
    # notices. These five methods are the "stop paying" path.

    def _await_with_doomed_watch(self, state: CampaignState, attempt_n: int, handle: Mapping[str, Any]) -> dict:
        """AWAIT, optionally under a doomed-run watch.

        Flag OFF (the default) -> a bare ``stages.await_result(handle)``: no
        thread, no disk reads, no ledger/SSE additions. Byte-identical.
        """
        if not doomed.enabled():
            return self.stages.await_result(dict(handle))

        watch, baseline_n = self._build_doomed_watch(state, attempt_n, handle)
        if watch is None:
            return self.stages.await_result(dict(handle))

        stop = threading.Event()
        thread = threading.Thread(
            target=self._doomed_watch_loop,
            args=(state, attempt_n, handle, watch, baseline_n, stop),
            name=f"doomed-watch-{attempt_n}",
            daemon=True,
        )
        thread.start()
        try:
            return self.stages.await_result(dict(handle))
        finally:
            stop.set()
            thread.join(timeout=_DOOMED_JOIN_TIMEOUT_S)

    def _build_doomed_watch(
        self, state: CampaignState, attempt_n: int, handle: Mapping[str, Any]
    ) -> tuple[Any, int | None]:
        """``(DoomedWatch, baseline_attempt_n)`` or ``(None, None)`` when a
        kill is not possible OR not safe. Every ``None`` return below is a
        deliberate refusal to kill, not a failure:

        * **No pid.** ``UnifiedRunDriver`` runs the attempt IN-PROCESS, so
          there is no process group to signal -- we could not actually stop
          paying, and a guard that reports a kill it did not perform is worse
          than no guard.
        * **No baseline.** Attempt 1, or a campaign whose only prior attempts
          are hard-quarantined/doomed-killed/curve-less. Structurally
          unkillable, by design.
        * **Baseline == live code dir.** Defensive: never compare a run with
          itself (would be a silent no-op, but an honest guard should not be
          able to reach that state at all).
        """
        try:
            pid = handle.get("pid")
            if not isinstance(pid, int):
                return None, None

            live_code = self._live_code_dir(handle)
            rows = self.ledger.read_rows()
            selected = doomed.select_baseline(
                rows,
                exclude_attempt_n=attempt_n,
                resolve_code_dir=lambda n: self._attempt_code_dir(rows, n),
            )
            if selected is None:
                return None, None

            baseline_n, baseline_curves = selected
            baseline_dir = self._attempt_code_dir(rows, baseline_n)
            if baseline_dir and Path(baseline_dir).resolve() == live_code.resolve():
                return None, None

            return doomed.watch_from_env(baseline_curves), baseline_n
        except Exception as exc:  # noqa: BLE001 -- a watchdog that cannot be built never breaks a campaign
            state.warnings.append(f"doomed_watch_unavailable:{type(exc).__name__}")
            return None, None

    def _live_code_dir(self, handle: Mapping[str, Any]) -> Path:
        return Path(str(handle.get("run_dir") or self.run_dir)) / "code"

    def _attempt_code_dir(self, rows: Sequence[Mapping[str, Any]], attempt_n: int) -> str | None:
        """An attempt's ``code/`` tree -- live while it is live, archived
        afterwards -- via the driver-maintained attempt-code index.

        Imported lazily so that with the flag off this module's import graph
        is completely unchanged. Resolution is delegated (never re-derived):
        the index format is ``attempt_driver``'s, and a second implementation
        of it here is exactly how the two would silently drift apart.
        """
        from backend.agents.rlm.attempt_driver import resolve_attempt_code_dir

        launched = CampaignLedger.latest_by_status(list(rows), attempt_n).get("launched") or {}
        run_dir = Path(str(launched.get("run_dir") or self.run_dir))
        return resolve_attempt_code_dir(run_dir, attempt_n)

    def _doomed_watch_loop(
        self,
        state: CampaignState,
        attempt_n: int,
        handle: Mapping[str, Any],
        watch: Any,
        baseline_n: int | None,
        stop: threading.Event,
    ) -> None:
        live_code = self._live_code_dir(handle)
        # ``stop.wait`` (not ``sleep``) so a natural completion unwinds the
        # thread instantly, and so the FIRST sample is taken one interval in
        # -- never at t=0, when nothing has been written yet.
        while not stop.wait(_DOOMED_POLL_INTERVAL_S):
            try:
                verdict = watch.observe(doomed.read_cell_curves(live_code))
            except Exception as exc:  # noqa: BLE001 -- observation failure => stop watching, never kill
                state.warnings.append(f"doomed_watch_failed:{type(exc).__name__}")
                return
            if verdict is None:
                continue
            self._fire_doomed_kill(state, attempt_n, handle, verdict, baseline_n)
            return

    def _fire_doomed_kill(
        self,
        state: CampaignState,
        attempt_n: int,
        handle: Mapping[str, Any],
        verdict: Any,
        baseline_n: int | None,
    ) -> None:
        pid = handle.get("pid")
        if not isinstance(pid, int):
            return

        evidence = {
            "attempt_n": attempt_n,
            "baseline_attempt_n": baseline_n,
            "killed_at": time.time(),
            **verdict.to_dict(),
        }
        # RECORD BEFORE SIGNALLING. A SIGTERM the child handles GRACEFULLY (the
        # CLI's own handler flips demo_status to "killed" and then finalizes)
        # lands on the status file long before the process actually exits --
        # so AWAIT can return, and the caller can reach _apply_doomed_kill,
        # while this thread is still inside the SIGTERM->SIGKILL grace loop
        # below. A kill we PERFORMED but failed to RECORD would be the one
        # dishonest outcome this whole feature exists to prevent (a run silently
        # stopped, reported as a science failure), so the record goes first and
        # is retracted only if it turns out we stopped nothing.
        self._doomed_kills[attempt_n] = evidence

        if not _terminate_process_group(pid):
            # Already gone: the attempt finished on its own inside this poll
            # window. We stopped nothing, so we claim nothing -- its real
            # result stands and is assessed exactly as it would have been.
            # (This branch is immediate -- os.getpgid/killpg raise at once on a
            # dead pid -- so the retraction always beats any reader.)
            self._doomed_kills.pop(attempt_n, None)
            return

        message = (
            f"attempt {attempt_n} stopped early (doomed): every matched training cell "
            f"({', '.join(verdict.cells)}) trailed attempt {baseline_n}'s MEASURED curve at the "
            f"same step by >= {verdict.margin:.0%} for {verdict.polls_required} consecutive "
            f"advancing polls. The attempt is recorded as '{doomed.DOOMED_KILLED_CLASS}' -- we "
            f"stopped paying; the science did not fail."
        )
        self._safe_emit(state, "attempt_doomed_killed", dict(evidence))
        self._safe_emit(
            state,
            "run_warning",
            {
                "code": "doomed_run_killed",
                "message": message,
                "attempt_n": attempt_n,
                "baseline_attempt_n": baseline_n,
                "evidence": list(verdict.detail),
            },
        )

    def _apply_doomed_kill(self, attempt_n: int, assessment: dict) -> dict:
        """Overlay the honest doomed-kill marks onto a killed attempt's
        assessment. A no-op (the same dict object) when no kill fired, which
        is always the case with the flag off.

        The marks are chosen so the attempt is STRUCTURALLY excluded from
        every downstream science decision without any of those consumers
        needing to learn a new field:

        * ``failure_class``/``failure_signature`` = ``doomed_killed`` -- a
          distinct terminal, so nothing reads it as a science failure. It
          also keeps the killed attempt out of ``report_missing_twice``
          (which it would otherwise trip -- a killed child writes no report)
          and out of ``infra_signature_repeated`` (scope is neither infra
          nor method: we interrupted it, we did not observe it fail).
        * ``hard_quarantined`` -> champion selection, the seeding pool, the
          campaign score floor, REPRODUCED and CONTRADICTED all already
          filter on this. A killed attempt therefore cannot become champion
          and cannot seed a lineage.
        * ``doomed_kill`` -- the full evidence packet, for the campaign report
          and any downstream triage consumer.

        What is deliberately NOT suppressed: the attempt's measured ``cost``.
        Real money was spent, so it still accrues to ``state.spent`` and the
        attempt still counts toward ``max_attempts`` -- which is also what
        bounds the number of kills a campaign can perform.
        """
        kill = self._doomed_kills.get(attempt_n)
        if kill is None:
            return assessment

        marked = dict(assessment)
        reasons = [*(marked.get("quarantine_reasons") or ()), f"{doomed.DOOMED_KILLED_CLASS}:{kill['reason']}"]
        marked["doomed_kill"] = dict(kill)
        marked["failure_class"] = doomed.DOOMED_KILLED_CLASS
        marked["failure_signature"] = doomed.DOOMED_KILLED_CLASS
        marked["failure_scope"] = None
        marked["hard_quarantined"] = True
        marked["quarantine_reasons"] = reasons
        return marked

    def run(self) -> dict:
        try:
            return self._run_body()
        except CampaignLedgerError:
            raise
        except Exception as exc:  # noqa: BLE001 -- campaign-level never-raises rule
            return self._degrade_to_campaign_error(exc)

    def _run_body(self) -> dict:
        try:
            self.campaign_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CampaignLedgerError(f"cannot create campaign dir: {exc}") from exc

        loaded = self.ledger.load_state()
        if loaded is None:
            return self._run_fresh()
        return self._run_resume(loaded)

    def _run_fresh(self) -> dict:
        state = self._new_state(state="init")
        self._state = state

        try:
            notices = self.stages.validate_init()
        except CampaignInitError as exc:
            state.warnings.append(f"campaign_init_error:{exc}")
            return self._finish_synthetic_terminal(
                state, kind="EXHAUSTED", rule="init", stop_reason=f"campaign_error:{exc}",
            )

        state.warnings.extend(notices)
        state.updated_at = time.time()
        self.ledger.write_state(state)
        self._safe_emit(state, "campaign_started", {
            "project_id": self.project_id, "paper_ref": self.paper_ref,
            "mode": self.mode, "driver": self.driver, "budget": dict(self.budget),
        })

        state.state = "understand"
        understood = self.stages.understand()
        state.understanding_sha256 = understood.get("sha256")
        state.rubric_sha256 = understood.get("rubric_sha256")
        blocking = understood.get("blocking") or []
        if blocking:
            return self._finish_synthetic_terminal(
                state, kind="INFEASIBLE", rule="understanding_gate",
                stop_reason=f"understanding_gate:{blocking[0]}",
            )

        state.state = "attempt_loop"
        state.updated_at = time.time()
        self.ledger.write_state(state)
        return self._loop(state)

    def _finish_synthetic_terminal(self, state: CampaignState, *, kind: str, rule: str, stop_reason: str) -> dict:
        # No stages.decide() ran (INIT error / UNDERSTAND block / PLAN refusal);
        # synthesize an equivalent decision dict and reuse the terminal path.
        decision = {"kind": kind, "rule": rule, "stop_reason": stop_reason, "next_plan": None}
        return self._finish_from_decision(state, decision)

    def _run_resume(self, state: CampaignState) -> dict:
        # Resume dispatch, spec §5 F7.
        self._state = state
        if state.state == "terminal":
            return state.terminal or {}
        if state.state in ("init", "understand"):
            # No money moved yet (UNDERSTAND is CPU-only, validate_init is $0);
            # safe + simplest to restart the whole pre-attempt-loop sequence.
            return self._run_fresh()

        if state.state == "paused":
            # U9: real operator-approval semantics land later; --resume IS the
            # approval for now.
            state.pending_approval = None
            state.state = "attempt_loop"
            state.updated_at = time.time()
            self.ledger.write_state(state)
            return self._loop(state)

        if state.in_flight is not None:
            return self._resume_in_flight(state)

        rows = self.ledger.read_rows()
        latest = CampaignLedger.latest_by_status(rows, state.next_attempt_n)

        if "launched" in latest and "assessed" not in latest:
            return self._resume_orphaned_intent(state, latest["launched"])

        if "assessed" in latest and "decided" not in latest:
            outcome = self._decide_and_continue(state, rows)
            return outcome if outcome is not None else self._loop(state)

        if "decided" in latest:
            decision = latest["decided"].get("decision") or {}
            if decision.get("kind") == "CONTINUE":
                self._apply_continue(state, decision)
                return self._loop(state)
            return self._finish_from_decision(state, decision)

        return self._loop(state)

    def _resume_in_flight(self, state: CampaignState) -> dict:
        in_flight = state.in_flight
        alive = self.stages.liveness_probe(in_flight)

        if alive:
            # A resumed campaign re-attaching to a still-running child is
            # AWAIT just as much as the fresh path is -- it must be watched
            # too, or a doomed attempt survives simply by outliving a campaign
            # restart.
            raw_result = self._await_with_doomed_watch(
                state,
                in_flight.attempt_n,
                {"pid": in_flight.pid, "run_dir": in_flight.run_dir, "lease_ref": in_flight.lease_ref},
            )
            rows = self.ledger.read_rows()
            planned = CampaignLedger.latest_by_status(rows, in_flight.attempt_n).get("launched", {})
            assessment = self._apply_doomed_kill(
                in_flight.attempt_n, self.stages.assess(raw_result, planned)
            )
            outcome = self._record_assessment_and_continue(state, in_flight.attempt_n, assessment)
            return outcome if outcome is not None else self._loop(state)

        # Dead. The attempt may have COMPLETED while the campaign process was
        # down (its own "assessed" row already landed before the crash) or it
        # may still be genuinely unassessed on disk. Check the ledger BEFORE
        # touching the run dir: quarantining first would archive a real,
        # possibly-reproduced result out from under assess_from_disk and
        # supersede it with report_missing (controller finding R1). Launch-time
        # force-quarantine (Codex F6, driver-side) already guarantees a clean
        # dir before any next launch, so no quarantine call belongs here at
        # all -- unlike ``_resume_orphaned_intent``, which quarantines because
        # launch() never even fired for that attempt.
        rows = self.ledger.read_rows()
        latest = CampaignLedger.latest_by_status(rows, in_flight.attempt_n)
        if "assessed" in latest:
            # Crash landed between the ASSESS tail's append_row("assessed")
            # and its write_state: the assessment is already durable. Reuse
            # it -- re-running assess_from_disk or re-appending would risk
            # judging a quarantined dir and double-counting spend.
            assessment = dict(latest["assessed"].get("assessment") or {})
            outcome = self._apply_assessment_tail(state, assessment)
            return outcome if outcome is not None else self._loop(state)

        assessment = self.stages.assess_from_disk(in_flight)
        outcome = self._record_assessment_and_continue(state, in_flight.attempt_n, assessment)
        return outcome if outcome is not None else self._loop(state)

    def _resume_orphaned_intent(self, state: CampaignState, launched_row: dict) -> dict:
        # Crash between write-ahead 2a (intent row) and 2b (in_flight write):
        # launch() never fired. Quarantine any residue, then relaunch the SAME
        # attempt_n -- the only path where a superseding intent row is allowed.
        synthetic = InFlight(
            attempt_n=int(launched_row.get("attempt_n", state.next_attempt_n)),
            driver=str(launched_row.get("driver", state.driver)),
            run_dir=str(launched_row.get("run_dir", "")),
            pid=None, lease_ref=None,
            launched_at=float(launched_row.get("launched_at", 0.0)),
        )
        self.stages.quarantine(synthetic)
        return self._loop(state, allow_supersede=True)

    def _loop(self, state: CampaignState, *, allow_supersede: bool = False) -> dict:
        # PLAN -> WRITE-AHEAD -> LAUNCH -> AWAIT -> ASSESS, repeated per attempt.
        while True:
            if self.stages.poll_steering is not None:
                try:
                    self.stages.poll_steering(state)
                except Exception as exc:  # noqa: BLE001 -- steering polling is fail-soft (observability)
                    state.warnings.append(f"poll_steering_failed:{type(exc).__name__}")

            rows = self.ledger.read_rows()
            planned = self.stages.plan_attempt(state, rows)
            refusal = planned.get("refusal")
            if refusal:
                if planned.get("downgrade_to_checkpoint") and state.mode == "unattended":
                    state.mode = "checkpoint"
                    state.pending_approval = {"reason": refusal, "planned": planned}
                    state.state = "paused"
                    state.updated_at = time.time()
                    self.ledger.write_state(state)
                    self._safe_emit(state, "campaign_awaiting_operator", {"reason": refusal})
                    return {"kind": "PAUSED", "pending_approval": dict(state.pending_approval)}
                return self._finish_synthetic_terminal(
                    state, kind="EXHAUSTED", rule="plan_refusal", stop_reason=refusal,
                )

            attempt_n = planned["attempt_n"]
            latest = CampaignLedger.latest_by_status(rows, attempt_n)
            if not allow_supersede and "launched" in latest and "assessed" not in latest:
                raise RuntimeError(
                    f"refusing double launch for attempt_n={attempt_n}: unassessed launch already recorded"
                )
            allow_supersede = False

            now = time.time()
            self.ledger.append_row({
                "attempt_n": attempt_n, "status": "launched",
                "directives_sha256": planned["directives_sha256"], "envelope": planned["envelope"],
                "driver": self.driver, "project_id": planned["project_id"],
                "run_dir": planned["run_dir"], "launched_at": now,
            })
            state.in_flight = InFlight(
                attempt_n=attempt_n, driver=self.driver, run_dir=planned["run_dir"],
                pid=None, lease_ref=None, launched_at=now,
            )
            state.updated_at = time.time()
            self.ledger.write_state(state)
            self._safe_emit(state, "attempt_started", {
                "attempt_n": attempt_n, "directives_sha256": planned["directives_sha256"],
                "envelope": planned["envelope"],
            })

            handle = self.stages.launch(planned["launch_payload"])
            state.in_flight = InFlight(
                attempt_n=attempt_n, driver=self.driver,
                run_dir=str(handle.get("run_dir", planned["run_dir"])),
                pid=handle.get("pid"), lease_ref=handle.get("lease_ref"), launched_at=now,
            )
            state.updated_at = time.time()
            self.ledger.write_state(state)

            raw_result = self._await_with_doomed_watch(state, attempt_n, handle)
            assessment = self._apply_doomed_kill(attempt_n, self.stages.assess(raw_result, planned))

            outcome = self._record_assessment_and_continue(state, attempt_n, assessment)
            if outcome is not None:
                return outcome

    def _record_assessment_and_continue(self, state: CampaignState, attempt_n: int, assessment: dict) -> dict | None:
        # ASSESS row + tail; shared by the normal loop and every resume path
        # that hasn't already recorded this attempt's assessment.
        now = time.time()
        self.ledger.append_row({
            "attempt_n": attempt_n, "status": "assessed",
            "assessment": dict(assessment), "assessed_at": now,
        })
        self._safe_emit(state, "attempt_assessed", {"attempt_n": attempt_n, "assessment": dict(assessment)})
        # Pin-at-first-sight: the assess stage computed the observed rubric
        # hash (never persisted it itself, which would race this method's
        # own writes); adopt it here so the very next write_state persists it.
        if state.rubric_sha256 is None and assessment.get("rubric_sha256_observed"):
            state.rubric_sha256 = assessment.get("rubric_sha256_observed")
        return self._apply_assessment_tail(state, assessment)

    def _apply_assessment_tail(self, state: CampaignState, assessment: dict) -> dict | None:
        # ASSESS tail (spend + in_flight clear) + DISTILL + DECIDE. Split out
        # of ``_record_assessment_and_continue`` so a resume that finds the
        # "assessed" row already durable can share this tail WITHOUT
        # appending a duplicate row or double-counting the cost into spent
        # (controller finding R1).
        state.in_flight = None
        cost = assessment.get("cost") or {}
        for meter in ("llm_usd", "gpu_usd", "gpu_hours", "wall_s"):
            state.spent[meter] = float(state.spent.get(meter, 0.0)) + float(cost.get(meter, 0.0))
        state.updated_at = time.time()
        self.ledger.write_state(state)

        # DISTILL mines CROSS-RUN memory (lessons / recipes / ExperienceMemory)
        # from what an attempt observed. A doomed-killed attempt observed
        # nothing -- we interrupted it mid-training -- so mining it would
        # teach the harness a "lesson" about a run WE stopped, permanently
        # poisoning memory that outlives this campaign. Skip it: the kill is
        # already recorded honestly in the ledger, the report and the SSE log.
        if assessment.get("doomed_kill"):
            state.warnings.append("distill_skipped:doomed_killed")
        else:
            try:
                self.stages.distill(assessment)
            except Exception as exc:  # noqa: BLE001 -- DISTILL is fail-soft by design (spec §13)
                state.warnings.append(f"distill_failed:{type(exc).__name__}")

        rows = self.ledger.read_rows()
        return self._decide_and_continue(state, rows)

    def _decide_and_continue(self, state: CampaignState, rows: list) -> dict | None:
        decision = self.stages.decide(state, rows)
        self.ledger.append_row({
            "attempt_n": state.next_attempt_n, "status": "decided", "decision": dict(decision),
        })
        self._safe_emit(state, "campaign_decision", {
            "attempt_n": state.next_attempt_n, "kind": decision.get("kind"), "rule": decision.get("rule"),
            "stop_reason": decision.get("stop_reason"),
            "fingerprint": self._directives_sha256_for_attempt(rows, state.next_attempt_n),
        })

        if decision.get("kind") == "CONTINUE":
            self._apply_continue(state, decision)
            if state.mode == "checkpoint":
                state.pending_approval = {"decision": decision}
                state.state = "paused"
                state.updated_at = time.time()
                self.ledger.write_state(state)
                self._safe_emit(state, "campaign_awaiting_operator", {"decision": decision})
                return {"kind": "PAUSED", "pending_approval": dict(state.pending_approval)}
            return None

        return self._finish_from_decision(state, decision)

    @staticmethod
    def _apply_continue(state: CampaignState, decision: dict) -> None:
        state.next_attempt_n += 1
        next_plan = decision.get("next_plan") or {}
        scope_rung = next_plan.get("scope_rung")
        if scope_rung is not None:
            state.scope_rung = scope_rung

    def _finish_from_decision(self, state: CampaignState, decision: dict) -> dict:
        terminal = {
            "kind": decision.get("kind"), "rule": decision.get("rule"),
            "stop_reason": decision.get("stop_reason"),
            "champion_attempt_n": decision.get("champion_attempt_n"),
            "spent": dict(state.spent),
        }
        state.terminal = terminal
        state.state = "terminal"
        state.updated_at = time.time()
        self.ledger.write_state(state)
        self._safe_emit(state, "campaign_terminal", dict(terminal))
        rows = self.ledger.read_rows()
        self.stages.write_reports(state, rows, decision)
        return state.terminal

    def _degrade_to_campaign_error(self, exc: Exception) -> dict:
        # Campaign-level never-raises fallback, spec §13.
        state = self._state
        if state is None:
            state = self._new_state(state="terminal")
            self._state = state

        decision = {
            "kind": "EXHAUSTED", "rule": "campaign_error",
            "stop_reason": f"campaign_error:{type(exc).__name__}", "next_plan": None,
        }
        try:
            return self._finish_from_decision(state, decision)
        except CampaignLedgerError:
            # A durable-ledger failure inside the degrade path must halt the
            # campaign exactly like it does in run()'s own dispatch --
            # swallowing it here would return a terminal-shaped dict while
            # campaign.json still shows a live in_flight (false "done", an
            # orphaned spending attempt; controller finding C1).
            raise
        except Exception as report_exc:  # noqa: BLE001 -- best-effort; own failure only warns
            state.warnings.append(f"campaign_error_report_failed:{type(report_exc).__name__}")
            return {
                "kind": decision["kind"], "rule": decision["rule"],
                "stop_reason": decision["stop_reason"], "champion_attempt_n": None,
                "spent": dict(state.spent),
            }
