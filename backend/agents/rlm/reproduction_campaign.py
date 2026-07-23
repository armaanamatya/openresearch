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
from typing import TYPE_CHECKING, Any

from backend.agents.rlm import doomed_run_comparator as doomed

if TYPE_CHECKING:
    from backend.agents.rlm.scheduler_authority_controller import SchedulerAuthorityController


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
        branch_tree_event_store: Any | None = None,
        scheduler_controller: "SchedulerAuthorityController | None" = None,
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
        # Test/host injection point.  Production lazily opens the configured
        # controller EventStore only when the tree flag is enabled; default-off
        # campaigns neither construct nor touch it.
        self._branch_tree_event_store = branch_tree_event_store
        # Optional authoritative scheduler controller, constructed by the
        # campaign composition root ONLY when both scheduler flags and an
        # explicit authority spec are present (byte-identical-OFF otherwise).
        # Mirrors the ``branch_tree_event_store`` optional-injection precedent.
        self.scheduler_controller = scheduler_controller

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
    def _scheduler_tree_enabled() -> bool:
        """Locked default-OFF truthiness gate for lineage observability."""
        return os.environ.get("OPENRESEARCH_SCHEDULER_TREE", "").strip().lower() in (
            "1", "true", "yes",
        )

    def _maybe_emit_root_branch_spawned(
        self,
        state: CampaignState,
        launched_row: Mapping[str, Any],
    ) -> None:
        """Record the only branch-tree fact the serial campaign owns today.

        The existing loop has one durable launch intent but no branch queue,
        paper-pinned optimizer-step receipt, checkpoint transition, or ASHA
        action owner.  Consequently tree mode records the root branch only;
        it must *not* manufacture rung climbs, promotions, freezes, revivals,
        or true deletes from a shadow advisory.  ``directives_sha256`` is the
        F10 novelty fingerprint already durable in the write-ahead row.

        Observability remains fail-soft while the scheduler is non-authoritative:
        a controller EventStore outage is recorded as a warning after the
        launch intent is durable and never changes money, verdict, or decision.
        """
        # Under authority the ``SchedulerAuthorityController`` owns
        # ``branch-tree:<campaign_id>`` lineage as the SOLE writer; since
        # ``campaign_id == project_id`` there, emitting here too would
        # double-write the same aggregate and risk an ``expected_version``
        # collision. ``scheduler_controller`` is None on every default path
        # today, so this is byte-identical when authority is not live.
        if self.scheduler_controller is not None:
            return
        if not self._scheduler_tree_enabled():
            return
        attempt_n = launched_row.get("attempt_n")
        # Later serial attempts are iterations of the same root, not genuine
        # branch forks.  Emitting them as children would misstate lineage.
        if attempt_n != 1:
            return
        fingerprint = launched_row.get("directives_sha256")
        if not isinstance(fingerprint, str) or not fingerprint:
            state.warnings.append("branch_lineage_skipped:missing_f10_fingerprint")
            return
        branch_type = launched_row.get("branch_type", "faithful")
        if branch_type not in {"faithful", "ambiguity", "discovery"}:
            state.warnings.append("branch_lineage_skipped:invalid_branch_type")
            return

        created_store = False
        store = self._branch_tree_event_store
        try:
            if store is None:
                # Keep imports and SQLite construction inside the enabled
                # branch so OFF runs remain side-effect-free and byte-identical.
                from backend.config import get_settings
                from backend.eventstore.sqlite_store import SqliteEventStore

                store = SqliteEventStore(get_settings().database_url)
                created_store = True

            from backend.agents.rlm.branch_lineage import BranchSpawned, branch_tree_aggregate_id
            from backend.messaging.envelope import AggregateId, CorrelationId, EventEnvelope, new_event_id

            aggregate_id = AggregateId(branch_tree_aggregate_id(state.project_id))
            branch_id = str(attempt_n)
            # A recovery/manual repair may revisit the durable launch.  The
            # existing F10 fingerprint is the only dedup identity: never hash
            # a scheduler-specific replacement key.
            for stored in store.load(aggregate_id):
                if (
                    stored.event_type == BranchSpawned.event_type
                    and stored.payload.get("branch_id") == branch_id
                    and stored.payload.get("hypothesis_fingerprint") == fingerprint
                ):
                    return
            store.append(
                aggregate_id=aggregate_id,
                aggregate_type="branch_tree",
                events=[BranchSpawned(
                    branch_id=branch_id,
                    branch_type=branch_type,
                    parent_branch_id=None,
                    rung=0,
                    hypothesis_fingerprint=fingerprint,
                )],
                expected_version=store.get_aggregate_version(aggregate_id),
                envelopes=[EventEnvelope(
                    event_id=new_event_id(),
                    correlation_id=CorrelationId(state.project_id),
                    source="agents.rlm.reproduction_campaign",
                )],
            )
        except Exception as exc:  # noqa: BLE001 -- scheduler is shadow/observability-only
            state.warnings.append(f"branch_lineage_emit_failed:{type(exc).__name__}")
        finally:
            if created_store:
                try:
                    store.close()
                except Exception:  # noqa: BLE001 -- cleanup never affects campaign semantics
                    pass

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
        return self._drive(state)

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
            # Track E §6.5: the operator resuming a paused campaign is a recorded
            # resume-approval intervention (display-only autonomy telemetry; no-op
            # + byte-identical when OPENRESEARCH_HUMAN_INTERVENTION_LOG is off).
            # Recorded BEFORE pending_approval is cleared so `why` keeps the reason.
            from backend.agents.rlm.human_intervention import record_intervention

            record_intervention(
                self.run_dir,
                kind="resume-approval",
                what="operator resumed a paused campaign",
                why=str((state.pending_approval or {}).get("reason", "")),
            )
            state.pending_approval = None
            state.state = "attempt_loop"
            state.updated_at = time.time()
            self.ledger.write_state(state)
            return self._drive(state)

        if state.in_flight is not None:
            return self._resume_in_flight(state)

        rows = self.ledger.read_rows()
        latest = CampaignLedger.latest_by_status(rows, state.next_attempt_n)

        if "launched" in latest and "assessed" not in latest:
            return self._resume_orphaned_intent(state, latest["launched"])

        if "assessed" in latest and "decided" not in latest:
            outcome = self._decide_and_continue(state, rows)
            return outcome if outcome is not None else self._drive(state)

        if "decided" in latest:
            decision = latest["decided"].get("decision") or {}
            if decision.get("kind") == "CONTINUE":
                self._apply_continue(state, decision)
                return self._drive(state)
            return self._finish_from_decision(state, decision)

        return self._drive(state)

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
            return outcome if outcome is not None else self._drive(state)

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
            return outcome if outcome is not None else self._drive(state)

        assessment = self.stages.assess_from_disk(in_flight)
        outcome = self._record_assessment_and_continue(state, in_flight.attempt_n, assessment)
        return outcome if outcome is not None else self._drive(state)

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
        return self._drive(state, allow_supersede=True)

    def _drive(self, state: CampaignState, *, allow_supersede: bool = False) -> dict:
        # Single dispatch seam between the serial attempt loop and the
        # receipt-gated cohort loop. OFF (no controller) is byte-identical to
        # the historical serial ``_loop`` -- this is the whole of the B3.2
        # flag-guard: an authority controller is constructed by
        # ``build_campaign`` ONLY when both scheduler flags AND an explicit
        # authority spec are present, so ``scheduler_controller is None`` on
        # every default path.
        if self.scheduler_controller is not None:
            return self._cohort_loop(state, allow_supersede=allow_supersede)
        return self._loop(state, allow_supersede=allow_supersede)

    def _cohort_loop(self, state: CampaignState, *, allow_supersede: bool = False) -> dict:
        # Receipt-gated authority cohort loop (Phase B3.2). Drives the already
        # bootstrapped ``SchedulerAuthorityController`` through
        # ``claim -> run -> receipt -> decide_rung -> apply``, then lets the
        # campaign's OWN deterministic ``stages.decide`` render a terminal
        # verdict that ALWAYS wins over the authority reordering. The
        # controller only reorders the CONTINUE cohort (promote/freeze/kill of
        # same-rung branches); it never flips or suppresses a terminal.
        #
        # ``allow_supersede`` is accepted for dispatch-signature parity with
        # ``_loop`` (the orphaned-intent relaunch seam); the cohort loop plans
        # per-branch launches from the controller's queue rather than the
        # serial write-ahead ledger, so there is no single-attempt double-launch
        # to supersede here.
        del allow_supersede

        controller = self.scheduler_controller
        assert controller is not None  # dispatch guard guarantees this
        from backend.agents.rlm import cell_checkpoint
        from backend.agents.rlm.scheduler_receipt_producer import build_raw_receipt
        from backend.agents.rlm.scheduler_runtime import SchedulerRuntimeError

        ladder = controller.spec.ladder
        # Monotonic per-branch attempt counter: each branch launch gets its own
        # ``attempt_n`` for its write-ahead + assessed ledger rows AND its
        # receipt, so a 2-wide wave never collides two rows on one key.
        attempt_counter = state.next_attempt_n
        # Per-branch cell id counter (audit-stable, per launch).
        cell_counter = 0
        # Deterministic provider GPU-$ per branch, ACCUMULATED across waves:
        # ``decide_rung`` requires a cost for EVERY branch at the rung, and an
        # under-cap rung is filled by several claim waves (a100_cap<cohort), so
        # a per-wave dict would drop the earlier waves' costs. Never sourced
        # from cost_ledger.jsonl -- only each branch's own deterministic assess.
        provider_gpu: dict[str, float] = {}

        while True:
            launches = controller.claim_launches(max_parallel=controller.spec.a100_cap)

            # Termination: nothing left to claim AND no branch still in an
            # active state ⇒ fall through to a normal terminal decision.
            if not launches:
                active = any(
                    branch.state in {"queued", "launching", "awaiting_receipt"}
                    for branch in controller.branches.values()
                )
                if not active:
                    break

            # The rung of the wave we are about to run (all claimed branches at
            # one wave share a rung by the claim policy, but read it per-branch
            # to be safe and decide the lowest claimed rung).
            wave_rungs: set[int] = set()

            for launch in launches:
                cell_counter += 1
                attempt_n = attempt_counter
                attempt_counter += 1
                wave_rungs.add(launch.rung)

                # Build a production-shaped launch payload keyed to this branch.
                # ``plan_attempt`` owns the campaign directives; we reuse it for
                # the CONTINUE-cohort attempt, then annotate the branch identity
                # so the runner writes into this branch's own run dir.
                rows = self.ledger.read_rows()
                planned = self.stages.plan_attempt(state, rows)
                payload = self._cohort_launch_payload(planned, launch)

                handle = self.stages.launch(payload)
                raw_result = self.stages.await_result(dict(handle))
                assessment = self.stages.assess(dict(raw_result), payload)

                # Provider GPU-$ comes ONLY from the deterministic assessment
                # (never cost_ledger.jsonl). Missing/malformed ⇒ conservative 0.
                cost = assessment.get("cost") or {}
                gpu_usd = cost.get("gpu_usd", 0.0)
                try:
                    gpu_usd = float(gpu_usd)
                    if gpu_usd < 0.0 or gpu_usd != gpu_usd:  # neg / NaN
                        gpu_usd = 0.0
                except (TypeError, ValueError):
                    gpu_usd = 0.0
                provider_gpu[launch.branch_id] = gpu_usd

                # The cell output dir + its checkpoint_components/ are produced
                # by the branch attempt's runner under its run dir. Ordinary
                # serial cells do not yet emit a controller-attested paper-step
                # checkpoint receipt (rlm CLAUDE.md), so the run_dir/cell layout
                # is the production-shaped source; the receipt producer verifies
                # the on-disk evidence and fails closed if it is absent.
                cell_out = self._cohort_cell_output_dir(handle, payload)

                # Resolve the REAL latest 5-component checkpoint the trainer
                # wrote under ``<cell_out>/checkpoints/step_<N>/`` (the dir
                # ``gpu_cell_runner`` points ``OPENRESEARCH_CELL_CHECKPOINT_DIR``
                # at). If none exists, this branch produced no resumable
                # checkpoint, so it CANNOT yield a verified receipt -- fail
                # CLOSED rather than fabricate one from a stub layout. The
                # message deliberately omits "incomplete" so the decide_rung
                # handler below does not swallow it as an under-cap tail.
                checkpoint_components_dir = cell_checkpoint.latest_checkpoint_dir(
                    cell_out / "checkpoints"
                )
                if checkpoint_components_dir is None:
                    raise SchedulerRuntimeError(
                        "cohort branch produced no resumable checkpoint under "
                        f"{cell_out / 'checkpoints'} -- refusing to fabricate a receipt"
                    )

                # termination_cause is 'training_diverged' ONLY when the
                # deterministic failure classifier says so; every other cause
                # (including a reversible underperformance freeze) is None.
                termination_cause = (
                    "training_diverged"
                    if assessment.get("failure_class") == "training_diverged"
                    else None
                )

                raw = build_raw_receipt(
                    run_dir=self.run_dir,
                    cell_output_dir=cell_out,
                    checkpoint_components_dir=checkpoint_components_dir,
                    ladder=ladder,
                    campaign_id=self.project_id,
                    branch_id=launch.branch_id,
                    parent_branch_id=launch.parent_branch_id,
                    attempt_n=attempt_n,
                    cell_id=f"{launch.branch_id}-cell-{cell_counter}",
                    from_step=launch.from_step,
                    to_step=launch.to_step,
                    seed=launch.seed,
                    termination_cause=termination_cause,
                    dataset_manifest_path=self._cohort_pinned_dataset_manifest(cell_out),
                    run_spec_path=self._cohort_pinned_run_spec(cell_out),
                )
                # ``attest`` MUST be the campaign's real fail-closed ledger
                # appender -- never a worker-authored write. Receipt rows carry
                # no "status" key, so ``latest_by_status`` ignores them and the
                # attempt-status ledger stays coherent.
                controller.record_cell_receipt(raw, attest=self.ledger.append_row)

            # When every branch at the current rung has a receipt, apply the
            # two-meter authority policy. A branch still lacking a receipt (the
            # under-cap tail of a wave) makes decide_rung raise "incomplete":
            # that is FAIL-CLOSED and correct -- loop back to claim_launches to
            # run the tail; never force a decision or synthesize a receipt.
            for rung in sorted(wave_rungs):
                try:
                    controller.decide_rung(
                        rung=rung, provider_gpu_usd_by_branch=provider_gpu,
                    )
                except SchedulerRuntimeError as exc:
                    if "incomplete" in str(exc):
                        continue
                    raise

            # Terminal deterministic decision ALWAYS wins. Run the campaign's
            # own decide against the durable ledger; a terminal verdict returns
            # immediately (authority can never flip or suppress it). On CONTINUE
            # we simply proceed to the next claim_launches wave -- authority has
            # already reordered the cohort (promotions re-queued at rung+1).
            #
            # We deliberately do NOT route CONTINUE through
            # ``_decide_and_continue`` here: that bumps ``next_attempt_n``, may
            # PAUSE in checkpoint mode, and returns a PAUSED dict -- all of which
            # would break the cohort drive. The cohort's forward motion is the
            # controller's re-queue, not a serial attempt increment.
            rows = self.ledger.read_rows()
            decision = self.stages.decide(state, rows)
            if decision.get("kind") != "CONTINUE":
                return self._finish_from_decision(state, decision)

        # No active branches remain: render the terminal decision.
        rows = self.ledger.read_rows()
        decision = self.stages.decide(state, rows)
        return self._finish_from_decision(state, decision)

    def _cohort_launch_payload(self, planned: Mapping[str, Any], launch: Any) -> dict:
        # Production-shaped per-branch launch payload. Starts from the campaign
        # directives ``plan_attempt`` produced (carrying the CONTINUE-cohort
        # envelope + run spec) and annotates the branch identity so the runner
        # keys artifacts to this branch's seed/rung/run dir. Kept a plain dict
        # so an injected stub ``launch`` can read the branch coordinates
        # directly without reconstructing directives.
        payload = dict(planned)
        payload["branch_id"] = launch.branch_id
        payload["branch_type"] = launch.branch_type
        payload["seed"] = launch.seed
        payload["rung"] = launch.rung
        payload["from_step"] = launch.from_step
        payload["to_step"] = launch.to_step
        payload["parent_branch_id"] = launch.parent_branch_id
        payload["resume_checkpoint_path"] = launch.resume_checkpoint_path
        return payload

    def _cohort_cell_output_dir(self, handle: Mapping[str, Any], payload: Mapping[str, Any]) -> Path:
        # The branch attempt's cell output dir (where the runner wrote
        # metrics.json + checkpoint_components/). Production shape mirrors the
        # ordinary attempt layout: ``<branch_run_dir>/code``. An injected stub
        # runner materializes the same layout under the handle's run dir.
        run_dir = handle.get("run_dir") or payload.get("run_dir") or str(self.run_dir)
        return Path(run_dir) / "code"

    @staticmethod
    def _cohort_pinned_dataset_manifest(cell_out: Path) -> Path:
        # Pinned dataset manifest for the receipt fingerprints. Deterministic,
        # per-cell path; the runner writes it beside the cell metrics.
        return Path(cell_out) / "dataset_manifest.json"

    @staticmethod
    def _cohort_pinned_run_spec(cell_out: Path) -> Path:
        # Pinned run spec for the receipt fingerprints (image/env pin).
        return Path(cell_out) / "run_spec.json"

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
                    # Track E §6.5: the run hit a blocking gate requiring the operator
                    # (a downgrade-to-checkpoint plan refusal, usually a budget/
                    # enforceability limit) — a blocking compute-approval gate.
                    from backend.agents.rlm.human_intervention import record_intervention

                    record_intervention(
                        self.run_dir,
                        kind="compute-approval",
                        what="campaign paused awaiting operator (plan refusal)",
                        why=str(refusal),
                        blocking=True,
                    )
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
            launched_row = {
                "attempt_n": attempt_n, "status": "launched",
                "directives_sha256": planned["directives_sha256"], "envelope": planned["envelope"],
                "driver": self.driver, "project_id": planned["project_id"],
                "run_dir": planned["run_dir"], "launched_at": now,
            }
            # Scheduler enrichments are durable only when explicitly enabled
            # by PLAN. Leaving the faithful/False defaults absent preserves
            # historical write-ahead rows byte-for-byte; ASSESS-on-resume
            # consumes this ledger copy rather than mutable directives.
            if "branch_type" in planned:
                if planned["branch_type"] not in {"faithful", "ambiguity", "discovery"}:
                    raise CampaignLedgerError(f"invalid planned branch_type: {planned['branch_type']!r}")
                launched_row["branch_type"] = planned["branch_type"]
            if "is_safety_bracket" in planned:
                if type(planned["is_safety_bracket"]) is not bool:
                    raise CampaignLedgerError("planned is_safety_bracket must be bool")
                if planned["is_safety_bracket"] and planned.get("branch_type", "faithful") != "faithful":
                    raise CampaignLedgerError("planned is_safety_bracket requires branch_type='faithful'")
                if planned["is_safety_bracket"] and os.environ.get(
                    "OPENRESEARCH_SCHEDULER_TREE", ""
                ).strip().lower() not in ("1", "true", "yes"):
                    raise CampaignLedgerError("planned is_safety_bracket requires OPENRESEARCH_SCHEDULER_TREE")
                launched_row["is_safety_bracket"] = planned["is_safety_bracket"]
            self.ledger.append_row(launched_row)
            # Event only after the write-ahead row makes the F10 fingerprint
            # durable.  No other scheduler transition is factual on this path.
            self._maybe_emit_root_branch_spawned(state, launched_row)
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
                # Track E §6.5: checkpoint mode pauses for operator sign-off before
                # the next attempt — a blocking config-choice gate.
                from backend.agents.rlm.human_intervention import record_intervention

                record_intervention(
                    self.run_dir,
                    kind="config-choice",
                    what="campaign paused for checkpoint sign-off",
                    blocking=True,
                )
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
