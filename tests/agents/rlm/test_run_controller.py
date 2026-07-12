"""Unit tests for ``backend.agents.rlm.run_controller`` -- pure/injectable
driver-controller logic (WS3, Phase 1).

Phase-1 scope: no cluster calls, no subprocess spawning -- every assertion
here is over pure function output or a fake injected lease double.
"""

from __future__ import annotations

from backend.agents.rlm.run_controller import (
    acquire_drive_lease,
    build_controller_command,
    classify_controller_exit,
    durable_controller_enabled,
)


# ---------------------------------------------------------------------------
# durable_controller_enabled -- default-OFF flag
# ---------------------------------------------------------------------------

class TestDurableControllerEnabled:
    def test_off_by_default_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENRESEARCH_DURABLE_CONTROLLER", raising=False)
        assert durable_controller_enabled() is False

    def test_on_when_truthy_token(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_DURABLE_CONTROLLER", "1")
        assert durable_controller_enabled() is True

    def test_off_when_explicit_zero(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_DURABLE_CONTROLLER", "0")
        assert durable_controller_enabled() is False


# ---------------------------------------------------------------------------
# build_controller_command
# ---------------------------------------------------------------------------

class TestBuildControllerCommand:
    def test_exact_argv(self) -> None:
        assert build_controller_command("2605.15155", "prj_abc123") == [
            "python",
            "-m",
            "backend.cli",
            "campaign",
            "2605.15155",
            "--project-id",
            "prj_abc123",
            "--resume",
        ]

    def test_uses_campaign_not_reproduce(self) -> None:
        argv = build_controller_command("1412.6980", "prj_xyz")
        assert "campaign" in argv
        assert "reproduce" not in argv


# ---------------------------------------------------------------------------
# classify_controller_exit
# ---------------------------------------------------------------------------

class TestClassifyControllerExit:
    def test_zero_is_complete(self) -> None:
        assert classify_controller_exit(0) == "complete"

    def test_two_is_paused(self) -> None:
        assert classify_controller_exit(2) == "paused"

    def test_sigkill_137_is_crash(self) -> None:
        assert classify_controller_exit(137) == "crash"

    def test_three_is_money_halt_not_crash(self) -> None:
        # Exit 3 is cmd_campaign's own [MONEY-HALT] (backend/cli.py, on
        # CampaignLedgerError) -- a deliberate fail-closed safety stop, not a
        # process crash. A durable reaper/restarter must be able to tell the
        # two apart so it never respawns/retries after a money-halt.
        assert classify_controller_exit(3) == "money_halt"

    def test_pre_existing_terminals_unchanged_by_the_money_halt_branch(self) -> None:
        # Regression guard: adding the exit-3 branch must not perturb the
        # pre-existing 0/2/137 mappings.
        assert classify_controller_exit(0) == "complete"
        assert classify_controller_exit(2) == "paused"
        assert classify_controller_exit(137) == "crash"


# ---------------------------------------------------------------------------
# acquire_drive_lease -- duck-typed lease double, contention semantics
# ---------------------------------------------------------------------------

class _FakeLease:
    """A single-slot lease double: only one acquirer may hold ``run_id``."""

    def __init__(self) -> None:
        self._owner: str | None = None
        self.calls: list[tuple[str, str, float]] = []

    def acquire(self, run_id: str, owner_id: str, now_epoch: float):
        self.calls.append((run_id, owner_id, now_epoch))
        if self._owner is None or self._owner == owner_id:
            self._owner = owner_id
            return object()  # stand-in fence token
        return None


class TestAcquireDriveLease:
    def test_solo_acquirer_gets_a_token(self) -> None:
        lease = _FakeLease()
        token = acquire_drive_lease(lease, "run-1", "owner-a", 100.0)
        assert token is not None

    def test_delegates_exact_arguments_to_the_lease(self) -> None:
        lease = _FakeLease()
        acquire_drive_lease(lease, "run-1", "owner-a", 100.0)
        assert lease.calls == [("run-1", "owner-a", 100.0)]

    def test_two_contending_acquirers_exactly_one_gets_a_token(self) -> None:
        lease = _FakeLease()
        token_a = acquire_drive_lease(lease, "run-1", "owner-a", 100.0)
        token_b = acquire_drive_lease(lease, "run-1", "owner-b", 100.0)

        assert (token_a is None) != (token_b is None), (
            "exactly one of the two contending acquirers must win"
        )
        assert token_a is not None
        assert token_b is None


# ---------------------------------------------------------------------------
# durable_controller_default_for_sandbox -- launch-boundary default (ON for gcp)
# ---------------------------------------------------------------------------

class TestDurableDefaultForSandbox:
    def test_default_on_for_gcp_when_unset(self, monkeypatch) -> None:
        from backend.agents.rlm import run_controller as rc

        monkeypatch.delenv("OPENRESEARCH_DURABLE_CONTROLLER", raising=False)
        assert rc.durable_controller_default_for_sandbox("gcp") is True

    def test_off_for_non_gcp_regardless_of_flag(self, monkeypatch) -> None:
        from backend.agents.rlm import run_controller as rc

        monkeypatch.setenv("OPENRESEARCH_DURABLE_CONTROLLER", "1")
        assert rc.durable_controller_default_for_sandbox("local") is False

    def test_opt_out_with_explicit_zero(self, monkeypatch) -> None:
        from backend.agents.rlm import run_controller as rc

        monkeypatch.setenv("OPENRESEARCH_DURABLE_CONTROLLER", "0")
        assert rc.durable_controller_default_for_sandbox("gcp") is False
