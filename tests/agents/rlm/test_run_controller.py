"""Unit tests for ``backend.agents.rlm.run_controller`` -- pure/injectable
driver-controller logic (WS3, Phase 1).

Phase-1 scope: no cluster calls, no subprocess spawning -- every assertion
here is over pure function output or a fake injected lease double.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.run_controller import (
    acquire_drive_lease,
    build_controller_command,
    classify_controller_exit,
    controller_job_exit_code,
    durable_controller_enabled,
    validate_controller_image,
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
            "backend.agents.rlm.controller_entry",
            "--cloud",
            "gcp",
            "--paper",
            "2605.15155",
            "--project-id",
            "prj_abc123",
            "--runs-root",
            "/mnt/reprolab/controller-runs",
            "--execution-mode",
            "max",
            "--gpu-mode",
            "auto",
        ]

    def test_threads_cloud_runs_root_run_spec_and_model(self) -> None:
        argv = build_controller_command(
            "1412.6980",
            "prj_xyz",
            cloud="azure",
            runs_root="/state/runs",
            run_spec="configs/verify_deep_run_spec.json",
            root_model="opus-foundry",
            execution_mode="efficient",
            gpu_mode="prefer",
            minimize_compute=True,
        )
        assert argv[argv.index("--cloud") + 1] == "azure"
        assert argv[argv.index("--runs-root") + 1] == "/state/runs"
        assert argv[argv.index("--run-spec") + 1] == "configs/verify_deep_run_spec.json"
        assert argv[argv.index("--root-model") + 1] == "opus-foundry"
        assert argv[argv.index("--execution-mode") + 1] == "efficient"
        assert argv[argv.index("--gpu-mode") + 1] == "prefer"
        assert "--minimize-compute" in argv


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


def test_controller_job_exit_code_does_not_retry_intentional_halts():
    assert controller_job_exit_code(0) == 0
    assert controller_job_exit_code(2) == 0
    assert controller_job_exit_code(3) == 0
    assert controller_job_exit_code(75) == 75


@pytest.mark.parametrize(
    "image",
    [
        "registry.example/reprolab/controller:sha-abc123",
        "registry.example/reprolab/controller@sha256:" + "a" * 64,
    ],
)
def test_validate_controller_image_accepts_pinned_reference(image: str):
    assert validate_controller_image(image, env_name="IMAGE") == image


@pytest.mark.parametrize(
    "image", ["", "registry.example/reprolab/controller", "controller:latest"]
)
def test_validate_controller_image_rejects_floating_reference(image: str):
    with pytest.raises(RuntimeError, match="pinned image"):
        validate_controller_image(image, env_name="IMAGE")


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
