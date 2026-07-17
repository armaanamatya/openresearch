"""In-Pod controller heartbeat loop (WS3).

The heartbeat is what makes ``sweep_orphaned_controllers`` safe: a live
controller renews its lease on a cadence, so an expired lease reliably means
the controller is gone. Tested here with a fake lease + injected clock/sleep —
no threads, no time.time().
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.rlm import controller_entry as ce


class _FakeLease:
    def __init__(self, renew_results):
        self._renew_results = list(renew_results)
        self.renew_calls = 0

    def renew(self, token, now_epoch):
        self.renew_calls += 1
        return self._renew_results.pop(0)


def test_heartbeat_renews_until_stopped():
    lease = _FakeLease(renew_results=["t1", "t2", "t3"])

    def is_running():
        return lease.renew_calls < 2   # stop once we have renewed twice

    lost = []
    ce.heartbeat_loop(
        lease=lease, token="t0", interval_s=60,
        sleep=lambda s: None, clock=lambda: 1000.0,
        is_running=is_running, on_lost=lost.append,
    )
    assert lease.renew_calls == 2
    assert lost == []   # stopped cleanly, never lost the lease


def test_heartbeat_stops_and_signals_when_superseded():
    lease = _FakeLease(renew_results=[None])   # first renew fails → superseded
    lost = []
    ce.heartbeat_loop(
        lease=lease, token="t0", interval_s=60,
        sleep=lambda s: None, clock=lambda: 1000.0,
        is_running=lambda: True, on_lost=lambda: lost.append("lost"),
    )
    assert lease.renew_calls == 1
    assert lost == ["lost"]   # a superseded controller must be told to stop


class _DriveLease:
    def __init__(self, *, renew_result: object = "renewed") -> None:
        self.renew_result = renew_result
        self.acquire_calls: list[tuple[str, str, float]] = []
        self.renew_calls = 0

    def acquire(self, project_id, owner_id, now_epoch):
        self.acquire_calls.append((project_id, owner_id, now_epoch))
        return SimpleNamespace(fence_epoch=7)

    def renew(self, token, now_epoch):
        self.renew_calls += 1
        return self.renew_result

    def is_current(self, token):
        return self.renew_result is not None


class _DriveProcess:
    def __init__(self, waits: list[object]) -> None:
        self._waits = list(waits)
        self.pid = 99999999
        self.terminated = False

    def wait(self, timeout=None):
        result = self._waits.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


def test_drive_controller_uses_unique_owner_reaps_and_heartbeats() -> None:
    lease = _DriveLease()
    process = _DriveProcess([subprocess.TimeoutExpired("campaign", 60), 0])
    spawn: dict[str, object] = {}
    acquired: list[object] = []
    checkpoints: list[str] = []

    def popen(command, **kwargs):
        spawn.update(command=command, **kwargs)
        return process

    result = ce.drive_controller(
        lease=lease,
        project_id="prj_x",
        owner_id="prj_x-launch-123",
        command=["campaign"],
        env={"BASE": "1"},
        heartbeat_s=60,
        clock=lambda: 100.0,
        popen_factory=popen,
        on_acquired=acquired.append,
        checkpoint=lambda: checkpoints.append("saved"),
    )

    assert result == 0
    assert lease.acquire_calls == [("prj_x", "prj_x-launch-123", 100.0)]
    assert lease.renew_calls == 1
    assert len(acquired) == 1
    assert checkpoints == ["saved", "saved"]
    assert spawn["env"]["OPENRESEARCH_CELL_FENCE_EPOCH"] == "7"


def test_drive_controller_fails_closed_when_lease_is_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _DriveLease(renew_result=None)
    process = _DriveProcess([subprocess.TimeoutExpired("campaign", 60), 0])
    monkeypatch.setattr(ce, "_terminate_process_group", lambda child: child.terminate())
    checkpoints: list[str] = []

    result = ce.drive_controller(
        lease=lease,
        project_id="prj_x",
        owner_id="owner-x",
        command=["campaign"],
        env={},
        heartbeat_s=60,
        clock=lambda: 100.0,
        popen_factory=lambda *args, **kwargs: process,
        checkpoint=lambda: checkpoints.append("stale"),
    )

    assert result == ce.LEASE_LOST_EXIT
    assert process.terminated is True
    assert checkpoints == []


def test_campaign_command_requires_budgets_and_uses_cloud_sandbox() -> None:
    env = {
        "OPENRESEARCH_CAMPAIGN_MAX_LLM_USD": "20",
        "OPENRESEARCH_CAMPAIGN_MAX_GPU_USD": "40",
        "OPENRESEARCH_CAMPAIGN_MAX_GPU_HOURS": "10",
    }
    command = ce._campaign_command(
        paper="/mnt/paper.pdf",
        project_id="prj_x",
        cloud="azure",
        runs_root=Path("/mnt/runs"),
        run_spec="/mnt/spec.json",
        root_model="opus-foundry",
        execution_mode="efficient",
        gpu_mode="prefer",
        minimize_compute=True,
        env=env,
    )
    assert command[:6] == [
        ce.sys.executable,
        "-m",
        "backend.cli",
        "--runs-root",
        "/mnt/runs",
        "campaign",
    ]
    assert command[6] == "/mnt/paper.pdf"
    assert command[7:13] == [
        "--max-llm-usd", "20", "--max-gpu-usd", "40", "--max-gpu-hours", "10"
    ]
    assert command[command.index("--execution-mode") + 1] == "efficient"
    assert command[command.index("--gpu-mode") + 1] == "prefer"
    assert "--minimize-compute" in command
    assert command[-4:] == [
        "--run-spec", "/mnt/spec.json", "--root-model", "opus-foundry"
    ]

    with pytest.raises(RuntimeError, match="MAX_LLM_USD"):
        ce._campaign_command(
            paper="paper.pdf",
            project_id="prj_x",
            cloud="gcp",
            runs_root=Path("/mnt/runs"),
            run_spec=None,
            env={},
        )


def test_load_mounted_secrets_does_not_override_existing(tmp_path: Path) -> None:
    (tmp_path / "anthropic-api-key").write_text("mounted-value\n", encoding="utf-8")
    (tmp_path / "azure-openai-api-key").write_text("azure-value", encoding="utf-8")
    (tmp_path / "azure-foundry-api-key").write_text("foundry-value", encoding="utf-8")
    env = {"ANTHROPIC_API_KEY": "existing"}
    ce.load_mounted_secrets(tmp_path, env)
    assert env == {
        "ANTHROPIC_API_KEY": "existing",
        "AZURE_OPENAI_API_KEY": "azure-value",
        "AZURE_FOUNDRY_API_KEY": "foundry-value",
    }


def test_heartbeat_interval_cannot_outlive_lease_safety_margin() -> None:
    assert ce._heartbeat_interval({}) == 60
    assert ce._heartbeat_interval({"OPENRESEARCH_CONTROLLER_HEARTBEAT_S": "30"}) == 30
    with pytest.raises(RuntimeError, match="one third"):
        ce._heartbeat_interval({"OPENRESEARCH_CONTROLLER_HEARTBEAT_S": "61"})
