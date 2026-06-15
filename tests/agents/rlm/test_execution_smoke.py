"""Phase 2B — preflight EXECUTION smoke (1-step dry-run before the GPU run).

String-level pins on the emitted sandbox command + the exit-code interpretation
(mirroring ``test_preflight_smoke`` — no real subprocess needed, since the command
runs the agent's OWN entry script inside the sandbox). Confirms the command caps the
run to one step, forces synchronous CUDA so a device-side assert surfaces at the real
line, wraps the entry script in ``timeout`` so a non-honoring script is killed rather
than hung, and that ``interpret_exit`` blocks only a real crash (not a timeout-kill).
"""
from __future__ import annotations

from pathlib import Path

from backend.agents.rlm import execution_smoke


def test_is_enabled_reads_flag(monkeypatch):
    # Opt-in (default OFF): unset → disabled; enabled only on explicit truthy.
    monkeypatch.delenv("REPROLAB_EXECUTION_SMOKE", raising=False)
    assert execution_smoke.is_enabled() is False
    for v in ("1", "true", "yes", "on", "ON"):
        monkeypatch.setenv("REPROLAB_EXECUTION_SMOKE", v)
        assert execution_smoke.is_enabled() is True
    monkeypatch.setenv("REPROLAB_EXECUTION_SMOKE", "0")
    assert execution_smoke.is_enabled() is False


def test_smoke_command_carries_marker_and_entry_script(tmp_path: Path):
    cmd = execution_smoke.smoke_command(tmp_path / "code")
    assert execution_smoke.MARKER in cmd
    assert "train.py" in cmd  # default entry script
    # cd's into the code dir.
    assert f'cd "{tmp_path / "code"}"' in cmd
    # Resolves an interpreter the same way preflight_smoke does.
    assert "command -v python3 || command -v python" in cmd


def test_smoke_command_forces_synchronous_cuda(tmp_path: Path):
    cmd = execution_smoke.smoke_command(tmp_path / "code")
    # CUDA_LAUNCH_BLOCKING=1 is the whole point — surfaces the async device-side assert
    # at the real line in seconds.
    assert "CUDA_LAUNCH_BLOCKING=1" in cmd


def test_smoke_command_caps_steps(tmp_path: Path):
    cmd = execution_smoke.smoke_command(tmp_path / "code")
    assert "REPROLAB_SMOKE_STEPS=1" in cmd
    assert execution_smoke.SMOKE_STEPS_ENV == "REPROLAB_SMOKE_STEPS"
    # A custom step count is honored.
    cmd5 = execution_smoke.smoke_command(tmp_path / "code", steps=5)
    assert "REPROLAB_SMOKE_STEPS=5" in cmd5


def test_smoke_command_wraps_in_timeout(tmp_path: Path):
    cmd = execution_smoke.smoke_command(tmp_path / "code")
    # Default timeout wraps the entry script so a full-run-attempting script is killed.
    assert "timeout 300" in cmd
    # Custom timeout is honored.
    cmd_custom = execution_smoke.smoke_command(tmp_path / "code", timeout_s=60)
    assert "timeout 60" in cmd_custom


def test_smoke_command_honors_custom_entry_script(tmp_path: Path):
    cmd = execution_smoke.smoke_command(tmp_path / "code", entry_script="run_baseline.py")
    assert "run_baseline.py" in cmd
    assert "train.py" not in cmd


def test_smoke_command_quoting_shape_matches_preflight(tmp_path: Path):
    # Same sh -c '...' single-quote shell wrapper as preflight_smoke.smoke_command, so
    # the wiring can treat both identically.
    cmd = execution_smoke.smoke_command(tmp_path / "code")
    assert cmd.startswith("sh -c '")
    assert cmd.rstrip().endswith(execution_smoke.MARKER)


def test_interpret_exit_ok_is_non_blocking():
    status, blocking = execution_smoke.interpret_exit(0)
    assert status == "ok"
    assert blocking is False


def test_interpret_exit_timeout_is_not_honored_non_blocking():
    # 124 == timeout-kill: the script ignored the step cap and ran long. NOT broken →
    # soft pass, skip, do NOT block the training run.
    status, blocking = execution_smoke.interpret_exit(124)
    assert status == "not_honored"
    assert blocking is False


def test_interpret_exit_other_nonzero_is_blocking_crash():
    # Any non-zero that isn't a timeout is a real entry-script crash (the device-side
    # assert) → blocking.
    for code in (1, 2, 3, 134, 139, 255):
        status, blocking = execution_smoke.interpret_exit(code)
        assert status == "crash", code
        assert blocking is True, code


def test_public_api_surface():
    assert set(execution_smoke.__all__) == {
        "MARKER",
        "SMOKE_STEPS_ENV",
        "is_enabled",
        "smoke_command",
        "interpret_exit",
    }
    assert execution_smoke.MARKER == "# reprolab:execution-smoke"
