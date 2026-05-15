"""Track 4 — environment build-and-repair loop contract tests.

Covers:
  A. build_image — success, BuildError, infrastructure failure
  B. _should_rebuild — pure function truth table
  C. _run_environment_build_loop — orchestrator integration via monkeypatching
  D. checkpoint round-trip for the three new PipelineState fields
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.agents.orchestrator import (
    PipelineState,
    ReproLabOrchestrator,
    _should_rebuild,
)
from backend.agents.schemas import EnvironmentSpec
from backend.services.runtime import SandboxRuntimeError, build_image
from backend.services.runtime.interface import RuntimeCauseKind


# ---------------------------------------------------------------------------
# A. build_image — inject a fake docker client
# ---------------------------------------------------------------------------


def _make_dockerfile(tmp_path: Path) -> Path:
    """Write a minimal Dockerfile to tmp_path and return its path."""
    df = tmp_path / "Dockerfile"
    df.write_text("FROM scratch\n")
    return df


def test_build_image_success(tmp_path: Path) -> None:
    """Fake client returns (image_obj, []) — build_image returns (True, tag, '')."""
    fake_client = MagicMock()
    fake_client.images.build.return_value = (object(), [])
    dockerfile = _make_dockerfile(tmp_path)

    ok, tag, err = asyncio.run(
        build_image(dockerfile, tmp_path, "myapp:test", client=fake_client)
    )

    assert ok is True
    assert tag == "myapp:test"
    assert err == ""
    fake_client.images.build.assert_called_once()


def test_build_image_broken_dockerfile(tmp_path: Path) -> None:
    """Fake client raises docker BuildError — returns (False, tag, non-empty err)."""
    import docker.errors

    build_log = [{"stream": "Step 1/2 : FROM badimage"}, {"error": "unknown base image"}]
    exc = docker.errors.BuildError("unknown base image", build_log)
    fake_client = MagicMock()
    fake_client.images.build.side_effect = exc
    dockerfile = _make_dockerfile(tmp_path)

    ok, tag, err = asyncio.run(
        build_image(dockerfile, tmp_path, "bad:image", client=fake_client)
    )

    assert ok is False
    assert tag == "bad:image"
    assert err  # non-empty
    assert "unknown base image" in err


def test_build_image_infrastructure_failure_raises(tmp_path: Path) -> None:
    """A plain RuntimeError (daemon unreachable) escalates to SandboxRuntimeError."""
    fake_client = MagicMock()
    fake_client.images.build.side_effect = RuntimeError("daemon unreachable")
    dockerfile = _make_dockerfile(tmp_path)

    with pytest.raises(SandboxRuntimeError):
        asyncio.run(
            build_image(dockerfile, tmp_path, "any:tag", client=fake_client)
        )


# ---------------------------------------------------------------------------
# B. _should_rebuild — pure function truth table
# ---------------------------------------------------------------------------


def test_should_rebuild_returns_true_when_not_ok_and_under_cap() -> None:
    assert _should_rebuild(False, 0, 3) is True


def test_should_rebuild_returns_false_when_ok() -> None:
    assert _should_rebuild(True, 0, 3) is False


def test_should_rebuild_returns_false_when_at_cap() -> None:
    assert _should_rebuild(False, 3, 3) is False


def test_should_rebuild_returns_true_when_one_attempt_remains() -> None:
    assert _should_rebuild(False, 2, 3) is True


# ---------------------------------------------------------------------------
# C. _run_environment_build_loop — orchestrator with monkeypatched build_image
# ---------------------------------------------------------------------------


def _make_orchestrator(tmp_path: Path, sandbox_mode: str = "docker") -> ReproLabOrchestrator:
    """Construct a ReproLabOrchestrator using the FakeRuntime pattern from
    test_agent_runtime_orchestrator.py — no real provider credentials needed."""
    from tests.test_agent_runtime_orchestrator import FakeRuntime

    runtime = FakeRuntime("openai", "{}")
    return ReproLabOrchestrator(
        "prj_track4",
        tmp_path,
        runtime=runtime,
        sandbox_mode=sandbox_mode,
    )


def _make_state_with_dockerfile(orchestrator: ReproLabOrchestrator) -> PipelineState:
    """Build a PipelineState whose environment_spec carries a non-empty dockerfile."""
    state = PipelineState(project_id=orchestrator.project_id)
    state.environment_spec = EnvironmentSpec(
        dockerfile="FROM python:3.11-slim\nRUN echo hello\n"
    )
    return state


def test_build_loop_fails_twice_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """build_image returns (False,…), (False,…), (True,…) — loop ends with ok=True, 3 attempts."""
    orc = _make_orchestrator(tmp_path)
    state = _make_state_with_dockerfile(orc)
    tag = f"reprolab/{orc.project_id}:env-check"

    call_results = [
        (False, tag, "err1"),
        (False, tag, "err2"),
        (True, tag, ""),
    ]
    call_index = 0

    async def fake_build_image(*_args: Any, **_kwargs: Any) -> tuple[bool, str, str]:
        nonlocal call_index
        result = call_results[call_index]
        call_index += 1
        return result

    repair_calls: list[str] = []

    async def fake_repair(s: PipelineState, error_text: str) -> PipelineState:
        repair_calls.append(error_text)
        return s

    monkeypatch.setattr("backend.services.runtime.build_image", fake_build_image)
    monkeypatch.setattr(orc, "_repair_environment", fake_repair)

    result = asyncio.run(orc._run_environment_build_loop(state))

    assert result.environment_build_ok is True
    assert result.environment_build_attempts == 3
    assert len(repair_calls) == 2


def test_build_loop_fails_to_cap_no_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """build_image always returns False — loop exits fail-soft, no exception, repair 2× (not after final)."""
    orc = _make_orchestrator(tmp_path)
    state = _make_state_with_dockerfile(orc)
    tag = f"reprolab/{orc.project_id}:env-check"

    async def always_fail(*_args: Any, **_kwargs: Any) -> tuple[bool, str, str]:
        return (False, tag, "build failed every time")

    repair_calls: list[str] = []

    async def fake_repair(s: PipelineState, error_text: str) -> PipelineState:
        repair_calls.append(error_text)
        return s

    monkeypatch.setattr("backend.services.runtime.build_image", always_fail)
    monkeypatch.setattr(orc, "_repair_environment", fake_repair)

    # Must not raise.
    result = asyncio.run(orc._run_environment_build_loop(state))

    assert result.environment_build_ok is False
    assert result.environment_build_attempts == 3  # default max_attempts
    # Repair is called after attempt 1 and 2, but NOT after attempt 3 (the cap).
    assert len(repair_calls) == 2


def test_build_loop_idempotent_after_cap_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second call once the cap is spent is a pure no-op — no re-build, no
    duplicate fail-soft signal. Regression test for run()'s resume hook
    re-entering a fresh run that already exhausted the cap."""
    orc = _make_orchestrator(tmp_path)
    state = _make_state_with_dockerfile(orc)
    tag = f"reprolab/{orc.project_id}:env-check"

    build_calls: list[Any] = []

    async def always_fail(*_args: Any, **_kwargs: Any) -> tuple[bool, str, str]:
        build_calls.append(True)
        return (False, tag, "build failed every time")

    async def fake_repair(s: PipelineState, error_text: str) -> PipelineState:
        return s

    monkeypatch.setattr("backend.services.runtime.build_image", always_fail)
    monkeypatch.setattr(orc, "_repair_environment", fake_repair)

    # First call exhausts the cap.
    state = asyncio.run(orc._run_environment_build_loop(state))
    assert state.environment_build_attempts == 3
    first_call_builds = len(build_calls)

    # Second call (as run()'s resume hook would make) must be a pure no-op.
    state = asyncio.run(orc._run_environment_build_loop(state))
    assert state.environment_build_attempts == 3
    assert len(build_calls) == first_call_builds  # build_image NOT called again


def test_build_loop_infrastructure_error_returns_without_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SandboxRuntimeError from build_image → returns immediately, _repair_environment not called."""
    orc = _make_orchestrator(tmp_path)
    state = _make_state_with_dockerfile(orc)

    async def infra_fail(*_args: Any, **_kwargs: Any) -> tuple[bool, str, str]:
        raise SandboxRuntimeError(RuntimeCauseKind.backend_unavailable, "daemon down")

    repair_calls: list[str] = []

    async def fake_repair(s: PipelineState, error_text: str) -> PipelineState:
        repair_calls.append(error_text)
        return s

    monkeypatch.setattr("backend.services.runtime.build_image", infra_fail)
    monkeypatch.setattr(orc, "_repair_environment", fake_repair)

    # Must not raise.
    result = asyncio.run(orc._run_environment_build_loop(state))

    assert result.environment_build_ok is False
    assert len(repair_calls) == 0


def test_build_loop_non_docker_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """sandbox_mode='local' — loop is a no-op; environment_build_attempts stays 0."""
    orc = _make_orchestrator(tmp_path, sandbox_mode="local")
    state = _make_state_with_dockerfile(orc)

    build_calls: list[Any] = []

    async def should_not_call(*_args: Any, **_kwargs: Any) -> tuple[bool, str, str]:
        build_calls.append(True)
        return (True, "x", "")

    monkeypatch.setattr("backend.services.runtime.build_image", should_not_call)

    result = asyncio.run(orc._run_environment_build_loop(state))

    assert result.environment_build_attempts == 0
    assert len(build_calls) == 0


def test_build_loop_disabled_via_settings_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """environment_build_validation_enabled=False — loop is a no-op."""
    orc = _make_orchestrator(tmp_path)
    state = _make_state_with_dockerfile(orc)

    build_calls: list[Any] = []

    async def should_not_call(*_args: Any, **_kwargs: Any) -> tuple[bool, str, str]:
        build_calls.append(True)
        return (True, "x", "")

    monkeypatch.setattr("backend.services.runtime.build_image", should_not_call)
    monkeypatch.setattr(
        "backend.agents.orchestrator.get_settings",
        lambda: SimpleNamespace(
            environment_build_validation_enabled=False,
            environment_build_max_attempts=3,
        ),
    )

    result = asyncio.run(orc._run_environment_build_loop(state))

    assert result.environment_build_attempts == 0
    assert len(build_calls) == 0


# ---------------------------------------------------------------------------
# D. Checkpoint round-trip for the three new PipelineState fields
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_new_fields_survive(tmp_path: Path) -> None:
    """Non-default values for the three new build fields survive save/load."""
    state = PipelineState(project_id="prj_cp_track4")
    state.environment_build_attempts = 2
    state.environment_build_ok = True
    state.environment_build_error = "some error text"

    state.save_checkpoint(tmp_path)
    loaded = PipelineState.load_checkpoint(tmp_path, "prj_cp_track4")

    assert loaded is not None
    assert loaded.environment_build_attempts == 2
    assert loaded.environment_build_ok is True
    assert loaded.environment_build_error == "some error text"


def test_checkpoint_legacy_missing_fields_load_with_defaults(tmp_path: Path) -> None:
    """A checkpoint dict without the three new keys loads with correct defaults."""
    import json

    # Simulate a legacy checkpoint that predates Track 4
    legacy_data = {
        "project_id": "prj_legacy",
        "stage": "ingested",
        "assumption_ledger": [],
        "decision_log": [],
        "seed": None,
        "attempt_id": None,
        "run_group_id": None,
        "blacklist_terms": [],
        # environment_build_ok / attempts / error intentionally absent
    }
    checkpoint_dir = tmp_path / "prj_legacy"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "pipeline_state.json").write_text(json.dumps(legacy_data))

    loaded = PipelineState.load_checkpoint(tmp_path, "prj_legacy")

    assert loaded is not None
    assert loaded.environment_build_attempts == 0
    assert loaded.environment_build_ok is False
    assert loaded.environment_build_error == ""
