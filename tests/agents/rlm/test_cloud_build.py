"""Unit tests for cloud_build — the Cloud Build (gcloud) client for build-on-miss
GKE images. Every subprocess boundary is an injected fake `runner`; no real
gcloud/subprocess/network call is ever made.
"""
from __future__ import annotations

from pathlib import Path

from backend.agents.rlm.cloud_build import BuildResult, image_exists, submit_build


def _make_fake_runner(rc: int, out: str = "", err: str = ""):
    calls: list[list[str]] = []

    def fake_runner(argv, timeout):
        calls.append(list(argv))
        return rc, out, err

    return fake_runner, calls


def _make_raising_runner(exc: Exception):
    calls: list[list[str]] = []

    def fake_runner(argv, timeout):
        calls.append(list(argv))
        raise exc

    return fake_runner, calls


def test_image_exists_true_on_rc0():
    fake_runner, calls = _make_fake_runner(0, out="digest", err="")
    result = image_exists("reg/env-verl:abc123", runner=fake_runner)
    assert result is True
    assert len(calls) == 1
    argv = calls[0]
    assert argv[:5] == ["gcloud", "artifacts", "docker", "images", "describe"]
    assert "reg/env-verl:abc123" in argv


def test_image_exists_false_on_rc1():
    fake_runner, _calls = _make_fake_runner(1, out="", err="NOT_FOUND")
    result = image_exists("reg/env-verl:abc123", runner=fake_runner)
    assert result is False


def test_image_exists_false_when_runner_raises():
    fake_runner, _calls = _make_raising_runner(RuntimeError("boom"))
    result = image_exists("reg/env-verl:abc123", runner=fake_runner)
    assert result is False


def test_submit_build_ok_on_rc0(tmp_path: Path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    fake_runner, calls = _make_fake_runner(0, out="built", err="")

    result = submit_build(tmp_path, "reg/env-verl:abc123", project="P", runner=fake_runner)

    assert isinstance(result, BuildResult)
    assert result.ok is True
    assert result.image_ref == "reg/env-verl:abc123"
    assert len(calls) == 1
    argv = calls[0]
    assert "--project" in argv and argv[argv.index("--project") + 1] == "P"
    assert "--tag" in argv and argv[argv.index("--tag") + 1] == "reg/env-verl:abc123"
    assert "--machine-type" in argv and argv[argv.index("--machine-type") + 1] == "E2_HIGHCPU_8"
    assert any(a.startswith("--timeout=") and a.endswith("s") for a in argv)
    assert str(tmp_path) in argv


def test_submit_build_no_dockerfile_never_calls_runner(tmp_path: Path):
    fake_runner, calls = _make_fake_runner(0)

    result = submit_build(tmp_path, "reg/env-verl:abc123", project="P", runner=fake_runner)

    assert result.ok is False
    assert "no Dockerfile" in result.error
    assert calls == []


def test_submit_build_nonzero_exit_reports_error_and_log_tail(tmp_path: Path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    fake_runner, _calls = _make_fake_runner(1, out="", err="boom stderr")

    result = submit_build(tmp_path, "reg/env-verl:abc123", project="P", runner=fake_runner)

    assert result.ok is False
    assert "exited 1" in result.error
    assert "boom stderr" in result.log_tail


def test_submit_build_runner_raises_reports_failed_to_launch(tmp_path: Path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    fake_runner, _calls = _make_raising_runner(RuntimeError("no gcloud binary"))

    result = submit_build(tmp_path, "reg/env-verl:abc123", project="P", runner=fake_runner)

    assert result.ok is False
    assert "failed to launch" in result.error


def test_submit_build_custom_machine_type_and_timeout_threaded_into_argv(tmp_path: Path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    fake_runner, calls = _make_fake_runner(0)

    submit_build(
        tmp_path,
        "reg/env-verl:abc123",
        project="P",
        machine_type="E2_HIGHCPU_32",
        timeout_s=1200,
        runner=fake_runner,
    )

    argv = calls[0]
    assert "--machine-type" in argv and argv[argv.index("--machine-type") + 1] == "E2_HIGHCPU_32"
    assert "--timeout=1200s" in argv
