"""BUG-NEW-042: deterministic Dockerfile shape guard.

`_validate_dockerfile_shape` rejects sub-agent prose dumped in place of a
Dockerfile; `build_environment` fails fast with failure_class=dockerfile_invalid
(repairable) before wasting a `docker build`.
"""
from types import SimpleNamespace

import pytest

from backend.agents.rlm import primitives
from backend.agents.rlm.primitives import _validate_dockerfile_shape


@pytest.mark.parametrize(
    "text,expected",
    [
        ("FROM python:3.11-slim", True),
        ("from python:3.11", True),  # docker is case-insensitive for instructions
        ("ARG BASE=runpod/pytorch\nFROM ${BASE}", True),
        ("# syntax=docker/dockerfile:1\nFROM scratch", True),
        ("# a plain comment\n\nFROM ubuntu:22.04", True),
        ("   \n\nFROM ubuntu", True),  # leading blank lines skipped
        ("Here is the Dockerfile you asked for:\n\nFROM ubuntu", False),  # prose
        ("I cannot complete this task.", False),
        ("```dockerfile\nFROM ubuntu\n```", False),  # markdown fence is prose
        ("", False),
        ("# only comments\n# nothing else", False),
    ],
)
def test_validate_dockerfile_shape(text, expected):
    assert _validate_dockerfile_shape(text) is expected


def test_build_environment_fails_fast_on_prose():
    ctx = SimpleNamespace(sandbox_mode="docker")
    res = primitives.build_environment(
        {"dockerfile": "Sure! Here is your Dockerfile:\n\nIt installs torch."},
        ctx=ctx,
    )
    assert res["ok"] is False
    assert res["failure_class"] == "dockerfile_invalid"
    assert res["error_code"] == "dockerfile_shape_guard"
    assert res["attempts"] == 0  # no build was attempted


def test_dockerfile_invalid_is_repairable():
    assert "dockerfile_invalid" in primitives._RUN_EXPERIMENT_REPAIRABLE_FAILURES


# --- BUG-NEW-046: _normalize_runpod_from_line ---


_DEFAULT_BASE = "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel"


class TestNormalizeRunpodFromLine:
    """Validate that hallucinated runpod/ image tags are replaced with a valid default."""

    def test_hallucinated_tag_is_replaced(self) -> None:
        from backend.agents.rlm.primitives import _normalize_runpod_from_line
        dockerfile = "FROM runpod/pytorch:1.12.1\nRUN pip install numpy\n"
        result = _normalize_runpod_from_line(dockerfile)
        assert _DEFAULT_BASE in result
        assert "runpod/pytorch:1.12.1" not in result

    def test_non_runpod_image_untouched(self) -> None:
        from backend.agents.rlm.primitives import _normalize_runpod_from_line
        dockerfile = "FROM python:3.11-slim\nRUN pip install numpy\n"
        result = _normalize_runpod_from_line(dockerfile)
        assert result == dockerfile

    def test_from_with_as_stage(self) -> None:
        from backend.agents.rlm.primitives import _normalize_runpod_from_line
        dockerfile = "FROM runpod/pytorch:wrong AS builder\nRUN pip install numpy\n"
        result = _normalize_runpod_from_line(dockerfile)
        assert f"{_DEFAULT_BASE} AS builder" in result

    def test_arg_before_from(self) -> None:
        from backend.agents.rlm.primitives import _normalize_runpod_from_line
        dockerfile = "ARG BASE_TAG=latest\nFROM runpod/pytorch:bad-tag\nRUN echo hi\n"
        result = _normalize_runpod_from_line(dockerfile)
        assert _DEFAULT_BASE in result
