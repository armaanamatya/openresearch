"""BUG-NEW-042: _validate_dockerfile_shape shape guard."""

from __future__ import annotations

import pytest

from backend.agents.rlm.primitives import _validate_dockerfile_shape


@pytest.mark.parametrize(
    "text",
    [
        "FROM python:3.11\nRUN pip install torch\n",
        "ARG BASE=python:3.11\nFROM ${BASE}\n",
        "# syntax=docker/dockerfile:1\nFROM python:3.11\n",
        "# Comment first\n# Another comment\nFROM python:3.11\n",
        "   \n\n  FROM python:3.11\n",
    ],
)
def test_valid_dockerfile_shape(text: str) -> None:
    ok, reason = _validate_dockerfile_shape(text)
    assert ok, reason


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   \n\n  \t  ",
        "You've already built the environment so this is just documentation.\n",
        "Here is the Dockerfile content:\nFROM python:3.11\n",  # prose first
        "# only comments\n# nothing else\n",
        "RUN pip install foo\nFROM python:3.11\n",  # RUN before FROM
    ],
)
def test_invalid_dockerfile_shape(text: str) -> None:
    ok, reason = _validate_dockerfile_shape(text)
    assert not ok
    assert reason is not None and len(reason) > 0


# --- BUG-NEW-046: _normalize_runpod_from_line ---


class TestNormalizeRunpodFromLine:
    """Validate that hallucinated runpod/ image tags are replaced."""

    def test_hallucinated_tag_is_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.agents.rlm.primitives import _normalize_runpod_from_line
        monkeypatch.setattr(
            "backend.config.get_settings",
            lambda: type("S", (), {"runpod_image": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"})(),
        )
        dockerfile = "FROM runpod/pytorch:1.12.1\nRUN pip install numpy\n"
        result = _normalize_runpod_from_line(dockerfile)
        assert "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" in result
        assert "runpod/pytorch:1.12.1" not in result

    def test_correct_tag_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.agents.rlm.primitives import _normalize_runpod_from_line
        configured = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
        monkeypatch.setattr(
            "backend.config.get_settings",
            lambda: type("S", (), {"runpod_image": configured})(),
        )
        dockerfile = f"FROM {configured}\nRUN pip install numpy\n"
        result = _normalize_runpod_from_line(dockerfile)
        assert result == dockerfile

    def test_non_runpod_image_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.agents.rlm.primitives import _normalize_runpod_from_line
        monkeypatch.setattr(
            "backend.config.get_settings",
            lambda: type("S", (), {"runpod_image": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"})(),
        )
        dockerfile = "FROM python:3.11-slim\nRUN pip install numpy\n"
        result = _normalize_runpod_from_line(dockerfile)
        assert result == dockerfile

    def test_from_with_as_stage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.agents.rlm.primitives import _normalize_runpod_from_line
        monkeypatch.setattr(
            "backend.config.get_settings",
            lambda: type("S", (), {"runpod_image": "runpod/pytorch:2.1.0-correct"})(),
        )
        dockerfile = "FROM runpod/pytorch:wrong AS builder\nRUN pip install numpy\n"
        result = _normalize_runpod_from_line(dockerfile)
        assert "runpod/pytorch:2.1.0-correct AS builder" in result

    def test_arg_before_from(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.agents.rlm.primitives import _normalize_runpod_from_line
        monkeypatch.setattr(
            "backend.config.get_settings",
            lambda: type("S", (), {"runpod_image": "runpod/pytorch:2.1.0-correct"})(),
        )
        dockerfile = "ARG BASE_TAG=latest\nFROM runpod/pytorch:bad-tag\nRUN echo hi\n"
        result = _normalize_runpod_from_line(dockerfile)
        assert "runpod/pytorch:2.1.0-correct" in result
