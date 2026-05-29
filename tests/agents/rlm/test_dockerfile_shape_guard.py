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
