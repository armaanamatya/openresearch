"""Pin the two shell .env parsers to python-dotenv's parse (audit 2026-06-09).

The values these parsers export become process env, which pydantic-settings
ranks ABOVE its own env_file parse — so any divergence silently shadows the
correct value. A Literal-typed field turns the divergence into a hard
ValidationError at boot: .env.example's own suggested header line
``OPENRESEARCH_DEFAULT_SANDBOX=local   # no Docker daemon / RunPod needed``
used to export the comment as part of the value, crash ``Settings()``, and
restart-loop the container.

Two parsers, one source of truth:
  * ``scripts/lib/env_file.sh::env_value_from_file`` — bash-3.2 reimplementation
    used by start.sh (runs before any venv exists).
  * ``docker/load_env.sh::load_env_file`` — delegates to python-dotenv itself
    (the venv is guaranteed in the container).
Both are asserted equal to ``dotenv_values`` on every fixture key.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from dotenv import dotenv_values

REPO = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")

# Every dotenv construct the repo's .env.example / docs actually use, plus the
# corruption cases the 2026-06-09 review confirmed (inline comments, CRLF,
# quoted hashes, export prefix, spaced '=', duplicate keys, '=' in values).
_FIXTURE = (
    "PLAIN=simple\n"
    "INLINE_COMMENT=local   # no Docker daemon / RunPod needed\n"
    "SPACED = spaced-value\n"
    "export EXPORTED=exported-value\n"
    'QUOTED_HASH="NVIDIA # GeForce"\n'
    "SINGLE_HASH='single # quoted'\n"
    'QUOTED_TRAILING="quoted-val"   # trailing comment\n'
    "GPU_TYPE=NVIDIA GeForce RTX 4090  # community tier\n"
    "EQ_IN_VALUE=a=b=c\n"
    "NOHASH=local# not-a-comment\n"
    "DUP=first\n"
    "DUP=last\n"
    "CRLF_KEY=crlf-value\r\n"
    'QUOTED_CRLF="qcrlf"\r\n'
)


@pytest.fixture()
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.env"
    path.write_bytes(_FIXTURE.encode("utf-8"))
    return path


def _expected(env_file: Path) -> dict[str, str]:
    expected = {k: v for k, v in dotenv_values(env_file).items() if v is not None}
    assert len(expected) >= 12, "fixture must exercise every construct"
    return expected


def test_start_sh_parser_matches_python_dotenv(env_file: Path) -> None:
    expected = _expected(env_file)
    for key, value in expected.items():
        proc = subprocess.run(
            [
                BASH,
                "-c",
                'source scripts/lib/env_file.sh && env_value_from_file "$1" "$2"',
                "_",
                key,
                str(env_file),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"{key}: {proc.stderr}"
        assert proc.stdout == value, f"{key}: bash={proc.stdout!r} dotenv={value!r}"


def test_start_sh_parser_absent_key_returns_nonzero(env_file: Path) -> None:
    proc = subprocess.run(
        [
            BASH,
            "-c",
            'source scripts/lib/env_file.sh && env_value_from_file MISSING_KEY "$1"',
            "_",
            str(env_file),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0


def _run_load_env(env_file: Path, *, preset: dict[str, str] | None = None) -> dict[str, str]:
    """Source docker/load_env.sh in a clean bash, load the fixture, dump env."""
    import os
    import sys

    base_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **(preset or {})}
    proc = subprocess.run(
        [
            BASH,
            "-c",
            "set -euo pipefail; source docker/load_env.sh; "
            'load_env_file "$1" "$2"; '
            '"$2" -c "import os, json, sys; sys.stdout.write(json.dumps(dict(os.environ)))"',
            "_",
            str(env_file),
            sys.executable,
        ],
        cwd=REPO,
        env=base_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_entrypoint_loader_matches_python_dotenv(env_file: Path) -> None:
    result = _run_load_env(env_file)
    for key, value in _expected(env_file).items():
        assert result.get(key) == value, (
            f"{key}: exported={result.get(key)!r} dotenv={value!r}"
        )


def test_entrypoint_loader_container_env_wins(env_file: Path) -> None:
    # Compose `environment:` / `docker run -e` precedence: an already-set var
    # must NOT be overwritten by .env (the regression that broke the
    # compose-set OPENRESEARCH_DATABASE_URL).
    result = _run_load_env(env_file, preset={"PLAIN": "from-container"})
    assert result["PLAIN"] == "from-container"
    assert result["INLINE_COMMENT"] == "local"  # other keys still load


def test_entrypoint_loader_missing_file_is_noop(tmp_path: Path) -> None:
    result = _run_load_env(tmp_path / "does-not-exist.env")
    assert "PLAIN" not in result
