"""Pin: the SDAR GCP driver scripts must honor an `OPENRESEARCH_REPRODUCTION_MODE`
environment override, never hardcode a bare `adapt` default.

Context (2026-07-05, B2 wiring): `scripts/gcp_sdar_preflight.sh` builds a
`.cache` run-spec that is consumed by `scripts/sdar_gcp_run.sh`. Both must
default to `adapt` only via the `${OPENRESEARCH_REPRODUCTION_MODE:-adapt}`
shell-parameter-expansion form (env wins), never via a bare
`OPENRESEARCH_REPRODUCTION_MODE=adapt`/`="adapt"` assignment that would
silently clobber an operator's `execute`/`reference` override. No network, no
execution beyond `bash -n` (a pure parse check).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

PREFLIGHT_SCRIPT = SCRIPTS_DIR / "gcp_sdar_preflight.sh"
RUN_SCRIPT = SCRIPTS_DIR / "sdar_gcp_run.sh"

# A "bare" hardcode: `OPENRESEARCH_REPRODUCTION_MODE` followed by `=`, an
# optional quote, the literal `adapt`, an optional quote, end of line. Deliberately
# does NOT match the env-override form `="${OPENRESEARCH_REPRODUCTION_MODE:-adapt}"`
# (the character right after `=` there is `"` immediately followed by `$`, not
# the literal `adapt`), and does NOT match the `_spec_add NAME VALUE`
# function-call form (no `=` between name and value at all).
_BARE_ADAPT_DEFAULT = re.compile(
    r'OPENRESEARCH_REPRODUCTION_MODE\s*=\s*["\']?adapt["\']?\s*$', re.MULTILINE
)


def _assert_no_bare_adapt_default(script: Path) -> None:
    text = script.read_text()
    match = _BARE_ADAPT_DEFAULT.search(text)
    assert match is None, (
        f"{script.name}: hardcodes a bare 'adapt' default "
        f"({match.group(0)!r}) instead of honoring an "
        f"OPENRESEARCH_REPRODUCTION_MODE environment override"
    )


def test_preflight_does_not_hardcode_adapt_default_only() -> None:
    _assert_no_bare_adapt_default(PREFLIGHT_SCRIPT)


def test_run_script_does_not_hardcode_adapt_default_only() -> None:
    _assert_no_bare_adapt_default(RUN_SCRIPT)


def test_preflight_forwards_env_override_form() -> None:
    """The preflight's run-spec entry must use the `${VAR:-adapt}` fallback form."""
    text = PREFLIGHT_SCRIPT.read_text()
    assert "${OPENRESEARCH_REPRODUCTION_MODE:-adapt}" in text, (
        "preflight must forward OPENRESEARCH_REPRODUCTION_MODE via the "
        "env-override fallback form, e.g. ${OPENRESEARCH_REPRODUCTION_MODE:-adapt}"
    )


def test_run_script_forwards_env_override_form() -> None:
    """The run script's own export must use the `${VAR:-adapt}` fallback form."""
    text = RUN_SCRIPT.read_text()
    assert "${OPENRESEARCH_REPRODUCTION_MODE:-adapt}" in text, (
        "sdar_gcp_run.sh must export OPENRESEARCH_REPRODUCTION_MODE via the "
        "env-override fallback form, e.g. ${OPENRESEARCH_REPRODUCTION_MODE:-adapt}"
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")
@pytest.mark.parametrize("script", [PREFLIGHT_SCRIPT, RUN_SCRIPT], ids=lambda p: p.name)
def test_bash_syntax_check(script: Path) -> None:
    """Both scripts must still parse cleanly under `bash -n` (no execution)."""
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{script.name}: bash -n failed\nstdout={result.stdout}\nstderr={result.stderr}"
    )
