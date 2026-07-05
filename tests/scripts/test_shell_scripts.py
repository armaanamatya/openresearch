"""Hermetic shell-script sanity: syntax checks + watchdog-invariant pins.

No network, no root, no subprocess side effects beyond `bash -n` (a pure parse
check — it never executes the scripts). See the 2026-07-04 GPU-idle-watchdog
incident: a finished SDAR training run idled a 4xA100 GCP VM for ~2.7h because
the watchdog treated process liveness as activity. These pins keep the fix
(evidence-based activity, not process liveness) from silently regressing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SHELL_SCRIPTS = sorted(SCRIPTS_DIR.glob("*.sh"))

PREFLIGHT_SCRIPT = SCRIPTS_DIR / "gcp_sdar_preflight.sh"
AUTHORS_REPRO_SCRIPT = SCRIPTS_DIR / "sdar_authors_repro.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")
@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax_check(script: Path) -> None:
    """Every scripts/*.sh must parse cleanly under `bash -n` (no execution)."""
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{script.name}: bash -n failed\nstdout={result.stdout}\nstderr={result.stderr}"
    )


# -- gcp_sdar_preflight.sh: WATCHDOG_INSTALL heredoc invariants --------------
# A merely-alive process is NOT activity: a hung teardown (e.g. wandb) can keep
# the driver process alive long after training has finished, so the watchdog
# must never treat process liveness alone as evidence of work.


def test_activity_evidence_function_present() -> None:
    assert "activity_evidence" in PREFLIGHT_SCRIPT.read_text()


def test_pgrep_liveness_clause_removed() -> None:
    stale_pattern = "pgrep -f 'backend.cli reproduce' >/dev/null 2>&1 && return 0"
    assert stale_pattern not in PREFLIGHT_SCRIPT.read_text(), (
        "process-liveness-as-activity clause must not reappear in the "
        "watchdog — a merely-alive process is not evidence of work"
    )


def test_run_sentinel_path_present() -> None:
    assert "/run/sdar_run_exited" in PREFLIGHT_SCRIPT.read_text()


def test_activity_dirs_env_var_present() -> None:
    assert "SDAR_ACTIVITY_DIRS" in PREFLIGHT_SCRIPT.read_text()


# -- sdar_authors_repro.sh: heartbeat removal + sentinel write ---------------
# The heartbeat `touch`ed a file whose mtime the watchdog never read (the
# watchdog reads the file's CONTENT via `cat`) — a convention mismatch that
# made the heartbeat provably ineffective, so it was removed outright.


def test_heartbeat_machinery_removed() -> None:
    text = AUTHORS_REPRO_SCRIPT.read_text()
    assert "heartbeat_start" not in text
    assert "heartbeat_stop" not in text


def test_run_exited_sentinel_present() -> None:
    assert "/run/sdar_run_exited" in AUTHORS_REPRO_SCRIPT.read_text()
