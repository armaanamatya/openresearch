"""Regression guard for the SDAR/GCP asset CLI's standalone runnability.

`scripts/sdar_gcp_assets.py` is invoked on the VM as a plain script
(`.venv/bin/python scripts/sdar_gcp_assets.py ...`) where the repo is NOT
pip-installed and PYTHONPATH is unset. Python then puts ``scripts/`` on
``sys.path[0]`` — never the repo root — so the script's lazy
``from backend... import`` calls raise ``ModuleNotFoundError: No module named
'backend'`` unless the script bootstraps the repo root onto ``sys.path`` itself.

This reproduced a live preflight failure on the GCP A100 VM (the unit tests for
``asset_provisioning`` never caught it because pytest sets ``pythonpath=["."]``).
The guard runs the script the way the VM does and asserts the import resolves.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "sdar_gcp_assets.py"


def test_cli_runs_standalone_without_backend_import_error(tmp_path):
    """Run the CLI as a bare script from outside the repo with no PYTHONPATH.

    The only way ``import backend`` can succeed here is the script's own
    ``sys.path`` bootstrap, so this fails loudly if that bootstrap regresses.
    """
    assert SCRIPT.exists(), SCRIPT

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--skip-models", "--allow-missing-webshop"],
        cwd=str(tmp_path),  # cwd != repo root, so CWD can't satisfy `import backend`
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    combined = result.stdout + result.stderr

    # The specific regression: the lazy backend import must resolve.
    assert "No module named 'backend'" not in combined, combined
    # And the script must have actually reached main() and emitted check output
    # (proving the import resolved rather than the process dying at import time).
    assert any(tok in combined for tok in ("[OK] python", "[GREEN]", "[RED]")), combined
