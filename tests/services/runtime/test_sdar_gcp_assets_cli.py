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

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "sdar_gcp_assets.py"


def _load_script_module():
    """Load scripts/sdar_gcp_assets.py as a module (it is not a package)."""
    spec = importlib.util.spec_from_file_location("sdar_gcp_assets_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass resolves cls.__module__ via sys.modules during exec; register first.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Excl:
    def __init__(self, item: str, reason: str):
        self.item, self.reason = item, reason


class _Result:
    def __init__(self, exclusions, env_vars):
        self.exclusions, self.env_vars = exclusions, env_vars
        self.released = False

    def release(self):
        self.released = True


def test_provision_envs_best_effort_skips_webshop_but_gates_required():
    """A best-effort env exclusion is skipped (run proceeds); a required one raises."""
    mod = _load_script_module()

    # Best-effort WebShop excluded → no raise, returns the envs that came up.
    good = _Result([_Excl("WebShop", "server did not become ready")], {"ALFWORLD_DATA": "/x"})
    with (
        patch("backend.services.runtime.env_cache.provision_scope", return_value=good),
        patch("backend.services.runtime.env_cache.EnvCacheManager", MagicMock()),
    ):
        out = mod.provision_envs(["ALFWorld", "WebShop"], best_effort={"webshop"})
    assert out == {"ALFWORLD_DATA": "/x"}
    assert good.released, "the scope lease must always be released"

    # Required ALFWorld excluded → raises (it gates the run).
    bad = _Result([_Excl("ALFWorld", "data download failed")], {})
    with (
        patch("backend.services.runtime.env_cache.provision_scope", return_value=bad),
        patch("backend.services.runtime.env_cache.EnvCacheManager", MagicMock()),
        pytest.raises(RuntimeError, match="required environment provisioning failed"),
    ):
        mod.provision_envs(["ALFWorld", "WebShop"], best_effort={"webshop"})
    assert bad.released


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
