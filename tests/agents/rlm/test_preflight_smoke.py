"""Phase 2B — preflight IMPORT smoke (executing TDD before the GPU run).

Runs the REAL emitted, self-contained smoke script via a subprocess (no sandbox
needed) and pins: it passes when every dependency resolves, fails (exit 3) on a
missing dependency, ignores the agent's own local modules, and — critically —
never EXECUTES the agent's modules (only AST-probes their imported deps), so a
missing dep is caught without any "training on import" side effect.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from backend.agents.rlm import preflight_smoke


def _emit_and_run(code_dir: Path) -> subprocess.CompletedProcess:
    script = preflight_smoke.emit(code_dir)
    assert script.exists()
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=str(code_dir), capture_output=True, text=True, timeout=120,
    )


def _result(code_dir: Path) -> dict:
    return json.loads((code_dir / "preflight_smoke_result.json").read_text(encoding="utf-8"))


def test_smoke_passes_when_all_imports_resolve(tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "train.py").write_text("import json\nimport os\nfrom pathlib import Path\n", encoding="utf-8")
    proc = _emit_and_run(code)
    assert proc.returncode == 0, proc.stderr
    res = _result(code)
    assert res["ok"] is True
    assert res["failures"] == []


def test_smoke_fails_on_missing_dependency_without_running_agent_code(tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    # train.py imports a guaranteed-missing dep AND has an import-time side effect.
    # The smoke must (a) catch the missing dep via AST probe -> exit 3, and (b) NEVER
    # execute train.py's body -> the side-effect file must NOT appear.
    (code / "train.py").write_text(
        "import definitely_not_a_real_pkg_xyz123\n"
        "open('SIDE_EFFECT_RAN', 'w').write('x')\n",
        encoding="utf-8",
    )
    proc = _emit_and_run(code)
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    res = _result(code)
    assert res["ok"] is False
    assert any(f["module"] == "definitely_not_a_real_pkg_xyz123" for f in res["failures"])
    assert any(f["error_type"] in ("ModuleNotFoundError", "ImportError") for f in res["failures"])
    # The agent's module body was NOT executed (no training on import).
    assert not (code / "SIDE_EFFECT_RAN").exists()


def test_smoke_ignores_local_sibling_modules(tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    # train.py imports a LOCAL sibling (sdar_env_base) — the smoke must not probe it as
    # a third-party dep (and so must not try to import the agent's code).
    (code / "sdar_env_base.py").write_text("RAISES = 1 / 0\n", encoding="utf-8")  # would crash if imported
    (code / "train.py").write_text("import sdar_env_base\nimport json\n", encoding="utf-8")
    proc = _emit_and_run(code)
    assert proc.returncode == 0, proc.stderr  # local module skipped → no crash
    res = _result(code)
    assert "sdar_env_base" not in res["probed"]


def test_smoke_skips_relative_imports(tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    pkg = code / "sdar"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (code / "train.py").write_text("from . import sdar\nfrom .sdar import envs\nimport json\n", encoding="utf-8")
    proc = _emit_and_run(code)
    assert proc.returncode == 0, proc.stderr


def test_is_enabled_reads_flag(monkeypatch):
    monkeypatch.delenv("REPROLAB_PREFLIGHT_SMOKE", raising=False)
    assert preflight_smoke.is_enabled() is False
    for v in ("1", "true", "yes", "on", "ON"):
        monkeypatch.setenv("REPROLAB_PREFLIGHT_SMOKE", v)
        assert preflight_smoke.is_enabled() is True
    monkeypatch.setenv("REPROLAB_PREFLIGHT_SMOKE", "0")
    assert preflight_smoke.is_enabled() is False


def test_smoke_command_carries_marker(tmp_path: Path):
    cmd = preflight_smoke.smoke_command(tmp_path / "code")
    assert preflight_smoke.MARKER in cmd
    assert "_preflight_smoke.py" in cmd
    assert 'CUDA_VISIBLE_DEVICES=""' in cmd  # GPU hidden for the probe
