"""_remote_command must cd to the REMOTE upload dir, not the orchestrator-host path.

Regression for the 2026-07-07 wedge: run_experiment on RunPod failed with
`sh: cd: can't cd to /Volumes/.../code` because the command wrapper cd'd to
config.workdir (the host path) and relied on a symlink created in
_prepare_workspace that wasn't present at exec time. The fix cd's to the
guaranteed remote_workdir instead.
"""

from __future__ import annotations

from pathlib import Path

from backend.services.runtime.interface import SandboxConfig
from backend.services.runtime.runpod_backend import _remote_command

_HOST = "/Volumes/CS_Stuff/scientific_article_generator/runs/prj_x/code"
_REMOTE = "/workspace/reprolab-abc123/work"


def _cfg(**over) -> SandboxConfig:
    kw = dict(project_id="prj_x", run_id="run_x", project_root=Path(_HOST), workdir=_HOST)
    kw.update(over)
    return SandboxConfig(**kw)


def test_uses_remote_workdir_when_supplied() -> None:
    script = _remote_command(_cfg(), "python train.py", remote_workdir=_REMOTE)
    assert f"cd '{_REMOTE}' && python train.py" in script
    # The fragile host path must NOT leak into the pod command.
    assert "/Volumes/CS_Stuff" not in script


def test_falls_back_to_config_workdir_without_remote() -> None:
    # Legacy/test-double path: no remote_workdir -> prior behavior (config.workdir).
    script = _remote_command(_cfg(), "python train.py")
    assert f"cd '{_HOST}' && python train.py" in script


def test_env_exports_precede_cd() -> None:
    cfg = _cfg(environment={"FOO": "bar"})
    script = _remote_command(cfg, "python train.py", remote_workdir=_REMOTE)
    assert "export FOO='bar'" in script
    assert script.index("export FOO") < script.index("cd ")


def test_remote_base_is_deterministic_and_pod_side() -> None:
    # The exec path recomputes remote_workdir from _remote_base (no in-memory
    # connection lookup) — it must match the create-time layout and never be the
    # host /Volumes path.
    from backend.services.runtime.runpod_backend import _remote_base
    base = _remote_base("/workspace", "prj_abc", "run_xyz")
    assert base == "/workspace/reprolab/prj-abc/run-xyz"
    assert "/Volumes" not in base
    # exec cd target = <base>/work — this is where _finish_create uploads code.
    assert f"{base}/work" == "/workspace/reprolab/prj-abc/run-xyz/work"
