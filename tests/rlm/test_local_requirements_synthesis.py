"""Phase 2A (2026-06-07) — local-sandbox requirements.txt synthesis.

The local sandbox previously gated its dependency install on requirements.txt
ALREADY existing; only the runpod block synthesized it. So a local run whose agent
forgot requirements.txt installed nothing and died at the first third-party import
(the matplotlib ModuleNotFoundError class). primitives._execute_in_sandbox now calls
ensure_requirements_txt on the local path when the file is missing — the same
synthesizer the runpod block uses. This pins that the synthesizer captures the
third-party import that was crashing local runs.
"""
from __future__ import annotations

from pathlib import Path

from backend.agents.rlm.requirements_derive import ensure_requirements_txt


def test_synthesis_from_dockerfile_on_local(tmp_path: Path):
    # ensure_requirements_txt derives from the project Dockerfile's pip installs.
    # The local path (primitives._execute_in_sandbox) computes
    # _project_dir_local = code_dir.parent (when code_dir.name == "code") and passes
    # _project_dir_local / "Dockerfile" — so the Dockerfile is the sibling of code/.
    code = tmp_path / "code"
    code.mkdir()
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.11-slim\nRUN pip install matplotlib numpy\n", encoding="utf-8")
    req = code / "requirements.txt"
    assert not req.exists()

    out = ensure_requirements_txt(code, dockerfile_path=tmp_path / "Dockerfile", base_image="")

    assert out is not None and req.exists(), "local synthesis must derive requirements.txt"
    content = req.read_text(encoding="utf-8").lower()
    assert "matplotlib" in content, f"matplotlib not captured: {content!r}"


def test_synthesis_none_without_dockerfile(tmp_path: Path):
    # No Dockerfile and no requirements.txt → synthesis returns None (fail-soft); the
    # local block then simply installs nothing extra (no crash). Documents the limit:
    # synthesis needs a dependency SOURCE (a Dockerfile), it does not scan .py imports.
    code = tmp_path / "code"
    code.mkdir()
    (code / "train.py").write_text("import matplotlib\n", encoding="utf-8")
    assert ensure_requirements_txt(code, dockerfile_path=tmp_path / "Dockerfile", base_image="") is None
    assert not (code / "requirements.txt").exists()


def test_synthesis_noop_when_requirements_present(tmp_path: Path):
    # If requirements.txt already exists, synthesis must not clobber it (the local
    # block only calls ensure_requirements_txt when the file is MISSING, but the
    # synthesizer itself is also idempotent/safe).
    code = tmp_path / "code"
    code.mkdir()
    (code / "train.py").write_text("import torch\n", encoding="utf-8")
    req = code / "requirements.txt"
    req.write_text("pinned-package==1.2.3\n", encoding="utf-8")

    ensure_requirements_txt(code, dockerfile_path=tmp_path / "Dockerfile", base_image="")

    assert "pinned-package==1.2.3" in req.read_text(encoding="utf-8")
