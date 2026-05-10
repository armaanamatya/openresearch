"""Tests for the dependency verification guardrails."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest

from backend.agents.dependency_verifier import (
    DepCheck,
    VerificationReport,
    _GIT_PIN_RE,
    _GIT_BRANCH_PIN_RE,
    _PIP_PACKAGE_RE,
    _GITHUB_CLONE_RE,
    verify_dockerfile,
)


# ---------------------------------------------------------------------------
# Regex extraction tests (offline, no network)
# ---------------------------------------------------------------------------

class TestRegexPatterns:
    def test_git_sha_pin(self):
        line = "pip install git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12bc"
        matches = _GIT_PIN_RE.findall(line)
        assert len(matches) == 1
        repo, sha = matches[0]
        assert "Farama-Foundation/Metaworld" in repo
        assert sha == "04be337a12bc"

    def test_git_sha_no_dot_git(self):
        line = "pip install git+https://github.com/org/repo@abcdef1234567890"
        matches = _GIT_PIN_RE.findall(line)
        assert len(matches) == 1
        assert matches[0][1] == "abcdef1234567890"

    def test_git_branch_pin(self):
        line = "pip install git+https://github.com/org/repo.git@main"
        matches = _GIT_BRANCH_PIN_RE.findall(line)
        assert len(matches) == 1
        assert matches[0][1] == "main"

    def test_pip_version_pin(self):
        text = "RUN pip install torch==2.2.0 numpy==1.24.3 scipy==1.11.4"
        matches = _PIP_PACKAGE_RE.findall(text)
        # Should find at least one
        packages = {m[0]: m[1] for m in matches}
        assert "torch" in packages or "numpy" in packages or "scipy" in packages

    def test_github_clone(self):
        line = "RUN git clone https://github.com/mikelma/componet.git /workspace/componet"
        matches = _GITHUB_CLONE_RE.findall(line)
        assert len(matches) == 1
        assert "mikelma/componet" in matches[0]

    def test_github_clone_no_dot_git(self):
        line = "RUN git clone --depth 1 https://github.com/org/repo /dest"
        matches = _GITHUB_CLONE_RE.findall(line)
        assert len(matches) == 1
        assert "org/repo" in matches[0]


# ---------------------------------------------------------------------------
# DepCheck and VerificationReport unit tests
# ---------------------------------------------------------------------------

class TestDepCheck:
    def test_summary_ok(self):
        check = DepCheck(
            kind="pypi_version",
            reference="torch==2.2.0",
            target="torch",
            pin="2.2.0",
            valid=True,
        )
        assert "[OK]" in check.summary()

    def test_summary_fail(self):
        check = DepCheck(
            kind="git_sha",
            reference="git+https://github.com/org/repo@deadbeef",
            target="github.com/org/repo",
            pin="deadbeef",
            valid=False,
            error="SHA not found",
        )
        assert "[FAIL]" in check.summary()
        assert "SHA not found" in check.summary()


class TestVerificationReport:
    def test_no_failures(self):
        report = VerificationReport(
            checks=[
                DepCheck("pypi_version", "torch==2.2.0", "torch", "2.2.0", valid=True),
            ]
        )
        assert not report.has_failures
        assert report.failures == []
        assert report.feedback_prompt() == ""

    def test_with_failures(self):
        report = VerificationReport(
            checks=[
                DepCheck("pypi_version", "torch==2.2.0", "torch", "2.2.0", valid=True),
                DepCheck(
                    "git_sha",
                    "git+https://github.com/org/repo@deadbeef",
                    "github.com/org/repo",
                    "deadbeef",
                    valid=False,
                    error="SHA not found",
                ),
            ]
        )
        assert report.has_failures
        assert len(report.failures) == 1
        feedback = report.feedback_prompt()
        assert "DEPENDENCY VERIFICATION FAILURES" in feedback
        assert "deadbeef" in feedback
        assert "git ls-remote" in feedback

    def test_feedback_prompt_pypi(self):
        report = VerificationReport(
            checks=[
                DepCheck(
                    "pypi_version",
                    "fakepackage==99.0.0",
                    "fakepackage",
                    "99.0.0",
                    valid=False,
                    error="package does not exist",
                ),
            ]
        )
        feedback = report.feedback_prompt()
        assert "pypi.org" in feedback


# ---------------------------------------------------------------------------
# Dockerfile scanning (filesystem, no network)
# ---------------------------------------------------------------------------

class TestVerifyDockerfile:
    @pytest.fixture
    def tmp_dockerfile(self, tmp_path: Path) -> Path:
        return tmp_path / "Dockerfile"

    def test_nonexistent_dockerfile(self, tmp_path: Path):
        report = asyncio.get_event_loop().run_until_complete(
            verify_dockerfile(tmp_path / "Dockerfile.missing")
        )
        assert not report.has_failures
        assert report.checks == []

    def test_empty_dockerfile(self, tmp_dockerfile: Path):
        tmp_dockerfile.write_text("FROM python:3.11-slim\nRUN echo hello\n")
        report = asyncio.get_event_loop().run_until_complete(
            verify_dockerfile(tmp_dockerfile)
        )
        # No dependency pins to check
        assert not report.has_failures

    def test_detects_git_sha_pin(self, tmp_dockerfile: Path):
        tmp_dockerfile.write_text(textwrap.dedent("""\
            FROM python:3.11-slim
            RUN pip install git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12bcdef0
        """))
        report = asyncio.get_event_loop().run_until_complete(
            verify_dockerfile(tmp_dockerfile)
        )
        # Should have at least one check for the git SHA
        sha_checks = [c for c in report.checks if c.kind == "git_sha"]
        assert len(sha_checks) >= 1
        # The fabricated SHA should fail (unless network is flaky)
        # We just verify it was detected
        assert sha_checks[0].pin == "04be337a12bcdef0"

    def test_detects_pip_version(self, tmp_dockerfile: Path):
        tmp_dockerfile.write_text(textwrap.dedent("""\
            FROM python:3.11-slim
            RUN pip install torch==2.2.0
        """))
        report = asyncio.get_event_loop().run_until_complete(
            verify_dockerfile(tmp_dockerfile)
        )
        pypi_checks = [c for c in report.checks if c.kind == "pypi_version"]
        assert len(pypi_checks) >= 1
        assert pypi_checks[0].target == "torch"
        assert pypi_checks[0].pin == "2.2.0"

    def test_detects_github_clone(self, tmp_dockerfile: Path):
        tmp_dockerfile.write_text(textwrap.dedent("""\
            FROM python:3.11-slim
            RUN git clone https://github.com/mikelma/componet.git /workspace
        """))
        report = asyncio.get_event_loop().run_until_complete(
            verify_dockerfile(tmp_dockerfile)
        )
        repo_checks = [c for c in report.checks if c.kind == "github_repo"]
        assert len(repo_checks) >= 1
        assert "componet" in repo_checks[0].target.lower() or "mikelma" in repo_checks[0].target.lower()


# ---------------------------------------------------------------------------
# Prompt hardening tests
# ---------------------------------------------------------------------------

class TestPromptHardening:
    def test_env_detective_has_anti_hallucination(self):
        from backend.agents.prompts.environment_detective import (
            ENVIRONMENT_DETECTIVE_PROMPT,
        )
        assert "NEVER fabricate git commit SHAs" in ENVIRONMENT_DETECTIVE_PROMPT
        assert "git ls-remote" in ENVIRONMENT_DETECTIVE_PROMPT

    def test_baseline_impl_has_anti_hallucination(self):
        from backend.agents.prompts.baseline_implementation import (
            BASELINE_IMPLEMENTATION_PROMPT,
        )
        assert "NEVER fabricate git commit SHAs" in BASELINE_IMPLEMENTATION_PROMPT
        assert "NEVER invent CLI wrapper scripts" in BASELINE_IMPLEMENTATION_PROMPT
