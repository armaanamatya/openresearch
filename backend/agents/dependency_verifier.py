"""Post-generation guardrails for hallucinated dependencies.

Scans Dockerfiles and pip install commands for git refs, PyPI packages,
and URLs that don't actually exist. Catches the class of failures where
an LLM fabricates a git SHA, invents a package version, or hallucinates
a repository URL.

Usage:
    report = await verify_dockerfile(Path("runs/prj_.../Dockerfile"))
    if report.has_failures:
        # re-run agent with feedback, or auto-fix
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Patterns that appear in Dockerfiles / pip install lines
_GIT_PIN_RE = re.compile(
    r"git\+https?://([^@\s]+?)(?:\.git)?@([0-9a-fA-F]{7,40})\b"
)
_GIT_BRANCH_PIN_RE = re.compile(
    r"git\+https?://([^@\s]+?)(?:\.git)?@([a-zA-Z][a-zA-Z0-9._/-]*)\b"
)
_PIP_PACKAGE_RE = re.compile(
    r"pip\s+install[^\\]*?(?:^|\s)([a-zA-Z0-9_-]+)==(\d+\.\d+[^\s\\]*)",
    re.MULTILINE,
)
_GITHUB_CLONE_RE = re.compile(
    r"git\s+clone\s+(?:--[^\s]+(?:\s+[^\s-][^\s]*)?\s+)*https?://github\.com/([^\s]+?)(?:\.git)?(?:\s|$)"
)


@dataclass
class DepCheck:
    """Result of verifying one dependency reference."""

    kind: str  # "git_sha", "git_branch", "pypi_version", "github_repo"
    reference: str  # the full reference string
    target: str  # repo URL, package name, etc.
    pin: str  # SHA, branch, version
    valid: bool
    error: str = ""

    def summary(self) -> str:
        status = "OK" if self.valid else "FAIL"
        return f"[{status}] {self.kind}: {self.target}@{self.pin} — {self.error or 'verified'}"


@dataclass
class VerificationReport:
    """Aggregated result of all dependency checks on a Dockerfile."""

    checks: list[DepCheck] = field(default_factory=list)
    source_path: str = ""

    @property
    def has_failures(self) -> bool:
        return any(not c.valid for c in self.checks)

    @property
    def failures(self) -> list[DepCheck]:
        return [c for c in self.checks if not c.valid]

    def feedback_prompt(self) -> str:
        """Generate a prompt fragment telling the agent what it got wrong."""
        if not self.has_failures:
            return ""
        lines = [
            "DEPENDENCY VERIFICATION FAILURES — fix these before proceeding:",
            "",
        ]
        for f in self.failures:
            lines.append(f"  - {f.summary()}")
            if f.kind == "git_sha":
                lines.append(
                    f"    Fix: Use a verified branch/tag instead of SHA, "
                    f"or run `git ls-remote https://{f.target}` to find valid refs."
                )
            elif f.kind == "pypi_version":
                lines.append(
                    f"    Fix: Check https://pypi.org/project/{f.target}/ for available versions."
                )
            elif f.kind == "github_repo":
                lines.append(
                    f"    Fix: Verify the repository exists at https://github.com/{f.target}"
                )
        lines.append("")
        lines.append(
            "RULE: Never fabricate git SHAs or package versions. "
            "Use branch names (main, master) when you cannot verify a specific commit."
        )
        return "\n".join(lines)


async def _verify_git_sha(repo_url: str, sha: str) -> DepCheck:
    """Verify a git SHA exists on the remote via git ls-remote."""
    check = DepCheck(
        kind="git_sha",
        reference=f"git+https://{repo_url}@{sha}",
        target=repo_url,
        pin=sha,
        valid=False,
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "ls-remote", f"https://{repo_url}.git", sha,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        output = stdout.decode().strip()
        if output and sha.lower() in output.lower():
            check.valid = True
        else:
            # Also try listing all refs and checking if any match the SHA prefix
            proc2 = await asyncio.create_subprocess_exec(
                "git", "ls-remote", f"https://{repo_url}.git",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout2, _ = await asyncio.wait_for(proc2.communicate(), timeout=15)
            all_refs = stdout2.decode().strip()
            sha_lower = sha.lower()
            if any(line.lower().startswith(sha_lower) for line in all_refs.splitlines()):
                check.valid = True
            else:
                check.error = f"SHA {sha[:12]}... not found in remote refs"
    except asyncio.TimeoutError:
        check.error = "git ls-remote timed out (15s)"
    except FileNotFoundError:
        check.error = "git not found on PATH"
    except Exception as exc:
        check.error = f"verification failed: {exc}"
    return check


async def _verify_git_branch(repo_url: str, branch: str) -> DepCheck:
    """Verify a git branch/tag exists on the remote."""
    check = DepCheck(
        kind="git_branch",
        reference=f"git+https://{repo_url}@{branch}",
        target=repo_url,
        pin=branch,
        valid=False,
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "ls-remote", "--heads", "--tags",
            f"https://{repo_url}.git", branch,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        output = stdout.decode().strip()
        if output:
            check.valid = True
        else:
            check.error = f"branch/tag '{branch}' not found in remote"
    except asyncio.TimeoutError:
        check.error = "git ls-remote timed out (15s)"
    except FileNotFoundError:
        check.error = "git not found on PATH"
    except Exception as exc:
        check.error = f"verification failed: {exc}"
    return check


async def _verify_pypi_version(package: str, version: str) -> DepCheck:
    """Verify a PyPI package+version exists via the JSON API."""
    check = DepCheck(
        kind="pypi_version",
        reference=f"{package}=={version}",
        target=package,
        pin=version,
        valid=False,
    )
    url = f"https://pypi.org/pypi/{package}/{version}/json"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                check.valid = True
            elif resp.status_code == 404:
                # Check if the package exists at all
                pkg_resp = await client.get(f"https://pypi.org/pypi/{package}/json")
                if pkg_resp.status_code == 404:
                    check.error = f"package '{package}' does not exist on PyPI"
                else:
                    data = pkg_resp.json()
                    available = sorted(data.get("releases", {}).keys())[-5:]
                    check.error = (
                        f"version {version} not found; "
                        f"latest available: {', '.join(available)}"
                    )
            else:
                check.error = f"PyPI returned HTTP {resp.status_code}"
    except Exception as exc:
        check.error = f"PyPI check failed: {exc}"
    return check


async def _verify_github_repo(repo_path: str) -> DepCheck:
    """Verify a GitHub repository exists via the API."""
    check = DepCheck(
        kind="github_repo",
        reference=f"https://github.com/{repo_path}",
        target=repo_path,
        pin="HEAD",
        valid=False,
    )
    # Clean up repo path (remove trailing paths after owner/repo)
    parts = repo_path.strip("/").split("/")
    if len(parts) < 2:
        check.error = f"invalid GitHub path: {repo_path}"
        return check
    owner_repo = f"{parts[0]}/{parts[1]}"
    check.target = owner_repo

    url = f"https://api.github.com/repos/{owner_repo}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers={"Accept": "application/json"})
            if resp.status_code == 200:
                check.valid = True
            elif resp.status_code == 404:
                check.error = f"repository github.com/{owner_repo} does not exist"
            elif resp.status_code == 403:
                # Rate limited — assume valid to avoid false positives
                check.valid = True
                check.error = "GitHub rate limit hit; assumed valid"
            else:
                check.error = f"GitHub API returned HTTP {resp.status_code}"
    except Exception as exc:
        check.error = f"GitHub check failed: {exc}"
    return check


async def verify_dockerfile(dockerfile_path: Path) -> VerificationReport:
    """Scan a Dockerfile for dependency references and verify each one."""
    report = VerificationReport(source_path=str(dockerfile_path))

    if not dockerfile_path.exists():
        return report

    content = dockerfile_path.read_text(encoding="utf-8", errors="replace")
    tasks: list[Any] = []

    # Check git+https://...@SHA pins
    for match in _GIT_PIN_RE.finditer(content):
        repo_url, sha = match.group(1), match.group(2)
        tasks.append(_verify_git_sha(repo_url, sha))

    # Check git+https://...@branch pins
    for match in _GIT_BRANCH_PIN_RE.finditer(content):
        repo_url, branch = match.group(1), match.group(2)
        # Skip if already checked as SHA
        if not re.match(r"^[0-9a-fA-F]{7,40}$", branch):
            tasks.append(_verify_git_branch(repo_url, branch))

    # Check pip package==version pins (only major packages to avoid noise)
    _SKIP_PACKAGES = {"pip", "setuptools", "wheel"}
    for match in _PIP_PACKAGE_RE.finditer(content):
        package, version = match.group(1), match.group(2)
        if package.lower() not in _SKIP_PACKAGES:
            tasks.append(_verify_pypi_version(package, version))

    # Check git clone GitHub repos
    for match in _GITHUB_CLONE_RE.finditer(content):
        repo_path = match.group(1)
        tasks.append(_verify_github_repo(repo_path))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, DepCheck):
                report.checks.append(result)
                logger.info("Dep check: %s", result.summary())
            elif isinstance(result, Exception):
                logger.warning("Dep check raised: %s", result)

    return report


async def verify_baseline_result(
    baseline_result: dict[str, Any],
    runs_root: Path,
) -> VerificationReport:
    """Verify dependencies in a baseline_result's Dockerfile and commands."""
    dockerfile_path = runs_root / baseline_result.get("dockerfile_path", "")
    if dockerfile_path.exists():
        return await verify_dockerfile(dockerfile_path)
    return VerificationReport()


__all__ = [
    "DepCheck",
    "VerificationReport",
    "verify_baseline_result",
    "verify_dockerfile",
]
