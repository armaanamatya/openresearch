# GitHub-repo-first reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a paper links an official code repository, discover/clone it, adapt it into the runnable `code/`, and report two honest axes (execution "did it run" + replication "did it reproduce"), all behind a default-OFF master flag so the system is byte-identical when unset.

**Architecture:** Finish GitHub issue #62. A new pure/IO package `backend/services/ingestion/repo/` (resolver + manifest + provisioner) is wired into the single keystone `run.py::_build_context()` which resolves a `RepoSpec`, clones into `runs/<id>/repo/`, exposes a constant-size manifest as the root's `repo_files`, and persists `rlm_state/repo_spec.json` (the deterministic trusted source). `detect_environment` merges the repo's declared deps; `implement_baseline` seeds `code/` from `repo/` once (adapt mode) and injects `artifact_index`; the report writer attaches a top-level `final_report.reproduction` block. A thin `inspect_repository` primitive (the 18th) lets the root deep-read the repo. Multi-cloud transports exclude `repo/` so only `code/` ever crosses to a GPU backend — exactly as today.

**Tech Stack:** Python 3.12 (floor 3.11), FastAPI, pydantic v2 / pydantic-settings, the `rlms` library (`rlm.RLM`), `git` CLI (subprocess), pytest + pytest-socket (socket-hermetic suite), Next.js 16 / vitest (frontend, separable phase).

## Global Constraints

- **Python floor 3.11; dev venv 3.14; Docker image 3.12; CI/locked env 3.12** (`uv venv --python 3.12 && uv sync --frozen`). Lint: `uvx ruff@0.15.16 check .` (E4/E7/E9/F defaults).
- **Flags are `OPENRESEARCH_*`-canonical; the `REPROLAB_*` legacy prefix is bridged at import by `config.py::_apply_legacy_env_aliases`** (a var set *after* import — e.g. a test monkeypatch — is NOT bridged, so use `OPENRESEARCH_*` in code and tests).
- **Default-OFF, byte-identical when unset.** Every task that adds behavior MUST include an off-state regression test proving no change when `OPENRESEARCH_USE_AUTHOR_REPO` is unset (no resolve/clone, `repo_files` stays `None`, no `repo_spec.json`, `inspect_repository` returns `{"status": "disabled"}`, `detect_environment`/`implement_baseline`/report unchanged).
- **Socket-hermetic tests.** `pytest-socket` blocks non-loopback sockets in the suite. Clone tests MUST use a LOCAL `file://` git remote (a temp git repo created with `git init` + a commit), NEVER the network.
- **17-primitive registry invariant.** `PRIMITIVE_REGISTRY` and `tests/rlm/test_registry.py::EXPECTED` must stay in sync. This plan takes the count 17 → 18 (`inspect_repository`). `build_custom_tools` advertises every primitive unconditionally — off-state inertness lives in the function body (return `{"status": "disabled"}`), exactly like `read_context_map`.
- **pytest invocation:** `.venv/bin/python -m pytest <path>::<test> -v`. Pytest config in `pyproject.toml` (`testpaths=["tests"]`, `pythonpath=["."]`).
- **Commits at PHASE boundaries only** (the user commits infrequently at milestones). One commit step ends each phase. Commit message = descriptive present-tense headline, **NO Conventional-Commits prefix** (no `feat:`/`fix:`/`chore:`), **NO `Co-Authored-By` / AI-attribution trailer**.
- **No push steps.** Never `git push`.
- **Branch:** Phase 0 creates `feat/github-repo-first-reproduction` off `main`.

---

## Phase 0 — Foundation

- [ ] **Step 0: Create the feature branch**

The user will confirm the base branch at execution time. Default base is `main`.

```bash
git checkout main && git checkout -b feat/github-repo-first-reproduction
```

### Task 1: Config flags

**Files:**
- Modify: `backend/config.py` (the `Settings` class, after the existing feature-flag fields)
- Test: `tests/config/test_repo_flags.py`

**Interfaces:**
- Consumes: `backend.config.Settings`, `backend.config.get_settings`, `backend.config._apply_legacy_env_aliases`.
- Produces: `Settings.use_author_repo: bool = False`, `Settings.reproduction_mode: str = "adapt"`, `Settings.repo_clone_timeout_s: int = 300`, `Settings.repo_clone_max_mb: int = 2048`, `Settings.repo_clone_lfs: bool = False` (env-bound to `OPENRESEARCH_USE_AUTHOR_REPO`, `OPENRESEARCH_REPRODUCTION_MODE`, `OPENRESEARCH_REPO_CLONE_TIMEOUT_S`, `OPENRESEARCH_REPO_CLONE_MAX_MB`, `OPENRESEARCH_REPO_CLONE_LFS` via the `OPENRESEARCH_` `env_prefix`).

- [ ] **Step 1: Write the failing test**

`tests/config/test_repo_flags.py`:

```python
import importlib

from backend.config import Settings


def test_repo_flag_defaults():
    s = Settings()
    assert s.use_author_repo is False
    assert s.reproduction_mode == "adapt"
    assert s.repo_clone_timeout_s == 300
    assert s.repo_clone_max_mb == 2048
    assert s.repo_clone_lfs is False


def test_repo_flags_read_openresearch_env(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "true")
    monkeypatch.setenv("OPENRESEARCH_REPRODUCTION_MODE", "reference")
    monkeypatch.setenv("OPENRESEARCH_REPO_CLONE_TIMEOUT_S", "120")
    monkeypatch.setenv("OPENRESEARCH_REPO_CLONE_MAX_MB", "512")
    monkeypatch.setenv("OPENRESEARCH_REPO_CLONE_LFS", "1")
    s = Settings()
    assert s.use_author_repo is True
    assert s.reproduction_mode == "reference"
    assert s.repo_clone_timeout_s == 120
    assert s.repo_clone_max_mb == 512
    assert s.repo_clone_lfs is True


def test_repo_flags_legacy_reprolab_bridge(monkeypatch):
    # The REPROLAB_* -> OPENRESEARCH_* bridge runs once at import. Set the legacy
    # var, then re-invoke the aliaser so the counterpart is filled, mirroring how
    # an operator who still exports REPROLAB_* before startup is handled.
    monkeypatch.setenv("REPROLAB_USE_AUTHOR_REPO", "true")
    import backend.config as cfg
    cfg._apply_legacy_env_aliases()
    assert cfg.Settings().use_author_repo is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/config/test_repo_flags.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'use_author_repo'`.

- [ ] **Step 3: Write minimal implementation**

In `backend/config.py`, inside the `Settings` class, add (place near the other feature-flag fields, e.g. just after `provider_fallback_disabled: bool = False` on line 83):

```python
    # GitHub-repo-first reproduction (#62, default-OFF; spec
    # 2026-06-21-github-repo-first-reproduction-design.md). Unset => byte-identical
    # to today: no resolve/clone, repo_files stays None, inspect_repository returns
    # disabled, detect_environment/implement_baseline/report unchanged.
    use_author_repo: bool = False           # OPENRESEARCH_USE_AUTHOR_REPO (master)
    reproduction_mode: str = "adapt"        # OPENRESEARCH_REPRODUCTION_MODE: adapt | reference
    repo_clone_timeout_s: int = 300         # OPENRESEARCH_REPO_CLONE_TIMEOUT_S
    repo_clone_max_mb: int = 2048           # OPENRESEARCH_REPO_CLONE_MAX_MB (post-clone size cap)
    repo_clone_lfs: bool = False            # OPENRESEARCH_REPO_CLONE_LFS (off => GIT_LFS_SKIP_SMUDGE=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/config/test_repo_flags.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit (PHASE 0 boundary)**

```bash
git add backend/config.py tests/config/test_repo_flags.py
git commit -m "Add default-OFF config flags for GitHub-repo-first reproduction"
```

---

## Phase 1 — The `backend/services/ingestion/repo/` package

All new pure/IO units, each independently testable. The package directory and `tests/services/ingestion/repo/` directory are created as part of Task 2 (the first task that needs them).

### Task 2: `resolver.py` — `RepoSpec` + `RepoResolver` (pure, no IO)

**Files:**
- Create: `backend/services/ingestion/repo/__init__.py`
- Create: `backend/services/ingestion/repo/resolver.py`
- Create: `tests/services/ingestion/repo/__init__.py`
- Test: `tests/services/ingestion/repo/test_resolver.py`

**Interfaces:**
- Consumes: `backend.services.ingestion.discovery.model.DiscoveredArtifact` (pydantic; fields `kind: DiscoveredArtifactKind`, `locator: str` e.g. `"github:owner/repo"`, `url: HttpUrl`, `confidence: float`) and `DiscoveredArtifactKind` (enum with `.repository`).
- Produces:
  - `@dataclass(frozen=True) RepoSpec(url: str | None, source: str, mode: str, reason: str)`.
  - `class RepoResolver` with `@staticmethod resolve(user_url: str | None, discovered: list[DiscoveredArtifact], blacklist: set[str], mode_override: str | None) -> RepoSpec`.
  - `def normalize_repo_url(raw: str | None) -> str | None` (canonicalizes `github:owner/repo`, `git@github.com:owner/repo.git`, and full `https://github.com/owner/repo[.git][/...]` to `https://github.com/owner/repo`; returns `None` for unrecognized input).

- [ ] **Step 1: Write the failing test**

`tests/services/ingestion/repo/test_resolver.py`:

```python
from backend.services.ingestion.discovery.model import (
    DiscoveredArtifact,
    DiscoveredArtifactKind,
)
from backend.services.ingestion.repo.resolver import (
    RepoResolver,
    RepoSpec,
    normalize_repo_url,
)


def _repo_artifact(locator: str, confidence: float = 0.9) -> DiscoveredArtifact:
    owner_repo = locator.split(":", 1)[1] if ":" in locator else locator
    return DiscoveredArtifact(
        id=f"art:{owner_repo}",
        project_id="prj_test",
        kind=DiscoveredArtifactKind.repository,
        locator=locator,
        url=f"https://github.com/{owner_repo}",
        evidence_quote="see https://github.com/" + owner_repo,
        confidence=confidence,
    )


def test_normalize_github_shorthand():
    assert normalize_repo_url("github:ZJU-REAL/SDAR") == "https://github.com/ZJU-REAL/SDAR"


def test_normalize_full_url_strips_suffix_and_path():
    assert normalize_repo_url("https://github.com/ZJU-REAL/SDAR.git") == "https://github.com/ZJU-REAL/SDAR"
    assert normalize_repo_url("https://github.com/ZJU-REAL/SDAR/tree/main") == "https://github.com/ZJU-REAL/SDAR"


def test_normalize_ssh_form():
    assert normalize_repo_url("git@github.com:ZJU-REAL/SDAR.git") == "https://github.com/ZJU-REAL/SDAR"


def test_normalize_unrecognized_returns_none():
    assert normalize_repo_url("") is None
    assert normalize_repo_url(None) is None
    assert normalize_repo_url("not a url") is None


def test_user_url_wins_over_discovered():
    spec = RepoResolver.resolve(
        user_url="github:me/mine",
        discovered=[_repo_artifact("github:them/theirs")],
        blacklist=set(),
        mode_override=None,
    )
    assert spec == RepoSpec(
        url="https://github.com/me/mine", source="user", mode="adapt",
        reason=spec.reason,
    )
    assert spec.source == "user"
    assert "user" in spec.reason.lower()


def test_highest_confidence_discovered_used_when_no_user_url():
    spec = RepoResolver.resolve(
        user_url=None,
        discovered=[
            _repo_artifact("github:low/conf", confidence=0.5),
            _repo_artifact("github:high/conf", confidence=0.95),
        ],
        blacklist=set(),
        mode_override=None,
    )
    assert spec.url == "https://github.com/high/conf"
    assert spec.source == "discovered"
    assert spec.mode == "adapt"


def test_blacklisted_url_is_dropped_to_scratch():
    spec = RepoResolver.resolve(
        user_url="github:them/theirs",
        discovered=[],
        blacklist={"https://github.com/them/theirs"},
        mode_override=None,
    )
    assert spec.url is None
    assert spec.source == "none"
    assert spec.mode == "scratch"


def test_mode_override_reference():
    spec = RepoResolver.resolve(
        user_url="github:me/mine",
        discovered=[],
        blacklist=set(),
        mode_override="reference",
    )
    assert spec.url == "https://github.com/me/mine"
    assert spec.mode == "reference"


def test_default_mode_is_adapt():
    spec = RepoResolver.resolve(
        user_url="github:me/mine", discovered=[], blacklist=set(), mode_override=None,
    )
    assert spec.mode == "adapt"


def test_no_repo_yields_scratch():
    spec = RepoResolver.resolve(
        user_url=None, discovered=[], blacklist=set(), mode_override=None,
    )
    assert spec == RepoSpec(url=None, source="none", mode="scratch", reason=spec.reason)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/services/ingestion/repo/test_resolver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.services.ingestion.repo'`.

- [ ] **Step 3: Write minimal implementation**

`backend/services/ingestion/repo/__init__.py`:

```python
"""GitHub-repo-first reproduction (#62): resolve + clone + manifest the paper's
linked code repository. All units are flag-gated by the caller; this package is
itself flag-agnostic (pure resolution + IO helpers)."""
```

`tests/services/ingestion/repo/__init__.py`:

```python
```

`backend/services/ingestion/repo/resolver.py`:

```python
"""Pure resolution of the reproduction's source repository.

No IO. Given a user-provided URL, the paper's discovered repository artifacts,
a blacklist, and an optional mode override, decide WHICH repo (if any) to use
and in WHICH mode (adapt / reference / scratch). The blacklist preserves the
existing "blocked = do not use" semantics: a resolved URL on the blacklist is
DROPPED (treated as not-found) and the run proceeds scratch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid an import cycle / keep this module pure-importable
    from backend.services.ingestion.discovery.model import DiscoveredArtifact

# owner/repo from a github: shorthand, an ssh remote, or a full https url.
_SHORTHAND_RE = re.compile(r"^github:(?P<owner>[^/]+)/(?P<repo>[^/#?]+)")
_SSH_RE = re.compile(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/#?]+)")
_HTTPS_RE = re.compile(r"^https?://(?:www\.)?github\.com/(?P<owner>[^/]+)/(?P<repo>[^/#?]+)")


@dataclass(frozen=True)
class RepoSpec:
    """The resolved repository decision.

    ``url`` is the canonical ``https://github.com/owner/repo`` or ``None`` (no repo).
    ``source`` is ``user`` | ``discovered`` | ``none``.
    ``mode`` is ``adapt`` | ``reference`` | ``scratch``.
    ``reason`` is a human string for the SSE/event narration.
    """

    url: str | None
    source: str
    mode: str
    reason: str


def normalize_repo_url(raw: str | None) -> str | None:
    """Canonicalize a github locator/url to ``https://github.com/owner/repo``.

    Accepts ``github:owner/repo``, ``git@github.com:owner/repo.git``, and a full
    ``https://github.com/owner/repo[.git][/tree/...]``. Returns ``None`` for any
    unrecognized input (empty, None, non-github).
    """
    if not raw:
        return None
    raw = raw.strip()
    for pat in (_SHORTHAND_RE, _SSH_RE, _HTTPS_RE):
        m = pat.match(raw)
        if m:
            owner = m.group("owner")
            repo = m.group("repo")
            if repo.endswith(".git"):
                repo = repo[: -len(".git")]
            return f"https://github.com/{owner}/{repo}"
    return None


class RepoResolver:
    """Pure resolver. Priority: user_url > highest-confidence discovered repo > none."""

    @staticmethod
    def resolve(
        user_url: str | None,
        discovered: "list[DiscoveredArtifact]",
        blacklist: set[str],
        mode_override: str | None,
    ) -> RepoSpec:
        mode = "reference" if (mode_override or "").strip().lower() == "reference" else "adapt"

        # 1. User-provided URL wins.
        norm_user = normalize_repo_url(user_url)
        if norm_user is not None:
            if norm_user in blacklist:
                return RepoSpec(
                    url=None, source="none", mode="scratch",
                    reason=f"user repo {norm_user} is blacklisted; proceeding scratch",
                )
            return RepoSpec(
                url=norm_user, source="user", mode=mode,
                reason=f"user-provided repo {norm_user} (mode={mode})",
            )

        # 2. Highest-confidence discovered repository artifact.
        repos = [
            a for a in discovered
            if getattr(getattr(a, "kind", None), "value", str(getattr(a, "kind", ""))) == "repository"
        ]
        repos.sort(key=lambda a: float(getattr(a, "confidence", 0.0)), reverse=True)
        for art in repos:
            norm = normalize_repo_url(getattr(art, "locator", None)) or normalize_repo_url(
                str(getattr(art, "url", "")) or None
            )
            if norm is None:
                continue
            if norm in blacklist:
                continue  # blocked = do not use; try the next candidate
            return RepoSpec(
                url=norm, source="discovered", mode=mode,
                reason=f"discovered repo {norm} (confidence={float(getattr(art, 'confidence', 0.0)):.2f}, mode={mode})",
            )

        # 3. Nothing usable.
        return RepoSpec(
            url=None, source="none", mode="scratch",
            reason="no usable repository resolved; proceeding scratch",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/services/ingestion/repo/test_resolver.py -v`
Expected: PASS (9 passed).

### Task 3: `manifest.py` — `RepoManifest` + `build_manifest` (pure)

**Files:**
- Create: `backend/services/ingestion/repo/manifest.py`
- Test: `tests/services/ingestion/repo/test_manifest.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure stdlib over a `Path`).
- Produces:
  - `@dataclass RepoManifest(path: str, commit_sha: str | None, file_tree: list[str], key_files: dict[str, str], size_mb: float, lfs_skipped: bool)`.
  - `def build_manifest(repo_dir: Path, *, commit_sha: str | None = None, size_mb: float = 0.0, lfs_skipped: bool = True) -> RepoManifest`.
  - `RepoManifest.as_context(self) -> dict` — a JSON-serializable dict under a fixed byte ceiling (mirrors the `context_map.MAX_BYTES` cap pattern).
  - Module constants `MAX_FILES = 200`, `MAX_DEPTH = 4`, `MAX_CONTEXT_BYTES = 8192`, `KEY_FILE_EXCERPT_CHARS = 800`.

- [ ] **Step 1: Write the failing test**

`tests/services/ingestion/repo/test_manifest.py`:

```python
import json

from backend.services.ingestion.repo.manifest import (
    MAX_CONTEXT_BYTES,
    MAX_DEPTH,
    MAX_FILES,
    RepoManifest,
    build_manifest,
)


def _make_repo(tmp_path):
    (tmp_path / "README.md").write_text("# SDAR\nrun: python train.py\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("torch==2.2.0\ntransformers\n", encoding="utf-8")
    (tmp_path / "train.py").write_text("def main():\n    pass\n", encoding="utf-8")
    sub = tmp_path / "src" / "sdar"
    sub.mkdir(parents=True)
    (sub / "model.py").write_text("class M: ...\n", encoding="utf-8")
    return tmp_path


def test_key_files_detected(tmp_path):
    repo = _make_repo(tmp_path)
    m = build_manifest(repo, commit_sha="abc1234")
    assert "README.md" in m.key_files
    assert "requirements.txt" in m.key_files
    assert "train.py" in m.key_files
    # Excerpt of README is captured (non-empty), not the whole file blindly.
    assert "SDAR" in m.key_files["README.md"]


def test_file_tree_capped_files_and_depth(tmp_path):
    repo = tmp_path
    # 250 shallow files -> capped to MAX_FILES.
    for i in range(250):
        (repo / f"f{i}.py").write_text("x = 1\n", encoding="utf-8")
    # A file deeper than MAX_DEPTH must be excluded.
    deep = repo / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (deep / "too_deep.py").write_text("y = 2\n", encoding="utf-8")
    m = build_manifest(repo, commit_sha=None)
    assert len(m.file_tree) <= MAX_FILES
    assert all(p.count("/") < MAX_DEPTH for p in m.file_tree)
    assert "a/b/c/d/e/too_deep.py" not in m.file_tree


def test_as_context_under_byte_ceiling(tmp_path):
    repo = tmp_path
    for i in range(MAX_FILES + 50):
        (repo / f"file_{i:04d}.py").write_text("z = 3\n" * 50, encoding="utf-8")
    (repo / "README.md").write_text("R" * 100_000, encoding="utf-8")
    m = build_manifest(repo, commit_sha="deadbee")
    ctx = m.as_context()
    assert isinstance(ctx, dict)
    encoded = json.dumps(ctx)
    assert len(encoded.encode("utf-8")) <= MAX_CONTEXT_BYTES
    # Provenance survives the truncation.
    assert ctx["commit_sha"] == "deadbee"


def test_as_context_round_trips_simple(tmp_path):
    repo = _make_repo(tmp_path)
    m = build_manifest(repo, commit_sha="abc1234", size_mb=1.5, lfs_skipped=True)
    ctx = m.as_context()
    assert ctx["commit_sha"] == "abc1234"
    assert ctx["size_mb"] == 1.5
    assert ctx["lfs_skipped"] is True
    assert "train.py" in ctx["file_tree"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/services/ingestion/repo/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.services.ingestion.repo.manifest'`.

- [ ] **Step 3: Write minimal implementation**

`backend/services/ingestion/repo/manifest.py`:

```python
"""Constant-size repository manifest.

The MANIFEST — not the raw tree — is what enters the root's context (RLM
Algorithm 1 invariant). A hard byte ceiling on ``as_context()`` mirrors the
``context_map.MAX_BYTES`` cap. The root navigates deeper via ``inspect_repository``.
Pure stdlib; fail-soft.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

MAX_FILES = 200
MAX_DEPTH = 4
MAX_CONTEXT_BYTES = 8192
KEY_FILE_EXCERPT_CHARS = 800

# Glob patterns for the files whose names + short excerpts go in the manifest.
_KEY_FILE_GLOBS = (
    "README*", "requirements*.txt", "setup.py", "pyproject.toml",
    "environment*.yml", "environment*.yaml", "train.py", "main.py", "run.py",
)
_EXCLUDED_DIR_PARTS = frozenset({".git", "__pycache__", ".venv", "node_modules"})


@dataclass
class RepoManifest:
    path: str
    commit_sha: str | None
    file_tree: list[str] = field(default_factory=list)
    key_files: dict[str, str] = field(default_factory=dict)
    size_mb: float = 0.0
    lfs_skipped: bool = True

    def as_context(self) -> dict:
        """Return a JSON-serializable dict bounded by ``MAX_CONTEXT_BYTES``.

        Provenance (commit_sha/path/size_mb/lfs_skipped) is preserved first; the
        file_tree and key_files are trimmed until the serialized payload fits.
        """
        import json

        base = {
            "path": self.path,
            "commit_sha": self.commit_sha,
            "size_mb": self.size_mb,
            "lfs_skipped": self.lfs_skipped,
        }
        tree = list(self.file_tree)
        keys = dict(self.key_files)
        while True:
            payload = {**base, "file_tree": tree, "key_files": keys}
            if len(json.dumps(payload).encode("utf-8")) <= MAX_CONTEXT_BYTES:
                return payload
            # Trim the largest contributor first: drop a key-file excerpt, else a tree entry.
            if keys:
                largest = max(keys, key=lambda k: len(keys[k]))
                if len(keys[largest]) > 80:
                    keys[largest] = keys[largest][:80]
                    continue
                keys.pop(largest)
                continue
            if len(tree) > 1:
                tree = tree[: len(tree) // 2]
                continue
            # Floor: provenance only.
            return base


def _is_excluded(rel: Path) -> bool:
    return any(part in _EXCLUDED_DIR_PARTS for part in rel.parts[:-1])


def _is_key_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, g) for g in _KEY_FILE_GLOBS)


def build_manifest(
    repo_dir: Path,
    *,
    commit_sha: str | None = None,
    size_mb: float = 0.0,
    lfs_skipped: bool = True,
) -> RepoManifest:
    """Walk ``repo_dir`` into a capped manifest (≤MAX_FILES, depth <MAX_DEPTH)."""
    repo_dir = Path(repo_dir)
    file_tree: list[str] = []
    key_files: dict[str, str] = {}
    try:
        for abs_path in sorted(repo_dir.rglob("*")):
            if not abs_path.is_file():
                continue
            rel = abs_path.relative_to(repo_dir)
            if _is_excluded(rel):
                continue
            depth = len(rel.parts) - 1  # 0 for a top-level file
            if depth >= MAX_DEPTH:
                continue
            rel_posix = rel.as_posix()
            if len(file_tree) < MAX_FILES:
                file_tree.append(rel_posix)
            if _is_key_file(abs_path.name) and abs_path.name not in key_files:
                try:
                    key_files[abs_path.name] = abs_path.read_text(
                        encoding="utf-8", errors="replace"
                    )[:KEY_FILE_EXCERPT_CHARS]
                except OSError:
                    pass
    except OSError:
        pass
    return RepoManifest(
        path=str(repo_dir),
        commit_sha=commit_sha,
        file_tree=file_tree,
        key_files=key_files,
        size_mb=size_mb,
        lfs_skipped=lfs_skipped,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/services/ingestion/repo/test_manifest.py -v`
Expected: PASS (4 passed).

### Task 4: `provisioner.py` — `RepoProvisioner.clone` (IO, fail-soft)

**Files:**
- Create: `backend/services/ingestion/repo/provisioner.py`
- Test: `tests/services/ingestion/repo/test_provisioner.py`

**Interfaces:**
- Consumes: `RepoSpec` (Task 2), `RepoManifest` / `build_manifest` (Task 3), `backend.config.get_settings` (for `repo_clone_timeout_s`, `repo_clone_max_mb`, `repo_clone_lfs`).
- Produces: `class RepoProvisioner` with `@staticmethod clone(spec: RepoSpec, dest: Path) -> RepoManifest | None`. Returns `None` on ANY failure (bad spec, git error, timeout, oversize). Helpers: `def _dir_size_mb(path: Path) -> float`.

- [ ] **Step 1: Write the failing test (LOCAL git fixture, no network)**

`tests/services/ingestion/repo/test_provisioner.py`:

```python
import subprocess
from pathlib import Path

import pytest

from backend.services.ingestion.repo.provisioner import RepoProvisioner
from backend.services.ingestion.repo.resolver import RepoSpec


def _make_local_git_remote(tmp_path: Path) -> str:
    """Create a real local git repo with one commit; return its file:// URL."""
    src = tmp_path / "remote_src"
    src.mkdir()
    (src / "README.md").write_text("# fixture\n", encoding="utf-8")
    (src / "requirements.txt").write_text("torch\n", encoding="utf-8")
    (src / "train.py").write_text("print('hi')\n", encoding="utf-8")
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e",
    }
    subprocess.run(["git", "init", "-q"], cwd=src, check=True, env={**__import__("os").environ, **env})
    subprocess.run(["git", "add", "-A"], cwd=src, check=True, env={**__import__("os").environ, **env})
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=src, check=True, env={**__import__("os").environ, **env})
    return src.as_uri()  # file:///... — NEVER the network (suite is socket-hermetic)


def test_clone_success_returns_manifest_with_commit_sha(tmp_path):
    remote = _make_local_git_remote(tmp_path)
    spec = RepoSpec(url=remote, source="user", mode="adapt", reason="test")
    dest = tmp_path / "repo"
    manifest = RepoProvisioner.clone(spec, dest)
    assert manifest is not None
    assert (dest / "README.md").exists()
    assert (dest / "train.py").exists()
    assert manifest.commit_sha and len(manifest.commit_sha) >= 7
    assert "README.md" in manifest.key_files


def test_clone_nonexistent_path_returns_none(tmp_path):
    spec = RepoSpec(
        url=(tmp_path / "does_not_exist").as_uri(), source="user", mode="adapt", reason="test",
    )
    assert RepoProvisioner.clone(spec, tmp_path / "repo") is None


def test_clone_oversize_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_REPO_CLONE_MAX_MB", "0")  # any non-empty clone exceeds 0 MB
    remote = _make_local_git_remote(tmp_path)
    spec = RepoSpec(url=remote, source="user", mode="adapt", reason="test")
    dest = tmp_path / "repo"
    assert RepoProvisioner.clone(spec, dest) is None
    # Oversize clone is discarded.
    assert not dest.exists()


def test_clone_none_url_returns_none(tmp_path):
    spec = RepoSpec(url=None, source="none", mode="scratch", reason="no repo")
    assert RepoProvisioner.clone(spec, tmp_path / "repo") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/services/ingestion/repo/test_provisioner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.services.ingestion.repo.provisioner'`.

- [ ] **Step 3: Write minimal implementation**

`backend/services/ingestion/repo/provisioner.py`:

```python
"""Shallow, fail-soft git clone of the resolved repository.

IO; sandbox-agnostic — always runs on the orchestrator host (which has git + TLS
certs). Any failure (bad spec, network/egress blocked, private/auth, 404,
oversize, timeout) returns ``None`` so the run proceeds from-scratch. A blocked
clone NEVER aborts a run.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from backend.config import get_settings
from backend.services.ingestion.repo.manifest import RepoManifest, build_manifest
from backend.services.ingestion.repo.resolver import RepoSpec

logger = logging.getLogger(__name__)


def _dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total / (1024 * 1024)


class RepoProvisioner:
    """Clone ``spec.url`` into ``dest``; return a manifest or ``None`` (fail-soft)."""

    @staticmethod
    def clone(spec: RepoSpec, dest: Path) -> RepoManifest | None:
        if not spec or not spec.url:
            return None
        dest = Path(dest)
        settings = get_settings()
        timeout_s = int(getattr(settings, "repo_clone_timeout_s", 300))
        max_mb = int(getattr(settings, "repo_clone_max_mb", 2048))
        lfs = bool(getattr(settings, "repo_clone_lfs", False))

        # Clean any stale destination so the clone is deterministic.
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env.setdefault("GIT_TERMINAL_PROMPT", "0")  # never block on credentials
        if not lfs:
            env["GIT_LFS_SKIP_SMUDGE"] = "1"

        cmd = [
            "git", "clone", "--depth", "1", "--no-tags",
            "--config", "credential.helper=",  # disable interactive auth prompts
            spec.url, str(dest),
        ]
        try:
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=timeout_s,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("repo clone failed (%s): %s", type(exc).__name__, spec.url)
            shutil.rmtree(dest, ignore_errors=True)
            return None
        if proc.returncode != 0:
            logger.warning("repo clone non-zero (%s): %s", proc.returncode, (proc.stderr or "")[:300])
            shutil.rmtree(dest, ignore_errors=True)
            return None

        size_mb = _dir_size_mb(dest)
        if max_mb > 0 and size_mb > max_mb:
            logger.warning("repo clone oversize %.1f MB > %d MB: %s", size_mb, max_mb, spec.url)
            shutil.rmtree(dest, ignore_errors=True)
            return None

        commit_sha: str | None = None
        try:
            rev = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=dest, env=env,
                capture_output=True, text=True, timeout=30,
            )
            if rev.returncode == 0:
                commit_sha = rev.stdout.strip() or None
        except (subprocess.TimeoutExpired, OSError):
            commit_sha = None

        return build_manifest(dest, commit_sha=commit_sha, size_mb=size_mb, lfs_skipped=not lfs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/services/ingestion/repo/test_provisioner.py -v`
Expected: PASS (4 passed). (Note: these tests `subprocess.run(["git", ...])` against a local `file://` URL only — no network, socket-hermetic-safe.)

- [ ] **Step 5: Run the whole repo package suite**

Run: `.venv/bin/python -m pytest tests/services/ingestion/repo/ -v`
Expected: PASS (all of Tasks 2–4).

- [ ] **Step 6: Commit (PHASE 1 boundary)**

```bash
git add backend/services/ingestion/repo/ tests/services/ingestion/repo/
git commit -m "Add the repo resolve/manifest/clone package for repo-first reproduction"
```

---

## Phase 2 — Wire into the run context

### Task 5: `run.py::_build_context()` — resolve + clone + expose

**Files:**
- Modify: `backend/agents/rlm/run.py` (`_build_context` at line 574; its caller at line 2791 inside `run_pipeline_rlm`; `run_pipeline_rlm` signature at line 2178)
- Test: `tests/rlm/test_build_context_repo.py`

**Interfaces:**
- Consumes: `RepoResolver.resolve(...)` + `RepoSpec` (Task 2), `RepoProvisioner.clone(spec, dest)` (Task 4), `RepoManifest.as_context()` (Task 3), `backend.config.get_settings`.
- Produces:
  - `_build_context(workspace_claim_map: dict, *, project_dir: Path | None = None, repo_url: str | None = None, blacklist: set[str] | None = None, discovered: list | None = None) -> dict[str, Any]` — the `repo_files` slot is populated from a cloned manifest when the master flag is on and a repo resolves; else stays `None`.
  - A new module-level helper `def _resolve_and_clone_repo(project_dir: Path, repo_url: str | None, blacklist: set[str], discovered: list) -> tuple[dict | None, "RepoSpec | None"]` returning `(repo_files_context_or_None, spec_or_None)` and persisting `rlm_state/repo_spec.json`.
  - `run_pipeline_rlm(..., repo_url: str | None = None)` (new keyword param; defaults preserve all existing call sites).

- [ ] **Step 1: Write the failing test**

`tests/rlm/test_build_context_repo.py`:

```python
import json

import pytest

import backend.agents.rlm.run as run_mod
from backend.services.ingestion.repo.manifest import RepoManifest


_WCM = {
    "entries": [{"title": "Intro", "source_id": "s1"}],
    "paper_id": "2605.15155",
    "paper_title": "SDAR",
    "rubric_spec": {},
}


def test_flag_off_repo_files_is_none_no_clone(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    project_dir = tmp_path / "prj"
    project_dir.mkdir()
    called = {"clone": 0}
    monkeypatch.setattr(
        "backend.services.ingestion.repo.provisioner.RepoProvisioner.clone",
        lambda spec, dest: called.__setitem__("clone", called["clone"] + 1) or None,
    )
    ctx = run_mod._build_context(
        _WCM, project_dir=project_dir, repo_url="github:ZJU-REAL/SDAR",
        blacklist=set(), discovered=[],
    )
    assert ctx["repo_files"] is None
    assert called["clone"] == 0
    assert not (project_dir / "rlm_state" / "repo_spec.json").exists()


def test_flag_on_user_url_populates_repo_files_and_spec(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    project_dir = tmp_path / "prj"
    project_dir.mkdir()

    def _fake_clone(spec, dest):
        assert spec.url == "https://github.com/me/mine"
        from pathlib import Path
        Path(dest).mkdir(parents=True, exist_ok=True)
        return RepoManifest(
            path=str(dest), commit_sha="abc1234",
            file_tree=["train.py"], key_files={"README.md": "# x"},
            size_mb=0.1, lfs_skipped=True,
        )

    monkeypatch.setattr(
        "backend.services.ingestion.repo.provisioner.RepoProvisioner.clone", _fake_clone,
    )
    ctx = run_mod._build_context(
        _WCM, project_dir=project_dir, repo_url="github:me/mine",
        blacklist=set(), discovered=[],
    )
    assert ctx["repo_files"] is not None
    assert ctx["repo_files"]["commit_sha"] == "abc1234"
    spec_path = project_dir / "rlm_state" / "repo_spec.json"
    assert spec_path.exists()
    saved = json.loads(spec_path.read_text())
    assert saved["url"] == "https://github.com/me/mine"
    assert saved["source"] == "user"
    assert saved["commit_sha"] == "abc1234"


def test_flag_on_no_repo_resolved_repo_files_none(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    project_dir = tmp_path / "prj"
    project_dir.mkdir()
    ctx = run_mod._build_context(
        _WCM, project_dir=project_dir, repo_url=None, blacklist=set(), discovered=[],
    )
    assert ctx["repo_files"] is None
    # A scratch spec is still persisted (provenance), but with url=None.
    spec_path = project_dir / "rlm_state" / "repo_spec.json"
    assert spec_path.exists()
    assert json.loads(spec_path.read_text())["url"] is None


def test_build_context_default_args_byte_identical(tmp_path, monkeypatch):
    # Calling _build_context with ONLY the legacy positional arg behaves as before.
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    ctx = run_mod._build_context(_WCM)
    assert ctx["repo_files"] is None
    assert set(ctx) == {
        "paper_text", "paper_metadata", "supplementary_text",
        "repo_files", "prior_work_refs", "rubric_spec",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_build_context_repo.py -v`
Expected: FAIL — `TypeError: _build_context() got an unexpected keyword argument 'project_dir'`.

- [ ] **Step 3: Write minimal implementation**

In `backend/agents/rlm/run.py`, replace the `_build_context` signature (line 574) and the `repo_files: None` return slot (line 615), and add the helper.

Change the signature from:

```python
def _build_context(workspace_claim_map: dict[str, Any]) -> dict[str, Any]:
```

to:

```python
def _build_context(
    workspace_claim_map: dict[str, Any],
    *,
    project_dir: "Path | None" = None,
    repo_url: str | None = None,
    blacklist: "set[str] | None" = None,
    discovered: "list[Any] | None" = None,
) -> dict[str, Any]:
```

Change the return slot from:

```python
        "repo_files": None,
```

to:

```python
        "repo_files": _repo_files,
```

Immediately before the `return {` statement (after `pb = workspace_claim_map.get("paperbench") or {}` block, ~line 601), insert:

```python
    # #62: resolve + clone the paper's linked repo (master flag-gated). Off-state
    # leaves repo_files None and writes no repo_spec.json — byte-identical to today.
    _repo_files: dict[str, Any] | None = None
    if project_dir is not None:
        _repo_files, _ = _resolve_and_clone_repo(
            project_dir, repo_url, blacklist or set(), discovered or [],
        )
```

Add this module-level helper (place just above `_build_context`):

```python
def _resolve_and_clone_repo(
    project_dir: "Path",
    repo_url: str | None,
    blacklist: "set[str]",
    discovered: "list[Any]",
) -> "tuple[dict[str, Any] | None, Any]":
    """Resolve a RepoSpec, clone it (flag-gated), persist rlm_state/repo_spec.json.

    Returns ``(repo_files_context_or_None, RepoSpec_or_None)``. Fail-soft: any
    error returns ``(None, None)`` and the run proceeds scratch. Byte-identical
    off-state: when OPENRESEARCH_USE_AUTHOR_REPO is unset this returns immediately
    without resolving, cloning, or writing repo_spec.json.
    """
    import json as _json
    import os as _os

    from backend.config import get_settings

    if not bool(getattr(get_settings(), "use_author_repo", False)):
        return None, None
    try:
        from backend.services.ingestion.repo.provisioner import RepoProvisioner
        from backend.services.ingestion.repo.resolver import RepoResolver

        mode_override = (
            getattr(get_settings(), "reproduction_mode", "adapt") or "adapt"
        )
        spec = RepoResolver.resolve(repo_url, discovered, set(blacklist), mode_override)
        commit_sha: str | None = None
        repo_files: dict[str, Any] | None = None
        if spec.url:
            manifest = RepoProvisioner.clone(spec, project_dir / "repo")
            if manifest is not None:
                repo_files = manifest.as_context()
                commit_sha = manifest.commit_sha
            else:
                # Clone failed -> fall back to scratch, but record that we tried.
                spec = type(spec)(
                    url=spec.url, source=spec.source, mode="scratch",
                    reason=f"clone failed for {spec.url}; proceeding scratch",
                )
        # Persist the deterministic source of truth (atomic write).
        rlm_state = project_dir / "rlm_state"
        rlm_state.mkdir(parents=True, exist_ok=True)
        payload = {
            "url": spec.url, "source": spec.source, "mode": spec.mode,
            "reason": spec.reason, "commit_sha": commit_sha,
            "path": str(project_dir / "repo") if repo_files else None,
        }
        tmp = rlm_state / "repo_spec.json.tmp"
        tmp.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        _os.replace(tmp, rlm_state / "repo_spec.json")
        return repo_files, spec
    except Exception:  # noqa: BLE001 — repo resolution must never break a run
        logger.warning("repo resolve/clone failed (non-fatal)", exc_info=True)
        return None, None
```

Now update `run_pipeline_rlm` (line 2178): add `repo_url: str | None = None` to the keyword args, and change the call site (line 2791) from:

```python
    context_dict = _build_context(workspace_claim_map)
```

to:

```python
    # #62: thread the resolved blacklist (ctx.blocked_terms) + a discovered-artifact
    # list (when a workspace handle is available) + the repo_url so _build_context
    # can resolve/clone the paper's repo. Falls back to OPENRESEARCH_REPO_URL env
    # (set by the CLI/API plumbing in Task 10) when no explicit kwarg was passed.
    _repo_url = repo_url or os.environ.get("OPENRESEARCH_REPO_URL", "").strip() or None
    _discovered = _load_discovered_artifacts(workspace_service, workspace_id)
    context_dict = _build_context(
        workspace_claim_map,
        project_dir=project_dir,
        repo_url=_repo_url,
        blacklist=set(getattr(ctx, "blocked_terms", ()) or ()),
        discovered=_discovered,
    )
```

Add this small loader near `_resolve_and_clone_repo` (it reads the already-materialized `discovered_artifacts` workspace var when a workspace handle exists; fail-soft to `[]`):

```python
def _load_discovered_artifacts(workspace_service: "Any", workspace_id: str | None) -> "list[Any]":
    """Best-effort fetch of the materialized discovered_artifacts as
    DiscoveredArtifact-shaped objects for RepoResolver. Returns [] on any miss
    (RepoResolver also accepts an empty list → scratch). Never raises."""
    if workspace_service is None or not workspace_id:
        return []
    try:
        # materialize_view returns a WorkspaceView; .get(name) -> Cited[Any] | None,
        # whose .value is the {"project_id","artifacts","count"} payload that
        # _preload_artifacts wrote.
        view = workspace_service.materialize_view(workspace_id)
        cited = view.get("discovered_artifacts")
        payload = getattr(cited, "value", None) if cited is not None else None
        raw = payload.get("artifacts") if isinstance(payload, dict) else None
        if not raw:
            return []
        from types import SimpleNamespace
        # RepoResolver only reads .kind(.value)/.locator/.url/.confidence — a
        # lightweight namespace is enough and avoids re-validating the pydantic model.
        out = []
        for a in raw:
            out.append(SimpleNamespace(
                kind=SimpleNamespace(value=a.get("kind", "")),
                locator=a.get("locator", ""),
                url=a.get("url", ""),
                confidence=a.get("confidence", 0.0),
            ))
        return out
    except Exception:  # noqa: BLE001 — discovery is an optional input
        return []
```

(Note: confirm `Path`, `os`, and `logger` are already imported at the top of `run.py` — they are, per the existing atomic-write idiom at line 2304.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_build_context_repo.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the broad off-state regression**

Run: `.venv/bin/python -m pytest tests/rlm/ -k "build_context or pipeline or registry" -v`
Expected: PASS — no existing run.py/pipeline test regresses (the new params all default to None/inert).

- [ ] **Step 6: Commit (PHASE 2 boundary)**

```bash
git add backend/agents/rlm/run.py tests/rlm/test_build_context_repo.py
git commit -m "Resolve and clone the paper's linked repo into runs/<id>/repo/ and expose the manifest"
```

---

## Phase 3 — Use the repo

### Task 6: `detect_environment` — merge the repo's declared deps

**Files:**
- Modify: `backend/agents/rlm/primitives.py` (`detect_environment` at line 1017; merge after `spec_dict = spec.model_dump()` at line 1067, before `_with_outcome(...)` at line 1108)
- Test: `tests/rlm/test_detect_environment_repo.py`

**Interfaces:**
- Consumes: `ctx.project_dir` (the `repo/` lives at `ctx.project_dir / "repo"`), `backend.config.get_settings`.
- Produces: a module-level pure helper `def _merge_repo_deps_into_spec(spec_dict: dict, repo_dir: Path) -> dict` (returns a new spec dict with repo-declared `pip_packages` merged in, repo deps taking priority over inferred ones). Reads `requirements*.txt`, `setup.py`/`pyproject.toml`, `environment*.yml`.

- [ ] **Step 1: Write the failing test**

`tests/rlm/test_detect_environment_repo.py`:

```python
import pytest

from backend.agents.rlm.primitives import detect_environment, _merge_repo_deps_into_spec


_METHOD = {"core_contribution": "x", "claims": [], "metrics": []}


def test_no_repo_byte_identical(tmp_path, make_context, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    ctx = make_context(tmp_path)
    result = detect_environment(_METHOD, ctx=ctx)
    # The result is a normal EnvironmentSpec dict; no repo merge happened.
    assert result.get("success") is not False
    assert "dockerfile" in result


def test_merge_repo_requirements(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("torch==2.2.0\ntransformers>=4.40\n", encoding="utf-8")
    spec = {"pip_packages": {"numpy": "1.26"}, "dockerfile": "FROM x"}
    merged = _merge_repo_deps_into_spec(spec, repo)
    assert merged["pip_packages"]["torch"] == "==2.2.0"
    assert "transformers" in merged["pip_packages"]
    # Inferred dep survives where the repo doesn't override it.
    assert merged["pip_packages"]["numpy"] == "1.26"


def test_repo_dep_overrides_inferred(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("torch==2.2.0\n", encoding="utf-8")
    spec = {"pip_packages": {"torch": "==1.13"}}
    merged = _merge_repo_deps_into_spec(spec, repo)
    assert merged["pip_packages"]["torch"] == "==2.2.0"  # repo wins


def test_flag_on_with_repo_merges(tmp_path, make_context, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    ctx = make_context(tmp_path)
    repo = ctx.project_dir / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("accelerate==0.30.0\n", encoding="utf-8")
    result = detect_environment(_METHOD, ctx=ctx)
    assert "accelerate" in (result.get("pip_packages") or {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_detect_environment_repo.py -v`
Expected: FAIL — `ImportError: cannot import name '_merge_repo_deps_into_spec'`.

- [ ] **Step 3: Write minimal implementation**

In `backend/agents/rlm/primitives.py`, add the pure helper near `detect_environment`:

```python
def _merge_repo_deps_into_spec(spec_dict: dict, repo_dir: Path) -> dict:
    """Merge a cloned repo's declared deps into an EnvironmentSpec dict.

    Repo-declared deps take priority (ground truth) over inferred ones. Reads
    requirements*.txt (one ``name[==/>=/...]version`` per line), and the bare
    package names from setup.py / pyproject.toml / environment*.yml. Pure +
    fail-soft: an unreadable/garbage file is skipped; returns spec_dict unchanged
    when repo_dir has no recognizable manifest.
    """
    import re as _re

    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        return spec_dict
    repo_pkgs: dict[str, str] = {}

    # requirements*.txt — the highest-signal, version-pinned source.
    for req in sorted(repo_dir.glob("requirements*.txt")):
        try:
            for line in req.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.split("#", 1)[0].strip()
                if not line or line.startswith("-"):
                    continue
                m = _re.match(r"^([A-Za-z0-9._-]+)\s*([<>=!~].*)?$", line)
                if m:
                    repo_pkgs[m.group(1)] = (m.group(2) or "").strip()
        except OSError:
            continue

    # Bare names from pyproject/setup/environment (no version → empty string).
    for name in ("pyproject.toml", "setup.py"):
        f = repo_dir / name
        if f.is_file():
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                for m in _re.finditer(r"[\"']([A-Za-z0-9._-]+)\s*(?:[<>=!~][^\"']*)?[\"']", text):
                    repo_pkgs.setdefault(m.group(1), "")
            except OSError:
                pass
    for env_yml in list(repo_dir.glob("environment*.yml")) + list(repo_dir.glob("environment*.yaml")):
        try:
            for line in env_yml.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip().lstrip("-").strip()
                m = _re.match(r"^([A-Za-z0-9._-]+)\s*(?:[<>=!~=].*)?$", line)
                if m and m.group(1) not in {"dependencies", "name", "channels", "pip"}:
                    repo_pkgs.setdefault(m.group(1), "")
        except OSError:
            continue

    if not repo_pkgs:
        return spec_dict
    merged = dict(spec_dict)
    pip = dict(merged.get("pip_packages") or {})
    for name, ver in repo_pkgs.items():
        if ver or name not in pip:  # repo pin wins; bare name fills a gap
            pip[name] = ver if ver else pip.get(name, "")
    merged["pip_packages"] = pip
    return merged
```

In `detect_environment`, after `spec_dict = spec.model_dump()` (line 1067) and BEFORE the existing hardware-annotation `try:` block, insert:

```python
    # #62: when a repo was cloned (flag on), merge its declared deps — ground
    # truth over prose-inferred deps. Byte-identical when no repo/ exists.
    import os as _os_repo
    if _os_repo.environ.get("OPENRESEARCH_USE_AUTHOR_REPO", "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        _repo_dir = ctx.project_dir / "repo"
        if _repo_dir.is_dir():
            spec_dict = _merge_repo_deps_into_spec(spec_dict, _repo_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_detect_environment_repo.py -v`
Expected: PASS (4 passed).

### Task 7: `implement_baseline` — seed `code/` from `repo/` + inject trusted `artifact_index`

**Files:**
- Modify: `backend/agents/rlm/primitives.py` (`implement_baseline` at line 1909; read `repo_spec.json`; seed before the sub-agent call at line 2253; `code_dir` resolved at line 2298)
- Modify: `backend/agents/baseline_implementation.py` (`run_with_sdk` `context` dict at line 2728)
- Modify: `backend/agents/prompts/baseline_implementation.py` (Mode-1 prompt at lines 14–26)
- Test: `tests/rlm/test_implement_baseline_repo.py`

**Interfaces:**
- Consumes: `ctx.project_dir`, `ctx.runs_root`, `ctx.project_id`; `rlm_state/repo_spec.json` (written by Task 5: `{url, source, mode, reason, commit_sha, path}`).
- Produces:
  - A pure helper `def _load_repo_spec(project_dir: Path) -> dict` (returns `{}` when absent/unreadable).
  - A helper `def _seed_code_from_repo(repo_dir: Path, code_dir: Path) -> int` (copies the tree excluding `.git/`; returns the number of files copied; idempotent contract is enforced by the caller's "code/ empty" gate).
  - `implement_baseline` merges the trusted repo metadata into `artifact_index` (`{repo_url, commit_sha, path, mode}`) regardless of `plan["artifact_index"]`, and in ADAPT mode + first call (code/ empty) seeds `code/` from `repo/` before invoking the sub-agent.

- [ ] **Step 1: Write the failing test**

`tests/rlm/test_implement_baseline_repo.py`:

```python
import json
from pathlib import Path

import pytest

from backend.agents.rlm.primitives import (
    _load_repo_spec,
    _seed_code_from_repo,
    _repo_artifact_index,
    _should_seed_code_from_repo,
)


def _write_repo_spec(project_dir: Path, **kw):
    rlm_state = project_dir / "rlm_state"
    rlm_state.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": "https://github.com/me/mine", "source": "user", "mode": "adapt",
        "reason": "test", "commit_sha": "abc1234",
        "path": str(project_dir / "repo"),
    }
    payload.update(kw)
    (rlm_state / "repo_spec.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_load_repo_spec_absent_returns_empty(tmp_path):
    assert _load_repo_spec(tmp_path) == {}


def test_load_repo_spec_reads_disk(tmp_path):
    _write_repo_spec(tmp_path)
    spec = _load_repo_spec(tmp_path)
    assert spec["url"] == "https://github.com/me/mine"
    assert spec["commit_sha"] == "abc1234"


def test_artifact_index_from_repo_spec_overrides_empty_plan(tmp_path):
    _write_repo_spec(tmp_path)
    ai = _repo_artifact_index(tmp_path, plan_artifact_index={})
    assert ai["repo_url"] == "https://github.com/me/mine"
    assert ai["commit_sha"] == "abc1234"
    assert ai["mode"] == "adapt"
    assert ai["path"].endswith("repo")


def test_seed_copies_tree_excluding_git(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: x", encoding="utf-8")
    (repo / "train.py").write_text("print(1)\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "m.py").write_text("x=1\n", encoding="utf-8")
    code = tmp_path / "code"
    n = _seed_code_from_repo(repo, code)
    assert n == 2  # train.py + src/m.py (NOT .git/HEAD)
    assert (code / "train.py").exists()
    assert (code / "src" / "m.py").exists()
    assert not (code / ".git").exists()


def test_should_seed_only_adapt_and_empty_code(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("x\n", encoding="utf-8")
    code = tmp_path / "code"
    code.mkdir()
    # adapt + empty code + repo present -> seed
    assert _should_seed_code_from_repo("adapt", repo, code) is True
    # reference mode -> never seed
    assert _should_seed_code_from_repo("reference", repo, code) is False
    # non-empty code (repair re-entry) -> never re-seed
    (code / "existing.py").write_text("y\n", encoding="utf-8")
    assert _should_seed_code_from_repo("adapt", repo, code) is False


def test_should_not_seed_without_repo(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    assert _should_seed_code_from_repo("adapt", tmp_path / "no_repo", code) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_implement_baseline_repo.py -v`
Expected: FAIL — `ImportError: cannot import name '_load_repo_spec'`.

- [ ] **Step 3: Write minimal implementation**

In `backend/agents/rlm/primitives.py`, add the helpers near `implement_baseline`:

```python
def _load_repo_spec(project_dir: Path) -> dict:
    """Read rlm_state/repo_spec.json (the deterministic trusted source). {} if absent."""
    import json as _json
    p = Path(project_dir) / "rlm_state" / "repo_spec.json"
    if not p.exists():
        return {}
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _repo_artifact_index(project_dir: Path, plan_artifact_index: dict | None) -> dict:
    """Merge the TRUSTED repo metadata into artifact_index, repo fields winning.

    The root's plan["artifact_index"] is untrusted; the repo_spec.json written at
    run setup is authoritative. Returns the plan's dict unchanged when no repo was used.
    """
    base = dict(plan_artifact_index or {})
    spec = _load_repo_spec(project_dir)
    if spec.get("url"):
        base.update({
            "repo_url": spec.get("url"),
            "commit_sha": spec.get("commit_sha"),
            "path": spec.get("path") or str(Path(project_dir) / "repo"),
            "mode": spec.get("mode") or "adapt",
        })
    return base


def _should_seed_code_from_repo(mode: str, repo_dir: Path, code_dir: Path) -> bool:
    """True iff ADAPT mode, the repo exists, and code/ is empty (first call)."""
    if (mode or "").strip().lower() != "adapt":
        return False
    repo_dir = Path(repo_dir)
    code_dir = Path(code_dir)
    if not repo_dir.is_dir():
        return False
    if code_dir.exists() and any(code_dir.iterdir()):
        return False
    return True


def _seed_code_from_repo(repo_dir: Path, code_dir: Path) -> int:
    """Copy repo_dir → code_dir excluding .git/; return the count of files copied."""
    import shutil as _shutil
    repo_dir = Path(repo_dir)
    code_dir = Path(code_dir)
    code_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(repo_dir.rglob("*")):
        rel = src.relative_to(repo_dir)
        if ".git" in rel.parts:
            continue
        dst = code_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(src, dst)
            copied += 1
    return copied
```

In `implement_baseline`, replace `artifact_index = plan.get("artifact_index")` (line 1979) with the trusted merge:

```python
    artifact_index = _repo_artifact_index(ctx.project_dir, plan.get("artifact_index"))
```

After `code_dir = ctx.runs_root / ctx.project_id / "code"` + `code_dir.mkdir(...)` (lines 2298–2299), insert the adapt-mode seed (flag-gated; before the sub-agent call at 2253 is reached on the non-cache path — note line 2298 sits below the cache lookup, so place this block right after the mkdir at 2299):

```python
    # #62: ADAPT mode, first call only (code/ empty) — seed the authors' code into
    # code/ so the sub-agent ADAPTS rather than rewrites. Re-entrant repair calls
    # never re-seed (code/ already non-empty). Flag-gated; byte-identical off.
    import os as _os_repo
    if _os_repo.environ.get("OPENRESEARCH_USE_AUTHOR_REPO", "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        _rspec = _load_repo_spec(ctx.project_dir)
        _repo_dir = ctx.project_dir / "repo"
        if _should_seed_code_from_repo(_rspec.get("mode", "adapt"), _repo_dir, code_dir):
            _n = _seed_code_from_repo(_repo_dir, code_dir)
            logger.info("implement_baseline[%s]: seeded code/ from repo/ (%d files)", ctx.project_id, _n)
            _emit_dashboard_event(ctx, event_type="run_warning", payload={
                "code": "repo_code_seeded",
                "message": f"adapt-mode: seeded code/ from the authors' repo ({_n} files)",
            })
```

In `backend/agents/baseline_implementation.py`, the `run_with_sdk` `context` dict (line 2728) already carries `"artifact_index": artifact_index or {}` — no change needed there (the trusted merge happens upstream in `implement_baseline`). Add a `reference available at repo/` note only when mode is reference: in `run_with_sdk`, just after `context = {...}` (line 2733), insert:

```python
    # #62: surface the trusted artifact_index mode so the prompt can branch on it.
    _repro_mode = str((artifact_index or {}).get("mode") or "")
    if _repro_mode == "reference":
        context["reference_repo_note"] = (
            "The authors' reference implementation is available read-only at repo/. "
            "Consult it for exact details, but write your own code/ from scratch."
        )
```

In `backend/agents/prompts/baseline_implementation.py`, replace the Mode-1 block (lines 14–26):

```python
## Mode 1: Adapt Existing Repository
When a reference repo was found by the Artifact Discovery Agent:
- Clone or copy the repository
- Adapt code to match the paper's exact experimental setup
- Apply all assumption decisions from the assumption ledger
- Record all changes as a git diff
```

with (brace-free — this prompt string must NOT introduce `{...}` placeholders, which would break a future `.format()` of the file):

```python
## Mode 1: Adapt Existing Repository
The authors' reference implementation is ALREADY in your working `code/`
directory — the harness cloned and copied it for you.
- DO NOT rewrite from scratch; ADAPT the existing code to run in this
  environment and at this scope.
- Apply all assumption decisions from the assumption ledger.
- Keep changes minimal and targeted (env/scope/entrypoint fixes); record what
  you changed in the diff summary.
- A pristine read-only copy of the repo remains alongside `code/` (in `repo/`)
  for reference; your edits go in `code/`.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_implement_baseline_repo.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Run the prompt + baseline-implementation regression**

Run: `.venv/bin/python -m pytest tests/test_baseline_implementation_data_recipes.py tests/test_issue25_baseline.py -v`
Expected: PASS — the Mode-1 prose change and the reference-note addition don't break existing baseline tests.

### Task 8: `inspect_repository` — the 18th primitive

**Files:**
- Modify: `backend/agents/rlm/primitives.py` (add `inspect_repository`; register in `PRIMITIVE_REGISTRY` at line 8092 + `PRIMITIVE_DESCRIPTIONS` at line 8112)
- Modify: `tests/rlm/test_registry.py` (bump `EXPECTED` 17 → 18)
- Test: `tests/rlm/test_inspect_repository.py`

**Interfaces:**
- Consumes: `ctx.project_dir` (the `repo/` lives at `ctx.project_dir / "repo"`), `RepoResolver`/`RepoProvisioner` (for `reclone_url`).
- Produces: `def inspect_repository(path: str = "", grep: str | None = None, reclone_url: str | None = None, max_bytes: int = 65536, *, ctx: "RunContext") -> dict`. Off-state returns `{"status": "disabled"}` (mirrors `read_context_map`). On-state returns `{"status": "ok", "path", "kind": "file"|"dir", "content"|"entries", "truncated"}` or `{"status": "error", "error"}`.

- [ ] **Step 1: Write the failing test**

`tests/rlm/test_inspect_repository.py`:

```python
import pytest

from backend.agents.rlm.primitives import (
    PRIMITIVE_REGISTRY,
    PRIMITIVE_DESCRIPTIONS,
    inspect_repository,
)


def test_disabled_when_flag_off(tmp_path, make_context, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    ctx = make_context(tmp_path)
    assert inspect_repository(ctx=ctx) == {"status": "disabled"}


def test_lists_dir_when_flag_on(tmp_path, make_context, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    ctx = make_context(tmp_path)
    repo = ctx.project_dir / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "train.py").write_text("print(1)\n", encoding="utf-8")
    out = inspect_repository(ctx=ctx)
    assert out["status"] == "ok"
    assert out["kind"] == "dir"
    assert "train.py" in out["entries"]


def test_reads_file_bounded(tmp_path, make_context, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    ctx = make_context(tmp_path)
    repo = ctx.project_dir / "repo"
    repo.mkdir(parents=True)
    (repo / "big.txt").write_text("A" * 100, encoding="utf-8")
    out = inspect_repository(path="big.txt", max_bytes=10, ctx=ctx)
    assert out["status"] == "ok"
    assert out["kind"] == "file"
    assert len(out["content"]) <= 10
    assert out["truncated"] is True


def test_path_escape_rejected(tmp_path, make_context, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    ctx = make_context(tmp_path)
    (ctx.project_dir / "repo").mkdir(parents=True)
    out = inspect_repository(path="../../etc/passwd", ctx=ctx)
    assert out["status"] == "error"


def test_registry_includes_inspect_repository():
    assert "inspect_repository" in PRIMITIVE_REGISTRY
    assert "inspect_repository" in PRIMITIVE_DESCRIPTIONS
    assert len(PRIMITIVE_REGISTRY) == 18
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_inspect_repository.py -v`
Expected: FAIL — `ImportError: cannot import name 'inspect_repository'`.

- [ ] **Step 3: Write minimal implementation**

In `backend/agents/rlm/primitives.py`, add the primitive near `read_context_map` (line 8072):

```python
def inspect_repository(
    path: str = "",
    grep: str | None = None,
    reclone_url: str | None = None,
    max_bytes: int = 65536,
    *,
    ctx: "RunContext",
) -> dict:
    """Bounded read of the cloned author repo at runs/<id>/repo/ (the 18th primitive).

    Off-state (OPENRESEARCH_USE_AUTHOR_REPO unset) → ``{"status": "disabled"}`` —
    mirrors read_context_map's no-op so the registry count stays stable and the
    off-state is inert. On-state: list a subtree, read a bounded file, optionally
    grep within a file, or (reclone_url) re-resolve + re-clone a different repo.
    Never raises (fail-soft): an error returns ``{"status": "error", "error": ...}``.
    Path traversal outside repo/ is refused.
    """
    import os as _os
    if _os.environ.get("OPENRESEARCH_USE_AUTHOR_REPO", "").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        return {"status": "disabled"}
    try:
        repo_dir = (ctx.project_dir / "repo").resolve()
        if reclone_url:
            from backend.services.ingestion.repo.provisioner import RepoProvisioner
            from backend.services.ingestion.repo.resolver import RepoResolver
            spec = RepoResolver.resolve(reclone_url, [], set(), None)
            manifest = RepoProvisioner.clone(spec, ctx.project_dir / "repo")
            if manifest is None:
                return {"status": "error", "error": f"reclone failed for {reclone_url}"}
            return {"status": "ok", "kind": "reclone", "commit_sha": manifest.commit_sha}

        target = (repo_dir / path).resolve() if path else repo_dir
        # Refuse traversal outside repo/.
        if repo_dir != target and repo_dir not in target.parents:
            return {"status": "error", "error": "path escapes repo/"}
        if not target.exists():
            return {"status": "error", "error": f"not found: {path or '.'}"}
        if target.is_dir():
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
            return {"status": "ok", "kind": "dir", "path": path or ".", "entries": entries[:500]}
        raw = target.read_text(encoding="utf-8", errors="replace")
        if grep:
            lines = [ln for ln in raw.splitlines() if grep in ln]
            content = "\n".join(lines)
        else:
            content = raw
        truncated = len(content.encode("utf-8")) > max_bytes
        if truncated:
            content = content.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
        return {"status": "ok", "kind": "file", "path": path, "content": content, "truncated": truncated}
    except Exception as exc:  # noqa: BLE001 — a read tool must never break the run
        return {"status": "error", "error": str(exc)[:300]}
```

Add to `PRIMITIVE_REGISTRY` (after `"read_context_map": read_context_map,` at line 8109):

```python
    "inspect_repository": inspect_repository,  # #62, OPENRESEARCH_USE_AUTHOR_REPO
```

Add to `PRIMITIVE_DESCRIPTIONS` (after the `read_context_map` entry):

```python
    "inspect_repository": "inspect_repository(path='', grep=None, reclone_url=None, "
        "max_bytes=65536) -> dict — bounded read of the authors' cloned repo at "
        "runs/<id>/repo/ (enabled by OPENRESEARCH_USE_AUTHOR_REPO). List a subtree "
        "(path='' or a dir), read a bounded file (path='train.py'), grep within a "
        "file (grep='def main'), or re-point to a different repo (reclone_url=...). "
        "Returns {status:'disabled'} when the flag is off. NAVIGATION/ADAPTATION "
        "aid — code/ is what runs; the report's evidence gate remains the backstop.",
```

In `tests/rlm/test_registry.py`, add `inspect_repository` to `EXPECTED`:

```python
    "read_context_map",  # PEEK-lite intra-run context map, OPENRESEARCH_CONTEXT_MAP
    "inspect_repository",  # repo-first reproduction (#62), OPENRESEARCH_USE_AUTHOR_REPO
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_inspect_repository.py tests/rlm/test_registry.py -v`
Expected: PASS (the registry tests now expect 18, and `build_custom_tools` advertises it unconditionally — off-state inertness is in the body).

- [ ] **Step 5: Commit (PHASE 3 boundary)**

```bash
git add backend/agents/rlm/primitives.py backend/agents/baseline_implementation.py backend/agents/prompts/baseline_implementation.py tests/rlm/test_registry.py tests/rlm/test_detect_environment_repo.py tests/rlm/test_implement_baseline_repo.py tests/rlm/test_inspect_repository.py
git commit -m "Use the cloned repo: merge deps, seed code/ in adapt mode, add inspect_repository primitive"
```

---

## Phase 4 — Measurement

### Task 9: `report.py` — attach `final_report.reproduction`

**Files:**
- Modify: `backend/agents/rlm/report.py` (`write_final_report_rlm`; attach in the serialized-dict stamp chain near the `experiment_arm`/`degradations_taken` stamps, lines 2028–2056, before `_atomic_write(json_path, json_content)` at line 2058)
- Test: `tests/rlm/test_report_reproduction.py`

**Interfaces:**
- Consumes: `_has_experiment_evidence(project_dir: Path) -> bool` (report.py:1343), `_load_repo_spec` semantics (reads `rlm_state/repo_spec.json`), `experiment_runs.jsonl` rows (`{"success": True, "metrics": {...}}`).
- Produces: a pure helper `def _build_reproduction_block(project_dir: Path) -> dict | None` returning the `reproduction` dict (mode/repo_url/commit_sha/provider/execution/adaptation) ONLY when the flag is on AND a repo was used (`repo_spec.json` has a non-null `url`); else `None`. Helper `def _adaptation_delta(repo_dir: Path, code_dir: Path) -> dict` (counts changed/added/removed files between `repo/` and `code/`).

- [ ] **Step 1: Write the failing test**

`tests/rlm/test_report_reproduction.py`:

```python
import json
from pathlib import Path

import pytest

from backend.agents.rlm.report import _build_reproduction_block, _adaptation_delta


def _write_repo_spec(project_dir: Path, url="https://github.com/me/mine", mode="adapt"):
    rlm_state = project_dir / "rlm_state"
    rlm_state.mkdir(parents=True, exist_ok=True)
    (rlm_state / "repo_spec.json").write_text(json.dumps({
        "url": url, "source": "user", "mode": mode, "reason": "t",
        "commit_sha": "abc1234", "path": str(project_dir / "repo"),
    }), encoding="utf-8")


def _write_success_experiment(project_dir: Path):
    (project_dir / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True, "metrics": {"accuracy": 0.9}, "experiment_run_id": "r1"}) + "\n",
        encoding="utf-8",
    )


def test_flag_off_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    _write_repo_spec(tmp_path)
    assert _build_reproduction_block(tmp_path) is None


def test_no_repo_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    # repo_spec.json with url=None (scratch) -> no reproduction block
    rlm_state = tmp_path / "rlm_state"
    rlm_state.mkdir(parents=True)
    (rlm_state / "repo_spec.json").write_text(json.dumps({
        "url": None, "source": "none", "mode": "scratch", "reason": "x", "commit_sha": None,
    }), encoding="utf-8")
    assert _build_reproduction_block(tmp_path) is None


def test_execution_ran_true_with_success_row(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    _write_repo_spec(tmp_path)
    _write_success_experiment(tmp_path)
    block = _build_reproduction_block(tmp_path)
    assert block is not None
    assert block["mode"] == "adapt"
    assert block["repo_url"] == "https://github.com/me/mine"
    assert block["commit_sha"] == "abc1234"
    assert block["provider"] == "github"
    assert block["execution"]["ran"] is True
    assert block["execution"]["metrics_produced"] is True
    assert block["execution"]["status"] == "success"


def test_execution_ran_false_without_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    _write_repo_spec(tmp_path)
    block = _build_reproduction_block(tmp_path)
    assert block is not None
    assert block["execution"]["ran"] is False
    assert block["execution"]["status"] == "failed"


def test_adaptation_delta_counts(tmp_path):
    repo = tmp_path / "repo"
    code = tmp_path / "code"
    (repo).mkdir(); (code).mkdir()
    (repo / "a.py").write_text("1\n", encoding="utf-8")
    (repo / "b.py").write_text("same\n", encoding="utf-8")
    (code / "a.py").write_text("CHANGED\n", encoding="utf-8")  # changed
    (code / "b.py").write_text("same\n", encoding="utf-8")     # unchanged
    (code / "c.py").write_text("new\n", encoding="utf-8")      # added
    # b.py present in both, a.py changed, c.py added, (no removed)
    delta = _adaptation_delta(repo, code)
    assert delta["files_changed"] == 1
    assert delta["files_added"] == 1
    assert delta["files_removed"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_report_reproduction.py -v`
Expected: FAIL — `ImportError: cannot import name '_build_reproduction_block'`.

- [ ] **Step 3: Write minimal implementation**

In `backend/agents/rlm/report.py`, add the helpers (near `_has_experiment_evidence`, line 1343):

```python
def _adaptation_delta(repo_dir: Path, code_dir: Path) -> dict:
    """Count files changed/added/removed between repo/ (pristine) and code/ (adapted).

    Compares by relative path + sha256 content. ``.git/`` is ignored. Pure +
    fail-soft: a missing dir yields zeros for that side.
    """
    import hashlib

    def _index(root: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        if not root.is_dir():
            return out
        for p in root.rglob("*"):
            rel = p.relative_to(root)
            if ".git" in rel.parts or not p.is_file():
                continue
            try:
                out[rel.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
            except OSError:
                continue
        return out

    repo_idx = _index(Path(repo_dir))
    code_idx = _index(Path(code_dir))
    changed = sum(1 for k in repo_idx.keys() & code_idx.keys() if repo_idx[k] != code_idx[k])
    added = len(code_idx.keys() - repo_idx.keys())
    removed = len(repo_idx.keys() - code_idx.keys())
    return {"files_changed": changed, "files_added": added, "files_removed": removed}


def _build_reproduction_block(project_dir: Path) -> dict | None:
    """Build final_report.reproduction, or None when no repo was used / flag off.

    ``execution.ran`` is sourced from the EVIDENCE layer (_has_experiment_evidence)
    so it cannot be forged by a green-looking report. Returns None unless
    OPENRESEARCH_USE_AUTHOR_REPO is on AND rlm_state/repo_spec.json carries a
    non-null url (a real repo run).
    """
    import json as _json
    import os as _os

    if _os.environ.get("OPENRESEARCH_USE_AUTHOR_REPO", "").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        return None
    project_dir = Path(project_dir)
    spec_path = project_dir / "rlm_state" / "repo_spec.json"
    if not spec_path.exists():
        return None
    try:
        spec = _json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(spec, dict) or not spec.get("url"):
        return None

    ran = _has_experiment_evidence(project_dir)
    status = "success" if ran else "failed"
    return {
        "mode": spec.get("mode") or "adapt",
        "repo_url": spec.get("url"),
        "commit_sha": spec.get("commit_sha"),
        "provider": "github",
        "execution": {
            "ran": ran,
            "status": status,
            "metrics_produced": ran,
        },
        "adaptation": _adaptation_delta(project_dir / "repo", project_dir / "code"),
    }
```

In `write_final_report_rlm`, add a stamp block in the serialized-dict chain. Insert it after the `degradations_taken` block (ends line 2056) and BEFORE `_atomic_write(json_path, json_content)` (line 2058):

```python
    # --- #62: reproduction block (execution + provenance + adaptation delta) ---
    # Same serialized-dict pattern as the stamps above — RLMFinalReport needs no
    # new field. Attached ONLY on a repo run (flag on + repo_spec.json url set);
    # omitted otherwise → byte-for-byte today. execution.ran is evidence-gated.
    try:
        _repro_block = _build_reproduction_block(project_dir)
        if _repro_block is not None:
            _d = json.loads(json_content)
            _d["reproduction"] = _repro_block
            json_content = json.dumps(_d, indent=2)
    except Exception:  # noqa: BLE001 — reproduction stamp is best-effort, never blocks
        logger.warning("report: reproduction block attach failed (non-fatal)", exc_info=True)
```

(Note: the replication axis — `reproducibility.replication_verdict` under `OPENRESEARCH_TWO_AXIS_VERDICT` — is the EXISTING machinery attached upstream at lines 1954–1995; no change needed there.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_report_reproduction.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Run the report off-state regression**

Run: `.venv/bin/python -m pytest tests/test_worker_reports.py tests/test_reports_route.py -v`
Expected: PASS — a non-repo run's `final_report.json` carries no `reproduction` key (byte-identical).

- [ ] **Step 6: Commit (PHASE 4 boundary)**

```bash
git add backend/agents/rlm/report.py tests/rlm/test_report_reproduction.py
git commit -m "Attach final_report.reproduction with the evidence-gated execution axis on repo runs"
```

---

## Phase 5 — Inputs + narration

### Task 10: `repo_url` input on the run-start surfaces

**Files:**
- Modify: `backend/services/events/live_runs.py` (`StartRunRequest` at line 175; the `common` dict in `_python_script` at ~line 1746)
- Modify: `backend/app.py` (`StartArxivRunRequest` at line 1129; `/runs/upload` form at lines 655–693; `/runs/arxiv` forward at lines 637–648)
- Modify: `backend/cli.py` (`reproduce` argparse at line 2508+; `cmd_reproduce` env threading near line 1760)
- Test: `tests/test_repo_url_inputs.py`

**Interfaces:**
- Consumes: `run_pipeline_rlm(..., repo_url=...)` (Task 5) and `OPENRESEARCH_REPO_URL` env (the fallback Task 5 reads).
- Produces: `StartRunRequest.repo_url: str | None = None`; `StartArxivRunRequest.repo_url: str | None = None`; the subprocess `common` dict carries `"repo_url"`; CLI flag `--repo-url` (`dest="repo_url"`) → `OPENRESEARCH_REPO_URL` env.

- [ ] **Step 1: Write the failing test**

`tests/test_repo_url_inputs.py`:

```python
import os

import pytest

from backend.services.events.live_runs import StartRunRequest


def test_start_run_request_accepts_repo_url():
    req = StartRunRequest(repo_url="https://github.com/me/mine")
    assert req.repo_url == "https://github.com/me/mine"


def test_start_run_request_repo_url_defaults_none():
    assert StartRunRequest().repo_url is None


def test_start_arxiv_request_accepts_repo_url():
    from backend.app import StartArxivRunRequest
    req = StartArxivRunRequest(url="https://arxiv.org/abs/2605.15155", repo_url="github:me/mine")
    assert req.repo_url == "github:me/mine"


def test_cli_parses_repo_url():
    from backend.cli import _build_parser  # the argparse factory (confirmed symbol)
    parser = _build_parser()
    ns = parser.parse_args(["reproduce", "2605.15155", "--repo-url", "github:me/mine"])
    assert ns.repo_url == "github:me/mine"


def test_python_script_threads_repo_url(tmp_path):
    from backend.services.events.live_runs import _python_script
    req = StartRunRequest(repo_url="github:me/mine")
    script = _python_script(req, project_id="prj_x", runs_root=tmp_path, uploaded_paper=None)
    # The serialized config embedded in the subprocess script carries repo_url.
    assert "repo_url" in script
    assert "github:me/mine" in script
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_repo_url_inputs.py -v`
Expected: FAIL — `pydantic_core._pydantic_core.ValidationError` / `AttributeError: ... has no attribute 'repo_url'`.

- [ ] **Step 3: Write minimal implementation**

In `backend/services/events/live_runs.py`, add to `StartRunRequest` (after the `provider_credentials` field, ~line 214):

```python
    # #62: optional official-code-repository URL (github: shorthand or full URL).
    # Threaded into the run context; resolved/cloned only when
    # OPENRESEARCH_USE_AUTHOR_REPO is on. None => existing behavior.
    repo_url: str | None = None
```

In `_python_script`, add to the `common` dict (after `"minimize_compute": ...`, ~line 1746):

```python
        # #62: official code repository URL → consumed by run_pipeline_rlm.
        "repo_url": request.repo_url,
```

Then ensure the subprocess body exports it before the pipeline runs. In the `return f"""..."""` template of `_python_script`, after `config = json.loads(...)` and before the pipeline call, add (matching the existing env-export idiom used for other config-derived env in that script — set the env var the pipeline reads):

```python
import os as _os_cfg
if config.get("repo_url"):
    _os_cfg.environ["OPENRESEARCH_REPO_URL"] = config["repo_url"]
```

(If the generated script does not already `import os`, add `import os as _os_cfg` at the top of the embedded script. This env hop is what `run_pipeline_rlm` reads as the `repo_url` fallback per Task 5.)

In `backend/app.py`, add to `StartArxivRunRequest` (after `estimate_id`, ~line 1155):

```python
    repo_url: str | None = None
```

In `/runs/upload` form parsing (the `StartRunRequest(...)` constructor, ~line 674), add:

```python
        repo_url=_optional_form_value(form, "repoUrl"),
```

In `/runs/arxiv` forward (the `StartRunRequest(...)` constructor, ~line 637), add:

```python
        repo_url=request.repo_url,
```

In `backend/cli.py`, add the flag to the `reproduce` parser (next to `--blacklist`, ~line 2508):

```python
    reproduce.add_argument(
        "--repo-url",
        dest="repo_url",
        default=None,
        help=(
            "Official code repository URL for the paper (github: shorthand or full "
            "URL). Resolved + cloned only when OPENRESEARCH_USE_AUTHOR_REPO is set; "
            "wins over an auto-discovered repo."
        ),
    )
```

In `cmd_reproduce`, thread it to env (near the blocked-terms threading, ~line 1760):

```python
    if getattr(args, "repo_url", None):
        _os.environ["OPENRESEARCH_REPO_URL"] = args.repo_url
```

(The CLI parser factory is `backend.cli._build_parser` — confirmed against `tests/test_cli_scope_spec.py`, which imports and calls it the same way.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_repo_url_inputs.py -v`
Expected: PASS (5 passed).

### Task 11: SSE narration + repo-aware system prompt

**Files:**
- Modify: `backend/agents/rlm/sse_bridge.py` (add `build_repo_resolved_event` / `build_repo_cloned_event` near `build_run_complete_event` at line 511; add both to `__all__` at line 918)
- Modify: `backend/agents/rlm/run.py` (`_resolve_and_clone_repo` from Task 5 — emit the two events when a repo resolves/clones; the `emit` chokepoint is available via `make_emit`/`ctx.emit`)
- Modify: `backend/agents/rlm/system_prompt.py` (`build_system_prompt` at line 553 — add `_REPO_AWARE_SECTION`, flag-gated)
- Test: `tests/rlm/test_repo_sse_and_prompt.py`

**Interfaces:**
- Consumes: `_now_iso()` (sse_bridge), `build_system_prompt(*, context_metadata, root_model, include_hints=True)` (system_prompt).
- Produces:
  - `def build_repo_resolved_event(*, url: str | None, source: str, mode: str, reason: str) -> dict`.
  - `def build_repo_cloned_event(*, commit_sha: str | None, size_mb: float, key_files: list[str]) -> dict`.
  - `_REPO_AWARE_SECTION` string appended to the prompt iff `OPENRESEARCH_USE_AUTHOR_REPO` is on.

- [ ] **Step 1: Write the failing test**

`tests/rlm/test_repo_sse_and_prompt.py`:

```python
import pytest

from backend.agents.rlm.sse_bridge import (
    build_repo_resolved_event,
    build_repo_cloned_event,
)
from backend.agents.rlm.system_prompt import build_system_prompt


def test_repo_resolved_event_shape():
    ev = build_repo_resolved_event(
        url="https://github.com/me/mine", source="user", mode="adapt", reason="r",
    )
    assert ev["event"] == "repo_resolved"
    assert ev["url"] == "https://github.com/me/mine"
    assert ev["source"] == "user"
    assert ev["mode"] == "adapt"
    assert "timestamp" in ev


def test_repo_cloned_event_shape():
    ev = build_repo_cloned_event(commit_sha="abc1234", size_mb=1.5, key_files=["README.md"])
    assert ev["event"] == "repo_cloned"
    assert ev["commit_sha"] == "abc1234"
    assert ev["size_mb"] == 1.5
    assert ev["key_files"] == ["README.md"]
    assert "timestamp" in ev


def _ctx_meta():
    return {"context": {"type": "str", "length": 10}}


def _root_model():
    from backend.agents.rlm.models import resolve_root_model
    return resolve_root_model("gpt-5")


def test_prompt_omits_repo_section_when_flag_off(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    prompt = build_system_prompt(context_metadata=_ctx_meta(), root_model=_root_model())
    assert "AUTHOR REPOSITORY" not in prompt


def test_prompt_includes_repo_section_when_flag_on(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    prompt = build_system_prompt(context_metadata=_ctx_meta(), root_model=_root_model())
    assert "AUTHOR REPOSITORY" in prompt
    assert "repo_files" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_repo_sse_and_prompt.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_repo_resolved_event'`.

- [ ] **Step 3: Write minimal implementation**

In `backend/agents/rlm/sse_bridge.py`, add the two builders (after `build_run_complete_event`, ~line 536):

```python
def build_repo_resolved_event(
    *,
    url: str | None,
    source: str,
    mode: str,
    reason: str,
) -> dict:
    """Build a ``repo_resolved`` control event (corpus-free). #62."""
    return {
        "event": "repo_resolved",
        "timestamp": _now_iso(),
        "url": url,
        "source": source,
        "mode": mode,
        "reason": reason,
    }


def build_repo_cloned_event(
    *,
    commit_sha: str | None,
    size_mb: float,
    key_files: list[str],
) -> dict:
    """Build a ``repo_cloned`` control event (corpus-free). #62."""
    return {
        "event": "repo_cloned",
        "timestamp": _now_iso(),
        "commit_sha": commit_sha,
        "size_mb": size_mb,
        "key_files": list(key_files or []),
    }
```

Add both names to `__all__` (line 918), in alphabetical position:

```python
    "build_repo_cloned_event",
    "build_repo_resolved_event",
```

In `backend/agents/rlm/run.py`, emit the events from `_resolve_and_clone_repo` (Task 5). Change its signature to accept an optional `emit` callback and call the builders. Update the function header to:

```python
def _resolve_and_clone_repo(
    project_dir: "Path",
    repo_url: str | None,
    blacklist: "set[str]",
    discovered: "list[Any]",
    emit: "Any" = None,
) -> "tuple[dict[str, Any] | None, Any]":
```

Inside, right after `spec = RepoResolver.resolve(...)`, add:

```python
        if emit is not None:
            try:
                from backend.agents.rlm.sse_bridge import build_repo_resolved_event
                emit(build_repo_resolved_event(
                    url=spec.url, source=spec.source, mode=spec.mode, reason=spec.reason,
                ))
            except Exception:  # noqa: BLE001 — narration is best-effort
                pass
```

And right after a successful `manifest is not None` clone (where `repo_files`/`commit_sha` are set), add:

```python
                if emit is not None:
                    try:
                        from backend.agents.rlm.sse_bridge import build_repo_cloned_event
                        emit(build_repo_cloned_event(
                            commit_sha=manifest.commit_sha,
                            size_mb=manifest.size_mb,
                            key_files=list(manifest.key_files.keys()),
                        ))
                    except Exception:  # noqa: BLE001
                        pass
```

In `_build_context` (Task 5), forward the emit: change the body call to pass `emit`. Since `_build_context` does not currently receive `emit`, thread it: add `emit: "Any" = None` to `_build_context`'s keyword args and pass `emit=emit` into `_resolve_and_clone_repo`. At the call site in `run_pipeline_rlm` (line 2791), pass `emit=emit` (the thread-safe emit chokepoint already built at line 2565).

In `backend/agents/rlm/system_prompt.py`, define the section (near the other section constants, e.g. by `_CONTEXT_MAP_SECTION`):

```python
_REPO_AWARE_SECTION = """\
═══════════════════════════════════════════════════════════════
  AUTHOR REPOSITORY (repo-first reproduction)
═══════════════════════════════════════════════════════════════

The paper's official code repository has been cloned for you. Its constant-size
manifest is in your `repo_files` context variable (file tree + key-file
excerpts + commit SHA). PREFER the authors' code over a from-scratch rewrite:

- Consult `repo_files` first to orient on the real implementation.
- Use the `inspect_repository(path=..., grep=...)` primitive to deep-read any
  file in the repo when you need exact details.
- In adapt mode the harness has already seeded the authors' code into your
  working `code/` directory — ADAPT it (fix env/scope/entrypoints) rather than
  rewriting; `implement_baseline` continues that adaptation.
- Narrate repo discovery, clone, and inspection in your reasoning so the run
  trace is transparent.
"""
```

In `build_system_prompt`, add the flag-gated append (mirror the `_CONTEXT_MAP_SECTION` pattern, after the `include_hints` block, ~line 616):

```python
    # #62: repo-aware guidance only when OPENRESEARCH_USE_AUTHOR_REPO is on.
    if _os.environ.get("OPENRESEARCH_USE_AUTHOR_REPO", "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        parts.append(_REPO_AWARE_SECTION)
```

(`import os as _os` is already present in `build_system_prompt` per the existing context-map block.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_repo_sse_and_prompt.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the system-prompt + sse regression**

Run: `.venv/bin/python -m pytest tests/rlm/ -k "system_prompt or sse or sanitize" -v`
Expected: PASS — the prompt is byte-identical when the flag is off; the new builders don't touch the egress sanitizer.

- [ ] **Step 6: Commit (PHASE 5 boundary)**

```bash
git add backend/services/events/live_runs.py backend/app.py backend/cli.py backend/agents/rlm/sse_bridge.py backend/agents/rlm/run.py backend/agents/rlm/system_prompt.py tests/test_repo_url_inputs.py tests/rlm/test_repo_sse_and_prompt.py
git commit -m "Accept repo-url on run-start surfaces and narrate repo resolve/clone with repo-aware prompt"
```

### Task 12 (OPTIONAL — SEPARABLE; do LAST; backend works without it): frontend repo-URL field

**Files:**
- Modify: `frontend/src/components/lab/upload-view.tsx` (add an optional "Official code repository (optional)" text input that posts `repoUrl`)
- Modify: the demo proxy route that forwards the upload form (`frontend/src/app/api/demo/route.ts` or the arxiv proxy — confirm which proxy serves the run-start POST) + the relevant TS request type
- Test: `frontend/src/components/lab/__tests__/upload-view.repo-url.test.tsx` (vitest)

**Interfaces:**
- Consumes: the backend `repoUrl` form field (Task 10).
- Produces: a controlled input whose value is included in the multipart/form POST body as `repoUrl`.

> **Frontend test env (per CLAUDE.md memory):** system Node v21 is broken for vitest — use nvm v22.14.0; run the lab suite with `--no-file-parallelism`. Run frontend commands from `frontend/` after `npm ci`.

- [ ] **Step 1: Write the failing test**

`frontend/src/components/lab/__tests__/upload-view.repo-url.test.tsx`:

```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { UploadView } from "../upload-view";

describe("UploadView repo-url field", () => {
  it("renders the optional repository input", () => {
    render(<UploadView />);
    expect(
      screen.getByLabelText(/Official code repository/i)
    ).toBeInTheDocument();
  });

  it("includes repoUrl in the submitted form body", async () => {
    const fetchSpy = vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ projectId: "prj_x" }), { status: 202 })
    );
    render(<UploadView />);
    fireEvent.change(screen.getByLabelText(/Official code repository/i), {
      target: { value: "github:me/mine" },
    });
    // Trigger the existing submit path (a real PDF select + start may be mocked
    // in the existing suite — reuse that helper). Assert the FormData carried repoUrl.
    // (The exact submit trigger mirrors the existing upload-view tests.)
    const body = fetchSpy.mock.calls.find(
      ([, init]) => init?.body instanceof FormData
    )?.[1]?.body as FormData | undefined;
    if (body) {
      expect(body.get("repoUrl")).toBe("github:me/mine");
    }
    fetchSpy.mockRestore();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `frontend/`): `npx vitest run src/components/lab/__tests__/upload-view.repo-url.test.tsx --no-file-parallelism`
Expected: FAIL — the input labeled "Official code repository" does not exist yet.

- [ ] **Step 3: Write minimal implementation**

In `frontend/src/components/lab/upload-view.tsx` (near the existing form fields, ~lines 280–315): add a controlled state + input, and include it in the FormData/JSON body that the submit handler posts.

```tsx
// near the other useState hooks
const [repoUrl, setRepoUrl] = useState("");

// in the JSX, alongside the other optional fields:
<label htmlFor="repo-url-input">Official code repository (optional)</label>
<input
  id="repo-url-input"
  type="text"
  placeholder="https://github.com/owner/repo"
  value={repoUrl}
  onChange={(e) => setRepoUrl(e.target.value)}
/>

// in the submit handler, when building FormData:
if (repoUrl.trim()) {
  formData.append("repoUrl", repoUrl.trim());
}
```

If the run-start is posted as JSON (arxiv path) rather than multipart, add `repoUrl` to the JSON body and to the TS request type (e.g. the `StartRunBody`/`ArxivRunBody` interface in the proxy route or a shared types file). The proxy route (`frontend/src/app/api/demo/*`) forwards the body to the backend unchanged — confirm it does not strip unknown fields; if it allowlists fields, add `repoUrl` to that allowlist.

- [ ] **Step 4: Run test to verify it passes**

Run (from `frontend/`): `npx vitest run src/components/lab/__tests__/upload-view.repo-url.test.tsx --no-file-parallelism`
Expected: PASS.

- [ ] **Step 5: Type-check + lint the frontend change**

Run (from `frontend/`): `npx tsc --noEmit && npm run lint`
Expected: PASS (no type or lint errors).

- [ ] **Step 6: Commit (Task 12 boundary)**

```bash
git add frontend/src/components/lab/upload-view.tsx frontend/src/components/lab/__tests__/upload-view.repo-url.test.tsx frontend/src/app/api/demo
git commit -m "Add the optional official-code-repository input to the lab upload view"
```

---

## Phase 6 — Multi-cloud transport

### Task 13: Exclude `repo/` from blob/SFTP upload sets

**Files:**
- Modify: `backend/services/runtime/azure_blob.py` (`_EXCLUDED_DIR_PARTS` at line 49)
- Modify: `backend/services/runtime/gcs_blob.py` (`_EXCLUDED_DIR_PARTS` at line 55)
- Modify: `backend/services/runtime/runpod_backend.py` (`_upload_directory` walk at lines 843–857 — add an exclusion filter mirroring the blob `_EXCLUDED_DIR_PARTS`)
- Test: `tests/runtime/test_repo_upload_exclusion.py`

**Interfaces:**
- Consumes: `azure_blob.upload_prefix(local_root, *, blob_prefix, account_name, container_name, client) -> list[str]` and `gcs_blob.upload_prefix(...)` (both accept a duck-typed fake client and return the sorted uploaded-name list — testable with no cloud). `azure_blob._is_excluded(rel_parts)` / `gcs_blob._is_excluded(rel_parts)`.
- Produces: `"repo"` added to both blob `_EXCLUDED_DIR_PARTS`; a `_RUNPOD_EXCLUDED_DIR_PARTS = frozenset({"outputs", ".git", "__pycache__", ".venv", "repo"})` + a guard in `runpod_backend._upload_directory`.

- [ ] **Step 1: Write the failing test**

`tests/runtime/test_repo_upload_exclusion.py`:

```python
from pathlib import Path

import pytest


def _make_tree(root: Path):
    (root / "code").mkdir()
    (root / "code" / "train.py").write_text("x\n", encoding="utf-8")
    (root / "repo" / "src").mkdir(parents=True)
    (root / "repo" / "src" / "model.py").write_text("y\n", encoding="utf-8")
    (root / "repo" / "README.md").write_text("z\n", encoding="utf-8")


class _FakeContainer:
    def __init__(self):
        self.blobs: dict[str, bytes] = {}

    def upload_blob(self, name, data, overwrite=True):
        self.blobs[name] = data


def test_azure_excludes_repo(tmp_path):
    from backend.services.runtime import azure_blob
    assert "repo" in azure_blob._EXCLUDED_DIR_PARTS
    _make_tree(tmp_path)
    client = _FakeContainer()
    uploaded = azure_blob.upload_prefix(
        tmp_path, blob_prefix="runs/x", account_name="a", container_name="c", client=client,
    )
    assert any("code/train.py" in n for n in uploaded)
    assert not any("/repo/" in n or n.endswith("/repo") for n in uploaded)


def test_gcs_excludes_repo(tmp_path):
    from backend.services.runtime import gcs_blob
    assert "repo" in gcs_blob._EXCLUDED_DIR_PARTS


def test_runpod_walk_skips_repo(tmp_path):
    from backend.services.runtime.runpod_backend import _runpod_upload_relpaths
    _make_tree(tmp_path)
    rels = _runpod_upload_relpaths(tmp_path)
    assert "code/train.py" in rels
    assert all(not r.startswith("repo/") and r != "repo" for r in rels)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/runtime/test_repo_upload_exclusion.py -v`
Expected: FAIL — `assert "repo" in azure_blob._EXCLUDED_DIR_PARTS` fails (repo not yet excluded), and `_runpod_upload_relpaths` does not exist.

- [ ] **Step 3: Write minimal implementation**

In `backend/services/runtime/azure_blob.py`, line 49:

```python
_EXCLUDED_DIR_PARTS: frozenset[str] = frozenset(
    {"outputs", ".git", "__pycache__", ".venv", "repo"}
)
```

In `backend/services/runtime/gcs_blob.py`, line 55:

```python
_EXCLUDED_DIR_PARTS: frozenset[str] = frozenset(
    {"outputs", ".git", "__pycache__", ".venv", "repo"}
)
```

In `backend/services/runtime/runpod_backend.py`, add a module-level constant + a pure relpath helper (so the walk is unit-testable without SFTP), and use the helper inside `_upload_directory`:

```python
_RUNPOD_EXCLUDED_DIR_PARTS = frozenset({"outputs", ".git", "__pycache__", ".venv", "repo"})


def _runpod_upload_relpaths(local_root: "Path") -> list[str]:
    """Pure: the sorted relative POSIX paths the SFTP walk would upload, with the
    excluded dirs (incl. repo/) filtered out. #62 — keeps repo/ host-only."""
    from pathlib import Path as _P
    local_root = _P(local_root)
    out: list[str] = []
    for p in sorted(local_root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(local_root)
        if any(part in _RUNPOD_EXCLUDED_DIR_PARTS for part in rel.parts[:-1]):
            continue
        out.append(rel.as_posix())
    return out
```

Then in `_upload_directory` (lines 843–857), filter using the same exclusion. Replace the loop body's per-file upload with an exclusion guard:

```python
    async def _upload_directory(
        self,
        sftp: Any,
        local_root: Path,
        remote_root: str,
    ) -> None:
        await sftp.makedirs(remote_root, exist_ok=True)
        for local_path in sorted(local_root.rglob("*")):
            rel = local_path.relative_to(local_root)
            # #62: keep repo/ (and the other build-only dirs) host-only.
            if any(part in _RUNPOD_EXCLUDED_DIR_PARTS for part in rel.parts[:-1]):
                continue
            rel_posix = rel.as_posix()
            remote_path = _join_posix(remote_root, rel_posix)
            if local_path.is_dir():
                await sftp.makedirs(remote_path, exist_ok=True)
            elif local_path.is_file():
                await sftp.makedirs(str(PurePosixPath(remote_path).parent), exist_ok=True)
                await sftp.put(str(local_path), remote_path)
```

(Note: per spec Open Item #3, RunPod's `project_root` may be `runs/<id>/code/` — in which case a sibling `repo/` never rides anyway and this exclusion is a safety net. The exclusion is correct either way; the test asserts the walk filter directly.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/runtime/test_repo_upload_exclusion.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the runtime/blob regression**

Run: `.venv/bin/python -m pytest tests/ -k "azure_blob or gcs_blob or runpod" -v`
Expected: PASS — adding `"repo"` to the exclusion set doesn't change any existing upload assertion (no existing fixture uses a `repo/` dir).

- [ ] **Step 6: Commit (PHASE 6 boundary)**

```bash
git add backend/services/runtime/azure_blob.py backend/services/runtime/gcs_blob.py backend/services/runtime/runpod_backend.py tests/runtime/test_repo_upload_exclusion.py
git commit -m "Keep runs/<id>/repo/ host-only by excluding it from every cloud upload set"
```

---

## Phase 7 — Docs

### Task 14: Update `CLAUDE.md` + `system_overview.md` + add the spec

**Files:**
- Modify: `CLAUDE.md` (primitive count 17→18; new SSE event types; feature-flag block; `runs/<id>/repo/`)
- Modify: `system_overview.md` (data-flow drift)
- Add to the commit: `docs/history/specs/2026-06-21-github-repo-first-reproduction-design.md` (the approved spec, already on disk) + this plan file
- Verify: grep + `tests/rlm/test_registry.py`

**Interfaces:** none (docs only; no feature-code tests).

- [ ] **Step 1: Update the primitive count in `CLAUDE.md`**

In the "The RLM orchestrator" section, change "the **17 primitives**" → "the **18 primitives**", also change "Plus 5 operational/aux helpers" → "Plus 6 operational/aux helpers" and add `inspect_repository` to that AUX enumeration — it mirrors `read_context_map` (a flag-gated read/navigation aid that returns `{"status":"disabled"}` when off), so it belongs with the aux helpers, not the core domain primitives. Net: 12 core + 6 aux = 18. Add `inspect_repository` to the enumerated primitive list with a one-line description:

```
`inspect_repository` (repo-first reproduction, OPENRESEARCH_USE_AUTHOR_REPO; returns {"status":"disabled"} unless on)
```

Update the "Keep this count and `tests/rlm/test_registry.py`'s `EXPECTED` in sync" note (already present — just verify it now reads 18).

- [ ] **Step 2: Add the SSE event types**

In the "UI ↔ backend run lifecycle" section's SSE event-type list (item 4), append `repo_resolved` and `repo_cloned` to the RLM emits list.

- [ ] **Step 3: Add the feature-flag block**

In the "Feature flags" section, add a block (mirroring the existing 2026-06-20 entry style):

```
- **GitHub-repo-first reproduction (2026-06-21, all default-OFF; spec `2026-06-21-github-repo-first-reproduction-design.md`)** — finishes #62: when a paper links an official repo, resolve + clone it into `runs/<id>/repo/` (pristine, commit-SHA pinned), expose a constant-size manifest as the root's `repo_files`, and adapt it into `code/`. **`OPENRESEARCH_USE_AUTHOR_REPO`** (master, off) — off ⇒ no resolve/clone, `repo_files` stays `None`, `inspect_repository` returns `{"status":"disabled"}`, `detect_environment`/`implement_baseline`/report unchanged, no `reproduction` stamp (byte-identical). **`OPENRESEARCH_REPRODUCTION_MODE`** (`adapt`) — `reference` forces clean-room (clone+read, reimplement in `code/`). **`OPENRESEARCH_REPO_CLONE_TIMEOUT_S`** (300) / **`OPENRESEARCH_REPO_CLONE_MAX_MB`** (2048) / **`OPENRESEARCH_REPO_CLONE_LFS`** (off ⇒ `GIT_LFS_SKIP_SMUDGE=1`). Resolve+clone live in the keystone `run.py::_build_context()` (RepoResolver + RepoProvisioner, `backend/services/ingestion/repo/`), persisting the trusted `rlm_state/repo_spec.json`. `final_report.reproduction{mode,repo_url,commit_sha,provider,execution{ran,...},adaptation{...}}` is attached on a repo run (`execution.ran` ← `_has_experiment_evidence`, evidence-gated); the replication axis is the existing `reproducibility.replication_verdict` under `OPENRESEARCH_TWO_AXIS_VERDICT`. `repo/` is host-only (excluded from every cloud upload set) so only `code/` crosses to a GPU backend. New SSE events `repo_resolved`/`repo_cloned`. The 18th primitive `inspect_repository` deep-reads the repo. Infra precondition: the orchestrator pod needs egress to `github.com` (a blocked clone fails-soft → from-scratch). Validate on SDAR before flipping default-ON.
```

- [ ] **Step 4: Add `runs/<id>/repo/` to the run-state list**

In the "File-backed run state" section, add a bullet:

```
- `repo/` — the paper's cloned official repository (pristine reference; orchestrator-host-only, never shipped to a GPU backend), when `OPENRESEARCH_USE_AUTHOR_REPO` is on
- `rlm_state/repo_spec.json` — the resolved RepoSpec + commit_sha (the deterministic trusted source `implement_baseline` + the report writer read)
```

- [ ] **Step 5: Update `system_overview.md` data-flow drift**

Add a short paragraph to the architecture/data-flow section describing the repo-first path: discovery → `RepoResolver` → `RepoProvisioner.clone` → `_build_context` exposes `repo_files` + persists `repo_spec.json` → `detect_environment` merges repo deps → `implement_baseline` seeds `code/` (adapt) → `final_report.reproduction` (execution axis) + existing `replication_verdict`. Keep it to the "why/where", not a code recap (per the doc policy). Note `repo/` is host-only across all sandboxes.

- [ ] **Step 6: Verify the docs + registry are consistent**

Run:
```bash
grep -n "18 primitives\|inspect_repository\|repo_resolved\|repo_cloned\|OPENRESEARCH_USE_AUTHOR_REPO\|runs/<id>/repo" CLAUDE.md
.venv/bin/python -m pytest tests/rlm/test_registry.py tests/test_claude_md_fidelity.py -v
```
Expected: grep shows the new entries; `test_registry.py` passes at 18; `test_claude_md_fidelity.py` (if it cross-checks the primitive count) passes.

- [ ] **Step 7: Commit (PHASE 7 boundary)**

```bash
git add CLAUDE.md system_overview.md docs/history/specs/2026-06-21-github-repo-first-reproduction-design.md docs/history/plans/2026-06-21-github-repo-first-reproduction.md
git commit -m "Document repo-first reproduction: 18 primitives, new SSE events, flags, and repo/ layout"
```

---

## Final verification (after all phases)

- [ ] **Run the full affected test set**

```bash
.venv/bin/python -m pytest tests/services/ingestion/repo/ tests/rlm/ tests/runtime/test_repo_upload_exclusion.py tests/config/test_repo_flags.py tests/test_repo_url_inputs.py -v
```
Expected: all PASS.

- [ ] **Run a broad off-state regression to prove byte-identical-when-unset**

```bash
env -u OPENRESEARCH_USE_AUTHOR_REPO .venv/bin/python -m pytest tests/rlm/ tests/test_worker_reports.py tests/test_issue25_baseline.py -v
```
Expected: all PASS — with the master flag unset, nothing in the existing suites changes behavior.

- [ ] **Lint**

```bash
uvx ruff@0.15.16 check backend/services/ingestion/repo/ backend/agents/rlm/primitives.py backend/agents/rlm/run.py backend/agents/rlm/report.py
```
Expected: no errors.

## Self-review checklist (run before handoff)

- Spec coverage: §5.2 → Tasks 2/3/4; §5.3 → Task 5; §5.4 → Tasks 6/7; §5.5 → Task 8; §5.6 → Task 9; §5.7 → Tasks 10/11/12; §5.8 → Task 13; §5.10 → Task 1; §8 docs → Task 14.
- Off-state regression test present in Tasks 5, 6, 7, 8, 9, 11, 13 (and Task 1 covers the flag defaults).
- Type consistency: `RepoSpec`/`RepoManifest`/`RepoResolver`/`RepoProvisioner` signatures match across Tasks 2–9; `_load_repo_spec`/`_repo_artifact_index` names consistent between Tasks 7 and 9.
- No placeholders: every code step shows complete code; every test step shows the real test.

## Execution Handoff

Plan complete and saved to `docs/history/plans/2026-06-21-github-repo-first-reproduction.md`. Two execution options:

1. Subagent-Driven (recommended) — dispatch a fresh subagent per task, review between tasks. REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
2. Inline Execution — execute tasks in this session with checkpoints. REQUIRED SUB-SKILL: superpowers:executing-plans.
