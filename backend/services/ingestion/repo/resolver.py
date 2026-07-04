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
    ``mode`` is ``adapt`` | ``reference`` | ``execute`` | ``scratch``.
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
        _mode_norm = (mode_override or "").strip().lower()
        mode = _mode_norm if _mode_norm in ("reference", "execute") else "adapt"

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
