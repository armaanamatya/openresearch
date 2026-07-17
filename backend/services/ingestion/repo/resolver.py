"""Pure resolution of the reproduction's source repository.

No IO. Given a user-provided URL, the paper's discovered repository artifacts,
a blacklist, and an optional mode override, decide WHICH repo (if any) to use
and in WHICH mode (adapt / reference / execute / scratch). The blacklist preserves
the existing "blocked = do not use" semantics: a resolved URL on the blacklist is
DROPPED (treated as not-found) and the run proceeds scratch.

The returned ``RepoSpec.mode`` is the **RESOLVED** mode — what will actually run —
never the raw request. That matters for the ``auto`` override (see ``resolve``):
``auto`` is a per-paper *request* that resolves to ``execute`` when a repo is
usable and to ``scratch`` when none is, and the whole harness downstream keys off
the resolved mode persisted to ``rlm_state/repo_spec.json``.
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
        """Resolve the repo decision. ``mode_override`` is the REQUEST; ``RepoSpec.mode``
        is the RESOLVED outcome.

        ``auto`` (the triage-funnel mode) resolves per-paper: a usable repo means
        ``execute`` (run the authors' published code — the high-evidence path), and no
        usable repo falls through to the terminal ``scratch`` branch below (an honest
        from-scratch attempt). The caller discloses that fallback loudly; a triage
        funnel's expensive error is a FALSE NEGATIVE, so a paper that published no
        code must still get a real attempt rather than hard-failing the run.

        ``adapt`` (default) / ``reference`` / ``execute`` keep their exact prior
        behavior — the request is stamped through verbatim.
        """
        _mode_norm = (mode_override or "").strip().lower()
        if _mode_norm == "auto":
            mode = "execute"
        elif _mode_norm in ("reference", "execute"):
            mode = _mode_norm
        else:
            mode = "adapt"

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
