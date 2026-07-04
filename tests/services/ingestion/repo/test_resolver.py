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


def test_mode_override_execute():
    spec = RepoResolver.resolve(
        user_url="github:me/mine",
        discovered=[],
        blacklist=set(),
        mode_override="execute",
    )
    assert spec.url == "https://github.com/me/mine"
    assert spec.mode == "execute"


def test_mode_override_normalizes_case_and_padding():
    spec = RepoResolver.resolve(
        user_url="github:me/mine", discovered=[], blacklist=set(), mode_override="  EXECUTE  ",
    )
    assert spec.mode == "execute"
    spec = RepoResolver.resolve(
        user_url="github:me/mine", discovered=[], blacklist=set(), mode_override="  Reference  ",
    )
    assert spec.mode == "reference"


def test_mode_override_unknown_value_falls_back_to_adapt():
    spec = RepoResolver.resolve(
        user_url="github:me/mine", discovered=[], blacklist=set(), mode_override="scratch-only",
    )
    assert spec.mode == "adapt"


def test_no_repo_yields_scratch():
    spec = RepoResolver.resolve(
        user_url=None, discovered=[], blacklist=set(), mode_override=None,
    )
    assert spec == RepoSpec(url=None, source="none", mode="scratch", reason=spec.reason)
