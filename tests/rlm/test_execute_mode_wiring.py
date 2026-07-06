"""T11: fail loudly when execute mode is requested but repo_spec stamps
something other than execute (B2 backstop) — e.g. the resolver found no
usable repo and downgraded to "scratch", or repo resolution raised entirely.
"""
import pytest

from backend.agents.rlm.run import assert_execute_mode_stamped


def test_raises_when_execute_requested_but_adapt_stamped():
    with pytest.raises(RuntimeError, match="execute"):
        assert_execute_mode_stamped("execute", {"mode": "adapt", "clone_succeeded": True})


def test_raises_when_execute_requested_but_scratch_stamped():
    """The real downgrade path: RepoResolver.resolve() falls back to mode="scratch"
    when nothing usable was found, even though execute was requested."""
    with pytest.raises(RuntimeError, match="execute"):
        assert_execute_mode_stamped("execute", {"mode": "scratch", "clone_succeeded": False})


def test_raises_when_execute_requested_but_repo_spec_is_none():
    """Total resolution failure (_resolve_and_clone_repo's except branch) -> None."""
    with pytest.raises(RuntimeError, match="execute"):
        assert_execute_mode_stamped("execute", None)


def test_ok_when_execute_stamped():
    assert assert_execute_mode_stamped(
        "execute", {"mode": "execute", "clone_succeeded": True}
    ) is None


def test_noop_when_not_execute_requested():
    assert assert_execute_mode_stamped("adapt", {"mode": "adapt"}) is None
    assert assert_execute_mode_stamped("", None) is None


def test_noop_when_reference_mode_requested():
    assert assert_execute_mode_stamped("reference", {"mode": "reference"}) is None
