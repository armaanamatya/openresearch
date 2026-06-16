"""2026-06-15: a corrupt system ``claude`` (a botched npm update symlinked it to a
Windows ``.exe``) made ``shutil.which("claude")`` find a non-executable binary, so
OAuth detection falsely returned False and EVERY run died at model resolution. The
harness now falls back to the claude-agent-sdk's OWN bundled claude binary, so a
broken/missing system install can't hide a valid subscription.
"""

from __future__ import annotations

import os

from backend.agents.runtime import factory


def test_bundled_claude_path_resolves_to_executable():
    p = factory._bundled_claude_path()
    assert p is not None, "the SDK's bundled claude should be detectable in this env"
    assert os.path.isfile(p) and os.access(p, os.X_OK)


def test_oauth_detected_via_bundled_when_system_claude_broken(monkeypatch, tmp_path):
    """Creds present + system `claude` gone (which→None) → bundled fallback → True."""
    creds = tmp_path / ".claude" / ".credentials.json"
    creds.parent.mkdir(parents=True)
    creds.write_text('{"claudeAiOauth": {}}', encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(factory.shutil, "which", lambda _name: None)  # system claude broken
    monkeypatch.setattr(factory.sys, "platform", "linux")
    assert factory._has_claude_subscription_oauth() is True


def test_false_when_no_creds_no_bundle(monkeypatch, tmp_path):
    """No creds file → False regardless of bundled (don't claim a session that isn't there)."""
    monkeypatch.setenv("HOME", str(tmp_path))  # empty home, no creds
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(factory.shutil, "which", lambda _name: None)
    monkeypatch.setattr(factory, "_bundled_claude_path", lambda: None)
    monkeypatch.setattr(factory.sys, "platform", "linux")
    assert factory._has_claude_subscription_oauth() is False
