"""Tests for --max-pod-seconds CLI flag wiring."""

from __future__ import annotations

from backend.cli import _build_parser, _resolve_max_pod_seconds


def test_max_pod_seconds_flag_is_recognized():
    parser = _build_parser()
    args = parser.parse_args(["reproduce", "dummy.pdf", "--max-pod-seconds", "1800"])
    assert args.max_pod_seconds == 1800.0


def test_max_pod_seconds_defaults_to_none():
    parser = _build_parser()
    args = parser.parse_args(["reproduce", "dummy.pdf"])
    assert args.max_pod_seconds is None


def test_sanity_flag_is_recognized():
    parser = _build_parser()
    args = parser.parse_args(["reproduce", "2512.24601", "--sanity"])
    assert args.sanity is True


def test_resolve_max_pod_seconds_prefers_cli_over_env(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_MAX_POD_SECONDS", "3600")
    assert _resolve_max_pod_seconds(1800.0) == 1800.0


def test_resolve_max_pod_seconds_falls_back_to_env_when_cli_none(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_MAX_POD_SECONDS", "3600")
    assert _resolve_max_pod_seconds(None) == 3600.0


def test_resolve_max_pod_seconds_returns_none_when_neither_set(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_MAX_POD_SECONDS", raising=False)
    assert _resolve_max_pod_seconds(None) is None


def test_resolve_max_pod_seconds_honors_explicit_zero_kill_switch(monkeypatch):
    """An explicit --max-pod-seconds 0 must NOT silently fall through to env.

    Zero is a legitimate "block-immediately" kill-switch value: check_pod_seconds
    raises when elapsed >= max_pod_seconds, so cap=0 fires on the first exec.
    Regression guard for the `or`-trap fixed in commit d250578.
    """
    monkeypatch.setenv("OPENRESEARCH_MAX_POD_SECONDS", "3600")
    assert _resolve_max_pod_seconds(0.0) == 0.0
