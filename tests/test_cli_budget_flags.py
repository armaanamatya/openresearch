"""Tests for --max-pod-seconds CLI flag wiring."""

from __future__ import annotations

from backend.cli import _build_parser


def test_max_pod_seconds_flag_is_recognized():
    parser = _build_parser()
    args = parser.parse_args(["reproduce", "dummy.pdf", "--max-pod-seconds", "1800"])
    assert args.max_pod_seconds == 1800.0


def test_max_pod_seconds_defaults_to_none():
    parser = _build_parser()
    args = parser.parse_args(["reproduce", "dummy.pdf"])
    assert args.max_pod_seconds is None
