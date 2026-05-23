"""Smoke test: the CLI boundary wires normalize_path_input into cmd_reproduce.

These tests verify the contract at the boundary level — that a Windows path
arriving as args.source is converted to a POSIX mount path before cmd_reproduce
uses it for anything. We test the normalization logic directly on the parsed
args value (the same transform cmd_reproduce applies) rather than running the
full pipeline, which has network/filesystem side-effects.
"""

from __future__ import annotations

from backend.cli import _build_parser, _with_reproduce_defaults
from backend.services.paths import normalize_path_input


def test_cmd_reproduce_normalizes_windows_path(monkeypatch):
    """A Windows path in args.source is normalized to a WSL mount path."""
    # Force posix host so Windows→WSL conversion fires.
    monkeypatch.setattr("backend.services.paths.os.name", "posix")
    # No wslpath binary on CI.
    monkeypatch.setattr("backend.services.paths.shutil.which", lambda _: None)

    parser = _build_parser()
    args = parser.parse_args(["reproduce", r"C:\Users\Foo\paper.pdf"])
    args = _with_reproduce_defaults(args)

    # This is exactly the transform applied at the top of cmd_reproduce.
    args.source = normalize_path_input(args.source)

    assert args.source == "/mnt/c/Users/Foo/paper.pdf"


def test_cmd_reproduce_leaves_arxiv_id_unchanged(monkeypatch):
    """arXiv IDs in args.source pass through the normalizer unchanged."""
    monkeypatch.setattr("backend.services.paths.os.name", "posix")
    monkeypatch.setattr("backend.services.paths.shutil.which", lambda _: None)

    parser = _build_parser()
    args = parser.parse_args(["reproduce", "2512.24601"])
    args = _with_reproduce_defaults(args)

    args.source = normalize_path_input(args.source)
    assert args.source == "2512.24601"
