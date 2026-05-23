"""Verify that the CLI --mode default is 'rlm'."""

from __future__ import annotations

from argparse import Namespace

import pytest


class TestDefaultModeIsRlm:
    """--mode defaults to 'rlm' when omitted."""

    def test_default_mode_is_rlm(self):
        """Verify _REPRODUCE_DEFAULTS carries mode='rlm' and _with_reproduce_defaults
        backfills it correctly for generated Namespace callers."""
        from backend.cli import _REPRODUCE_DEFAULTS, _with_reproduce_defaults

        # 1. The dict itself carries the right default.
        assert _REPRODUCE_DEFAULTS["mode"] == "rlm", (
            f"Expected _REPRODUCE_DEFAULTS['mode'] == 'rlm', got {_REPRODUCE_DEFAULTS['mode']!r}"
        )

        # 2. _with_reproduce_defaults backfills mode when not provided.
        args = _with_reproduce_defaults(Namespace(source="paper.pdf"))
        assert args.mode == "rlm", (
            f"Expected args.mode == 'rlm' after backfill, got {args.mode!r}"
        )


def test_module_main_bypasses_atexit_for_reproduce(monkeypatch):
    """`python -m backend.cli reproduce` must not hang on SDK atexit cleanup."""
    from backend import cli

    exit_codes: list[int] = []

    def _fake_exit(code: int) -> None:
        exit_codes.append(code)
        raise RuntimeError("os._exit intercepted")

    monkeypatch.setattr(cli, "main", lambda argv=None: 3)
    monkeypatch.setattr(cli.os, "_exit", _fake_exit)

    with pytest.raises(RuntimeError, match="intercepted"):
        cli._module_main(["reproduce", "ftrl"])

    assert exit_codes == [3]


def test_module_main_uses_system_exit_for_non_reproduce(monkeypatch):
    from backend import cli

    monkeypatch.setattr(cli, "main", lambda argv=None: 0)

    with pytest.raises(SystemExit) as exc:
        cli._module_main(["ingest", "paper.pdf"])

    assert exc.value.code == 0
