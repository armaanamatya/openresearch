"""Verify that the CLI --mode default is 'rlm'."""

from __future__ import annotations

from argparse import Namespace


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
