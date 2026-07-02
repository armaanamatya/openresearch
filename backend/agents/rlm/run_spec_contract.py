"""run_spec_contract.py — canonical run-spec -> env-sink application (Codex F15).

Extracted from ``cli.py::_load_run_spec`` so the campaign's INIT stage can
validate, BEFORE any money moves, that every key its canonical run-spec
profile (``configs/campaign_run_spec.json``) intends to apply will actually
land in the env sink instead of silently no-op'ing on a renamed/typo'd knob.
``apply_run_spec`` mirrors ``_load_run_spec``'s key-application semantics
exactly (same three branches, same ``str()`` coercion, same silent-skip of
unknown keys); ``cli.py`` delegates its per-key loop to it so there is exactly
one place this logic lives.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

# Keys that map to a DIFFERENT env var name than the spec key itself.
_SPECIAL_KEYS: dict[str, str] = {
    "models": "OPENRESEARCH_ROLE_MODELS",
    "baseline_extra_guidance": "OPENRESEARCH_BASELINE_EXTRA_GUIDANCE",
}

# Any key with one of these prefixes is applied verbatim.
_ENV_PREFIXES: tuple[str, ...] = ("OPENRESEARCH_", "REPROLAB_")


class RunSpecValidationError(Exception):
    """A run-spec key would be silently ignored by the env-sink applier."""

    def __init__(self, rejected_keys: tuple[str, ...]) -> None:
        self.rejected_keys = tuple(rejected_keys)
        keys_repr = ", ".join(repr(k) for k in self.rejected_keys)
        super().__init__(
            "run-spec key(s) rejected by the env-sink applier (would be "
            f"silently ignored): {keys_repr}"
        )


def run_spec_key_applies(key: str) -> bool:
    """True iff *key* is one ``apply_run_spec``/``_load_run_spec`` actually apply.

    Mirrors ``cli.py::_load_run_spec``'s condition exactly: the two
    special-cased keys (``models``, ``baseline_extra_guidance``) or any key
    prefixed ``OPENRESEARCH_``/``REPROLAB_``. Everything else is silently
    ignored by the loader (forward-compatibility) — and therefore rejected
    here too.
    """
    return key in _SPECIAL_KEYS or key.startswith(_ENV_PREFIXES)


def apply_run_spec(spec: Mapping[str, Any], sink: MutableMapping[str, str]) -> int:
    """Apply *spec* into *sink* exactly as ``cli.py::_load_run_spec`` does.

    Returns the number of keys applied. Unknown keys are silently skipped
    (excluded from the count) — the same forward-compatible semantics as the
    original inline loop.
    """
    applied = 0
    for key, val in spec.items():
        if key in _SPECIAL_KEYS:
            sink[_SPECIAL_KEYS[key]] = str(val)
            applied += 1
        elif key.startswith(_ENV_PREFIXES):
            sink[key] = str(val)
            applied += 1
        # Unknown keys silently ignored.
    return applied


def validate_run_spec_keys(spec_keys: Iterable[str]) -> None:
    """Raise :class:`RunSpecValidationError` listing every key that would be
    silently ignored by :func:`apply_run_spec` (the F15 round-trip check).

    A campaign's canonical run-spec profile calls this at INIT — before any
    money moves — so a renamed/typo'd ``OPENRESEARCH_*`` key (or a bare key
    outside the two special cases) fails fast at $0 instead of quietly doing
    nothing for the rest of the campaign.
    """
    rejected = tuple(key for key in spec_keys if not run_spec_key_applies(key))
    if rejected:
        raise RunSpecValidationError(rejected)


__all__ = [
    "RunSpecValidationError",
    "apply_run_spec",
    "run_spec_key_applies",
    "validate_run_spec_keys",
]
