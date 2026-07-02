"""Canonical secret resolver + redaction for the reproduction pipeline.

WHY this module exists
-----------------------
Phase 1d's resolve/triage/stage/provision seams (CPU-tier asset resolution,
two-VM ``cpu_warm_disk_then_gpu_attach`` staging, generic ``AssetResolver``)
all need the same thing a handful of existing call sites already do ad hoc:
look up a configured secret (HF token, cloud API key, ...) without ever
putting a raw ``.env`` on the wire or leaking a value into a log line, a
staged bundle, or a redacted-command string.

This module is that one canonical home. It mirrors two already-proven
patterns instead of inventing new ones:

* the **resolution order** — ``os.environ`` candidates first, then a
  ``Settings``-backed ``.env`` attribute, fail-soft on any Settings import
  error — is ``backend.agents.runtime.foundry_endpoint._env_or_settings``,
  generalised from one secret to a registry of named secrets;
* the **redaction semantics** — drop a whole env var when its KEY looks
  secret-shaped, drop a whole TEXT LINE when it mentions a secret substring —
  are ``backend.agents.rlm.failure_capsule._redact_env`` /
  ``_redact_text``, lifted here to be the shared implementation (that module
  keeps its own copies for now; a future consolidation can delegate to this
  one without behaviour change since the semantics are identical).

The red line (enforced by the test suite, not just documented): ``get`` never
logs or returns a secret into anything other than its direct return value;
``redact_env``/``redact_text`` never let a secret VALUE survive into their
output, even when the input key/line also carries other data; and
``gated_exclusion`` reports only presence/absence of a secret, never the
secret itself.

Stdlib + ``backend.config`` only (lazy-imported, see ``_env_or_settings``)
and ``backend.agents.rlm.exclusion`` (pure, stdlib-only) — no network, no
subprocess, no other backend layer. Hermetic by construction: ``env`` and
``settings_getter`` are injectable so tests never touch real process
environment or real Settings.
"""

from __future__ import annotations

import os
import re
from typing import Callable

from backend.agents.rlm.exclusion import AXIS_DATASET, KIND_ENV_SETUP_FAILED, Exclusion

__all__ = [
    "CredentialBroker",
]

# Canonical registry: logical secret name -> (env-var candidates, Settings attr).
# ``env_candidates`` are tried in order, first non-empty wins; ``settings_attr``
# (may be ``None``) is the ``Settings`` fallback attribute name. Extend this
# table, never hardcode a secret lookup at a call site.
_SECRET_REGISTRY: dict[str, tuple[tuple[str, ...], str | None]] = {
    "hf_token": (("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"), None),
    "anthropic_api_key": (("ANTHROPIC_API_KEY",), "anthropic_api_key"),
    "openai_api_key": (("OPENAI_API_KEY",), "openai_api_key"),
    "runpod_api_key": (("RUNPOD_API_KEY",), "runpod_api_key"),
    "azure_foundry_api_key": (("AZURE_FOUNDRY_API_KEY",), "azure_foundry_api_key"),
    "aws_access_key_id": (("AWS_ACCESS_KEY_ID",), None),
    "aws_secret_access_key": (("AWS_SECRET_ACCESS_KEY",), None),
    "kaggle_key": (("KAGGLE_KEY",), None),
}

# A key/line "looks secret-shaped" iff it mentions one of these substrings,
# case-insensitively. Matches the ``failure_capsule`` denylist semantics.
_SECRET_KEY_RE = re.compile(r"key|token|secret|password", re.IGNORECASE)


def _default_settings_getter() -> object:
    """Lazy import so this module never drags in ``backend.config`` at import time."""
    from backend.config import get_settings

    return get_settings()


class CredentialBroker:
    """Resolves configured secrets (os.environ -> Settings), fail-soft, never logged.

    ``env`` and ``settings_getter`` are injectable for hermetic testing;
    defaults are ``os.environ`` and a lazy ``backend.config.get_settings``.
    """

    def __init__(
        self,
        *,
        env: dict | None = None,
        settings_getter: Callable[[], object] | None = None,
    ) -> None:
        self._env: dict = os.environ if env is None else env
        self._settings_getter: Callable[[], object] = (
            _default_settings_getter if settings_getter is None else settings_getter
        )

    def get(self, name: str) -> str | None:
        """Resolve ``name`` (a ``_SECRET_REGISTRY`` key) to its value, or ``None``.

        Tries each env-var candidate against ``self._env`` first (first
        non-empty wins), then the Settings attribute if one is registered.
        Fail-soft: an unknown ``name``, a Settings error, or an all-empty
        result all resolve to ``None`` rather than raising. Never logs the
        resolved value.
        """
        entry = _SECRET_REGISTRY.get(name)
        if entry is None:
            return None
        env_candidates, settings_attr = entry

        for var in env_candidates:
            try:
                value = str(self._env.get(var, "") or "").strip()
            except Exception:  # noqa: BLE001 — a hostile fake env must not raise
                value = ""
            if value:
                return value

        if settings_attr:
            try:
                settings = self._settings_getter()
                value = str(getattr(settings, settings_attr, "") or "").strip()
            except Exception:  # noqa: BLE001 — Settings import/attr error => absent
                value = ""
            if value:
                return value

        return None

    def available(self, name: str) -> bool:
        """``True`` iff :meth:`get` resolves ``name`` to a non-empty string."""
        return bool(self.get(name))

    def require(self, name: str) -> str:
        """Return the secret or raise ``KeyError(name)``.

        Callers on a fail-soft path should prefer ``available()`` +
        :meth:`gated_exclusion` over catching this.
        """
        value = self.get(name)
        if value is None:
            raise KeyError(name)
        return value

    @staticmethod
    def redact_env(env: dict | None) -> dict:
        """Return a copy of ``env`` with every secret-shaped KEY dropped wholesale.

        A key "looks secret-shaped" iff it matches ``_SECRET_KEY_RE``
        (case-insensitive substring: key/token/secret/password) — the value is
        never inspected or echoed, only the key name decides. Fail-soft: any
        error yields ``{}`` rather than raising or leaking partial data.
        """
        if not env:
            return {}
        out: dict = {}
        try:
            for k, v in env.items():
                if _SECRET_KEY_RE.search(str(k)):
                    continue  # drop the whole var — never persist or return the value
                out[str(k)] = str(v)
        except Exception:  # noqa: BLE001 — redaction must never raise
            return {}
        return out

    @staticmethod
    def redact_text(text: str) -> str:
        """Drop whole LINES of ``text`` that mention a secret-shaped substring.

        Line-granular (not token-granular) because a traceback/log line can
        embed a credential in many shapes (``KEY=value``, ``Bearer ...``, a
        URL query param); we cannot reliably parse every shape, so dropping
        the entire line is the conservative, safe default. Fail-soft: any
        error yields ``""``.
        """
        if not text:
            return ""
        try:
            kept = [
                line for line in text.splitlines() if not _SECRET_KEY_RE.search(line)
            ]
            return "\n".join(kept)
        except Exception:  # noqa: BLE001 — redaction must never raise
            return ""

    def gated_exclusion(
        self, *, item: str, secret_name: str, axis: str = AXIS_DATASET
    ) -> Exclusion | None:
        """A verified, harness-produced ``Exclusion`` iff ``secret_name`` is absent.

        Returns ``None`` when the secret is available (nothing to gate).
        Never includes the secret value itself — only its logical NAME
        appears in the reason string.
        """
        if self.available(secret_name):
            return None
        return Exclusion(
            item=item,
            axis=axis,
            kind=KIND_ENV_SETUP_FAILED,
            reason=f"gated: needs {secret_name}",
            verified=True,
            evidence=f"credential '{secret_name}' not configured",
        )
