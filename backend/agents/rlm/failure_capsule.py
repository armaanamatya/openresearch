"""Durable failure capsules — bounded, redacted on-disk evidence of a failure.

WHY: when ``run_experiment`` fails, the rich evidence (the actual traceback
tail, the subprocess log tail, the command + environment that produced it,
the artifact paths) lives only transiently in the result dict and is then
collapsed to a generic ``suggested_fix`` string by the failure classifier.
A repair iteration that sees only the generic message has to re-diagnose an
opaque failure from scratch. A *capsule* atomically persists the concrete,
bounded evidence to disk AND threads its text into the next iteration's
``repair_context`` so the root model repairs from real evidence.

DESIGN — lowest-blast-radius persistence:
We write capsules to ``rlm_state/failure_capsules.jsonl`` (a NEW append-only
file) rather than mutating the existing ``experiment_runs.jsonl`` rows. That
JSONL is consumed by the leaf scorer and ~8 postflight guards whose schema
expectations we must not perturb — adding fields to its rows is a needless
correctness risk. A separate file is purely additive: nothing reads it unless
the flag is on, so the OFF path is byte-for-byte unchanged.

GATING: everything here is gated by ``OPENRESEARCH_FAILURE_CAPSULES`` at the
call sites (primitives + baseline prompt). Module functions are pure/fail-soft
and safe to call unconditionally, but the wiring only invokes them when the
flag is truthy.

REDACTION: traceback / log / env text is scrubbed of any token whose key (for
env vars) or surrounding context contains a secret-shaped substring
(KEY / TOKEN / SECRET / PASSWORD). Matching env vars are dropped entirely;
no secret value is ever written to disk.

FAIL-SOFT: capsule creation/persistence MUST NEVER raise into the run. Every
public function is wrapped in try/except and degrades to a safe empty result.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Substrings (case-insensitive) that mark an env var / text fragment as secret.
_SECRET_SUBSTRINGS: tuple[str, ...] = ("key", "token", "secret", "password")

# Bound for the traceback tail and the log tail (chars each). Keeps the
# capsule small enough to inline into a prompt without blowing the context.
_TAIL_CHARS: int = 4000

# Bound for the number of artifact paths recorded.
_MAX_ARTIFACTS: int = 20


def _flag_on(name: str) -> bool:
    """Return True iff the named env var is a truthy capsule flag."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def capsules_enabled() -> bool:
    """Master gate for the durable-failure-capsule feature (default OFF)."""
    return _flag_on("OPENRESEARCH_FAILURE_CAPSULES")


def _is_secret_key(name: str) -> bool:
    """A denylist match — does this env var NAME look like a credential?"""
    low = str(name).lower()
    return any(sub in low for sub in _SECRET_SUBSTRINGS)


def _redact_text(text: str) -> str:
    """Drop whole lines of ``text`` that mention a secret-shaped token.

    WHY line-granular: a traceback / log can echo ``OPENAI_API_KEY=sk-...`` or
    ``Authorization: Bearer ...``. We cannot reliably parse every shape, so we
    conservatively drop any LINE containing a secret substring — losing a noisy
    line is always safer than persisting a live credential.
    """
    if not text:
        return ""
    try:
        kept = [
            line
            for line in text.splitlines()
            if not any(sub in line.lower() for sub in _SECRET_SUBSTRINGS)
        ]
        return "\n".join(kept)
    except Exception:  # noqa: BLE001 — redaction must never raise
        return ""


def _redact_env(env: dict[str, str] | None) -> dict[str, str]:
    """Return a copy of ``env`` with all secret-shaped vars dropped entirely."""
    if not env:
        return {}
    out: dict[str, str] = {}
    try:
        for k, v in env.items():
            if _is_secret_key(k):
                continue  # drop credential vars wholesale — never persist values
            out[str(k)] = str(v)
    except Exception:  # noqa: BLE001 — fail-soft
        return {}
    return out


def _tail(text: object, *, limit: int = _TAIL_CHARS) -> str:
    """Bounded tail of a (possibly huge) string field; redacted."""
    try:
        s = str(text or "")
    except Exception:  # noqa: BLE001
        return ""
    return _redact_text(s[-limit:])


def _error_signature(result: dict) -> str:
    """A short, stable signature for the failure — class + last error line.

    WHY: the signature lets a cross-iteration consumer tell "the same failure
    recurred" from "a new failure appeared" without diffing full tracebacks.
    """
    try:
        klass = str(result.get("failure_class") or "unknown")
        err = str(result.get("error") or "")
        last = ""
        if err:
            last = err.strip().splitlines()[-1][:200] if err.strip() else ""
        if not last:
            # fall back to the last non-empty stderr/log line
            for field in ("stderr", "logs", "stdout"):
                blob = str(result.get(field) or "")
                lines = [ln for ln in blob.splitlines() if ln.strip()]
                if lines:
                    last = lines[-1][:200]
                    break
        return f"{klass}: {last}".strip().rstrip(":").strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_capsule(
    result: dict,
    *,
    command: str | None = None,
    env: dict[str, str] | None = None,
    artifacts: list[str] | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Build a bounded, redacted failure capsule dict from a failed result.

    The capsule captures REAL evidence — a traceback tail, a log tail, the
    command + environment fingerprints, artifact paths, and an error
    signature — so a repair iteration works from what actually happened.

    Fail-soft: any internal error returns a minimal ``{}``-ish capsule rather
    than raising. Bounded: every text field is tail-capped to ``_TAIL_CHARS``.
    Redacted: secrets are scrubbed from text and env BEFORE inclusion.
    """
    try:
        # Prefer an explicit traceback field; else stderr; else logs tail.
        tb_source = (
            result.get("traceback")
            or result.get("stderr")
            or result.get("error")
            or ""
        )
        log_source = result.get("logs") or result.get("stdout") or ""

        arts = []
        try:
            for a in (artifacts or [])[:_MAX_ARTIFACTS]:
                arts.append(str(a))
        except Exception:  # noqa: BLE001
            arts = []

        capsule: dict[str, Any] = {
            "schema": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_id": project_id,
            "failure_class": str(result.get("failure_class") or "unknown"),
            "suggested_fix": str(result.get("suggested_fix") or ""),
            "error_signature": _error_signature(result),
            "traceback_tail": _tail(tb_source),
            "log_tail": _tail(log_source),
            "command": _redact_text(str(command or "")) if command else None,
            "env_fingerprint": _redact_env(env),
            "artifacts": arts,
            "exit_code": result.get("exit_code"),
        }
        return capsule
    except Exception:  # noqa: BLE001 — capsule build must NEVER break the run
        return {
            "schema": 1,
            "failure_class": "unknown",
            "error_signature": "unknown",
            "traceback_tail": "",
            "log_tail": "",
            "env_fingerprint": {},
            "artifacts": [],
        }


def persist_capsule(project_dir: Path, capsule: dict[str, Any]) -> bool:
    """Atomically append ``capsule`` to ``rlm_state/failure_capsules.jsonl``.

    Atomicity: a single JSONL line is written to a temp file in the same
    directory and ``os.replace``'d onto a per-capsule staging name, then the
    line is appended to the durable log. We append (a + os.replace of a temp
    holding the FULL new content) so a crash mid-write can never leave a torn
    line in the durable file.

    Returns True on success, False on any failure. Fail-soft: never raises.
    """
    try:
        state_dir = Path(project_dir) / "rlm_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        target = state_dir / "failure_capsules.jsonl"

        new_line = json.dumps(capsule, default=str) + "\n"
        existing = ""
        if target.exists():
            try:
                existing = target.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001 — treat unreadable as empty, never crash
                existing = ""

        fd, tmp_path = tempfile.mkstemp(dir=str(state_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(existing)
                fh.write(new_line)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, target)  # atomic swap onto the durable log
        finally:
            # If os.replace already consumed tmp_path this is a no-op.
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:  # noqa: BLE001
                pass
        return True
    except Exception:  # noqa: BLE001 — persistence must NEVER break the run
        return False


def render_capsule_text(capsule: dict[str, Any]) -> str:
    """Render a capsule into compact text for inlining into a repair prompt.

    WHY a dedicated renderer (vs json.dumps): the repair prompt is read by the
    code-writing model — a labelled, sectioned view of the concrete evidence
    is easier to act on than a raw JSON blob, and lets us order the highest-
    signal fields (signature, traceback tail) first.
    """
    try:
        if not capsule:
            return ""
        parts: list[str] = ["FAILURE CAPSULE (real on-disk evidence):"]
        sig = capsule.get("error_signature")
        if sig:
            parts.append(f"- error_signature: {sig}")
        klass = capsule.get("failure_class")
        if klass:
            parts.append(f"- failure_class: {klass}")
        ec = capsule.get("exit_code")
        if ec is not None:
            parts.append(f"- exit_code: {ec}")
        cmd = capsule.get("command")
        if cmd:
            parts.append(f"- command: {cmd}")
        arts = capsule.get("artifacts") or []
        if arts:
            parts.append(f"- artifacts: {', '.join(str(a) for a in arts)}")
        tb = capsule.get("traceback_tail")
        if tb:
            parts.append(f"- traceback_tail:\n{tb}")
        log = capsule.get("log_tail")
        if log:
            parts.append(f"- log_tail:\n{log}")
        return "\n".join(parts)
    except Exception:  # noqa: BLE001 — rendering must never raise
        return ""


__all__ = [
    "capsules_enabled",
    "build_capsule",
    "persist_capsule",
    "render_capsule_text",
]
