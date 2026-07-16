"""W1-M1 grading-input integrity — rubric pinning (grader-integrity workstream).

The RLM root writes Python in a **sandboxed** REPL: it cannot mutate the host's
``leaf_scorer.py``, but it *can* rewrite the grading-input artifacts it
legitimately produces into the run dir — ``rubric_tree.json`` (the scoring
criteria), ``provenance.json`` and ``metrics.json``. A weakened rubric
("test_error <= 8.75" → "test_error <= 99.0") makes a bad run grade well while
every existing evidence check still passes, because those check the *result*,
not the *yardstick*.

This module pins the yardstick. At rubric-generation time the tree is
fingerprinted (:func:`rubric_fingerprint`); the pin is persisted next to the run
(:func:`write_rubric_pin`). At grade time the current tree is re-fingerprinted
and compared to the pin (:func:`verify_rubric_integrity`); a post-gen change is a
fail-closed ``evidence_tampered`` verdict.

Pure & deterministic (no clock, no network, no randomness); fail-soft on bad
input. The gate wiring reads ``OPENRESEARCH_GRADER_INTEGRITY`` (default OFF) so
that when the flag is unset this module is never consulted and behavior is
byte-identical to the prior baseline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "rubric_fingerprint",
    "write_rubric_pin",
    "verify_rubric_integrity",
    "check_grading_input_integrity",
]

_FLAG = "OPENRESEARCH_GRADER_INTEGRITY"
_PIN_SCHEMA = 1


def _flag_on() -> bool:
    """Canonical default-OFF flag idiom (matches the repo-wide convention)."""
    return os.environ.get(_FLAG, "").strip().lower() in ("1", "true", "yes")


def _rubric_path(run_dir: Path) -> Path:
    return Path(run_dir) / "rubric_tree.json"


def _pin_path(run_dir: Path) -> Path:
    return Path(run_dir) / "rlm_state" / "rubric_pin.json"


def _load_json(path: Path) -> Any | None:
    """Read + parse JSON; fail-soft to ``None`` (missing / unreadable / bad JSON)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — fail-soft: any read/parse error → None.
        return None


def rubric_fingerprint(rubric: Any) -> str:
    """Return a deterministic lowercase-hex sha256 of a rubric object.

    The object is serialized canonically (``sort_keys=True``, compact
    separators) so that a re-ordering of dict keys yields the same digest while
    any change to a value — a weakened leaf criterion, a dropped leaf, a nudged
    weight — yields a different digest. List order is significant (reordering
    siblings changes the digest); that is acceptable and conservative for a
    tamper check.
    """
    canonical = json.dumps(
        rubric,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_rubric_pin(run_dir: Path | str, rubric: Any) -> Path:
    """Persist the rubric fingerprint next to the run (call at rubric-gen time).

    Writes ``<run_dir>/rlm_state/rubric_pin.json`` = ``{"schema", "rubric_tree_sha256"}``.
    Returns the pin path. Idempotent: re-pinning an unchanged rubric is a no-op-equivalent.
    """
    run_dir = Path(run_dir)
    pin = _pin_path(run_dir)
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(
        json.dumps({"schema": _PIN_SCHEMA, "rubric_tree_sha256": rubric_fingerprint(rubric)}),
        encoding="utf-8",
    )
    return pin


def verify_rubric_integrity(run_dir: Path | str) -> dict[str, Any]:
    """Fail-closed check that the graded rubric matches the pinned fingerprint.

    Returns ``{"ok", "reason", "pinned_sha", "current_sha"}``. We veto only on
    *positive* evidence of tampering, never on a missing baseline:

    * no pin present            → ``ok=True,  reason="no_pin"``   (nothing to compare)
    * pinned, rubric present, match    → ``ok=True,  reason="match"``
    * pinned, rubric present, differ   → ``ok=False, reason="rubric_mismatch"``  (the attack)
    * pinned, rubric file gone         → ``ok=False, reason="rubric_missing"``
    """
    run_dir = Path(run_dir)
    pin_obj = _load_json(_pin_path(run_dir))
    pinned_sha = pin_obj.get("rubric_tree_sha256") if isinstance(pin_obj, dict) else None
    if not pinned_sha:
        return {"ok": True, "reason": "no_pin", "pinned_sha": None, "current_sha": None}

    rubric_obj = _load_json(_rubric_path(run_dir))
    if rubric_obj is None:
        return {"ok": False, "reason": "rubric_missing", "pinned_sha": pinned_sha, "current_sha": None}

    current_sha = rubric_fingerprint(rubric_obj)
    if current_sha == pinned_sha:
        return {"ok": True, "reason": "match", "pinned_sha": pinned_sha, "current_sha": current_sha}
    return {"ok": False, "reason": "rubric_mismatch", "pinned_sha": pinned_sha, "current_sha": current_sha}


def check_grading_input_integrity(run_dir: Path | str) -> dict[str, Any] | None:
    """Flag-gated entrypoint. Returns ``None`` when ``OPENRESEARCH_GRADER_INTEGRITY``
    is off (byte-identical no-op), else the :func:`verify_rubric_integrity` verdict.

    Never raises: an unexpected error fails soft to ``None`` so an integrity-check
    bug can never break grading of an otherwise-fine run.
    """
    if not _flag_on():
        return None
    try:
        return verify_rubric_integrity(run_dir)
    except Exception:  # noqa: BLE001 — an integrity bug must never break grading.
        logger.exception("grading_input_integrity: unexpected error on %r", run_dir)
        return None
