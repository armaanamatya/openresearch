"""P4 (2026-05-29): root-cause signature for the same-failure-twice rule.

The SDAR 7-attempt retry sprint surfaced a gap in the original
"same-failure-twice halts the sweep" rule: it keyed on the
``failure_class`` field of a ``run_experiment`` result, but SDAR died
from things WITHOUT a ``failure_class``:

  * REPL safe-builtins errors (BUG-LR-011/012)
  * SDK pre-emit stall in ``implement_baseline`` (BUG-NEW-012)
  * OAuth refusal loop (BUG-NEW-038)
  * Ingest failure
  * Backend wedge

A single helper that normalizes any attempt's failure into a comparable
``signature`` string makes the rule actually fire across the full failure
surface, not just `run_experiment`'s slice.

The helper is intentionally simple — it picks the most specific field
available and returns a stable string. Two attempts whose signatures
match are "the same failure" for the purposes of the retry rule.
"""

from __future__ import annotations

from typing import Any

ROOT_CAUSE_UNKNOWN = "unknown:unclassified"


def root_cause_signature(attempt_info: dict[str, Any]) -> str:
    """Return a stable signature for an attempt's root cause.

    The signature is chosen in priority order:

    1. ``run_experiment`` ``failure_class`` (e.g. ``cuda_oom``,
       ``dockerfile_invalid``, ``commands_missing_file``) — most specific
       for in-sandbox failures.
    2. ``<primitive>:<error_code>`` when a primitive (other than
       ``run_experiment``) failed with a typed error code — e.g.
       ``implement_baseline:sdk_pre_emit_stall``,
       ``build_environment:dockerfile_parse_error``.
    3. ``<exception_class>:<first_line_of_message>`` when the failure
       was a top-level exception with no typed error code — e.g.
       ``RuntimeError:credit balance too low``,
       ``KeyboardInterrupt:received signal 15``. The first line is
       truncated to 120 chars to keep signatures comparable across
       attempts whose error messages share a class but differ in trailing
       trivia (timestamps, ids, etc.).
    4. ``ROOT_CAUSE_UNKNOWN`` as a stable fallback when nothing
       classifiable is present.

    Args:
        attempt_info: a dict describing the failed attempt. Recognized
            keys (all optional):

            * ``failure_class`` — `run_experiment` result field
            * ``primitive`` + ``error_code`` — primitive failure
            * ``exception_class`` + ``error`` (or ``message``) — harness
              exception
            * ``error`` / ``message`` as a fallback when class is set
              but no message dict was nested

    Returns:
        A signature string suitable for direct comparison with another
        attempt's signature.

    Notes:
        Trailing whitespace, ANSI codes, and surrounding quotes are
        stripped from the message line so signatures don't drift on
        cosmetic differences. Case is preserved — ``Traceback`` vs
        ``traceback`` are distinct attempts.
    """
    if not isinstance(attempt_info, dict):
        return ROOT_CAUSE_UNKNOWN

    # 1. run_experiment failure_class
    failure_class = attempt_info.get("failure_class")
    if isinstance(failure_class, str) and failure_class.strip():
        return f"run_experiment:{failure_class.strip().lower()}"

    # 2. primitive failure with error_code
    primitive = attempt_info.get("primitive")
    error_code = attempt_info.get("error_code")
    if (
        isinstance(primitive, str) and primitive.strip()
        and isinstance(error_code, str) and error_code.strip()
    ):
        return f"{primitive.strip()}:{error_code.strip()}"

    # 3. exception_class + message
    exception_class = attempt_info.get("exception_class")
    message = attempt_info.get("error") or attempt_info.get("message")
    if isinstance(exception_class, str) and exception_class.strip():
        first_line = _normalize_message_line(message)
        if first_line:
            return f"{exception_class.strip()}:{first_line[:120]}"
        return f"{exception_class.strip()}:<no-message>"

    return ROOT_CAUSE_UNKNOWN


def _normalize_message_line(message: Any) -> str:
    """Return the first non-blank line of ``message`` with cosmetic trim.

    Strips surrounding quotes, ANSI escape codes, and trailing whitespace
    so two messages that differ only in those don't produce distinct
    signatures.
    """
    if message is None:
        return ""
    text = str(message)
    # ANSI escape codes — e.g. "\x1b[31mError\x1b[0m"
    import re as _re
    text = _re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    for line in text.splitlines():
        stripped = line.strip().strip('"').strip("'")
        if stripped:
            return stripped
    return ""
