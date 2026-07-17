"""verdict_ceiling.py — a tier-aware ceiling on the terminal verdict.

WHY THIS EXISTS (2026-07-13, deepinvent two-tier triage funnel)
--------------------------------------------------------------
The product runs papers through two tiers:

* **Tier 1 "screen"** — cheap, broad, recall-oriented. Reduced scope, one seed.
  Its job is to RANK papers for a shortlist, not to certify them.
* **Tier 2 "verify"** — precision-first, full scope, multi-seed, external
  validator. The ONLY tier allowed to certify a paper as ``reproduced``.

The invariant that must hold: **a screening-tier run can never mint a
``reproduced`` verdict** — because a false ``reproduced`` is the catastrophic
error for patent-viability triage (you build IP strategy on a result that does
not hold), while a false ``partial``/``inconclusive`` is a recoverable
opportunity cost.

The obvious-looking mechanism does NOT deliver this. ``campaign_policy`` gates
``REPRODUCED`` on reaching the FULL scope rung — but that rule is unreachable
in three separate ways:

1. ``campaign_policy`` is imported only by the campaign-family modules. A plain
   ``backend.cli reproduce`` run — which is exactly what a cheap Tier-1 screen
   would be — never enters that code path at all, so the rung rule simply does
   not apply.
2. Under ``campaign``, the ladder length comes from the CLI ``--scope-ladder``
   token count. Omit the flag and you get a single-rung ladder, where the
   reduced rung IS the full rung.
3. A multi-attempt campaign advances up the ladder on green, so it reaches the
   full rung on its own.

A ceiling that depends on remembering a CLI flag is not an invariant — it is
the "config that silently fails to protect" pattern this repo has been bitten
by repeatedly (``learn.md`` 2026-07-07: "Reliability fixes ship default-OFF and
protect NOTHING until enabled in the run-spec").

So the ceiling is enforced HERE, at the single report write-chokepoint
(``report.write_final_report_rlm``) that every finalize path routes through —
the same seam that already hosts the evidence gate and the two-axis upgrade
clamp. It is run-spec-settable, so it travels with the tier profile rather than
with an operator's shell history.

SEMANTICS
---------
* ``OPENRESEARCH_MAX_TERMINAL_VERDICT`` unset/empty ⇒ **no cap**, byte-identical
  to prior behaviour (the repo's default-OFF invariant).
* A cap can only ever LOWER a verdict, never raise one. It composes safely with
  every other gate: they all downgrade, and downgrades commute.
* The measured score is left **untouched**. That is deliberate — the score is
  the Tier-1 ranking signal. A screened paper that scored 0.78 but is capped to
  ``partial`` is a strong shortlist candidate, and the report says so honestly
  via the ``verdict_ceiling`` block (which records what the verdict WOULD have
  been).

FAIL-CLOSED
-----------
Its neighbours in the write chokepoint are additive stamps and are fail-soft
("never blocks the write"). This is a TRUST gate, so it fails CLOSED instead: a
gate that fails open is how a screening run would ship a ``reproduced`` verdict,
which is the exact failure class of the fail-open ``FINAL_VAR`` guard. If a
ceiling is configured and cannot be applied cleanly, the verdict is forced DOWN
to the ceiling anyway and the reason is stamped.

An unrecognized non-empty value (a typo'd ceiling) is likewise treated
conservatively — capped to ``partial`` with a loud warning — rather than
silently ignored. The operator plainly intended to cap; honouring that intent
too strictly is a recoverable false-negative, whereas ignoring it is the
catastrophic false-positive.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

#: Verdict ordering. Higher = stronger claim about the paper.
_RANK: dict[str, int] = {"failed": 0, "partial": 1, "reproduced": 2}

_FLAG = "OPENRESEARCH_MAX_TERMINAL_VERDICT"

#: What an unrecognized (typo'd) ceiling value collapses to. See FAIL-CLOSED.
_CONSERVATIVE_FALLBACK = "partial"


def configured_ceiling() -> str | None:
    """The configured terminal-verdict ceiling, or ``None`` when uncapped.

    Unset/empty ⇒ ``None`` (no cap — byte-identical to prior behaviour).
    A recognized value ⇒ that value.
    An unrecognized NON-EMPTY value ⇒ ``_CONSERVATIVE_FALLBACK`` + a loud log:
    a typo must not silently disable a trust gate.
    """
    raw = os.environ.get(_FLAG, "").strip().lower()
    if not raw:
        return None
    if raw in _RANK:
        return raw
    logger.error(
        "verdict_ceiling: %s=%r is not one of %s — treating as %r (fail-closed). "
        "A typo must never silently disable the ceiling.",
        _FLAG, raw, sorted(_RANK), _CONSERVATIVE_FALLBACK,
    )
    return _CONSERVATIVE_FALLBACK


def apply_verdict_ceiling(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Clamp ``payload["verdict"]`` down to the configured ceiling.

    Returns ``(payload, warning_message_or_None)``. The payload is mutated in
    place and also returned, matching the serialized-dict convention the other
    write-chokepoint blocks use.

    Never raises a verdict. Never touches the score. No-ops entirely when the
    flag is unset.
    """
    ceiling = configured_ceiling()
    if ceiling is None:
        return payload, None

    ceiling_rank = _RANK[ceiling]

    try:
        current = str(payload.get("verdict") or "").strip().lower()
        current_rank = _RANK.get(current)

        if current_rank is None:
            # An unknown verdict string cannot be proven to sit at or below the
            # ceiling, so it does not get the benefit of the doubt.
            logger.error(
                "verdict_ceiling: unrecognized verdict %r — forcing to ceiling %r "
                "(fail-closed).", current, ceiling,
            )
            payload["verdict"] = ceiling
            payload["verdict_ceiling"] = {
                "applied": True,
                "ceiling": ceiling,
                "uncapped_verdict": current or None,
                "reason": (
                    f"verdict {current!r} is not a recognized terminal verdict; "
                    f"forced to the configured ceiling {ceiling!r}"
                ),
            }
            return payload, (
                f"terminal verdict {current!r} unrecognized — forced to ceiling {ceiling!r}"
            )

        if current_rank <= ceiling_rank:
            # Already at or below the ceiling — record that a ceiling was in
            # force (so a Tier-1 'partial' is distinguishable from a Tier-2
            # 'partial'), but change nothing.
            payload["verdict_ceiling"] = {
                "applied": False,
                "ceiling": ceiling,
                "uncapped_verdict": current,
                "reason": f"verdict {current!r} already at or below ceiling {ceiling!r}",
            }
            return payload, None

        # The real clamp.
        payload["verdict"] = ceiling
        payload["verdict_ceiling"] = {
            "applied": True,
            "ceiling": ceiling,
            "uncapped_verdict": current,
            "reason": (
                f"this run's tier may not certify above {ceiling!r}; the measured "
                f"evidence would otherwise have supported {current!r}. The score is "
                "preserved unchanged as the ranking signal."
            ),
        }
        logger.warning(
            "verdict_ceiling: capped verdict %r -> %r (%s=%s). Score preserved.",
            current, ceiling, _FLAG, ceiling,
        )
        return payload, (
            f"terminal verdict capped {current!r} -> {ceiling!r} by the run's tier "
            f"ceiling ({_FLAG}={ceiling}); measured score preserved for ranking"
        )

    except Exception as exc:  # noqa: BLE001 — trust gate: fail CLOSED, never open.
        logger.error(
            "verdict_ceiling: failed to apply ceiling %r (%s) — forcing verdict to "
            "the ceiling rather than letting an uncapped verdict ship.",
            ceiling, exc, exc_info=True,
        )
        try:
            payload["verdict"] = ceiling
            payload["verdict_ceiling"] = {
                "applied": True,
                "ceiling": ceiling,
                "uncapped_verdict": None,
                "reason": f"ceiling application errored ({exc!r}); forced to ceiling",
            }
        except Exception:  # noqa: BLE001 — payload is not a usable mapping
            pass
        return payload, f"terminal verdict forced to ceiling {ceiling!r} after gate error"


__all__ = ["apply_verdict_ceiling", "configured_ceiling"]
