"""Pure classifier for the root-validation gate.

Given a resolved RLM root model, classify it as validated/risky so the CLI can
predict the degenerate-loop failure BEFORE a long run: stamp the verdict into
``demo_status.json``, emit a loud operator warning for ``claude-oauth``, and
fail-fast when ``OPENRESEARCH_REQUIRE_VALIDATED_ROOT`` is set.

The function is **pure**: it reads only ``root_model.{paper_validated,
rlm_backend, key}`` and performs no I/O and no environment reads. It is
duck-typed on those three attributes and deliberately does NOT import
``models.py`` (avoids an import cycle).

oauth-root-reliability plan, P2.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "RISK_NONE",
    "RISK_DEGENERATE_LOOP",
    "RISK_UNVALIDATED",
    "RootValidation",
    "classify_root_model",
    "is_degenerate_loop_risk",
]

# ---------------------------------------------------------------------------
# Public contract — risk values
# ---------------------------------------------------------------------------

RISK_NONE = "none"
"""Paper-validated root — no known root-reliability risk."""

RISK_DEGENERATE_LOOP = "degenerate_loop"
"""``claude-oauth``: documented FINAL_VAR refusal-loop risk (never reaches
``implement_baseline``)."""

RISK_UNVALIDATED = "unvalidated"
"""Any other non-paper-validated root (no documented degenerate-loop risk)."""


@dataclass(frozen=True)
class RootValidation:
    """Classification of a resolved root model for the validation gate."""

    validated: bool
    risk: str  # one of the RISK_* values
    model_key: str


def classify_root_model(root_model) -> RootValidation:
    """Classify a resolved ``RootModel`` for the root-validation gate.

    PURE: reads only ``root_model.{paper_validated, rlm_backend, key}``; no
    I/O, no environment reads.

    Risk precedence:

    1. ``rlm_backend == "anthropic-oauth"`` → :data:`RISK_DEGENERATE_LOOP`
       (the documented claude-oauth refusal-loop), regardless of validation.
    2. ``not paper_validated``              → :data:`RISK_UNVALIDATED`
    3. otherwise                            → :data:`RISK_NONE`
    """
    validated = bool(root_model.paper_validated)
    if root_model.rlm_backend == "anthropic-oauth":
        risk = RISK_DEGENERATE_LOOP
    elif not validated:
        risk = RISK_UNVALIDATED
    else:
        risk = RISK_NONE
    return RootValidation(validated=validated, risk=risk, model_key=root_model.key)


def is_degenerate_loop_risk(root_model) -> bool:
    """``True`` iff *root_model* carries the documented degenerate-loop risk."""
    return root_model.rlm_backend == "anthropic-oauth"
