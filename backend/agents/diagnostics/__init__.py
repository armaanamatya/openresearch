"""Diagnostic helpers for run-failure triage.

This module hosts small, pure-function helpers used by the runbook author
(and any future automated triage) to classify failure modes across attempts.
Nothing here writes state, calls LLMs, or contacts external services.
"""

from backend.agents.diagnostics.root_cause import (
    ROOT_CAUSE_UNKNOWN,
    root_cause_signature,
)

__all__ = ["root_cause_signature", "ROOT_CAUSE_UNKNOWN"]
