"""
P1 — pre-GPU code-review gate (§4.1 of 2026-06-20-pre-gpu-code-review-and-report-validation-design.md).

Sends the executor's training code through a separate-model reviewer (grok,
cross-family from the Sonnet executor) BEFORE any GPU grid is dispatched.
A finding is blocking only when severity ∈ {will_produce_fake_metrics,
will_produce_wrong_metrics} AND the cited file:line exists on disk — so a
transient LLM hallucination cannot false-block a run.

Default-OFF: OPENRESEARCH_CODE_REVIEW_GATE (also requires EXTERNAL_VALIDATOR=1).
Fail-OPEN at runtime: client None / call error / parse error → non-blocking
(proceed; P2 + post-run guards remain the backstop).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


def code_review_gate_enabled() -> bool:
    """True iff both CODE_REVIEW_GATE and EXTERNAL_VALIDATOR flags are on."""
    from backend.agents.rlm.external_validator import external_validator_enabled  # noqa: PLC0415
    return (
        os.environ.get("OPENRESEARCH_CODE_REVIEW_GATE", "").strip().lower()
        in _ENABLED_VALUES
        and external_validator_enabled()
    )


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class CodeReviewVerdict:
    """Result of the code review panel call."""
    blocking: bool
    findings: list[dict]   # each: {file, line, severity, anti_pattern, detail}
    raw: str               # raw LLM response text


# ---------------------------------------------------------------------------
# Training-file discovery
# ---------------------------------------------------------------------------

# File names that are considered training/env code worth reviewing.
_TRAINING_FILENAMES = frozenset({
    "train.py",
    "train_cell.py",
    "trainer.py",
    "run.py",
    "main.py",
    "model.py",
    "env.py",
    "environment.py",
    "reward.py",
    "policy.py",
})

# Max chars we read per file to stay within context limits.
_MAX_FILE_CHARS = 12_000
# Max number of training files to include.
_MAX_FILES = 6


def _read_training_files(code_dir: Path) -> dict[str, str]:
    """Return {relative_path: content_truncated} for key training files."""
    result: dict[str, str] = {}
    try:
        candidates: list[Path] = []
        # Priority: exact filename matches first, then any .py in top-level.
        for name in _TRAINING_FILENAMES:
            p = code_dir / name
            if p.is_file():
                candidates.append(p)
        # Also sweep top-level *.py that weren't already included.
        for p in sorted(code_dir.glob("*.py")):
            if p not in candidates and p.name not in _TRAINING_FILENAMES:
                candidates.append(p)
        for p in candidates[:_MAX_FILES]:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                result[p.name] = text[:_MAX_FILE_CHARS]
            except OSError:
                continue
    except Exception:  # noqa: BLE001
        pass
    return result


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an adversarial code reviewer specialising in ML reproducibility.
Your job is to find code bugs that would cause training to produce FAKE or WRONG metrics
— not bugs that merely reduce model quality. Be conservative: only flag things that
produce zeros, constants, or systematically-wrong numbers even when training appears
to run successfully.

Respond with a JSON array (nothing else) where each element has:
{
  "file": "<filename>",
  "line": <integer or null>,
  "severity": "<will_produce_fake_metrics | will_produce_wrong_metrics | style | uncertain>",
  "anti_pattern": "<short label>",
  "detail": "<one sentence>"
}

Severity meanings:
  will_produce_fake_metrics  — the code will always produce zeros, constants, or
                               placeholder values (e.g. metrics never updated,
                               hardcoded return values, loss not connected to model).
  will_produce_wrong_metrics — the code will produce non-zero but systematically
                               biased values (e.g. teacher and student logprobs swapped,
                               wrong aggregation, train/eval split leak).
  style                      — not a correctness issue; do not flag training quality.
  uncertain                  — might be a bug but you cannot be sure from reading alone.

Focus ONLY on:
1. Is the loss computed from a REAL forward pass over sampled rollouts?
2. Are rewards from REAL environment outcomes (not constants / torch.randn / hardcoded)?
3. Is the teacher model loaded and the KL/gap computed from real logprobs?
4. Does eval score against gold labels (not against itself or a constant)?
5. Are ANY metrics hardcoded (assigned a literal float instead of computed)?
6. Is the model a real `from_pretrained` load, not a stub/placeholder?

If everything looks fine, return an empty array: []
"""

_USER_TEMPLATE = """\
## Method context
{method_context}

## Training code files
{file_blocks}

Review the code for anti-fabrication issues only.
Return a JSON array of findings (empty [] if none).
"""


def _build_user_prompt(files: dict[str, str], method_context: str) -> str:
    file_blocks = "\n\n".join(
        f"### {name}\n```python\n{content}\n```"
        for name, content in files.items()
    )
    return _USER_TEMPLATE.format(
        method_context=method_context[:2000],
        file_blocks=file_blocks or "(no training files found)",
    )


# ---------------------------------------------------------------------------
# Finding file:line existence check
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(r"^\d+$")


def _finding_is_grounded(finding: dict, code_dir: Path) -> bool:
    """Return True iff the cited file:line exists on disk."""
    try:
        fname = finding.get("file") or ""
        line = finding.get("line")
        if not fname:
            return False
        fpath = code_dir / fname
        if not fpath.is_file():
            return False
        if line is None:
            # File exists but no line cited — count it as grounded at file level.
            return True
        line_int = int(line)
        # Check line count without reading the whole file twice.
        with fpath.open(encoding="utf-8", errors="replace") as fh:
            for i, _ in enumerate(fh, 1):
                if i >= line_int:
                    return True
        return False
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# JSON-findings parser
# ---------------------------------------------------------------------------

def _parse_findings(raw: str) -> list[dict]:
    """Extract a list of finding dicts from the LLM response.

    Tries JSON parse of the whole response; falls back to finding the first
    '[' … ']' block.  Fail-soft → [].
    """
    try:
        raw = raw.strip()
        # Fast path: entire response is valid JSON.
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [f for f in parsed if isinstance(f, dict)]
        except json.JSONDecodeError:
            pass
        # Fallback: find first JSON array in the text.
        start = raw.find("[")
        if start == -1:
            return []
        # Find the matching ']' (accounting for nesting).
        depth = 0
        for i, ch in enumerate(raw[start:], start=start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(raw[start : i + 1])
                        if isinstance(parsed, list):
                            return [f for f in parsed if isinstance(f, dict)]
                    except json.JSONDecodeError:
                        pass
                    break
    except Exception:  # noqa: BLE001
        pass
    return []


# ---------------------------------------------------------------------------
# High-severity set
# ---------------------------------------------------------------------------

_BLOCKING_SEVERITIES = frozenset({"will_produce_fake_metrics", "will_produce_wrong_metrics"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def review_executor_code(
    *,
    validator_client: Any,
    code_dir: Path,
    method_context: str,
) -> CodeReviewVerdict:
    """Send training code to the validator for anti-fabrication review.

    Returns a CodeReviewVerdict.  `blocking=True` iff ≥1 high-severity finding
    cites an existing file:line.

    Fail-OPEN: any error (client None, call raises, parse fails) → non-blocking
    verdict with empty findings.
    """
    if validator_client is None:
        logger.debug("code_review_gate: validator_client is None — fail-open")
        return CodeReviewVerdict(blocking=False, findings=[], raw="")

    files = _read_training_files(code_dir)
    if not files:
        logger.debug("code_review_gate: no training files found in %s — skip", code_dir)
        return CodeReviewVerdict(blocking=False, findings=[], raw="")

    user_prompt = _build_user_prompt(files, method_context)

    # Call the validator (single completion — this is a structured review, not a panel).
    try:
        from backend.agents.rlm.grader_transport import sample_completions  # noqa: PLC0415
        completions = sample_completions(
            validator_client,
            system=_SYSTEM,
            user=user_prompt,
            n=1,
        )
        raw = completions[0] if completions else ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("code_review_gate: validator call failed (%s) — fail-open", exc)
        return CodeReviewVerdict(blocking=False, findings=[], raw="")

    findings = _parse_findings(raw)

    # Determine blocking: high-severity AND cited file:line exists.
    blocking = False
    for f in findings:
        if f.get("severity") in _BLOCKING_SEVERITIES and _finding_is_grounded(f, code_dir):
            blocking = True
            break

    if blocking:
        logger.warning(
            "code_review_gate: blocking finding(s): %s",
            [f for f in findings if f.get("severity") in _BLOCKING_SEVERITIES],
        )
    else:
        logger.debug("code_review_gate: non-blocking (findings=%d)", len(findings))

    return CodeReviewVerdict(blocking=blocking, findings=findings, raw=raw)
