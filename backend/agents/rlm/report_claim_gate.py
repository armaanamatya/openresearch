"""Report-claim teeth — §4.3 of 2026-06-20-pre-gpu-code-review-and-report-validation-design.md.

Deterministic, universal backstop: extracts result claims from the assembled
report narrative (reproduction_summary + reported_metrics ONLY — NOT
paper_claims) and caps the verdict when ungrounded claims exist and measured
evidence is present.

Flag: OPENRESEARCH_REPORT_CLAIM_GATE (default OFF).
Hook: write_final_report_rlm — the FINAL verdict mutation, after the
best-of-run floor and two-axis attach, immediately before model_dump_json.
Fail-soft: any exception → skip (never break the write).
Byte-identical-off: when flag is off, no key is emitted and verdict is
untouched.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Verdict rank — used to cap without upgrading.
_VERDICT_RANK: dict[str, int] = {"failed": 0, "partial": 1, "reproduced": 2}


def report_claim_gate_enabled() -> bool:
    """True iff OPENRESEARCH_REPORT_CLAIM_GATE=1."""
    return os.environ.get("OPENRESEARCH_REPORT_CLAIM_GATE", "").strip() == "1"


def _text_from_reported_metrics(reported_metrics: Any) -> str:
    """Flatten reported_metrics (dict or str) to a searchable string."""
    if not reported_metrics:
        return ""
    if isinstance(reported_metrics, str):
        return reported_metrics
    try:
        return json.dumps(reported_metrics, default=str)
    except Exception:  # noqa: BLE001
        return str(reported_metrics)


def apply_report_claim_gate(
    report: Any,
    report_dict: dict,
    project_dir: Path,
    emit_warning: Any | None = None,
) -> dict:
    """Apply the claim-grounding gate to `report_dict` (already serialized).

    Mutates `report_dict` in place and returns it.  `report` is the
    RLMFinalReport pydantic model (read-only here — we work on the dict).
    `emit_warning` is an optional callable(code, message) for SSE warnings.

    Steps:
    1. Extract claims from reproduction_summary + reported_metrics ONLY.
    2. Flatten measured values from disk.
    3. If ≥1 ungrounded RESULT claim AND measured non-empty AND verdict in
       {reproduced, partial}: cap verdict to "partial", append cited note,
       cap two-axis verdicts, recompute meets_target.
    4. Stamp "claim_grounding" key into the dict (flag is already confirmed on).
    """
    from backend.agents.rlm.claim_grounding import (
        check_claims_grounded,
        extract_result_claims,
        flatten_measured_values,
    )

    try:
        # Build search text from the root's ACHIEVED assertions only.
        summary = getattr(report, "reproduction_summary", "") or ""
        reported_metrics = getattr(report, "reported_metrics", None)
        search_text = summary + "\n" + _text_from_reported_metrics(reported_metrics)

        claims = extract_result_claims(search_text)
        measured = flatten_measured_values(project_dir)
        grounding = check_claims_grounded(claims, measured)

        ungrounded = grounding.get("ungrounded", [])
        stamp: dict[str, Any] = {
            "claims_found": len(claims),
            "grounded": len(grounding.get("grounded", [])),
            "ungrounded": len(ungrounded),
        }

        current_verdict = report_dict.get("verdict", "")
        if (
            ungrounded
            and measured  # no-op when no evidence (evidence_gate owns that)
            and current_verdict in ("reproduced", "partial")
        ):
            # Cap verdict at "partial" — never upgrade, never force to "failed".
            report_dict["verdict"] = "partial"

            # Append cited note to reproduction_summary.
            # Build one representative citation (first ungrounded claim).
            first = ungrounded[0]
            note = (
                f"\n\n[harness] report claimed {first.value} ({first.term}); "
                "code/metrics.json has no matching measured value within 5% "
                "— verdict capped at partial."
            )
            existing_summary = report_dict.get("reproduction_summary") or ""
            report_dict["reproduction_summary"] = existing_summary + note

            # Cap two-axis reproducibility/implementation_verdict if present.
            repro = report_dict.get("reproducibility")
            if isinstance(repro, dict):
                for key in ("verdict", "implementation_verdict", "replication_verdict"):
                    if repro.get(key) == "reproduced":
                        repro[key] = "partial"

            # Recompute meets_target from the capped verdict.
            rubric = report_dict.get("rubric")
            if isinstance(rubric, dict):
                overall = rubric.get("overall_score")
                target = rubric.get("target_score")
                try:
                    if overall is not None and target is not None:
                        report_dict.setdefault("rubric", {})
                        report_dict["rubric"]["meets_target"] = float(overall) >= float(target)
                except (TypeError, ValueError):
                    pass
            # Top-level mirror.
            top_overall = report_dict.get("overall_score")
            top_target = report_dict.get("target_score")
            try:
                if top_overall is not None and top_target is not None:
                    report_dict["meets_target"] = float(top_overall) >= float(top_target)
            except (TypeError, ValueError):
                pass

            # Sync the cap back to the report OBJECT — the markdown render and the
            # best-attempt preservation marker read report.verdict (NOT this dict), so
            # without this final_report.md would emit a stale 'reproduced' while
            # final_report.json says 'partial' (codex Area-4).
            try:
                report.verdict = "partial"
                report.reproduction_summary = (
                    getattr(report, "reproduction_summary", "") or ""
                ) + note
                if "meets_target" in report_dict:
                    report.meets_target = report_dict["meets_target"]
                _r_rubric = getattr(report, "rubric", None)
                if (
                    isinstance(_r_rubric, dict)
                    and isinstance(report_dict.get("rubric"), dict)
                    and "meets_target" in report_dict["rubric"]
                ):
                    _r_rubric["meets_target"] = report_dict["rubric"]["meets_target"]
                _r_repro = getattr(report, "reproducibility", None)
                if isinstance(_r_repro, dict):
                    for key in ("verdict", "implementation_verdict", "replication_verdict"):
                        if _r_repro.get(key) == "reproduced":
                            _r_repro[key] = "partial"
            except Exception:  # noqa: BLE001 — object sync best-effort; the json dict is authoritative
                logger.debug("report_claim_gate: report-object sync skipped", exc_info=True)

            stamp["verdict_capped"] = True
            stamp["first_ungrounded"] = {"term": first.term, "value": first.value}
            logger.info(
                "report_claim_gate: capped verdict from %r to 'partial' — "
                "%d ungrounded result claim(s) found in report narrative",
                current_verdict,
                len(ungrounded),
            )
            if emit_warning is not None:
                try:
                    emit_warning(
                        "report_claim_ungrounded",
                        f"Report claims {len(ungrounded)} result value(s) not found in "
                        "code/metrics.json (within 5%); verdict capped at 'partial'.",
                    )
                except Exception:  # noqa: BLE001
                    pass
        else:
            stamp["verdict_capped"] = False

        # Stamp into the dict (flag is already confirmed on by the caller).
        report_dict["claim_grounding"] = stamp

    except Exception:  # noqa: BLE001 — fail-soft, never break the write
        logger.debug("report_claim_gate: skipped due to exception", exc_info=True)

    return report_dict
