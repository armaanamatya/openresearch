"""Post-run PaperBench rubric leaf scorer.

Grades a reproduction run against a PaperBench rubric.json tree by:
1. Flattening the tree to leaves.
2. LLM-grading leaves in batches against gathered run evidence.
3. Rolling up leaf scores through the weighted tree.
4. Amending final_report.json with the rubric block.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class LlmClient(Protocol):
    def complete(self, *, system: str, user: str) -> str:
        ...


# ---------------------------------------------------------------------------
# 1. flatten_leaves
# ---------------------------------------------------------------------------


def flatten_leaves(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Recursively collect all leaf nodes (nodes with empty/missing sub_tasks)."""
    children: list[dict[str, Any]] = [
        c for c in (node.get("sub_tasks") or []) if isinstance(c, dict)
    ]
    if not children:
        return [node]
    leaves: list[dict[str, Any]] = []
    for child in children:
        leaves.extend(flatten_leaves(child))
    return leaves


# ---------------------------------------------------------------------------
# 2. roll_up
# ---------------------------------------------------------------------------


def roll_up(node: dict[str, Any], leaf_scores: dict[str, float]) -> float:
    """Recursive weighted roll-up.

    Leaf: return leaf_scores.get(node["id"], 0.0).
    Non-leaf: weighted average of children scores.
    """
    children: list[dict[str, Any]] = [
        c for c in (node.get("sub_tasks") or []) if isinstance(c, dict)
    ]
    if not children:
        return leaf_scores.get(str(node.get("id", "")), 0.0)

    total_weight = sum(float(c.get("weight", 0.0) or 0.0) for c in children)
    if total_weight == 0.0:
        return 0.0

    weighted_sum = sum(
        roll_up(c, leaf_scores) * float(c.get("weight", 0.0) or 0.0)
        for c in children
    )
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Honesty backstop (C2b)
#
# A run that reached _finalize() without producing measured numeric metrics
# (baseline_metrics={}) is "degraded": the experiment either never ran or ran
# without writing metrics.json. A lenient LLM grader on metric-less evidence
# can still hand out high leaf scores by reading the code; that score does not
# describe a reproduction. Cap each leaf at DEGRADED_LEAF_CEILING so the
# rolled-up overall_score is bounded by the same ceiling.
#
# The 0.35 number is inherited from the verify_against_rubric backstop that
# lived in primitives.py before 2e1ce37 consolidated the in-loop and post-run
# scoring paths through score_reproduction.
# ---------------------------------------------------------------------------

DEGRADED_LEAF_CEILING: float = 0.35

# Minimal field set that distinguishes an RLM-mode final_report from an SDK-mode
# one. Used by _rerender_report_markdown to detect RLM reports without requiring
# ALL RLMFinalReport fields — that prior approach re-broke every time the schema
# gained a new field (regression of T21: primitive_provider + degraded added).
_RLM_SIGNATURE_FIELDS: frozenset[str] = frozenset({"verdict", "baseline_metrics", "paper", "rubric"})


def _is_degraded_run(run_dir: Path) -> bool:
    """Decide whether the run produced no measured metrics.

    A run is degraded when final_report.json exists with baseline_metrics
    empty/missing — the RLMFinalReport contract for "no metrics were measured."
    Missing or unreadable final_report.json is treated as NOT degraded (do not
    cap on uncertainty) so this is safe to call in-loop, before the report has
    been written.

    Callers with a results dict in hand (verify_against_rubric) should NOT
    rely on this auto-detection alone — pass `degraded` explicitly via
    score_reproduction's kwarg so the in-loop signal is correct too.
    """
    report_path = run_dir / "final_report.json"
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — unreadable → don't cap on uncertainty
        return False
    if not isinstance(report, dict):
        return False
    metrics = report.get("baseline_metrics") or {}
    verdict = report.get("verdict", "")
    return (not metrics) or verdict == "failed"


# ---------------------------------------------------------------------------
# Evidence gathering
# ---------------------------------------------------------------------------

_MAX_FILE_BYTES = 6 * 1024          # 6 KB per file
_MAX_TOTAL_EVIDENCE_BYTES = 40 * 1024  # 40 KB total


def _gather_evidence(run_dir: Path) -> str:
    """Gather bounded reproduction evidence from a run directory."""
    parts: list[str] = []
    total = 0

    # final_report.json — reproduction_summary + measured metrics + paper id
    # C2a fix: read the RLMFinalReport schema's real keys.  The previous list
    # ("metrics", "paper_title") was a guess at SDK-mode field names; RLM-mode
    # reports carry "baseline_metrics" (dict) and "paper" (dict).  Reading the
    # wrong keys meant every RLM run was graded against evidence with no
    # metrics and no paper identity — the grader had nothing to ground on.
    report_path = run_dir / "final_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            snippet = {
                k: report[k]
                for k in ("reproduction_summary", "baseline_metrics", "verdict", "paper")
                if k in report
            }
            text = f"=== final_report.json (key fields) ===\n{json.dumps(snippet, indent=2)}\n"
            parts.append(text)
            total += len(text)
        except Exception as exc:
            logger.warning("Could not read final_report.json: %s", exc)

    # code/ directory listing
    code_dir = run_dir / "code"
    if code_dir.exists():
        listing_lines: list[str] = []
        for path in sorted(code_dir.rglob("*"))[:200]:
            if path.is_file():
                listing_lines.append(str(path.relative_to(code_dir)))
        listing = "=== code/ listing (first 200 files) ===\n" + "\n".join(listing_lines) + "\n"
        parts.append(listing)
        total += len(listing)

    # Key code files
    if code_dir.exists() and total < _MAX_TOTAL_EVIDENCE_BYTES:
        priority_extensions = {".py", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".txt"}
        for path in sorted(code_dir.rglob("*")):
            if total >= _MAX_TOTAL_EVIDENCE_BYTES:
                break
            if not path.is_file():
                continue
            if path.suffix not in priority_extensions:
                continue
            try:
                raw = path.read_bytes()[:_MAX_FILE_BYTES]
                content = raw.decode("utf-8", errors="replace")
                header = f"\n=== code/{path.relative_to(code_dir)} ===\n"
                chunk = header + content + "\n"
                parts.append(chunk)
                total += len(chunk)
            except Exception:
                pass

    return "".join(parts) if parts else "(no reproduction evidence found)"


# ---------------------------------------------------------------------------
# 3. score_reproduction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a strict research reproducibility judge evaluating whether a paper reproduction \
satisfies specific rubric requirements.

You will be given:
1. Evidence from the reproduction run (code, reports, logs).
2. A batch of rubric leaf tasks, each with an id and requirements text.

For EACH leaf task, output a JSON object with:
- "leaf_id": the task id (string, copy exactly)
- "score": float 0.0 to 1.0 (0.0 = not satisfied at all, 1.0 = fully satisfied)
- "justification": one sentence explaining the score

Output ONLY a JSON array of these objects, no other text. Example:
[{"leaf_id": "abc-123", "score": 0.8, "justification": "The model is implemented but missing dropout."}]

Be conservative: score 0.0 when there is no evidence either way.
"""

_USER_TEMPLATE = """\
## Reproduction evidence

{evidence}

## Rubric leaf tasks to grade (batch {batch_num})

{tasks_json}

Grade EACH task based solely on what the evidence shows. Return a JSON array.
"""


def score_reproduction(
    rubric_tree: dict[str, Any],
    run_dir: Path,
    llm_client: LlmClient,
    *,
    batch_size: int = 15,
    rubric_source: str = "paperbench_bundle",
    degraded: bool | None = None,
) -> dict[str, Any]:
    """Grade a reproduction run against a PaperBench rubric tree.

    Returns a dict with overall_score, leaf_count, graded, rubric_source,
    leaf_scores, degraded, target_score.

    ``rubric_source`` is passed through to the result dict unchanged — callers set
    it to "generated" when the rubric was derived at run-time rather than from a
    vendored bundle.

    ``degraded`` (C2b): when True, every leaf score is capped at
    DEGRADED_LEAF_CEILING (0.35) before roll-up — the honesty backstop for runs
    that produced no measured metrics. ``None`` (default) auto-detects via
    :func:`_is_degraded_run` (reads ``final_report.json`` for an empty
    ``baseline_metrics``). Callers with a results dict in hand should pass
    ``degraded`` explicitly so the in-loop case (no final_report.json on disk
    yet) is also capped.
    """
    leaves = flatten_leaves(rubric_tree)
    evidence = _gather_evidence(run_dir)
    if degraded is None:
        degraded = _is_degraded_run(run_dir)

    leaf_scores: dict[str, float] = {}
    leaf_score_records: list[dict[str, Any]] = []
    graded_count = 0

    if degraded:
        for leaf in leaves:
            lid = str(leaf.get("id", ""))
            leaf_scores[lid] = 0.0
            leaf_score_records.append(
                {
                    "id": lid,
                    "score": 0.0,
                    "justification": "degraded_no_metrics",
                }
            )

        raw_target = rubric_tree.get("target_score")
        try:
            target_score: float | None = (
                None if raw_target is None else max(0.0, min(1.0, float(raw_target)))
            )
        except (TypeError, ValueError):
            target_score = None

        return {
            "overall_score": roll_up(rubric_tree, leaf_scores),
            "leaf_count": len(leaves),
            "graded": graded_count,
            "rubric_source": rubric_source,
            "leaf_scores": leaf_score_records,
            "degraded": True,
            "target_score": target_score,
        }

    for batch_num, start in enumerate(range(0, len(leaves), batch_size), 1):
        batch = leaves[start : start + batch_size]
        tasks_payload = [
            {"leaf_id": str(leaf.get("id", "")), "requirements": str(leaf.get("requirements", ""))}
            for leaf in batch
        ]
        user_msg = _USER_TEMPLATE.format(
            evidence=evidence,
            tasks_json=json.dumps(tasks_payload, indent=2),
            batch_num=batch_num,
        )

        try:
            raw = llm_client.complete(system=_SYSTEM_PROMPT, user=user_msg)
            results = _parse_batch_response(raw, batch)
        except Exception as exc:
            logger.warning(
                "Batch %d LLM call failed (%s); defaulting all %d leaves to 0.0",
                batch_num,
                exc,
                len(batch),
            )
            results = [
                {
                    "id": str(leaf.get("id", "")),
                    "score": 0.0,
                    "justification": "batch_error",
                    "_graded": False,
                }
                for leaf in batch
            ]

        for rec in results:
            lid = rec["id"]
            score = rec["score"]
            # C2b: clamp degraded leaves to the honesty ceiling before storing
            # so the rolled-up overall_score, the returned leaf_score_records,
            # and any "weak leaves" surface all reflect the cap consistently.
            if degraded and score > DEGRADED_LEAF_CEILING:
                score = DEGRADED_LEAF_CEILING
            leaf_scores[lid] = score
            leaf_score_records.append(
                {"id": lid, "score": score, "justification": rec["justification"]}
            )
            if rec.get("_graded", True):
                graded_count += 1

    overall_score = roll_up(rubric_tree, leaf_scores)

    # C2c: surface target_score so amend_final_report can compute meets_target
    # honestly. None when the rubric tree has no target — never fabricate.
    raw_target = rubric_tree.get("target_score")
    try:
        target_score: float | None = (
            None if raw_target is None else max(0.0, min(1.0, float(raw_target)))
        )
    except (TypeError, ValueError):
        target_score = None

    return {
        "overall_score": overall_score,
        "leaf_count": len(leaves),
        "graded": graded_count,
        "rubric_source": rubric_source,
        "leaf_scores": leaf_score_records,
        "degraded": degraded,
        "target_score": target_score,
    }


def _parse_batch_response(
    raw: str, batch: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Parse LLM batch response robustly. Ungraded/malformed leaves -> 0.0."""
    batch_ids = {str(leaf.get("id", "")): leaf for leaf in batch}
    results: dict[str, dict[str, Any]] = {}

    # Try to extract JSON array from response
    raw = raw.strip()
    try:
        from backend.agents.rlm.primitives import _extract_json_array
        parsed = _extract_json_array(raw)
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                lid = str(item.get("leaf_id", ""))
                if not lid or lid not in batch_ids:
                    continue
                try:
                    score = max(0.0, min(1.0, float(item.get("score", 0.0))))
                except (TypeError, ValueError):
                    score = 0.0
                justification = str(item.get("justification", ""))
                results[lid] = {"id": lid, "score": score, "justification": justification, "_graded": True}
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Could not parse batch response as JSON: %s", exc)

    # Fill in any missing leaves with 0.0
    out: list[dict[str, Any]] = []
    for lid in batch_ids:
        if lid in results:
            out.append(results[lid])
        else:
            out.append({"id": lid, "score": 0.0, "justification": "ungraded", "_graded": False})
    return out


# ---------------------------------------------------------------------------
# 4. amend_final_report
# ---------------------------------------------------------------------------


def amend_final_report(run_dir: Path, score: dict[str, Any]) -> None:
    """Load final_report.json, set its rubric field, write back atomically.

    Also re-renders final_report.md so ``GET /runs/{id}/final-report`` (which
    serves the markdown) reflects this authoritative leaf score — not the stale
    in-loop ``verify_against_rubric`` score the run wrote at finish time.
    """
    report_path = run_dir / "final_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {}

    # C2c: compute meets_target from the real target_score score_reproduction
    # now threads through. When the rubric tree has no target_score (e.g. a
    # self-generated arXiv rubric without a configured target), both
    # target_score and meets_target are written as null — never a fabricated
    # False, which used to flip a legitimate high score to "✘ below target".
    target_score = score.get("target_score")
    if target_score is None:
        meets_target: bool | None = None
    else:
        meets_target = bool(score["overall_score"] >= target_score)

    # T5: preserve the in-loop tree-rubric areas list so the markdown areas
    # table is not silently dropped when we replace report["rubric"].
    previous_rubric = report.get("rubric", {}) or {}
    report["rubric"] = {
        "overall_score": score["overall_score"],
        "rubric_source": score.get("rubric_source", "paperbench_bundle"),
        "leaf_count": score["leaf_count"],
        "graded": score["graded"],
        "target_score": target_score,
        "meets_target": meets_target,
        # C2b: surface the degraded flag so the UI / human reviewer can see
        # *why* a low score was reached. False/missing → run was honest.
        "degraded": bool(score.get("degraded", False)),
        "areas": previous_rubric.get("areas", []),
    }

    # Reconcile the self-reported verdict against the authoritative leaf score.
    # Symptom: the `ftrl` run wrote verdict="reproduced" at overall_score=0.0.
    # This must happen BEFORE the atomic write and before _rerender_report_markdown
    # so the markdown re-render picks up the corrected verdict automatically.
    if "verdict" in report:
        try:
            from backend.agents.rlm.report import reconcile_verdict_with_score  # lazy import
            report["verdict"] = reconcile_verdict_with_score(
                report["verdict"], score["overall_score"]
            )
        except Exception as exc:  # noqa: BLE001 — reconciliation is best-effort
            logger.warning(
                "amend_final_report: verdict reconciliation failed (%s) — "
                "verdict may be inconsistent with rubric score",
                exc,
            )

    tmp_fd, tmp_path = tempfile.mkstemp(dir=run_dir, prefix=".final_report_", suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        os.replace(tmp_path, report_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    _rerender_report_markdown(run_dir, report)


def _rerender_report_markdown(run_dir: Path, report: dict[str, Any]) -> None:
    """Re-render final_report.md from an amended RLM report dict.

    The post-run leaf scorer updates final_report.json's rubric block; the
    markdown the HTTP layer serves must stay consistent with it. Only RLM-mode
    reports are re-rendered — the markdown renderer is RLM-specific; for any
    other report shape (or a missing markdown file) this is a no-op.
    """
    md_path = run_dir / "final_report.md"
    if not md_path.exists():
        return
    try:
        # Lazy import — keeps backend.evals import-light and breaks no cycle.
        from backend.agents.rlm.report import RLMFinalReport, _render_markdown

        # Detect RLM-mode reports by signature fields, not by full-set equality —
        # the schema can grow without breaking this re-render path (regression of T21).
        if not _RLM_SIGNATURE_FIELDS.issubset(report.keys()):
            return  # not an RLM-mode report — leave its markdown untouched
        all_fields = set(RLMFinalReport.model_fields)
        obj = RLMFinalReport(**{k: v for k, v in report.items() if k in all_fields})
        md = _render_markdown(obj)
    except Exception as exc:  # noqa: BLE001 — markdown refresh is best-effort
        logger.warning(
            "amend_final_report: could not re-render final_report.md (%s) — "
            "it may show a stale rubric score",
            exc,
        )
        return
    tmp_fd, tmp_path = tempfile.mkstemp(dir=run_dir, prefix=".final_report_", suffix=".md")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(md)
        os.replace(tmp_path, md_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
