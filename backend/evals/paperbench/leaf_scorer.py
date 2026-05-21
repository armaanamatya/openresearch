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
# Evidence gathering
# ---------------------------------------------------------------------------

_MAX_FILE_BYTES = 6 * 1024          # 6 KB per file
_MAX_TOTAL_EVIDENCE_BYTES = 40 * 1024  # 40 KB total


def _gather_evidence(run_dir: Path) -> str:
    """Gather bounded reproduction evidence from a run directory."""
    parts: list[str] = []
    total = 0

    # final_report.json — reproduction_summary + metrics
    report_path = run_dir / "final_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            snippet = {
                k: report[k]
                for k in ("reproduction_summary", "metrics", "verdict", "paper_title")
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
) -> dict[str, Any]:
    """Grade a reproduction run against a PaperBench rubric tree.

    Returns a dict with overall_score, leaf_count, graded, rubric_source, leaf_scores.
    ``rubric_source`` is passed through to the result dict unchanged — callers set
    it to "generated" when the rubric was derived at run-time rather than from a
    vendored bundle.
    """
    leaves = flatten_leaves(rubric_tree)
    evidence = _gather_evidence(run_dir)

    leaf_scores: dict[str, float] = {}
    leaf_score_records: list[dict[str, Any]] = []
    graded = 0

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
                {"id": str(leaf.get("id", "")), "score": 0.0, "justification": "batch_error"}
                for leaf in batch
            ]

        for rec in results:
            lid = rec["id"]
            score = rec["score"]
            leaf_scores[lid] = score
            leaf_score_records.append(
                {"id": lid, "score": score, "justification": rec["justification"]}
            )
            if rec.get("_graded", True):
                graded += 1

    overall_score = roll_up(rubric_tree, leaf_scores)

    return {
        "overall_score": overall_score,
        "leaf_count": len(leaves),
        "graded": graded,
        "rubric_source": rubric_source,
        "leaf_scores": leaf_score_records,
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
        # Find first '[' and last ']'
        start = raw.index("[")
        end = raw.rindex("]") + 1
        parsed = json.loads(raw[start:end])
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
    """Load final_report.json, set its rubric field, write back atomically."""
    report_path = run_dir / "final_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {}

    report["rubric"] = {
        "overall_score": score["overall_score"],
        "rubric_source": score.get("rubric_source", "paperbench_bundle"),
        "leaf_count": score["leaf_count"],
        "graded": score["graded"],
        "meets_target": False,
    }

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
