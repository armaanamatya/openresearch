"""rubric_gen.py — self-generate a PaperBench-shaped rubric tree from paper text.

For arXiv runs that arrive without a vendored rubric.json, this module derives
a structurally compatible rubric from the paper itself so the run is scorable
by ``backend.evals.paperbench.leaf_scorer`` (flatten_leaves / roll_up).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Protocol

logger = logging.getLogger(__name__)


class LlmClient(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...


# ---------------------------------------------------------------------------
# System prompt — instructs the LLM to produce the six-category rubric JSON.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a research-reproduction rubric author for ReproLab.

You are given the full text of a research paper. Produce a PaperBench-style
weighted rubric that a grader will use to score an attempted reproduction of
that paper. The rubric grades only concrete reproduction artifacts — source
code, the environment, executed runs, produced metrics and plots — never
process, effort, or how the reproduction was carried out.

Organize the rubric under these six categories. The weight of each category
should fall in the range shown (weights are relative — they need not sum to
exactly 1):

  Method and code fidelity to the paper             0.30 - 0.45
  Data and preprocessing fidelity                   0.10 - 0.20
  Experiment execution and reproducibility          0.15 - 0.25
  Evaluation protocol and metric correctness        0.15 - 0.25
  Result match versus the paper's reported targets  0.15 - 0.30
  Artifact completeness and provenance              0.05 - 0.10

For each category write 2 to 5 leaf criteria. Each leaf is ONE concrete,
independently checkable requirement, specific to THIS paper: name the actual
model component, dataset, hyperparameter, equation, training detail, metric, or
reported result it refers to. A grader must be able to score it from the
reproduction's artifacts alone.

Good leaf:  "train.py implements the two-layer bidirectional GRU encoder with
            hidden size 256 described in Section 3.1."
Weak leaf:  "The model is implemented correctly."

Give every leaf a relative weight within its category.

Return ONLY this JSON object and nothing else:

{
  "categories": [
    {
      "name": "Method and code fidelity to the paper",
      "weight": 0.40,
      "leaves": [
        {"requirements": "<concrete paper-specific criterion>", "weight": 0.3}
      ]
    }
  ]
}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_rubric_tree(
    paper_text: str,
    llm_client: LlmClient,
    *,
    paper_title: str = "",
    max_attempts: int = 3,
    max_paper_chars: int = 48000,
) -> dict | None:
    """Derive a PaperBench-shaped rubric tree from a paper's full text.

    Returns a rubric dict compatible with ``flatten_leaves`` / ``roll_up``, or
    ``None`` if the paper is too short to derive a rubric from, or if all LLM
    attempts fail (honest degradation — the run proceeds rubric-less).
    """
    if len(paper_text.strip()) < 500:
        logger.warning(
            "generate_rubric_tree: paper text too short (%d chars) — skipping rubric generation",
            len(paper_text.strip()),
        )
        return None

    user_msg = (
        f"Paper title: {paper_title}\n\nPaper text:\n\n{paper_text[:max_paper_chars]}"
    )

    last_error: str = "unknown"
    for attempt in range(1, max_attempts + 1):
        try:
            raw = llm_client.complete(system=_SYSTEM_PROMPT, user=user_msg)
        except Exception as exc:
            last_error = f"LLM exception on attempt {attempt}: {exc}"
            logger.warning("generate_rubric_tree: %s", last_error)
            continue

        parsed = _extract_json_object(raw)
        if parsed is None:
            last_error = f"unparseable JSON on attempt {attempt}"
            logger.warning("generate_rubric_tree: %s", last_error)
            continue

        categories = _clean_categories(parsed.get("categories") or [])
        if not categories or sum(len(c["leaves"]) for c in categories) == 0:
            last_error = f"empty categories/leaves after cleaning on attempt {attempt}"
            logger.warning("generate_rubric_tree: %s", last_error)
            continue

        tree = _build_tree(categories, paper_title)
        leaf_count = sum(len(c["leaves"]) for c in categories)
        logger.info(
            "generate_rubric_tree: built rubric — %d leaves across %d categories",
            leaf_count,
            len(categories),
        )
        return tree

    logger.warning(
        "generate_rubric_tree: all %d attempts failed — last: %s",
        max_attempts,
        last_error,
    )
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_json_object(raw: str) -> dict | None:
    """Extract the first complete JSON object from a string (first { to last })."""
    raw = raw.strip()
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def _clean_categories(raw_categories: list) -> list[dict]:
    """Drop malformed categories and leaves; return a clean list."""
    cleaned: list[dict] = []
    for cat in raw_categories:
        if not isinstance(cat, dict):
            continue
        name = cat.get("name", "")
        if not isinstance(name, str) or not name.strip():
            continue
        raw_leaves = cat.get("leaves") or []
        good_leaves = [
            lf for lf in raw_leaves
            if isinstance(lf, dict)
            and isinstance(lf.get("requirements", ""), str)
            and lf.get("requirements", "").strip()
        ]
        if not good_leaves:
            continue
        cleaned.append({"name": name.strip(), "weight": cat.get("weight"), "leaves": good_leaves})
    return cleaned


def _normalize_weights(weights: list) -> list[float]:
    """Normalize raw weights to sum to 1.0.

    A weight that is None, <= 0, or non-numeric is filled with the **mean of the
    valid weights** in the level — so a leaf with a missing weight still counts,
    rather than silently dropping to weight 0 and being excluded from `roll_up`.
    If no weight in the level is valid, every entry gets an equal share.
    """
    coerced: list[float | None] = []
    for w in weights:
        try:
            v = float(w)
        except (TypeError, ValueError):
            v = None
        coerced.append(v if (v is not None and v > 0.0) else None)

    valid = [v for v in coerced if v is not None]
    if not valid:
        n = len(coerced)
        return [1.0 / n] * n if n else []

    fill = sum(valid) / len(valid)
    filled = [v if v is not None else fill for v in coerced]
    total = sum(filled)
    return [v / total for v in filled]


def _build_tree(categories: list[dict], paper_title: str) -> dict:
    """Build the rubric tree from cleaned categories."""
    cat_weights_raw = [c.get("weight") for c in categories]
    cat_weights = _normalize_weights(cat_weights_raw)

    category_nodes: list[dict] = []
    for cat, cat_w in zip(categories, cat_weights):
        leaf_weights_raw = [lf.get("weight") for lf in cat["leaves"]]
        leaf_weights = _normalize_weights(leaf_weights_raw)

        leaf_nodes: list[dict] = [
            {
                "id": uuid.uuid4().hex,
                "requirements": lf["requirements"].strip(),
                "weight": lw,
                "task_category": cat["name"],
                "finegrained_task_category": None,
                "sub_tasks": [],
            }
            for lf, lw in zip(cat["leaves"], leaf_weights)
        ]

        category_nodes.append({
            "id": uuid.uuid4().hex,
            "requirements": cat["name"],
            "weight": cat_w,
            "task_category": None,
            "finegrained_task_category": None,
            "sub_tasks": leaf_nodes,
        })

    return {
        "id": uuid.uuid4().hex,
        "requirements": f"Reproduce: {paper_title or 'the paper'}",
        "weight": 1.0,
        "task_category": None,
        "finegrained_task_category": None,
        "sub_tasks": category_nodes,
    }
