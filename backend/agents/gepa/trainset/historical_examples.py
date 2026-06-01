"""Load GEPA training examples from previous runs of the same arxiv_id."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_historical_examples(
    *,
    runs_root: Path,
    arxiv_id: str | None,
    primitive_name: str,
    max_examples: int = 20,
) -> list[dict]:
    """Scan runs_root for gepa_examples.jsonl files matching arxiv_id + primitive_name.

    Returns up to max_examples entries sorted by recency (most recent first).
    """
    if not arxiv_id:
        return []
    matches: list[tuple[float, dict]] = []
    try:
        for jsonl_path in runs_root.rglob("gepa_examples.jsonl"):
            try:
                mtime = jsonl_path.stat().st_mtime
                with jsonl_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if (row.get("arxiv_id") == arxiv_id
                                and row.get("primitive") == primitive_name):
                            matches.append((mtime, row))
            except Exception:
                continue
    except Exception as exc:
        logger.debug("load_historical_examples error: %s", exc)
        return []
    matches.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in matches[:max_examples]]
