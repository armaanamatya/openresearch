"""Persist gepa_example_buffer to runs/{id}/gepa_examples.jsonl at run end.

Called from run.py in the finalization block.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def save_gepa_examples(ctx: Any) -> None:
    """Append ctx.gepa_example_buffer entries to gepa_examples.jsonl."""
    if not getattr(ctx, "gepa_example_buffer", None):
        return
    out_path = ctx.project_dir / "gepa_examples.jsonl"
    try:
        with out_path.open("a", encoding="utf-8") as f:
            for primitive_name, entries in ctx.gepa_example_buffer.items():
                for entry in entries:
                    row = {
                        "primitive": primitive_name,
                        "arxiv_id": getattr(ctx, "arxiv_id", None),
                        **entry,
                    }
                    f.write(json.dumps(row, default=str) + "\n")
        total = sum(len(v) for v in ctx.gepa_example_buffer.values())
        logger.info("saved %d gepa examples to %s", total, out_path)
    except Exception as exc:
        logger.warning("save_gepa_examples failed: %s", exc)
