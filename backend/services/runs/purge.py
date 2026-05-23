"""Atomic two-store purge — clears both the run dir and the event_store aggregates for a project_id.

Re-running a paper with the same project_id used to ConcurrencyError because
`rm -rf runs/<id>` does not touch the SQLite event_store_events table. This
module purges both surfaces in one call so the next run starts at version 0.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from backend.config import get_settings

logger = logging.getLogger(__name__)


def purge_project(project_id: str, runs_root: Path) -> dict:
    """Delete runs/<project_id>/ AND every event_store aggregate with that prefix.

    Delegates to ``SqliteEventStore.purge_project_aggregates`` which covers:
      - ``<project_id>``                 (the intake/root aggregate)
      - ``<project_id>:<suffix>``        (parsed, index, discovery, …)
      - ``rlm-run:<project_id>``         (the RLM iteration checkpointer)

    Returns: ``{"run_dir_removed": bool, "aggregates_removed": int}``.

    Failure modes:
      - Missing run dir: not an error (idempotent).
      - SQLite errors: propagate (caller decides whether to retry).
    """
    from backend.eventstore.sqlite_store import SqliteEventStore

    runs_root = Path(runs_root)
    run_dir = runs_root / project_id
    run_dir_removed = False
    if run_dir.exists():
        shutil.rmtree(run_dir)
        run_dir_removed = True
        logger.info("purge_project: removed run dir %s", run_dir)

    store = SqliteEventStore(get_settings().database_url)
    try:
        aggregates_removed = store.purge_project_aggregates(project_id)
        logger.info(
            "purge_project: removed %d event_store aggregate(s) for %s",
            aggregates_removed, project_id,
        )
    finally:
        store.close()

    return {"run_dir_removed": run_dir_removed, "aggregates_removed": aggregates_removed}
