"""Persistence helpers for Hermes audit reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.hermes_audit.models import HermesAuditReport, HermesAuditScope


class HermesAuditStorage:
    """Stores audit reports under a run-scoped `hermes/` directory."""

    _INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

    def __init__(self, runs_root: Path, project_id: str) -> None:
        self._base_dir = Path(runs_root) / project_id / "hermes"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._base_dir / "index.json"

    def _report_path(self, report: HermesAuditReport) -> Path:
        prefix = "step" if report.scope == HermesAuditScope.step else "checkpoint"
        safe_target = self._INVALID_FILENAME_CHARS.sub("_", report.target)
        return self._base_dir / f"{prefix}-{safe_target}.json"

    def save_report(self, report: HermesAuditReport) -> Path:
        path = self._report_path(report)
        # Explicit UTF-8 on every write — audit reports routinely contain
        # arrows, em-dashes, Greek letters, etc., and Path.write_text() on
        # Windows defaults to cp1252 (charmap), which crashes on those.
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        index = self.load_index()
        key = f"{report.scope.value}:{report.target}"
        index[key] = {
            "path": str(path),
            "status": report.status.value,
            "recommended_intervention": report.recommended_intervention.value,
            "summary": report.summary,
        }
        self._index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        return path

    def load_index(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {}
        return json.loads(self._index_path.read_text(encoding="utf-8"))
