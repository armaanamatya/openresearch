"""Opt-in terminal run notification.

POST a short message to a Slack / Discord / generic incoming webhook when a run
reaches a terminal state. Opt-in and fail-soft: a no-op unless
``OPENRESEARCH_NOTIFY_WEBHOOK_URL`` (or the ``webhook_url`` argument) is set, and
it never raises — a notification failure must not change a run's outcome.

The function reads the run's own on-disk terminal truth (``final_report.json``
first, ``demo_status.json`` as a fallback) so every terminal path in ``run.py``
can fire it with a single argument. Stdlib-only (``urllib``); no new dependency.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WEBHOOK_ENV = "OPENRESEARCH_NOTIFY_WEBHOOK_URL"
_BUCKET_ENV = "OPENRESEARCH_GCP_GCS_BUCKET"
_TIMEOUT_S = 10.0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _extract_score(report: dict[str, Any]) -> float | None:
    """Defensive score read: canonical nested rubric first, then flat fallbacks."""
    rubric = report.get("rubric")
    if isinstance(rubric, dict):
        val = rubric.get("overall_score")
        if isinstance(val, (int, float)):
            return float(val)
    for key in ("overall_score", "rubric_overall_score", "rubric_score"):
        val = report.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _extract_target(report: dict[str, Any]) -> float | None:
    rubric = report.get("rubric")
    if isinstance(rubric, dict):
        val = rubric.get("target_score")
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _stop_reason_text(report: dict[str, Any]) -> str | None:
    sr = report.get("stop_reason")
    if isinstance(sr, dict):
        for key in ("reason", "kind", "code"):
            val = sr.get(key)
            if isinstance(val, str) and val:
                return val
        return None
    if isinstance(sr, str) and sr:
        return sr
    return None


def _is_success(report: dict[str, Any], status: str) -> bool:
    """Prefer the explicit meets_target flag; else infer from the status string."""
    rubric = report.get("rubric")
    if isinstance(rubric, dict) and isinstance(rubric.get("meets_target"), bool):
        return bool(rubric["meets_target"])
    lowered = (status or "").lower()
    return not any(bad in lowered for bad in ("fail", "kill", "interrupt", "error"))


def _payload_for_host(url: str, message: str) -> dict[str, str]:
    """Slack incoming webhooks accept ``text``; Discord accepts ``content``."""
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    if "discord" in host:
        return {"content": message}
    return {"text": message}


def notify_run_terminal(project_dir: str | Path, *, webhook_url: str | None = None) -> bool:
    """POST a terminal-status message for the run rooted at ``project_dir``.

    Returns True on a 2xx response, False on no-op (no webhook configured) or on
    any failure. Never raises.
    """
    url = (webhook_url if webhook_url is not None else os.environ.get(WEBHOOK_ENV, "")).strip()
    if not url:
        return False
    try:
        pdir = Path(project_dir)
        report = _read_json(pdir / "final_report.json")
        status_doc = _read_json(pdir / "demo_status.json")

        run_id = (
            report.get("run_id")
            or report.get("project_id")
            or status_doc.get("projectId")
            or pdir.name
        )
        status = report.get("verdict") or status_doc.get("status") or "unknown"
        score = _extract_score(report)
        target = _extract_target(report)
        stop_reason = _stop_reason_text(report)
        success = _is_success(report, str(status))

        bucket = os.environ.get(_BUCKET_ENV, "").strip()
        gcs_prefix = f"gs://{bucket}/runs/{run_id}/" if bucket else None

        parts = [f"{'✅' if success else '❌'} OpenResearch run `{run_id}` — {status}"]
        if score is not None:
            parts.append(f"score {score:.3f}{f'/{target:.3f}' if target is not None else ''}")
        if stop_reason:
            parts.append(f"stop: {stop_reason}")
        if gcs_prefix:
            parts.append(f"artifacts: {gcs_prefix}")
        message = " · ".join(parts)

        data = json.dumps(_payload_for_host(url, message)).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            ok = 200 <= int(getattr(resp, "status", 0)) < 300
        if not ok:
            logger.warning("run-notify: webhook POST returned non-2xx")
        return ok
    except Exception as exc:  # noqa: BLE001 — a notification must never break a run
        logger.warning("run-notify: terminal notification failed: %s", exc)
        return False
