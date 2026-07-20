#!/usr/bin/env python3
"""Append a row to the coworker-facing SDAR / GCP run ledger.

Dynamic + fail-soft. Static fields you know up front (command, root, scope,
sandbox/instance) come from flags; the outcome fields (status, score, verdict,
cost, cells, stop) are *derived from the run directory on disk* when present:

  runs/<project_id>/final_report.json   -> verdict, rubric score, target, cost, stop
  runs/<project_id>/demo_status.json    -> status
  runs/<project_id>/cost_ledger.jsonl   -> cost (fallback sum if report has none)
  runs/<project_id>/experiment_runs.jsonl -> cells ok/total

Every field degrades to "-" rather than raising, so logging never blocks a run.
The ledger is append-only markdown (newest row at the bottom) so concurrent
appends from different runs can never corrupt an earlier row.

Usage:
  # at launch (static fields known up front):
  python scripts/sdar_runlog.py --project-id sdar_gcp_x --event launched \
      --root claude-oauth --scope full-grid --sandbox "gcp/sdar-ultra/us-central1-c" \
      --command "PRIMARY=1 ROOT=claude-oauth scripts/sdar_gcp_e2e.sh run" \
      --note "first run on merged code"

  # any time after (outcome auto-derives from the run dir):
  python scripts/sdar_runlog.py --project-id sdar_gcp_x --event finished
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO / "runs" / "sdar-runlog.md"

COLUMNS = [
    "UTC", "project_id", "event", "root/model", "scope", "sandbox/instance",
    "status", "score", "verdict", "cost USD", "cells ok/total", "stop",
    "run dir", "note",
]


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%MZ")


def _load_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _score(report: dict):
    rub = report.get("rubric")
    if isinstance(rub, dict):
        for key in ("overall_score", "adjusted_score", "overall"):
            v = _num(rub.get(key))
            if v is not None:
                meets = rub.get("meets_target")
                tag = "" if meets is None else (" ✓" if meets else " ✗")
                tgt = _num(rub.get("target_score"))
                return f"{round(v, 4)}{f'/{round(tgt, 4)}' if tgt is not None else ''}{tag}"
    v = _num(report.get("rubric_score"))
    return str(round(v, 4)) if v is not None else None


def _verdict(report: dict):
    repro = report.get("reproducibility")
    if isinstance(repro, dict):
        v = repro.get("replication_verdict") or repro.get("verdict")
        if v:
            return v
    return report.get("verdict")


def _cost(report: dict, run_dir: pathlib.Path):
    c = report.get("cost")
    v = _num(c)
    if v is not None:
        return round(v, 4)
    if isinstance(c, dict):
        for key in ("total_usd", "usd", "total", "amount_usd"):
            v = _num(c.get(key))
            if v is not None:
                return round(v, 4)
    total, found = 0.0, False
    try:
        for line in (run_dir / "cost_ledger.jsonl").read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            for key in ("usd", "cost_usd", "amount_usd"):
                v = _num(entry.get(key))
                if v is not None:
                    total += v
                    found = True
                    break
    except Exception:
        return None
    return round(total, 4) if found else None


def _cells(run_dir: pathlib.Path):
    total = ok = 0
    try:
        for line in (run_dir / "experiment_runs.jsonl").read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            total += 1
            if entry.get("success") is True:
                ok += 1
    except Exception:
        return None
    return f"{ok}/{total}" if total else None


def _stop(report: dict):
    sr = report.get("stop_reason")
    if isinstance(sr, dict):
        return sr.get("kind") or sr.get("detail")
    return sr


def main() -> None:
    ap = argparse.ArgumentParser(description="Append a row to the SDAR/GCP coworker run ledger.")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--event", default="update", help="launched | update | finished | killed | <free text>")
    ap.add_argument("--root", default="")
    ap.add_argument("--scope", default="")
    ap.add_argument("--sandbox", default="")
    ap.add_argument("--command", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    args = ap.parse_args()

    run_dir = REPO / "runs" / args.project_id
    report = _load_json(run_dir / "final_report.json") or {}
    status_doc = _load_json(run_dir / "demo_status.json") or {}

    models = report.get("models") if isinstance(report.get("models"), dict) else {}
    root = args.root or models.get("planner") or models.get("executor")

    note = args.note
    if args.command:
        note = f"{note} · " if note else ""
        note += f"`{args.command}`"

    def cell(x):
        return "-" if x is None or x == "" else str(x).replace("|", "/").replace("\n", " ")

    values = [
        _utc(), args.project_id, args.event, root, args.scope, args.sandbox,
        status_doc.get("status"), _score(report), _verdict(report),
        _cost(report, run_dir), _cells(run_dir), _stop(report),
        f"runs/{args.project_id}/", note,
    ]
    row = "| " + " | ".join(cell(v) for v in values) + " |"

    ledger = pathlib.Path(args.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if not ledger.exists():
        header = (
            "# SDAR-on-GCP e2e — coworker run ledger\n\n"
            "Append-only log of every SDAR/GCP run (auto-written by `scripts/sdar_runlog.py`).\n\n"
            "| " + " | ".join(COLUMNS) + " |\n"
            "|" + "---|" * len(COLUMNS) + "\n"
        )
        ledger.write_text(header)
    with ledger.open("a") as fh:
        fh.write(row + "\n")
    print(row)


if __name__ == "__main__":
    main()
