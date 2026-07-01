#!/usr/bin/env python3
"""Render dashboard_events.jsonl lines as readable one-liners.

Usage:
  python3 scripts/render_run_events.py [--tail N] [--all] [FILE]
  cat dashboard_events.jsonl | python3 scripts/render_run_events.py [--all]

Each line is rendered as:
  [{HH:MM:SS}|it{iteration}] {EVENT}: {detail}
"""

from __future__ import annotations  # `str | None` annotations on Python < 3.10 (VM system python3)

import json
import sys
import argparse
from datetime import datetime


# Low-signal events skipped by default
_SKIP_EVENTS = frozenset({"primitive_resource"})
_SKIP_EVENT_TYPES = frozenset({"primitive_resource"})


def parse_timestamp(ts: str) -> str:
    """Extract HH:MM:SS from an ISO-8601 timestamp string."""
    try:
        # Handle both offset-aware and naive forms
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(ts[:len(fmt) + 6], fmt)
                return dt.strftime("%H:%M:%S")
            except ValueError:
                continue
        # Fallback: grab the time portion directly
        if "T" in ts:
            part = ts.split("T", 1)[1]
            return part[:8]
    except Exception:
        pass
    return "??:??:??"


def render_event(obj: dict, include_all: bool) -> str | None:
    """Return a formatted one-liner for the event, or None to skip."""
    event = obj.get("event", "")

    # Default-skip low-signal events
    if not include_all:
        if event in _SKIP_EVENTS:
            return None
        if event == "dashboard_event":
            etype = obj.get("event_type", obj.get("payload", {}).get("event_type", ""))
            if etype in _SKIP_EVENT_TYPES:
                return None
        # Skip heartbeat-ok spam: iteration_heartbeat with no error/warning
        if event == "iteration_heartbeat":
            return None

    ts = parse_timestamp(obj.get("timestamp", ""))
    iteration = obj.get("iteration", "?")
    prefix = f"[{ts}|it{iteration}]"

    # Determine detail string
    primitive = obj.get("primitive", "")
    status = obj.get("status", "")
    note = obj.get("note", "")
    message = obj.get("message", "")

    # Extract warning code from top-level or payload
    code = obj.get("code", "") or obj.get("payload", {}).get("code", "") if isinstance(obj.get("payload"), dict) else obj.get("code", "")

    # run_warning: the diagnostic gold — always show
    if event == "run_warning":
        code_part = f" [{code}]" if code else ""
        return f"⚠  {prefix} {event.upper()}{code_part}: {message}"

    # primitive_call with error/failure status
    if event == "primitive_call" and status in ("error", "failed"):
        err = obj.get("error", obj.get("result_summary", ""))
        detail = f"{primitive} ({status})"
        if err:
            detail += f": {err}"
        return f"✗  {prefix} {event}: {detail}"

    # Normal primitive_call
    if event == "primitive_call":
        detail = primitive + (f" ({status})" if status else "")
        return f"   {prefix} {event}: {detail}"

    # sub_rlm_spawned
    if event == "sub_rlm_spawned":
        model = obj.get("model", obj.get("payload", {}).get("model", ""))
        detail = f"spawn {model}" if model else "spawn"
        return f"   {prefix} {event}: {detail}"

    # Remaining events: first present of: note | message | event_type
    event_type = obj.get("event_type", "")
    if isinstance(obj.get("payload"), dict):
        event_type = event_type or obj["payload"].get("event_type", "")
    detail = note or message or event_type or ""

    tag = "   "
    return f"{tag}{prefix} {event}: {detail}" if detail else f"{tag}{prefix} {event}"


def main():
    parser = argparse.ArgumentParser(description="Render dashboard_events.jsonl as readable one-liners.")
    parser.add_argument("file", nargs="?", help="Path to dashboard_events.jsonl (default: stdin)")
    parser.add_argument("--tail", type=int, default=0, metavar="N",
                        help="Show only the last N rendered lines (default: all)")
    parser.add_argument("--all", action="store_true",
                        help="Include low-signal heartbeat/resource events")
    args = parser.parse_args()

    if args.file:
        try:
            fh = open(args.file, "r", encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"render_run_events: cannot open {args.file}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        fh = sys.stdin

    lines_rendered = []
    try:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            rendered = render_event(obj, include_all=args.all)
            if rendered is not None:
                lines_rendered.append(rendered)
    finally:
        if args.file:
            fh.close()

    if args.tail and args.tail > 0:
        lines_rendered = lines_rendered[-args.tail:]

    for line in lines_rendered:
        print(line)


if __name__ == "__main__":
    main()
