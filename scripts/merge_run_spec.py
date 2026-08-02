#!/usr/bin/env python3
"""Merge the ablation baseline run-spec with one arm's feature flags.

Emits ``baseline_run_spec.json`` UNION ``arms.json[arms][<arm>]`` (arm flags win)
as a JSON object to stdout — the per-arm ``--run-spec`` the GCP launcher stages.

    python scripts/merge_run_spec.py bes > /tmp/arm_bes.json
    python scripts/merge_run_spec.py --list          # list arm names

Pure stdlib. Keys are validated against run_spec_contract when --validate is set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CFG = Path(__file__).resolve().parent.parent / "configs" / "ablation"


def _load() -> tuple[dict, dict]:
    base = json.loads((_CFG / "baseline_run_spec.json").read_text())
    arms = json.loads((_CFG / "arms.json").read_text())["arms"]
    return base, arms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", nargs="?", help="arm name (see --list)")
    ap.add_argument("--list", action="store_true", help="list arm names and exit")
    ap.add_argument("--validate", action="store_true", help="reject keys run_spec_contract won't apply")
    args = ap.parse_args()

    base, arms = _load()
    if args.list:
        print("\n".join(sorted(arms)))
        return 0
    if args.arm not in arms:
        print(f"unknown arm {args.arm!r}; choose from: {sorted(arms)}", file=sys.stderr)
        return 2

    merged = {**base, **arms[args.arm]}
    if args.validate:
        from backend.agents.rlm import run_spec_contract as rsc
        bad = [k for k in merged if not rsc.run_spec_key_applies(k)]
        if bad:
            print(f"rejected keys (would fail INIT at $0): {bad}", file=sys.stderr)
            return 3
    json.dump(merged, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
