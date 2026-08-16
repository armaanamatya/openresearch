#!/usr/bin/env python3
"""Audit the single-VM GCP campaign path for stray billing.

READ-ONLY: this never stops or deletes a VM. It describes the configured
GCP instance(s) (same env vars / defaults as VmComputeProvider) and warns
loudly when one is RUNNING with no locally-tracked run to justify it, or
notes a STOPPED instance's ongoing (much smaller) persistent-disk cost.
Turns the manual "run `gcloud compute instances list` after every GCP run"
step from docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md into one
command an operator (or a cron job) can run any time.

Usage:
    python scripts/gcp_vm_audit.py
    python scripts/gcp_vm_audit.py --runs-root /elsewhere/runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/gcp_vm_audit.py` from the repo root without installs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.runtime.gcp_vm_audit import audit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)

    findings = audit(runs_root=args.runs_root.resolve())
    exit_code = 0
    for finding in findings:
        prefix = {"warn": "[WARN]", "info": "[info]", "error": "[ERROR]"}[finding.level]
        print(f"{prefix} {finding.message}")
        if finding.level in ("warn", "error"):
            exit_code = 1
    if not findings:
        print("[info] No GCP VM audit targets configured.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
