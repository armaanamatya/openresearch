#!/usr/bin/env python3
"""Replay the evidence-integrity mechanisms over existing runs/ — free, offline audit.

Runs the three flag-gated evidence mechanisms (W1-M1 rubric integrity, the
eval-coverage floor, and W2 state-contracts) over real on-disk run artifacts and
reports what each would find. Also demonstrates each gate FIRING via a controlled
injection on a throwaway copy (never mutates a real run).

Usage:
    .venv/bin/python scripts/evidence_replay.py [runs_root]   # default: ./runs

Pure/offline: reads artifacts, makes no network calls, spends nothing.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.evals.paperbench.deterministic_leaf_checker import check_leaf  # noqa: E402
from backend.evals.paperbench.grading_input_integrity import (  # noqa: E402
    check_grading_input_integrity,
    rubric_fingerprint,
    write_rubric_pin,
)


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _report_verdict(run_dir: Path) -> tuple[str, str]:
    fr = _load(run_dir / "final_report.json")
    if not isinstance(fr, dict):
        return ("-", "-")
    verdict = str(fr.get("verdict") or fr.get("status") or "-")
    score = fr.get("overall_score")
    if score is None:
        score = (fr.get("rubric") or {}).get("overall_score") if isinstance(fr.get("rubric"), dict) else None
    if score is None:
        score = fr.get("rubric_score")
    return (verdict, f"{score:.3f}" if isinstance(score, (int, float)) else "-")


def _fingerprint_consistency(run_dir: Path) -> str:
    gen = _load(run_dir / "generated_rubric.json")
    graded = _load(run_dir / "rubric_tree.json")
    if gen is None and graded is None:
        return "no rubric files"
    if gen is None:
        return "graded-only (no generated_rubric to compare)"
    if graded is None:
        return "generated-only (never graded / no rubric_tree)"
    fg, fr = rubric_fingerprint(gen), rubric_fingerprint(graded)
    return "MATCH" if fg == fr else f"DIFFER (gen {fg[:8]} vs graded {fr[:8]})"


def _count_sidecars(run_dir: Path) -> int:
    code = run_dir / "code"
    return len(list(code.rglob("eval_provenance.json"))) if code.exists() else 0


def audit_runs(runs_root: Path) -> None:
    print("=" * 100)
    print("EVIDENCE-INTEGRITY REPLAY over real runs (offline, $0)")
    print("=" * 100)
    header = f"{'run':<26}{'verdict':<14}{'score':<8}{'rubric consistency (W1-M1)':<40}{'ep-sidecars':<12}"
    print(header)
    print("-" * 100)
    run_dirs = sorted(
        d for d in runs_root.glob("prj_*") if d.is_dir() and " " not in d.name
    )
    for d in run_dirs:
        verdict, score = _report_verdict(d)
        consistency = _fingerprint_consistency(d)
        if len(consistency) > 38:
            consistency = consistency[:37] + "…"
        sidecars = _count_sidecars(d)
        print(f"{d.name:<26}{verdict:<14}{score:<8}{consistency:<40}{sidecars:<12}")
    print("-" * 100)
    print(
        "Note: rubric 'DIFFER' can be legitimate (the spec-validator may drop leaves after "
        "generation)\n      OR tampering — the live W1-M1 pin disambiguates (new runs carry "
        "rlm_state/rubric_pin.json).\n      ep-sidecars=0 ⇒ eval-coverage floor + W2 "
        "state-contracts are inert (these runs predate record_eval)."
    )


def demo_gates_firing(sample_run: Path) -> None:
    print("\n" + "=" * 100)
    print(f"CONTROLLED DEMO — inject attacks into a THROWAWAY COPY of {sample_run.name}")
    print("=" * 100)
    rubric = _load(sample_run / "rubric_tree.json") or _load(sample_run / "generated_rubric.json")
    if not isinstance(rubric, dict):
        print("(sample run has no usable rubric — skipping demo)")
        return

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "run"
        tmp.mkdir()
        (tmp / "rubric_tree.json").write_text(json.dumps(rubric), encoding="utf-8")

        # --- W1-M1: pin the real rubric, then weaken it ---
        import os
        os.environ["OPENRESEARCH_GRADER_INTEGRITY"] = "1"
        write_rubric_pin(tmp, rubric)
        clean = check_grading_input_integrity(tmp, rubric=rubric)
        weakened = json.loads(json.dumps(rubric))
        weakened["_tamper"] = "target_score lowered"  # any content change
        if isinstance(weakened.get("target_score"), (int, float)):
            weakened["target_score"] = 0.0
        tampered = check_grading_input_integrity(tmp, rubric=weakened)
        os.environ.pop("OPENRESEARCH_GRADER_INTEGRITY", None)
        print(f"  W1-M1 rubric integrity:  honest rubric -> ok={clean['ok']} ({clean['reason']})")
        print(f"                           weakened rubric -> ok={tampered['ok']} ({tampered['reason']})  <-- CAUGHT")

        # --- eval-coverage floor: inject a tiny-n_eval sidecar ---
        os.environ["OPENRESEARCH_EVAL_PROVENANCE_GUARD"] = "1"
        from backend.agents.rlm.eval_provenance import eval_provenance_should_veto
        cell = tmp / "code" / "outputs" / "run1" / "cell0"
        cell.mkdir(parents=True)
        (cell / "metrics.json").write_text(json.dumps({"status": "ok", "accuracy": 0.5}), encoding="utf-8")
        (cell / "eval_provenance.json").write_text(
            json.dumps({"records": [{"id": "a", "outcome": 1.0}, {"id": "b", "outcome": 0.0}],
                        "metric_value": 0.5, "n_eval": 2, "held_out": True}),
            encoding="utf-8",
        )
        os.environ.pop("OPENRESEARCH_MIN_EVAL_N", None)
        no_floor = eval_provenance_should_veto(tmp / "code")
        os.environ["OPENRESEARCH_MIN_EVAL_N"] = "100"
        with_floor = eval_provenance_should_veto(tmp / "code")
        os.environ.pop("OPENRESEARCH_MIN_EVAL_N", None)
        os.environ.pop("OPENRESEARCH_EVAL_PROVENANCE_GUARD", None)
        print(f"  eval-coverage floor:     MIN_EVAL_N unset -> veto={no_floor[0]}")
        print(f"                           MIN_EVAL_N=100 (2 evals) -> veto={with_floor[0]}  <-- CAUGHT")

        # --- W2 state-contract: eval-coverage as a per-leaf predicate ---
        os.environ["OPENRESEARCH_STATE_CONTRACTS"] = "1"
        leaf = {"id": "L1", "check_kind": "deterministic:state_contract",
                "assertion": {"min_eval_n": 100}}
        sc = check_leaf(leaf, tmp)
        os.environ.pop("OPENRESEARCH_STATE_CONTRACTS", None)
        print(f"  W2 state-contract:       min_eval_n=100 (2 evals) -> score={sc['score']}  <-- CAUGHT")


def main() -> None:
    runs_root = Path(sys.argv[1]) if len(sys.argv) > 1 else _REPO / "runs"
    if not runs_root.is_dir():
        print(f"no runs dir at {runs_root}")
        return
    audit_runs(runs_root)
    # Demo on the first run that has a rubric.
    for d in sorted(runs_root.glob("prj_*")):
        if " " in d.name:
            continue
        if (d / "rubric_tree.json").exists() or (d / "generated_rubric.json").exists():
            demo_gates_firing(d)
            break


if __name__ == "__main__":
    main()
