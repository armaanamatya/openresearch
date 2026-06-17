"""A1: fresh N>=3 candidate capture -> archive snapshots -> K temp=0 re-grades ->
stability verdict. The capture/regrade loop calls the existing BES candidate path
and scripts/calibrate_grader.py and is operator-run; this module owns the verdict
logic and is unit-tested."""
from __future__ import annotations
import argparse
from typing import Any
from backend.agents.rlm.select_stability import stability_report


def summarize_regrades(regrades: list[dict[str, float]], *, repeatability_sigma: float) -> dict[str, Any]:
    """A SELECT signal is noise when the top-1 winner flips across re-grades AND the
    smallest top-2 margin is within the grader's measured repeatability."""
    rep = stability_report(regrades)
    noisy = rep["top1_flip_rate"] > 0.0 and rep["margin_min"] <= repeatability_sigma
    return {"report": rep, "verdict": "select_is_noise" if noisy else "select_stable"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A1 SELECT-stability capture + verdict")
    parser.add_argument("--paper", required=True, help="arXiv id / paper to capture candidates on")
    parser.add_argument("--candidates", type=int, default=3, help="N>=3 candidates to capture")
    parser.add_argument("--k", type=int, default=10, help="re-grades per candidate (temp=0)")
    parser.add_argument("--sigma", type=float, default=0.02, help="grader repeatability sigma")
    args = parser.parse_args(argv)
    # NOTE (operator-gated, not unit-tested): the live loop must
    #   (1) run args.candidates candidates on args.paper via the BES candidate path
    #       (OPENRESEARCH_BES_CANDIDATES_PER_CLUSTER>=3), zero GPU,
    #   (2) archive each candidates/rlm_impl_*/ snapshot (archival-completeness),
    #   (3) re-grade each K times at temperature=0 via scripts/calibrate_grader.py,
    #   (4) assemble `regrades: list[dict[candidate_id, score]]` and print
    #       summarize_regrades(regrades, repeatability_sigma=args.sigma).
    raise SystemExit(
        "bes_a1_capture: live capture is operator-run; wire the candidate/regrade "
        "loop per the NOTE in main(). summarize_regrades() is the unit-tested core."
    )


if __name__ == "__main__":
    main()
