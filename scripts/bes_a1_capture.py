"""A1: fresh N>=3 candidate capture -> archive snapshots -> K temp=0 re-grades ->
stability verdict. The capture/regrade loop calls the existing BES candidate path
and scripts/calibrate_grader.py and is operator-run; this module owns the verdict
logic and is unit-tested."""
from __future__ import annotations
import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agents.rlm.select_stability import stability_report

logger = logging.getLogger(__name__)


def summarize_regrades(regrades: list[dict[str, float]], *, repeatability_sigma: float) -> dict[str, Any]:
    """A SELECT signal is noise when the top-1 winner flips across re-grades AND the
    smallest top-2 margin is within the grader's measured repeatability."""
    rep = stability_report(regrades)
    noisy = rep["top1_flip_rate"] > 0.0 and rep["margin_min"] <= repeatability_sigma
    return {"report": rep, "verdict": "select_is_noise" if noisy else "select_stable"}


def _default_score_one(code_dir: Path, rubric_tree: dict) -> float:
    """Real leaf-scorer SELECT call mirroring BES's _static_grade.

    Uses the same score_reproduction path BES uses for candidate selection:
    degraded=False (code-only grade, no metrics.json required), temperature=0
    (the grader client always runs at temp=0 by default). Returns overall_score
    as a float, or 0.0 on any error.
    """
    from backend.evals.paperbench.leaf_scorer import score_reproduction
    from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient

    llm_client = ClaudeLlmClient(model=None)  # default_oauth_model() → Sonnet
    run_dir = code_dir.parent  # candidate dir: candidates/rlm_impl_N/
    scored = score_reproduction(
        rubric_tree=rubric_tree,
        run_dir=run_dir,
        llm_client=llm_client,
        rubric_source="generated",
        degraded=False,
    )
    overall = scored.get("overall_score")
    try:
        return float(overall) if overall is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _load_rubric(run_dir: Path) -> dict:
    """Load the rubric tree from the run dir.

    Tries generated_rubric.json first (the BES path), then rubric_tree.json
    (the alternate name used by some code paths in leaf_scorer.py).
    """
    for name in ("generated_rubric.json", "rubric_tree.json"):
        p = run_dir / name
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"No rubric tree found in {run_dir} "
        "(expected generated_rubric.json or rubric_tree.json)"
    )


def regrade_candidates(
    run_dir: Path | str,
    k: int,
    *,
    score_one: Callable[[Path, dict], float] | None = None,
) -> list[dict[str, float]]:
    """Re-grade each candidate snapshot K times and return the per-round scores.

    Discovers ``candidates/rlm_impl_*/`` under ``run_dir``, loads the rubric tree,
    and for each of K rounds scores every candidate via ``score_one(code_dir,
    rubric_tree)``. Returns a list of K dicts ``{candidate_id: score}``.

    ``score_one`` defaults to the real leaf-scorer SELECT call (LLM, temp=0, no
    GPU). It is injectable so tests can pass a deterministic fake with no LLM.

    Fail-soft: a candidate that errors during scoring is skipped for that round
    with a warning — it will not appear in that round's dict. If every candidate
    fails in a round the round dict is still included (empty), so ``len(result)``
    is always K.
    """
    run_dir = Path(run_dir)
    if score_one is None:
        score_one = _default_score_one

    rubric_tree = _load_rubric(run_dir)

    candidates_root = run_dir / "candidates"
    cand_dirs = sorted(candidates_root.glob("rlm_impl_*/")) if candidates_root.is_dir() else []
    if not cand_dirs:
        warnings.warn(
            f"regrade_candidates: no candidates/rlm_impl_*/ dirs found under {run_dir}",
            stacklevel=2,
        )

    regrades: list[dict[str, float]] = []
    for round_idx in range(k):
        round_scores: dict[str, float] = {}
        for cand_dir in cand_dirs:
            cid = cand_dir.name  # e.g. rlm_impl_0
            code_dir = cand_dir / "code"
            try:
                score = score_one(code_dir, rubric_tree)
                round_scores[cid] = score
            except Exception as exc:  # noqa: BLE001 — fail-soft per candidate
                logger.warning(
                    "regrade_candidates: round %d candidate %s failed to score (%s: %s) — skipped",
                    round_idx, cid, type(exc).__name__, exc,
                )
        regrades.append(round_scores)
    return regrades


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A1 SELECT-stability re-grade verdict")
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to an existing BES run dir with candidates/rlm_impl_*/ snapshots",
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of re-grades per candidate at temperature=0 (default: 10)",
    )
    parser.add_argument(
        "--sigma", type=float, default=0.02,
        help="Grader repeatability sigma threshold (default: 0.02)",
    )
    args = parser.parse_args(argv)
    regrades = regrade_candidates(Path(args.run_dir), args.k)
    result = summarize_regrades(regrades, repeatability_sigma=args.sigma)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
