#!/usr/bin/env python3
"""Grader-noise calibration harness (Lane 0 of the 2026-06-16 grader-fidelity remediation).

Re-grade a *fixed* saved run directory K times through the post-run leaf scorer
(:func:`backend.evals.paperbench.leaf_scorer.score_reproduction`) and measure how
much the shipped score wobbles between draws. The leaf grade is a non-deterministic
LLM call (no temp/seed), so the same evidence graded twice produces slightly
different numbers; this tool quantifies that drift as a per-leaf and overall
*standard deviation* (σ).

It is a **pure measurement tool** — it never mutates the run dir or the run's
``final_report.json`` (unlike ``scripts/score_run.py``). It only appends a record
to ``data/grader_calibration.json``. The integrator uses it to prove the overall
grader σ before/after each grader-fidelity (workstream A) step:

    σ before any A-flag flips default-ON  →  measure here
    σ after A5+A1 (median-of-N)           →  re-measure, confirm it dropped
    promotion criterion                   →  overall σ ≤ ~0.02

Why a separate file from ``data/calibration.json``: that file is the *cost*
calibration (per-primitive token averages); this is *grader noise*. They are
unrelated artifacts and ``data/calibration.json`` carries uncommitted user state.

Usage
-----
    python -m scripts.calibrate_grader --run-dir runs/prj_xxx --rubric path/to/rubric_tree.json -k 5 [--label "pre-A5"]
    python -m scripts.calibrate_grader --run-dir runs/prj_xxx -k 5          # --rubric resolved from the run dir

Rubric resolution (when ``--rubric`` omitted), in order:
    1. <run_dir>/rubric_tree.json
    2. <run_dir>/generated_rubric.json

The grading client is built from env the same way ``scripts/score_run.py`` does
(Featherless Qwen root via ``resolve_root_model`` + ``OpenAILlmClient``) so no
extra credential is needed beyond what a normal scored run requires. The client
is *injectable* (the ``calibrate`` function takes an ``llm_client``) so the harness
is unit-testable without any real LLM — see ``tests/rlm/test_calibrate_grader.py``.

Standard-deviation convention
-----------------------------
Per-leaf and overall σ use the **sample** standard deviation (``statistics.stdev``,
the n-1 / Bessel-corrected estimator): the K draws are a finite sample from the
grader-noise distribution, and the sample stdev is its unbiased estimator. When
fewer than two numeric draws exist for a series, σ is reported as ``0.0`` (a single
point has no spread). A per-leaf σ is computed only over the draws in which that
leaf received a *numeric* score — ``None`` scores (data-unavailable / theory-only /
unscored leaves) are excluded, and the count of contributing draws is recorded
as ``n`` per leaf so a leaf graded in 3 of 5 draws is honestly distinguishable
from one graded in all 5.

The all-0.0 outlier (``batch_error``): ``score_reproduction`` catches an LLM/parse
failure *inside* ``_grade_batch`` and defaults that whole batch to 0.0 rather than
raising. So a draw whose LLM call blows up still returns a (low) overall_score and
leaf scores — and this harness folds it into the spread exactly as the live grader
would experience it, which is the point: median-of-N is what shrugs the outlier off.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

# Top-level so callers/tests can pin it without importing private state.
SCHEMA_VERSION = 1

# Default location of the calibration ledger (repo-relative). Deliberately NOT
# data/calibration.json (that's the cost ledger with uncommitted user changes).
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER_PATH = _REPO_ROOT / "data" / "grader_calibration.json"

# Filenames tried (in order) when --rubric is omitted.
_RUBRIC_CANDIDATES = ("rubric_tree.json", "generated_rubric.json")


class _LlmClient(Protocol):
    """The grader transport contract (mirror of leaf_scorer.LlmClient).

    We re-declare it here so the harness imports cleanly without dragging the
    scorer module in at import time (the real ``score_reproduction`` is imported
    lazily inside :func:`calibrate`). The stub in the unit test satisfies this.
    """

    def complete(self, *, system: str, user: str) -> str: ...


# ---------------------------------------------------------------------------
# Rubric / client resolution helpers
# ---------------------------------------------------------------------------


def resolve_rubric_path(run_dir: Path, explicit: Optional[Path] = None) -> Path:
    """Return the rubric_tree path: ``explicit`` if given, else first found in run_dir.

    Raises ``FileNotFoundError`` if nothing resolves so the CLI fails loudly rather
    than grading against an empty tree.
    """
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--rubric path does not exist: {p}")
        return p
    for name in _RUBRIC_CANDIDATES:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no rubric found in {run_dir} (tried {', '.join(_RUBRIC_CANDIDATES)}); "
        f"pass --rubric explicitly"
    )


def _build_real_llm_client() -> _LlmClient:
    """Build the grading client from env, identical to scripts/score_run.py.

    Featherless Qwen root via ``resolve_root_model`` + ``OpenAILlmClient`` — the
    same backend a normal scored run uses, so no extra credential is needed beyond
    ``FEATHERLESS_API_KEY``. Imported lazily so ``--help`` and the unit tests never
    touch network/credential code. We intentionally do NOT edit the client builder;
    we reuse the existing public constructors.
    """
    from backend.agents.rlm.models import resolve_root_model
    from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

    root = resolve_root_model("qwen3-coder-featherless")
    bk = root.backend_kwargs
    return OpenAILlmClient(
        model=bk["model_name"], api_key=bk["api_key"], base_url=bk["base_url"]
    )


# ---------------------------------------------------------------------------
# σ computation
# ---------------------------------------------------------------------------


def _stdev(values: list[float]) -> float:
    """Sample standard deviation (n-1); 0.0 for <2 points. See module docstring."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _extract_leaf_scores(result: dict[str, Any]) -> dict[str, float]:
    """Turn one ``score_reproduction`` result into ``{leaf_id: numeric_score}``.

    ``result["leaf_scores"]`` is a *list of records* (``{id, score, justification, ...}``),
    NOT a dict. Records whose ``score`` is ``None`` (data-unavailable / theory-only /
    unscored) are dropped — they carry no signal for noise measurement. A leaf that
    appears more than once keeps the last numeric value (matches the scorer's own
    ``leaf_scores`` dict semantics).
    """
    out: dict[str, float] = {}
    for rec in result.get("leaf_scores") or []:
        if not isinstance(rec, dict):
            continue
        lid = str(rec.get("id", ""))
        if not lid:
            continue
        score = rec.get("score")
        if score is None:
            continue
        try:
            out[lid] = float(score)
        except (TypeError, ValueError):
            continue
    return out


def summarize_draws(
    overall_scores: list[float],
    per_leaf_draws: dict[str, list[float]],
) -> dict[str, Any]:
    """Reduce the K draws into the calibration record's numeric body.

    Pure (no I/O, no clock) so it is trivially unit-testable. ``per_leaf_draws``
    maps leaf_id -> the list of numeric scores observed across draws (one entry
    per draw in which the leaf was numerically graded).
    """
    n = len(overall_scores)
    overall = {
        "n": n,
        "mean": (statistics.fmean(overall_scores) if overall_scores else 0.0),
        "stdev": _stdev(overall_scores),
        "min": (min(overall_scores) if overall_scores else 0.0),
        "max": (max(overall_scores) if overall_scores else 0.0),
        "scores": list(overall_scores),
    }
    per_leaf: dict[str, dict[str, Any]] = {}
    for lid in sorted(per_leaf_draws):
        draws = per_leaf_draws[lid]
        per_leaf[lid] = {
            "n": len(draws),
            "mean": (statistics.fmean(draws) if draws else 0.0),
            "stdev": _stdev(draws),
            "min": (min(draws) if draws else 0.0),
            "max": (max(draws) if draws else 0.0),
            "scores": list(draws),
        }
    return {"overall": overall, "per_leaf": per_leaf}


# ---------------------------------------------------------------------------
# Ledger I/O (append, never clobber)
# ---------------------------------------------------------------------------


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "records": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Corrupt/empty file — start fresh rather than crash, but never silently
        # drop prior records on a *parseable* file.
        return {"schema_version": SCHEMA_VERSION, "records": []}
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        return {"schema_version": SCHEMA_VERSION, "records": []}
    data.setdefault("schema_version", SCHEMA_VERSION)
    return data


def append_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append ``record`` to the ledger at ``path`` and persist. Returns the full ledger."""
    ledger = _load_ledger(path)
    ledger["records"].append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return ledger


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def calibrate(
    rubric_tree: dict[str, Any],
    run_dir: Path,
    llm_client: _LlmClient,
    *,
    k: int = 5,
    run_id: Optional[str] = None,
    label: Optional[str] = None,
    rubric_source: str = "generated",
    degraded: bool = False,
    score_fn: Optional[Callable[..., dict[str, Any]]] = None,
    grader_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Re-grade ``run_dir`` K times and return the calibration record (does NOT write it).

    Parameters
    ----------
    rubric_tree : the parsed rubric tree dict.
    run_dir : the saved run directory (read-only; never mutated).
    llm_client : injectable grader transport (real client or a unit-test stub).
    k : number of independent re-grades (draws). Default 5 per the validation gate.
    run_id : key for the record; defaults to ``run_dir.name``.
    label : free-text marker passed by the caller (e.g. "pre-A5"); recorded as-is.
        Accepted as an argument so library code never calls ``datetime.now()``.
    rubric_source : passed straight through to ``score_reproduction``.
    degraded : passed EXPLICITLY to ``score_reproduction`` (default ``False`` — we
        want to measure the live, non-degraded grading path; the integrator can set
        True to calibrate the degraded branch).
    score_fn : the scorer. Defaults to the real ``score_reproduction``; injectable
        only to keep the unit test fully self-contained (the test injects the real
        scorer with a stub client, which is the realistic seam).
    grader_kwargs : extra kwargs forwarded to ``score_fn`` (e.g. ``batch_size``,
        ``invariants``). ``run_dir``/``llm_client``/``rubric_source``/``degraded``
        are passed by this function and must not be duplicated here.

    Returns
    -------
    A record dict ready for :func:`append_record`.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if score_fn is None:
        from backend.evals.paperbench.leaf_scorer import score_reproduction as score_fn  # type: ignore

    extra = dict(grader_kwargs or {})

    overall_scores: list[float] = []
    per_leaf_draws: dict[str, list[float]] = {}
    leaf_counts: list[int] = []
    graded_counts: list[int] = []

    for _ in range(k):
        result = score_fn(
            rubric_tree,
            run_dir,
            llm_client,
            rubric_source=rubric_source,
            degraded=degraded,
            **extra,
        )
        try:
            overall_scores.append(float(result.get("overall_score", 0.0)))
        except (TypeError, ValueError):
            overall_scores.append(0.0)
        leaf_counts.append(int(result.get("leaf_count", 0) or 0))
        graded_counts.append(int(result.get("graded", 0) or 0))
        for lid, score in _extract_leaf_scores(result).items():
            per_leaf_draws.setdefault(lid, []).append(score)

    summary = summarize_draws(overall_scores, per_leaf_draws)

    record: dict[str, Any] = {
        "run_id": run_id or run_dir.name,
        "label": label,
        "k": k,
        "rubric_source": rubric_source,
        "degraded": degraded,
        "leaf_count": (max(leaf_counts) if leaf_counts else 0),
        "graded_counts": graded_counts,
        **summary,
    }
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.calibrate_grader",
        description="Re-grade a saved run K times and record per-leaf + overall grader σ.",
    )
    p.add_argument("--run-dir", required=True, help="runs/<id> directory to re-grade")
    p.add_argument(
        "--rubric",
        default=None,
        help="path to rubric_tree.json (resolved from --run-dir if omitted)",
    )
    p.add_argument("-k", type=int, default=5, help="number of re-grades (draws); default 5")
    p.add_argument(
        "--label",
        default=None,
        help="free-text marker recorded with the run (e.g. 'pre-A5'); avoids clock calls in library code",
    )
    p.add_argument(
        "--rubric-source",
        default=None,
        help="rubric_source passed to the scorer; inferred from the rubric filename if omitted",
    )
    p.add_argument(
        "--degraded",
        action="store_true",
        help="grade the degraded path (default: non-degraded — measures live grading)",
    )
    p.add_argument(
        "--ledger",
        default=str(DEFAULT_LEDGER_PATH),
        help=f"calibration ledger to append to (default {DEFAULT_LEDGER_PATH})",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    args = _build_arg_parser().parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"error: run dir does not exist: {run_dir}", file=sys.stderr)
        return 1

    try:
        rubric_path = resolve_rubric_path(
            run_dir, Path(args.rubric) if args.rubric else None
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rubric_tree = json.loads(rubric_path.read_text(encoding="utf-8"))

    # Infer rubric_source from the filename if not explicitly given:
    # generated_rubric.json => "generated"; otherwise the bundle default.
    rubric_source = args.rubric_source or (
        "generated" if rubric_path.name == "generated_rubric.json" else "paperbench_bundle"
    )

    llm_client = _build_real_llm_client()

    print(f"calibrating grader on {run_dir.name} with k={args.k} "
          f"(rubric={rubric_path.name}, source={rubric_source})")
    record = calibrate(
        rubric_tree,
        run_dir,
        llm_client,
        k=args.k,
        label=args.label,
        rubric_source=rubric_source,
        degraded=bool(args.degraded),
    )

    ledger_path = Path(args.ledger)
    append_record(ledger_path, record)

    overall = record["overall"]
    print(f"  overall σ : {overall['stdev']:.4f}  "
          f"(mean {overall['mean']:.4f}, min {overall['min']:.4f}, max {overall['max']:.4f}, n={overall['n']})")
    max_leaf_sigma = max(
        (lv["stdev"] for lv in record["per_leaf"].values()), default=0.0
    )
    print(f"  max leaf σ: {max_leaf_sigma:.4f} over {len(record['per_leaf'])} numerically-graded leaves")
    print(f"  appended to: {ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
