"""Calibration data derived from preserved runs.

Reads `.preserved` + `cost_ledger.jsonl` from every preserved run under
`runs_root/` and computes per-primitive average token counts.  The result
is persisted to `data/calibration.json` (relative to the repo root, adjacent
to `backend/`).

Hooked into `write_final_report_rlm` post-marker write so each successful run
tightens the next estimate.

Spec: docs/superpowers/specs/2026-05-25-budget-estimation-design.md §calibration.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CALIBRATION_SCHEMA_VERSION: int = 2

_MIN_RUNS_FOR_CATEGORY: int = 5

# Fallback per-primitive priors when calibration has fewer than _MIN_RUNS_FOR_CATEGORY.
# Estimated from typical RLM runs on NLP and vision papers.
_DEFAULT_PRIORS: dict[str, dict[str, float]] = {
    "understand_section": {"avg_input_tokens": 12_000, "avg_output_tokens": 800},
    "extract_hyperparameters": {"avg_input_tokens": 8_000, "avg_output_tokens": 400},
    "detect_environment": {"avg_input_tokens": 6_000, "avg_output_tokens": 600},
    "build_environment": {"avg_input_tokens": 4_000, "avg_output_tokens": 300},
    "plan_reproduction": {"avg_input_tokens": 15_000, "avg_output_tokens": 1_200},
    "implement_baseline": {"avg_input_tokens": 30_000, "avg_output_tokens": 8_000},
    "run_experiment": {"avg_input_tokens": 5_000, "avg_output_tokens": 200},
    "verify_against_rubric": {"avg_input_tokens": 10_000, "avg_output_tokens": 600},
    "propose_improvements": {"avg_input_tokens": 8_000, "avg_output_tokens": 1_000},
}

_DEFAULT_PRIMITIVE_CALL_COUNTS: dict[str, dict[str, int]] = {
    "strict": {
        "understand_section": 4,
        "extract_hyperparameters": 2,
        "detect_environment": 1,
        "build_environment": 1,
        "plan_reproduction": 1,
        "implement_baseline": 2,
        "run_experiment": 3,
        "verify_against_rubric": 2,
        "propose_improvements": 2,
    },
    "compressed": {
        "understand_section": 2,
        "extract_hyperparameters": 1,
        "detect_environment": 1,
        "build_environment": 1,
        "plan_reproduction": 1,
        "implement_baseline": 1,
        "run_experiment": 2,
        "verify_against_rubric": 1,
        "propose_improvements": 1,
    },
}


def _calibration_path() -> Path:
    """Canonical path for calibration.json — `data/` adjacent to `backend/`."""
    return Path(__file__).resolve().parents[3] / "data" / "calibration.json"


def recompute_calibration(runs_root: Path) -> dict:
    """Aggregate preserved runs into per-primitive token averages.

    Reads runs/<id>/.preserved + runs/<id>/cost_ledger.jsonl for every
    preserved run.  Persists result to data/calibration.json atomically.
    Returns the computed calibration dict.
    """
    runs_root = Path(runs_root)
    per_primitive: dict[str, list[tuple[int, int]]] = {}  # name → [(input, output), ...]
    n_preserved = 0

    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        marker = run_dir / ".preserved"
        if not marker.exists():
            continue
        ledger = run_dir / "cost_ledger.jsonl"
        if not ledger.exists():
            continue
        n_preserved += 1
        try:
            for line in ledger.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                primitive = row.get("primitive") or row.get("primitive_name")
                input_tok = row.get("input_tokens") or row.get("prompt_tokens") or 0
                output_tok = row.get("output_tokens") or row.get("completion_tokens") or 0
                # PR-ε.7: skip rows where both token counts are zero.  OAuth
                # runs do not return token usage from the SDK; those rows have
                # input_tokens=0 / output_tokens=0 and would poison the
                # calibration averages (driving them toward zero, which in turn
                # makes API-cost estimates trend to ~$0).  A row with zero
                # tokens is "missing data", not "zero-cost data".
                if int(input_tok) == 0 and int(output_tok) == 0:
                    continue
                if primitive and isinstance(input_tok, (int, float)) and isinstance(output_tok, (int, float)):
                    per_primitive.setdefault(primitive, []).append(
                        (int(input_tok), int(output_tok))
                    )
        except OSError:
            continue

    averages: dict[str, dict[str, float]] = {}
    for primitive, samples in per_primitive.items():
        if samples:
            avg_in = sum(s[0] for s in samples) / len(samples)
            avg_out = sum(s[1] for s in samples) / len(samples)
            averages[primitive] = {
                "avg_input_tokens": avg_in,
                "avg_output_tokens": avg_out,
                "n_samples": len(samples),
            }

    calibration = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "based_on_n_preserved_runs": n_preserved,
        "per_primitive": averages,
    }

    cal_path = _calibration_path()
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cal_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    os.replace(tmp, cal_path)
    logger.info(
        "calibration: recomputed from %d preserved runs, %d primitives tracked",
        n_preserved,
        len(averages),
    )
    return calibration


def get_primitive_priors(
    paper_category: str,
    recipe_mode: str,
) -> dict[str, dict[str, float]]:
    """Return per-primitive token averages from calibration.json, or static defaults.

    Falls back to static defaults when:
    - calibration.json does not exist yet
    - fewer than _MIN_RUNS_FOR_CATEGORY preserved runs
    - a specific primitive has no calibration data
    """
    cal_path = _calibration_path()
    cal_data: dict = {}
    if cal_path.exists():
        try:
            cal_data = json.loads(cal_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — corrupt calibration → defaults
            logger.warning("calibration: could not read %s, using defaults", cal_path)

    n_runs = cal_data.get("based_on_n_preserved_runs", 0)
    per_primitive_cal = cal_data.get("per_primitive", {})

    result: dict[str, dict[str, float]] = {}
    for primitive, defaults in _DEFAULT_PRIORS.items():
        if n_runs >= _MIN_RUNS_FOR_CATEGORY and primitive in per_primitive_cal:
            entry = per_primitive_cal[primitive]
            # PR-ε.7: treat zero-valued calibration entries as missing data.
            # Legacy calibration.json files (schema_version=1) may contain
            # averages that were computed from OAuth runs with zero-token rows.
            # `entry.get(k) or default` returns the stored value when it is a
            # truthy non-zero float, and falls back to the static default when
            # the stored value is 0 / 0.0 / None — exactly the right semantics.
            result[primitive] = {
                "avg_input_tokens": float(
                    entry.get("avg_input_tokens") or defaults["avg_input_tokens"]
                ),
                "avg_output_tokens": float(
                    entry.get("avg_output_tokens") or defaults["avg_output_tokens"]
                ),
            }
        else:
            result[primitive] = dict(defaults)

    return result
