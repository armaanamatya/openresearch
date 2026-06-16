"""Unit tests for scripts/calibrate_grader.py (Lane 0 grader-noise harness).

No real LLM: a stub ``complete()`` cycles slightly-varying canned batch responses
across the K draws, driven through the *real* ``score_reproduction`` so the
exercise covers the actual grading + roll-up path (and the scorer's own
``batch_error`` all-0.0 exception handler). σ math is asserted both against the
pure ``summarize_draws`` reducer and end-to-end through ``calibrate``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
from pathlib import Path

import pytest

# scripts/ is not an importable package on the default pythonpath (src/), so load
# the module by file path. This also mirrors how the integrator runs `python -m
# scripts.calibrate_grader` without coupling the test to that invocation.
_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "calibrate_grader.py"
_spec = importlib.util.spec_from_file_location("calibrate_grader", _MODULE_PATH)
assert _spec and _spec.loader
calibrate_grader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calibrate_grader)

from backend.evals.paperbench.leaf_scorer import score_reproduction  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: a tiny in-memory rubric tree (same weight shape as the scorer's own
# test tree) + a tmp run dir whose final_report carries non-empty baseline_metrics
# so the degraded auto-cap never fires (we also pass degraded=False explicitly).
# ---------------------------------------------------------------------------

TINY_TREE = {
    "id": "root",
    "requirements": "root",
    "weight": 1,
    "sub_tasks": [
        {
            "id": "branch-a",
            "requirements": "branch a",
            "weight": 3,
            "sub_tasks": [
                {"id": "leaf-a1", "requirements": "leaf a1", "weight": 1, "sub_tasks": []},
                {"id": "leaf-a2", "requirements": "leaf a2", "weight": 1, "sub_tasks": []},
            ],
        },
        {"id": "leaf-b", "requirements": "leaf b", "weight": 1, "sub_tasks": []},
    ],
}


def _roll_up_overall(a1: float, a2: float, b: float) -> float:
    """The exact weighted roll-up for TINY_TREE: ((a1+a2)/2*3 + b*1)/4."""
    branch_a = (a1 + a2) / 2.0
    return (branch_a * 3.0 + b * 1.0) / 4.0


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "prj_calibtest"
    d.mkdir()
    (d / "final_report.json").write_text(
        json.dumps({"reproduction_summary": "calib test", "baseline_metrics": {"accuracy": 0.5}}),
        encoding="utf-8",
    )
    return d


class CyclingStubClient:
    """Returns canned batch JSON, cycling through ``per_draw`` on each call.

    Each element of ``per_draw`` is a dict ``{leaf_id: score}`` (one draw's grade).
    A sentinel value of the string ``"RAISE"`` makes that call raise — the scorer
    catches it inside ``_grade_batch`` and defaults the whole batch to 0.0 (the
    ``batch_error`` outlier we must measure the spread of).
    """

    def __init__(self, per_draw: list[dict]):
        self._per_draw = per_draw
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        draw = self._per_draw[self.calls % len(self._per_draw)]
        self.calls += 1
        if draw == "RAISE":
            raise RuntimeError("simulated transient LLM/parse failure")
        return json.dumps(
            [
                {"leaf_id": lid, "score": score, "justification": f"stub {lid}={score}"}
                for lid, score in draw.items()
            ]
        )


# ---------------------------------------------------------------------------
# 1. Pure reducer math (no scorer, no I/O).
# ---------------------------------------------------------------------------


def test_summarize_draws_overall_sigma_is_sample_stdev():
    overall = [0.50, 0.55, 0.60]
    out = calibrate_grader.summarize_draws(overall, {})
    assert out["overall"]["n"] == 3
    assert out["overall"]["min"] == 0.50
    assert out["overall"]["max"] == 0.60
    assert math.isclose(out["overall"]["mean"], statistics.fmean(overall))
    # sample stdev (n-1), not population
    assert math.isclose(out["overall"]["stdev"], statistics.stdev(overall))
    assert not math.isclose(out["overall"]["stdev"], statistics.pstdev(overall))


def test_summarize_draws_single_point_sigma_zero():
    out = calibrate_grader.summarize_draws([0.7], {})
    assert out["overall"]["n"] == 1
    assert out["overall"]["stdev"] == 0.0
    assert out["overall"]["mean"] == 0.7


def test_summarize_draws_per_leaf_n_reflects_contributing_draws():
    # leaf graded in 3 draws, another in only 2 (missing/None in one draw).
    per_leaf = {"leaf-a1": [1.0, 0.9, 1.0], "leaf-a2": [0.0, 0.1]}
    out = calibrate_grader.summarize_draws([0.5, 0.5, 0.5], per_leaf)
    assert out["per_leaf"]["leaf-a1"]["n"] == 3
    assert out["per_leaf"]["leaf-a2"]["n"] == 2
    assert math.isclose(out["per_leaf"]["leaf-a1"]["stdev"], statistics.stdev([1.0, 0.9, 1.0]))
    assert math.isclose(out["per_leaf"]["leaf-a2"]["stdev"], statistics.stdev([0.0, 0.1]))


def test_extract_leaf_scores_drops_none_and_keeps_numeric():
    result = {
        "leaf_scores": [
            {"id": "leaf-a1", "score": 0.8, "justification": "ok"},
            {"id": "leaf-a2", "score": None, "state": "skipped_data_unavailable"},
            {"id": "leaf-b", "score": 0.3, "justification": "partial"},
            {"id": "", "score": 0.9},  # blank id dropped
        ]
    }
    extracted = calibrate_grader._extract_leaf_scores(result)
    assert extracted == {"leaf-a1": 0.8, "leaf-b": 0.3}


# ---------------------------------------------------------------------------
# 2. End-to-end through the REAL score_reproduction with a stub client.
# ---------------------------------------------------------------------------


def test_calibrate_end_to_end_sigma_matches_canned_draws(run_dir: Path):
    # Three draws with slightly varying leaf grades — exactly the ~2.5% wobble
    # the real grader exhibits.
    draws = [
        {"leaf-a1": 1.0, "leaf-a2": 0.0, "leaf-b": 0.50},
        {"leaf-a1": 0.9, "leaf-a2": 0.1, "leaf-b": 0.55},
        {"leaf-a1": 1.0, "leaf-a2": 0.0, "leaf-b": 0.45},
    ]
    stub = CyclingStubClient(draws)

    record = calibrate_grader.calibrate(
        TINY_TREE,
        run_dir,
        stub,
        k=3,
        label="unit-test",
        score_fn=score_reproduction,  # the real scorer, stub transport
    )

    # The stub returns one batch per draw (3 leaves <= batch_size) → 3 calls.
    assert stub.calls == 3
    assert record["run_id"] == "prj_calibtest"
    assert record["label"] == "unit-test"
    assert record["k"] == 3
    assert record["degraded"] is False
    assert record["leaf_count"] == 3

    # Expected overall scores per draw from the exact roll-up.
    expected_overall = [_roll_up_overall(d["leaf-a1"], d["leaf-a2"], d["leaf-b"]) for d in draws]
    got_overall = record["overall"]["scores"]
    assert got_overall == pytest.approx(expected_overall)
    assert record["overall"]["stdev"] == pytest.approx(statistics.stdev(expected_overall))
    assert record["overall"]["mean"] == pytest.approx(statistics.fmean(expected_overall))
    assert record["overall"]["min"] == pytest.approx(min(expected_overall))
    assert record["overall"]["max"] == pytest.approx(max(expected_overall))

    # Per-leaf σ matches the canned per-leaf draws.
    for lid in ("leaf-a1", "leaf-a2", "leaf-b"):
        leaf_draws = [d[lid] for d in draws]
        assert record["per_leaf"][lid]["n"] == 3
        assert record["per_leaf"][lid]["stdev"] == pytest.approx(statistics.stdev(leaf_draws))
        assert record["per_leaf"][lid]["scores"] == pytest.approx(leaf_draws)


def test_calibrate_captures_all_zero_batch_error_outlier(run_dir: Path):
    # Two clean draws + one that RAISES → the scorer zeroes that batch (all 0.0),
    # producing an overall of 0.0 for that draw. The harness must fold the outlier
    # into the spread (this is exactly what median-of-N later shrugs off).
    draws = [
        {"leaf-a1": 1.0, "leaf-a2": 0.0, "leaf-b": 0.50},  # overall = 0.5
        "RAISE",                                            # batch_error → all 0.0
        {"leaf-a1": 1.0, "leaf-a2": 0.0, "leaf-b": 0.50},  # overall = 0.5
    ]
    stub = CyclingStubClient(draws)

    record = calibrate_grader.calibrate(
        TINY_TREE, run_dir, stub, k=3, score_fn=score_reproduction
    )

    assert stub.calls == 3
    overall_scores = record["overall"]["scores"]
    assert len(overall_scores) == 3
    # exactly one draw is the 0.0 outlier; the other two are the clean 0.5.
    assert overall_scores.count(pytest.approx(0.0)) == 1
    assert overall_scores.count(pytest.approx(0.5)) == 2
    assert record["overall"]["min"] == pytest.approx(0.0)
    assert record["overall"]["max"] == pytest.approx(0.5)
    # The spread is real (non-zero σ) and equals the sample stdev of [0.5, 0.0, 0.5].
    assert record["overall"]["stdev"] == pytest.approx(statistics.stdev([0.5, 0.0, 0.5]))
    assert record["overall"]["stdev"] > 0.0

    # The leaves are 0.0 on the error draw too → each leaf saw all 3 draws,
    # one of which is 0.0 (for leaf-b: [0.5, 0.0, 0.5]).
    assert record["per_leaf"]["leaf-b"]["n"] == 3
    assert sorted(record["per_leaf"]["leaf-b"]["scores"]) == pytest.approx([0.0, 0.5, 0.5])
    assert record["per_leaf"]["leaf-b"]["stdev"] == pytest.approx(statistics.stdev([0.5, 0.0, 0.5]))


# ---------------------------------------------------------------------------
# 3. Ledger I/O: append, never clobber.
# ---------------------------------------------------------------------------


def test_append_record_creates_and_appends(tmp_path: Path):
    ledger = tmp_path / "grader_calibration.json"
    assert not ledger.exists()

    r1 = {"run_id": "run-1", "label": "a", "overall": {"stdev": 0.01}}
    out = calibrate_grader.append_record(ledger, r1)
    assert ledger.exists()
    assert out["schema_version"] == calibrate_grader.SCHEMA_VERSION
    assert len(out["records"]) == 1

    # second append must NOT clobber the first
    r2 = {"run_id": "run-2", "label": "b", "overall": {"stdev": 0.02}}
    out2 = calibrate_grader.append_record(ledger, r2)
    assert len(out2["records"]) == 2
    assert [r["run_id"] for r in out2["records"]] == ["run-1", "run-2"]

    # re-read from disk to confirm persistence
    on_disk = json.loads(ledger.read_text(encoding="utf-8"))
    assert [r["run_id"] for r in on_disk["records"]] == ["run-1", "run-2"]


def test_append_record_preserves_prior_records_on_existing_ledger(tmp_path: Path):
    ledger = tmp_path / "grader_calibration.json"
    ledger.write_text(
        json.dumps({"schema_version": 1, "records": [{"run_id": "old"}]}), encoding="utf-8"
    )
    out = calibrate_grader.append_record(ledger, {"run_id": "new"})
    assert [r["run_id"] for r in out["records"]] == ["old", "new"]


# ---------------------------------------------------------------------------
# 4. Rubric resolution + arg validation.
# ---------------------------------------------------------------------------


def test_resolve_rubric_prefers_rubric_tree(tmp_path: Path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "rubric_tree.json").write_text("{}", encoding="utf-8")
    (d / "generated_rubric.json").write_text("{}", encoding="utf-8")
    assert calibrate_grader.resolve_rubric_path(d).name == "rubric_tree.json"


def test_resolve_rubric_falls_back_to_generated(tmp_path: Path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "generated_rubric.json").write_text("{}", encoding="utf-8")
    assert calibrate_grader.resolve_rubric_path(d).name == "generated_rubric.json"


def test_resolve_rubric_explicit_wins(tmp_path: Path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "rubric_tree.json").write_text("{}", encoding="utf-8")
    explicit = tmp_path / "custom.json"
    explicit.write_text("{}", encoding="utf-8")
    assert calibrate_grader.resolve_rubric_path(d, explicit).name == "custom.json"


def test_resolve_rubric_missing_raises(tmp_path: Path):
    d = tmp_path / "run"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        calibrate_grader.resolve_rubric_path(d)


def test_calibrate_rejects_k_below_one(run_dir: Path):
    with pytest.raises(ValueError):
        calibrate_grader.calibrate(
            TINY_TREE, run_dir, CyclingStubClient([{}]), k=0, score_fn=score_reproduction
        )
