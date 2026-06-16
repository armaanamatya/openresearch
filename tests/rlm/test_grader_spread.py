"""F7 — surface grader-noise spread under median-of-N (2026-06-16).

When OPENRESEARCH_GRADER_SAMPLES > 1 (A1 median-of-N), score_reproduction surfaces the
grader provenance + worst per-leaf noise spread so the report/UI can show the
denoising headroom. At N=1 (default) none of these keys appear → byte-for-byte.
"""

from __future__ import annotations

import json
import re

import pytest

from backend.evals.paperbench.leaf_scorer import score_reproduction

RUBRIC = {
    "id": "root", "requirements": "r", "weight": 1.0, "target_score": 0.7,
    "sub_tasks": [
        {"id": "a", "requirements": "leaf a", "weight": 1.0, "sub_tasks": []},
    ],
}


class _VaryingStub:
    """Returns a different score on each successive call (drives a real spread)."""

    def __init__(self, scores):
        self._it = iter(scores)

    def complete(self, *, system: str, user: str) -> str:
        s = next(self._it)
        ids = re.findall(r'"leaf_id":\s*"([^"]+)"', user)
        return json.dumps([{"leaf_id": i, "score": s, "justification": "x"} for i in ids])


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    for v in ("OPENRESEARCH_GRADER_SAMPLES", "OPENRESEARCH_GRADER_BACKEND", "OPENRESEARCH_EVIDENCE_GATE"):
        monkeypatch.delenv(v, raising=False)


def test_n1_no_spread_keys(tmp_path):
    result = score_reproduction(RUBRIC, tmp_path, _VaryingStub([0.6]), degraded=False)
    assert "grader_samples" not in result
    assert "grader_max_spread" not in result
    rec = result["leaf_scores"][0]
    assert "score_spread" not in rec


def test_median_of_n_surfaces_spread(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_GRADER_SAMPLES", "3")
    # 0.2, 0.6, 1.0 → median 0.6, spread 0.8
    result = score_reproduction(RUBRIC, tmp_path, _VaryingStub([0.2, 0.6, 1.0]), degraded=False)
    assert result["grader_samples"] == 3
    assert result["grader_temperature"] == 0.0
    assert result["grader_max_spread"] == pytest.approx(0.8)
    rec = result["leaf_scores"][0]
    assert rec["score"] == pytest.approx(0.6)  # median, not mean
    assert rec["score_spread"] == {"n": 3, "min": 0.2, "max": 1.0}
