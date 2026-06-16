"""A1 median-of-N grader denoising + A5 transport passthrough (2026-06-16).

The leaf grade is a non-deterministic LLM call. A1 takes N samples per batch and
the per-leaf MEDIAN so transient noise (and the all-0.0 batch_error outlier) is
shrugged off. N=1 (OPENRESEARCH_GRADER_SAMPLES default) must reproduce today's
single-sample behavior byte-for-byte. A5 routes through grader_transport, which
falls back to N× complete() on a client without complete_samples.
"""

from __future__ import annotations

import json

import pytest

from backend.evals.paperbench.leaf_scorer import score_reproduction

# Same minimal 2-level tree rubric as test_verify_against_rubric.
RUBRIC = {
    "id": "root",
    "requirements": "reproduce the paper",
    "weight": 1.0,
    "source": "generated",
    "target_score": 0.7,
    "sub_tasks": [
        {"id": "code", "requirements": "code is implemented", "weight": 0.6, "sub_tasks": []},
        {"id": "results", "requirements": "results are reported", "weight": 0.4, "sub_tasks": []},
    ],
}


def _resp(code_score, results_score) -> str:
    return json.dumps([
        {"leaf_id": "code", "score": code_score, "justification": "c"},
        {"leaf_id": "results", "score": results_score, "justification": "r"},
    ])


class _CyclingClient:
    """Stub LlmClient cycling canned batch responses across successive complete()
    calls. Has NO complete_samples → grader_transport.sample_completions falls
    back to N× complete (the universal floor), consuming N responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        r = self._responses[self.calls % len(self._responses)]
        self.calls += 1
        return r


@pytest.fixture(autouse=True)
def _clear_grader_env(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_GRADER_SAMPLES", raising=False)
    monkeypatch.delenv("OPENRESEARCH_GRADER_BACKEND", raising=False)
    monkeypatch.delenv("OPENRESEARCH_GRADER_MODEL", raising=False)


def test_default_n1_is_single_sample(tmp_path):
    """OPENRESEARCH_GRADER_SAMPLES unset → exactly one grader call, today's score."""
    client = _CyclingClient([_resp(0.9, 0.8)])
    score = score_reproduction(RUBRIC, tmp_path, client, degraded=False)
    assert score["overall_score"] == pytest.approx(0.86)  # 0.9*0.6 + 0.8*0.4
    assert client.calls == 1


def test_n3_takes_per_leaf_median(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_GRADER_SAMPLES", "3")
    client = _CyclingClient([_resp(0.9, 0.8), _resp(0.7, 0.8), _resp(0.8, 0.8)])
    score = score_reproduction(RUBRIC, tmp_path, client, degraded=False)
    # per-leaf median: code median(0.9,0.7,0.8)=0.8; results median(0.8,0.8,0.8)=0.8
    # overall = 0.8*0.6 + 0.8*0.4 = 0.8 (distinct from the N=1 single draw 0.86)
    assert score["overall_score"] == pytest.approx(0.8)
    assert client.calls == 3


def test_n3_median_shrugs_off_zero_outlier(monkeypatch, tmp_path):
    """One unparseable draw (→ all-0.0) must not sink a clearly-good grade."""
    monkeypatch.setenv("OPENRESEARCH_GRADER_SAMPLES", "3")
    client = _CyclingClient([_resp(0.85, 0.85), "TOTALLY NOT JSON", _resp(0.85, 0.85)])
    score = score_reproduction(RUBRIC, tmp_path, client, degraded=False)
    # garbage sample → 0.0 for both leaves; median(0.85, 0.0, 0.85) = 0.85
    assert score["overall_score"] == pytest.approx(0.85)


def test_n3_even_sample_count_still_medians(monkeypatch, tmp_path):
    """Non-odd N is clamped-friendly: statistics.median averages the two middle."""
    monkeypatch.setenv("OPENRESEARCH_GRADER_SAMPLES", "2")
    client = _CyclingClient([_resp(0.9, 0.8), _resp(0.7, 0.8)])
    score = score_reproduction(RUBRIC, tmp_path, client, degraded=False)
    # code median(0.9,0.7)=0.8; results median(0.8,0.8)=0.8 → 0.8
    assert score["overall_score"] == pytest.approx(0.8)
    assert client.calls == 2


def test_invalid_samples_env_falls_back_to_one(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_GRADER_SAMPLES", "not-a-number")
    client = _CyclingClient([_resp(0.9, 0.8)])
    score = score_reproduction(RUBRIC, tmp_path, client, degraded=False)
    assert score["overall_score"] == pytest.approx(0.86)
    assert client.calls == 1
