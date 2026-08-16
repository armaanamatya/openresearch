"""T5 grader hardening — in-run cross-provider fallback, never silent-zero.

Spec: docs/history/specs/2026-07-09-eval-integrity-track-a-design.md §4.5.

Today ``leaf_scorer._grade_batch`` catches any parse/SDK error from the
grader transport and silently defaults EVERY leaf in the batch to ``0.0``.
This is wrong twice over: (1) a single transport hiccup (rate limit, provider
outage) permanently poisons the score even though another provider could
have graded the batch fine, and (2) a leaf that genuinely could not be
graded gets counted as a measured *failure* (a phantom ``0.0``) in the
weighted roll-up instead of being excluded like a data-unavailable leaf.

Two changes under test, both socket-hermetic (every transport is a plain
Python fake — no network, no real SDK):

1. ``grader_transport.build_fallback_chain()`` — an ordered list of
   INDEPENDENT transports composed from the existing single-swap
   ``build_transport_client`` backends.
2. ``leaf_scorer._grade_batch`` (via ``score_reproduction``) retries down
   that chain before giving up; a leaf still ungradable after the whole
   chain is exhausted is marked ``ungraded`` (score ``None``, excluded from
   the roll-up) with a loud ``grader_unavailable`` entry in the returned
   ``run_warnings`` list — never a silent ``0.0``.
"""

from __future__ import annotations

import json

import pytest

import backend.agents.rlm.grader_transport as grader_transport
from backend.evals.paperbench.leaf_scorer import score_reproduction

# Same minimal 2-level tree rubric as tests/rlm/test_grader_median.py.
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


class _AlwaysRaisesClient:
    """A transport that always raises (simulates an SDK/provider outage)."""

    def __init__(self, message: str = "transport down") -> None:
        self._message = message
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        raise RuntimeError(self._message)


class _CannedClient:
    """A transport that always succeeds with a fixed canned response."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        return self._response


class _LeafAwareClient:
    """Raises for any batch whose prompt mentions ``fails_for``; otherwise
    returns a canned score for ``ok_leaf_id`` (a batch's prompt embeds the
    JSON task payload, so the leaf id it is grading is inspectable in
    ``user``)."""

    def __init__(self, fails_for: str, ok_leaf_id: str, ok_score: float) -> None:
        self._fails_for = fails_for
        self._ok_leaf_id = ok_leaf_id
        self._ok_score = ok_score
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        if self._fails_for in user:
            raise RuntimeError("transport down for this leaf's batch")
        return json.dumps(
            [{"leaf_id": self._ok_leaf_id, "score": self._ok_score, "justification": "ok"}]
        )


@pytest.fixture(autouse=True)
def _clear_grader_env(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_GRADER_SAMPLES", raising=False)
    monkeypatch.delenv("OPENRESEARCH_GRADER_BACKEND", raising=False)
    monkeypatch.delenv("OPENRESEARCH_GRADER_MODEL", raising=False)


# ---------------------------------------------------------------------------
# grader_transport.build_fallback_chain — composition of the existing
# single-swap builders; never raises; preserves order; skips unconstructable
# backends (identity-sentinel fail-soft, mirrors build_validator_client).
# ---------------------------------------------------------------------------


def test_build_fallback_chain_order_and_skip_on_missing_creds(monkeypatch):
    seen_backends: list[str] = []
    built = {"anthropic-foundry": object(), "oauth": object()}  # "grok" absent below

    def _fake_build_transport_client(*, backend, model, fallback_client, fallback_label, role_label):
        seen_backends.append(backend)
        client = built.get(backend)
        if client is None:
            return fallback_client, fallback_label  # simulate missing credentials
        return client, f"{role_label}:{backend}"

    monkeypatch.setattr(grader_transport, "build_transport_client", _fake_build_transport_client)

    chain = grader_transport.build_fallback_chain()

    # Fixed order: anthropic-foundry -> grok -> oauth. "grok" has no entry in
    # `built` so it must be silently skipped, never appended.
    assert seen_backends == ["anthropic-foundry", "grok", "oauth"]
    assert chain == [built["anthropic-foundry"], built["oauth"]]


def test_build_fallback_chain_survives_a_candidate_construction_exception(monkeypatch):
    ok_client = object()

    def _fake_build_transport_client(*, backend, model, fallback_client, fallback_label, role_label):
        if backend == "anthropic-foundry":
            raise RuntimeError("boom during construction")
        if backend == "grok":
            return fallback_client, fallback_label  # missing creds
        return ok_client, f"{role_label}:{backend}"  # oauth succeeds

    monkeypatch.setattr(grader_transport, "build_transport_client", _fake_build_transport_client)

    chain = grader_transport.build_fallback_chain()
    assert chain == [ok_client]


# ---------------------------------------------------------------------------
# _grade_batch (via score_reproduction) — primary failure recovers via the
# fallback chain; total exhaustion marks leaves ungraded (never 0.0) with a
# loud grader_unavailable run_warning; a partially-exhausted grid excludes
# only the ungradable leaf from the roll-up instead of dragging it to 0.
# ---------------------------------------------------------------------------


def test_primary_raises_falls_back_to_chain_and_grades(monkeypatch, tmp_path):
    fallback = _CannedClient(_resp(0.9, 0.8))
    monkeypatch.setattr(grader_transport, "build_fallback_chain", lambda: [fallback])

    primary = _AlwaysRaisesClient()
    score = score_reproduction(RUBRIC, tmp_path, primary, degraded=False)

    assert score["overall_score"] == pytest.approx(0.86)  # 0.9*0.6 + 0.8*0.4
    assert score["run_warnings"] == []
    assert score["graded"] == 2
    assert primary.calls == 1   # the primary WAS tried first...
    assert fallback.calls == 1  # ...then the fallback graded the batch
    for rec in score["leaf_scores"]:
        assert rec["score"] is not None
        assert rec.get("state") != "ungraded"


def test_whole_chain_exhausted_marks_leaves_ungraded_never_zero_with_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(
        grader_transport,
        "build_fallback_chain",
        lambda: [_AlwaysRaisesClient("fallback #1 down"), _AlwaysRaisesClient("fallback #2 down")],
    )

    primary = _AlwaysRaisesClient("primary down")
    score = score_reproduction(RUBRIC, tmp_path, primary, degraded=False)

    # Nothing was measurable -> the honest default, but with full transparency
    # (never a fabricated per-leaf 0.0 masquerading as a real failing grade).
    assert score["overall_score"] == 0.0
    assert score["graded"] == 0
    assert score["coverage_pct"] == 0.0

    seen_ids = set()
    for rec in score["leaf_scores"]:
        assert rec["score"] is None  # NEVER a phantom 0.0
        assert rec["state"] == "ungraded"
        seen_ids.add(rec["id"])
    assert seen_ids == {"code", "results"}

    assert len(score["run_warnings"]) == 1
    warning = score["run_warnings"][0]
    assert warning["code"] == "grader_unavailable"
    assert warning["level"] == "warn"
    assert set(warning["leaf_ids"]) == {"code", "results"}


def test_partial_batch_exhaustion_excludes_leaf_from_rollup_not_zero(monkeypatch, tmp_path):
    """Two separate batches (batch_size=1): "code" grades fine, "results"'
    entire chain fails. The excluded leaf must NOT drag overall_score toward
    0 -- it must be excluded from BOTH the numerator and denominator of the
    weighted roll-up, exactly like a data-unavailable leaf. If "results" were
    (today's bug) silently scored 0.0, overall_score would be
    0.9*0.6 + 0.0*0.4 = 0.54; excluded, it must equal 0.9 (the "code"-only
    weighted average).
    """
    monkeypatch.setattr(
        grader_transport, "build_fallback_chain", lambda: [_AlwaysRaisesClient("fallback down")]
    )

    client = _LeafAwareClient(fails_for="results", ok_leaf_id="code", ok_score=0.9)
    score = score_reproduction(RUBRIC, tmp_path, client, degraded=False, batch_size=1)

    assert score["overall_score"] == pytest.approx(0.9)
    assert score["graded"] == 1
    assert score["coverage_pct"] == pytest.approx(0.5)  # 1 of 2 eligible leaves graded

    by_id = {rec["id"]: rec for rec in score["leaf_scores"]}
    assert by_id["code"]["score"] == pytest.approx(0.9)
    assert by_id["results"]["score"] is None
    assert by_id["results"]["state"] == "ungraded"

    assert len(score["run_warnings"]) == 1
    assert score["run_warnings"][0]["leaf_ids"] == ["results"]
