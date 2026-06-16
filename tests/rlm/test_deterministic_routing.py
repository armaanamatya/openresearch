"""A2: deterministic-by-construction leaf routing (2026-06-16).

REPROLAB_DETERMINISTIC_LEAVES default OFF → every eligible leaf goes to the LLM
(byte-for-byte today). On → leaves carrying a structured check_kind+assertion are
graded by the pure-Python deterministic_leaf_checker (no LLM); un-annotated
leaves still go to the LLM. The two routes merge before roll-up.
"""

from __future__ import annotations

import json

import pytest

from backend.evals.paperbench.leaf_scorer import score_reproduction

RUBRIC = {
    "id": "root", "requirements": "r", "weight": 1.0, "target_score": 0.7,
    "sub_tasks": [
        {
            "id": "hp", "requirements": "seed is 42", "weight": 0.5, "sub_tasks": [],
            "check_kind": "deterministic:hparam",
            "assertion": {"field": "seed", "op": "==", "value": 42},
        },
        {"id": "judgment", "requirements": "method faithful", "weight": 0.5, "sub_tasks": []},
    ],
}


class _Stub:
    def __init__(self, responses):
        self._r = list(responses)
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        r = self._r[self.calls % len(self._r)]
        self.calls += 1
        return r


def _resp(*pairs):
    return json.dumps([{"leaf_id": lid, "score": s, "justification": "x"} for lid, s in pairs])


def _write_prov(run_dir, obj):
    (run_dir / "code").mkdir(parents=True, exist_ok=True)
    (run_dir / "code" / "provenance.json").write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv("REPROLAB_DETERMINISTIC_LEAVES", raising=False)
    monkeypatch.delenv("REPROLAB_GRADER_SAMPLES", raising=False)
    monkeypatch.delenv("REPROLAB_GRADER_BACKEND", raising=False)


def test_off_all_leaves_go_to_llm(tmp_path):
    _write_prov(tmp_path, {"seed": 42})
    # flag off: hp + judgment BOTH LLM-graded — hp gets the LLM 0.6, not the
    # deterministic 1.0 it would earn from provenance.
    client = _Stub([_resp(("hp", 0.6), ("judgment", 0.8))])
    score = score_reproduction(RUBRIC, tmp_path, client, degraded=False)
    assert score["overall_score"] == pytest.approx(0.7)  # 0.6*0.5 + 0.8*0.5
    assert client.calls == 1


def test_on_routes_hparam_deterministically(monkeypatch, tmp_path):
    monkeypatch.setenv("REPROLAB_DETERMINISTIC_LEAVES", "1")
    _write_prov(tmp_path, {"seed": 42})
    # flag on: hp checked deterministically (seed==42 → 1.0); only judgment → LLM
    client = _Stub([_resp(("judgment", 0.8))])
    score = score_reproduction(RUBRIC, tmp_path, client, degraded=False)
    assert score["overall_score"] == pytest.approx(0.9)  # 1.0*0.5 + 0.8*0.5
    assert client.calls == 1  # only the judgment batch reached the LLM
    assert score["graded"] == 2  # deterministic + LLM both counted


def test_on_hparam_mismatch_is_a_real_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("REPROLAB_DETERMINISTIC_LEAVES", "1")
    _write_prov(tmp_path, {"seed": 7})  # provenance disagrees with the assertion
    client = _Stub([_resp(("judgment", 0.8))])
    score = score_reproduction(RUBRIC, tmp_path, client, degraded=False)
    # hp deterministic 0.0 (a well-formed failing check is a verdict, not a fall-through)
    assert score["overall_score"] == pytest.approx(0.4)  # 0.0*0.5 + 0.8*0.5
    assert client.calls == 1
