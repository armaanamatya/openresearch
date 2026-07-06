"""Tests for backend.agents.rlm.literature_claim_gate — the advisory
claim-grounding gate on the RUBRIC INPUT (WS② of the 2026-07-05
openscience-skill-library design), and its flag-gated wiring into
rubric_gen.generate_rubric_tree.

OFF: the gate returns [] without touching any injected connector, and wiring
it into generate_rubric_tree changes neither the returned rubric tree nor
triggers emit_warning.

ON: given a paper with two claimed numbers, a claim whose textual context
fuzzy-title-matches an injected connector's search result is NOT flagged; a
claim with no matching result IS flagged as an advisory LiteratureFinding,
and emit_warning is called once per finding.
"""

from __future__ import annotations

import json

from backend.agents.rlm.literature_claim_gate import (
    LiteratureFinding,
    gather_literature_findings,
    literature_claim_gate_enabled,
    run_literature_claim_gate,
)
from backend.agents.rlm.rubric_gen import generate_rubric_tree
from backend.services.knowledge.connectors import Connector, LiteratureRecord

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# extract_result_claims(_PAPER_TEXT) yields exactly two Claims:
#   Claim(term="accuracy", value=65.0, context="reports accuracy of 65.0 on the benchmar")
#   Claim(term="success_rate", value=91.2, context="s a success_rate of 91.2 on a novel unst")
# (verified directly against backend.agents.rlm.claim_grounding.extract_result_claims;
# the trailing filler paragraph exists ONLY to clear generate_rubric_tree's
# 500-char minimum-length guard — it is appended AFTER both claims and
# contains no digits or result-term vocabulary, so it cannot shift the two
# claims above or introduce a third).
_PAPER_TEXT = (
    "We compare our approach against the GRPO baseline. "
    "GRPO reports accuracy of 65.0 on the benchmark. "
    "Our method achieves a success_rate of 91.2 on a novel unstudied task."
    " This work builds on prior studies of agentic training pipelines, environment "
    "design, and reward shaping for language model policies. We discuss related "
    "architectural choices, ablation considerations, and implementation details "
    "relevant to reproducibility. The appendix provides additional discussion of "
    "hyperparameter selection and dataset curation. Future work will explore "
    "additional environments and extended training budgets across a range of "
    "compute-constrained settings."
)

# A title sharing enough tokens with the "accuracy" claim's context
# ("reports"/"accuracy"/"benchmar[k]") to clear the overlap threshold.
_CORROBORATING_TITLE = (
    "GRPO: A Group Relative Policy Optimization Approach with Strong "
    "Accuracy on Reports Benchmark"
)
# A title sharing nothing with the "success_rate" claim's context.
_UNRELATED_TITLE = "Advances in Molecular Dynamics Simulation for Battery Materials"


class _FakeConnector(Connector):
    """A Connector whose search() returns a canned response keyed by a
    substring of the query — no network, no injected transport needed since
    search()/fetch() are overridden directly."""

    source = "fake"

    def __init__(self, responses_by_query_substring: dict[str, list[LiteratureRecord]]) -> None:
        super().__init__()
        self._responses = responses_by_query_substring
        self.calls: list[str] = []

    def search(self, query: str, *, limit: int = 10) -> list[LiteratureRecord]:
        self.calls.append(query)
        for needle, records in self._responses.items():
            if needle in query:
                return records
        return []

    def fetch(self, record_id: str):  # pragma: no cover - unused by the gate
        return None


class _BoomConnector(Connector):
    """Raises if ever called — proves the OFF path never touches a connector."""

    source = "boom"

    def search(self, query: str, *, limit: int = 10) -> list[LiteratureRecord]:
        raise AssertionError("connector.search must not be called when the gate is OFF")

    def fetch(self, record_id: str):
        raise AssertionError("connector.fetch must not be called when the gate is OFF")


def _make_fake_connector() -> _FakeConnector:
    return _FakeConnector({
        "accuracy": [LiteratureRecord(source="fake", id="1", title=_CORROBORATING_TITLE)],
        "success_rate": [LiteratureRecord(source="fake", id="2", title=_UNRELATED_TITLE)],
    })


class _Recorder:
    """A simple emit_warning(code, message) recorder."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, code: str, message: str) -> None:
        self.calls.append((code, message))


class _FixedClient:
    """Returns the same canned rubric JSON on every call (mirrors test_rubric_gen.py)."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    def complete(self, *, system: str, user: str) -> str:
        self.call_count += 1
        return self.response


_VALID_RUBRIC_RESPONSE = json.dumps({
    "categories": [
        {
            "name": "Method fidelity",
            "weight": 1.0,
            "leaves": [
                {"requirements": "Implements the GRPO baseline as described in Section 3.", "weight": 1.0},
            ],
        },
    ]
})


# ---------------------------------------------------------------------------
# OFF — flag unset
# ---------------------------------------------------------------------------


def test_flag_off_by_default(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_CLAIM_GATE", raising=False)
    assert not literature_claim_gate_enabled()


def test_gate_returns_empty_and_never_touches_connectors_when_off(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_CLAIM_GATE", raising=False)
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_GROUNDING", "1")  # even if this were on...
    recorder = _Recorder()
    boom = _BoomConnector()

    findings = run_literature_claim_gate(
        _PAPER_TEXT, emit_warning=recorder, connectors=[boom],
    )

    assert findings == []
    assert recorder.calls == []


def test_gather_literature_findings_empty_when_literature_grounding_off(monkeypatch):
    # OPENRESEARCH_LITERATURE_CLAIM_GATE is a separate flag from
    # OPENRESEARCH_LITERATURE_GROUNDING; gather_literature_findings only
    # checks the latter (the connectors' own call-site gate) directly.
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_GROUNDING", raising=False)
    boom = _BoomConnector()

    findings = gather_literature_findings(_PAPER_TEXT, connectors=[boom])

    assert findings == []


def test_rubric_gen_unchanged_when_flags_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_CLAIM_GATE", raising=False)
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_GROUNDING", raising=False)
    recorder = _Recorder()
    client = _FixedClient(_VALID_RUBRIC_RESPONSE)

    tree = generate_rubric_tree(
        _PAPER_TEXT, client, paper_title="Test Paper",
        project_dir=tmp_path, emit_warning=recorder,
    )

    assert tree is not None
    leaves = tree["sub_tasks"][0]["sub_tasks"]
    assert len(leaves) == 1
    assert leaves[0]["requirements"] == "Implements the GRPO baseline as described in Section 3."
    assert recorder.calls == []  # the hook never ran


def test_rubric_gen_identical_with_or_without_new_kwargs_when_off(monkeypatch, tmp_path):
    """Supplying project_dir/emit_warning (the new optional kwargs) produces a
    structurally identical tree to the old call signature when the flag is off."""
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_CLAIM_GATE", raising=False)

    tree_old_signature = generate_rubric_tree(_PAPER_TEXT, _FixedClient(_VALID_RUBRIC_RESPONSE), paper_title="P")
    tree_new_signature = generate_rubric_tree(
        _PAPER_TEXT, _FixedClient(_VALID_RUBRIC_RESPONSE), paper_title="P",
        project_dir=tmp_path, emit_warning=_Recorder(),
    )

    def _strip_ids(node):
        return {
            "requirements": node["requirements"],
            "weight": node["weight"],
            "sub_tasks": [_strip_ids(c) for c in node.get("sub_tasks", [])],
        }

    assert _strip_ids(tree_old_signature) == _strip_ids(tree_new_signature)


# ---------------------------------------------------------------------------
# ON — flag set
# ---------------------------------------------------------------------------


def test_gate_flags_uncorroborated_claim_but_not_corroborated_one(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_CLAIM_GATE", "1")
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_GROUNDING", "1")
    fake = _make_fake_connector()
    recorder = _Recorder()

    findings = run_literature_claim_gate(_PAPER_TEXT, emit_warning=recorder, connectors=[fake])

    assert len(findings) == 1
    finding = findings[0]
    assert isinstance(finding, LiteratureFinding)
    assert finding.term == "success_rate"
    assert finding.value == 91.2
    assert finding.checked_sources == ("fake",)

    # The corroborated "accuracy" claim never appears in findings.
    assert all(f.term != "accuracy" for f in findings)

    # Exactly one advisory emitted, tagged with the expected code.
    assert len(recorder.calls) == 1
    code, message = recorder.calls[0]
    assert code == "literature_claim_ungrounded"
    assert "success_rate" in message

    # Both claims were actually queried against the connector.
    assert len(fake.calls) == 2


def test_gate_off_when_only_claim_gate_flag_set_and_grounding_off(monkeypatch):
    """OPENRESEARCH_LITERATURE_CLAIM_GATE alone does not authorize network use —
    OPENRESEARCH_LITERATURE_GROUNDING (the connectors' own call-site gate) must
    also be on."""
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_CLAIM_GATE", "1")
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_GROUNDING", raising=False)
    boom = _BoomConnector()

    findings = run_literature_claim_gate(_PAPER_TEXT, connectors=[boom])

    assert findings == []


def test_rubric_gen_still_returns_llm_tree_unmodified_when_on(monkeypatch, tmp_path):
    """Flag ON, wired through generate_rubric_tree with the REAL default
    connectors (no injection) — pytest-socket blocks the network, so this
    exercises the full fail-soft chain end-to-end and proves the returned
    tree is the LLM's rubric, untouched, no matter what the gate concludes.

    A blocked network means every connector search returns [] — with no
    records to corroborate against, both claims are (correctly, conservatively)
    treated as unable to be corroborated: "network unreachable" must never
    silently suppress the advisory the way it would if the code assumed
    corroboration by default. What this test pins is: no exception escapes,
    and the RUBRIC TREE itself is completely unaffected either way.
    """
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_CLAIM_GATE", "1")
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_GROUNDING", "1")
    recorder = _Recorder()
    client = _FixedClient(_VALID_RUBRIC_RESPONSE)

    tree = generate_rubric_tree(
        _PAPER_TEXT, client, paper_title="Test Paper",
        project_dir=tmp_path, emit_warning=recorder,
    )

    assert tree is not None
    leaves = tree["sub_tasks"][0]["sub_tasks"]
    assert len(leaves) == 1
    assert leaves[0]["requirements"] == "Implements the GRPO baseline as described in Section 3."
    # Blocked network -> every connector returns [] -> neither claim can be
    # corroborated -> both are conservatively flagged (never silently hidden).
    assert len(recorder.calls) == 2
    assert all(code == "literature_claim_ungrounded" for code, _ in recorder.calls)


def test_max_claims_bounds_network_calls(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_GROUNDING", "1")
    fake = _make_fake_connector()

    findings = gather_literature_findings(_PAPER_TEXT, connectors=[fake], max_claims=1)

    # Only the first (deduped) claim is checked.
    assert len(fake.calls) == 1
    assert len(findings) <= 1


def test_connector_response_is_cached_across_calls_for_the_same_project_dir(monkeypatch, tmp_path):
    """Reuses primitive_cache.make_key (see connectors/base.py::cache_get/put) to
    persist a connector response to rlm_state/literature_cache.jsonl, keyed by
    project_dir — a second identical call must not hit the connector again."""
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_GROUNDING", "1")
    fake = _make_fake_connector()

    first = gather_literature_findings(_PAPER_TEXT, connectors=[fake], project_dir=tmp_path)
    calls_after_first = len(fake.calls)
    assert calls_after_first == 2  # one search per (deduped) claim

    second = gather_literature_findings(_PAPER_TEXT, connectors=[fake], project_dir=tmp_path)

    # No new connector calls on the second pass — the cache served both queries.
    assert len(fake.calls) == calls_after_first
    assert [f.term for f in second] == [f.term for f in first]

    cache_path = tmp_path / "rlm_state" / "literature_cache.jsonl"
    assert cache_path.exists()


def test_cache_is_scoped_per_project_dir(monkeypatch, tmp_path):
    """A different project_dir (no prior cache) still hits the connector —
    proves the cache is keyed by project_dir, not process-global."""
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_GROUNDING", "1")
    fake = _make_fake_connector()
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    gather_literature_findings(_PAPER_TEXT, connectors=[fake], project_dir=tmp_path)
    calls_after_first_project = len(fake.calls)

    gather_literature_findings(_PAPER_TEXT, connectors=[fake], project_dir=other_dir)

    assert len(fake.calls) == calls_after_first_project + 2
