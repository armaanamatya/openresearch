"""Lock the paper_claims schema coercion contract.

The root model occasionally returns paper_claims as a LIST of claim dicts
(e.g. `[{"method": "RLM(GPT-5,…)", "expected_result": "62.0"}, …]`) instead
of the keyed dict the RLMFinalReport schema expects. The schema's
`_coerce_paper_claims` field validator turns lists into dicts (keyed by an
identity field, falling back to integer index) so a 30-minute run is not
rejected at the final-report step.

Tests pin the four shapes the validator must handle:
    dict passthrough · list → dict with `method` key · list → dict with
    fallback `claim_N` key · non-iterable garbage → `{}`.
"""

from __future__ import annotations

from backend.agents.rlm.report import RLMFinalReport


class TestPaperClaimsCoercion:
    def test_dict_passes_through_unchanged(self) -> None:
        r = RLMFinalReport(paper_claims={"baseline": {"value": 0.95}})
        assert r.paper_claims == {"baseline": {"value": 0.95}}

    def test_list_with_method_key_coerced_to_dict(self) -> None:
        r = RLMFinalReport(
            paper_claims=[
                {"method": "RLM(GPT-5)", "expected_result": "62.0"},
                {"method": "Baseline",   "expected_result": "45.0"},
            ]
        )
        assert r.paper_claims == {
            "RLM(GPT-5)": {"method": "RLM(GPT-5)", "expected_result": "62.0"},
            "Baseline":   {"method": "Baseline",   "expected_result": "45.0"},
        }

    def test_list_without_identity_key_falls_back_to_index(self) -> None:
        r = RLMFinalReport(
            paper_claims=[
                {"no_method": "x", "expected_result": "1.0"},
                {"no_method": "y", "expected_result": "2.0"},
            ]
        )
        assert r.paper_claims == {
            "claim_0": {"no_method": "x", "expected_result": "1.0"},
            "claim_1": {"no_method": "y", "expected_result": "2.0"},
        }

    def test_list_uses_first_available_identity_field(self) -> None:
        # method / claim / claim_id / id / name precedence — pick the first one set.
        r = RLMFinalReport(
            paper_claims=[
                {"claim": "Theorem 3.1", "expected_result": "true"},
                {"claim_id": "T-4-2",    "expected_result": "false"},
                {"id": "ablation-A",     "expected_result": "0.42"},
                {"name": "Section 5",    "expected_result": "75 ± 3"},
            ]
        )
        assert set(r.paper_claims.keys()) == {
            "Theorem 3.1", "T-4-2", "ablation-A", "Section 5",
        }

    def test_mixed_list_and_non_dict_entries_skips_garbage(self) -> None:
        r = RLMFinalReport(
            paper_claims=[
                {"method": "Real", "expected_result": "1.0"},
                "garbage",      # not a dict — skipped
                42,             # not a dict — skipped
                None,           # not a dict — skipped
                {"method": "Also real"},
            ]
        )
        assert r.paper_claims == {
            "Real":      {"method": "Real", "expected_result": "1.0"},
            "Also real": {"method": "Also real"},
        }

    def test_non_iterable_garbage_becomes_empty_dict(self) -> None:
        # Per `if not isinstance(v, list): return {}` — anything not dict/list is {}.
        r = RLMFinalReport(paper_claims="not a dict")  # type: ignore[arg-type]
        assert r.paper_claims == {}

    def test_default_is_empty_dict(self) -> None:
        r = RLMFinalReport()
        assert r.paper_claims == {}
