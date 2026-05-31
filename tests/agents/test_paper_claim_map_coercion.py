"""2026-05-30: PaperClaimMap.claims must coerce non-string dict values.

claims is list[dict[str,str]]. LLM output (esp. gpt-5) emits None / lists /
numbers as values (e.g. dataset=null, dataset=["ALFWorld","WebShop"]), which
died with `claims.0.dataset: Input should be a valid string` and killed the run
at implement_baseline's input validation. We coerce instead of reject.
"""
from __future__ import annotations

from backend.agents.schemas import PaperClaimMap


def test_none_value_becomes_empty_string():
    m = PaperClaimMap(core_contribution="c", claims=[{"method": "GRPO", "dataset": None}])
    assert m.claims[0]["dataset"] == ""
    assert m.claims[0]["method"] == "GRPO"


def test_list_value_is_comma_joined():
    m = PaperClaimMap(
        core_contribution="c",
        claims=[{"dataset": ["ALFWorld", "WebShop", "Search-QA"]}],
    )
    assert m.claims[0]["dataset"] == "ALFWorld, WebShop, Search-QA"


def test_number_value_is_stringified():
    m = PaperClaimMap(core_contribution="c", claims=[{"metric": 0.95}])
    assert m.claims[0]["metric"] == "0.95"


def test_dict_value_is_json():
    m = PaperClaimMap(core_contribution="c", claims=[{"expected_result": {"acc": 1}}])
    assert m.claims[0]["expected_result"] == '{"acc": 1}'


def test_bare_string_item_still_coerced_to_claim_dict():
    # existing behavior preserved: a bare string item -> {"claim": item}
    m = PaperClaimMap(core_contribution="c", claims=["the model converges"])
    assert m.claims[0] == {"claim": "the model converges"}


def test_plain_string_values_pass_through():
    m = PaperClaimMap(
        core_contribution="c",
        claims=[{"method": "GRPO", "dataset": "ALFWorld", "expected_result": "85%"}],
    )
    assert m.claims[0] == {"method": "GRPO", "dataset": "ALFWorld", "expected_result": "85%"}
