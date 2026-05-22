"""Guard test: ReproductionContract accepts list-valued plan fields.

Symptom (catalogue debug-harden session, run 1 log): `plan_reproduction` built
`ReproductionContract(**data)` from LLM JSON whose `evaluation_plan` was a list
of steps — Pydantic raised `Input should be a valid string [input_type=list]`,
so the primitive fail-softed to an empty contract on every paper whose LLM
returned a step-list plan. The plan/definition fields now accept `str | list[str]`.
"""

from __future__ import annotations

from backend.agents.schemas import ReproductionContract


def test_reproduction_contract_accepts_list_plan_fields():
    """List-valued plan/definition fields must validate, not raise."""
    c = ReproductionContract(
        reproduction_definition=["criterion 1", "criterion 2"],
        smoke_test_plan=["step 1", "step 2"],
        full_run_plan=["run step 1"],
        dataset_plan=["download X", "preprocess Y"],
        evaluation_plan=["Compute C2ST scores for SNRE on benchmarks."],
    )
    assert c.evaluation_plan == ["Compute C2ST scores for SNRE on benchmarks."]
    assert c.smoke_test_plan == ["step 1", "step 2"]


def test_reproduction_contract_still_accepts_string_plan_fields():
    """String-valued plan fields must still validate (back-compat)."""
    c = ReproductionContract(
        reproduction_definition="a single-string definition",
        evaluation_plan="a single-string plan",
    )
    assert c.evaluation_plan == "a single-string plan"
    assert c.reproduction_definition == "a single-string definition"
