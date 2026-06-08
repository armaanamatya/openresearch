"""ScopeSpec.skip_datasets — the verified operator-scope source for the env axis
(2026-06-01). Mirrors skip_models: explicit entries (and the datasets the operator
dropped by narrowing) are removed from the effective datasets and become the
input to the rubric's operator_scope exclusions.
"""
from __future__ import annotations

from backend.agents.schemas import DatasetSlice, ScopeSpec


def _paper_default() -> ScopeSpec:
    return ScopeSpec(
        models=["Qwen3-1.7B", "Qwen2.5-3B-Instruct", "Qwen2.5-7B"],
        datasets=[DatasetSlice(name="ALFWorld"), DatasetSlice(name="WebShop"), DatasetSlice(name="Search-QA")],
    )


def test_narrowing_datasets_derives_skip_datasets():
    # Mirrors the live smallest-two scope file (datasets=[Search-QA]).
    eff = ScopeSpec(
        models=["Qwen3-1.7B", "Qwen2.5-3B-Instruct"],
        skip_models=["Qwen2.5-7B"],
        datasets=["Search-QA"],
    ).merge_with_paper_default(_paper_default())
    assert [d.name for d in eff.datasets] == ["Search-QA"]
    assert sorted(eff.skip_datasets) == ["ALFWorld", "WebShop"]
    assert "Qwen2.5-7B" not in eff.models  # skip_models still reconciled


def test_narrowing_models_derives_skip_models():
    # SHOULD-FIX #3: symmetry with datasets — an operator-narrowed `models` list
    # auto-de-scopes the dropped paper-default models into skip_models, so their
    # rubric leaves are excluded rather than scored 0.
    eff = ScopeSpec(models=["Qwen3-1.7B", "Qwen2.5-3B-Instruct"]).merge_with_paper_default(_paper_default())
    assert "Qwen2.5-7B" in eff.skip_models
    assert "Qwen2.5-7B" not in eff.models
    assert sorted(eff.models) == ["Qwen2.5-3B-Instruct", "Qwen3-1.7B"]


def test_pure_default_run_derives_no_model_skips():
    eff = ScopeSpec().merge_with_paper_default(_paper_default())
    assert eff.skip_models == []
    assert "Qwen2.5-7B" in eff.models


def test_explicit_skip_datasets_reconciled_out_of_datasets():
    eff = ScopeSpec(datasets=["Search-QA", "ALFWorld"], skip_datasets=["ALFWorld"]).merge_with_paper_default(None)
    assert [d.name for d in eff.datasets] == ["Search-QA"]
    assert eff.skip_datasets == ["ALFWorld"]


def test_pure_default_run_derives_no_skips():
    # Operator constrains nothing → full paper scope, no exclusions invented.
    eff = ScopeSpec().merge_with_paper_default(_paper_default())
    assert eff.skip_datasets == []
    assert sorted(d.name for d in eff.datasets) == ["ALFWorld", "Search-QA", "WebShop"]


def test_explicit_skip_datasets_union_with_derived():
    eff = ScopeSpec(datasets=["Search-QA"], skip_datasets=["ExtraEnv"]).merge_with_paper_default(_paper_default())
    assert sorted(eff.skip_datasets) == ["ALFWorld", "ExtraEnv", "WebShop"]


def test_skip_datasets_default_empty_and_field_present():
    s = ScopeSpec()
    assert s.skip_datasets == []
    # round-trips through pydantic JSON like the rest of the spec
    assert "skip_datasets" in ScopeSpec.model_validate_json(s.model_dump_json()).model_dump()
