"""Hermetic contract-store coverage for the round-two foundation."""

from types import SimpleNamespace

import json

import pytest

from backend.agents.schemas import ReproductionContract


def _contract() -> ReproductionContract:
    return ReproductionContract(
        reproduction_definition="Train the stated model on the stated task.",
        expected_outputs=["metrics.json"],
    )


def test_contract_store_is_off_and_does_not_create_state(monkeypatch, tmp_path):
    from backend.agents.rlm import reproduction_contract_store as store

    monkeypatch.delenv("OPENRESEARCH_REPRO_CONTRACT", raising=False)
    ctx = SimpleNamespace(project_dir=tmp_path, reproduction_contract=None)

    assert store.activate(ctx, _contract().model_dump()) is None
    assert ctx.reproduction_contract is None
    assert not (tmp_path / "rlm_state" / "reproduction_contract.json").exists()


def test_contract_store_activates_and_round_trips(monkeypatch, tmp_path):
    from backend.agents.rlm import reproduction_contract_store as store

    monkeypatch.setenv("OPENRESEARCH_REPRO_CONTRACT", "1")
    ctx = SimpleNamespace(project_dir=tmp_path, reproduction_contract=None)

    activated = store.activate(ctx, _contract().model_dump())

    assert activated is not None
    assert ctx.reproduction_contract is activated
    path = tmp_path / "rlm_state" / "reproduction_contract.json"
    assert path.exists()
    loaded = store.load(tmp_path)
    assert loaded is not None
    assert loaded.expected_outputs == ["metrics.json"]


def test_contract_store_is_fail_soft_for_bad_payload(monkeypatch, tmp_path):
    from backend.agents.rlm import reproduction_contract_store as store

    monkeypatch.setenv("OPENRESEARCH_REPRO_CONTRACT", "true")
    ctx = SimpleNamespace(project_dir=tmp_path, reproduction_contract=None)

    assert store.activate(ctx, {"metrics_shape": "not-a-list"}) is None
    assert ctx.reproduction_contract is None
    assert not (tmp_path / "rlm_state" / "reproduction_contract.json").exists()


@pytest.mark.parametrize(
    ("name", "method_spec", "env_spec"),
    [
        ("sdar", {"core_contribution": "OPSD-gated GRPO"}, {"framework": "pytorch"}),
        ("vision", {"core_contribution": "All-CNN"}, {"framework": "pytorch"}),
        ("sweep", {"core_contribution": "optimizer ablation"}, {"framework": "jax"}),
    ],
)
def test_planner_contract_opt_in_is_hermetic_paired_ab(
    monkeypatch, tmp_path, make_context, name, method_spec, env_spec,
):
    """A/B: the planner's public result is identical; ON adds only the artifact/context."""
    from backend.agents.rlm.primitives import plan_reproduction

    response = json.dumps({
        "reproduction_definition": f"Reproduce {name}.",
        "expected_outputs": ["metrics.json"],
        "evaluation_plan": "Evaluate held-out results.",
    })

    monkeypatch.delenv("OPENRESEARCH_REPRO_CONTRACT", raising=False)
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    off_ctx = make_context(off_dir, llm_responses=[response])
    off = plan_reproduction(method_spec, env_spec, ctx=off_ctx)

    monkeypatch.setenv("OPENRESEARCH_REPRO_CONTRACT", "1")
    on_dir = tmp_path / "on"
    on_dir.mkdir()
    on_ctx = make_context(on_dir, llm_responses=[response])
    on = plan_reproduction(method_spec, env_spec, ctx=on_ctx)

    assert on == off
    assert off_ctx.reproduction_contract is None
    assert on_ctx.reproduction_contract is not None
    assert (on_ctx.project_dir / "rlm_state" / "reproduction_contract.json").is_file()
    assert not (off_ctx.project_dir / "rlm_state" / "reproduction_contract.json").exists()
