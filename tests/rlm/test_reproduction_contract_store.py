"""Hermetic contract-store coverage for the round-two foundation."""

from types import SimpleNamespace

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
