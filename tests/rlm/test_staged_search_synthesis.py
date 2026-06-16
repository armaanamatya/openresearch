"""Issue #1 (2026-06-15): harness auto-synthesis of the cells.json `search` block.

When a paper hint declares an `lr_search` grid, the harness synthesizes a staged
search from the agent's emitted cells × the grid — so a per-model LR search fires
even when the agent ships a single fixed lr (the observed All-CNN failure: every
model at lr=0.05, base-A 15.61% vs paper 12.5%, All-CNN-C inverted).

These tests pin the pure core: the synthesized block is valid (round-trips through
parse_search_spec), the searched value reaches BOTH cell shapes, and the winner's
tuned value is promoted to the top level (so a whole-cell-reading trainer honors it).
"""

from __future__ import annotations

from backend.agents.rlm import staged_search as ss

# All-CNN-shape cell: lr/epochs at TOP LEVEL (the trainer reads p.get("lr")).
ALLCNN_CELL = {
    "id": "c_allcnn_cifar10_noaug",
    "model_key": "c_allcnn",
    "env": "cifar10_noaug",
    "baseline": "allcnn",
    "lr": 0.05,
    "epochs": 350,
    "batch_size": 128,
    "seed": 42,
}

LR_SEARCH = {
    "grid": [0.25, 0.1, 0.05, 0.01],
    "param_key": "lr",
    "epochs_key": "epochs",
    "probe_epochs": 8,
    "select_metric": "final_train_loss",
    "select_objective": "min",
}


def test_synth_produces_one_group_per_cell_with_grid_candidates():
    search = ss.synthesize_search_from_hint([ALLCNN_CELL], LR_SEARCH)
    assert len(search) == 1
    g = search[0]
    assert g["group"] == "c_allcnn_cifar10_noaug"
    assert g["select_metric"] == "final_train_loss"
    assert g["select_objective"] == "min"
    assert g["param_from_winner"] == ["lr"]
    assert len(g["candidates"]) == 4
    # Each candidate carries its grid lr + the probe epochs in BOTH shapes.
    for cand, lr in zip(g["candidates"], LR_SEARCH["grid"]):
        assert cand["id"] == f"c_allcnn_cifar10_noaug__lr_{lr}"
        assert cand["lr"] == lr and cand["params"]["lr"] == lr
        assert cand["epochs"] == 8 and cand["params"]["epochs"] == 8
    # Promote keeps the agent's full epochs (mirrored into params for the budget calc).
    assert g["promote"]["epochs"] == 350 and g["promote"]["params"]["epochs"] == 350


def test_synth_empty_on_unusable_input():
    assert ss.synthesize_search_from_hint([], LR_SEARCH) == []
    assert ss.synthesize_search_from_hint([ALLCNN_CELL], {"grid": []}) == []
    assert ss.synthesize_search_from_hint([{"no_id": 1}], LR_SEARCH) == []


def test_synth_roundtrips_through_parse_search_spec():
    """The synthesized block must be a VALID search spec the existing parser accepts."""
    search = ss.synthesize_search_from_hint([ALLCNN_CELL], LR_SEARCH)
    groups = ss.parse_search_spec({"cells": [ALLCNN_CELL], "search": search})
    assert len(groups) == 1
    assert len(groups[0].candidates) == 4
    assert groups[0].param_from_winner == ["lr"]


def test_materialize_copies_winner_to_top_level_and_params():
    """The Issue #1 fix: a whole-cell-reading trainer (All-CNN) must see the tuned lr
    at the TOP LEVEL, not only under ['params']."""
    search = ss.synthesize_search_from_hint([ALLCNN_CELL], LR_SEARCH)
    groups = ss.parse_search_spec({"search": search})
    # Pretend lr=0.1 (params + top-level both set by the synth) won.
    winner = next(c for c in groups[0].candidates if c["params"]["lr"] == 0.1)
    full = ss.materialize_full_cells(groups, {"c_allcnn_cifar10_noaug": winner})
    assert len(full) == 1
    promoted = full[0]
    assert promoted["lr"] == 0.1, "tuned lr must reach the top level (trainer reads p.get('lr'))"
    assert promoted["params"]["lr"] == 0.1
    assert promoted["epochs"] == 350, "promote keeps the full epochs, not the probe budget"
    assert promoted["_tuned_from"] == winner["id"]


def test_end_to_end_pure_select_then_promote():
    """synth → select_winner (by final_train_loss) → materialize at the tuned lr."""
    search = ss.synthesize_search_from_hint([ALLCNN_CELL], LR_SEARCH)
    groups = ss.parse_search_spec({"search": search})
    g = groups[0]
    # lr=0.1 has the lowest probe train loss → it should win and be promoted.
    losses = {0.25: 1.9, 0.1: 0.7, 0.05: 1.1, 0.01: 2.3}
    candidate_results = {
        c["id"]: {"metrics": {"final_train_loss": losses[c["params"]["lr"]]}}
        for c in g.candidates
    }
    winner = ss.select_winner(g, candidate_results)
    assert winner is not None and winner["params"]["lr"] == 0.1
    full = ss.materialize_full_cells(groups, {g.group: winner})
    assert full[0]["lr"] == 0.1 and full[0]["epochs"] == 350


# ---------------------------------------------------------------------------
# synthesize_search_from_leaf (L4, 2026-06-16) — the same staged tune-then-run,
# triggered by a result_quality leaf DIAGNOSIS instead of a paper-hint grid.
# ---------------------------------------------------------------------------

# Adam-shape cell: lr/epochs live under ["params"] (the trainer reads p["params"]["lr"]).
ADAM_CELL = {
    "id": "mnist_mlp_adam",
    "model_key": "mnist_mlp",
    "env": "plain",
    "baseline": "adam",
    "params": {"lr": 0.05, "epochs": 200},
}


def test_leaf_synth_one_group_per_condition_cell():
    # Each emitted cell is one per-condition unit (one optimizer) → one group each,
    # so every condition gets tuned at ITS OWN lr (the inverted-ordering fix).
    cells = [dict(ADAM_CELL, id="mnist_adam", baseline="adam"),
             dict(ADAM_CELL, id="mnist_sgd", baseline="sgd")]
    search = ss.synthesize_search_from_leaf(cells)
    assert len(search) == 2
    assert all(g["param_from_winner"] == ["lr"] for g in search)
    # Uses the default grid (3 candidates) when no explicit grid is given.
    assert all(len(g["candidates"]) == 3 for g in search)


def test_leaf_synth_roundtrips_through_parse_search_spec():
    search = ss.synthesize_search_from_leaf([ADAM_CELL])
    groups = ss.parse_search_spec({"search": search})
    assert len(groups) == 1
    assert groups[0].param_from_winner == ["lr"]


def test_leaf_synth_honors_explicit_grid():
    search = ss.synthesize_search_from_leaf([ADAM_CELL], lr_grid=[1e-4, 1e-3])
    cand_lrs = {c["params"]["lr"] for c in search[0]["candidates"]}
    assert cand_lrs == {1e-4, 1e-3}


def test_leaf_synth_empty_on_unusable_input():
    assert ss.synthesize_search_from_leaf([]) == []
    assert ss.synthesize_search_from_leaf(None) == []  # type: ignore[arg-type]
    # A falsy/empty grid falls back to the DEFAULT grid (still a usable search).
    assert ss.synthesize_search_from_leaf([ADAM_CELL], lr_grid=[]) != []
    # A grid with no numeric entries collapses to empty → legacy fallback.
    assert ss.synthesize_search_from_leaf([ADAM_CELL], lr_grid=["bad"]) == []  # type: ignore[list-item]
