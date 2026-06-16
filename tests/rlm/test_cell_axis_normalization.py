"""Cell-axis normalization — a ran cell must NEVER vanish from the aggregate.

Regression suite for the 2026-06-09 All-CNN collapse: the agent's cells.json
carried its own axis vocabulary (no model_key/env/baseline), run_matrix trained
14 cells to paper-grade accuracy, and ``aggregate_cell_metrics`` silently
skipped every one → ``{"status": "failed", "per_model": {}}`` → the scorer saw
"no measured metrics" for a successful run. Adam's 30-cell VAE sweep
(model_key present, env/baseline absent) hit the same family.
"""

from __future__ import annotations

from backend.agents.rlm.cell_matrix import (
    aggregate_cell_metrics,
    normalize_cell_axes,
)


def _ok_result(cells):
    return {c["id"]: {"status": "ok", "metrics": {"status": "ok", "metric": 0.9}}
            for c in cells}


# ------------------------------------------------------------ normalize_cell_axes

def test_explicit_axes_pass_through_unchanged():
    cells = [{"id": "c1", "model_key": "qwen3_1_7b", "env": "alfworld", "baseline": "grpo"}]
    out, notes = normalize_cell_axes(cells)
    assert out == cells
    assert out[0] is cells[0]  # no copy when nothing changed
    assert notes == []


def test_synonyms_fill_missing_axes():
    # All-CNN-shaped cell: letter/variant/dataset vocabulary, no harness axes.
    cells = [{
        "id": "a_base_cifar10_noaug", "letter": "A", "variant": "base",
        "dataset": "cifar10", "augment": False, "lr": 0.05,
    }]
    out, notes = normalize_cell_axes(cells)
    cell = out[0]
    assert cell["env"] == "cifar10"        # dataset → env
    assert cell["baseline"] == "base"      # variant → baseline
    assert cell["model_key"] == "a_base_cifar10_noaug"  # falls back to id
    assert notes and "derived" in notes[0]
    # input not mutated
    assert "env" not in cells[0]


def test_adam_vae_shape_partial_axes():
    # Adam VAE manifest: model_key present, env/baseline absent.
    cells = [{
        "id": "vae_adam_b2_099_la_-5", "model_key": "vae_bias_correction",
        "variant": "adam", "beta2": 0.99,
    }]
    out, _ = normalize_cell_axes(cells)
    assert out[0]["model_key"] == "vae_bias_correction"
    assert out[0]["baseline"] == "adam"    # variant → baseline
    assert out[0]["env"] == "default"      # nothing dataset-like → default


def test_synonym_source_feeds_at_most_one_axis():
    # `variant` must not become BOTH model_key and baseline.
    cells = [{"id": "x", "variant": "adam"}]
    out, _ = normalize_cell_axes(cells)
    assert out[0]["baseline"] == "adam"
    assert out[0]["model_key"] == "x"  # id fallback, NOT "adam"


def test_derived_duplicate_triples_are_disambiguated():
    cells = [
        {"id": "s0", "dataset": "mnist", "variant": "adam", "seed": 0},
        {"id": "s1", "dataset": "mnist", "variant": "adam", "seed": 1},
    ]
    out, _ = normalize_cell_axes(cells)
    triples = {(c["model_key"], c["env"], c["baseline"]) for c in out}
    assert len(triples) == 2  # second cell suffixed, no silent leaf overwrite


def test_explicit_duplicate_triples_are_disambiguated_and_warned():
    # C5 (2026-06-16): two cells with IDENTICAL explicit model_key/env/baseline
    # used to be preserved verbatim — both kept the same triple, so the later
    # leaf silently overwrote the earlier one in aggregate's per_model tree
    # (last-writer-wins). Now the later baseline is id-suffixed so BOTH survive,
    # AND a contract_warning rides back so the agent emits distinct axes.
    cells = [
        {"id": "s0", "model_key": "m", "env": "e", "baseline": "b"},
        {"id": "s1", "model_key": "m", "env": "e", "baseline": "b"},
    ]
    out, notes = normalize_cell_axes(cells)
    # Both cells survive (never dropped).
    assert len(out) == 2
    # The two leaves now occupy DISTINCT triples — no silent overwrite.
    triples = {(c["model_key"], c["env"], c["baseline"]) for c in out}
    assert len(triples) == 2
    # First cell keeps its explicit triple verbatim; the second is suffixed.
    assert out[0]["baseline"] == "b"
    assert out[1]["baseline"] == "b__s1"
    # model_key/env are untouched by the disambiguation (only baseline is suffixed).
    assert out[1]["model_key"] == "m" and out[1]["env"] == "e"
    # A single contract warning names the disambiguation (mirror of the
    # derived-dup / cell_axes_derived note); no "derived" note since axes were explicit.
    assert len(notes) == 1
    assert "already used by an earlier cell" in notes[0]
    assert "derived" not in notes[0]


def test_explicit_dup_leaf_survives_aggregate_not_overwritten():
    # End-to-end: normalize → aggregate must keep BOTH explicit-dup leaves, with
    # distinct metrics, instead of last-writer-wins collapsing them to one.
    cells = [
        {"id": "s0", "model_key": "m", "env": "e", "baseline": "b"},
        {"id": "s1", "model_key": "m", "env": "e", "baseline": "b"},
    ]
    norm, _ = normalize_cell_axes(cells)
    result = {
        "s0": {"status": "ok", "metrics": {"metric": 0.10}},
        "s1": {"status": "ok", "metrics": {"metric": 0.20}},
    }
    agg = aggregate_cell_metrics(result, norm)
    baselines = agg["per_model"]["m"]["e"]
    assert set(baselines) == {"b", "b__s1"}          # two distinct leaves
    assert baselines["b"]["metric"] == 0.10           # earlier cell NOT lost
    assert baselines["b__s1"]["metric"] == 0.20


def test_non_list_and_non_dict_entries():
    assert normalize_cell_axes(None) == ([], [])  # type: ignore[arg-type]
    out, _ = normalize_cell_axes([{"id": "a", "model_key": "m", "env": "e", "baseline": "b"}, "junk"])
    assert len(out) == 1


def test_boolean_field_never_becomes_an_axis():
    # augment=True must not be claimed by any synonym (bool is an int subclass).
    cells = [{"id": "c", "augment": True, "dataset": "cifar10"}]
    out, _ = normalize_cell_axes(cells)
    assert out[0]["baseline"] == "default"


# ------------------------------------------------------- aggregate_cell_metrics

def test_aggregate_never_drops_axisless_cells():
    """The exact All-CNN shape: axis-less manifest + ok results → real per_model."""
    cells = [
        {"id": f"{l}_{v}_cifar10_noaug", "letter": l.upper(), "variant": v,
         "dataset": "cifar10"}
        for l in ("a", "b") for v in ("base", "strided")
    ]
    agg = aggregate_cell_metrics(_ok_result(cells), cells)
    assert agg["status"] == "complete"
    assert agg["per_model"], "ran cells must never aggregate to per_model={}"
    # every cell landed somewhere in the tree
    leaves = [
        leaf
        for envs in agg["per_model"].values()
        for baselines in envs.values()
        for leaf in baselines.values()
    ]
    assert len(leaves) == len(cells)
    assert all(leaf["status"] == "ok" for leaf in leaves)


def test_aggregate_mixed_ok_and_diverged_is_partial_not_failed():
    cells = [
        {"id": "ok_cell", "dataset": "cifar10", "variant": "allcnn"},
        {"id": "dead_cell", "dataset": "cifar10", "variant": "convpool"},
    ]
    result = {
        "ok_cell": {"status": "ok", "metrics": {"metric": 0.89}},
        "dead_cell": {"status": "training_diverged", "error": "loss pinned",
                      "metrics": {}},
    }
    agg = aggregate_cell_metrics(result, cells)
    assert agg["status"] == "partial"
    assert agg["scope"]["models_run"]  # the ok cell's model is recorded


def test_aggregate_explicit_axes_unchanged_sdar_path():
    """SDAR-shaped manifests aggregate byte-identically to before."""
    cells = [{
        "id": "q17_alf_grpo", "model_key": "qwen3_1_7b", "env": "alfworld",
        "baseline": "grpo",
    }]
    agg = aggregate_cell_metrics(_ok_result(cells), cells)
    assert agg["per_model"] == {
        "qwen3_1_7b": {"alfworld": {"grpo": {"status": "ok", "metric": 0.9}}}
    }
