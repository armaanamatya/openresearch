"""Paper-declared COEFFICIENTS on the deterministic evidence layer (2026-07-13).

A paper's algorithmic constants — SDAR's ``β=10`` gate sharpness, its ``λ=0.1``
distillation weight, a temperature, a clip ε — ARE the method. They are exactly
what a surrogate implementation gets wrong and exactly what a fidelity rubric
inspects. Until the provenance contract carried them they could only be graded by
an LLM's opinion of the code, which is the precise thing the project's
"evidence, not grade" red line forbids.

The chain under test:

    rubric_gen  ──annotates──▶  coefficients.<name> assertion
        │                              │
        │ coefficient_fields()          │ deterministic_leaf_checker
        ▼                              ▼
    implementer prompt  ──emits──▶  provenance.json {"coefficients": {...}}

THE LOAD-BEARING PROPERTY IS FALSE-NEGATIVE SAFETY, NOT COVERAGE.
``emit_provenance`` is fail-soft and OPTIONAL, so a faithful run may carry no
manifest at all. A coefficient the manifest does not carry MUST route to the LLM,
never auto-zero. Only a coefficient that is LOCATED AND WRONG (β=1.0 where the
paper says 10) is a deterministic 0.0 — that is a surrogate, and catching it is
the entire point. This file therefore asserts REFUSAL at least as hard as it
asserts coverage.

Env is injected explicitly per test (the suite is env-hermetic).
"""

from __future__ import annotations

import json

import pytest

from backend.agents.baseline_implementation import _coefficient_contract_block
from backend.agents.rlm.paper_invariants import declared_coefficients, load_paper_invariants
from backend.agents.rlm.provenance import (
    COEFFICIENTS_KEY,
    build_cell_provenance,
    canonical_coefficient_name,
    coefficient_field,
    emit_provenance,
)
from backend.agents.rlm.rubric_gen import (
    annotation_coverage,
    coefficient_fields,
    generate_rubric_tree,
)
from backend.evals.paperbench.deterministic_leaf_checker import check_leaf
from backend.evals.paperbench.leaf_scorer import flatten_leaves, score_reproduction

FLAG = "OPENRESEARCH_DETERMINISTIC_LEAVES"


# --------------------------------------------------------------------------- #
# An SDAR-shaped paper: the gate, its sharpening constant, its loss weight.
# Every value a rubric asserts must occur HERE or grounding drops the annotation.
# --------------------------------------------------------------------------- #
_PAPER = """
Self-Distilled Agentic Reinforcement Learning (SDAR).

Section 3.3 Method. We introduce a sigmoid gate g_t = sigma(beta * Delta_t) with a
stop-gradient applied to the gate. The gate sharpening coefficient is set to
beta = 10 throughout, and the self-distillation loss weight is lambda = 0.1.

Section 4.1 Training setup. All models are trained with AdamW at a learning rate of
1e-4 and a batch size of 64, for 45 epochs.
""" * 4  # comfortably past the 500-char floor


def _leaf(requirements: str, check: dict | None = None, weight: float = 1.0) -> dict:
    lf: dict = {"requirements": requirements, "weight": weight}
    if check is not None:
        lf["check"] = check
    return lf


#: What a rubric author writes once the prompt tells it to split the mechanism
#: from its constant. The fidelity leaf keeps the mechanism (LLM); the two value
#: leaves pin the constants (deterministic).
_RUBRIC_RESPONSE = json.dumps({
    "categories": [{
        "name": "Method and code fidelity to the paper",
        "weight": 1.0,
        "leaves": [
            # The MECHANISM — stays LLM-graded. A check on beta's value could never
            # stand in for "implements the gate with a stop-gradient".
            _leaf(
                "train.py implements the sigmoid gate g_t = sigma(beta * Delta_t) "
                "with a stop-gradient applied to the gate (Section 3.3).",
                weight=0.5,
            ),
            # The CONSTANTS — atomic value leaves, deterministically checkable.
            _leaf(
                "The gate sharpening coefficient beta is set to 10 (Section 3.3).",
                {"kind": "deterministic:hparam", "field": "beta", "value": 10},
                weight=0.25,
            ),
            _leaf(
                "The self-distillation loss weight lambda is set to 0.1 (Section 3.3).",
                {"kind": "deterministic:hparam", "field": "lambda", "value": 0.1},
                weight=0.25,
            ),
        ],
    }],
})


class _FixedClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.systems: list[str] = []

    def complete(self, *, system: str, user: str) -> str:
        self.systems.append(system)
        return self.response


def _tree(monkeypatch, *, on: bool, response: str = _RUBRIC_RESPONSE, project_dir=None):
    if on:
        monkeypatch.setenv(FLAG, "1")
    else:
        monkeypatch.delenv(FLAG, raising=False)
    tree = generate_rubric_tree(
        _PAPER, _FixedClient(response), paper_title="SDAR", project_dir=project_dir
    )
    assert tree is not None
    return tree


def _find(tree: dict, needle: str) -> dict:
    for lf in flatten_leaves(tree):
        if needle in lf["requirements"]:
            return lf
    raise AssertionError(f"no leaf containing {needle!r}")


def _run_dir(tmp_path, coefficients=None, experiments=None):
    """A run whose train.py emitted a provenance manifest (the agent route)."""
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)
    emit_provenance(
        code,
        experiments=experiments if experiments is not None else {"e1": {"model_key": "qwen3_1_7b"}},
        coefficients=coefficients,
    )
    return tmp_path


# --------------------------------------------------------------------------- #
# 1. The unlock: an SDAR-shaped coefficient is emitted AND deterministically checked.
# --------------------------------------------------------------------------- #
def test_paper_declared_coefficients_are_annotated_and_addressed_by_namespace(monkeypatch):
    tree = _tree(monkeypatch, on=True)

    beta = _find(tree, "sharpening coefficient beta")
    assert beta["check_kind"] == "deterministic:hparam"
    assert beta["assertion"]["field"] == "coefficients.beta"
    assert beta["assertion"]["value"] == pytest.approx(10.0)
    assert beta["assertion"]["on_missing"] == "llm"  # the false-negative valve

    lam = _find(tree, "loss weight lambda")
    assert lam["assertion"]["field"] == "coefficients.lambda"
    assert lam["assertion"]["value"] == pytest.approx(0.1)

    # The MECHANISM leaf must stay with the LLM — a stub that hardcodes beta=10 and
    # implements nothing must still lose the paper's core fidelity claim.
    assert "check_kind" not in _find(tree, "implements the sigmoid gate")


def test_emitted_coefficient_is_checked_true(tmp_path):
    run_dir = _run_dir(tmp_path, coefficients={"beta": 10, "lambda": 0.1})
    leaf = {
        "id": "beta",
        "check_kind": "deterministic:hparam",
        "assertion": {
            "field": "coefficients.beta", "op": "~=", "value": 10.0,
            "tolerance": 1e-5, "on_missing": "llm",
        },
    }
    rec = check_leaf(leaf, run_dir)
    assert rec is not None and rec["score"] == 1.0
    assert rec["_graded"] is True


def test_end_to_end_rubric_gen_to_scored_leaf(monkeypatch, tmp_path):
    """rubric_gen's own assertion resolves against emit_provenance's own manifest."""
    monkeypatch.setenv(FLAG, "1")
    tree = _tree(monkeypatch, on=True)
    contract = coefficient_fields(tree)
    assert contract == {"beta": 10.0, "lambda": 0.1}

    # The agent emits exactly the contract it was handed.
    run_dir = _run_dir(tmp_path, coefficients=contract)
    for leaf in flatten_leaves(tree):
        if leaf.get("check_kind"):
            rec = check_leaf(leaf, run_dir)
            assert rec is not None, leaf["requirements"]
            assert rec["score"] == 1.0, leaf["requirements"]


# --------------------------------------------------------------------------- #
# 2. Located-and-WRONG is a real deterministic 0.0 — the surrogate is caught.
# --------------------------------------------------------------------------- #
def test_surrogate_beta_is_a_deterministic_zero(tmp_path):
    run_dir = _run_dir(tmp_path, coefficients={"beta": 1.0})  # paper says 10
    leaf = {
        "id": "beta",
        "check_kind": "deterministic:hparam",
        "assertion": {
            "field": "coefficients.beta", "op": "~=", "value": 10.0,
            "tolerance": 1e-5, "on_missing": "llm",
        },
    }
    rec = check_leaf(leaf, run_dir)
    assert rec is not None, "a LOCATED-but-wrong coefficient must NOT route to the LLM"
    assert rec["score"] == 0.0
    assert "fails" in rec["justification"]


def test_surrogate_is_caught_through_the_real_scorer(monkeypatch, tmp_path):
    """The 0.0 survives the leaf_scorer's routing, not just the checker in isolation."""
    monkeypatch.setenv(FLAG, "1")
    _run_dir(tmp_path, coefficients={"beta": 1.0})
    rubric = {
        "id": "root", "requirements": "r", "weight": 1.0,
        "sub_tasks": [{
            "id": "beta", "requirements": "beta is 10", "weight": 1.0, "sub_tasks": [],
            "check_kind": "deterministic:hparam",
            "assertion": {
                "field": "coefficients.beta", "op": "~=", "value": 10.0,
                "tolerance": 1e-5, "on_missing": "llm",
            },
        }],
    }

    class _NoLlm:
        calls = 0

        def complete(self, *, system: str, user: str) -> str:
            _NoLlm.calls += 1
            return json.dumps([{"leaf_id": "beta", "score": 1.0, "justification": "looks fine"}])

    score = score_reproduction(rubric, tmp_path, _NoLlm(), degraded=False)
    assert score["overall_score"] == 0.0, "an LLM opinion must not rescue a wrong constant"
    assert _NoLlm.calls == 0, "the leaf must never have reached the LLM at all"


# --------------------------------------------------------------------------- #
# 3. FALSE-NEGATIVE SAFETY: a coefficient provenance does not carry falls back to
#    the LLM. NEVER an auto-zero. This is the whole game.
# --------------------------------------------------------------------------- #
def test_missing_coefficient_routes_to_llm_never_auto_zero(tmp_path):
    # A faithful run that emitted a manifest but no coefficients (emit_provenance's
    # `coefficients` kwarg is optional and the whole call is fail-soft).
    run_dir = _run_dir(tmp_path, coefficients=None)
    leaf = {
        "id": "beta",
        "check_kind": "deterministic:hparam",
        "assertion": {
            "field": "coefficients.beta", "op": "~=", "value": 10.0,
            "tolerance": 1e-5, "on_missing": "llm",
        },
    }
    assert check_leaf(leaf, run_dir) is None, "missing evidence must route to the LLM"


def test_no_manifest_at_all_routes_to_llm(tmp_path):
    (tmp_path / "code").mkdir(parents=True)  # code/, but the agent never emitted provenance
    leaf = {
        "id": "beta",
        "check_kind": "deterministic:hparam",
        "assertion": {
            "field": "coefficients.beta", "op": "~=", "value": 10.0, "on_missing": "llm",
        },
    }
    assert check_leaf(leaf, tmp_path) is None


def test_partial_manifest_zeroes_nothing(tmp_path):
    """lambda emitted, beta not: lambda is graded, beta falls back — not zeroed."""
    run_dir = _run_dir(tmp_path, coefficients={"lambda": 0.1})

    def _leaf_for(name, value):
        return {
            "id": name, "check_kind": "deterministic:hparam",
            "assertion": {
                "field": f"coefficients.{name}", "op": "~=", "value": value,
                "tolerance": abs(value) * 1e-6 or 1e-12, "on_missing": "llm",
            },
        }

    assert check_leaf(_leaf_for("lambda", 0.1), run_dir)["score"] == 1.0
    assert check_leaf(_leaf_for("beta", 10.0), run_dir) is None  # NOT 0.0


def test_every_generated_coefficient_assertion_carries_the_valve(monkeypatch):
    """Structural guarantee: rubric_gen never mints a coefficient check that can auto-zero."""
    tree = _tree(monkeypatch, on=True)
    seen = 0
    for leaf in flatten_leaves(tree):
        assertion = leaf.get("assertion") or {}
        if str(assertion.get("field", "")).startswith(f"{COEFFICIENTS_KEY}."):
            assert assertion.get("on_missing") == "llm"
            seen += 1
    assert seen == 2


# --------------------------------------------------------------------------- #
# 4. learn.md 2026-07-07 REGRESSION — a legitimate alpha=0.0 ablation must not be
#    false-blocked. An over-broad LR guard that keyed on the NAME `alpha` (rather
#    than the variable's ROLE) hard-blocked prj_618's faithful sharpening ablation.
# --------------------------------------------------------------------------- #
def test_alpha_zero_ablation_is_a_legitimate_declared_value(tmp_path):
    """A paper that DECLARES alpha=0.0 is satisfied by a run that emits 0.0.

    0.0 must be treated as a value, never as "absent"/falsy — and it must not be
    range-checked out of existence.
    """
    run_dir = _run_dir(tmp_path, coefficients={"alpha": 0.0})
    leaf = {
        "id": "alpha", "check_kind": "deterministic:hparam",
        "assertion": {
            "field": "coefficients.alpha", "op": "~=", "value": 0.0,
            "tolerance": 1e-12, "on_missing": "llm",
        },
    }
    rec = check_leaf(leaf, run_dir)
    assert rec is not None and rec["score"] == 1.0


def test_alpha_zero_ablation_cell_does_not_drag_down_the_paper_cell(tmp_path):
    """The ablation sweeping alpha to 0.0 must NOT fail the paper's alpha=1.0 leaf.

    Any-match semantics: the paper cell carries alpha=1.0 and the ablation cell
    carries alpha=0.0. The run did what the paper said AND ran an extra ablation —
    crediting it is correct; failing it is the exact false-block learn.md records.
    """
    code = tmp_path / "code"
    code.mkdir(parents=True)
    emit_provenance(
        code,
        experiments={
            "paper_cell": {"baseline": "ucpo", "coefficients": {"alpha": 1.0}},
            "ablation": {"baseline": "ucpo_no_sharpening", "coefficients": {"alpha": 0.0}},
        },
    )
    leaf = {
        "id": "alpha", "check_kind": "deterministic:hparam",
        "assertion": {
            "field": "coefficients.alpha", "op": "~=", "value": 1.0,
            "tolerance": 1e-6, "on_missing": "llm",
        },
    }
    rec = check_leaf(leaf, tmp_path)
    assert rec is not None and rec["score"] == 1.0


def test_no_coefficient_value_is_ever_range_checked(tmp_path):
    """0.0, 10, and 1e4 are all legitimate declared values — none is 'absurd'."""
    for value in (0.0, 10.0, 1e4, -1.0):
        run_dir = _run_dir(tmp_path / f"r{value}", coefficients={"tau": value})
        leaf = {
            "id": "tau", "check_kind": "deterministic:hparam",
            "assertion": {
                "field": "coefficients.tau", "op": "~=", "value": value,
                "tolerance": max(abs(value) * 1e-6, 1e-12), "on_missing": "llm",
            },
        }
        rec = check_leaf(leaf, run_dir)
        assert rec is not None and rec["score"] == 1.0, value


# --------------------------------------------------------------------------- #
# 5. Float tolerance — the checker's EXISTING `~=` semantics, no new ones invented.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("emitted", "expect"),
    [
        (10, 1.0),        # int 10 == declared 10
        (10.0, 1.0),      # float 10.0 == declared 10 (JSON round-trip / int-vs-float)
        (9.999, 0.0),     # a DIFFERENT declared constant — an identity check, not a fit
        (1.0, 0.0),       # the surrogate
    ],
)
def test_float_tolerance_is_an_identity_check(tmp_path, emitted, expect):
    run_dir = _run_dir(tmp_path / f"r{emitted}", coefficients={"beta": emitted})
    leaf = {
        "id": "beta", "check_kind": "deterministic:hparam",
        "assertion": {
            "field": "coefficients.beta", "op": "~=", "value": 10.0,
            "tolerance": 1e-5, "on_missing": "llm",
        },
    }
    rec = check_leaf(leaf, run_dir)
    assert rec is not None and rec["score"] == expect


def test_lambda_authors_value_fails_the_paper_text_assertion(tmp_path):
    """0.01 (authors' scripts) != 0.1 (paper text) — a real, deterministic miss.

    This is precisely WHY the CONTESTED gate exists: when the operator's registry
    records the authors' value, the assertion is never minted in the first place.
    """
    run_dir = _run_dir(tmp_path, coefficients={"lambda": 0.01})
    leaf = {
        "id": "lam", "check_kind": "deterministic:hparam",
        "assertion": {
            "field": "coefficients.lambda", "op": "~=", "value": 0.1,
            "tolerance": 1e-7, "on_missing": "llm",
        },
    }
    assert check_leaf(leaf, run_dir)["score"] == 0.0


# --------------------------------------------------------------------------- #
# 6. The CONTESTED gate — the operator registry vetoes a symbol the paper text and
#    the authors' released code disagree about (SDAR: text beta=10, scripts beta=5).
# --------------------------------------------------------------------------- #
def test_sdar_registry_declares_the_contested_values():
    inv = load_paper_invariants("2605.15155")
    assert inv is not None and inv.algorithm is not None
    assert inv.algorithm.coefficients == {"lambda": 0.01, "beta": 5.0}


def test_contested_coefficient_is_refused_and_routed_to_the_llm(monkeypatch, tmp_path):
    """A run faithful to the authors' code must not be zeroed by the paper's number."""
    (tmp_path / "artifact_index.json").write_text(
        json.dumps({"paper": {"arxiv_id": "2605.15155"}}), encoding="utf-8"
    )
    assert declared_coefficients(tmp_path) == {"lambda": 0.01, "beta": 5.0}

    tree = _tree(monkeypatch, on=True, project_dir=tmp_path)
    # Both SDAR coefficients are contested (10 vs 5.0; 0.1 vs 0.01) → NO annotation.
    assert "check_kind" not in _find(tree, "sharpening coefficient beta")
    assert "check_kind" not in _find(tree, "loss weight lambda")
    assert coefficient_fields(tree) == {}


def test_uncontested_paper_still_annotates(monkeypatch, tmp_path):
    """A paper with no registry entry keeps its annotations — the veto is targeted."""
    (tmp_path / "artifact_index.json").write_text(
        json.dumps({"paper": {"arxiv_id": "9999.99999"}}), encoding="utf-8"
    )
    assert declared_coefficients(tmp_path) == {}
    tree = _tree(monkeypatch, on=True, project_dir=tmp_path)
    assert coefficient_fields(tree) == {"beta": 10.0, "lambda": 0.1}


# --------------------------------------------------------------------------- #
# 7. The other refusal gates (widened vocabulary, NOT a lowered bar).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("requirements", "check", "why"),
    [
        (
            "The gate is computed as g_t = sigma(beta * Delta_t) with beta=10 (Sec 3.3).",
            {"kind": "deterministic:hparam", "field": "beta", "value": 10},
            "restates the FORMULA — a stub hardcoding beta=10 would score 1.0",
        ),
        (
            "train.py implements the sigmoid gate with beta=10 (Section 3.3).",
            {"kind": "deterministic:hparam", "field": "beta", "value": 10},
            "an IMPLEMENTATION claim can never reduce to a scalar compare",
        ),
        (
            "The sampling constant is set to 10 (Section 3.3).",
            {"kind": "deterministic:hparam", "field": "beta", "value": 10},
            "ROLE gate: the leaf never names beta, so the field may be the wrong symbol",
        ),
        (
            "The frobnication coefficient is set to 10 (Section 3.3).",
            {"kind": "deterministic:hparam", "field": "frobnication", "value": 10},
            "VOCABULARY gate: not a known coefficient symbol",
        ),
        (
            "Sets lambda to 0.1 and a batch size of 64 (Section 4.1).",
            {"kind": "deterministic:hparam", "field": "lambda", "value": 0.1},
            "pins TWO numbers — one check would over-credit a run that missed the other",
        ),
        (
            "The gate sharpening coefficient beta is set to 3 (Section 3.3).",
            {"kind": "deterministic:hparam", "field": "beta", "value": 3},
            "GROUNDING gate: 3 is not a value this paper states",
        ),
    ],
)
def test_refusal_gates_hold(monkeypatch, requirements, check, why):
    response = json.dumps({
        "categories": [{
            "name": "Method and code fidelity to the paper",
            "weight": 1.0,
            "leaves": [_leaf(requirements, check)],
        }],
    })
    tree = _tree(monkeypatch, on=True, response=response)
    leaf = flatten_leaves(tree)[0]
    assert "check_kind" not in leaf, why
    assert coefficient_fields(tree) == {}


def test_beta_never_binds_to_adams_betas(tmp_path):
    """`coefficients.beta` is DOTTED, so it can never reach per_optimizer.adam.betas.

    That separation is the namespace's whole purpose: the address encodes the
    symbol's ROLE (a constant the PAPER declared), so an ambiguous name cannot be
    silently resolved against an unrelated optimizer knob.
    """
    code = tmp_path / "code"
    code.mkdir(parents=True)
    emit_provenance(
        code,
        experiments={"e1": {"per_optimizer": {"adam": {"lr": 1e-4, "betas": [0.9, 0.999]}}}},
    )
    leaf = {
        "id": "beta", "check_kind": "deterministic:hparam",
        "assertion": {
            "field": "coefficients.beta", "op": "~=", "value": 10.0, "on_missing": "llm",
        },
    }
    assert check_leaf(leaf, tmp_path) is None  # routes to LLM, does NOT bind to 0.9


# --------------------------------------------------------------------------- #
# 8. The producer: schema, canonicalization, and the harness-route merge.
# --------------------------------------------------------------------------- #
def test_greek_glyphs_canonicalize_to_the_manifest_key():
    assert canonical_coefficient_name("β") == "beta"
    assert canonical_coefficient_name("\\lambda") == "lambda"
    assert canonical_coefficient_name("Top-K") == "top_k"
    assert coefficient_field("β") == "coefficients.beta"


def test_emit_provenance_canonicalizes_and_drops_uncomparable_values(tmp_path):
    path = emit_provenance(
        tmp_path,
        experiments={"e1": {}},
        coefficients={"β": 10, "λ": 0.1, "optimizer": "adamw", "flag": True},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Non-numeric ("adamw") and bool entries are dropped: a value that cannot be
    # compared is worse than absent — absent routes to the LLM, garbage would grade.
    assert payload[COEFFICIENTS_KEY] == {"beta": 10, "lambda": 0.1}


def test_no_coefficients_means_no_key_at_all(tmp_path):
    """A caller that passes nothing writes the manifest it wrote before this existed."""
    payload = json.loads(
        emit_provenance(tmp_path, experiments={"e1": {}}).read_text(encoding="utf-8")
    )
    assert COEFFICIENTS_KEY not in payload


def test_harness_cell_route_preserves_agent_coefficients(tmp_path):
    """build_cell_provenance OVERWRITES provenance.json — it must not eat the evidence.

    The harness cannot re-derive a paper-declared coefficient from cells.json (nothing
    mechanical knows what beta is), so clobbering the agent's block would silently
    delete the only record of the algorithmic constants.
    """
    code = tmp_path / "code"
    code.mkdir(parents=True)
    (code / "cells.json").write_text(
        json.dumps({"cells": [{"id": "c1", "model_key": "m", "params": {"lr": 1e-4}}]}),
        encoding="utf-8",
    )
    emit_provenance(code, experiments={"c1": {"model_key": "m"}}, coefficients={"beta": 10})

    build_cell_provenance(code)  # the harness route runs afterwards and rewrites the file

    payload = json.loads((code / "provenance.json").read_text(encoding="utf-8"))
    assert payload["source"] == "harness_cell_provenance"  # the harness really did rewrite it
    assert payload[COEFFICIENTS_KEY] == {"beta": 10}       # ...and preserved the coefficients
    assert payload["experiments"]["c1"]["lr"] == pytest.approx(1e-4)  # harness facts still merged


def test_per_cell_override_is_addressable(tmp_path):
    """An ablation cell's own coefficients sub-dict resolves via any-match."""
    code = tmp_path / "code"
    code.mkdir(parents=True)
    emit_provenance(
        code,
        experiments={"sweep_a": {"coefficients": {"λ": 0.05}}, "sweep_b": {"coefficients": {"λ": 0.1}}},
    )
    leaf = {
        "id": "lam", "check_kind": "deterministic:hparam",
        "assertion": {
            "field": "coefficients.lambda", "op": "~=", "value": 0.1,
            "tolerance": 1e-7, "on_missing": "llm",
        },
    }
    rec = check_leaf(leaf, tmp_path)
    assert rec is not None and rec["score"] == 1.0


# --------------------------------------------------------------------------- #
# 9. The implementer's emit-contract is DERIVED from the rubric, never improvised.
# --------------------------------------------------------------------------- #
def test_implementer_block_lists_exactly_the_graded_coefficients(monkeypatch, tmp_path):
    monkeypatch.setenv(FLAG, "1")
    tree = _tree(monkeypatch, on=True)
    (tmp_path / "generated_rubric.json").write_text(json.dumps(tree), encoding="utf-8")

    block = _coefficient_contract_block(tmp_path)
    assert "'beta': 10" in block
    assert "'lambda': 0.1" in block
    assert "coefficients=" in block
    # It carries ONLY coefficients — a bookkeeping hparam has its own contract.
    assert "'epochs'" not in block
    # And it tells the agent to emit the variable it actually used, not a literal.
    assert "ACTUALLY USES" in block


def test_implementer_block_is_absent_without_a_rubric(monkeypatch, tmp_path):
    monkeypatch.setenv(FLAG, "1")
    assert _coefficient_contract_block(tmp_path) == ""


# --------------------------------------------------------------------------- #
# 10. FLAG OFF ⇒ byte-identical. The gate is the existing one; nothing flipped.
# --------------------------------------------------------------------------- #
def test_off_emits_no_coefficient_annotations(monkeypatch):
    tree = _tree(monkeypatch, on=False)
    for leaf in flatten_leaves(tree):
        assert "check_kind" not in leaf
        assert "assertion" not in leaf
    assert annotation_coverage(tree)["deterministic"] == 0
    assert coefficient_fields(tree) == {}


def test_off_omits_the_implementer_block_even_with_a_coefficient_rubric(monkeypatch, tmp_path):
    """Rubric on disk carries coefficients (from an earlier ON run) — flag off ⇒ silent."""
    monkeypatch.setenv(FLAG, "1")
    tree = _tree(monkeypatch, on=True)
    (tmp_path / "generated_rubric.json").write_text(json.dumps(tree), encoding="utf-8")

    monkeypatch.delenv(FLAG, raising=False)
    assert _coefficient_contract_block(tmp_path) == ""


def test_off_never_routes_a_coefficient_leaf_deterministically(monkeypatch, tmp_path):
    """With the flag off the checker is not consulted at all — even a wrong beta."""
    monkeypatch.delenv(FLAG, raising=False)
    _run_dir(tmp_path, coefficients={"beta": 1.0})  # a surrogate
    rubric = {
        "id": "root", "requirements": "r", "weight": 1.0,
        "sub_tasks": [{
            "id": "beta", "requirements": "beta is 10", "weight": 1.0, "sub_tasks": [],
            "check_kind": "deterministic:hparam",
            "assertion": {"field": "coefficients.beta", "op": "~=", "value": 10.0},
        }],
    }

    class _Llm:
        def complete(self, *, system: str, user: str) -> str:
            return json.dumps([{"leaf_id": "beta", "score": 0.9, "justification": "x"}])

    score = score_reproduction(rubric, tmp_path, _Llm(), degraded=False)
    assert score["overall_score"] == pytest.approx(0.9)  # the LLM's grade, unchanged
