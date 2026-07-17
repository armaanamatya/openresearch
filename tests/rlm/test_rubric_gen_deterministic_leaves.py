"""Deterministic-leaf ANNOTATION at rubric-gen time (makes A2 reachable).

Before this, ``rubric_gen`` never emitted ``check_kind``, so
``deterministic_leaf_checker`` was dead code and every leaf of every arXiv paper
was graded by LLM opinion — the thing the "evidence, not grade" red line forbids.

The load-bearing property under test is NOT coverage, it is SAFETY: an annotation
is written before the run, so it can only *predict* the artifacts. Every gate is
one-directional — it may only ever REFUSE to annotate. A refused leaf is graded by
the LLM exactly as today, which is always a correct fallback; a WRONG annotation
would deterministically fail a faithful reproduction, which is the expensive error
(learn.md 2026-07-07: an over-broad LR guard hard-blocked a faithful alpha=0.0
ablation). So most of these tests assert that we DECLINE to annotate.

Env is injected explicitly per test (the suite is env-hermetic).
"""

from __future__ import annotations

import json
import re

import pytest

from backend.agents.rlm.rubric_gen import (
    _SYSTEM_PROMPT,
    _build_system_prompt,
    annotation_coverage,
    generate_rubric_tree,
    strip_invalid_annotations,
)
from backend.evals.paperbench.deterministic_leaf_checker import check_leaf
from backend.evals.paperbench.leaf_scorer import flatten_leaves, score_reproduction

# --------------------------------------------------------------------------- #
# A representative paper + the rubric an LLM writes for it.
# Every number the rubric asserts must occur HERE or the annotation is dropped.
# --------------------------------------------------------------------------- #
_PAPER = """
Self-Distilled Agentic Reinforcement Learning (SDAR).

Section 3.3 Method. We introduce a sigmoid gate g_t = sigma(beta * Delta_t) with a
stop-gradient applied to the gate. The sharpening coefficient is set to beta = 10
throughout, and the self-distillation loss weight is lambda = 0.1.

Section 4.1 Training setup. All models are trained with AdamW at a learning rate of
1e-4 and a batch size of 64. We use a single random seed of 42 for every run, with a
weight decay of 0.01. Table 2 lists the full hyper-parameters.

Section 4.2 Schedule. Each model is trained for 45 epochs on the ALFWorld corpus.

Section 5 Results. SDAR reaches an ALFWorld success rate of 72.3, improving over the
GRPO baseline. Training loss falls to 0.08 by the end of the schedule.
""" * 3  # comfortably past the 500-char floor


def _leaf(requirements: str, check: dict | None = None, weight: float = 1.0) -> dict:
    lf: dict = {"requirements": requirements, "weight": weight}
    if check is not None:
        lf["check"] = check
    return lf


#: The nine-leaf rubric the model returns. Four SHOULD annotate; five must not —
#: each for a different, individually-tested reason.
_RUBRIC_RESPONSE = json.dumps({
    "categories": [
        {
            "name": "Method and code fidelity to the paper",
            "weight": 0.35,
            "leaves": [
                # (a) DROP: field `beta` is not written by ANY provenance producer,
                #     and the leaf's substance is algorithmic. Annotating it would
                #     zero a faithful SDAR run.
                _leaf(
                    "train.py implements the sigmoid gate g_t = sigma(beta * Delta_t) "
                    "with beta=10 and a stop-gradient on the gate (Section 3.3).",
                    {"kind": "deterministic:hparam", "field": "beta", "value": 10},
                ),
                # (b) DROP: pure judgment, no check proposed. The correct outcome.
                _leaf("The method is faithfully described in the README."),
            ],
        },
        {
            "name": "Experiment execution and reproducibility",
            "weight": 0.25,
            "leaves": [
                # (c) ANNOTATE: single whitelisted hparam, single pinned number.
                _leaf(
                    "Trains for 45 epochs as specified in Section 4.2.",
                    {"kind": "deterministic:hparam", "field": "epochs", "value": 45},
                ),
                # (d) ANNOTATE: float hparam; 1e-4 grounds against the paper.
                _leaf(
                    "Uses a learning rate of 1e-4 (Section 4.1).",
                    {"kind": "deterministic:hparam", "field": "learning_rate",
                     "value": 0.0001},
                ),
                # (e) DROP: pins TWO numbers — one assertion would over-credit a run
                #     that got lambda right and the batch size wrong.
                _leaf(
                    "Sets lambda=0.1 for the loss weight and batch size 64, "
                    "matching Section 4.1 Table 2.",
                    {"kind": "deterministic:hparam", "field": "batch_size", "value": 64},
                ),
            ],
        },
        {
            "name": "Evaluation protocol and metric correctness",
            "weight": 0.20,
            "leaves": [
                # (f) ANNOTATE: a genuine existence question.
                _leaf(
                    "A metrics.json file is written containing the per-model results.",
                    {"kind": "deterministic:artifact", "glob": "metrics.json"},
                ),
            ],
        },
        {
            "name": "Result match versus the paper's reported targets",
            "weight": 0.15,
            "leaves": [
                # (g) ANNOTATE: reported result, unambiguous polarity, grounded.
                _leaf(
                    "The ALFWorld success rate is at least 72.3 (Section 5).",
                    {"kind": "deterministic:numeric", "metric_key": "success_rate",
                     "target": 72.3},
                ),
                # (h) DROP: 99.9 appears NOWHERE in the paper — a hallucinated target
                #     would deterministically fail a correct reproduction.
                _leaf(
                    "Achieves an accuracy of 99.9 on the held-out split.",
                    {"kind": "deterministic:numeric", "metric_key": "accuracy",
                     "target": 99.9},
                ),
            ],
        },
        {
            "name": "Artifact completeness and provenance",
            "weight": 0.05,
            "leaves": [
                # (i) DROP: an existence check must never stand in for a fidelity
                #     claim — an empty train.py would score 1.0 on the algorithm.
                _leaf(
                    "train.py implements the training loop and exists in code/.",
                    {"kind": "deterministic:artifact", "glob": "train.py"},
                ),
            ],
        },
    ]
})


class _FixedClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
        self.systems: list[str] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        self.systems.append(system)
        return self.response


def _tree(monkeypatch, *, on: bool) -> tuple[dict, _FixedClient]:
    if on:
        monkeypatch.setenv("OPENRESEARCH_DETERMINISTIC_LEAVES", "1")
    else:
        monkeypatch.delenv("OPENRESEARCH_DETERMINISTIC_LEAVES", raising=False)
    client = _FixedClient(_RUBRIC_RESPONSE)
    tree = generate_rubric_tree(_PAPER, client, paper_title="SDAR")
    assert tree is not None
    return tree, client


def _by_text(tree: dict) -> dict[str, dict]:
    return {lf["requirements"]: lf for lf in flatten_leaves(tree)}


def _find(tree: dict, needle: str) -> dict:
    for req, lf in _by_text(tree).items():
        if needle in req:
            return lf
    raise AssertionError(f"no leaf containing {needle!r}")


# --------------------------------------------------------------------------- #
# OFF: byte-identical to today (the flag stays the gate; we made it functional).
# --------------------------------------------------------------------------- #
def test_off_emits_no_annotations_and_leaves_the_prompt_unchanged(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_DETERMINISTIC_LEAVES", raising=False)
    assert _build_system_prompt() == _SYSTEM_PROMPT

    tree, client = _tree(monkeypatch, on=False)
    assert client.systems[0] == _SYSTEM_PROMPT  # the LLM saw today's exact prompt
    for lf in flatten_leaves(tree):
        assert "check_kind" not in lf
        assert "assertion" not in lf
    assert annotation_coverage(tree)["deterministic"] == 0


def test_on_extends_the_prompt(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_DETERMINISTIC_LEAVES", "1")
    prompt = _build_system_prompt()
    assert prompt.startswith(_SYSTEM_PROMPT)
    assert "deterministic:hparam" in prompt
    assert "OMIT" in prompt  # the omit-if-unsure bias is stated to the model


# --------------------------------------------------------------------------- #
# ON: the three mechanically-checkable kinds are emitted, and VALID.
# --------------------------------------------------------------------------- #
def test_hparam_leaf_emits_valid_annotation(monkeypatch):
    tree, _ = _tree(monkeypatch, on=True)
    lf = _find(tree, "Trains for 45 epochs")
    assert lf["check_kind"] == "deterministic:hparam"
    assert lf["assertion"]["field"] == "epochs"
    assert lf["assertion"]["op"] == "=="
    assert lf["assertion"]["value"] == 45
    # The false-negative valve: a faithful run that skipped the optional
    # provenance manifest must fall through to the LLM, never grade 0.0.
    assert lf["assertion"]["on_missing"] == "llm"


def test_float_hparam_is_canonicalized_and_grounded(monkeypatch):
    tree, _ = _tree(monkeypatch, on=True)
    lf = _find(tree, "learning rate of 1e-4")
    # "learning_rate" is normalized to the canonical provenance field name `lr`.
    assert lf["check_kind"] == "deterministic:hparam"
    assert lf["assertion"]["field"] == "lr"
    assert lf["assertion"]["op"] == "~="
    assert lf["assertion"]["value"] == pytest.approx(1e-4)
    assert lf["assertion"]["tolerance"] > 0


def test_artifact_leaf_emits_valid_annotation(monkeypatch):
    tree, _ = _tree(monkeypatch, on=True)
    lf = _find(tree, "A metrics.json file is written")
    assert lf["check_kind"] == "deterministic:artifact"
    assert lf["assertion"]["glob"] == ["metrics.json"]
    # Existence is exactly the question asked, so a missing file IS a real 0.0 —
    # no on_missing valve here, deliberately.
    assert "on_missing" not in lf["assertion"]


def test_numeric_leaf_emits_valid_annotation(monkeypatch):
    tree, _ = _tree(monkeypatch, on=True)
    lf = _find(tree, "ALFWorld success rate")
    assert lf["check_kind"] == "deterministic:numeric"
    a = lf["assertion"]
    assert a["metric_key"] == "success_rate"
    assert a["target"] == pytest.approx(72.3)
    assert a["direction"] == "higher_better"  # polarity inferred from the metric
    assert a["tolerance"] > 0  # a faithful repro lands NEAR, not exactly on, 72.3
    assert a["on_missing"] == "llm"


# --------------------------------------------------------------------------- #
# The refusals. Each is a distinct false-negative / over-credit the gates stop.
# --------------------------------------------------------------------------- #
def test_judgment_leaf_gets_no_annotation(monkeypatch):
    tree, _ = _tree(monkeypatch, on=True)
    lf = _find(tree, "faithfully described")
    assert "check_kind" not in lf and "assertion" not in lf


def test_ungrounded_value_is_never_guessed(monkeypatch):
    """A target the paper never states must NOT be asserted.

    This is the whole anti-hallucination bias: a made-up 99.9 target would
    deterministically fail a correct reproduction.
    """
    tree, _ = _tree(monkeypatch, on=True)
    lf = _find(tree, "accuracy of 99.9")
    assert "check_kind" not in lf


def test_non_whitelisted_hparam_field_is_refused(monkeypatch):
    """`beta` is real, stated, and load-bearing — and NO provenance producer
    writes a field by that name. Asserting it would hard-fail a faithful SDAR
    run with `provenance_missing:beta`. It must stay LLM-graded.
    """
    tree, _ = _tree(monkeypatch, on=True)
    lf = _find(tree, "sigmoid gate")
    assert "check_kind" not in lf


def test_compound_leaf_is_refused(monkeypatch):
    """A leaf pinning two numbers cannot be checked by one assertion without
    crediting a run that got the other number wrong."""
    tree, _ = _tree(monkeypatch, on=True)
    lf = _find(tree, "lambda=0.1")
    assert "check_kind" not in lf


def test_existence_check_never_stands_in_for_fidelity(monkeypatch):
    """"train.py implements the training loop" must not decay into "does
    train.py exist" — a stub with an empty train.py would score 1.0 on the
    paper's core algorithmic claim."""
    tree, _ = _tree(monkeypatch, on=True)
    lf = _find(tree, "implements the training loop")
    assert "check_kind" not in lf


# --------------------------------------------------------------------------- #
# Malformed annotations are DROPPED before persistence, never written to disk.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kind,assertion",
    [
        ("deterministic:hparam", {"field": "beta", "op": "==", "value": 10}),   # field not emitted
        ("deterministic:hparam", {"field": "lr", "op": "≈", "value": 1e-4}),    # bad op
        ("deterministic:hparam", {"field": "lr", "op": "=="}),                  # no value
        ("deterministic:artifact", {"glob": "../../etc/passwd"}),               # path escape
        ("deterministic:artifact", {"glob": "/abs/path"}),                      # absolute
        ("deterministic:artifact", {"glob": []}),                               # empty
        ("deterministic:numeric", {"metric_key": "acc", "direction": "sideways",
                                   "target": 1}),                               # bad direction
        ("deterministic:numeric", {"metric_key": "acc", "direction": "higher_better"}),  # no target
        ("bogus:kind", {"field": "lr", "op": "==", "value": 1}),                # unknown kind
    ],
)
def test_malformed_annotation_is_stripped_not_persisted(kind, assertion):
    tree = {
        "id": "root", "requirements": "r", "weight": 1.0,
        "sub_tasks": [{
            "id": "leaf", "requirements": "x", "weight": 1.0, "sub_tasks": [],
            "check_kind": kind, "assertion": assertion,
        }],
    }
    assert strip_invalid_annotations(tree) == 1
    leaf = tree["sub_tasks"][0]
    assert "check_kind" not in leaf and "assertion" not in leaf
    # And what would have been persisted carries no trace of it.
    assert "check_kind" not in json.dumps(tree)
    # A stripped leaf is simply an un-annotated leaf → the LLM grades it.
    assert check_leaf(leaf, "/nonexistent") is None


def test_valid_annotation_survives_the_strip():
    tree = {
        "id": "root", "requirements": "r", "weight": 1.0,
        "sub_tasks": [{
            "id": "leaf", "requirements": "x", "weight": 1.0, "sub_tasks": [],
            "check_kind": "deterministic:hparam",
            "assertion": {"field": "epochs", "op": "==", "value": 45},
        }],
    }
    assert strip_invalid_annotations(tree) == 0
    assert tree["sub_tasks"][0]["check_kind"] == "deterministic:hparam"


# --------------------------------------------------------------------------- #
# End-to-end: the annotated leaves are graded by the CHECKER, not the LLM.
# --------------------------------------------------------------------------- #
class _Grader:
    """Records every leaf id the LLM grader was actually shown."""

    def __init__(self) -> None:
        self.graded_ids: list[str] = []
        self.prompts: list[str] = []

    def complete(self, *, system: str, user: str) -> str:
        self.prompts.append(user)
        ids = re.findall(r'"leaf_id":\s*"([^"]+)"', user)
        self.graded_ids.extend(ids)
        return json.dumps(
            [{"leaf_id": i, "score": 0.5, "justification": "llm"} for i in ids]
        )


def _write_faithful_run(run_dir):
    """A CORRECT reproduction: every asserted value present and matching."""
    code = run_dir / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "provenance.json").write_text(json.dumps({
        "schema_version": 1,
        "experiments": {
            # The agent-emitter shape: lr lives under per_optimizer.<opt>.lr.
            "sdar__alfworld": {
                "model_key": "qwen", "epochs": 45, "batch_size": 64, "seed": 42,
                "per_optimizer": {"adamw": {"lr": 1e-4}},
            },
        },
    }), encoding="utf-8")
    (code / "metrics.json").write_text(json.dumps({
        "per_model": {"qwen": {"alfworld": {"sdar": {
            "status": "ok", "success_rate": 72.5,
        }}}},
    }), encoding="utf-8")


def test_annotated_leaves_bypass_the_llm_grader(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_DETERMINISTIC_LEAVES", "1")
    monkeypatch.delenv("OPENRESEARCH_GRADER_SAMPLES", raising=False)
    monkeypatch.delenv("OPENRESEARCH_GRADER_BACKEND", raising=False)
    tree, _ = _tree(monkeypatch, on=True)
    _write_faithful_run(tmp_path)

    det_ids = {
        lf["id"] for lf in flatten_leaves(tree) if lf.get("check_kind")
    }
    assert len(det_ids) == 4  # the four mechanically-checkable leaves

    grader = _Grader()
    result = score_reproduction(tree, tmp_path, grader, degraded=False)

    # THE ASSERTION THAT MATTERS: no annotated leaf was ever shown to the LLM —
    # not in the parsed task list, and not anywhere in the raw prompt text.
    assert det_ids.isdisjoint(grader.graded_ids)
    blob = "".join(grader.prompts)
    for lid in det_ids:
        assert lid not in blob
    # ...and they were still all graded (by the checker), carrying its provenance.
    scored = {r["id"] for r in result["leaf_scores"]}
    assert det_ids <= scored
    # The five judgment leaves DID reach the LLM — we did not silently drop them.
    assert len(set(grader.graded_ids)) == 5


def test_a_correct_reproduction_is_not_false_failed(monkeypatch, tmp_path):
    """The regression that matters most: every deterministic leaf must PASS on a
    faithful run. A deterministic leaf that fires wrongly is a false negative,
    and for a triage product a false negative is the expensive error."""
    monkeypatch.setenv("OPENRESEARCH_DETERMINISTIC_LEAVES", "1")
    tree, _ = _tree(monkeypatch, on=True)
    _write_faithful_run(tmp_path)

    for lf in flatten_leaves(tree):
        if not lf.get("check_kind"):
            continue
        rec = check_leaf(lf, tmp_path)
        assert rec is not None, lf["requirements"]
        assert rec["score"] == 1.0, (
            f"FALSE NEGATIVE — a faithful reproduction failed a deterministic leaf:\n"
            f"  {lf['requirements']!r}\n  {rec['justification']}"
        )


def test_missing_provenance_falls_through_to_llm_not_zero(monkeypatch, tmp_path):
    """emit_provenance is fail-soft and OPTIONAL. A faithful run that simply
    skipped the manifest must NOT have every hyperparameter leaf zeroed — that
    would be strictly worse than the LLM, which can read lr=1e-4 out of train.py.
    """
    monkeypatch.setenv("OPENRESEARCH_DETERMINISTIC_LEAVES", "1")
    tree, _ = _tree(monkeypatch, on=True)
    (tmp_path / "code").mkdir(parents=True, exist_ok=True)  # no provenance.json

    hparams = [
        lf for lf in flatten_leaves(tree)
        if lf.get("check_kind") == "deterministic:hparam"
    ]
    assert hparams
    for lf in hparams:
        assert check_leaf(lf, tmp_path) is None  # → LLM, NOT a 0.0


def test_wrong_value_still_fails_deterministically(monkeypatch, tmp_path):
    """The valve must not defang the check: a value that is PRESENT and WRONG is
    still a real, deterministic 0.0."""
    monkeypatch.setenv("OPENRESEARCH_DETERMINISTIC_LEAVES", "1")
    tree, _ = _tree(monkeypatch, on=True)
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "provenance.json").write_text(
        json.dumps({"experiments": {"e1": {"epochs": 3}}}), encoding="utf-8"
    )
    lf = _find(tree, "Trains for 45 epochs")
    rec = check_leaf(lf, tmp_path)
    assert rec is not None and rec["score"] == 0.0


# --------------------------------------------------------------------------- #
# The number this exercise exists to move.
# --------------------------------------------------------------------------- #
def test_coverage_fraction_on_a_representative_rubric(monkeypatch):
    tree, _ = _tree(monkeypatch, on=True)
    cov = annotation_coverage(tree)
    assert cov["total"] == 9
    assert cov["deterministic"] == 4
    assert cov["llm"] == 5
    assert cov["by_kind"] == {
        "deterministic:hparam": 2,
        "deterministic:artifact": 1,
        "deterministic:numeric": 1,
    }
    assert cov["fraction"] == pytest.approx(4 / 9)
