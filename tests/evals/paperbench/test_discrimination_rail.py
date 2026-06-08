"""Discrimination-rail fixtures + dual-run regression (spec 2026-06-07 §8, D7a/D7b).

The anchored-grader calibration (D1) flips the grader's default posture from
"start at 0, be conservative" to "start at 1.0, deduct only named gaps." That is
inflation-prone, so D7 is the mandatory counterweight: a committed, **LLM-free**
regression that proves a *faithful* reproduction climbs while a *fake* stays
pinned low, and the spread between them WIDENS rather than narrows.

This module is the instrument (D8 build-order step 1): it ships with *stub*
LlmClients only — no live grader, fully deterministic. Two stubs simulate the
prompt-anchor change without changing the grader model:

  * ``_OldAnchorLlm`` — the floor-anchored "be conservative" grader (~0.6/leaf).
  * ``_NewAnchorLlm`` — the anchored-to-1.0 grader (~0.97/leaf).

Both ignore prompt content (the change they model is the *prompt*, not the
evidence). The gate / ceiling / roll-up are deterministic Python, so old-vs-new
over the same fixtures is a legitimate A/B (spec §8: "a **prompt** A/B on the
*same* grader model — not a model swap").

Four hand-authored synthetic run-dirs (D7a), each built in ``tmp_path`` so
nothing is committed:

  | class      | built from                                        | band under new anchor |
  |------------|---------------------------------------------------|-----------------------|
  | surrogate  | ``class TinyLM`` + ``# surrogate model``          | == 0.0  (hard gate)   |
  | degraded   | faithful code, ``baseline_metrics={}``            | <= 0.35 (degraded)    |
  | faithful   | real Qwen + sigmoid(beta*Δ).detach(), λ=0.1, β=10 | >= 0.9                |
  | sabotaged  | faithful w/ ``.detach()`` + ``from_pretrained`` removed | <= 0.5  (soft gate) |

The bands are enforced by the deterministic discrimination floor
(``INVARIANT_HARD_CAP=0.0``, ``INVARIANT_SOFT_CAP=0.5``,
``DEGRADED_LEAF_CEILING=0.35``) which runs *after* grading and is never bypassed
by the calibration change (spec §0 design principle 4).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.prompts.paper_hints import lookup_paper_hint
from backend.evals.paperbench.leaf_scorer import (
    DEGRADED_LEAF_CEILING,
    INVARIANT_HARD_CAP,
    INVARIANT_SOFT_CAP,
    score_reproduction,
)

# ---------------------------------------------------------------------------
# Shared rubric tree (shape copied from test_leaf_scorer_invariant_gate.py).
# Two leaves so a batch is exercised; weights are equal.
# ---------------------------------------------------------------------------

TINY_TREE: dict = {
    "id": "root",
    "requirements": "root",
    "weight": 1,
    "sub_tasks": [
        {
            "id": "leaf-a1",
            "requirements": "sigmoid gate present",
            "weight": 1,
            "sub_tasks": [],
        },
        {
            "id": "leaf-a2",
            "requirements": "real model weights used",
            "weight": 1,
            "sub_tasks": [],
        },
    ],
}


def _sdar_invariants() -> list:
    """The real SDAR InvariantSpec list (the deterministic discrimination floor)."""
    hint = lookup_paper_hint("2605.15155")
    assert hint is not None, "SDAR paper hint not found in PAPER_HINTS"
    return hint.invariants


# ---------------------------------------------------------------------------
# Stub LlmClients — content-blind; they model the PROMPT-anchor change only.
# ---------------------------------------------------------------------------


def _scores_json(score: float) -> str:
    """A grader response giving every leaf in TINY_TREE the same ``score``."""
    return json.dumps(
        [
            {"leaf_id": "leaf-a1", "score": score, "justification": "stub"},
            {"leaf_id": "leaf-a2", "score": score, "justification": "stub"},
        ]
    )


class _OldAnchorLlm:
    """Floor-anchored ("be conservative") grader — ~0.6 on every leaf."""

    def complete(self, *, system: str, user: str) -> str:  # noqa: ARG002
        return _scores_json(0.6)


class _NewAnchorLlm:
    """Anchored-to-1.0 grader — ~0.97 on every leaf."""

    def complete(self, *, system: str, user: str) -> str:  # noqa: ARG002
        return _scores_json(0.97)


# ---------------------------------------------------------------------------
# Fixture writer — builds a minimal run-dir for a named class in tmp_path.
# ---------------------------------------------------------------------------

# A faithful SDAR train.py: hits every must_match invariant and trips no
# must_not_match.  Variable names match the paper_hints regexes
# (lambda_ = 0.1, beta = 10, sigmoid(beta * …).detach(), grpo_loss + opsd_loss,
# from_pretrained('Qwen/…')).
_FAITHFUL_TRAIN_PY = (
    "import torch\n"
    "from transformers import AutoModelForCausalLM\n"
    "\n"
    "# Hyper-parameters (SDAR paper §3.2)\n"
    "self_distill_weight = 0.1   # OPSD self-distillation weight lambda (eq. 5)\n"
    "beta = 10                   # gate sharpness\n"
    "\n"
    "model = AutoModelForCausalLM.from_pretrained(\"Qwen/Qwen3-1.7B-Instruct\")\n"
    "\n"
    "def compute_loss(advantages, delta_t, teacher_logprobs, student_logprobs):\n"
    "    grpo_loss = -(advantages * student_logprobs).mean()\n"
    "    # OPSD on-policy self-distillation gate, stop-gradient through the gate\n"
    "    g_t = torch.sigmoid(beta * delta_t).detach()\n"
    "    opsd_loss = -(g_t * teacher_logprobs).mean()\n"
    "    return grpo_loss + self_distill_weight * opsd_loss\n"
)

# Sabotaged-faithful: same body but with the stop-gradient removed (.detach()
# stripped, no torch.no_grad replacement) AND the real-weights line deleted.
# Trips two SOFT gates (stop_gradient_on_gate + real_qwen_weights_not_surrogate
# must_match miss); trips NO hard gate (no surrogate marker) → capped at 0.5.
_SABOTAGED_TRAIN_PY = (
    "import torch\n"
    "from transformers import AutoModelForCausalLM\n"
    "\n"
    "self_distill_weight = 0.1\n"
    "beta = 10\n"
    "\n"
    "def compute_loss(advantages, delta_t, teacher_logprobs, student_logprobs):\n"
    "    grpo_loss = -(advantages * student_logprobs).mean()\n"
    "    # stop-gradient removed below (sabotage)\n"
    "    g_t = torch.sigmoid(beta * delta_t)\n"
    "    opsd_loss = -(g_t * teacher_logprobs).mean()\n"
    "    return grpo_loss + self_distill_weight * opsd_loss\n"
)

# Surrogate: a TinyLM stub with the explicit surrogate marker → hard gate.
_SURROGATE_TRAIN_PY = (
    "import torch\n"
    "# surrogate model — avoid downloading real Qwen weights\n"
    "class TinyLM(torch.nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.lm_head = torch.nn.Linear(128, 50257)\n"
    "\n"
    "lambda_ = 0.1\n"
    "beta = 10\n"
    "g_t = torch.sigmoid(beta * delta).detach()\n"
    "grpo_loss = -advantages.mean()\n"
    "opsd_loss = -g_t.mean()\n"
)

_REAL_METRICS = {
    "per_model": {
        "Qwen3-1.7B-Instruct": {
            "ALFWorld": {"SDAR": {"success_rate": 0.62}, "GRPO": {"success_rate": 0.54}}
        }
    },
    "comparison": {"SDAR_vs_GRPO": {"direction": "SDAR>GRPO", "delta": 0.08}},
}


def _write_fixture(tmp: Path, cls: str) -> Path:
    """Write a minimal run-dir (code/train.py + final_report.json + code/metrics.json).

    ``cls`` ∈ {"surrogate", "degraded", "faithful", "sabotaged"}.  Returns the
    run-dir path (== ``tmp``).  All four share the SDAR paper identity so the
    invariant set applies.
    """
    code_dir = tmp / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    if cls == "surrogate":
        train_py = _SURROGATE_TRAIN_PY
    elif cls == "sabotaged":
        train_py = _SABOTAGED_TRAIN_PY
    else:  # faithful, degraded both use faithful code
        train_py = _FAITHFUL_TRAIN_PY
    (code_dir / "train.py").write_text(train_py, encoding="utf-8")

    # code/metrics.json — present for every class except degraded (which has no
    # measured metrics anywhere).  This feeds _gather_evidence, not the degraded
    # detector (that keys on final_report.json::baseline_metrics).
    if cls != "degraded":
        (code_dir / "metrics.json").write_text(
            json.dumps(_REAL_METRICS, indent=2), encoding="utf-8"
        )

    # final_report.json — baseline_metrics drives _is_degraded_run.
    if cls == "degraded":
        baseline_metrics: dict = {}  # empty → _is_degraded_run → True
    else:
        baseline_metrics = {"success_rate": 0.62}
    (tmp / "final_report.json").write_text(
        json.dumps(
            {
                "reproduction_summary": f"{cls} fixture",
                "baseline_metrics": baseline_metrics,
                "verdict": "reproduced",
                "paper": {"id": "2605.15155", "title": "SDAR"},
            }
        ),
        encoding="utf-8",
    )
    return tmp


def _score(run_dir: Path, client) -> float:
    """Score a fixture with the SDAR invariants (the deterministic floor)."""
    result = score_reproduction(
        TINY_TREE, run_dir, client, invariants=_sdar_invariants()
    )
    return result["overall_score"]


# ---------------------------------------------------------------------------
# Sanity: the helper actually classifies each fixture as intended.
# ---------------------------------------------------------------------------


class TestFixtureSanity:
    def test_degraded_fixture_is_degraded(self, tmp_path):
        run_dir = _write_fixture(tmp_path, "degraded")
        result = score_reproduction(
            TINY_TREE, run_dir, _NewAnchorLlm(), invariants=_sdar_invariants()
        )
        assert result["degraded"] is True

    def test_faithful_fixture_not_degraded_and_passes_invariants(self, tmp_path):
        run_dir = _write_fixture(tmp_path, "faithful")
        result = score_reproduction(
            TINY_TREE, run_dir, _NewAnchorLlm(), invariants=_sdar_invariants()
        )
        assert result["degraded"] is False
        assert result["invariant_gate_applied"] is False, (
            "faithful code must satisfy every SDAR invariant: "
            f"{[r for r in result['invariant_results'] if not r['passed']]}"
        )

    def test_surrogate_fixture_trips_hard_gate(self, tmp_path):
        run_dir = _write_fixture(tmp_path, "surrogate")
        result = score_reproduction(
            TINY_TREE, run_dir, _NewAnchorLlm(), invariants=_sdar_invariants()
        )
        assert result["invariant_gate_applied"] is True
        assert any(r["hard_gate_tripped"] for r in result["invariant_results"])

    def test_sabotaged_fixture_trips_soft_gate_only(self, tmp_path):
        run_dir = _write_fixture(tmp_path, "sabotaged")
        result = score_reproduction(
            TINY_TREE, run_dir, _NewAnchorLlm(), invariants=_sdar_invariants()
        )
        assert result["invariant_gate_applied"] is True
        # Soft trips present, no hard trip (no surrogate marker).
        assert any(r["soft_gate_tripped"] for r in result["invariant_results"])
        assert not any(r["hard_gate_tripped"] for r in result["invariant_results"])


# ---------------------------------------------------------------------------
# D7a — per-class band assertions under the NEW (anchored-to-1.0) grader.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls, predicate, label",
    [
        ("surrogate", lambda s: s == pytest.approx(INVARIANT_HARD_CAP), "== 0.0 (hard gate)"),
        ("degraded", lambda s: s <= DEGRADED_LEAF_CEILING + 1e-9, "<= 0.35 (degraded)"),
        ("faithful", lambda s: s >= 0.9, ">= 0.9 (anchored, all invariants pass)"),
        ("sabotaged", lambda s: s <= INVARIANT_SOFT_CAP + 1e-9, "<= 0.5 (soft gate)"),
    ],
)
def test_class_band_under_new_anchor(tmp_path, cls, predicate, label):
    """Each fixture class lands in its expected band under ``_NewAnchorLlm``.

    The bands are enforced by the deterministic floor, NOT the grader — even with
    the new anchor handing out ~0.97 per leaf, surrogate→0.0, degraded≤0.35,
    sabotaged≤0.5; only the faithful run is free to climb to ≥0.9.
    """
    run_dir = _write_fixture(tmp_path, cls)
    score = _score(run_dir, _NewAnchorLlm())
    assert predicate(score), f"class={cls} expected {label}, got {score}"


# ---------------------------------------------------------------------------
# D7b — the dual-run kill criterion.
# ---------------------------------------------------------------------------


def test_dual_run_kill_criterion(tmp_path):
    """The mandatory anti-inflation gate (spec §8, D7b).

    Score the FAITHFUL fixture and the SURROGATE fixture (the "fake") under BOTH
    the old (floor-anchored) and new (anchored-to-1.0) grader stubs, then assert
    ALL FOUR clauses hold:

        fake_new      <= 0.4                              # absolute fake ceiling
        fake_new      <= fake_old + 0.05                  # fakes must NOT rise
        faithful_new  >  faithful_old                     # faithful rises
        (faithful_new - fake_new) > (faithful_old - fake_old)   # spread WIDENS

    The ``fake_new <= fake_old + ε`` clause is load-bearing (Codex review
    2026-06-07): "spread widens" alone is gameable — both scores rising
    proportionally widens the spread yet lets the fake climb.  Requiring the fake
    to stay flat closes that hole.  Sound because only the per-leaf LLM scores
    change between stubs; the gate / ceiling / roll-up are deterministic Python.
    """
    faithful_dir = _write_fixture(tmp_path / "faithful", "faithful")
    fake_dir = _write_fixture(tmp_path / "fake", "surrogate")

    faithful_old = _score(faithful_dir, _OldAnchorLlm())
    faithful_new = _score(faithful_dir, _NewAnchorLlm())
    fake_old = _score(fake_dir, _OldAnchorLlm())
    fake_new = _score(fake_dir, _NewAnchorLlm())

    eps = 0.05

    assert fake_new <= 0.4, f"fake_new={fake_new} breached the absolute fake ceiling 0.4"
    assert fake_new <= fake_old + eps, (
        f"fake rose: fake_new={fake_new} > fake_old={fake_old} + {eps} "
        "(anti-gaming clause violated)"
    )
    assert faithful_new > faithful_old, (
        f"faithful did not rise: faithful_new={faithful_new} <= "
        f"faithful_old={faithful_old}"
    )
    spread_new = faithful_new - fake_new
    spread_old = faithful_old - fake_old
    assert spread_new > spread_old, (
        f"spread did not widen: new={spread_new} (faithful_new={faithful_new} - "
        f"fake_new={fake_new}) <= old={spread_old} (faithful_old={faithful_old} - "
        f"fake_old={fake_old})"
    )
