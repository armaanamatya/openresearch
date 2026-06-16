"""Tests for the deterministic paper-hint invariant gate in leaf_scorer.

The invariant gate (2026-05-29) deterministically caps ``overall_score``
independently of the LLM leaf grader when paper-hint InvariantSpec patterns
fire against the agent's code:

  * A ``must_not_match`` violation (e.g. surrogate model ``class TinyLM``)
    → hard gate → overall_score capped to INVARIANT_HARD_CAP (0.0).
  * A ``must_match`` miss (e.g. sigmoid gate absent from all .py files)
    → soft gate → overall_score capped to INVARIANT_SOFT_CAP (0.5).
  * All invariants pass → overall_score unchanged.
  * Hard gate wins over soft gate.

These tests use a synthetic rubric tree + a mock LLM that gives high scores
(0.9) so any capping is 100% attributable to the invariant gate, not the grader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.schemas import InvariantSpec
from backend.evals.paperbench.leaf_scorer import (
    INVARIANT_HARD_CAP,
    INVARIANT_SOFT_CAP,
    _apply_invariant_gate,
    run_invariant_checks,
    score_reproduction,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TINY_TREE = {
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


class _HighScoreLlm:
    """Stub LLM that gives every leaf 0.9 — any cap is from the invariant gate."""

    def complete(self, *, system: str, user: str) -> str:
        return json.dumps([
            {"leaf_id": "leaf-a1", "score": 0.9, "justification": "looks good"},
            {"leaf_id": "leaf-a2", "score": 0.9, "justification": "looks good"},
        ])


def _write_report(tmp: Path, metrics: dict | None = None) -> None:
    """Write a minimal final_report.json (non-degraded so the gate is the only cap)."""
    (tmp / "final_report.json").write_text(
        json.dumps({
            "reproduction_summary": "test",
            "baseline_metrics": metrics if metrics is not None else {"acc": 0.9},
            "verdict": "reproduced",
            "paper": {"id": "2605.15155", "title": "SDAR"},
        }),
        encoding="utf-8",
    )


def _write_train_py(code_dir: Path, content: str) -> None:
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "train.py").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# run_invariant_checks unit tests
# ---------------------------------------------------------------------------


class TestRunInvariantChecks:
    def test_empty_invariants_returns_empty(self, tmp_path):
        assert run_invariant_checks([], tmp_path) == []

    def test_missing_code_dir_returns_empty(self, tmp_path):
        inv = InvariantSpec(name="x", rationale="y", must_match=[r"sigmoid"])
        # tmp_path / "nonexistent" does not exist
        result = run_invariant_checks([inv], tmp_path / "nonexistent")
        assert result == []

    def test_must_not_match_fires_on_surrogate(self, tmp_path):
        """A file containing 'class TinyLM' trips the hard gate."""
        code_dir = tmp_path / "code"
        _write_train_py(code_dir, "class TinyLM(nn.Module):\n    pass\n")

        inv = InvariantSpec(
            name="no_surrogate",
            rationale="no surrogate",
            must_not_match=[r"class\s+TinyLM\b"],
        )
        results = run_invariant_checks([inv], code_dir)
        assert len(results) == 1
        r = results[0]
        assert r["name"] == "no_surrogate"
        assert r["hard_gate_tripped"] is True
        assert r["passed"] is False
        assert r["files_scanned"] > 0
        # The violation must carry a file:line: excerpt entry
        violations = r["must_not_match_violations"]
        assert r"class\s+TinyLM\b" in violations
        assert len(violations[r"class\s+TinyLM\b"]) > 0

    def test_must_not_match_absent_does_not_trip(self, tmp_path):
        """A clean file does not trip must_not_match."""
        code_dir = tmp_path / "code"
        _write_train_py(code_dir, "import torch\ngate = torch.sigmoid(beta * delta)\n")

        inv = InvariantSpec(
            name="no_surrogate",
            rationale="no surrogate",
            must_not_match=[r"class\s+TinyLM\b"],
        )
        results = run_invariant_checks([inv], code_dir)
        assert results[0]["hard_gate_tripped"] is False
        assert results[0]["passed"] is True

    def test_must_match_found_passes(self, tmp_path):
        """must_match that IS present does not trip the soft gate."""
        code_dir = tmp_path / "code"
        _write_train_py(code_dir, "gate = torch.sigmoid(beta * delta_t).detach()\n")

        inv = InvariantSpec(
            name="sigmoid_gate",
            rationale="gate",
            must_match=[r"(?:torch\.)?sigmoid\s*\(\s*(?:self\.)?beta\s*\*"],
        )
        results = run_invariant_checks([inv], code_dir)
        r = results[0]
        assert r["soft_gate_tripped"] is False
        assert r["passed"] is True
        # Evidence must be populated
        pat = r"(?:torch\.)?sigmoid\s*\(\s*(?:self\.)?beta\s*\*"
        assert pat in r["must_match_evidence"]
        assert len(r["must_match_evidence"][pat]) > 0

    def test_must_match_absent_trips_soft_gate(self, tmp_path):
        """must_match pattern absent from all files trips the soft gate."""
        code_dir = tmp_path / "code"
        _write_train_py(code_dir, "loss = grpo_loss + 0.1 * distill_loss\n")

        inv = InvariantSpec(
            name="sigmoid_gate",
            rationale="gate",
            must_match=[r"(?:torch\.)?sigmoid\s*\(\s*(?:self\.)?beta\s*\*"],
        )
        results = run_invariant_checks([inv], code_dir)
        r = results[0]
        assert r["soft_gate_tripped"] is True
        assert r["passed"] is False

    def test_must_match_or_semantics_either_pattern_satisfies(self, tmp_path):
        """Any one must_match pattern matching is sufficient (OR semantics).

        stop_gradient_on_gate has two patterns: .detach() OR torch.no_grad().
        Code with .detach() but not torch.no_grad() must PASS (not trip soft gate).
        Code with torch.no_grad() but not .detach() must also PASS.
        Only when NEITHER is present should the soft gate fire.
        """
        code_dir = tmp_path / "code"

        # Case 1: .detach() present — must pass
        _write_train_py(code_dir, "gate = torch.sigmoid(beta * delta).detach()\n")
        inv = InvariantSpec(
            name="stop_grad",
            rationale="stop grad",
            must_match=[r"\.detach\(\)", r"with\s+torch\.no_grad\s*\(\s*\)"],
        )
        results = run_invariant_checks([inv], code_dir)
        assert results[0]["soft_gate_tripped"] is False, (
            ".detach() present — should satisfy OR, not trip soft gate"
        )

        # Case 2: neither present — must trip soft gate
        _write_train_py(code_dir, "gate = torch.sigmoid(beta * delta)  # no stop-gradient\n")
        results = run_invariant_checks([inv], code_dir)
        assert results[0]["soft_gate_tripped"] is True, (
            "Neither .detach() nor no_grad present — should trip soft gate"
        )

    def test_files_scanned_count(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        for i in range(3):
            (code_dir / f"module{i}.py").write_text(f"# module {i}\n", encoding="utf-8")
        # Non-py file should not be counted
        (code_dir / "README.md").write_text("# readme", encoding="utf-8")

        inv = InvariantSpec(name="x", rationale="y", must_match=[r"module"])
        results = run_invariant_checks([inv], code_dir)
        assert results[0]["files_scanned"] == 3

    def test_multiple_invariants_independent(self, tmp_path):
        """Each InvariantSpec is checked independently."""
        code_dir = tmp_path / "code"
        _write_train_py(
            code_dir,
            "gate = torch.sigmoid(self.beta * delta_t).detach()\n"
            "class TinyLM(nn.Module): pass\n",
        )

        invs = [
            InvariantSpec(
                name="sigmoid_gate",
                rationale="sigmoid",
                must_match=[r"(?:torch\.)?sigmoid\s*\(\s*(?:self\.)?beta\s*\*"],
            ),
            InvariantSpec(
                name="no_surrogate",
                rationale="no stub",
                must_not_match=[r"class\s+TinyLM\b"],
            ),
        ]
        results = run_invariant_checks(invs, code_dir)
        assert len(results) == 2
        by_name = {r["name"]: r for r in results}
        assert by_name["sigmoid_gate"]["passed"] is True
        assert by_name["no_surrogate"]["hard_gate_tripped"] is True


# ---------------------------------------------------------------------------
# _apply_invariant_gate unit tests
# ---------------------------------------------------------------------------


class TestApplyInvariantGate:
    def test_no_results_no_gate(self):
        score, applied = _apply_invariant_gate(0.8, [])
        assert score == pytest.approx(0.8)
        assert applied is False

    def test_all_pass_no_gate(self):
        results = [{"hard_gate_tripped": False, "soft_gate_tripped": False}]
        score, applied = _apply_invariant_gate(0.8, results)
        assert score == pytest.approx(0.8)
        assert applied is False

    def test_hard_gate_caps_to_zero(self):
        results = [{"hard_gate_tripped": True, "soft_gate_tripped": False}]
        score, applied = _apply_invariant_gate(0.8, results)
        assert score == pytest.approx(INVARIANT_HARD_CAP)
        assert applied is True

    def test_soft_gate_caps_high_score(self):
        results = [{"hard_gate_tripped": False, "soft_gate_tripped": True}]
        score, applied = _apply_invariant_gate(0.8, results)
        assert score == pytest.approx(INVARIANT_SOFT_CAP)
        assert applied is True

    def test_soft_gate_does_not_raise_low_score(self):
        """A soft gate must not raise a score below the cap."""
        results = [{"hard_gate_tripped": False, "soft_gate_tripped": True}]
        score, applied = _apply_invariant_gate(0.2, results)
        assert score == pytest.approx(0.2)  # 0.2 < INVARIANT_SOFT_CAP, unchanged
        assert applied is True

    def test_hard_gate_wins_over_soft(self):
        """When both hard and soft trips exist, hard gate wins (score = 0.0)."""
        results = [
            {"hard_gate_tripped": True, "soft_gate_tripped": False},
            {"hard_gate_tripped": False, "soft_gate_tripped": True},
        ]
        score, applied = _apply_invariant_gate(0.8, results)
        assert score == pytest.approx(INVARIANT_HARD_CAP)
        assert applied is True


# ---------------------------------------------------------------------------
# score_reproduction integration with invariants
# ---------------------------------------------------------------------------


class TestScoreReproductionInvariantGate:
    """End-to-end: the LLM grader gives high scores; the gate caps or not."""

    def test_no_invariants_no_gate(self, tmp_path):
        """Without invariants, LLM score is returned unchanged."""
        _write_report(tmp_path)
        result = score_reproduction(
            TINY_TREE, tmp_path, _HighScoreLlm(), invariants=None
        )
        assert result["overall_score"] == pytest.approx(0.9)
        assert result["invariant_gate_applied"] is False
        assert result["invariant_results"] == []

    def test_passing_invariants_no_gate(self, tmp_path):
        """Code satisfies all invariants → LLM score unchanged."""
        _write_report(tmp_path)
        code_dir = tmp_path / "code"
        # Write code that satisfies both must_match patterns.
        _write_train_py(
            code_dir,
            "gate = torch.sigmoid(beta * delta_t).detach()\n"
            "from transformers import AutoModelForCausalLM\n"
            "model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B-Instruct')\n",
        )

        invs = [
            InvariantSpec(
                name="sigmoid_gate",
                rationale="gate",
                must_match=[r"(?:torch\.)?sigmoid\s*\(\s*(?:self\.)?beta\s*\*"],
            ),
            InvariantSpec(
                name="real_weights",
                rationale="real weights",
                must_match=[r"from_pretrained\s*\(\s*['\"]Qwen/Qwen"],
                must_not_match=[r"class\s+TinyLM\b"],
            ),
        ]
        result = score_reproduction(TINY_TREE, tmp_path, _HighScoreLlm(), invariants=invs)
        assert result["overall_score"] == pytest.approx(0.9)
        assert result["invariant_gate_applied"] is False
        # All invariant results should be passed=True
        for r in result["invariant_results"]:
            assert r["passed"] is True, f"Expected pass but got: {r}"

    def test_must_not_match_surrogate_caps_score_to_zero(self, tmp_path):
        """A surrogate model (class TinyLM) triggers the hard gate → 0.0.

        This is the primary regression test for the SDAR use-case: an agent
        that cuts corners by using a tiny surrogate instead of real Qwen weights
        must NEVER pass the rubric, regardless of what the LLM grader says.
        """
        _write_report(tmp_path)
        code_dir = tmp_path / "code"
        # Write code with a surrogate model — this MUST be detected and gated.
        _write_train_py(
            code_dir,
            "# surrogate model for testing\n"
            "class TinyLM(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.lm_head = nn.Linear(128, 50257)\n",
        )

        surrogate_inv = InvariantSpec(
            name="real_qwen_weights_not_surrogate",
            rationale="Rubric verifies real HuggingFace Qwen weights",
            must_match=[r"from_pretrained\s*\(\s*['\"]Qwen/Qwen"],
            must_not_match=[
                r"class\s+TinyLM\b",
                r"#\s*surrogate\s+model",
            ],
        )
        result = score_reproduction(
            TINY_TREE, tmp_path, _HighScoreLlm(), invariants=[surrogate_inv]
        )

        # Hard gate must fire regardless of LLM score.
        assert result["overall_score"] == pytest.approx(INVARIANT_HARD_CAP), (
            f"Expected hard gate to cap score to {INVARIANT_HARD_CAP} but got "
            f"{result['overall_score']}. Surrogate model was not detected."
        )
        assert result["invariant_gate_applied"] is True
        # The invariant result must record the violation.
        inv_res = result["invariant_results"]
        assert len(inv_res) == 1
        r = inv_res[0]
        assert r["hard_gate_tripped"] is True
        assert r["passed"] is False
        # Violations must list the offending lines.
        assert len(r["must_not_match_violations"]) > 0

    def test_must_match_miss_sigmoid_caps_to_soft_cap(self, tmp_path):
        """Code missing the sigmoid gate pattern trips the soft gate → score ≤ 0.5."""
        _write_report(tmp_path)
        code_dir = tmp_path / "code"
        # Code missing sigmoid(beta * …) but not containing a surrogate.
        _write_train_py(
            code_dir,
            "gate = torch.tanh(delta_t)  # wrong gate formula\n"
            "from transformers import AutoModelForCausalLM\n"
            "model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B')\n",
        )

        sigmoid_inv = InvariantSpec(
            name="sigmoid_gate_on_advantage",
            rationale="SDAR gate must be sigmoid(beta*delta_t)",
            must_match=[r"(?:torch\.)?sigmoid\s*\(\s*(?:self\.)?beta\s*\*"],
        )
        result = score_reproduction(
            TINY_TREE, tmp_path, _HighScoreLlm(), invariants=[sigmoid_inv]
        )

        assert result["overall_score"] <= INVARIANT_SOFT_CAP + 1e-9, (
            f"Expected soft gate to cap score at {INVARIANT_SOFT_CAP} but got "
            f"{result['overall_score']}"
        )
        assert result["invariant_gate_applied"] is True
        inv_res = result["invariant_results"]
        assert len(inv_res) == 1
        assert inv_res[0]["soft_gate_tripped"] is True

    def test_hard_gate_wins_over_soft_gate_together(self, tmp_path):
        """When both hard and soft trips exist in different invariants, 0.0 wins."""
        _write_report(tmp_path)
        code_dir = tmp_path / "code"
        # Surrogate present (hard gate) AND sigmoid absent (soft gate).
        _write_train_py(
            code_dir,
            "class TinyLM(nn.Module):\n    pass\n"
            "# no sigmoid gate anywhere\n",
        )

        invs = [
            InvariantSpec(
                name="no_surrogate",
                rationale="real weights required",
                must_not_match=[r"class\s+TinyLM\b"],
            ),
            InvariantSpec(
                name="sigmoid_gate",
                rationale="sigmoid gate required",
                must_match=[r"(?:torch\.)?sigmoid\s*\(\s*(?:self\.)?beta\s*\*"],
            ),
        ]
        result = score_reproduction(TINY_TREE, tmp_path, _HighScoreLlm(), invariants=invs)
        assert result["overall_score"] == pytest.approx(INVARIANT_HARD_CAP)
        assert result["invariant_gate_applied"] is True

    def test_invariant_results_in_score_dict(self, tmp_path):
        """invariant_results and invariant_gate_applied are always in the dict."""
        _write_report(tmp_path)
        # No code dir — run_invariant_checks returns [] when dir missing.
        invs = [
            InvariantSpec(name="x", rationale="y", must_match=[r"sigmoid"])
        ]
        result = score_reproduction(TINY_TREE, tmp_path, _HighScoreLlm(), invariants=invs)
        assert "invariant_results" in result
        assert "invariant_gate_applied" in result

    def test_invariant_results_empty_when_no_invariants(self, tmp_path):
        """invariant_results=[] and gate_applied=False when invariants=None."""
        _write_report(tmp_path)
        result = score_reproduction(TINY_TREE, tmp_path, _HighScoreLlm(), invariants=None)
        assert result["invariant_results"] == []
        assert result["invariant_gate_applied"] is False

    def test_surrogate_gate_on_degraded_run(self, tmp_path):
        """Surrogate gate fires even on a degraded (metric-less) run.

        The gate should be visible in invariant_results regardless of whether
        the run is degraded.  The degraded path short-circuits to score=0.0 so
        the gate is numerically a no-op, but invariant_results still records
        the violation so the rubric block is informative.
        """
        _write_report(tmp_path, metrics={})  # degraded run (empty metrics)
        code_dir = tmp_path / "code"
        _write_train_py(code_dir, "class TinyLM(nn.Module): pass\n")

        surrogate_inv = InvariantSpec(
            name="no_surrogate",
            rationale="real weights",
            must_not_match=[r"class\s+TinyLM\b"],
        )
        result = score_reproduction(
            TINY_TREE, tmp_path, _HighScoreLlm(), invariants=[surrogate_inv]
        )
        # Degraded → score already 0.0.
        assert result["degraded"] is True
        assert result["overall_score"] == pytest.approx(0.0)
        # But invariant_results must still record the violation.
        assert len(result["invariant_results"]) == 1
        assert result["invariant_results"][0]["hard_gate_tripped"] is True


# ---------------------------------------------------------------------------
# amend_final_report persists invariant_results
# ---------------------------------------------------------------------------


class TestAmendFinalReportInvariantGate:
    def test_amend_persists_invariant_results(self, tmp_path):
        """amend_final_report writes invariant_results to final_report.json."""
        from backend.evals.paperbench.leaf_scorer import amend_final_report

        report = {
            "verdict": "reproduced",
            "baseline_metrics": {"acc": 0.9},
            "paper": {"id": "2605.15155", "title": "SDAR"},
            "paper_claims": {},
            "rubric": {"overall_score": 0.0},
            "improvements": [],
            "primitive_trace": {"calls": 0, "by_primitive": {}},
            "cost": {"llm_usd": 0.0, "primitives": 0.0},
            "iterations": 1,
            "reproduction_summary": "",
        }
        (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

        inv_results = [
            {
                "name": "no_surrogate",
                "passed": False,
                "hard_gate_tripped": True,
                "soft_gate_tripped": False,
                "must_match_evidence": {},
                "must_not_match_violations": {
                    r"class\s+TinyLM\b": ["train.py:5: class TinyLM(nn.Module):"]
                },
                "files_scanned": 1,
                "rationale": "real weights required",
            }
        ]
        amend_final_report(
            tmp_path,
            {
                "overall_score": 0.0,
                "rubric_source": "paperbench_bundle",
                "leaf_count": 2,
                "graded": 2,
                "target_score": 0.6,
                "invariant_results": inv_results,
                "invariant_gate_applied": True,
            },
        )

        out = json.loads((tmp_path / "final_report.json").read_text(encoding="utf-8"))
        assert out["rubric"]["invariant_gate_applied"] is True
        assert len(out["rubric"]["invariant_results"]) == 1
        assert out["rubric"]["invariant_results"][0]["name"] == "no_surrogate"
        assert out["rubric"]["invariant_results"][0]["hard_gate_tripped"] is True

    def test_amend_no_invariants_writes_empty_list(self, tmp_path):
        """When no invariants were checked, amend writes empty list and False."""
        from backend.evals.paperbench.leaf_scorer import amend_final_report

        report = {
            "verdict": "reproduced",
            "baseline_metrics": {"acc": 0.9},
            "paper": {"id": "x", "title": "Y"},
            "paper_claims": {},
            "rubric": {},
            "improvements": [],
            "primitive_trace": {"calls": 0, "by_primitive": {}},
            "cost": {"llm_usd": 0.0, "primitives": 0.0},
            "iterations": 1,
            "reproduction_summary": "",
        }
        (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")
        amend_final_report(
            tmp_path,
            {
                "overall_score": 0.7,
                "rubric_source": "generated",
                "leaf_count": 2,
                "graded": 2,
                "target_score": 0.6,
                # No invariant keys — should default to [] / False.
            },
        )
        out = json.loads((tmp_path / "final_report.json").read_text(encoding="utf-8"))
        assert out["rubric"]["invariant_results"] == []
        assert out["rubric"]["invariant_gate_applied"] is False


# ---------------------------------------------------------------------------
# SDAR full-invariant-set smoke test
# ---------------------------------------------------------------------------


class TestSdarInvariantSet:
    """Smoke tests against the full SDAR InvariantSpec list from paper_hints.py."""

    @staticmethod
    def _sdar_invariants() -> list[InvariantSpec]:
        from backend.agents.prompts.paper_hints import lookup_paper_hint
        hint = lookup_paper_hint("2605.15155")
        assert hint is not None, "SDAR paper hint not found in PAPER_HINTS"
        return hint.invariants

    def test_sdar_all_pass_on_correct_code(self, tmp_path):
        """Code implementing SDAR correctly passes all 6 invariants.

        Variable names are chosen to match the InvariantSpec must_match patterns:
          - lambda = 0.1   (pattern: lambda[:=]0.1)
          - beta = 10      (pattern: beta[:=]10)
          - torch.sigmoid(beta * …) for sigmoid gate
          - .detach()      for stop-gradient (OR torch.no_grad — either satisfies)
          - grpo_loss + opsd_loss for combined loss
          - from_pretrained('Qwen/…') for real weights
        """
        code_dir = tmp_path / "code"
        _write_train_py(
            code_dir,
            "import torch\n"
            "from transformers import AutoModelForCausalLM\n"
            "\n"
            "# Hyper-parameters (paper §3.2)\n"
            "lambda = 0.1   # self-distillation weight (eq. 5)\n"
            "beta = 10      # gate sharpness\n"
            "\n"
            "model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B-Instruct')\n"
            "\n"
            "def compute_loss(advantages, delta_t, teacher_logprobs, student_logprobs):\n"
            "    # GRPO loss\n"
            "    grpo_loss = -(advantages * student_logprobs).mean()\n"
            "    # OPSD self-distillation gate (sigmoid gate, stop-gradient)\n"
            "    gate = torch.sigmoid(beta * delta_t).detach()  # stop-gradient\n"
            "    opsd_loss = -(gate * teacher_logprobs).mean()\n"
            "    return grpo_loss + lambda * opsd_loss\n",
        )

        results = run_invariant_checks(self._sdar_invariants(), code_dir)
        failures = [r for r in results if not r["passed"]]
        assert failures == [], (
            f"Expected all SDAR invariants to pass on correct code, "
            f"but these failed: {[f['name'] for f in failures]}\n"
            f"Details: {[{k: v for k, v in f.items() if k != 'must_match_evidence'} for f in failures]}"
        )

    def test_sdar_surrogate_model_trips_hard_gate(self, tmp_path):
        """SDAR surrogate model detection: class TinyLM trips must_not_match."""
        code_dir = tmp_path / "code"
        _write_train_py(
            code_dir,
            "import torch\n"
            "# surrogate model — avoid downloading real Qwen weights\n"
            "class TinyLM(torch.nn.Module):\n"
            "    def __init__(self): super().__init__()\n"
            "\n"
            "lambda_ = 0.1\n"
            "beta = 10\n"
            "gate = torch.sigmoid(beta * delta).detach()\n"
            "grpo_loss = -advantages.mean()\n"
            "opsd_loss = -gate.mean()\n",
        )

        results = run_invariant_checks(self._sdar_invariants(), code_dir)
        # The real_qwen_weights_not_surrogate invariant must trip hard.
        surrogate_result = next(
            (r for r in results if r["name"] == "real_qwen_weights_not_surrogate"),
            None,
        )
        assert surrogate_result is not None
        assert surrogate_result["hard_gate_tripped"] is True, (
            "SDAR must_not_match for 'class TinyLM' did not fire"
        )

    def test_sdar_score_gated_to_zero_on_surrogate(self, tmp_path):
        """Full integration: SDAR surrogate model → overall_score == 0.0."""
        _write_report(tmp_path)
        code_dir = tmp_path / "code"
        _write_train_py(
            code_dir,
            "class TinyLM(torch.nn.Module): pass\n"
            "lambda_ = 0.1\nbeta = 10\n"
            "grpo_loss = grpo()\nopsd_loss = opsd()\n"
            "gate = torch.sigmoid(beta * delta).detach()\n",
        )

        invs = self._sdar_invariants()
        result = score_reproduction(TINY_TREE, tmp_path, _HighScoreLlm(), invariants=invs)

        assert result["overall_score"] == pytest.approx(INVARIANT_HARD_CAP), (
            f"SDAR surrogate model must cap overall_score to {INVARIANT_HARD_CAP}, "
            f"got {result['overall_score']}"
        )
        assert result["invariant_gate_applied"] is True
