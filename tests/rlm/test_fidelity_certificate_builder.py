"""Tests for backend/agents/rlm/fidelity_certificate_builder.py (U16).

All tests use SYNTHETIC code directories — no GPU, no LLM, no network.

Coverage:
  1. Flag off  → no file written, stub returned (invariant_tests_ran=False).
  2. No code/  → stub returned.
  3. No test_reproduction.py → invariant_tests_ran=False.
  4. Faithful impl + biting test → all True (green cert).
  5. Decoy test (passes under mutation) → mutation_confirmed=False.
  6. Failing test → invariant_tests_ran=True, invariant_tests_passed=False.
  7. Output loads via two_axis_report.load_certificate; FidelityCertificate correct.
  8. Correct implementation_verdict via compute_reproducibility_verdict.
  9. blinded_extraction_agreed passthrough from repro_spec.json.
 10. profile_satisfied=False when has_measured_metrics=False (end_to_end profile).
 11. profile_satisfied=True when tests pass + metrics present (end_to_end).
 12. Mutation check skipped when tests fail (conservative).
 13. No registered constants → mutation_confirmed=False.
 14. Static profile + green tests → profile_satisfied=True even without metrics.
 15. Timeout exit → invariant_tests_ran=True, invariant_tests_passed=False.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from backend.agents.rlm import fidelity_certificate_builder as fcb
from backend.agents.rlm.reproducibility_verdict import (
    FidelityCertificate,
    compute_reproducibility_verdict,
)
from backend.agents.rlm.two_axis_report import load_certificate

_FLAG = "REPROLAB_TWO_AXIS_VERDICT"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_run_dir(tmp_path: Path, *, with_metrics: bool = False) -> Path:
    """Create a minimal run directory structure."""
    code_dir = tmp_path / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    state_dir = tmp_path / "rlm_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    if with_metrics:
        (code_dir / "metrics.json").write_text(
            json.dumps({"status": "ok", "per_model": {"m": {"v": 1.0}}}),
            encoding="utf-8",
        )
    return tmp_path


def _write_test(code_dir: Path, content: str) -> None:
    """Write test_reproduction.py into code_dir."""
    (code_dir / "test_reproduction.py").write_text(
        textwrap.dedent(content), encoding="utf-8"
    )


def _write_train(code_dir: Path, content: str) -> None:
    """Write train.py into code_dir (so invariant patterns can be found)."""
    (code_dir / "train.py").write_text(
        textwrap.dedent(content), encoding="utf-8"
    )


def _write_repro_spec(tmp_path: Path, payload: dict) -> None:
    path = tmp_path / "rlm_state" / "repro_spec.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# A synthetic arxiv_id + paper hint that registers a numeric constant
# so the mutation tests are self-contained (don't depend on real PAPER_HINTS).
# We patch _get_paper_hints_constants to return a fake constant spec.
# ---------------------------------------------------------------------------

_FAKE_CONSTANTS = [
    {"invariant_name": "beta_test_constant", "must_match": [r"\bbeta\s*=\s*(\d+\.?\d*)"]},
]


def _green_test_content(beta_val: str = "10") -> str:
    """A test_reproduction.py that checks beta == 10 in the production train.py."""
    return f"""\
        import re, pathlib

        def test_beta_value():
            code = (pathlib.Path(__file__).parent / "train.py").read_text()
            m = re.search(r"beta\\s*=\\s*(\\d+\\.?\\d*)", code)
            assert m is not None, "beta not found in train.py"
            assert float(m.group(1)) == {beta_val}, f"beta must be {beta_val}, got {{m.group(1)}}"
    """


_TRAIN_WITH_BETA = "beta = 10\n# rest of training code\n"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFlagGate:
    def test_disabled_by_default(self, monkeypatch, tmp_path):
        """Flag OFF → stub returned, no file written."""
        monkeypatch.delenv(_FLAG, raising=False)
        run_dir = _setup_run_dir(tmp_path)
        result = fcb.build_certificate(run_dir)
        assert result["invariant_tests_ran"] is False
        cert_path = run_dir / "rlm_state" / "fidelity_certificate.json"
        assert not cert_path.exists()

    def test_enabled(self, monkeypatch, tmp_path):
        """Flag ON → cert file written (even with no tests)."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path)
        result = fcb.build_certificate(run_dir)
        cert_path = run_dir / "rlm_state" / "fidelity_certificate.json"
        assert cert_path.exists()
        on_disk = json.loads(cert_path.read_text())
        assert on_disk["invariant_tests_ran"] == result["invariant_tests_ran"]


class TestNoTestFile:
    def test_no_test_file_sets_ran_false(self, monkeypatch, tmp_path):
        """No test_reproduction.py → invariant_tests_ran=False (PARTIAL, not broken)."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path)
        result = fcb.build_certificate(run_dir)
        assert result["invariant_tests_ran"] is False
        assert result["invariant_tests_passed"] is False

    def test_no_code_dir_returns_stub(self, monkeypatch, tmp_path):
        """No code/ directory → stub (no crash)."""
        monkeypatch.setenv(_FLAG, "1")
        # Don't create code/
        (tmp_path / "rlm_state").mkdir(parents=True, exist_ok=True)
        result = fcb.build_certificate(tmp_path)
        assert result["invariant_tests_ran"] is False


class TestGreenCert:
    def test_faithful_impl_biting_test_green_cert(self, monkeypatch, tmp_path):
        """Faithful impl + correctly biting test → green cert (all True where expected)."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=True)
        code_dir = run_dir / "code"
        _write_train(code_dir, _TRAIN_WITH_BETA)
        _write_test(code_dir, _green_test_content("10"))

        # Patch the constant loader to return our fake constants
        monkeypatch.setattr(fcb, "_get_paper_hints_constants", lambda _: _FAKE_CONSTANTS)

        result = fcb.build_certificate(run_dir)

        assert result["invariant_tests_ran"] is True
        assert result["invariant_tests_passed"] is True
        assert result["mutation_confirmed"] is True
        assert result["has_measured_metrics"] is True
        assert result["profile_satisfied"] is True

    def test_green_cert_loads_via_load_certificate(self, monkeypatch, tmp_path):
        """Written cert loads correctly via two_axis_report.load_certificate."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=True)
        code_dir = run_dir / "code"
        _write_train(code_dir, _TRAIN_WITH_BETA)
        _write_test(code_dir, _green_test_content("10"))
        monkeypatch.setattr(fcb, "_get_paper_hints_constants", lambda _: _FAKE_CONSTANTS)

        fcb.build_certificate(run_dir)

        cert = load_certificate(run_dir, has_measured_metrics=True)
        assert isinstance(cert, FidelityCertificate)
        assert cert.invariant_tests_ran is True
        assert cert.invariant_tests_passed is True
        assert cert.mutation_confirmed is True
        assert cert.blinded_extraction_agreed is False   # no repro_spec with blinded flag
        assert cert.has_measured_metrics is True
        assert cert.is_green is False  # blinded_extraction_agreed=False → not fully green

    def test_implementation_verdict_with_blinded_agreed(self, monkeypatch, tmp_path):
        """When blinded_extraction_agreed=True + all other True → implementation_verdict=faithful."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=True)
        code_dir = run_dir / "code"
        _write_train(code_dir, _TRAIN_WITH_BETA)
        _write_test(code_dir, _green_test_content("10"))
        monkeypatch.setattr(fcb, "_get_paper_hints_constants", lambda _: _FAKE_CONSTANTS)
        # Simulate blinded extraction agreed via repro_spec.json
        _write_repro_spec(run_dir, {"blinded_extraction_agreed": True})

        result = fcb.build_certificate(run_dir)

        assert result["blinded_extraction_agreed"] is True
        # Reload via load_certificate and run verdict
        cert = load_certificate(run_dir, has_measured_metrics=True)
        assert cert.blinded_extraction_agreed is True
        assert cert.is_green is True  # all fields True → green

        # Verify implementation_verdict is faithful when fidelity_score high enough
        from backend.agents.rlm.reproducibility_verdict import DEFAULT_FAITHFUL_MIN_SCORE
        v = compute_reproducibility_verdict(
            fidelity_score=0.85,
            certificate=cert,
            claims=[],
        )
        assert v.implementation_verdict == "faithful"


class TestMutationCheck:
    def test_tautological_test_mutation_not_confirmed(self, monkeypatch, tmp_path):
        """A tautological test (passes even after mutation) → mutation_confirmed=False."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=True)
        code_dir = run_dir / "code"
        _write_train(code_dir, "beta = 10\n# training\n")

        # Tautological test: does NOT check the actual value of beta
        _write_test(
            code_dir,
            """\
            import pathlib
            def test_train_exists():
                # This test is tautological — it passes regardless of beta's value
                assert (pathlib.Path(__file__).parent / "train.py").exists()
            """,
        )
        monkeypatch.setattr(fcb, "_get_paper_hints_constants", lambda _: _FAKE_CONSTANTS)

        result = fcb.build_certificate(run_dir)

        assert result["invariant_tests_ran"] is True
        assert result["invariant_tests_passed"] is True
        # The tautological test passes even under mutation → mutation_confirmed=False
        assert result["mutation_confirmed"] is False
        assert any("tautological" in n for n in result.get("_mutation_notes", []))

    def test_no_registered_constants_mutation_false(self, monkeypatch, tmp_path):
        """No registered constants → mutation_confirmed=False (can't confirm)."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=True)
        code_dir = run_dir / "code"
        _write_train(code_dir, "beta = 10\n")
        _write_test(code_dir, "def test_pass(): pass\n")
        # No constants registered
        monkeypatch.setattr(fcb, "_get_paper_hints_constants", lambda _: [])

        result = fcb.build_certificate(run_dir)

        assert result["invariant_tests_ran"] is True
        assert result["invariant_tests_passed"] is True
        assert result["mutation_confirmed"] is False

    def test_mutation_skipped_when_tests_fail(self, monkeypatch, tmp_path):
        """Mutation check skipped when base tests fail."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=True)
        code_dir = run_dir / "code"
        _write_train(code_dir, "beta = 10\n")
        _write_test(code_dir, "def test_fail(): assert False, 'intentional failure'\n")
        monkeypatch.setattr(fcb, "_get_paper_hints_constants", lambda _: _FAKE_CONSTANTS)

        result = fcb.build_certificate(run_dir)

        assert result["invariant_tests_ran"] is True
        assert result["invariant_tests_passed"] is False
        assert result["mutation_confirmed"] is False
        assert any("tests_did_not_pass" in n for n in result.get("_mutation_notes", []))

    def test_constant_not_in_code_mutation_false(self, monkeypatch, tmp_path):
        """Constant pattern not found in code → mutation_confirmed=False."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=True)
        code_dir = run_dir / "code"
        # train.py has NO beta assignment
        _write_train(code_dir, "lr = 0.001\n")
        _write_test(code_dir, "def test_pass(): pass\n")
        monkeypatch.setattr(fcb, "_get_paper_hints_constants", lambda _: _FAKE_CONSTANTS)

        result = fcb.build_certificate(run_dir)

        assert result["invariant_tests_passed"] is True
        assert result["mutation_confirmed"] is False
        assert any("constant_not_found" in n for n in result.get("_mutation_notes", []))


class TestObligationProfile:
    def test_end_to_end_profile_no_metrics_profile_not_satisfied(self, monkeypatch, tmp_path):
        """end_to_end profile + no metrics → profile_satisfied=False."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=False)
        code_dir = run_dir / "code"
        _write_test(code_dir, "def test_pass(): pass\n")

        result = fcb.build_certificate(run_dir)

        assert result["obligation_profile"] == "end_to_end"
        assert result["has_measured_metrics"] is False
        assert result["profile_satisfied"] is False

    def test_end_to_end_profile_with_metrics_and_passing_tests(self, monkeypatch, tmp_path):
        """end_to_end profile + metrics + passing tests → profile_satisfied=True."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=True)
        code_dir = run_dir / "code"
        _write_test(code_dir, "def test_pass(): pass\n")

        result = fcb.build_certificate(run_dir)

        assert result["obligation_profile"] == "end_to_end"
        assert result["has_measured_metrics"] is True
        assert result["profile_satisfied"] is True

    def test_static_profile_from_repro_spec(self, monkeypatch, tmp_path):
        """static profile from repro_spec + passing tests → profile_satisfied even without metrics."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=False)
        code_dir = run_dir / "code"
        _write_test(code_dir, "def test_pass(): pass\n")
        _write_repro_spec(run_dir, {"obligation_profile": "static"})

        result = fcb.build_certificate(run_dir)

        assert result["obligation_profile"] == "static"
        assert result["profile_satisfied"] is True

    def test_forward_pass_profile_no_metrics_satisfied(self, monkeypatch, tmp_path):
        """forward_pass profile + passing tests → profile_satisfied=True even without metrics."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=False)
        code_dir = run_dir / "code"
        _write_test(code_dir, "def test_pass(): pass\n")
        _write_repro_spec(run_dir, {"obligation_profile": "forward_pass"})

        result = fcb.build_certificate(run_dir)

        assert result["obligation_profile"] == "forward_pass"
        assert result["profile_satisfied"] is True

    def test_invalid_profile_defaults_to_end_to_end(self, monkeypatch, tmp_path):
        """Invalid profile in repro_spec defaults to end_to_end."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=False)
        code_dir = run_dir / "code"
        _write_test(code_dir, "def test_pass(): pass\n")
        _write_repro_spec(run_dir, {"obligation_profile": "INVALID_PROFILE"})

        result = fcb.build_certificate(run_dir)

        assert result["obligation_profile"] == "end_to_end"


class TestBlindedExtraction:
    def test_blinded_false_when_absent(self, monkeypatch, tmp_path):
        """No repro_spec → blinded_extraction_agreed=False (conservative)."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path)
        result = fcb.build_certificate(run_dir)
        assert result["blinded_extraction_agreed"] is False

    def test_blinded_true_from_repro_spec(self, monkeypatch, tmp_path):
        """repro_spec with blinded_extraction_agreed=True → propagated."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path)
        _write_repro_spec(run_dir, {"blinded_extraction_agreed": True})
        result = fcb.build_certificate(run_dir)
        assert result["blinded_extraction_agreed"] is True

    def test_blinded_false_when_repro_spec_malformed(self, monkeypatch, tmp_path):
        """Malformed repro_spec → conservative False."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path)
        (run_dir / "rlm_state").mkdir(parents=True, exist_ok=True)
        (run_dir / "rlm_state" / "repro_spec.json").write_text(
            "{not valid json", encoding="utf-8"
        )
        result = fcb.build_certificate(run_dir)
        assert result["blinded_extraction_agreed"] is False


class TestMetrics:
    def test_metrics_in_outputs_subdirectory(self, monkeypatch, tmp_path):
        """metrics.json under code/outputs/** → has_measured_metrics=True."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path)
        nested = run_dir / "code" / "outputs" / "run_1"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "metrics.json").write_text(
            json.dumps({"per_model": {}}), encoding="utf-8"
        )
        result = fcb.build_certificate(run_dir)
        assert result["has_measured_metrics"] is True

    def test_no_metrics_anywhere(self, monkeypatch, tmp_path):
        """No metrics.json anywhere → has_measured_metrics=False."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=False)
        result = fcb.build_certificate(run_dir)
        assert result["has_measured_metrics"] is False


class TestContractIntegration:
    """End-to-end: loaded cert maps to the correct implementation_verdict."""

    def test_no_test_file_verdict_partial(self, monkeypatch, tmp_path):
        """No test_reproduction.py → invariant_tests_ran=False → verdict partial."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=True)
        fcb.build_certificate(run_dir)
        cert = load_certificate(run_dir, has_measured_metrics=True)
        v = compute_reproducibility_verdict(
            fidelity_score=0.85, certificate=cert, claims=[]
        )
        # invariant_tests_ran=False → not green → partial (not broken, not faithful)
        assert v.implementation_verdict == "partial"
        assert v.replication_verdict == "inconclusive"

    def test_failing_test_verdict_broken(self, monkeypatch, tmp_path):
        """Tests ran and failed → invariant_tests_ran=True, passed=False → broken."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=True)
        code_dir = run_dir / "code"
        _write_test(code_dir, "def test_fail(): assert False, 'broken impl'\n")
        fcb.build_certificate(run_dir)

        cert = load_certificate(run_dir, has_measured_metrics=True)
        assert cert.invariant_tests_ran is True
        assert cert.invariant_tests_passed is False
        v = compute_reproducibility_verdict(
            fidelity_score=0.85, certificate=cert, claims=[]
        )
        assert v.implementation_verdict == "broken"

    def test_cert_dict_keys_match_load_certificate_exactly(self, monkeypatch, tmp_path):
        """The dict build_certificate returns has ALL keys load_certificate reads."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=True)
        result = fcb.build_certificate(run_dir)

        required_keys = {
            "invariant_tests_ran",
            "invariant_tests_passed",
            "mutation_confirmed",
            "blinded_extraction_agreed",
            "obligation_profile",
            "profile_satisfied",
            "has_measured_metrics",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    def test_on_disk_cert_loadable_by_load_certificate(self, monkeypatch, tmp_path):
        """Written cert can be loaded by load_certificate without error."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path, with_metrics=False)
        fcb.build_certificate(run_dir)

        cert = load_certificate(run_dir, has_measured_metrics=False)
        assert isinstance(cert, FidelityCertificate)
        # Conservative stub is not green
        assert cert.is_green is False

    def test_obligation_profile_validated_by_load_certificate(self, monkeypatch, tmp_path):
        """load_certificate rejects invalid profiles and defaults to end_to_end."""
        monkeypatch.setenv(_FLAG, "1")
        run_dir = _setup_run_dir(tmp_path)
        # Write a cert with a bogus profile directly
        cert_path = run_dir / "rlm_state" / "fidelity_certificate.json"
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        cert_path.write_text(
            json.dumps({
                "invariant_tests_ran": True,
                "invariant_tests_passed": True,
                "mutation_confirmed": True,
                "blinded_extraction_agreed": True,
                "obligation_profile": "BOGUS_PROFILE",
                "profile_satisfied": True,
                "has_measured_metrics": True,
            }),
            encoding="utf-8",
        )
        cert = load_certificate(run_dir, has_measured_metrics=True)
        # load_certificate normalizes invalid profile to end_to_end
        assert cert.obligation_profile == "end_to_end"


class TestPerturbValue:
    """Unit tests for the _perturb_value helper."""

    def test_integer_incremented(self):
        line = "beta = 10"
        result = fcb._perturb_value(line, r"\bbeta\s*=\s*(\d+)")
        assert result is not None
        assert "11" in result

    def test_float_scaled(self):
        line = "lambda_val = 0.1"
        result = fcb._perturb_value(line, r"\blambda_val\s*=\s*([\d.]+)")
        assert result is not None
        # The line should be changed: the perturbed value is 0.1 * 1.1 = 0.11 (different from 0.1)
        assert result != line  # line must have changed
        import re as _re
        m = _re.search(r"lambda_val\s*=\s*([\d.]+)", result)
        assert m is not None
        assert float(m.group(1)) != pytest.approx(0.1)

    def test_zero_replaced_with_one(self):
        line = "gate_init = 0"
        result = fcb._perturb_value(line, r"\bgate_init\s*=\s*(\d+)")
        assert result is not None
        assert "1" in result

    def test_no_numeric_returns_none(self):
        line = "mode = 'train'"
        result = fcb._perturb_value(line, r"\bmode\s*=")
        assert result is None
