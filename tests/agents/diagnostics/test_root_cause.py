"""P4 (2026-05-29): root_cause_signature for the same-failure-twice rule."""

from __future__ import annotations

import pytest

from backend.agents.diagnostics import ROOT_CAUSE_UNKNOWN, root_cause_signature


class TestRunExperimentFailureClass:
    def test_basic_failure_class(self) -> None:
        sig = root_cause_signature({"failure_class": "cuda_oom"})
        assert sig == "run_experiment:cuda_oom"

    def test_failure_class_normalized_lower(self) -> None:
        sig = root_cause_signature({"failure_class": "CUDA_OOM"})
        assert sig == "run_experiment:cuda_oom"

    def test_failure_class_wins_over_primitive(self) -> None:
        """When both fields are present, run_experiment failure_class wins
        (most specific signal)."""
        sig = root_cause_signature({
            "failure_class": "dockerfile_invalid",
            "primitive": "build_environment",
            "error_code": "parse_error",
        })
        assert sig == "run_experiment:dockerfile_invalid"

    def test_real_failure_classes_from_sdar(self) -> None:
        for fc in (
            "cuda_oom",
            "commands_missing_file",
            "dockerfile_invalid",
            "preflight_blocked",
            "runpod_capacity",
        ):
            assert root_cause_signature({"failure_class": fc}) == f"run_experiment:{fc}"


class TestPrimitiveErrorCode:
    def test_primitive_with_error_code(self) -> None:
        sig = root_cause_signature({
            "primitive": "implement_baseline",
            "error_code": "sdk_pre_emit_stall",
        })
        assert sig == "implement_baseline:sdk_pre_emit_stall"

    def test_primitive_alone_falls_through(self) -> None:
        """Missing error_code → can't form a primitive signature → falls to
        next tier."""
        sig = root_cause_signature({"primitive": "implement_baseline"})
        assert sig == ROOT_CAUSE_UNKNOWN

    def test_real_primitive_failures(self) -> None:
        for primitive, code in (
            ("implement_baseline", "sdk_pre_emit_stall"),
            ("implement_baseline", "sdk_aclose_incomplete_artifacts"),
            ("implement_baseline", "commands_missing_file"),
            ("build_environment", "dockerfile_parse_error"),
            ("plan_reproduction", "envelope_failed"),
        ):
            sig = root_cause_signature({"primitive": primitive, "error_code": code})
            assert sig == f"{primitive}:{code}"


class TestExceptionClassMessage:
    def test_exception_with_message(self) -> None:
        sig = root_cause_signature({
            "exception_class": "RuntimeError",
            "error": "credit balance too low",
        })
        assert sig == "RuntimeError:credit balance too low"

    def test_first_line_only(self) -> None:
        sig = root_cause_signature({
            "exception_class": "ValueError",
            "message": "first line\nsecond line\nthird",
        })
        assert sig == "ValueError:first line"

    def test_message_truncated_at_120(self) -> None:
        long_msg = "x" * 500
        sig = root_cause_signature({
            "exception_class": "RuntimeError",
            "error": long_msg,
        })
        # Total length cap: "RuntimeError:" + 120 chars
        assert len(sig) == len("RuntimeError:") + 120

    def test_strips_ansi_codes(self) -> None:
        sig = root_cause_signature({
            "exception_class": "RuntimeError",
            "error": "\x1b[31mError\x1b[0m: thing failed",
        })
        assert "\x1b" not in sig
        assert "Error: thing failed" in sig

    def test_strips_quotes(self) -> None:
        sig = root_cause_signature({
            "exception_class": "ValueError",
            "error": '"quoted message"',
        })
        assert sig == "ValueError:quoted message"

    def test_no_message_fallback(self) -> None:
        sig = root_cause_signature({"exception_class": "KeyError"})
        assert sig == "KeyError:<no-message>"


class TestFallback:
    def test_empty_dict(self) -> None:
        assert root_cause_signature({}) == ROOT_CAUSE_UNKNOWN

    def test_non_dict(self) -> None:
        assert root_cause_signature("not a dict") == ROOT_CAUSE_UNKNOWN  # type: ignore[arg-type]
        assert root_cause_signature(None) == ROOT_CAUSE_UNKNOWN  # type: ignore[arg-type]

    def test_unrelated_keys_only(self) -> None:
        assert root_cause_signature({"foo": "bar", "iteration": 3}) == ROOT_CAUSE_UNKNOWN

    def test_whitespace_only_failure_class(self) -> None:
        """Whitespace-only string should be treated as absent."""
        assert root_cause_signature({"failure_class": "  "}) == ROOT_CAUSE_UNKNOWN


class TestSignatureComparability:
    """The whole point of P4 is two attempts comparing equal when they
    failed for the same root cause. These assert signatures are stable
    across cosmetic message differences."""

    def test_two_cuda_oom_attempts_compare_equal(self) -> None:
        sig1 = root_cause_signature({"failure_class": "cuda_oom"})
        sig2 = root_cause_signature({"failure_class": "cuda_oom"})
        assert sig1 == sig2

    def test_two_sdk_stall_attempts_compare_equal(self) -> None:
        sig1 = root_cause_signature({
            "primitive": "implement_baseline",
            "error_code": "sdk_pre_emit_stall",
        })
        sig2 = root_cause_signature({
            "primitive": "implement_baseline",
            "error_code": "sdk_pre_emit_stall",
        })
        assert sig1 == sig2

    def test_distinct_failures_do_not_collide(self) -> None:
        """SDAR ran 7 attempts with 7 distinct root causes — those must
        all have distinct signatures or the rule fires falsely."""
        sigs = {
            root_cause_signature({"failure_class": "cuda_oom"}),
            root_cause_signature({"failure_class": "dockerfile_invalid"}),
            root_cause_signature({"primitive": "implement_baseline", "error_code": "sdk_pre_emit_stall"}),
            root_cause_signature({"primitive": "implement_baseline", "error_code": "oauth_refusal"}),
            root_cause_signature({"exception_class": "RuntimeError", "error": "credit balance too low"}),
        }
        assert len(sigs) == 5, "5 distinct attempts must produce 5 distinct signatures"
