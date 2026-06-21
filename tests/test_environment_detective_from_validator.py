"""Tests for the static FROM-base validator in environment_detective.

Covers:
  - Malformed / empty FROM → ok=False
  - Known-good base families → ok=True, no warning
  - Unknown / hallucinated base → ok=True (fail-soft), warning + suggested_image
  - Devel-vs-runtime hint for CUDA-compile requirements
  - Plain CPU requirements → no devel hint
  - Helpers: _extract_from_image, _is_known_good_base, _requirements_need_devel

Pure unit tests — no Docker, no network, no filesystem I/O.
"""

from __future__ import annotations

import pytest

from backend.agents.environment_detective import (
    _RUNPOD_PYTORCH_BASE,
    _extract_from_image,
    _is_known_good_base,
    _requirements_need_devel,
    _suggest_devel_base,
    validate_from_base,
)


# ---------------------------------------------------------------------------
# _extract_from_image
# ---------------------------------------------------------------------------


class TestExtractFromImage:
    def test_simple_from(self):
        assert _extract_from_image("FROM python:3.11-slim\n") == "python:3.11-slim"

    def test_from_with_as_stage(self):
        assert _extract_from_image("FROM ubuntu:22.04 AS builder") == "ubuntu:22.04"

    def test_skips_comment_lines(self):
        df = "# comment\n# another\nFROM nvidia/cuda:12.1.1-runtime-ubuntu22.04\n"
        assert _extract_from_image(df) == "nvidia/cuda:12.1.1-runtime-ubuntu22.04"

    def test_skips_arg_before_from(self):
        df = "ARG BASE=latest\nFROM runpod/pytorch:2.1.0\n"
        assert _extract_from_image(df) == "runpod/pytorch:2.1.0"

    def test_empty_dockerfile_returns_none(self):
        assert _extract_from_image("") is None

    def test_comments_only_returns_none(self):
        assert _extract_from_image("# only comments\n# no FROM\n") is None

    def test_malformed_from_no_image_returns_none(self):
        # "FROM" with nothing after — malformed
        assert _extract_from_image("FROM\n") is None

    def test_skips_blank_lines(self):
        df = "\n\n   \nFROM python:3.10-slim\n"
        assert _extract_from_image(df) == "python:3.10-slim"

    def test_case_insensitive_from(self):
        # Docker FROM is case-insensitive
        assert _extract_from_image("from python:3.11-slim") == "python:3.11-slim"


# ---------------------------------------------------------------------------
# _is_known_good_base
# ---------------------------------------------------------------------------


class TestIsKnownGoodBase:
    @pytest.mark.parametrize(
        "image",
        [
            "python:3.11-slim",
            "python:3.10",
            "ubuntu:22.04",
            "ubuntu:focal",
            "debian:bullseye-slim",
            "debian:12",
            "nvidia/cuda:12.1.1-runtime-ubuntu22.04",
            "nvidia/cuda:11.8.0-devel-ubuntu20.04",
            "nvcr.io/nvidia/pytorch:24.01-py3",
            "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel",
            "tensorflow/tensorflow:2.15.0-gpu",
            "rocm/pytorch:latest",
            "continuumio/miniconda3",
            "mambaorg/micromamba:latest",
            "quay.io/biocontainers/python:3.11",
            "gcr.io/distroless/python3",
            "us-central1-docker.pkg.dev/my-project/images/base:v1",
            "scratch",
            "busybox",
            "alpine:3.19",
        ],
    )
    def test_known_good_accepted(self, image: str):
        assert _is_known_good_base(image) is True

    @pytest.mark.parametrize(
        "image",
        [
            "hallucinated/image:v99",
            "totally-made-up:latest",
            "runpod-fake/pytorch:2.0",  # not runpod/ prefix
            "my-registry.internal/torch:latest",
        ],
    )
    def test_unknown_not_accepted(self, image: str):
        assert _is_known_good_base(image) is False

    def test_arg_interpolated_trusted(self):
        # $VAR and ${VAR} forms are treated as trusted
        assert _is_known_good_base("$BASE_IMAGE") is True
        assert _is_known_good_base("${BASE}") is True

    def test_empty_string_not_accepted(self):
        assert _is_known_good_base("") is False


# ---------------------------------------------------------------------------
# _requirements_need_devel
# ---------------------------------------------------------------------------


class TestRequirementsNeedDevel:
    @pytest.mark.parametrize(
        "text",
        [
            "bitsandbytes>=0.41.0\ntorch>=2.0",
            "flash-attn==2.3.0",
            "deepspeed>=0.12",
            "apex",
            "xformers",
            "triton==2.2.0",
        ],
    )
    def test_cuda_compile_pkg_triggers_devel(self, text: str):
        assert _requirements_need_devel(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "torch==2.2.0\nnumpy==1.26.4\nmatplotlib==3.8.0",
            "gymnasium>=0.29.1\ntqdm>=4.66.0",
            "",
            "transformers==4.40.0\naccelerater==0.30.0",
        ],
    )
    def test_plain_cpu_requirements_no_devel(self, text: str):
        assert _requirements_need_devel(text) is False

    def test_case_insensitive_match(self):
        # Package names in requirements may appear in mixed case
        assert _requirements_need_devel("Flash-Attn==2.0") is True
        assert _requirements_need_devel("BitsAndBytes>=0.41") is True


# ---------------------------------------------------------------------------
# _suggest_devel_base
# ---------------------------------------------------------------------------


class TestSuggestDevelBase:
    def test_runtime_to_devel(self):
        base = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04"
        result = _suggest_devel_base(base)
        assert "-devel-" in result
        assert "-runtime-" not in result

    def test_devel_unchanged(self):
        base = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
        assert _suggest_devel_base(base) == base

    def test_no_runtime_substring_unchanged(self):
        base = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel"
        assert _suggest_devel_base(base) == base

    def test_nvidia_cuda_runtime_to_devel(self):
        base = "nvidia/cuda:12.1.1-runtime-ubuntu22.04"
        result = _suggest_devel_base(base)
        assert "devel" in result


# ---------------------------------------------------------------------------
# validate_from_base — core scenarios
# ---------------------------------------------------------------------------


class TestValidateFromBase:
    # (a) malformed / empty FROM

    def test_empty_dockerfile_is_not_ok(self):
        result = validate_from_base("")
        assert result.ok is False
        assert result.warning is not None
        assert result.image is None

    def test_comments_only_is_not_ok(self):
        result = validate_from_base("# just a comment\n")
        assert result.ok is False
        assert result.image is None

    def test_malformed_from_no_image_is_not_ok(self):
        result = validate_from_base("FROM\n")
        assert result.ok is False
        assert result.image is None

    def test_suggested_image_is_fallback_for_empty(self):
        fallback = "python:3.11-slim"
        result = validate_from_base("", fallback_base=fallback)
        assert result.suggested_image == fallback

    def test_default_fallback_is_runpod_pytorch_base(self):
        result = validate_from_base("")
        assert result.suggested_image == _RUNPOD_PYTORCH_BASE

    # (b) known-good base — accepted unchanged

    def test_python_slim_is_ok(self):
        result = validate_from_base("FROM python:3.11-slim\nRUN pip install numpy\n")
        assert result.ok is True
        assert result.image == "python:3.11-slim"
        assert result.warning is None
        assert result.suggested_image is None

    def test_runpod_pytorch_devel_is_ok(self):
        df = "FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04\n"
        result = validate_from_base(df)
        assert result.ok is True
        assert result.warning is None

    def test_nvidia_cuda_is_ok(self):
        df = "FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04\nRUN echo hi\n"
        result = validate_from_base(df)
        assert result.ok is True
        assert result.warning is None

    def test_pytorch_pytorch_is_ok(self):
        df = "FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel\n"
        result = validate_from_base(df)
        assert result.ok is True
        assert result.warning is None

    # (c) unknown / hallucinated base — fail-soft

    def test_hallucinated_base_still_ok_but_warns(self):
        df = "FROM hallucinated/image:v99\nRUN echo hi\n"
        result = validate_from_base(df)
        assert result.ok is True  # fail-soft
        assert result.warning is not None
        assert "hallucinated/image:v99" in result.warning

    def test_hallucinated_base_suggests_fallback(self):
        df = "FROM totally-made-up:latest\n"
        fallback = "python:3.11-slim"
        result = validate_from_base(df, fallback_base=fallback)
        assert result.ok is True
        assert result.suggested_image == fallback

    def test_garbage_from_image_is_flagged(self):
        df = "FROM !!bad!!image\n"
        result = validate_from_base(df)
        assert result.ok is True  # still fail-soft
        assert result.warning is not None

    # (d) devel-vs-runtime hint

    def test_flash_attn_on_runtime_base_triggers_devel_hint(self):
        df = "FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04\n"
        reqs = "flash-attn==2.3.0\ntorch>=2.0\n"
        result = validate_from_base(df, requirements_text=reqs)
        assert result.ok is True
        assert result.devel_hint is True
        assert result.suggested_image is not None
        assert "-devel-" in result.suggested_image
        assert result.warning is not None

    def test_bitsandbytes_on_runtime_base_triggers_devel_hint(self):
        df = "FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04\n"
        reqs = "bitsandbytes>=0.41.0\ntransformers==4.40.0\n"
        result = validate_from_base(df, requirements_text=reqs)
        assert result.devel_hint is True
        assert "-devel-" in (result.suggested_image or "")

    def test_deepspeed_on_runtime_base_triggers_devel_hint(self):
        df = "FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04\n"
        reqs = "deepspeed>=0.12\n"
        result = validate_from_base(df, requirements_text=reqs)
        assert result.devel_hint is True

    def test_plain_torch_requirements_no_devel_hint(self):
        df = "FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04\n"
        reqs = "torch==2.2.0\nnumpy==1.26.4\nmatplotlib==3.8.0\n"
        result = validate_from_base(df, requirements_text=reqs)
        assert result.devel_hint is False
        # No warning from devel hint; may or may not have suggested_image from other rules
        assert "-devel-" not in (result.suggested_image or "")

    def test_devel_base_with_flash_attn_no_hint_already_devel(self):
        # Base is already -devel-; _suggest_devel_base returns the same string → no hint
        df = "FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04\n"
        reqs = "flash-attn==2.3.0\n"
        result = validate_from_base(df, requirements_text=reqs)
        # devel_hint should be False because suggested_image == image (no change)
        assert result.devel_hint is False
        assert result.suggested_image is None

    def test_empty_requirements_no_devel_hint(self):
        df = "FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04\n"
        result = validate_from_base(df, requirements_text="")
        assert result.devel_hint is False

    # ARG before FROM — pass-through

    def test_arg_before_from_accepted(self):
        df = "ARG BASE=latest\nFROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04\n"
        result = validate_from_base(df)
        assert result.ok is True
        assert result.image == "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04"

    def test_arg_interpolated_image_trusted(self):
        df = "ARG BASE\nFROM ${BASE}\n"
        result = validate_from_base(df)
        assert result.ok is True
        assert result.devel_hint is False

    # FromValidationResult attribute sanity

    def test_result_has_expected_attributes(self):
        result = validate_from_base("FROM python:3.11-slim\n")
        assert hasattr(result, "ok")
        assert hasattr(result, "image")
        assert hasattr(result, "warning")
        assert hasattr(result, "suggested_image")
        assert hasattr(result, "devel_hint")
