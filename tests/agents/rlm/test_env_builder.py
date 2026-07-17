"""Unit tests for env_builder — deterministic env-image assembly (pure, no
cloud/subprocess/network). Covers EnvSpec resolution (dedupe/sort/import-name
overrides), Dockerfile assembly shape, content-hash stability/sensitivity, and
built_image_ref formatting.
"""
from __future__ import annotations

import pytest

from backend.agents.rlm.env_builder import (
    EnvSpec,
    _import_name,
    assemble_dockerfile,
    built_image_ref,
    content_hash,
    resolve_env_spec,
)


def test_resolve_env_spec_verl_no_extra_deps():
    spec = resolve_env_spec(framework="verl", base_image="IMG", extra_deps=())
    assert spec.build_layers == ()
    assert "import torch" in spec.assertions
    assert "import flash_attn" in spec.assertions


def test_resolve_env_spec_verl_with_extra_deps_dedup_sort_and_import_override():
    spec = resolve_env_spec(
        framework="verl", base_image="IMG", extra_deps=("einops", "math-verify")
    )
    assert spec.build_layers == ("einops", "math-verify")
    assert "import einops" in spec.assertions
    assert "import math_verify" in spec.assertions


def test_resolve_env_spec_extra_deps_dedupe_and_sort_determinism():
    spec = resolve_env_spec(framework="verl", base_image="IMG", extra_deps=("b", "a", "a"))
    assert spec.build_layers == ("a", "b")


def test_resolve_env_spec_unknown_framework_has_no_base_stack():
    spec = resolve_env_spec(framework="unknown", base_image="IMG", extra_deps=("foo",))
    assert spec.assertions == ("import foo",)


def test_assemble_dockerfile_with_layers():
    spec = resolve_env_spec(
        framework="verl", base_image="IMG", extra_deps=("einops", "math-verify")
    )
    dockerfile = assemble_dockerfile(spec)
    lines = dockerfile.splitlines()

    assert lines[0] == "FROM IMG"

    pip_lines = [ln for ln in lines if ln.startswith("RUN pip install")]
    assert len(pip_lines) == 1
    assert "einops" in pip_lines[0]
    assert "math-verify" in pip_lines[0]

    assertion_run_lines = [ln for ln in lines if ln.startswith("RUN python3 -c")]
    assert len(assertion_run_lines) == len(spec.assertions)

    assert "CMD" not in dockerfile
    assert "ENTRYPOINT" not in dockerfile


def test_assemble_dockerfile_no_layers_has_no_pip_install_line():
    spec = resolve_env_spec(framework="verl", base_image="IMG", extra_deps=())
    dockerfile = assemble_dockerfile(spec)
    lines = dockerfile.splitlines()

    assert lines[0] == "FROM IMG"
    assert not any(ln.startswith("RUN pip install") for ln in lines)
    assertion_run_lines = [ln for ln in lines if ln.startswith("RUN python3 -c")]
    assert len(assertion_run_lines) == len(spec.assertions)


def test_content_hash_stable_for_same_spec():
    spec1 = resolve_env_spec(framework="verl", base_image="IMG", extra_deps=("einops",))
    spec2 = resolve_env_spec(framework="verl", base_image="IMG", extra_deps=("einops",))
    assert content_hash(spec1) == content_hash(spec2)


def test_content_hash_changes_with_base_image_or_deps():
    base = resolve_env_spec(framework="verl", base_image="IMG", extra_deps=("einops",))
    diff_base_image = resolve_env_spec(framework="verl", base_image="IMG2", extra_deps=("einops",))
    diff_deps = resolve_env_spec(framework="verl", base_image="IMG", extra_deps=("einops", "numpy"))
    diff_framework = resolve_env_spec(framework="unknown", base_image="IMG", extra_deps=("einops",))

    base_hash = content_hash(base)
    assert content_hash(diff_base_image) != base_hash
    assert content_hash(diff_deps) != base_hash
    assert content_hash(diff_framework) != base_hash


def test_content_hash_invariant_to_extra_deps_order():
    spec_ab = resolve_env_spec(framework="verl", base_image="IMG", extra_deps=("a", "b"))
    spec_ba = resolve_env_spec(framework="verl", base_image="IMG", extra_deps=("b", "a"))
    assert content_hash(spec_ab) == content_hash(spec_ba)


def test_built_image_ref_strips_trailing_slash():
    ref = built_image_ref("reg/host/proj/repo/", "verl", "abc123")
    assert ref == "reg/host/proj/repo/env-verl:abc123"


def test_import_name_overrides_and_passthrough():
    assert _import_name("flash-attn==2.7.4") == "flash_attn"
    assert _import_name("uvicorn[standard]") == "uvicorn"
    assert _import_name("numpy") == "numpy"


# --- Codex P1: reject control chars that could inject Dockerfile instructions ----------

def test_assemble_dockerfile_rejects_newline_in_base_image():
    spec = EnvSpec(framework="verl", base_image="IMG\nRUN evil", build_layers=(), assertions=())
    with pytest.raises(ValueError, match="control character"):
        assemble_dockerfile(spec)


def test_assemble_dockerfile_rejects_newline_in_dep():
    spec = EnvSpec(
        framework="verl", base_image="IMG", build_layers=("ok", "bad\nRUN evil"), assertions=()
    )
    with pytest.raises(ValueError, match="control character"):
        assemble_dockerfile(spec)


def test_assemble_dockerfile_rejects_control_char_in_assertion():
    spec = EnvSpec(
        framework="verl", base_image="IMG", build_layers=(), assertions=("import x\rRUN evil",)
    )
    with pytest.raises(ValueError, match="control character"):
        assemble_dockerfile(spec)


def test_assemble_dockerfile_clean_spec_still_renders():
    spec = resolve_env_spec(framework="verl", base_image="IMG", extra_deps=("einops",))
    df = assemble_dockerfile(spec)
    assert df.startswith("FROM IMG")
    assert "RUN pip install --no-cache-dir einops" in df
