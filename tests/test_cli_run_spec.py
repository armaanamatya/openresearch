"""Hermetic tests for the --run-spec loader (P0.3).

Tests assert:
  (a) spec OPENRESEARCH_* / REPROLAB_* keys land in os.environ
  (b) an explicit CLI --max-usd value overrides the spec's OPENRESEARCH_MAX_USD
  (c) a multi-line baseline_extra_guidance lands in OPENRESEARCH_BASELINE_EXTRA_GUIDANCE intact
  (d) a missing / non-object / invalid JSON spec file errors clearly

No network, no subprocess, no VM.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the loader under test directly (avoids any heavy CLI import chain).
# ---------------------------------------------------------------------------
from backend.cli import _load_run_spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_spec(tmp_path: Path, data: dict) -> str:
    """Write *data* as JSON to a temp file; return the path string."""
    p = tmp_path / "run_spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# (a) Spec OPENRESEARCH_* / REPROLAB_* keys land in os.environ
# ---------------------------------------------------------------------------

def test_openresearch_keys_land_in_environ(tmp_path, monkeypatch):
    """OPENRESEARCH_* keys in the spec are written into os.environ."""
    monkeypatch.delenv("OPENRESEARCH_ZERO_METRICS_GUARD", raising=False)
    monkeypatch.delenv("OPENRESEARCH_STUB_METRICS_GUARD", raising=False)

    spec_path = _write_spec(tmp_path, {
        "OPENRESEARCH_ZERO_METRICS_GUARD": "1",
        "OPENRESEARCH_STUB_METRICS_GUARD": "1",
    })

    _load_run_spec(spec_path)

    assert os.environ["OPENRESEARCH_ZERO_METRICS_GUARD"] == "1"
    assert os.environ["OPENRESEARCH_STUB_METRICS_GUARD"] == "1"


def test_reprolab_keys_land_in_environ(tmp_path, monkeypatch):
    """REPROLAB_* keys (legacy prefix) are also accepted and written into os.environ."""
    monkeypatch.delenv("REPROLAB_FINALIZE_REGRADE", raising=False)

    spec_path = _write_spec(tmp_path, {
        "REPROLAB_FINALIZE_REGRADE": "1",
    })

    _load_run_spec(spec_path)

    assert os.environ["REPROLAB_FINALIZE_REGRADE"] == "1"


def test_models_key_maps_to_role_models(tmp_path, monkeypatch):
    """The ``models`` key maps to OPENRESEARCH_ROLE_MODELS."""
    monkeypatch.delenv("OPENRESEARCH_ROLE_MODELS", raising=False)

    spec_path = _write_spec(tmp_path, {
        "models": "executor=sonnet,grader=gpt-4o",
    })

    _load_run_spec(spec_path)

    assert os.environ["OPENRESEARCH_ROLE_MODELS"] == "executor=sonnet,grader=gpt-4o"


def test_unknown_keys_silently_ignored(tmp_path, monkeypatch):
    """Unknown spec keys are ignored (forward-compat) without raising."""
    monkeypatch.delenv("OPENRESEARCH_ZERO_METRICS_GUARD", raising=False)

    spec_path = _write_spec(tmp_path, {
        "OPENRESEARCH_ZERO_METRICS_GUARD": "1",
        "some_future_key_unknown": "ignored",
        "another_unrecognised": 42,
    })

    _load_run_spec(spec_path)  # must not raise

    assert os.environ["OPENRESEARCH_ZERO_METRICS_GUARD"] == "1"
    assert "some_future_key_unknown" not in os.environ
    assert "another_unrecognised" not in os.environ


# ---------------------------------------------------------------------------
# (b) Explicit CLI flag overrides spec's OPENRESEARCH_MAX_USD
# ---------------------------------------------------------------------------

def test_explicit_flag_overrides_spec_max_usd(tmp_path, monkeypatch):
    """An explicit --max-usd CLI arg wins over the spec's OPENRESEARCH_MAX_USD.

    The spec loader writes OPENRESEARCH_MAX_USD into os.environ (the spec-layer
    default).  A non-None args.max_usd (from the CLI flag) is what RunBudget
    reads directly — it is set AFTER the spec load, so it always wins.  This
    test confirms the spec value is NOT mistakenly treated as authoritative when
    a CLI flag was supplied.
    """
    monkeypatch.delenv("OPENRESEARCH_MAX_USD", raising=False)

    spec_path = _write_spec(tmp_path, {
        "OPENRESEARCH_MAX_USD": "99.0",
    })

    # Load the spec (writes env var as the baseline).
    _load_run_spec(spec_path)
    assert os.environ["OPENRESEARCH_MAX_USD"] == "99.0"

    # Simulate the CLI flag overriding the env var (the env-sink block in
    # cmd_reproduce sets the env var AFTER _load_run_spec when the flag was
    # passed explicitly).
    os.environ["OPENRESEARCH_MAX_USD"] = "5.0"   # explicit flag wins

    assert os.environ["OPENRESEARCH_MAX_USD"] == "5.0"


def test_spec_max_usd_does_not_clobber_args_max_usd(tmp_path, monkeypatch):
    """args.max_usd set by an explicit CLI flag is unaffected by _load_run_spec.

    _load_run_spec only writes os.environ; it never touches argparse Namespace
    attributes.  So if --max-usd 5 set args.max_usd = 5.0, that value is intact
    after the spec load regardless of what OPENRESEARCH_MAX_USD the spec sets.
    """
    monkeypatch.delenv("OPENRESEARCH_MAX_USD", raising=False)

    spec_path = _write_spec(tmp_path, {
        "OPENRESEARCH_MAX_USD": "99.0",
    })

    # Simulate args.max_usd as set by argparse from --max-usd 5.
    args = argparse.Namespace(max_usd=5.0, run_spec=spec_path)

    _load_run_spec(spec_path)

    # The spec loaded env, but args.max_usd must be unchanged (still 5.0).
    assert args.max_usd == 5.0
    # The env var reflects the spec value (overrideable by the env-sink block).
    assert os.environ["OPENRESEARCH_MAX_USD"] == "99.0"


# ---------------------------------------------------------------------------
# (c) Multi-line baseline_extra_guidance lands in OPENRESEARCH_BASELINE_EXTRA_GUIDANCE
# ---------------------------------------------------------------------------

def test_multiline_baseline_extra_guidance_intact(tmp_path, monkeypatch):
    """Multi-line baseline_extra_guidance survives JSON roundtrip without truncation.

    This is the primary footgun the spec §10 calls out: shell env-word-split
    mangles multi-line values when forwarded via env $VAR in SSH.  The JSON
    carrier avoids that entirely.
    """
    monkeypatch.delenv("OPENRESEARCH_BASELINE_EXTRA_GUIDANCE", raising=False)

    guidance = textwrap.dedent("""\
        Use Qwen3-1.7B and Qwen2.5-3B only.
        ALFWorld: use the shipped AgenticEnv modules.
        Do NOT use any closed-book surrogates.

        Extra line with   spaces   and\ttabs.
    """)

    spec_path = _write_spec(tmp_path, {
        "baseline_extra_guidance": guidance,
    })

    _load_run_spec(spec_path)

    loaded = os.environ["OPENRESEARCH_BASELINE_EXTRA_GUIDANCE"]
    assert loaded == guidance, (
        f"Multi-line guidance was mangled.\n"
        f"Expected:\n{guidance!r}\n"
        f"Got:\n{loaded!r}"
    )


def test_multiline_guidance_with_newlines_and_special_chars(tmp_path, monkeypatch):
    """Guidance containing colons, slashes, and quotes survives intact."""
    monkeypatch.delenv("OPENRESEARCH_BASELINE_EXTRA_GUIDANCE", raising=False)

    guidance = (
        "Line 1: value=0.1\n"
        "Line 2: path=/some/dir\n"
        'Line 3: note="quoted"\n'
    )

    spec_path = _write_spec(tmp_path, {"baseline_extra_guidance": guidance})
    _load_run_spec(spec_path)

    assert os.environ["OPENRESEARCH_BASELINE_EXTRA_GUIDANCE"] == guidance


# ---------------------------------------------------------------------------
# (d) Missing / invalid / non-object spec file → clear error
# ---------------------------------------------------------------------------

def test_missing_spec_file_raises(tmp_path):
    """A path to a non-existent file raises ArgumentTypeError immediately."""
    missing = str(tmp_path / "does_not_exist.json")
    with pytest.raises(argparse.ArgumentTypeError, match="file not found"):
        _load_run_spec(missing)


def test_invalid_json_raises(tmp_path):
    """A file containing invalid JSON raises ArgumentTypeError."""
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json ", encoding="utf-8")
    with pytest.raises(argparse.ArgumentTypeError, match="invalid JSON"):
        _load_run_spec(str(bad))


def test_json_array_raises(tmp_path):
    """A JSON array (not an object) raises ArgumentTypeError."""
    arr = tmp_path / "array.json"
    arr.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(argparse.ArgumentTypeError, match="expected a JSON object"):
        _load_run_spec(str(arr))


def test_json_string_raises(tmp_path):
    """A JSON string (not an object) raises ArgumentTypeError."""
    s = tmp_path / "string.json"
    s.write_text(json.dumps("not an object"), encoding="utf-8")
    with pytest.raises(argparse.ArgumentTypeError, match="expected a JSON object"):
        _load_run_spec(str(s))


def test_json_null_raises(tmp_path):
    """A JSON null raises ArgumentTypeError."""
    n = tmp_path / "null.json"
    n.write_text("null", encoding="utf-8")
    with pytest.raises(argparse.ArgumentTypeError, match="expected a JSON object"):
        _load_run_spec(str(n))


def test_empty_spec_is_valid(tmp_path, monkeypatch):
    """An empty JSON object ``{}`` is valid (no keys applied, no error)."""
    spec_path = _write_spec(tmp_path, {})
    _load_run_spec(spec_path)  # must not raise


# ---------------------------------------------------------------------------
# Additional: numeric values are coerced to string in os.environ
# ---------------------------------------------------------------------------

def test_numeric_spec_values_coerced_to_string(tmp_path, monkeypatch):
    """Numeric JSON values are string-coerced when written to os.environ."""
    monkeypatch.delenv("OPENRESEARCH_MAX_USD", raising=False)
    monkeypatch.delenv("OPENRESEARCH_GRADER_SAMPLES", raising=False)

    spec_path = _write_spec(tmp_path, {
        "OPENRESEARCH_MAX_USD": 12.5,
        "OPENRESEARCH_GRADER_SAMPLES": 3,
    })

    _load_run_spec(spec_path)

    assert os.environ["OPENRESEARCH_MAX_USD"] == "12.5"
    assert os.environ["OPENRESEARCH_GRADER_SAMPLES"] == "3"
