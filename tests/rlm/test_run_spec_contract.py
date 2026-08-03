"""tests/rlm/test_run_spec_contract.py — Codex F15 round-trip validation.

Covers the extracted ``run_spec_contract`` module (key-acceptance predicate,
the env-sink applier, and the round-trip validator campaign INIT calls
before any money moves) plus a regression pin that ``cli.py::_load_run_spec``
still behaves identically after delegating its per-key loop to
``apply_run_spec``.
"""

from __future__ import annotations

import argparse
import json
import os

import pytest

from backend.agents.rlm.run_spec_contract import (
    RunSpecValidationError,
    apply_run_spec,
    run_spec_key_applies,
    validate_run_spec_keys,
)


# ---------------------------------------------------------------------------
# test_key_acceptance_matrix
# ---------------------------------------------------------------------------


def test_key_acceptance_matrix():
    """Accepted: OPENRESEARCH_*, REPROLAB_*, models, baseline_extra_guidance.
    Rejected: anything else."""
    accepted = [
        "OPENRESEARCH_ZERO_METRICS_GUARD",
        "OPENRESEARCH_MAX_USD",
        "REPROLAB_FINALIZE_REGRADE",
        "REPROLAB_BES_ENABLED",
        "models",
        "baseline_extra_guidance",
    ]
    rejected = [
        "some_future_key_unknown",
        "another_unrecognised",
        "MAX_USD",
        "openresearch_lowercase",  # prefix match is case-sensitive
        "",
        "model",  # near-miss of "models"
        "baseline_extra_guidance_typo",
    ]
    for key in accepted:
        assert run_spec_key_applies(key), f"{key!r} should be accepted"
    for key in rejected:
        assert not run_spec_key_applies(key), f"{key!r} should be rejected"


# ---------------------------------------------------------------------------
# test_apply_matches_cli_semantics
# ---------------------------------------------------------------------------


def test_apply_matches_cli_semantics():
    """Special-key mapping + str coercion + applied count, against a plain
    dict sink (no os.environ dependency)."""
    sink: dict[str, str] = {}
    spec = {
        "models": "executor=sonnet,grader=gpt-4o",
        "baseline_extra_guidance": "line one\nline two\n",
        "OPENRESEARCH_MAX_USD": 12.5,
        "OPENRESEARCH_GRADER_SAMPLES": 3,
        "REPROLAB_FINALIZE_REGRADE": "1",
        "some_future_key_unknown": "ignored",
        "another_unrecognised": 42,
    }

    applied = apply_run_spec(spec, sink)

    assert applied == 5
    assert sink["OPENRESEARCH_ROLE_MODELS"] == "executor=sonnet,grader=gpt-4o"
    assert sink["OPENRESEARCH_BASELINE_EXTRA_GUIDANCE"] == "line one\nline two\n"
    assert sink["OPENRESEARCH_MAX_USD"] == "12.5"
    assert sink["OPENRESEARCH_GRADER_SAMPLES"] == "3"
    assert sink["REPROLAB_FINALIZE_REGRADE"] == "1"
    assert "some_future_key_unknown" not in sink
    assert "another_unrecognised" not in sink
    assert "models" not in sink
    assert "baseline_extra_guidance" not in sink


def test_apply_run_spec_empty_spec_applies_nothing():
    sink: dict[str, str] = {}
    assert apply_run_spec({}, sink) == 0
    assert sink == {}


# ---------------------------------------------------------------------------
# test_validate_raises_listing_every_rejected_key
# ---------------------------------------------------------------------------


def test_validate_raises_listing_every_rejected_key():
    keys = [
        "OPENRESEARCH_EVIDENCE_GATE",
        "models",
        "typo_key_one",
        "REPROLAB_REUSE_RUBRIC",
        "typo_key_two",
    ]
    with pytest.raises(RunSpecValidationError) as excinfo:
        validate_run_spec_keys(keys)

    err = excinfo.value
    assert err.rejected_keys == ("typo_key_one", "typo_key_two")
    # The message names every rejected key (an operator sees the whole
    # picture — the F15 round-trip check fails at $0, not one key at a time).
    assert "typo_key_one" in str(err)
    assert "typo_key_two" in str(err)
    assert "OPENRESEARCH_EVIDENCE_GATE" not in str(err)


def test_validate_passes_when_every_key_applies():
    keys = [
        "OPENRESEARCH_SEED_BEST_ATTEMPT",
        "OPENRESEARCH_TARGET_BEST_FLOOR",
        "OPENRESEARCH_PRIOR_ATTEMPT_EVIDENCE",
        "OPENRESEARCH_FAILURE_CAPSULES",
        "OPENRESEARCH_NEGATIVE_LESSONS",
        "OPENRESEARCH_EXPERIENCE_MEMORY",
        "OPENRESEARCH_EVIDENCE_GATE",
        "OPENRESEARCH_REPRO_CONTRACT",
        "OPENRESEARCH_USE_AUTHOR_REPO",
        "OPENRESEARCH_REUSE_RUBRIC",
        # Scheduler-authority rung-step -> trainer iteration-budget wire: the
        # campaign threads these into the branch's enforcement env; they must be
        # recognized run-spec keys so no run-spec assertion silently drops them.
        "OPENRESEARCH_CELL_ITER_BUDGET",
        "OPENRESEARCH_CELL_ITER_FROM",
        "models",
        "baseline_extra_guidance",
    ]
    validate_run_spec_keys(keys)  # must not raise


def test_validate_empty_iterable_passes():
    validate_run_spec_keys([])  # must not raise


# ---------------------------------------------------------------------------
# test_cli_load_run_spec_still_applies_and_ignores_unknown
# ---------------------------------------------------------------------------


def test_cli_load_run_spec_still_applies_and_ignores_unknown(tmp_path, monkeypatch):
    """Regression pin: cli.py::_load_run_spec, after delegating its per-key
    loop to apply_run_spec, still applies known keys and silently skips
    unknown ones — identical to its pre-refactor behavior."""
    from backend.cli import _load_run_spec

    monkeypatch.delenv("OPENRESEARCH_ZERO_METRICS_GUARD", raising=False)
    monkeypatch.delenv("OPENRESEARCH_ROLE_MODELS", raising=False)
    monkeypatch.delenv("some_future_key_unknown", raising=False)

    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text(
        json.dumps({
            "OPENRESEARCH_ZERO_METRICS_GUARD": "1",
            "models": "executor=sonnet",
            "some_future_key_unknown": "ignored",
        }),
        encoding="utf-8",
    )

    _load_run_spec(str(spec_path))

    assert os.environ["OPENRESEARCH_ZERO_METRICS_GUARD"] == "1"
    assert os.environ["OPENRESEARCH_ROLE_MODELS"] == "executor=sonnet"
    assert "some_future_key_unknown" not in os.environ


def test_cli_load_run_spec_bad_json_raises_same_error_type(tmp_path):
    from backend.cli import _load_run_spec

    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json ", encoding="utf-8")
    with pytest.raises(argparse.ArgumentTypeError, match="invalid JSON"):
        _load_run_spec(str(bad))


def test_cli_load_run_spec_missing_file_raises_same_error_type(tmp_path):
    from backend.cli import _load_run_spec

    missing = tmp_path / "nope.json"
    with pytest.raises(argparse.ArgumentTypeError, match="file not found"):
        _load_run_spec(str(missing))


def test_cli_load_run_spec_non_object_raises_same_error_type(tmp_path):
    from backend.cli import _load_run_spec

    arr = tmp_path / "array.json"
    arr.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(argparse.ArgumentTypeError, match="expected a JSON object"):
        _load_run_spec(str(arr))
