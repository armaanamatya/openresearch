"""Unit tests for backend.agents.rlm.evidence_key (Workstream A3).

Properties under test (from the design test plan):
  * identical metrics+scope -> identical key
  * dict-key reordering (metrics and scope) -> SAME key
  * one extra epoch appended to a series (len changes) -> DIFFERENT key (desired)
  * scope change -> DIFFERENT key
  * None scope works
"""

from __future__ import annotations

import copy
import json

import pytest

from backend.agents.rlm.evidence_key import evidence_key, normalize_scope


def _metrics(epochs: int = 50, err: float = 9.1) -> dict:
    return {
        "per_model": {
            "plain_20": {
                "cifar10": {
                    "plain": {
                        "status": "ok",
                        "test_error_pct": err,
                        "best_test_accuracy": 0.909,
                        "history": {
                            "epoch": list(range(epochs)),
                            "train_loss": [1.0 / (i + 1) for i in range(epochs)],
                            "test_accuracy": [0.5 + i * 0.001 for i in range(epochs)],
                        },
                    }
                }
            },
        },
        "comparison": {"winner": "plain_20"},
    }


# --- identity / determinism ------------------------------------------------

def test_identical_metrics_and_scope_identical_key():
    m = _metrics()
    scope = {"models": ["plain_20"], "environments": ["cifar10"]}
    assert evidence_key(m, scope) == evidence_key(copy.deepcopy(m), copy.deepcopy(scope))


def test_returns_64_char_hex():
    k = evidence_key(_metrics(), None)
    assert isinstance(k, str)
    assert len(k) == 64
    int(k, 16)  # parses as hex


# --- order independence ----------------------------------------------------

def test_metrics_dict_key_reorder_same_key():
    m = _metrics()
    # Rebuild the same dict with keys inserted in a different order.
    cell = m["per_model"]["plain_20"]["cifar10"]["plain"]
    reordered_cell = {
        "history": cell["history"],
        "best_test_accuracy": cell["best_test_accuracy"],
        "test_error_pct": cell["test_error_pct"],
        "status": cell["status"],
    }
    m2 = {
        "comparison": {"winner": "plain_20"},
        "per_model": {"plain_20": {"cifar10": {"plain": reordered_cell}}},
    }
    assert evidence_key(m, None) == evidence_key(m2, None)


def test_scope_list_order_independent():
    m = _metrics()
    k1 = evidence_key(m, {"models": ["a", "b", "c"], "env": ["x", "y"]})
    k2 = evidence_key(m, {"env": ["y", "x"], "models": ["c", "a", "b"]})
    assert k1 == k2


def test_scope_set_and_list_collapse():
    m = _metrics()
    # A leaf-id set vs the same ids as a reordered list -> same normalized form.
    k_set = evidence_key(m, {"leaves": {"leaf3", "leaf1", "leaf2"}})
    k_list = evidence_key(m, {"leaves": ["leaf1", "leaf2", "leaf3"]})
    assert k_set == k_list


def test_bare_list_scope_order_independent():
    m = _metrics()
    assert evidence_key(m, ["leaf_b", "leaf_a"]) == evidence_key(m, ["leaf_a", "leaf_b"])


# --- evidence growth changes the key (desired) -----------------------------

def test_one_extra_epoch_changes_key():
    k_short = evidence_key(_metrics(epochs=50), None)
    k_long = evidence_key(_metrics(epochs=51), None)
    assert k_short != k_long


def test_short_series_value_change_changes_key():
    # A short series (<= keep threshold) is preserved verbatim, so changing a
    # value changes the key.
    m1 = {"per_model": {"m": {"e": {"b": {"status": "ok", "vals": [1, 2, 3]}}}}}
    m2 = {"per_model": {"m": {"e": {"b": {"status": "ok", "vals": [1, 2, 4]}}}}}
    assert evidence_key(m1, None) != evidence_key(m2, None)


def test_headline_scalar_change_changes_key():
    assert evidence_key(_metrics(err=9.1), None) != evidence_key(_metrics(err=8.7), None)


# --- scope sensitivity -----------------------------------------------------

def test_scope_change_changes_key():
    m = _metrics()
    k1 = evidence_key(m, {"models": ["plain_20", "plain_32"]})
    k2 = evidence_key(m, {"models": ["plain_20"]})
    assert k1 != k2


def test_none_vs_empty_scope_distinct_but_both_valid():
    m = _metrics()
    k_none = evidence_key(m, None)
    k_empty = evidence_key(m, {})
    assert len(k_none) == 64 and len(k_empty) == 64
    # None and an empty dict are different scope descriptors.
    assert k_none != k_empty


def test_none_scope_works():
    assert isinstance(evidence_key(_metrics(), None), str)
    assert isinstance(evidence_key(_metrics()), str)  # default arg


# --- normalize_scope unit --------------------------------------------------

def test_normalize_scope_none():
    assert normalize_scope(None) is None


def test_normalize_scope_nested_sorted():
    out = normalize_scope({"b": [3, 1, 2], "a": ["z", "x", "y"]})
    assert out == {"a": ["x", "y", "z"], "b": [1, 2, 3]}


def test_normalize_scope_is_json_serializable():
    out = normalize_scope({"models": {"m2", "m1"}, "seeds": [3, 1]})
    json.dumps(out)  # must not raise


def test_normalize_scope_scalar_passthrough():
    assert normalize_scope("abc") == "abc"
    assert normalize_scope(7) == 7


# --- fail-soft on odd inputs ----------------------------------------------

def test_non_dict_metrics_does_not_raise():
    # Defensive: a partial/unusual metrics shape still fingerprints.
    k = evidence_key([], None)  # type: ignore[arg-type]
    assert len(k) == 64


def test_default_str_handles_unserializable_scope_element():
    class Weird:
        def __repr__(self):
            return "WEIRD"

    # default=str at the dump site keeps it from raising.
    k = evidence_key(_metrics(), {"x": Weird()})
    assert len(k) == 64
