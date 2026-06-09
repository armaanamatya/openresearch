"""Tests for fair_comparison — identical-init snapshot + verifiable fingerprint (Module B).

torch-free: a small ``FakeModel`` exercises snapshot/restore so the suite needs no GPU/torch.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm import fair_comparison as fc


class FakeModel:
    """Minimal state_dict()/load_state_dict() double standing in for an nn.Module."""

    def __init__(self, weights):
        self._w = dict(weights)

    def state_dict(self):
        return dict(self._w)

    def load_state_dict(self, sd):
        self._w = dict(sd)


# --------------------------------------------------------------------------- flag

def test_is_enabled_default_off(monkeypatch):
    monkeypatch.delenv(fc.ENV_FLAG, raising=False)
    assert fc.is_enabled() is False


def test_is_enabled_on(monkeypatch):
    monkeypatch.setenv(fc.ENV_FLAG, "1")
    assert fc.is_enabled() is True


# --------------------------------------------------------------------------- snapshot / restore

def test_snapshot_is_immune_to_later_mutation():
    m = FakeModel({"w": [1.0, 2.0], "b": [0.0]})
    snap = fc.snapshot_init_state(m)
    # train: mutate the live model
    m.load_state_dict({"w": [9.0, 9.0], "b": [9.0]})
    # snapshot must be unchanged (deep copy)
    assert snap["w"] == [1.0, 2.0]


def test_restore_brings_back_initial_weights():
    m = FakeModel({"w": [1.0, 2.0]})
    snap = fc.snapshot_init_state(m)
    m.load_state_dict({"w": [5.0, 5.0]})
    assert fc.restore_init_state(m, snap) is True
    assert m.state_dict()["w"] == [1.0, 2.0]


def test_snapshot_failsoft_on_bad_model():
    assert fc.snapshot_init_state(object()) == {}


def test_restore_failsoft_on_bad_model():
    assert fc.restore_init_state(object(), {"w": [1.0]}) is False


# --------------------------------------------------------------------------- fingerprint = evidence

def test_fingerprint_deterministic_and_stable():
    a = fc.init_fingerprint({"w": [1.0, 2.0], "b": [0.0]})
    b = fc.init_fingerprint({"b": [0.0], "w": [1.0, 2.0]})  # key order must not matter
    assert a == b
    assert len(a) == 16


def test_identical_init_same_fingerprint_across_methods():
    # the WHOLE point: two models started from the same init hash identically →
    # the grader can confirm "identical initialization across optimizers".
    init = {"w": [0.1, 0.2, 0.3]}
    m_adam = FakeModel(init)
    m_sgd = FakeModel(init)
    fp_adam = fc.init_fingerprint(fc.snapshot_init_state(m_adam))
    fp_sgd = fc.init_fingerprint(fc.snapshot_init_state(m_sgd))
    assert fp_adam == fp_sgd


def test_different_init_different_fingerprint():
    assert fc.init_fingerprint({"w": [1.0]}) != fc.init_fingerprint({"w": [2.0]})


def test_fingerprint_empty_is_sentinel_not_error():
    assert isinstance(fc.init_fingerprint({}), str)
    assert fc.init_fingerprint({}) == fc.init_fingerprint(None)  # type: ignore[arg-type]
