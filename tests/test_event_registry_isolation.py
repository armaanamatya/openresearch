"""Guard tests: the global event registry must survive a registry-clearing test.

Symptom (catalogue debug-harden session): a test calling
`_clear_registry_for_tests()` wiped the global event registry, and the next
test collected — `tests/rlm/test_checkpoint.py::test_registered_in_registry` —
failed with::

    KeyError: No DomainEvent class registered for event_type='rlm_run_iteration'

The conftest autouse fixture `_isolate_event_registry` together with
`_restore_registry_for_tests()` close that hole: the registry is restored after
every test, so a clearing test can no longer pollute a later one.
"""

from __future__ import annotations

import pytest

from backend.messaging.event import (
    _clear_registry_for_tests,
    _restore_registry_for_tests,
    resolve_event_class,
)


def test_restore_reregisters_production_events_after_clear():
    """`_restore_registry_for_tests()` re-registers `rlm_run_iteration` after a clear."""
    # Importing checkpoint makes RLMRunIteration a loaded DomainEvent subclass.
    from backend.agents.rlm.checkpoint import RLMRunIteration

    _clear_registry_for_tests()
    with pytest.raises(KeyError):
        resolve_event_class("rlm_run_iteration", 1)

    _restore_registry_for_tests()
    assert resolve_event_class("rlm_run_iteration", 1) is RLMRunIteration


def test_a_clearing_test_does_not_restore():
    """A test that clears the registry and deliberately does NOT restore it.

    The autouse `_isolate_event_registry` fixture must heal this in teardown —
    `test_b_registry_intact_despite_prior_clear` verifies it did.
    """
    _clear_registry_for_tests()
    # Intentionally no restore — the autouse fixture is responsible.


def test_b_registry_intact_despite_prior_clear():
    """Despite the prior test clearing the registry, this test sees it intact.

    Proof that the autouse `_isolate_event_registry` fixture restored the
    registry in the previous test's teardown.
    """
    from backend.agents.rlm.checkpoint import RLMRunIteration

    assert resolve_event_class("rlm_run_iteration", 1) is RLMRunIteration
