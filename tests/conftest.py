"""Shared test fixtures and helpers."""

from __future__ import annotations

import pytest

from backend.messaging.event import register_event


def _re_register_production_events() -> None:
    """Re-register production event classes after `_clear_registry_for_tests()`.

    Foundation tests intentionally clear the registry; subsequent tests in
    other modules need production events present (otherwise replay /
    `StoredEvent.into()` raise KeyError). We re-register the existing
    class objects so isinstance() checks in aggregates still work.

    Each module is imported lazily; missing modules (because that part of
    the slice hasn't landed yet) are skipped silently.
    """
    classes: list[type] = []

    try:
        from backend.services.ingestion.intake.events import (
            PaperFetchFailed,
            PaperFetched,
            ProjectCreated,
        )

        classes.extend([ProjectCreated, PaperFetched, PaperFetchFailed])
    except ImportError:
        pass

    try:
        from backend.services.ingestion.parser.events import (
            FigureExtracted,
            ParsingCompleted,
            ParsingFailed,
            ParsingStarted,
            ReferenceExtracted,
            SectionExtracted,
        )

        classes.extend(
            [
                ParsingStarted,
                SectionExtracted,
                ReferenceExtracted,
                FigureExtracted,
                ParsingCompleted,
                ParsingFailed,
            ]
        )
    except ImportError:
        pass

    try:
        from backend.services.runtime.events import (
            CommandExecuted,
            CommandFailed,
            SandboxCreated,
            SandboxDestroyed,
            SandboxFailed,
            SandboxRequested,
        )

        classes.extend(
            [
                SandboxRequested,
                SandboxCreated,
                SandboxFailed,
                CommandExecuted,
                CommandFailed,
                SandboxDestroyed,
            ]
        )
    except ImportError:
        pass

    for cls in classes:
        try:
            register_event(cls)
        except Exception:
            # Already registered or conflict — both fine for fixture setup.
            pass


@pytest.fixture
def production_events_registered():
    """Force production event modules to (re)register before a test."""
    _re_register_production_events()
    yield


@pytest.fixture(autouse=True)
def _isolate_event_registry():
    """Restore the production event registry after every test.

    Several tests call `_clear_registry_for_tests()` to exercise event
    registration in isolation. Without this autouse guard the global registry
    stays cleared, and any later test that resolves a production event — e.g.
    `tests/rlm/test_checkpoint.py::test_registered_in_registry` resolving
    `rlm_run_iteration` — fails with KeyError depending purely on pytest
    collection order. Restoring after every test makes the registry
    order-independent for the whole suite.
    """
    yield
    from backend.messaging.event import _restore_registry_for_tests

    _restore_registry_for_tests()


@pytest.fixture(autouse=True)
def _disable_disk_floor_preflight(monkeypatch):
    """Keep the suite hermetic to host free-disk.

    ``run_experiment``'s disk-floor preflight (primitives.py, default
    ``OPENRESEARCH_DISK_FLOOR_GB=15``) probes the REAL host filesystem even
    when the sandbox backend is fully mocked — on any machine with <15 GB
    free, 31 otherwise-green tests fail with ``disk_exhausted``. Disable it
    by default; the floor behaviour itself is covered by
    tests/agents/rlm/test_harness_enforcement.py, which sets the variable
    explicitly (its in-test monkeypatch.setenv overrides this fixture).
    """
    monkeypatch.setenv("OPENRESEARCH_DISK_FLOOR_GB", "0")
