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
