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


# Ambient real credentials the documented local-dev .env carries. python-dotenv /
# pydantic-settings load .env into os.environ on backend import, so these are
# present for the whole pytest session even though a bare shell doesn't export
# them — which flips every "logged-out / no-provider-configured" detection test
# red on a real developer machine. We strip them by default so credential and
# provider-resolution tests are hermetic; a test that needs one present sets it
# explicitly via monkeypatch.setenv, which overrides this clear. Same doctrine as
# the disk-floor fixture above (tests/CLAUDE.md: no test may depend on an ambient
# host credential). Both bare and OPENRESEARCH_-prefixed forms are cleared.
_AMBIENT_CREDENTIAL_ENV = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_ADMIN_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_FOUNDRY_API_KEY",
    "AZURE_FOUNDRY_ENDPOINT",
    "AZURE_FOUNDRY_DEPLOYMENT",
    "FEATHERLESS_API_KEY",
    "OPENROUTER_API_KEY",
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "ambient_env: opt out of the .env-isolation / credential-clearing autouse "
        "fixtures — for tests that intentionally exercise real .env-file loading or "
        "ambient provider-credential resolution.",
    )


@pytest.fixture(autouse=True)
def _isolate_dev_dotenv(request, monkeypatch):
    """Don't let the developer's real ``.env`` file leak into ``Settings``.

    ``Settings.model_config`` hardcodes ``env_file=".env"``, so on a real dev
    machine pydantic-settings loads live secrets (Azure Foundry endpoint/key,
    provider keys, gcp_project, …) into every ``Settings()`` — and credential /
    provider resolution (``resolve_foundry_credentials``, ``resolve_accelerator``)
    falls through to Settings, so clearing ``os.environ`` alone isn't enough
    (test_accelerator's TestResolveAuto, test_gcp_orchestrator_settings's
    defaults regressions). Point ``env_file`` at nothing and bust the cache so
    the suite reads only what a test explicitly sets via ``monkeypatch.setenv``
    (env vars still override — this disables *file* loading only, not env reads).
    Tests that need the file open it explicitly with ``Settings(_env_file=...)``.
    Opt out with ``@pytest.mark.ambient_env`` (e.g. the dotenv-loading test).
    """
    if request.node.get_closest_marker("ambient_env"):
        return
    from backend.config import Settings

    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setattr("backend.config._settings_cache", None, raising=False)


@pytest.fixture(autouse=True)
def _clear_ambient_provider_credentials(request, monkeypatch):
    """Keep credential + provider-resolution detection hermetic to the dev's env.

    See ``_AMBIENT_CREDENTIAL_ENV`` above. Complements ``_isolate_dev_dotenv``:
    that stops ``.env`` *file* loading; this strips the same secrets when they
    leak from the *shell* (a real ``CLAUDE_CODE_OAUTH_TOKEN`` export short-circuits
    ``_has_claude_subscription_oauth`` True; raw ``AZURE_*`` exports resolve a live
    accelerator). Together they make the "no credentials / returns None" tests
    (test_agent_runtime_factory, test_runtime_oauth_wsl, test_accelerator's
    TestResolveAuto, test_bundled_claude_detection, test_catalogue_i8_i11_i13,
    test_gcp_orchestrator_settings) pass on any machine with populated creds.
    Opt out with ``@pytest.mark.ambient_env`` (e.g. the foundry-alias test that
    needs a real AZURE_FOUNDRY_API_KEY present to resolve).
    """
    if request.node.get_closest_marker("ambient_env"):
        return
    for name in _AMBIENT_CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(f"OPENRESEARCH_{name}", raising=False)
