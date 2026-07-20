"""Shared test fixtures and helpers."""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# ENV HERMETICITY — this block MUST stay above every ``backend.*`` import.
#
# The suite is socket-hermetic (pytest-socket, see pyproject `addopts`) but was
# NOT env-hermetic: `Settings` (backend/config.py) is declared with
# ``SettingsConfigDict(env_file=".env")``, and pydantic-settings re-reads that
# file FROM DISK on every ``Settings()`` construction, independent of what
# os.environ contains. That is deliberate in production (providers must work
# when spawned from the Next.js dev server or a docker entrypoint that never
# loaded the dotenv) — but under test it meant the suite silently asserted
# against whatever the *developer's* .env happened to say:
#
#   * credential exposure — a failing assertion on any Settings-derived object
#     printed live API keys straight into pytest output / CI logs (this actually
#     happened: tests/rlm/test_accelerator.py::TestResolveAuto dumped a live
#     Azure Foundry key when its `is None` assertion failed);
#   * a correctness hole — every "default flag / default setting" assertion was
#     really asserting "whatever this machine's .env says", which cannot prove
#     the repo's central default-OFF / byte-identical-when-off invariant.
#
# ``monkeypatch.delenv`` cannot fix this: the value never came from os.environ,
# it came off disk. Two layers close it:
#
#   1. Import-time (below): scrub the process env, then set
#      ``Settings.model_config["env_file"] = None``. pydantic-settings reads
#      ``self.model_config.get("env_file")`` inside ``_settings_build_values``
#      at *construction* time (not at class-creation time), so this
#      post-class-creation mutation is honoured — verified against
#      pydantic-settings 2.14 and pinned by tests/test_env_hermeticity.py.
#      This is a TEST-SIDE change only: backend/config.py is untouched and
#      production behaviour is byte-identical.
#   2. Per-test (`_isolate_environment` autouse fixture): re-assert both, via
#      monkeypatch, so one test cannot leak env/config state into the next.
#
# A test that genuinely needs a credential must set it explicitly with
# ``monkeypatch.setenv`` (it runs in the test body, i.e. after every fixture, so
# it always wins). A test that genuinely exercises the dotenv-read path must
# request the `dotenv_disk_reads_enabled` fixture below.
# ---------------------------------------------------------------------------

# Whole namespaces we own or that carry provider credentials/config overrides.
_SCRUBBED_ENV_PREFIXES: tuple[str, ...] = (
    "OPENRESEARCH_",
    "REPROLAB_",
    "ANTHROPIC_",
    "OPENAI_",
    "AZURE_",
    "GCP_",
    "RUNPOD_",
)

# Unprefixed credentials the code (or a vendored SDK) reads straight from env.
# NB: only CLAUDE_CODE_OAUTH_TOKEN is scrubbed from the CLAUDE_* namespace —
# CLAUDE_CODE_ENTRYPOINT / _SESSION_ID etc. belong to the harness running pytest.
_SCRUBBED_ENV_NAMES: frozenset[str] = frozenset(
    {
        "CLAUDE_CODE_OAUTH_TOKEN",
        "APIFY_API_TOKEN",
        "TAVILY_API_KEY",
        "FEATHERLESS_API_KEY",
        "OPENROUTER_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "WANDB_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }
)


def _leaking_env_names(env=os.environ) -> list[str]:
    """Every env var that could bleed a host credential/override into a test."""
    return [
        key
        for key in list(env)
        if key.startswith(_SCRUBBED_ENV_PREFIXES) or key in _SCRUBBED_ENV_NAMES
    ]


# Layer 1 — scrub before anything imports backend.config (whose import-time
# `_apply_legacy_env_aliases()` would otherwise mirror REPROLAB_*/OPENRESEARCH_*
# leaks across both spellings).
for _leaked in _leaking_env_names():
    os.environ.pop(_leaked, None)

import pytest

import backend.config as _backend_config
from backend.config import Settings
from backend.messaging.event import register_event

# Layer 1 (cont.) — stop pydantic-settings reading .env off disk, session-wide.
Settings.model_config["env_file"] = None
_backend_config._settings_cache = None


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
def _isolate_environment(monkeypatch):
    """Make every test env-hermetic — see the ENV HERMETICITY block up top.

    Layer 2 of the fix. The import-time block already scrubbed the process env
    and disabled the ``.env`` disk read session-wide; this re-asserts both per
    test (through monkeypatch, so it self-restores) and resets the module-level
    Settings cache, so no test can inherit — or bequeath — host config.

    Precedence, by construction:
      * this fixture           — strips host credentials/overrides;
      * fixtures that DEPEND on it (`_disable_disk_floor_preflight`,
        `dotenv_disk_reads_enabled`) — run strictly after, so their setenv
        survives the scrub;
      * a test's own ``monkeypatch.setenv`` — runs in the test body, i.e. after
        every fixture, so an explicitly-injected credential always wins.
    Both halves are pinned by tests/test_env_hermeticity.py.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for key in _leaking_env_names():
        monkeypatch.delenv(key, raising=False)

    # Not monkeypatch.setattr: that would restore the *previous* test's cached
    # Settings on teardown. Force a cold cache on both sides instead.
    _backend_config._settings_cache = None
    yield
    _backend_config._settings_cache = None


@pytest.fixture
def dotenv_disk_reads_enabled(_isolate_environment, monkeypatch):
    """Opt one test back into pydantic-settings' ``.env`` disk read.

    For the handful of tests whose SUBJECT is the dotenv-loading behaviour
    itself (they chdir into a tmp_path and write their own .env). Depending on
    `_isolate_environment` guarantees this runs after the session-wide block is
    applied, so the re-enable is not immediately undone.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", ".env")


@pytest.fixture(autouse=True)
def _disable_disk_floor_preflight(_isolate_environment, monkeypatch):
    """Keep the suite hermetic to host free-disk.

    ``run_experiment``'s disk-floor preflight (primitives.py, default
    ``OPENRESEARCH_DISK_FLOOR_GB=15``) probes the REAL host filesystem even
    when the sandbox backend is fully mocked — on any machine with <15 GB
    free, 31 otherwise-green tests fail with ``disk_exhausted``. Disable it
    by default; the floor behaviour itself is covered by
    tests/agents/rlm/test_harness_enforcement.py, which sets the variable
    explicitly (its in-test monkeypatch.setenv overrides this fixture).

    Depends on `_isolate_environment` so this setenv lands AFTER that fixture's
    OPENRESEARCH_* scrub — otherwise the scrub would delete the floor override
    and the preflight would probe the host disk again.
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
