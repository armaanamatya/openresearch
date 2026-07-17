"""Env-hermeticity guard — keeps the suite honest about what it is asserting.

`Settings` is declared with ``SettingsConfigDict(env_file=".env")`` and
pydantic-settings re-reads that file from disk on EVERY ``Settings()``
construction, regardless of os.environ. Deliberate in production; corrosive
under test, in two distinct ways:

  1. CREDENTIAL EXPOSURE. Any failing assertion on a Settings-derived object
     prints the live values into pytest output / CI logs. This was not
     hypothetical — ``tests/rlm/test_accelerator.py::TestResolveAuto`` asserts
     ``resolve_accelerator("auto") is None``, got a live Azure Foundry endpoint
     from the developer's .env instead, and dumped the API key in full into the
     assertion diff.
  2. SILENT FALSIFICATION. Every "this flag defaults to X" assertion was really
     asserting "whatever this machine's .env says". A suite that reads a
     developer's .env cannot prove the repo's central default-OFF /
     byte-identical-when-off invariant — it can only prove it for that laptop.

``monkeypatch.delenv`` cannot close this: the value never travelled through
os.environ. tests/conftest.py blocks the disk read instead. These tests pin that
block in place.

RULE FOR THIS FILE: no assertion here may ever put a credential VALUE into its
failure message. Compare on field NAMES and booleans, never on the secret. The
`model_dump` diff below is deliberately reduced to a list of differing keys for
exactly this reason.
"""

from __future__ import annotations

import os

import pytest

from backend.config import Settings, get_settings
from tests.conftest import _SCRUBBED_ENV_NAMES, _leaking_env_names

# Credential-bearing Settings fields. Under a hermetic suite every one of these
# is the empty-string default; a non-empty value means the host bled through.
_CREDENTIAL_FIELDS = (
    "anthropic_api_key",
    "openai_api_key",
    "openai_admin_key",
    "azure_openai_api_key",
    "azure_foundry_api_key",
    "azure_foundry_endpoint",
    "azure_foundry_deployment",
    "runpod_api_key",
    "apify_api_token",
    "claude_code_oauth_token",
    "demo_secret",
)


# ---------------------------------------------------------------------------
# The headline guard: no host credential reaches Settings.
# ---------------------------------------------------------------------------
def test_no_credential_from_dotenv_reaches_settings():
    settings = Settings()

    # NAMES ONLY in the failure message — never `assert settings.x == ""`,
    # which would print the secret it is meant to protect.
    leaked = sorted(f for f in _CREDENTIAL_FIELDS if getattr(settings, f))
    assert not leaked, (
        "env hermeticity REGRESSED: the host .env / process env bled credentials "
        f"into Settings() for fields {leaked}. A failing test can now print live "
        "secrets into CI logs. See the ENV HERMETICITY block in tests/conftest.py."
    )


def test_no_credential_from_dotenv_reaches_get_settings():
    """`get_settings()` caches a module-global — it must be hermetic too."""
    settings = get_settings(_force_reload=True)

    leaked = sorted(f for f in _CREDENTIAL_FIELDS if getattr(settings, f))
    assert not leaked, f"get_settings() leaked host credentials for fields {leaked}"


def test_no_host_credential_env_var_is_visible_to_a_test():
    present = sorted(name for name in _SCRUBBED_ENV_NAMES if name in os.environ)
    assert not present, (
        f"host credential env vars visible inside a test: {present}. "
        "They must be scrubbed by conftest's `_isolate_environment`."
    )


# ---------------------------------------------------------------------------
# The complete guard: a plain Settings() must be indistinguishable from one
# built with the env file explicitly disabled. Covers every field, present and
# future, without enumerating any of them.
# ---------------------------------------------------------------------------
def test_plain_settings_is_identical_to_dotenv_disabled_settings():
    plain = Settings().model_dump()
    hermetic = Settings(_env_file=None).model_dump()

    differing = sorted(k for k in plain if plain[k] != hermetic[k])
    assert not differing, (
        "env hermeticity REGRESSED: Settings() diverges from "
        f"Settings(_env_file=None) on fields {differing} — i.e. the suite is "
        "reading this machine's .env, so any 'default value' assertion is really "
        "asserting whatever that file says."
    )


def test_settings_defaults_are_the_declared_code_defaults():
    """Spot-check fields the developer's .env is known to override.

    These are safe to compare by value (no secrets) and they are the ones that
    silently rewrite behaviour: a .env that sets OPENRESEARCH_DEFAULT_SANDBOX=gcp
    turns every "defaults to runpod" assertion into a lie.
    """
    settings = Settings()

    assert settings.default_sandbox == "runpod"
    assert settings.force_sandbox == ""
    assert settings.gcp_project == ""
    assert settings.gcp_gcs_bucket == ""
    assert settings.gcp_base_image == ""
    assert settings.gcp_gpu_skus == ["gcp_a100_80", "gcp_a100_80x8"]
    assert settings.environment == "development"
    assert settings.llm_provider == "anthropic"


# ---------------------------------------------------------------------------
# Anti-vacuity: prove the BLOCK is what stops the read — not an absent/unreadable
# .env. Uses a fake dotenv in tmp_path, so it holds on CI (no repo .env) and on a
# developer box alike, and never touches a real credential.
#
# This also pins the pydantic-settings behaviour the fix depends on: `env_file`
# is read from `model_config` at CONSTRUCTION time, so a post-class-creation
# mutation is honoured. If a pydantic-settings upgrade ever bakes `env_file` in
# at class-creation time instead, this test fails loudly instead of the suite
# quietly going non-hermetic again.
# ---------------------------------------------------------------------------
def test_isolation_actually_blocks_a_dotenv_that_exists_on_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENRESEARCH_DEFAULT_SANDBOX=docker\n")

    # Blocked: the autouse isolation is in force, so the file is not read.
    assert Settings().default_sandbox == "runpod"

    # Re-enable the disk read and the very same file IS honoured — proving the
    # assertion above passed because of the block, not because the file was
    # missing, empty, or unreadable.
    monkeypatch.setitem(Settings.model_config, "env_file", ".env")
    assert Settings().default_sandbox == "docker"


def test_dotenv_disk_reads_enabled_fixture_opts_back_in(
    dotenv_disk_reads_enabled, tmp_path, monkeypatch
):
    """The documented escape hatch for tests whose subject IS the dotenv path."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENRESEARCH_DEFAULT_SANDBOX=docker\n")

    assert Settings().default_sandbox == "docker"


# ---------------------------------------------------------------------------
# Fixture ordering. The scrub strips OPENRESEARCH_*; two things must survive it.
# ---------------------------------------------------------------------------
def test_disk_floor_fixture_survives_the_env_scrub():
    """`_disable_disk_floor_preflight` depends on `_isolate_environment`, so its
    setenv lands after the OPENRESEARCH_* scrub rather than being deleted by it."""
    assert os.environ.get("OPENRESEARCH_DISK_FLOOR_GB") == "0"


def test_a_tests_own_monkeypatch_setenv_beats_the_scrub(monkeypatch):
    """A test body runs after every fixture, so explicit injection always wins.

    This is the supported way to give a test a credential: set it yourself. The
    token below is a literal fake — never a real key.
    """
    monkeypatch.setenv("OPENRESEARCH_DISK_FLOOR_GB", "42")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")
    monkeypatch.setenv("OPENRESEARCH_DEFAULT_SANDBOX", "local")

    assert os.environ["OPENRESEARCH_DISK_FLOOR_GB"] == "42"

    settings = Settings()
    assert settings.anthropic_api_key == "sk-ant-fake-for-test"
    assert settings.default_sandbox == "local"


def test_env_scrub_is_restored_between_tests(monkeypatch):
    """Sanity: the scrub is monkeypatch-based, so a var a test sets does not
    survive into the next one. Paired with the assertion in
    `test_no_host_credential_env_var_is_visible_to_a_test`."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-test")
    assert "OPENAI_API_KEY" in os.environ
    # Teardown (monkeypatch + `_isolate_environment`) removes it; the companion
    # test above proves no credential name is visible at test start.


@pytest.mark.parametrize("prefix", ["OPENRESEARCH_", "REPROLAB_", "AZURE_", "GCP_"])
def test_leak_detector_recognises_each_owned_prefix(prefix, monkeypatch):
    """`_leaking_env_names` must actually match the namespaces it claims to."""
    monkeypatch.setenv(f"{prefix}HERMETICITY_PROBE", "x")
    assert f"{prefix}HERMETICITY_PROBE" in _leaking_env_names()
