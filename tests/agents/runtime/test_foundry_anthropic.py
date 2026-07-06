from backend.agents.runtime import foundry_anthropic as fa


def test_derives_anthropic_base_from_openai_endpoint(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://res-x.services.ai.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k-123")
    monkeypatch.delenv("AZURE_FOUNDRY_ANTHROPIC_ENDPOINT", raising=False)
    base_url, api_key, models = fa.resolve_foundry_anthropic_credentials()
    assert base_url == "https://res-x.services.ai.azure.com/anthropic"
    assert api_key == "k-123"
    assert models == {"opus": "claude-opus-4-8", "sonnet": "claude-sonnet-5"}
    assert fa.has_foundry_anthropic_credentials() is True


def test_explicit_anthropic_endpoint_wins_and_strips_messages(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ANTHROPIC_ENDPOINT",
                       "https://res-x.services.ai.azure.com/anthropic/v1/messages")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k-9")
    base_url, _, _ = fa.resolve_foundry_anthropic_credentials()
    assert base_url == "https://res-x.services.ai.azure.com/anthropic"


def test_explicit_bare_host_endpoint_gets_anthropic_appended(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ANTHROPIC_ENDPOINT",
                       "https://res-x.services.ai.azure.com")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k-9")
    base_url, _, _ = fa.resolve_foundry_anthropic_credentials()
    assert base_url == "https://res-x.services.ai.azure.com/anthropic"


def test_explicit_bare_host_endpoint_with_trailing_slash(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ANTHROPIC_ENDPOINT",
                       "https://res-x.services.ai.azure.com/")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k-9")
    base_url, _, _ = fa.resolve_foundry_anthropic_credentials()
    assert base_url == "https://res-x.services.ai.azure.com/anthropic"


def test_model_overrides(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://r.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k")
    monkeypatch.setenv("AZURE_FOUNDRY_ANTHROPIC_OPUS", "claude-opus-4-9")
    _, _, models = fa.resolve_foundry_anthropic_credentials()
    assert models["opus"] == "claude-opus-4-9"


def test_unset_is_empty_and_absent(monkeypatch):
    for v in ("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_ANTHROPIC_ENDPOINT",
              "AZURE_FOUNDRY_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    # Neutralize the Settings/.env fallback for every field this module resolves
    # (base_url, api_key, model overrides) — this checkout's real .env may carry
    # live Foundry credentials (see MEMORY.md), so os.environ deletion alone is
    # not hermetic; patching just `_settings_endpoint` only covers base_url.
    monkeypatch.setattr(fa, "_env_or_settings", lambda *_a, **_kw: "")
    base_url, api_key, _ = fa.resolve_foundry_anthropic_credentials()
    assert base_url == "" and api_key == ""
    assert fa.has_foundry_anthropic_credentials() is False


# ---------------------------------------------------------------------------
# normalize_anthropic_base_url robustness gaps: scheme-less hosts, case-
# insensitive suffix matching (casing preserved in the output), and
# query/fragment stripping.
# ---------------------------------------------------------------------------

def test_scheme_less_bare_host_gets_https_and_anthropic_appended():
    assert (fa.normalize_anthropic_base_url("res-x.services.ai.azure.com")
            == "https://res-x.services.ai.azure.com/anthropic")


def test_uppercase_anthropic_suffix_recognized_and_casing_preserved():
    assert (fa.normalize_anthropic_base_url("https://X/Anthropic")
            == "https://X/Anthropic")


def test_uppercase_anthropic_v1_suffix_stripped_case_insensitively():
    assert (fa.normalize_anthropic_base_url("https://X/ANTHROPIC/V1")
            == "https://X/ANTHROPIC")


def test_query_string_is_stripped():
    assert (fa.normalize_anthropic_base_url("https://X/anthropic?api-version=2024")
            == "https://X/anthropic")


def test_fragment_is_stripped():
    assert (fa.normalize_anthropic_base_url("https://X/anthropic#x")
            == "https://X/anthropic")


def test_regression_bare_host_with_scheme():
    assert (fa.normalize_anthropic_base_url("https://res-x.services.ai.azure.com")
            == "https://res-x.services.ai.azure.com/anthropic")


def test_regression_anthropic_suffix_returned_as_is():
    assert (fa.normalize_anthropic_base_url("https://res-x.services.ai.azure.com/anthropic")
            == "https://res-x.services.ai.azure.com/anthropic")


def test_regression_anthropic_v1_suffix_stripped():
    assert (fa.normalize_anthropic_base_url("https://res-x.services.ai.azure.com/anthropic/v1")
            == "https://res-x.services.ai.azure.com/anthropic")


def test_regression_anthropic_v1_messages_suffix_stripped():
    assert (fa.normalize_anthropic_base_url(
                "https://res-x.services.ai.azure.com/anthropic/v1/messages")
            == "https://res-x.services.ai.azure.com/anthropic")


def test_regression_genuine_custom_path_returned_unchanged():
    assert (fa.normalize_anthropic_base_url("https://X/custom/route")
            == "https://X/custom/route")
