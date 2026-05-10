"""Application configuration via Pydantic Settings."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REPROLAB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # pydantic-settings does NOT mutate os.environ, but it does read the
        # .env file with full precedence rules: shell env > .env file >
        # default. That's exactly what we want for API keys — see the
        # ``anthropic_api_key`` / ``openai_api_key`` fields below.
        populate_by_name=True,
    )

    environment: Literal["development", "testing", "production"] = "development"
    database_url: str = "sqlite:///reprolab.db"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_default_model: str = "claude-sonnet-4-6"
    anthropic_reasoning_model: str = "claude-opus-4-7"
    openai_default_model: str = "gpt-4o"
    openai_reasoning_model: str = "o4-mini"
    agent_provider_overrides: dict[str, str] = Field(default_factory=dict)

    # External provider API keys. We read both the unprefixed names that
    # the upstream SDKs (anthropic, openai) and most CI conventions use,
    # AND the REPROLAB_-prefixed forms, because some deployments reserve
    # the unprefixed names for a different scope. First match wins.
    #
    # WHY THIS LIVES IN SETTINGS, NOT os.environ:
    # The Hermes audit providers used to read these directly from
    # ``os.environ.get(...)`` and were skipped whenever the spawning
    # process (Lab UI's Next.js dev server, docker entrypoint without
    # env_file, pytest from a fresh shell) hadn't loaded the .env. The
    # values were always in .env, but never in os.environ. Funnelling
    # through Settings makes pydantic-settings the single source of
    # truth: it reads .env from disk on every ``Settings()`` construction
    # regardless of what os.environ contains. Providers pass these
    # values explicitly to ``anthropic.Anthropic(api_key=...)`` /
    # ``openai.OpenAI(api_key=...)`` so the SDKs don't fall back to
    # their own os.environ lookup either.
    anthropic_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "ANTHROPIC_API_KEY",
            "REPROLAB_ANTHROPIC_API_KEY",
        ),
    )
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "OPENAI_API_KEY",
            "REPROLAB_OPENAI_API_KEY",
        ),
    )
    codex_cli_path: str = ""
    codex_auth_path: str = ""

    # Default sandbox mode for the dashboard's "start a run" form.
    # CLI defaults remain controlled separately by argparse flags.
    default_sandbox: Literal["auto", "local", "docker", "runpod"] = "runpod"

    runpod_api_key: str = ""
    runpod_api_base_url: str = "https://rest.runpod.io/v1"
    runpod_image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
    runpod_gpu_type: str = "NVIDIA GeForce RTX 4090"
    runpod_gpu_count: int = 1
    runpod_cloud_type: Literal["SECURE", "COMMUNITY"] = "SECURE"
    runpod_container_disk_gb: int = 50
    runpod_volume_gb: int = 20
    runpod_volume_mount_path: str = "/workspace"
    runpod_network_volume_id: str = ""
    runpod_data_center_ids: str = ""
    runpod_ssh_key_path: str = ""
    runpod_ssh_public_key: str = ""
    runpod_ssh_user: str = "root"
    runpod_boot_timeout_seconds: int = 900
    runpod_delete_on_destroy: bool = True
    runpod_bootstrap_command: str = ""
    # When set, the Runpod backend attaches to this existing pod ID
    # instead of creating a fresh pod per run. The pod is NEVER deleted
    # by the backend (the _owned_pod_ids allowlist enforces this even
    # if delete_on_destroy=true). Useful for persistent shared workers.
    runpod_pod_id: str = ""


_settings_cache: Settings | None = None


def get_settings(_force_reload: bool = False) -> Settings:
    """Return application settings, cached after first call."""
    global _settings_cache
    if _force_reload or _settings_cache is None:
        _settings_cache = Settings()
    return _settings_cache


# Marker so tests can check: hasattr(get_settings, '_force_reload')
get_settings._force_reload = True
