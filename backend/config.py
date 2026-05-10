"""Application configuration via Pydantic Settings."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REPROLAB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
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


_settings_cache: Settings | None = None


def get_settings(_force_reload: bool = False) -> Settings:
    """Return application settings, cached after first call."""
    global _settings_cache
    if _force_reload or _settings_cache is None:
        _settings_cache = Settings()
    return _settings_cache


# Marker so tests can check: hasattr(get_settings, '_force_reload')
get_settings._force_reload = True
