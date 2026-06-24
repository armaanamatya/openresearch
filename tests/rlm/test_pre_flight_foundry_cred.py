"""Focused test: the azure-foundry cred preflight reads the run's single source
of truth (resolve_foundry_credentials), not a raw os.environ read.

Asserts the two NON-network outcomes so the test is socket-hermetic (the suite
blocks non-loopback sockets):
  * empty key                → (False, ...)  — definitive missing-cred rejection.
  * key present, no endpoint  → (True, ...)  — "skipping probe, proceeding".
"""

from __future__ import annotations

import backend.agents.runtime.foundry_endpoint as foundry_endpoint
from backend.agents.rlm.pre_flight_validator import validate_root_credentials


def test_foundry_empty_key_fails(monkeypatch):
    # Key empty (only present in .env-equivalent → resolver returns "") → False.
    monkeypatch.setattr(
        foundry_endpoint,
        "resolve_foundry_credentials",
        lambda: ("https://x.services.ai.azure.com/openai/v1", "gpt-chat-latest", ""),
    )
    ok, msg = validate_root_credentials("azure-foundry", model="azure-foundry")
    assert ok is False
    assert "AZURE_FOUNDRY_API_KEY" in msg


def test_foundry_key_present_no_endpoint_skips_probe(monkeypatch):
    # Key present but endpoint empty → fail-open "skipping probe" (no network).
    monkeypatch.setattr(
        foundry_endpoint,
        "resolve_foundry_credentials",
        lambda: ("", "", "k"),
    )
    ok, msg = validate_root_credentials("azure-foundry", model="azure-foundry")
    assert ok is True
    assert "skipping probe" in msg
