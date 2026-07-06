# scripts/foundry_anthropic_smoke.py
"""Opt-in live smoke of the Foundry Anthropic endpoint (opus + sonnet). Costs a few tokens.
Run: .venv/bin/python scripts/foundry_anthropic_smoke.py"""
from __future__ import annotations

import sys

from backend.agents.runtime.foundry_anthropic import resolve_foundry_anthropic_credentials


def main() -> int:
    import anthropic

    base_url, api_key, models = resolve_foundry_anthropic_credentials()
    if not (base_url and api_key):
        print("FAIL: missing base_url/api_key (set AZURE_FOUNDRY_ENDPOINT + _API_KEY)")
        return 1
    client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
    ok = True
    for label, model in models.items():
        try:
            msg = client.messages.create(
                model=model, max_tokens=16,
                messages=[{"role": "user", "content": "reply with one word: pong"}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            print(f"[{label}:{model}] OK stop={msg.stop_reason} text={text!r}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[{label}:{model}] FAIL {type(exc).__name__}: {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
