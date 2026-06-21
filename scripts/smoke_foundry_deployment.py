#!/usr/bin/env python3
"""Smoke-test an Azure AI Foundry (OpenAI-compatible v1) deployment.

Reuses the canonical credential resolver so the test exercises the exact
(base_url, api_key) the harness uses. The deployment/model is overridable so a
new deployment on the same resource can be validated without touching .env:

    .venv/bin/python scripts/smoke_foundry_deployment.py gpt-chat-latest

Prints only status + the model echo + a short reply. Never prints the API key.
"""

from __future__ import annotations

import sys


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        from backend.agents.runtime.foundry_endpoint import resolve_foundry_credentials
    except Exception as exc:  # noqa: BLE001
        print(f"[RED] cannot import foundry resolver: {exc!r}")
        return 2

    base_url, deployment, api_key = resolve_foundry_credentials()
    model = model or deployment
    if not (base_url and api_key and model):
        print(
            "[RED] missing Foundry credentials "
            f"(base_url={'set' if base_url else 'MISSING'}, "
            f"api_key={'set' if api_key else 'MISSING'}, model={model or 'MISSING'})"
        )
        return 2

    print(f"[info] base_url = {base_url}")
    print(f"[info] model    = {model}")
    print(f"[info] api_key  = ****{api_key[-4:]} ({len(api_key)} chars)")

    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        print(f"[RED] openai SDK unavailable: {exc!r}")
        return 2

    client = OpenAI(base_url=base_url, api_key=api_key)
    messages = [{"role": "user", "content": "Reply with exactly: OPENRESEARCH_SMOKE_OK"}]
    # Newer GPT-class deployments reject `max_tokens`/non-default temperature; fall
    # back to the modern `max_completion_tokens` shape on that specific 400.
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=20, temperature=0,
        )
    except Exception as exc:  # noqa: BLE001
        if "max_completion_tokens" in str(exc) or "unsupported_parameter" in str(exc):
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, max_completion_tokens=20,
                )
            except Exception as exc2:  # noqa: BLE001
                print(f"[RED] request failed: {type(exc2).__name__}: {exc2}")
                return 1
        else:
            print(f"[RED] request failed: {type(exc).__name__}: {exc}")
            return 1

    reply = (resp.choices[0].message.content or "").strip()
    print(f"[info] resp.model = {getattr(resp, 'model', '?')}")
    print(f"[info] reply      = {reply!r}")
    usage = getattr(resp, "usage", None)
    if usage is not None:
        print(f"[info] usage      = prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
    print("[GREEN] Foundry deployment reachable and responding." if reply else "[YELLOW] empty reply (auth ok, but no content)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
