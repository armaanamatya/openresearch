"""Claude-backed vision client for paper-page description.

Modeled on ``backend.hermes_audit.providers.ClaudeAuditProvider``:
  - resolves the API key through Settings (not os.environ directly)
  - importlib-guards the ``anthropic`` import so the module is importable
    in environments where the SDK is absent
  - ``is_available()`` is the single gate; callers must check before use
"""

from __future__ import annotations

import base64
import importlib
import logging

from backend.config import get_settings

logger = logging.getLogger(__name__)

_DESCRIBE_PROMPT = (
    "You are helping a research-paper reproduction agent build context.\n"
    "Page {page_number}.\n\n"
    "Please do two things:\n"
    "1. Transcribe ALL readable text on this page exactly as it appears "
    "(preserve equations, tables, captions, footnotes).\n"
    "2. After the transcription, add a concise structured description of "
    "every figure, table, and equation you see: what it shows, axis labels, "
    "key values, and its relationship to the paper's contribution.\n\n"
    "Text hint from embedded PDF layer (may be empty for scanned pages):\n"
    "{text_hint}\n\n"
    "Respond in plain text only — no markdown headers."
)


class ClaudeVisionClient:
    """Calls the Anthropic messages API with an image + text prompt."""

    def __init__(
        self,
        model: str,
        max_tokens: int = 1500,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key_override = api_key

    def _resolve_api_key(self) -> str:
        if self._api_key_override is not None:
            return self._api_key_override
        return get_settings().anthropic_api_key

    def is_available(self) -> bool:
        if not self._resolve_api_key():
            return False
        try:
            importlib.import_module("anthropic")
            return True
        except ImportError:
            return False

    def describe_page(
        self,
        *,
        png_bytes: bytes,
        page_number: int,
        text_hint: str,
    ) -> str:
        anthropic = importlib.import_module("anthropic")
        client = anthropic.Anthropic(api_key=self._resolve_api_key())
        b64 = base64.standard_b64encode(png_bytes).decode("ascii")
        prompt = _DESCRIBE_PROMPT.format(
            page_number=page_number,
            text_hint=text_hint or "(empty)",
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                    ],
                }
            ],
        )
        parts = [getattr(b, "text", "") for b in response.content]
        return "".join(parts)


__all__ = ["ClaudeVisionClient"]
