"""Phase 8 — the read_context_map instruction is FLAG-GATED.

The default (REPROLAB_CONTEXT_MAP off) path must stay inert: the root must NOT
be told to call read_context_map (which would burn a REPL action on an empty
read every fact-derivation). The instruction appears only when the flag is on.
"""
from backend.agents.rlm.models import ROOT_MODELS
from backend.agents.rlm.system_prompt import build_system_prompt


def _prompt():
    return build_system_prompt(
        context_metadata={"paper_text": {"type": "str", "length": 80_000}},
        root_model=ROOT_MODELS["claude"],
    )


def test_prompt_mentions_read_context_map_when_enabled(monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    assert "read_context_map" in _prompt()


def test_prompt_omits_read_context_map_when_disabled(monkeypatch):
    monkeypatch.delenv("REPROLAB_CONTEXT_MAP", raising=False)
    assert "read_context_map" not in _prompt()
