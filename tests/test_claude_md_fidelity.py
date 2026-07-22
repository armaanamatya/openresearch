"""Content-fidelity guards for CLAUDE.md (audit R5): keep the day-to-day doc's
load-bearing claims in sync with the code, and ensure every doc citation
resolves to a real file."""
import re
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _read_claude_docs() -> str:
    """Root CLAUDE.md + every nested CLAUDE.md — the doc system is split across subtrees.

    Since 2026-07-05 the day-to-day reference is a lean root plus per-subtree nested
    ``<dir>/CLAUDE.md`` files (loaded on-demand). A documented env var / primitive count /
    doc citation may live in the root OR in the nested file that owns it, so the fidelity
    guard validates the whole set, not just the root.
    """
    _SKIP = {"runs", "node_modules", ".venv", ".git", "openscience-ref", "site-packages"}
    texts = [(_REPO / "CLAUDE.md").read_text()]
    for path in sorted(_REPO.glob("**/CLAUDE.md")):
        rel = path.relative_to(_REPO)
        if str(rel) == "CLAUDE.md" or any(part in _SKIP for part in rel.parts):
            continue
        texts.append(path.read_text())
    return "\n".join(texts)


_CLAUDE = _read_claude_docs()


def _grep_backend(token: str) -> bool:
    """True if token appears anywhere under backend/."""
    r = subprocess.run(
        ["git", "grep", "-q", token, "--", "backend/"], cwd=_REPO
    )
    return r.returncode == 0


# Feature-flag / auth env vars CLAUDE.md documents must actually be read in code.
_DOCUMENTED_ENV_VARS = [
    "OPENRESEARCH_CONTEXT_MAP",
    "OPENRESEARCH_NEGATIVE_LESSONS",
    "OPENRESEARCH_ACCELERATOR_API_KEY",
    "OPENRESEARCH_SUBRLM_OPENAI_TIMEOUT_S",
    "OPENRESEARCH_DISABLE_TORCHRUN_WRAP",
    "OPENROUTER_API_KEY",
]


@pytest.mark.parametrize("var", _DOCUMENTED_ENV_VARS)
def test_documented_env_var_is_read_in_code(var):
    assert var in _CLAUDE, f"{var} expected in CLAUDE.md"
    assert _grep_backend(var), f"CLAUDE.md documents {var} but no code under backend/ reads it"


def test_custom_tools_count_matches_doc():
    from backend.agents.rlm.primitives import PRIMITIVE_REGISTRY

    n = len(PRIMITIVE_REGISTRY)
    assert n == 19, f"PRIMITIVE_REGISTRY has {n} entries; update the doc + this test together"
    assert "19" in _CLAUDE, "CLAUDE.md should state the bound custom_tools count (19)"


def test_all_doc_citations_resolve():
    """Every docs/...(.md) path cited in CLAUDE.md must exist (SCR-7)."""
    cited = set(re.findall(r"docs/[A-Za-z0-9_./-]+\.md", _CLAUDE))
    missing = sorted(p for p in cited if not (_REPO / p).exists())
    assert missing == [], f"CLAUDE.md cites nonexistent docs: {missing}"
