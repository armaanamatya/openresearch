"""Content-fidelity guards for CLAUDE.md (audit R5): keep the day-to-day doc's
load-bearing claims in sync with the code, and ensure every doc citation
resolves to a real file."""
import re
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_CLAUDE = (_REPO / "CLAUDE.md").read_text()


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
    "OPENRESEARCH_RUNPOD_CLOUD_TYPE",
]


@pytest.mark.parametrize("var", _DOCUMENTED_ENV_VARS)
def test_documented_env_var_is_read_in_code(var):
    assert var in _CLAUDE, f"{var} expected in CLAUDE.md"
    assert _grep_backend(var), f"CLAUDE.md documents {var} but no code under backend/ reads it"


def test_custom_tools_count_matches_doc():
    from backend.agents.rlm.primitives import PRIMITIVE_REGISTRY

    n = len(PRIMITIVE_REGISTRY)
    assert n == 17, f"PRIMITIVE_REGISTRY has {n} entries; update the doc + this test together"
    assert "17" in _CLAUDE, "CLAUDE.md should state the bound custom_tools count (17)"


def test_runpod_cloud_type_default_matches_config():
    from backend.config import Settings

    default = Settings.model_fields["runpod_cloud_type"].default
    assert default == "SECURE"
    # the doc must name SECURE as the default (SBX-2)
    m = re.search(r"OPENRESEARCH_RUNPOD_CLOUD_TYPE.*", _CLAUDE)
    assert m and "SECURE" in m.group(0) and "default" in m.group(0).lower()


def test_all_doc_citations_resolve():
    """Every docs/...(.md) path cited in CLAUDE.md must exist (SCR-7)."""
    cited = set(re.findall(r"docs/[A-Za-z0-9_./-]+\.md", _CLAUDE))
    missing = sorted(p for p in cited if not (_REPO / p).exists())
    assert missing == [], f"CLAUDE.md cites nonexistent docs: {missing}"
