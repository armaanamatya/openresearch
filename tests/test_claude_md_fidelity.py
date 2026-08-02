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


# --- Cloud/auth posture guards (operator directives 2026-07-22 and 2026-08-01) ---

_POSTURE_DOCS = [
    "README.md",
    "CLAUDE.md",
    "ONBOARDING.md",
    "coworker.md",
    "docs/architecture.md",
    "docs/engineering-guide.md",
    "docs/operations.md",
    "docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md",
    "docs/runbooks/2026-08-01-remote-run-llm-auth.md",
    "docs/runbooks/2026-08-01-feature-ablation-gcp-runbook.md",
    "backend/agents/rlm/CLAUDE.md",
    "backend/services/runtime/CLAUDE.md",
    "infra/gcp/README.md",
    "infra/gcp/helm/README.md",
    "infra/azure/README.md",
    "docker/gke-cell-base/README.md",
]


def _posture_texts():
    for rel in _POSTURE_DOCS:
        path = _REPO / rel
        assert path.exists(), f"posture doc missing: {rel}"
        yield rel, path.read_text(encoding="utf-8")


def test_gke_posture_says_not_used_never_parked():
    """Directive 2026-07-22: every cloud-posture doc says GKE is NOT USED, never 'parked'."""
    # A '"' between the words exempts meta-rules that merely QUOTE the forbidden word,
    # e.g. runtime CLAUDE.md's: "GKE is not used," never "parked."
    parked_near_gke = re.compile(r'\bgke\b[^.!?\n"]{0,60}\bparked\b|\bparked\b[^.!?\n"]{0,60}\bgke\b')
    failures = []
    for rel, text in _posture_texts():
        low = text.lower()
        if "gke" not in low:
            continue
        if "not used" not in low:
            failures.append(f"{rel}: mentions GKE but never states it is NOT USED")
        if parked_near_gke.search(low):
            failures.append(f"{rel}: uses forbidden 'parked' wording for GKE")
    assert not failures, "\n".join(failures)


def test_oauth_marked_forbidden_where_mentioned():
    """Directive 2026-08-01: any current doc that mentions OAuth must mark it forbidden."""
    marker = re.compile(r"(?i)(never use oauth|oauth is forbidden|never oauth|⛔[^\n]*oauth)")
    failures = []
    for rel, text in _posture_texts():
        if "oauth" not in text.lower():
            continue
        if not marker.search(text):
            failures.append(f"{rel}: mentions OAuth without the forbidden marker")
    assert not failures, "\n".join(failures)


def test_no_oauth_recommendation_phrases_survive():
    """Exact stale phrases that recommended OAuth as a usable path must be gone."""
    banned = [
        "Leave empty to use Claude CLI OAuth",
        "`claude login` for OAuth",
        "falls back to Claude CLI OAuth (free on subscription)",
        "Anthropic key/OAuth",
    ]
    failures = []
    for rel, text in _posture_texts():
        for phrase in banned:
            if phrase in text:
                failures.append(f"{rel}: stale OAuth recommendation survives: {phrase!r}")
    assert not failures, "\n".join(failures)
