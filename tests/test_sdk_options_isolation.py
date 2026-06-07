"""BUG-NEW-038: every ClaudeAgentOptions(...) construction must isolate the
inner model from the developer's ~/.claude settings + MCP servers.

The three construction sites build options inline inside methods with heavy
dependencies, so we assert at the source/AST level: each ClaudeAgentOptions(...)
call passes setting_sources=[] (an empty list) and an explicit mcp_servers.

claude_runtime.py builds its kwargs via _agent_options_kwargs (gated by
OPENRESEARCH_SDK_HERMETIC, default true); the other two sites isolate
unconditionally.
"""
import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

# (path, whether ClaudeAgentOptions is called directly with the kwargs inline)
_INLINE_SITES = [
    "backend/services/context/workspace/tools/rlm_query.py",
    "backend/hermes_audit/providers.py",
]


def _options_calls(tree: ast.AST) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "ClaudeAgentOptions":
                calls.append(node)
    return calls


@pytest.mark.parametrize("rel", _INLINE_SITES)
def test_inline_site_isolates(rel):
    src = (_REPO / rel).read_text()
    tree = ast.parse(src)
    calls = _options_calls(tree)
    assert calls, f"no ClaudeAgentOptions(...) call found in {rel}"
    for call in calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        assert "setting_sources" in kwargs, f"{rel}: missing setting_sources"
        val = kwargs["setting_sources"]
        assert isinstance(val, ast.List) and not val.elts, (
            f"{rel}: setting_sources must be [] (empty list)"
        )
        assert "mcp_servers" in kwargs, f"{rel}: missing explicit mcp_servers"


def test_claude_runtime_root_kwargs_isolate():
    """claude_runtime builds kwargs in _agent_options_kwargs; assert the
    hermetic branch sets setting_sources=[] and mcp_servers is always explicit."""
    src = (_REPO / "backend/agents/runtime/claude_runtime.py").read_text()
    assert 'kwargs["setting_sources"] = []' in src
    assert '"mcp_servers": mcp_servers' in src
