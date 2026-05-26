"""Codebase scanner: claude_agent_sdk.query() / Client() may only be called
from SDK adapter modules. Any other caller risks the aclose race — route through
sdk_isolation.run_isolated or collect_agent_text instead."""
import ast
from pathlib import Path

# These files are the only legitimate direct callers of claude_agent_sdk.query()/Client():
# - claude_runtime.py: the SDK adapter (calls query() inside run_agent async generator)
# - rlm_query.py: ClaudeLlmClient already self-isolates via ThreadPoolExecutor
# - sdk_isolation.py: the isolation helper itself (does NOT call query directly,
#   but is listed for completeness in case it grows test fixtures)
ALLOWED_FILES = {
    "backend/agents/runtime/claude_runtime.py",
    "backend/services/context/workspace/tools/rlm_query.py",
    "backend/agents/runtime/sdk_isolation.py",
}


def test_no_direct_claude_agent_sdk_query_calls():
    repo_root = Path(__file__).resolve().parent.parent
    violations: list[str] = []
    for py_file in (repo_root / "backend").rglob("*.py"):
        rel = str(py_file.relative_to(repo_root))
        if rel in ALLOWED_FILES:
            continue
        if "claude_agent_sdk" not in py_file.read_text():
            continue
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match `claude_agent_sdk.query(...)` and `claude_agent_sdk.Client(...)`
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    if func.value.id == "claude_agent_sdk" and func.attr in ("query", "Client"):
                        violations.append(
                            f"{rel}:{node.lineno}: direct claude_agent_sdk.{func.attr}() call"
                        )
    assert not violations, (
        "Direct claude_agent_sdk calls must go through sdk_isolation.run_isolated "
        "or collect_agent_text (which uses run_isolated internally):\n"
        + "\n".join(violations)
    )
