from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator

import pytest

from backend.agents.runtime.base import AgentRuntimeSpec, ProviderName, StreamEvent, StreamText, StreamToolCall
from backend.agents.runtime.invoke import collect_agent_text
from backend.agents.worker_reports import parse_worker_report_sections, worker_reports_path


class FakeRuntime:
    provider_name: ProviderName = "anthropic"

    def __init__(self) -> None:
        self.seen_prompt = ""

    async def run_agent(
        self,
        *,
        agent: AgentRuntimeSpec,
        user_input: str,
    ) -> AsyncIterator[StreamEvent]:
        self.seen_prompt = user_input
        yield StreamToolCall("tool-1", "Bash", {"command": "python train.py"})
        yield StreamText(
            "Done.\n\n"
            "Worker report\n"
            "What was implemented\n"
            "- Added train.py and metrics.json output\n"
            "What was left undone\n"
            "- None\n"
            "Commands run + exit codes\n"
            "- python train.py exit 0\n"
            "Issues discovered\n"
            "- Dataset URL was unavailable\n"
            "Whether procedures were followed\n"
            "- Followed the reproduction checklist\n"
        )


def test_parse_worker_report_sections_extracts_required_fields() -> None:
    parsed = parse_worker_report_sections(
        "Worker report\n"
        "What was implemented:\n"
        "- Added evaluator\n"
        "Commands run + exit codes:\n"
        "- pytest tests/test_eval.py exit code 0\n"
        "Whether procedures were followed:\n"
        "- yes, followed\n"
    )

    assert parsed["implemented"] == ["Added evaluator"]
    assert parsed["commands"] == [
        {"command": "pytest tests/test_eval.py", "exit_code": 0, "source": "worker_report"}
    ]
    assert parsed["procedures_followed"] is True


@pytest.mark.asyncio
async def test_collect_agent_text_persists_worker_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True)
    (run_dir / "demo_status.json").write_text("{}", encoding="utf-8")
    runtime = FakeRuntime()

    output = await collect_agent_text(
        "baseline-implementation",
        "Implement the baseline.",
        project_dir=code_dir,
        runtime=runtime,
    )

    assert "Worker report" in runtime.seen_prompt
    assert "Added train.py" in output
    lines = worker_reports_path(code_dir).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    report = json.loads(lines[0])
    assert report["agent_id"] == "baseline-implementation"
    assert report["implemented"] == ["Added train.py and metrics.json output"]
    assert {"command": "python train.py", "exit_code": 0, "source": "worker_report"} in report["commands"]
    assert {"command": "python train.py", "exit_code": None, "source": "tool_call"} in report["commands"]
    assert report["procedures_followed"] is True
