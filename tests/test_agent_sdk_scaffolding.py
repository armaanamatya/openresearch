"""Tests for SDK telemetry and structured-output scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

from backend.agents.orchestrator import ReproLabOrchestrator
from backend.agents.schemas import PaperClaimMap
from backend.agents.structured_output import append_structured_output_instruction
from backend.agents.telemetry import AgentInvocationRecord, AgentTelemetryRecorder


def test_structured_output_instruction_names_schema() -> None:
    prompt = append_structured_output_instruction("Analyze paper.", PaperClaimMap)
    assert "Structured Output Contract" in prompt
    assert "PaperClaimMap" in prompt
    assert "core_contribution" in prompt


def test_agent_telemetry_recorder_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "agent_telemetry.jsonl"
    recorder = AgentTelemetryRecorder(path)
    recorder.append(
        AgentInvocationRecord(
            agent_id="paper-understanding",
            model="test-model",
            started_at="2026-05-09T00:00:00+00:00",
            finished_at="2026-05-09T00:00:01+00:00",
            duration_seconds=1.0,
            message_count=2,
            output_chars=100,
            success=True,
            usage={"input_tokens": 10, "output_tokens": 20},
        )
    )
    payload = json.loads(path.read_text().splitlines()[0])
    assert payload["agent_id"] == "paper-understanding"
    assert payload["usage"]["input_tokens"] == 10


def test_orchestrator_extract_json_prefers_valid_file(tmp_path: Path) -> None:
    orchestrator = ReproLabOrchestrator("prj_test", tmp_path)
    out_file = tmp_path / "prj_test" / "paper_claim_map.json"
    out_file.write_text(json.dumps({"core_contribution": "x"}))
    assert orchestrator._extract_json("", fallback_file=str(out_file)) == {
        "core_contribution": "x"
    }
